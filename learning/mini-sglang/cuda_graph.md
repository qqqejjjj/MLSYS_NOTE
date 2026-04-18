# CUDA Graph 技术分析笔记

CUDA Graph 是 Mini-SGLang 中用于消除 Decode 阶段 **CPU kernel launch 开销**的核心优化技术。它通过提前录制 GPU 计算图并在推理时直接回放，将每次 forward 的数百次 CPU→GPU 内核提交压缩为**一次 replay 调用**，大幅降低延迟，提升 decode 吞吐。

## 1. 核心问题：Decode 阶段的 CPU 开销

### 1.1 为什么 Decode 是瓶颈？

在自回归 LLM 推理中，每次 decode step 只生成 **1 个 token**，计算量极小，GPU 利用率极低。此时单步延迟的主要来源不是 GPU 计算，而是 **CPU 侧的 kernel launch 开销**：

```
没有 CUDA Graph 的 Decode 步骤（batch_size=8）：

CPU 线程      │ 提交 kernel 1 │ 提交 kernel 2 │ ... │ 提交 kernel N │  (Python + CUDA Driver)
              │←  ~10μs/kernel →│←  ~10μs/kernel →│     │←  ~10μs/kernel →│
              │                 │                 │     │                 │
GPU 流水线    │   [k1 执行]    │   [k2 执行]    │     │   [kN 执行]    │
              │← GPU 计算极短  →│← GPU 等 CPU 提交→│     │                │
              时间轴 ─────────────────────────────────────────────────────────►

典型参数：Transformer 70B，每层 ~30 个 kernel，32 层 → 960 次提交
CPU kernel launch 总开销：960 × 10μs = ~10ms/step
GPU 实际计算（decode batch=8）：~0.5ms
CPU 开销 >> GPU 计算时间！
```

### 1.2 CUDA Graph 的解法

```
使用 CUDA Graph 的 Decode 步骤：

录制阶段（启动时，只做一次）：
  CPU：执行一次 forward，同时录制所有 GPU kernel 到 Graph 对象
                          └─► [k1, k2, ..., kN] 全部存入 graph

回放阶段（每次 decode step）：
  CPU：graph.replay()   ← 只提交一次！
  GPU：立即按录制顺序执行 k1→k2→...→kN，无 CPU 干预

CPU kernel launch 总开销：1 次 ~5μs（无论模型多深）
延迟改善：10ms → 0.005ms（对于 kernel launch 部分）
```

---

## 2. 系统架构：CUDA Graph 在 Mini-SGLang 中的位置

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Engine（engine/engine.py）                                                   │
│                                                                             │
│  初始化阶段（一次性）：                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ dummy_req = Req(...)          ← 用于填充固定 batch size 的哑请求       │ │
│  │ GraphRunner.__init__(...)     ← 决定 batch sizes，录制所有 graphs      │ │
│  │   _determine_cuda_graph_bs()  → [1, 2, 4, 8, 16, 24, ..., 160]      │ │
│  │   _capture_graphs()           → 对每个 bs 录制一张 CUDAGraph          │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  推理阶段（每个 decode step）：                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ forward_batch(batch):                                                 │ │
│  │   if graph_runner.can_use_cuda_graph(batch):    ← decode & size ≤ max│ │
│  │       graph_runner.pad_batch(batch)             ← 向上取整到录制 bs   │ │
│  │       logits = graph_runner.replay(batch)       ← 一次 replay()      │ │
│  │   else:                                                               │ │
│  │       logits = model.forward(batch)             ← 普通 forward（prefill）│ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ GraphRunner（engine/graph.py）                        │
│                                                      │
│  graph_map: Dict[int, CUDAGraph]                     │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐     │
│  │graph[1]│  │graph[2]│  │graph[4]│  │graph[8]│ ... │
│  └────────┘  └────────┘  └────────┘  └────────┘     │
│                                                      │
│  buffer: GraphCaptureBuffer                          │
│  ┌──────────────────────────────────────────────┐   │
│  │ input_ids[max_bs]  positions[max_bs]         │   │
│  │ out_loc[max_bs]    logits[max_bs, vocab_size] │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## 3. 核心数据结构：`GraphCaptureBuffer`

`GraphCaptureBuffer`（`engine/graph.py`）持有**固定大小的 GPU 张量**，作为 CUDA Graph 录制和回放的"输入/输出接口"：

```python
@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor   # shape: [max_graph_bs]，录制时固定地址
    out_loc: torch.Tensor     # shape: [max_graph_bs]，KV Cache 中的槽位
    positions: torch.Tensor   # shape: [max_graph_bs]，序列位置
    logits: torch.Tensor      # shape: [max_graph_bs, vocab_size]，输出 logits

    def set_batch(self, batch: Batch) -> None:
        # 让 batch 内部张量指向 buffer 的切片（录制时调用）
        batch.input_ids = self.input_ids[:batch.padded_size]
        batch.out_loc = self.out_loc[:batch.padded_size]
        batch.positions = self.positions[:batch.padded_size]

    def copy_from(self, batch: Batch) -> None:
        # 在回放前将实际数据拷入固定 buffer（每次 replay 前调用）
        self.input_ids[:batch.padded_size].copy_(batch.input_ids)
        self.out_loc[:batch.padded_size].copy_(batch.out_loc)
        self.positions[:batch.padded_size].copy_(batch.positions)
```

**为什么需要固定地址的 buffer？**

CUDA Graph 在录制时记录的是**特定 GPU 地址**的操作序列。如果每次推理时 batch 张量的地址不同，回放时 GPU 会读写错误地址。`GraphCaptureBuffer` 提供稳定的"占位符"——每次推理只需将新数据 `copy_` 进固定 buffer，让录制好的 graph 始终操作相同地址。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 录制时（capture）：                                                           │
│                                                                             │
│  buffer.input_ids ──地址 0xA000──► [kernel1 读取 0xA000]                    │
│  buffer.logits    ──地址 0xB000──► [kernel_last 写入 0xB000]                │
│                                                                             │
│  CUDAGraph 内记录：                                                          │
│  "从地址 0xA000 读输入，经过 N 个 kernels，结果写到 0xB000"                    │
└─────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────┐
│ 回放时（replay）：                                                            │
│                                                                             │
│  新 batch 数据 ─── copy_() ──► buffer（地址不变！0xA000）                    │
│  g.replay()：GPU 直接执行已录制的 kernel 序列，读 0xA000，写 0xB000           │
│  取结果：buffer.logits[:actual_batch_size]                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 录制阶段：`_capture_graphs()`

录制是在 Engine 初始化时**一次性**完成的（`engine/graph.py`）：

```python
def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: nn.Module):
    pool = None

    # 1. 分配 buffer（固定大小的 GPU 张量，最大 batch size）
    self.buffer = GraphCaptureBuffer(
        input_ids=torch.zeros(self.max_graph_bs, dtype=torch.long, device=self.device),
        out_loc=torch.zeros(self.max_graph_bs, dtype=torch.int32, device=self.device),
        positions=torch.zeros(self.max_graph_bs, dtype=torch.long, device=self.device),
        logits=torch.zeros(self.max_graph_bs, vocab_size, dtype=torch.float32, device=self.device),
    )

    # 2. 从最大 batch size 开始录制（建立共享内存池）
    for bs in sorted(self.graph_bs_list, reverse=True):  # 160, 152, ..., 8, 4, 2, 1

        # 2a. Warmup pass（不录制，让 CUDA 消除初始化开销）
        dummy_batch = self._make_dummy_batch(bs)
        self.buffer.set_batch(dummy_batch)
        with torch.cuda.stream(self.stream):
            self.buffer.logits[:bs] = model.forward(dummy_batch)

        # 2b. 录制 pass
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=self.stream):
            self.buffer.logits[:bs] = model.forward(dummy_batch)

        # 2c. 第一张 graph 建立内存池，后续 graph 复用（节省显存）
        if pool is None:
            pool = graph.pool()

        self.graph_map[bs] = graph
```

**关键细节**：

```
录制顺序（从大到小）的原因：

第一张 graph（bs=160）分配了最大内存池：
  ┌───────────────────────────────────────────────────┐
  │ Memory Pool（pool）                                │
  │ ┌──────────────────────────────────────────────┐  │
  │ │  activations for bs=160  (最大中间激活显存)   │  │
  │ └──────────────────────────────────────────────┘  │
  └───────────────────────────────────────────────────┘

后续 bs=152, 144, ..., 1 的 graph 复用同一个 pool（因为更小）：
  内存不重复分配，所有 graph 共享同一块 GPU 内存！
  总额外显存 = 最大 bs 的激活显存（而非 N × 每 bs 激活显存）
```

---

## 5. 回放阶段：`replay()`

每次 decode step 调用 `replay()`：

```python
def replay(self, batch: Batch) -> torch.Tensor:
    # 1. 将实际 batch 数据拷入固定 buffer 地址
    self.buffer.copy_from(batch)

    # 2. 准备 attention backend（更新 KV Cache 索引等辅助信息）
    self.attn_backend.prepare_for_replay(batch)

    # 3. 回放录制好的 graph（一次 CPU 调用，GPU 执行全部 kernels）
    g = self.graph_map[batch.padded_size]
    g.replay()

    # 4. 只返回实际 batch size 的 logits（去掉 padding 部分）
    return self.buffer.logits[: batch.size]
```

**`pad_batch()` 的作用**（回放前调用）：

```python
def pad_batch(self, batch: Batch) -> None:
    # 找到 >= batch.size 的最小录制 batch size
    padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size)
    # 用 dummy_req 填充至 padded_size（保持 batch 维度与录制时相同）
    batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
    batch.padded_size = padded_size
```

例如 `graph_bs_list = [1, 2, 4, 8, 16, ...]`，当实际 batch_size=6 时：

```
实际: batch.size = 6
填充: padded_size = 8（下一个录制的 bs）
      batch.padded_reqs = [req1, req2, req3, req4, req5, req6, dummy, dummy]

回放 graph[8]（为 bs=8 录制的 graph）
返回 buffer.logits[:6]（丢弃最后 2 行 dummy 的 logits）
```

---

## 6. Batch Size 预确定：`_determine_cuda_graph_bs()`

```python
def _determine_cuda_graph_bs(
    cuda_graph_bs: Optional[List[int]],
    cuda_graph_max_bs: Optional[int],
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs   # 用户手动指定

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        # H200（>80GB）能录制更多 bs，覆盖更大 batch
        cuda_graph_max_bs = 256 if free_memory_gb > 80 else 160

    # 稀疏小值（1,2,4），密集大值（8,16,24,...,max_bs）
    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    # 例：[1, 2, 4, 8, 16, 24, 32, ..., 160]  共 23 个 graph
```

**设计考量**：

```
为什么小 batch size 稀疏（1,2,4 之后间隔 8）？

Decode batch 普遍较大时：精细覆盖 8,16,24,... 最常见范围
小 batch（1,2,4）是特殊情况（冷启动、低负载），单独覆盖
8 步长的 padding 损失：最坏 7 个 dummy（如 batch=9 → padded=16）
内存浪费 = dummy 数 × 单 token 显存（极小，可忽略）

录制 23 张 graph 的显存代价 ≈ 1 张（最大 bs）的激活显存（共享内存池）
```

---

## 7. `dummy_req`：填充机制的关键

CUDA Graph 要求每次同一张 graph 回放时，输入的**形状完全相同**。当实际 batch size 不是录制时的 bs 时，需要用哑请求填充：

```python
# engine/engine.py 中的初始化
self.dummy_req = Req(
    input_ids=tensor([0]),             # 哑 token id = 0
    table_idx=max_running_req,         # 超出正常范围的独占 page table 槽
)
# 为 dummy_req 分配一个哑 page（所有 dummy 指向同一个无害 page）
self.page_table[self.dummy_req.table_idx].fill_(num_tokens)
```

```
page_table 示意（max_running_req = 256，dummy 占 index 256）：

正常请求 page_table[0..255]：指向各自的 KV Cache page
dummy_req.table_idx = 256，page_table[256]：指向一个固定的哑 page

填充后的 batch（实际 6 + dummy 2）：
  ┌──────────────────────────────────────────────────────┐
  │ padded_reqs = [req1, req2, req3, req4, req5, req6,  │
  │                dummy, dummy]                         │
  │                                                      │
  │ input_ids: [t1, t2, t3, t4, t5, t6, 0, 0]           │
  │ out_loc:   [loc1, ..., loc6, dummy_loc, dummy_loc]   │
  └──────────────────────────────────────────────────────┘

dummy 的 attention 写入哑 page（不影响正常 KV Cache）
dummy 的 logits 被裁剪掉（buffer.logits[:6]）
```

---

## 8. 使用条件：`can_use_cuda_graph()`

```python
def can_use_cuda_graph(self, batch: Batch) -> bool:
    return (
        batch.is_decode                  # 只有 decode phase 使用（prefill 不适用）
        and batch.size <= self.max_graph_bs  # batch 不超过最大录制 bs
    )
```

**Prefill 为什么不用 CUDA Graph？**

```
Prefill 请求：序列长度各不相同（可能 10~4096 token 不等）

CUDA Graph 的限制：录制时 batch 形状固定
  - 如果为所有可能的序列长度各录制一张 graph → 数千张 graph，显存爆炸
  - 序列长度动态变化 → 无法简单做 padding

Decode 请求：每个序列每步只处理 1 个 token
  - batch 形状仅由 batch_size 决定（与序列长度无关！）
  - 只需按 batch_size 录制有限张 graph（23 张覆盖 1~160）
  - 完美匹配 CUDA Graph 固定形状的要求 ✓
```

---

## 9. 完整实例演示：batch_size=6 的 Decode Step

```
初始状态：
  graph_bs_list = [1, 2, 4, 8, 16, ...]
  graph_map = { 1: graph_1, 2: graph_2, 4: graph_4, 8: graph_8, ... }
  buffer = GraphCaptureBuffer（固定 GPU 内存，地址不变）
  actual batch: 6 个请求 [req1, req2, req3, req4, req5, req6]

Step 1: can_use_cuda_graph(batch)?
  batch.is_decode = True  ✓
  batch.size = 6 ≤ max_graph_bs = 160  ✓
  → 使用 CUDA Graph

Step 2: graph_runner.pad_batch(batch)
  padded_size = first bs >= 6 in [1,2,4,8,...] = 8
  batch.padded_reqs = [req1, ..., req6, dummy, dummy]
  batch.padded_size = 8

Step 3: graph_runner.replay(batch)
  3a. buffer.copy_from(batch):
      buffer.input_ids[0:8] ← [t1, t2, t3, t4, t5, t6, 0, 0]
      buffer.out_loc[0:8]   ← [l1, l2, l3, l4, l5, l6, d, d]
      buffer.positions[0:8] ← [p1, p2, p3, p4, p5, p6, 0, 0]

  3b. attn_backend.prepare_for_replay(batch)
      更新 attention 所需的辅助 metadata（page table 索引等）

  3c. graph_map[8].replay()
      ← CPU 只发起这一次调用！
      ← GPU 立即执行为 bs=8 录制的全部 kernel 序列
      ← 从 buffer.input_ids 读取，结果写入 buffer.logits[0:8]

  3d. return buffer.logits[:6]
      ← 取前 6 行（丢弃 dummy 的 logits）
      ← logits.shape = [6, vocab_size]

Step 4: 采样
  sampler.forward(logits)  → 6 个 next token ids
  每个 req 追加 next token，继续下一轮 decode
```

---

## 10. 与 Engine Stream 和 Overlap Scheduling 的集成

### 10.1 Engine Stream 的一致性

```python
# engine/engine.py 初始化
self.stream = torch.cuda.Stream()    # engine 专用 CUDA stream

self.graph_runner = GraphRunner(
    stream=self.stream,    # ← 录制和回放必须在同一个 stream 上！
    ...
)
```

CUDA Graph 在录制时将所有 kernel 绑定到特定的 CUDA stream（`self.stream`）。回放时也必须在相同 stream 上执行。Mini-SGLang 用统一的 `engine.stream` 确保一致性，避免 stream 切换导致回放顺序错乱。

### 10.2 与 Overlap Scheduling 的配合

```
Overlap Scheduling 流水线：

时间轴 ──────────────────────────────────────────────────────────────────────►
CPU (prepare):  [prepare batch N-1]  →   [prepare batch N]   →  [prepare batch N+1]
                                  ↓                     ↓
GPU (forward):  [forward batch N-2] →  [forward batch N-1]   →  [forward batch N]
                                  ↓                     ↓
CPU (process):  [process N-3 result]→ [process N-2 result]   →  [process N-1 result]

当 forward 是 decode 且使用 CUDA Graph 时：
GPU 执行 forward batch N-1：
  → graph.replay() 一次调用
  → GPU 立即执行，没有逐 kernel 的 CPU 等待！
  → CPU 可以立即切换去 prepare batch N（更少 CPU 时间消耗）
  → CPU / GPU 并行程度更高，Overlap Scheduling 效果提升
```

发起 `replay()` 的 CPU 耗时从 ~10ms 降至 ~0.005ms，CPU 可用于 Overlap 的时间窗口大幅扩大。

---

## 11. 总结

CUDA Graph 在 Mini-SGLang 中的核心价值是**将 Decode 阶段数百次 kernel launch 压缩为一次 replay**。

五条核心要点：

1. **固定 buffer 是关键**：`GraphCaptureBuffer` 提供地址不变的 GPU 张量，CUDA Graph 录制和回放始终操作相同地址，每次 replay 前只需 `copy_` 新数据。

2. **从大向小录制，共享内存池**：最大 bs 的 graph 建立内存池，后续所有 graph 复用，总额外显存 ≈ 1 张最大 graph 的激活显存。

3. **向上取整 padding**：`pad_batch()` 用 `dummy_req` 填充使 batch size 对齐到下一个录制 bs，`replay()` 后裁掉 dummy 的输出。

4. **仅对 decode 有效**：prefill 序列长度动态变化，无法高效用 CUDA Graph；decode 每步只处理 1 token，batch shape 仅由 batch_size 决定，非常适合固定形状要求。

5. **与 Overlap Scheduling 协同**：CUDA Graph 大幅降低 GPU forward 阶段的 CPU 占用，CPU 可用时间增多，Overlap Scheduling 的流水线效率进一步提升。

| 特性 | Normal Forward | CUDA Graph |
|---|---|---|
| 每步 kernel launch 次数 | ~960 次（32层×30个） | 1 次 replay |
| CPU kernel launch 开销 | ~10ms/step | ~0.005ms/step |
| 适用阶段 | Prefill + Decode | Decode only |
| batch size 限制 | 无 | ≤ max_graph_bs（160/256） |
| 额外显存开销 | 无 | ≈ 最大 bs 的激活显存 |
| 初始化时间 | 无 | 录制所有 graph（~30秒） |
