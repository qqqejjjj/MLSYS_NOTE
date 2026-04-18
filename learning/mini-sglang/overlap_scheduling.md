# Overlap Scheduling 技术分析笔记

Overlap Scheduling 是 Mini-SGLang 中用于提升 GPU 利用率的核心优化技术，源自 [NanoFlow](https://arxiv.org/abs/2408.12757) 论文的思想。它的本质是：**将第 N 批次的 CPU 调度开销，与第 N-1 批次的 GPU 计算并行执行**，从而把 CPU 侧 ~1ms 的调度延迟隐藏在 GPU 侧 10~50ms 的前向计算中，显著提升推理系统的整体吞吐量。

## 1. 核心概念与问题背景

### 1.1 为什么需要 Overlap Scheduling？

在朴素的 LLM 推理调度循环中，每一轮迭代都遵循以下严格串行的执行顺序：

```
[CPU 调度] → [launch GPU 内核] → 等待 GPU 完成 → [CPU 处理结果] → [CPU 调度] → ...
```

具体地，CPU 调度阶段需要完成：
1. 从消息队列接收新请求（ZMQ）
2. 决定本轮处理哪些请求（Prefill / Decode）
3. 分配 KV Cache 物理页（`allocate_paged`）
4. 构建 GPU 计算所需的各种索引张量（`_make_positions`、`_make_input_tuple`、`_make_write_tuple`），含 Host → Device 的异步内存拷贝
5. 准备 Attention 元数据（`prepare_metadata`）

这些 CPU 工作在传统模式下会直接延迟下一个 batch 的提交时间，导致 GPU 处于空闲等待状态，是典型的 **CPU-bound bottleneck**。

### 1.2 Overlap Scheduling 的硬件逻辑位置

Mini-SGLang 使用**双 CUDA Stream**架构来实现 CPU 和 GPU 工作的并行。整个系统分为两个流：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CPU (Scheduler Process) - 主线程在 scheduler stream 上执行                   │
│                                                                             │
│   ████ self.stream（scheduler stream）████                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ [步骤1] 接收消息（ZMQ）                                              │   │
│   │   • self.receive_msg(blocking) - 非阻塞接收用户请求                  │   │
│   │   • _process_one_msg() - 将请求加入 prefill/decode manager          │   │
│   │                                                                      │   │
│   │ [步骤2] CPU 调度决策（当前批次 N）                                   │   │
│   │   • decode_manager.schedule_next_batch() - 选择哪些请求进入批次     │   │
│   │   • graph_runner.pad_batch() - CUDA Graph 对齐                      │   │
│   │                                                                      │   │
│   │ [步骤3] 分配资源                                                     │   │
│   │   • cache_manager.allocate_paged() - 分配物理 KV Cache 页           │   │
│   │   • table_manager 更新 page_table                                   │   │
│   │                                                                      │   │
│   │ [步骤4] 构建元数据（异步 H2D，提交到 scheduler stream）              │   │
│   │   • _make_positions() - 构建 position 索引 [pin_memory→H2D async]   │   │
│   │   • _make_input_tuple() - 构建输入映射 [pin_memory→H2D async]       │   │
│   │   • _make_write_tuple() - 构建写回映射 [pin_memory→H2D async]       │   │
│   │   • prepare_metadata() - FlashAttention/PagedAttention 元数据       │   │
│   │     （这些 H2D 拷贝是异步的，CPU 提交后立即返回！）                  │   │
│   │                                                                      │   │
│   │ [步骤5] 处理上一批次（N-1）的结果 ◄── 关键！与 GPU 并行             │   │
│   │   • copy_done_{N-1}.synchronize() - 等待 D2H 完成（唯一同步点）     │   │
│   │   • 遍历 batch_{N-1}.reqs，读取 next_tokens_cpu                     │   │
│   │   • EOS 检测（next_token == eos_token_id）                          │   │
│   │   • 释放完成请求的资源（table_idx, KV Cache）                        │   │
│   │   • send_result(reply) - ZMQ 发送 token 给 detokenizer             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ████ self.engine.stream（engine stream）████  ◄── 临时切换执行             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ [步骤6] launch GPU 计算（当前批次 N，非阻塞！）                       │   │
│   │   • engine.stream.wait_stream(self.stream) ◄─ GPU 流间同步           │   │
│   │     （engine stream 等待 scheduler stream 上 H2D 完成）               │   │
│   │   • batch.input_ids = token_pool[mapping] - 读取输入 token           │   │
│   │   • engine.forward_batch():                                          │   │
│   │     - model.forward() 或 graph_runner.replay() ◄─ GPU 前向          │   │
│   │     - sampler.sample(logits) ◄─ GPU 采样                             │   │
│   │     - next_tokens_cpu = gpu→cpu (non_blocking) ◄─ 异步 D2H           │   │
│   │     - copy_done.record(engine.stream) ◄─ 记录 D2H 完成事件           │   │
│   │   • token_pool[mapping] = next_tokens_gpu - 写回新 token             │   │
│   │                                                                      │   │
│   │   （CPU 在这里立即返回，不等 GPU！）                                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                  ↓ 提交到 GPU                                │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │ PCIe / NVLink
┌─────────────────────────────────▼───────────────────────────────────────────┐
│ GPU (Device) - 硬件异步执行队列                                              │
│                                                                             │
│   ████ scheduler stream 队列 ████                                            │
│   [H2D: positions] → [H2D: input_mapping] → [H2D: write_mapping] ...        │
│                                                                             │
│   ████ engine stream 队列 ████                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ [GPU 步骤1] 等待 H2D 完成                                             │   │
│   │   • wait_stream 同步点：读取 scheduler stream 传来的所有数据          │   │
│   │                                                                      │   │
│   │ [GPU 步骤2] 模型前向（10~50ms，大头！）                               │   │
│   │   • PagedAttention/FlashAttention Kernel - 读取 KV Cache             │   │
│   │   • LayerNorm, Linear, GeLU, ... (FFN layers)                        │   │
│   │   • 多层 Transformer block 迭代                                      │   │
│   │   • logits = final_layer(hidden_states)                              │   │
│   │                                                                      │   │
│   │ [GPU 步骤3] Token 采样（~0.1ms）                                      │   │
│   │   • Top-K / Top-P / Temperature 采样                                 │   │
│   │   • next_tokens_gpu = sample(logits)                                 │   │
│   │                                                                      │   │
│   │ [GPU 步骤4] 异步 D2H 拷贝（后台 DMA）                                 │   │
│   │   • GPU → CPU: next_tokens_cpu (non_blocking copy)                  │   │
│   │   • copy_done_event.record() - 标记此拷贝完成                        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   关键：GPU 步骤2-4（批次 N）与 CPU 步骤5（批次 N-1 后处理）完全并行！       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**关键洞察**：GPU 内核的 `launch` 调用是非阻塞的——CPU 只是把任务"塞进" engine stream 的队列，然后立刻返回继续做别的事情（处理上一批结果），GPU 在后台异步执行。CPU 和 GPU 就像两条并行的生产线，通过 CUDA Stream 实现生产者-消费者解耦。

---

## 2. 核心数据结构分析

### 2.1 `ForwardOutput`（基于 `python/minisgl/engine/engine.py`）

```python
class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor  # 采样结果，保留在 GPU，用于写回 token_pool
    next_tokens_cpu: torch.Tensor  # 同一份数据的 CPU 副本，用于 EOS 检测和发送
    copy_done_event: torch.cuda.Event  # D2H 拷贝完成后被 record 的 CUDA 事件
```

这三个字段的设计极为精妙：

| 字段 | 位置 | 用途 |
|---|---|---|
| `next_tokens_gpu` | GPU 显存 | 写回 `token_pool`（`token_pool[output_mapping] = ...`），用于下一轮推理 |
| `next_tokens_cpu` | CPU 固定内存 | CPU 端检查 EOS，通过 ZMQ 发给 detokenizer |
| `copy_done_event` | CUDA 事件 | D2H 异步拷贝完成的精准同步点，**避免同步整个 GPU** |

`copy_done_event` 的关键价值在于：CPU 只需调用 `copy_done_event.synchronize()` 等待这一个拷贝操作完成，而**不需要** `torch.cuda.synchronize()` 去等待 engine stream 上所有内核完成，最大限度减少 CPU-GPU 同步代价。

### 2.2 `ForwardInput` 与 `ForwardData`（基于 `python/minisgl/scheduler/scheduler.py`）

```python
# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)，CPU→GPU 的输入索引
    write_tuple: Indice2D  # (req_mapping, seq_lens)，GPU→GPU 的写出索引

ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"
```

`ForwardInput` 是一个**"快照"对象**。在 Overlap 模式中，当第 N 批次的 GPU 计算正在进行时，CPU 已经在准备第 N+1 批次的调度数据（修改 `batch`、`req` 等对象）。如果不把第 N 批次所需的所有输入数据封装成独立的不可变快照，就会产生**数据竞争（IMA, Illegal Memory Access）**——CPU 修改的数据可能已经被 GPU 正在读取。

`ForwardData = (ForwardInput, ForwardOutput)` 将一次完整迭代的"输入快照"和"计算结果"打包在一起，作为跨迭代的"交接包"传递。

### 2.3 双流初始化

**`Engine.__init__`** 中（`python/minisgl/engine/engine.py`）：

```python
class Engine:
    def __init__(self, config: EngineConfig):
        ...
        self.stream = torch.cuda.Stream()   # ← engine stream：GPU 计算专用流
        torch.cuda.set_stream(self.stream)  # Engine 初始化时默认使用 engine stream
```

**`Scheduler.__init__`** 中（`python/minisgl/scheduler/scheduler.py`）：

```python
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        self.engine = Engine(config)  # Engine 初始化后，torch 默认流仍是 engine stream

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)     # ← scheduler stream：CPU 元数据流
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)  # context manager
        torch.cuda.set_stream(self.stream)  # ← 主线程切换到 scheduler stream
```

初始化完成后的状态：

| 流 | 引用路径 | 当前线程默认流？ | 职责 |
|---|---|---|---|
| scheduler stream | `self.stream` | ✅ 是 | CPU 发起的所有异步 H2D 拷贝 |
| engine stream | `self.engine.stream` | ❌ 否 | GPU 模型前向、采样、D2H 拷贝 |

---

## 3. 核心方法逐行解析

### 3.1 `run_forever`：模式选择入口

```python
@torch.inference_mode()
def run_forever(self) -> NoReturn:
    if ENV.DISABLE_OVERLAP_SCHEDULING:
        # Normal 模式：强制在 engine stream 上同步执行，CPU 和 GPU 串行
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            while True:
                self.normal_loop()
    else:
        # Overlap 模式：在 scheduler stream 上跑，data 存储上一轮的 ForwardData
        assert torch.cuda.current_stream() == self.stream
        data = None
        while True:
            data = self.overlap_loop(data)  # ← data 是"上一批"的结果，本轮处理
```

`data` 这个变量是 Overlap 机制的核心载体：**每次 `overlap_loop` 返回的是"本次"刚 launch 的 batch 的数据，而下一次迭代中，它会作为 `last_data` 被处理**。这形成了一个"错位一拍"的流水线。

### 3.2 `overlap_loop`：主循环核心（完整注释版）

```python
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    """
    The main loop of overlapping scheduling and execution.
    It will overlap the execution of current batch and processing of last batch's results,
    which can effectively hide CPU latency and improve GPU utilization.
    """
    # ① 决定是否阻塞等待新消息
    # 只有在三种条件都不满足时才阻塞：没有"上批结果"待处理、无可运行的 prefill、无可运行的 decode
    blocking = not (
        last_data is not None       # 有上一批结果待处理 → 不阻塞
        or self.prefill_manager.runnable  # 有待 prefill 的请求 → 不阻塞
        or self.decode_manager.runnable   # 有待 decode 的请求 → 不阻塞
    )
    for msg in self.receive_msg(blocking=blocking):  # ZMQ 接收（在 scheduler stream 上）
        self._process_one_msg(msg)

    # ② CPU 调度当前批次（运行在 scheduler stream 上）:
    # 包括分配 KV Cache 页、 构建 H2D 索引（全异步 non_blocking）、准备 Attention 元数据
    forward_input = self._schedule_next_batch()

    ongoing_data = None
    if forward_input is not None:
        # ③ 切换到 engine stream 并 launch GPU 前向（非阻塞！立即返回）
        with self.engine_stream_ctx:               # 临时切换到 engine stream
            # 关键同步：engine stream 等待 scheduler stream 上的 H2D 拷贝完成
            # GPU 内核读取数据前，必须确保 CPU 已经把数据写好
            self.engine.stream.wait_stream(self.stream)
            # launch GPU 前向（_forward 内部调用 engine.forward_batch，非阻塞）
            ongoing_data = (forward_input, self._forward(forward_input))
        # ← 此时 GPU 在后台跑批次 N，CPU 立即往下执行 ④

    # ④ 处理上一批（last_data）的结果 —— 全是 CPU 操作，与 ③ 的 GPU 计算并行！
    # 包括：等待 D2H copy 完成、EOS 检测、更新请求状态、ZMQ 发送 token
    self._process_last_data(last_data)

    # 返回本批数据，下次迭代时作为 last_data 被处理
    return ongoing_data
```

**`normal_loop` 对比**（不使用 Overlap）：

```python
def normal_loop(self) -> None:
    blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    forward_input = self._schedule_next_batch()
    ongoing_data = None
    if forward_input is not None:
        ongoing_data = (forward_input, self._forward(forward_input))

    # ← 关键区别：传入的是"当前批次" ongoing_data，而不是"上一批次"
    # 必须等 GPU 完成当前批次，CPU 才能继续，没有任何重叠！
    self._process_last_data(ongoing_data)
```

### 3.3 `_prepare_batch`：CPU 元数据构建（所有 H2D 均为异步）

```python
def _prepare_batch(self, batch: Batch) -> ForwardInput:
    # 全部运行在 self.stream（scheduler stream）上
    self.engine.graph_runner.pad_batch(batch)               # CUDA Graph 尺寸对齐
    self.cache_manager.allocate_paged(batch.reqs)           # 分配物理 KV Cache 页

    # ↓ 以下三步全部使用 pin_memory=True + .to(device, non_blocking=True)
    # → 触发异步 H2D DMA 拷贝，提交到 scheduler stream 后立即返回
    batch.positions  = _make_positions(batch, self.device)   # positions 索引
    input_mapping    = _make_input_tuple(batch, self.device) # (req_mapping, positions)
    write_mapping    = _make_write_tuple(batch, self.device) # (req_mapping, seq_lens)

    batch.out_loc = self.engine.page_table[input_mapping]    # GPU 上查 page_table 得写出位置
    self.engine.attn_backend.prepare_metadata(batch)         # Attention 元数据（FlashInfer 等）

    return ForwardInput(
        batch=batch,
        sample_args=self.engine.sampler.prepare(batch),
        input_tuple=input_mapping,
        write_tuple=write_mapping,
    )
```

以 `_make_positions` 为例，异步 H2D 拷贝的标准写法：

```python
def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)  # ← 固定内存
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(req.cached_len, req.device_len, dtype=torch.int32,
                     out=indices_host[offset : offset + length])
        offset += length
    return indices_host.to(device, non_blocking=True)  # ← 异步 DMA，立即返回！
```

`pin_memory=True`（页锁定内存）是异步 H2D DMA 的前提条件。普通内存（可分页内存）无法直接 DMA，CUDA 驱动需要先把数据拷贝到内部的 pin_memory 临时区再传输，这会强制同步。使用 `pin_memory=True` 直接分配固定内存，`non_blocking=True` 才能真正做到异步传输。

### 3.4 `_forward`：launch GPU 前向（在 engine stream 内执行）

```python
def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
    # 此函数在 engine stream 上下文内被调用（由 with self.engine_stream_ctx 保障）
    batch, sample_args, input_mapping, output_mapping = forward_input
    batch.input_ids = self.token_pool[input_mapping]  # 从 token_pool 读输入 token（共享 CPU 固定内存）
    if ENV.OVERLAP_EXTRA_SYNC:  # issue #58 的 workaround
        self.stream.synchronize()  # 强制额外同步（调试/稳定性用）
    forward_output = self.engine.forward_batch(batch, sample_args)  # ← 核心 GPU 前向
    self.token_pool[output_mapping] = forward_output.next_tokens_gpu  # 写回新生成的 token
    self.decode_manager.filter_reqs(forward_input.batch.reqs)         # 更新 decode 请求集合
    return forward_output
```

`engine.forward_batch` 的内部实现（`python/minisgl/engine/engine.py`）：

```python
def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
    assert torch.cuda.current_stream() == self.stream  # 必须在 engine stream 上运行
    with self.ctx.forward_batch(batch):
        if self.graph_runner.can_use_cuda_graph(batch):
            logits = self.graph_runner.replay(batch)  # Decode 阶段：CUDA Graph 重放
        else:
            logits = self.model.forward()             # Prefill 阶段：普通前向

    for req in batch.reqs:
        req.complete_one()  # cached_len = device_len; device_len += 1

    next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)  # ← 异步 D2H 拷贝！
    copy_done_event = torch.cuda.Event()
    copy_done_event.record(self.stream)  # ← 在 D2H 拷贝完成后 record，记录完成时间戳
    return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
```

`forward_batch` 的最后两行是 Overlap 机制的另一个要点：
- `next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)` 将 D2H 拷贝异步提交到 engine stream，立即返回
- `copy_done_event.record(self.stream)` 在 D2H 拷贝完成时自动 record 事件

这样，下一轮迭代处理结果时，CPU 只需 `copy_done_event.synchronize()`，**精准地只等这一次 D2H 拷贝**，无需等待下一个批次的 GPU 计算。

### 3.5 `_process_last_data`：处理上批结果（纯 CPU，与当前 GPU 计算并行）

```python
def _process_last_data(self, last_data: ForwardData | None) -> None:
    if last_data is None:
        return

    batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
    copy_done.synchronize()  # ← 系统中唯一的 CPU-GPU 同步点！等待 D2H 拷贝完成
    reply: List[DetokenizeMsg] = []
    new_finished_reqs: Set[Req] = set()
    with self.cache_manager.lazy_free_region():
        for i, req in enumerate(batch.reqs):
            if isinstance(req, ChunkedReq):
                continue  # Chunked Prefill 请求不做采样，跳过
            next_token = next_tokens_cpu[i]
            req.append_host(next_token.unsqueeze(0))  # 追加生成的 token 到 CPU 侧 input_ids
            next_token = int(next_token.item())
            finished = not req.can_decode
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id  # EOS 检测

            reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            if finished and req not in self.finished_reqs:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)   # 释放 table_idx + KV Cache 页面
                new_finished_reqs.add(req)
            elif batch.is_prefill:
                self.cache_manager.cache_req(req, finished=False)  # 缓存 Prefill 前缀到 Radix Tree

    self.finished_reqs = new_finished_reqs
    self.send_result(reply)  # ZMQ 发送生成的 token 给 detokenizer
```

`_process_last_data` 是纯 CPU 操作，**在时间上与当前批次的 GPU 前向完全重叠**。正是这个重叠，消除了 CPU 调度开销对系统吞吐的负面影响。

---

## 4. 完整实例演示：三个连续迭代

以三个 decode 批次（batch A、B、C）为例，逐步追踪 Overlap Scheduling 的执行状态。

### 初始状态

- batch A 已经完成 GPU 前向，`ForwardData_A = (ForwardInput_A, ForwardOutput_A)` 持有结果
- 系统中有待 decode 的请求集合

### Round 1：迭代处理 batch B，同时处理 batch A 的结果

```
进入 overlap_loop(last_data=ForwardData_A)

① blocking=False（last_data 非 None），接收 ZMQ 消息（若有）

② _schedule_next_batch() → 在 scheduler stream 上：
   - decode_manager.schedule_next_batch() → batch B（含 N 个请求）
   - allocate_paged(batch_B.reqs)           ← CPU 分配 KV Cache 页
   - _make_positions(batch_B)               ← pin_memory → 异步 H2D，提交到 scheduler stream
   - _make_input_tuple(batch_B)             ← 同上
   - _make_write_tuple(batch_B)             ← 同上
   - prepare_metadata(batch_B)              ← FlashInfer 等 backend 准备元数据
   → 返回 ForwardInput_B（快照）

③ with engine_stream_ctx:
   engine.stream.wait_stream(self.stream)   ← 等待 scheduler stream 上所有 H2D 完成
   _forward(ForwardInput_B):
     batch_B.input_ids = token_pool[input_mapping]
     forward_batch(batch_B):
       logits = graph_runner.replay(batch_B)   ← GPU CUDA Graph 重放（异步）
       next_tokens_gpu = sampler.sample(logits) ← GPU 采样（异步）
       next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)  ← 异步 D2H
       copy_done_B.record(engine.stream)       ← 记录 D2H 完成事件
     token_pool[output_mapping] = next_tokens_gpu  ← 写回 GPU（异步）
   → 返回 ForwardData_B
   ← GPU 在后台执行 batch B，CPU 立即往下

④ _process_last_data(ForwardData_A):
   copy_done_A.synchronize()                 ← 等待 batch A 的 D2H 完成（通常已经完成）
   for req in batch_A.reqs:
     next_token = next_tokens_cpu_A[i]       ← 读取 batch A 的 CPU 端 token
     检测 EOS、更新状态、发 ZMQ
   send_result(reply_A)                      ← 发送 batch A 的结果给 detokenizer

返回 ForwardData_B
```

### Round 2：迭代处理 batch C，同时处理 batch B 的结果

```
进入 overlap_loop(last_data=ForwardData_B)

  此时：GPU 仍在执行 batch B（如果还没完成）
  ↓ 与之完全并行：

① 接收新消息

② CPU 调度 batch C（all H2D 异步，提交到 scheduler stream）
   → ForwardInput_C

③ engine.stream.wait_stream(self.stream)
   _forward(ForwardInput_C)   ← GPU launch batch C（异步）
   → ForwardData_C

④ _process_last_data(ForwardData_B)：
   copy_done_B.synchronize()  ← 等 batch B 的 D2H（此时 batch C 的 GPU 计算已在后台跑）
   处理 batch B：EOS 检测 + ZMQ 发送
   ← 这些 CPU 工作与 GPU 执行 batch C 完全重叠！

返回 ForwardData_C
```

### 关键时序图（超详细 ASCII 甘特图）

```
时间轴（毫秒级）────────────────────────────────────────────────────────────────────────────────►
          0ms    1ms   2ms              12ms                                    22ms

════════════════════════════════════════════════════════════════════════════════════════════════
【Round 1】处理 batch B，并行后处理 batch A
════════════════════════════════════════════════════════════════════════════════════════════════

▼ CPU 主线程（scheduler stream）
│
├─[0.0-0.2ms] receive_msg()          ◄── 接收 ZMQ 新请求（非阻塞）
│             _process_one_msg()
│
├─[0.2-0.5ms] schedule_next_batch()  ◄── 决定哪些请求进入 batch B
│             decode_manager.schedule() → 返回 8 个 decode 请求
│
├─[0.5-0.7ms] allocate_paged()       ◄── CPU 分配 KV Cache 物理页
│             cache_manager: evict LRU, allocate 16 pages
│             更新 page_table[req_id] 映射
│
├─[0.7-1.0ms] _make_positions()      ◄── pin_memory + H2D async (提交到 scheduler stream)
│             _make_input_tuple()    ◄── pin_memory + H2D async
│             _make_write_tuple()    ◄── pin_memory + H2D async
│             prepare_metadata()     ◄── FlashInfer 构建 qo_indptr, kv_indptr
│             （这些 H2D 拷贝在 GPU 上异步执行，CPU 已返回！）
│
├─[1.0-1.1ms] 切换到 engine stream   ◄── with self.engine_stream_ctx:
│             wait_stream()          ◄── GPU 流间同步：等 H2D 完成
│             _forward(B):           ◄── launch GPU 内核（非阻塞！）
│               token_pool[mapping] → batch.input_ids
│               engine.forward_batch(B)  → 提交 Attention+FFN+Sample 内核
│               token_pool[mapping] ← next_tokens_gpu
│             （CPU 立即返回，不等 GPU 完成！）
│
├─[1.1-1.5ms] _process_last_data(A)  ◄──┐
│             copy_done_A.synchronize() │  关键！CPU 处理 batch A 的结果
│               （通常立即返回，D2H 早已完成）│  与 GPU 执行 batch B 并行
│             for req in batch_A:      │
│               next_token = next_tokens_cpu_A[i]  ← 读 CPU 端数据
│               if next_token == eos:  │
│                 free_resources(req)  │
│             send_result(reply_A)     │  ← ZMQ 发送 batch A 的 token
│                                      │
└─[1.5ms] 返回 ForwardData_B ────────►│
                                       │
════════════════════════════════════════════════════════════════════════════════
▼ GPU 硬件（engine stream）            │
│                                      │  ┌─────────────────────────────────┐
├─[0.7-2.0ms] H2D 异步拷贝             │  │ CPU 处理 A (1.1-1.5ms)           │
│   (scheduler stream DMA)             │  │     与                          │
│   positions: 512 bytes               │  │ GPU 计算 B (2.0-12.0ms)         │
│   input_mapping: 1024 bytes          │  │     完全并行！                   │
│   write_mapping: 512 bytes           │  │     ▲                          │
│                                      │  │     这就是 Overlap 的核心收益！ │
├─[2.0-2.1ms] wait_stream 同步点       │  └─────────────────────────────────┘
│   （GPU 硬件等待 H2D 完成）           │
│                                      │
├─[2.1-11.5ms] 模型前向 ★★★ 大头 ★★★  ◄─┘ （batch B，8 个请求，每个 seq_len=100）
│   ┌─ Layer 0-39（40 层 Transformer）:
│   │   [2.1-2.8ms]  Attention (PagedAttention kernel)
│   │                - 读取 KV Cache：8 reqs × 100 tokens × 32 heads
│   │                - Q @ K^T, Softmax, @ V
│   │   [2.8-3.3ms]  FFN (Gemm + GeLU)
│   │   ...
│   │   [10.5-11.0ms] Layer 39 Attention
│   │   [11.0-11.5ms] Layer 39 FFN
│   └─ lm_head: [11.5-11.8ms] Linear(4096 → vocab_size)
│
├─[11.8-12.0ms] Sampling              ◄── Top-P / Temperature 采样
│   next_tokens_gpu = sampler.sample(logits)
│
├─[12.0-12.2ms] D2H 异步拷贝          ◄── GPU → CPU (non_blocking)
│   next_tokens_cpu ← next_tokens_gpu
│   copy_done_B.record(engine.stream)  ← 记录事件
│
└─[12.2ms] batch B GPU 计算完成


════════════════════════════════════════════════════════════════════════════════════════════════
【Round 2】处理 batch C，并行后处理 batch B
════════════════════════════════════════════════════════════════════════════════════════════════

▼ CPU 主线程（scheduler stream）
│
├─[12.2-12.4ms] receive_msg() ...
├─[12.4-12.7ms] schedule_next_batch() → batch C
├─[12.7-13.0ms] allocate_paged() + _make_*() → H2D async
├─[13.0-13.1ms] wait_stream() + _forward(C) → launch GPU
│
├─[13.1-13.5ms] _process_last_data(B) ◄──┐
│             copy_done_B.synchronize()   │  CPU 处理 batch B 结果
│               （D2H 在 12.2ms 完成，立即返回）│  与 GPU 执行 batch C 并行
│             处理 batch B 的 8 个请求     │
│             send_result(reply_B)        │
│                                         │
└─[13.5ms] 返回 ForwardData_C ──────────►│
                                          │
════════════════════════════════════════════════════════════════════════════════
▼ GPU 硬件（engine stream）               │
│                                         │
├─[12.7-14.0ms] H2D 拷贝（batch C）       │  ┌─────────────────────────────────┐
├─[14.0-14.1ms] wait_stream 同步          │  │ CPU 处理 B (13.1-13.5ms)         │
├─[14.1-23.5ms] 模型前向（batch C）◄──────┘  │     与                          │
│   40 层 Transformer × Attention+FFN      │ GPU 计算 C (14.1-23.5ms)        │
├─[23.5-23.7ms] Sampling                    │     完全并行！                   │
├─[23.7-23.9ms] D2H 拷贝 + record           │                                  │
│                                           └─────────────────────────────────┘
└─[23.9ms] batch C GPU 计算完成


════════════════════════════════════════════════════════════════════════════════════════════════
【性能对比】Normal vs Overlap 模式
════════════════════════════════════════════════════════════════════════════════════════════════

Normal 模式（串行）：
  时间轴: [CPU调度1ms] [GPU前向10ms] ◄─ CPU等待 ─► [CPU处理0.5ms] [CPU调度1ms] ...
  每批次总耗时: 1 + 10 + 0.5 = 11.5ms
  GPU 利用率: 10 / 11.5 = 87%

Overlap 模式（并行）：
  时间轴: [CPU调度1ms+launch] [GPU前向10ms 与 CPU处理0.5ms并行] [CPU调度1ms+launch] ...
  每批次有效耗时: max(10, 1+0.5) = 10ms（CPU开销完全隐藏！）
  GPU 利用率: ~100%（几乎无空闲）

吞吐量提升: 11.5 / 10 = 1.15x （提升 15%）

在 CPU 调度开销更大的场景（如复杂的 Radix Cache 查找），提升可达 20-30%！
```

---

## 4.4 CPU 和 GPU 的详细操作分解表

下表详细列出了在 Overlap 模式下，CPU 和 GPU 在每个时间片内的具体操作：

| 时间片 | CPU 操作（scheduler stream） | GPU 操作（engine stream） | 数据流向 | 同步点 |
|---|---|---|---|---|
| **T0: 迭代开始** | | | | |
| 0-200μs | `receive_msg(blocking=False)` - ZMQ 接收新请求<br>`_process_one_msg()` - 将请求加入队列 | [空闲] | ZMQ → CPU 内存 | 无 |
| 200-500μs | `schedule_next_batch()` - 遍历 prefill/decode 队列<br>`decode_manager.schedule()` - 选择 batch_N 的请求<br>`pad_batch()` - 对齐到 CUDA Graph 尺寸 | [空闲] | 无 | 无 |
| 500-700μs | `allocate_paged(batch_N.reqs)` - CPU 侧分配逻辑<br>• 查找 Radix Cache 共享前缀<br>• 从 free_pages 分配新页<br>• 更新 page_table 映射表（CPU 端） | [空闲] | 无 | 无 |
| **T1: H2D 准备** | | | | |
| 700-800μs | `_make_positions(batch_N)` - CPU 构建 position 索引<br>• `torch.empty(pin_memory=True)` - 分配固定内存<br>• 计算每个 token 的位置编码索引<br>• `.to(device, non_blocking=True)` - 提交 H2D DMA | [开始] scheduler stream 的 DMA 引擎启动传输 | CPU pin_memory → GPU VRAM | 异步，无阻塞 |
| 800-900μs | `_make_input_tuple(batch_N)` - 构建 token 读取映射<br>• 计算 (req_id, token_offset) 的二维索引<br>• `.to(device, non_blocking=True)` | scheduler stream DMA 继续传输 | CPU pin_memory → GPU VRAM | 异步，无阻塞 |
| 900-1000μs | `_make_write_tuple(batch_N)` - 构建 token 写回映射<br>`prepare_metadata(batch_N)` - FlashInfer/PagedAttention 元数据 | scheduler stream DMA 继续传输 | CPU pin_memory → GPU VRAM | 异步，无阻塞 |
| **T2: launch GPU** | | | | |
| 1000-1020μs | 切换到 engine stream：<br>`with self.engine_stream_ctx:` | [等待] engine stream 等待被唤醒 | 无 | 无 |
| 1020-1030μs | `engine.stream.wait_stream(self.stream)` - CPU 调用 | **[同步点1]** GPU 硬件插入栅栏：<br>engine stream 等待 scheduler stream 上所有已提交的 H2D 完成 | 无 | GPU 流间同步<br>（CPU 不阻塞） |
| 1030-1050μs | `batch_N.input_ids = token_pool[mapping]` - 读取输入<br>`_forward(batch_N)` - 提交 GPU 内核 | [等待 H2D 完成中...] | 无 | 无 |
| 1050-1100μs | `engine.forward_batch(batch_N)` 内部：<br>• `model.forward()` - 提交 Attention 内核到 engine stream<br>• `sampler.sample()` - 提交 Sampling 内核<br>• `next_tokens_cpu = .to('cpu', non_blocking=True)` - 提交 D2H<br>• `copy_done_N.record(engine.stream)` - 标记 D2H 事件<br>• `token_pool[mapping] = next_tokens_gpu` - 提交写回内核<br>`return ForwardOutput` - **CPU 立即返回！** | [准备就绪] H2D 完成（~2.0ms），GPU 开始执行 batch_N 的内核队列 | GPU VRAM（内部读写）<br>GPU → CPU (D2H 异步) | 无（全异步） |
| **T3: 并行阶段** | | | | |
| 1100-1500μs | **`_process_last_data(batch_{N-1})`** ◄─ 关键！<br>`copy_done_{N-1}.synchronize()` - CPU 阻塞等待 | **batch_N 的 GPU 前向计算** ◄─ 关键！<br>Layer 0 Attention Kernel 执行 | 无 | **[同步点2]** CPU 等待上一批次的 D2H<br>（通常立即返回，D2H 早已完成） |
| 1500-1600μs | • `next_token = next_tokens_cpu_{N-1}[i]` - 读取 CPU 端数据<br>• `if next_token == eos_token_id:` - EOS 检测<br>• `free_resources(req)` - 释放 table_idx, KV Cache 页 | Layer 0 FFN → Layer 1 Attention → ... | CPU 本地内存读取 | 无 |
| 1600-1800μs | • `send_result(reply)` - ZMQ 发送 token 给 detokenizer<br>• 返回到主循环，准备下一次迭代 | Layer 5 → Layer 10 → ... | CPU → ZMQ | 无 |
| **T4: GPU 继续** | | | | |
| 1800-11500μs | [已返回主循环，可能在处理下一批次 batch_{N+1}] | **batch_N 的 40 层 Transformer 计算**<br>• PagedAttention 读取 KV Cache<br>• Gemm, LayerNorm, GeLU, ...<br>• 每层约 250μs，共 10ms | GPU VRAM 内部<br>（HBM 高带宽访问） | 无 |
| 11500-11800μs | [已在处理 batch_{N+1}] | **batch_N Sampling**<br>• Top-K/Top-P kernel<br>• `next_tokens_gpu` 生成完成 | GPU VRAM | 无 |
| 11800-12000μs | [已在处理 batch_{N+1}] | **batch_N D2H 拷贝**<br>• `next_tokens_cpu ← next_tokens_gpu`<br>• DMA 引擎后台传输<br>• `copy_done_N.record()` - 标记完成 | GPU VRAM → CPU pin_memory | 异步 DMA |
| **T5: 下一轮** | | | | |
| 12000+μs | 进入下一次 `overlap_loop(last_data=ForwardData_N)`<br>• 重复 T0-T4，但这次 `_process_last_data` 处理 batch_N | 开始 batch_{N+1} 的计算 | ... | ... |

**关键观察**：
1. **1100-12000μs（~11ms）期间**，CPU 和 GPU 完全并行工作，无任何互等
2. **唯一的 CPU 阻塞同步点**是 `copy_done_{N-1}.synchronize()`（1100μs），但由于 D2H 拷贝很快（<200μs），通常立即返回
3. **CPU 的调度开销**（0-1100μs，共 1.1ms）完全被 GPU 的计算时间（2000-12000μs，共 10ms）吸收
4. **吞吐率 = 1 / max(T_gpu, T_cpu)** ≈ 1 / T_gpu，CPU 不再是瓶颈

---

## 5. 流间同步机制详解

Overlap Scheduling 的正确性依赖以下三个同步原语。

### 5.1 `engine.stream.wait_stream(scheduler_stream)` — 跨流数据依赖同步

```
scheduler stream:  [H2D: positions] → [H2D: input_mapping] → [H2D: write_mapping]
                                                                    ↓
                                                    engine.stream.wait_stream(scheduler_stream)
                                                                    ↓
engine stream:                                         [Attention Kernel（读取上面的数据）]
```

这是一个 GPU 内部的流同步，**不会阻塞 CPU**。它告诉 GPU 驱动："engine stream 上的后续工作，必须等 scheduler stream 上的所有已提交工作全部完成"。CPU 调完 `wait_stream` 后立即继续执行，GPU 硬件负责等待依赖。

### 5.2 `copy_done_event.record()` + `copy_done_event.synchronize()` — 精准 D2H 等待

```python
# 在 engine stream 上 record（D2H 拷贝完成后）：
copy_done_event = torch.cuda.Event()
copy_done_event.record(self.stream)  # record 时，stream 上前面的所有操作已完成

# 在下一轮迭代的 CPU 端等待：
copy_done.synchronize()  # CPU 阻塞，直到该事件被 record（即 D2H 完成）
```

`copy_done_event.synchronize()` 是整个系统中**唯一的 CPU 阻塞等待 GPU 点**，且它等待的只是 D2H 拷贝，不是整个模型前向。在实际系统中，`batch N` 的 D2H 拷贝通常在 `batch N+1` 的调度工作（~1ms）结束前就已完成，因此这个 `synchronize()` 几乎是零延迟等待。

### 5.3 `pin_memory=True` + `non_blocking=True` — 异步 H2D 的前提

```python
mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
# ... 填充 CPU 数据 ...
return mapping_host.to(device, non_blocking=True)  # ← 真正的异步 DMA
```

| 属性 | 无 pin_memory | 有 pin_memory |
|---|---|---|
| `non_blocking=True` 效果 | 伪异步（CUDA 内部仍需同步拷贝到临时缓冲区） | 真正异步 DMA，CPU 立即返回 |
| 内存带宽 | 较低（pageable memory） | 较高（locked memory）|
| 额外开销 | 无 | 内存分配稍慢，长期持有占用物理内存 |

---

## 6. Overlap vs Normal 对比

### 6.1 时序图对比

**Normal 模式（`normal_loop`）— 完全串行**：

```
时间轴 ──────────────────────────────────────────────────────────────────────────────►

CPU: [调度A]─[launch A]            [处理A:EOS,ZMQ]  [调度B]─[launch B]            [处理B]
GPU:          ────────[batch A 前向+D2H]─────────────          ────────[batch B 前向+D2H]────

瓶颈：
• GPU 空闲：[调度A] + [处理A] 期间，GPU 完全空闲
• CPU 阻塞："处理A" 要等 GPU 完成后才能开始
• 每轮总时间 = T_cpu_schedule + T_gpu_forward + T_cpu_post
```

**Overlap 模式（`overlap_loop`）— 流水线并行**：

```
时间轴 ──────────────────────────────────────────────────────────────────────────────►

CPU: [调度A]─[launch A] [调度B+处理A]─[launch B] [调度C+处理B]─[launch C] [调度D+处理C]
GPU:          [── batch A 前向+D2H ──]          [── batch B 前向+D2H ──]   [── batch C ──]

收益：
• T_cpu_schedule（~1ms）被完全隐藏在 T_gpu_forward（~10~50ms）中
• GPU 几乎没有空闲，利用率大幅提升
• 每轮有效时间 ≈ max(T_gpu_forward, T_cpu_total)，通常约等于 T_gpu_forward
```

### 6.2 核心差异对比表

| 维度 | Normal 模式 | Overlap 模式 |
|---|---|---|
| `_process_last_data` 处理时机 | 当前批次 GPU 完成后 | 下一批次 GPU 计算期间 |
| CPU-GPU 同步点 | `copy_done.synchronize()`（阻塞 GPU → CPU 全流程）| `copy_done.synchronize()`（仅等 D2H，GPU 仍可继续） |
| `last_data` 参数 | 传入 `ongoing_data`（当前批） | 传入上一次迭代的 `ForwardData`（上一批） |
| CUDA stream 数量 | 1（engine stream，全程） | 2（scheduler stream + engine stream） |
| `ForwardInput` 快照必要性 | 不需要（串行，无竞争） | **必须**（避免 IMA 数据竞争） |
| GPU 空闲时间 | 每轮 `T_cpu_schedule + T_cpu_post` | 几乎为 0 |
| 实现复杂度 | 简单 | 需仔细处理流同步与数据生命周期 |

---

## 7. 环境变量与调试

`python/minisgl/env.py` 中定义了两个相关的环境变量：

```python
DISABLE_OVERLAP_SCHEDULING = EnvBool(False)  # 默认开启 overlap
OVERLAP_EXTRA_SYNC = EnvBool(False)          # 默认关闭额外同步
```

使用方式（`MINISGL_` 前缀）：

```bash
# 关闭 Overlap Scheduling，切换回 normal_loop（用于调试或对比性能）
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl --model Qwen/Qwen2.5-7B-Instruct

# 开启额外同步（issue #58 的 workaround，解决特定环境下的 H2D 数据不一致问题）
MINISGL_OVERLAP_EXTRA_SYNC=1 python -m minisgl --model Qwen/Qwen2.5-7B-Instruct
```

`OVERLAP_EXTRA_SYNC` 在 `_forward` 中插入 `self.stream.synchronize()`，强制等待 scheduler stream 上所有 H2D 拷贝完成，牺牲部分并行性来保证 CPU 侧对 `token_pool` 的写入对 GPU 完全可见。通常不需要开启，仅在遇到生成结果异常时用于排查。

---

## 8. 系统架构中的 Overlap Scheduling

Overlap Scheduling 与 Mini-SGLang 其他核心特性的关系：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mini-SGLang 调度层 整体架构                   │
│                                                                  │
│  ┌─────────────────────┐     ┌──────────────────────────────┐   │
│  │  Radix Cache         │     │  Chunked Prefill              │   │
│  │  - 共享前缀复用      │────▶│  - 长上下文分块处理           │   │
│  │  - 减少 Prefill 计算 │     │  - 避免 OOM                  │   │
│  └──────────┬───────────┘     └──────────────┬───────────────┘   │
│             │                                │                   │
│             └────────────────┬───────────────┘                   │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              Overlap Scheduling（本文主题）              │     │
│  │                                                          │     │
│  │  scheduler stream          engine stream                 │     │
│  │  [调度N+1批次]  ───────────▶  [GPU 执行 N 批次]          │     │
│  │  [处理N-1结果]  ◀─ D2H ──   （前向 + 采样 + D2H 拷贝）  │     │
│  │                                                          │     │
│  │  "CPU 1ms 调度" 隐藏在 "GPU 10~50ms 计算" 之中           │     │
│  └─────────────────────────────────────────────────────────┘     │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              CUDA Graph（engine.py GraphRunner）         │     │
│  │  - Decode 阶段固定形状 → CUDA Graph 重放                 │     │
│  │  - 进一步减少 CPU launch overhead                        │     │
│  └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

Overlap Scheduling 处于整个调度系统的"中枢"位置：
- 它依赖 Chunked Prefill 提供的 `ChunkedReq` 标识（跳过采样）
- 它依赖 Radix Cache 提供的高效前缀缓存（减少单批次 GPU 计算量，间接扩大 Overlap 收益）
- 它为 CUDA Graph 的使用提供了正确的 stream 上下文

---

## 9. 总结

Overlap Scheduling 的核心是利用 CUDA Stream 的异步特性，将 CPU 工作和 GPU 工作**错位一拍**地并行执行，消除传统串行推理系统中 CPU 调度对 GPU 利用率的限制。

五条核心要点：

1. **双 Stream 架构**：scheduler stream 负责 CPU 侧元数据（H2D 异步拷贝），engine stream 负责 GPU 前向计算。两个流之间通过 `wait_stream` 传递数据依赖。

2. **"错位一拍"流水线**：`overlap_loop(last_data)` 中，GPU 执行 batch N，CPU 同时处理 batch N-1 的结果。`ForwardData` 作为跨迭代的"交接包"向后传递。

3. **ForwardInput 快照防止 IMA**：Overlap 模式下 CPU 与 GPU 并发访问请求数据，必须在 GPU launch 前将所有输入数据封装为不可变快照，防止 CPU 修改 GPU 正在读取的数据。

4. **CUDA Event 实现精准同步**：`copy_done_event.synchronize()` 是系统唯一的 CPU 阻塞等待 GPU 点，且只等 D2H 拷贝（不等整个前向），保证等待时间趋近于零。

5. **pin_memory + non_blocking 是异步 H2D 的基础**：无固定内存则无法实现真正的零拷贝 DMA，`non_blocking=True` 才能让 H2D 提交后立即返回 CPU。

| 特性 | NanoFlow（原始论文）| Mini-SGLang 实现 |
|---|---|---|
| 核心思路 | CPU/GPU Overlap | CPU/GPU Overlap |
| 实现机制 | 双 batch / 双 stream | 双 stream + 迭代 last_data 传递 |
| 同步原语 | CUDA Event | CUDA Event + wait_stream |
| 额外内存开销 | 双 batch 显存 | ForwardInput 快照（极小）|
| 与单进程兼容 | 需要特殊调度器 | 完全在单 Scheduler 进程内实现 |
