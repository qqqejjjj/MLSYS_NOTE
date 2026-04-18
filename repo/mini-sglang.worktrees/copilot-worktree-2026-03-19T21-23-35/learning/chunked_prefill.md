# Chunked Prefill 技术分析笔记

Chunked Prefill 是 SGLang 及其精简版 Mini-SGLang 中用于解决长上下文推理瓶颈的核心技术。它通过将长 prompt 切分为较小的块（Chunks）进行处理，显著降低了峰值显存占用，避免了 OOM（Out-Of-Memory），并且能够使得 Prefill 和 Decode 请求在一个批次中更好地混合调度。

## 1. 核心概念与系统地位

### 1.1 什么是 Chunked Prefill？
在传统的 LLM 推理调度中，一个新请求的 Prompt 无论多长（如 32K, 128K tokens），都会在第一次（Prefill阶段）被**全量**输入到模型中计算。
- **痛点**：对于超长文本，单次全量 Attention 计算会导致极高（二次方或线性）的峰值显存占用，挤压 KV Cache 空间，极易 OOM；并且会造成计算阻塞，使得其他正在 Decode 的请求面临严重的饥饿（延迟飙升）。
- **解决方案 (Chunked Prefill)**：将一个长 Prompt 切分成多个固定大小的块（Chunk，例如长度为 4096）。每个调度周期只处理其中一个 Chunk，像流水线一样将超大计算量平摊到多个阶段中。

### 1.2 Chunked Prefill 的硬件逻辑位置

```
┌────────────────────────────────────────────────────────┐
│ Scheduler (CPU)                                        │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Waiting Queue                                      │ │
│ │ [Req1(10K tokens)], [Req2(20 tokens)]              │ │
│ └─────────────────────────┬──────────────────────────┘ │
│ ┌─────────────────────────▼──────────────────────────┐ │
│ │ Prefill Manager / Prefill Adder                    │ │
│ │ 设定的 token_budget = 4096                         │ │
│ │ 遇到 Req1: 10K > 4096，切分成 Chunk1(4096)         │ │
│ │ 将 Req1 标记为 ChunkedReq，保留在队列中等待下轮    │ │
│ └─────────────────────────┬──────────────────────────┘ │
└───────────────────────────┼────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│ GPU (Device)                                           │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Attention Kernel                                   │ │
│ │ Round 1: 计算 Req1 的 Chunk1 (4096 tokens)         │ │
│ │ Round 2: 计算 Req1 的 Chunk2 (4096 tokens)         │ │
│ │ Round 3: 计算 Req1 的 Chunk3 (1808 tokens) -> 完成 │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

## 2. 核心代码组件分析 (基于 `python/minisgl/scheduler/prefill.py`)

在 `minisgl` 中，Chunked Prefill 的逻辑主要实现在 `prefill.py` 和调度器中。

### 2.1 `ChunkedReq` (中间态请求)
```python
class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager
```
当一个超长请求正在被分块处理时，它还未产生首个输出 Token，系统会用一个特定的实体 `ChunkedReq` 来表示它的状态：
- 它的 `can_decode` 为 `False`，这确保它不会过早被扔进 `decode_manager` 里进行自回归生成。

### 2.2 `PrefillAdder._add_one_req` (核心切分逻辑)
```python
remain_len = pending_req.input_len - cached_len
chunk_size = min(self.token_budget, remain_len)
is_chunked = chunk_size < remain_len
CLS = ChunkedReq if is_chunked else Req
```
逻辑非常直白：
1. 计算当前请求还有多少 token 没被计算（`remain_len`）。
2. 从设定的全局预算 `token_budget` 和 `remain_len` 取最小值，作为当前的计算块大小 `chunk_size`。
3. 如果 `chunk_size` 小于 `remain_len`，说明无法一次算完，该请求被标记为 `ChunkedReq`。

### 2.3 `PrefillManager.schedule_next_batch` (队列维护)
如果请求被 chunked，它不能丢，必须在下一轮继续算：
```python
if isinstance(req, ChunkedReq):
    pending_req.chunked_req = req
    chunked_list.append(pending_req)
# ...
# 将未完成的 chunk 请求重新放回 pending_list 头部
self.pending_list = chunked_list + self.pending_list[len(reqs) :]
```
它会被放回 `pending_list` 最前面，保证在后续调度轮次中优先吃完自己的后续 Chunk。只有当所有 Chunk 都算完，它才会变成普通的 `Req`，并进入 Decode 阶段。

## 3. 实际案例：逐步演示 Chunked Prefill

假设设定调度器的 **token_budget = 4096**（即每一轮最多计算 4096 个 prefill tokens）。

### 场景设定
**请求 R1**：输入长度为 **10,000 tokens**。
**请求 R2**：输入长度为 **1,000 tokens**，紧接着 R1 到达。

---

### **轮次 (Round) 1**
1. 调度器从队列头部拿到 R1 (10,000 tokens)。
2. 发现 $10,000 > 4096$ (`token_budget`)。
3. 执行切分：
   - 提取 **Chunk 1** (0 ~ 4095 tokens) 组装进 batch。
   - `token_budget` 降为 0。
   - R1 状态：变为 `ChunkedReq`，挂起在 `pending_list` 头部，剩余 5904 tokens 待处理。
4. **GPU 执行**：计算 R1 前 4096 个 token 的 Attention。

---

### **轮次 (Round) 2**
1. 调度器检查 `pending_list`，第一个依然是 R1 (剩余 5904 tokens)。
2. 发现 $5904 > 4096$。
3. 执行切分：
   - 提取 **Chunk 2** (4096 ~ 8191 tokens) 组装进 batch。
   - `token_budget` 降为 0。
   - R1 状态：继续为 `ChunkedReq`，剩余 1808 tokens 待处理。
4. **GPU 执行**：在 Chunk 1 的 KV Cache 基础上，继续计算 Chunk 2 的 Attention。

---

### **轮次 (Round) 3**
1. 调度器检查，拿到 R1 (剩余 1808 tokens)。
2. 发现 $1808 \le 4096$。
3. 提取 R1 最后的 **Chunk 3** (8192 ~ 9999 tokens)。
4. 此时，`token_budget` 剩余：$4096 - 1808 = 2288$。
5. 调度器继续看下一个请求 R2 (1000 tokens)。
6. 发现 $1000 \le 2288$，将 R2 的全量 Prompt 打包进当前 batch。
7. R1 状态：由于已全部计算完毕，转变为普通的 `Req`。
8. **GPU 执行**：同时计算 (R1 的后 1808 个 token) 和 (R2 的 1000 个 token)。

---

### **轮次 (Round) 4**
R1 和 R2 均完成 Prefill，现在它们同时进入 `decode_manager` 开始逐字生成（Decode）阶段。

## 4. Chunked Prefill 中如何保证 KV Cache 和上下文隔离？

将一个请求切分成多个 Chunk 跨越多个轮次计算时，如何保证它计算出来的 Attention 结果是对的，并且多个并行的 Chunked Request 的上下文不互串？这依赖于底层的 **CacheManager** 虚拟化内存和 **Paged Attention / FlashAttention Kernel** 的隔离机制。

### 4.1 独立且连续的逻辑视图 (table_idx)

在 `mini-sglang` 中，每个 Request 无论是否被 Chunk，一旦开始被处理，就会在 `TableManager` 中被分配一个独一无二的 `table_idx` (Slot)。

```python
# python/minisgl/scheduler/table.py
class TableManager:
    def allocate(self) -> int:
        return self._free_slots.pop() # 分配全局唯一的 table_idx
```
- 这个 `table_idx` 就像是该请求专属的虚拟内存空间索引。
- 请求 R1 在整个生命周期（所有 Chunk 的 prefill + 所有的 decode）内，都独占这个 `table_idx`。

### 4.2 物理页表的累积写入与映射

对于被切分为 Chunk 1, Chunk 2, Chunk 3 的请求，物理 KV Cache 页面是如何被管理和查找的呢？

在 `PrefillAdder._add_one_req` 和 `_prepare_batch` 阶段：
1. **保留进度状态**：`ChunkedReq` 实例中维护着 `cached_len`（已经计算并缓存的 token 数量，即历史 KV）和本轮需要处理的新 `input_ids` 长度（即 `extend_len`）。
2. **按需分配物理页**：每一轮只为**当前 Chunk 新增的 token** 分配物理页，并将映射关系写到该请求专属的页表行中。

```python
# python/minisgl/scheduler/cache.py 里的页表分配和写入逻辑：
def allocate_paged(self, reqs: List[Req]) -> None:
    # 针对当前批次里的请求（无论它是一个完整的请求还是一个 Chunk）
    for req in reqs:
        # req.cached_len 是之前所有 Chunk 累积计算完的长度
        first_page = div_ceil(req.cached_len, self.page_size)
        # req.device_len 是包含当前 Chunk 后，在 GPU 上即将达到的总长度
        last_page = div_ceil(req.device_len, self.page_size)
        
        # 只为两者之间的差值（即本轮 Chunk 新增的 token）分配物理页
        if last_page > first_page:
            # ... 分配空闲物理页面 ...
```

**详细实例：物理页表的累积过程**
假设 **page_size = 2**（每个物理页存 2 个 token 的 KV Cache），请求 **R1** 总长度为 5 tokens。
假设系统 `token_budget = 2`，因此 R1 会被切分成 3 个 Chunk：
- Chunk 1: 2 tokens (index 0, 1)
- Chunk 2: 2 tokens (index 2, 3)
- Chunk 3: 1 token (index 4)

**Round 1: 处理 Chunk 1 (tokens 0~1)**
- `cached_len = 0`，目标 `device_len = 2`。
- `first_page = 0`，`last_page = 1`。
- **分配**：分配 1 个物理页，假设拿到 **P8**。
- **更新页表**：`page_table[R1_idx]` = `[P8, Null, Null...]`
- **计算**：GPU 算完 token 0~1，将 KV 存入 **P8**。

**Round 2: 处理 Chunk 2 (tokens 2~3)**
- `cached_len = 2`，目标 `device_len = 4`。
- `first_page = 1`，`last_page = 2`。
- **分配**：再分配 1 个物理页，假设拿到 **P15**。
- **更新页表**：`page_table[R1_idx]` = `[P8, P15, Null...]`（累积映射！）
- **计算**：GPU 算完 token 2~3，将 KV 存入 **P15**。

**Round 3: 处理 Chunk 3 (token 4)**
- `cached_len = 4`，目标 `device_len = 5`。
- `first_page = 2`，`last_page = 3`。
- **分配**：再分配 1 个物理页，假设拿到 **P23**。
- **更新页表**：`page_table[R1_idx]` = `[P8, P15, P23...]`。
- **计算**：GPU 算完 token 4，存入 **P23**。

通过这种“按需分配、增量追加映射”的机制，R1 的全量上下文在物理显存中虽然是不连续的（P8 -> P15 -> P23），但通过 `page_table[R1_idx]` 完美串联了起来。

### 4.3 硬件内核层级 (FlashAttention/FlashInfer) 的调度与隔离

当一个 Batch 里混杂着 `Req A (Chunk 2)`, `Req B (Chunk 1)`, `Req C (全量)` 时，计算单元是如何保证 Attention Map 的隔离的？

**1. Query 端的隔离 (`cu_seqlens_q`)：**
在进行 Attention 计算时，整个 batch 的新 token（Query）会被 `torch.cat` 打平拼接成一个连续的 1D Tensor（例如 `[Q_A, Q_B, Q_C]`）。为了告诉 Kernel 谁是谁，调度器会构建一个**累加序列长度数组 (Cumulative Sequence Lengths, 简称 `cu_seqlens_q`)**。
- 假设 A(新增 4096), B(新增 2000), C(新增 1000)。
- `cu_seqlens_q` = `[0, 4096, 6096, 7096]`。
- Kernel 依据这根标尺，严格卡死计算边界，保证 A 的 Query 只跟 A 的历史计算，**天然隔离了不同的 Request**。

**2. KV 端的上下文缝合 (`cu_seqlens_k` 和 `page_table`)：**
由于采用了 Chunked Prefill，请求 A 此时是第 2 个 Chunk。在算 Attention 时，不仅要算 A 自己内部新产生的 KV，**还需要去加载 A 在上一轮算好的历史 KV**。
- `minisgl` 通过 `FlashInfer` / `PagedAttention` 的 Wrapper (如 `BatchPrefillWithPagedKVCacheWrapper`) 解决：
  - 它不仅传 `cu_seqlens_q`，还会传 `cu_seqlens_k` (KV的总长度标尺) 和 `indices` (当前 Batch 里所有请求用到的物理页索引数组)。
  - 对 Req A，Kernel 拿到它的虚拟 `table_idx`，去 `page_table` 里读出 A 这个请求独有的前几轮存好的物理页（历史 K, V），再结合当前这 4096 个新 Query 算 Attention，算完后，又将这 4096 个新生成的 K, V 存入它刚分到的新物理页里。

**可视化视图：**
```
【同一批次调度】 Batch = [Req A(Chunk 2), Req B(Chunk 1)]
- A 历史已存: tokens 0~4095 (已在物理页 P1~P4 中)
- A 本轮新增: tokens 4096~8191 (需要分配 P5~P8)
- B 本轮新增: tokens 0~2000 (需要分配 P9~P10)

【1D 拼接打平的输入序列】
Q_flatten = [ A的新 Query(4096) | B的新 Query(2000) ]
cu_seqlens_q = [0, 4096, 6096]   <-- 保证 Q 不会跨请求算！
cu_seqlens_k = [0, 8192, 10192]  <-- 告诉内核: A的KV总长是8192, B是2000

【GPU Kernel 内部读取与计算 (以 Req A 为例)】
1. Kernel 根据 cu_seqlens 提取出 A 的 4096 个 Query。
2. Kernel 顺藤摸瓜找 Req A 的 page_table 行。
3. 从 P1~P4 (历史) 和 P5~P8 (当前) 中加载出总共 8192 长度的 Key 和 Value。
4. 计算 Attention Map：[4096, 8192] (当前Chunk vs 完整上下文)。
5. 天然应用 Causal Mask，不会读到未来，也不会读到 Req B 的信息。
```

**总结**：
- **不同 Request 的隔离**：靠 1D Flatten + `cu_seqlens` 边界指针实现，Kernel 在并行时绝对不会越界。
- **跨 Chunk 的上下文衔接**：靠 `table_idx` 固定绑定的虚拟页表，使得每个后续的 Chunk 能准确无误地从物理显存里“捞回”自己前序轮次算好的历史 KV 张量。对底层的 FlashAttention 内核而言，无论是全量计算还是增量 Extend，只是 `cu_seqlens_q`（当前长度）和 `cu_seqlens_k`（含历史的总长度）的差异罢了。

### 4.4 计算视角的差异：Linear 投影 vs Attention
在一个 Batch 包含多个 Request（无论是 Chunk 还是全量）时，系统在 Transformer Block 内部的计算方式是**分裂**的：

1. **Linear 层（QKV 投影、FFN 等）—— 完美打平，一起算**：
   - 因为 `Linear` 只是单纯的矩阵乘法，没有任何时序上下文的依赖（每个 token 的计算独立）。
   - 所有的输入 token `[Q_A(4096), Q_B(2000)]` 被当作一个纯粹的二维矩阵 `[6096, hidden_dim]`。
   - GPU 直接执行一次大矩阵乘法 `MatMul([6096, hidden_dim], Weights)`，极大提高了计算密度和算力利用率。

2. **Attention 层 —— 严格按 Request 分隔算**：
   - 到了 Attention 阶段，token 之间需要发生交互（计算 Score）。
   - 这时 `cu_seqlens_q` 和 `cu_seqlens_k` 开始发挥作用。Kernel 会在内部把打平的矩阵“逻辑上”切分开。
   - 保证 A 的 Query 只跟 A 自己的 Key (包含历史 KV) 计算，B 的 Query 只跟 B 的 Key 计算。

这就是为什么 Chunked Prefill 能够把碎片化的请求拼成一个大 Batch 的核心原因：在耗费大量算力的 Linear 和 FFN 层，所有的 token 可以毫无缝隙地被拼接成一个大矩阵享受高并发的张量核心（Tensor Core）加速，而在需要上下文隔离的 Attention 层，则通过高效的指针控制实现。

---

## 5. 核心优势与性能对比

| **特性** | **传统全量 Prefill** | **Chunked Prefill** |
| :--- | :--- | :--- |
| **超长文本的峰值显存** | 极高（易 OOM） | 受限于 `token_budget`，大幅降低且可控 |
| **Decode 饥饿问题** | 严重（处理一个 128K prompt 可能耗时几秒，期间其他用户的生成停滞） | 解决（Decode 请求可以和 Chunked Prefill 混合组 batch，每步开销稳定）|
| **GPU 利用率** | 不均匀（长文本时计算绑，生成时带宽绑） | 极高（Prefill 填缝机制提高了整体计算密度） |

## 6. 总结
Chunked Prefill 将大块的、易导致 OOM 和阻塞的计算任务，切分成了**大小均匀的流水线微步（Micro-steps）**。通过引入 `ChunkedReq` 这种中间状态，系统保证了超长请求即使被拆散计算也能正确累积状态。在服务长文本模型（如 128K/1M 上下文）时，这是确保系统不崩溃和保障其他请求时效性的底层基石。