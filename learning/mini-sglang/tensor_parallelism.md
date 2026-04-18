# Tensor Parallelism 技术分析笔记

Tensor Parallelism（张量并行，TP）是 Mini-SGLang 中用于将单个 Transformer 模型的推理计算分散到**多张 GPU** 上执行的核心分布式技术。它通过将模型权重矩阵按特定维度切分，让每张 GPU 只持有并计算权重的一部分，从而突破单卡显存和计算瓶颈，支持更大的模型和更高的吞吐量。

## 1. 核心概念与问题背景

### 1.1 为什么需要 Tensor Parallelism？

大型 LLM（如 Llama-3 70B）的参数通常超过单张 GPU 的显存容量。即使模型能放入单卡，单卡的计算带宽也可能成为瓶颈。Tensor Parallelism 从两个维度解决问题：

- **显存问题**：权重矩阵按 GPU 数量均分，每张 GPU 只存 `1/N` 的权重
- **计算问题**：矩阵乘法按维度切分，多卡并行计算，吞吐量线性扩展

```
单卡（TP=1）：
┌─────────────────────────────────────────────────────────┐
│ GPU 0                                                   │
│ 持有全量权重 W ∈ R[hidden_size × intermediate_size]     │
│ 计算：Y = X @ W  (完整矩阵乘法)                          │
│ 显存：W = 4096 × 11008 × 2bytes = ~88MB (单层 FFN)      │
└─────────────────────────────────────────────────────────┘

四卡（TP=4）：
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ GPU 0            │ │ GPU 1            │ │ GPU 2            │ │ GPU 3            │
│ W[:,0:2752]      │ │ W[:,2752:5504]   │ │ W[:,5504:8256]   │ │ W[:,8256:11008]  │
│ Y0 = X @ W_local │ │ Y1 = X @ W_local │ │ Y2 = X @ W_local │ │ Y3 = X @ W_local │
│ 显存只需 ~22MB   │ │ 显存只需 ~22MB   │ │ 显存只需 ~22MB   │ │ 显存只需 ~22MB   │
└──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘
          ↓                     ↓                     ↓                     ↓
                    AllReduce / AllGather 合并结果
                    Y = Y0 + Y1 + Y2 + Y3（Row Parallel 情况）
```

### 1.2 Tensor Parallelism 在系统中的位置

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 进程启动层（CPU, launch.py）                                                   │
│                                                                              │
│   launch_server()                                                            │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  for i in range(world_size):  # world_size = tp_size               │    │
│   │      mp.Process(target=_run_scheduler,                            │    │
│   │                 args=(DistributedInfo(rank=i, size=tp_size), ...)  │    │
│   │      ).start()                                                     │    │
│   │  # 每个进程独立持有一个 GPU，独立运行 Scheduler + Engine            │    │
│   └────────────────────────────────────────────────────────────────────┘    │
└──────────┬────────────────┬─────────────────┬─────────────────┬─────────────┘
           │                │                 │                 │
   ┌───────▼──────┐ ┌───────▼──────┐ ┌───────▼──────┐ ┌───────▼──────┐
   │ Scheduler 0  │ │ Scheduler 1  │ │ Scheduler 2  │ │ Scheduler 3  │
   │   GPU 0      │ │   GPU 1      │ │   GPU 2      │ │   GPU 3      │
   │ rank=0(主)   │ │   rank=1     │ │   rank=2     │ │   rank=3     │
   │              │ │              │ │              │ │              │
   │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
   │ │ Engine   │ │ │ │ Engine   │ │ │ │ Engine   │ │ │ │ Engine   │ │
   │ │QKV[0:H/4]│ │ │ │QKV[H/4:] │ │ │ │QKV[2H/4:]│ │ │ │QKV[3H/4:]│ │
   │ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │
   └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
          │                │                 │                 │
          └───────────────►│◄────────────────┘◄────────────────┘
                     NCCL AllReduce / AllGather
                     （PyNCCL 通信，NVLink / PCIe）
```

**关键设计**：每个 TP rank 是一个完全独立的 Python 进程，持有不同的权重切片。**rank 0（primary）** 额外负责接收 ZMQ 请求并广播给其他 rank。

---

## 2. 核心数据结构与初始化

### 2.1 `DistributedInfo`（`distributed/info.py`）

```python
@dataclass(frozen=True)
class DistributedInfo:
    rank: int   # 本进程的 GPU 编号，从 0 开始
    size: int   # GPU 总数（= tp_size）

    def is_primary(self) -> bool:
        return self.rank == 0  # rank 0 是"主节点"，负责协调
```

`DistributedInfo` 是全局单例，通过 `set_tp_info(rank, size)` 设置，任何层只需调用 `get_tp_info()` 即可获取当前进程的 rank 信息，以此决定自己持有权重的哪一份。

### 2.2 通信后端（`distributed/impl.py`）

```python
class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)  # 调用最新注册的 backend

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)
```

Mini-SGLang 支持两套通信后端，通过工厂模式动态切换：

| 后端 | 类 | 使用场景 | 特点 |
|---|---|---|---|
| PyNCCL | `PyNCCLDistributedImpl` | 默认（`use_pynccl=True`） | 自定义封装，绑定 CUDA stream，低延迟 |
| TorchDistributed | `TorchDistributedImpl` | 备用（`use_pynccl=False`） | 标准接口，兼容性好 |

**PyNCCL 的优势**：通过 `enable_pynccl_distributed()` 注册后，NCCL 通信内核直接运行在 engine stream 上，与模型计算无需额外同步，避免了 CUDA Stream 切换的开销。

### 2.3 `EngineConfig` 中的 TP 配置（`engine/config.py`）

```python
@dataclass(frozen=True)
class EngineConfig:
    tp_info: DistributedInfo    # 当前进程的 rank/size 信息
    use_pynccl: bool = True     # 是否使用 PyNCCL 通信后端
    distributed_timeout: float = 60.0
```

KV Cache 的大小也随 TP 自动缩减（`engine/engine.py`）：

```python
cache_per_page = (
    2  # key + value
    * config.model_config.head_dim
    * div_even(config.model_config.num_kv_heads, config.tp_info.size)  # ← 按 tp_size 均分
    * config.page_size
    * self.dtype.itemsize
    * config.model_config.num_layers
)
```

TP=4 时，每张 GPU 只需存储 `1/4` 的 KV Cache（因为 KV heads 也被分发到各 GPU），显著节省显存。

---

## 3. 权重切分策略：四类并行线性层

Mini-SGLang 在 `layers/linear.py` 中实现了四种 TP 线性层，对应不同的权重切分方式。

### 3.1 架构总览

```
Transformer Block（一层）
│
├── QKV Projection  ←── LinearQKVMerged（列并行，Q/K/V 按 head 均分到各 GPU）
│                        GPU0: Q_heads[0:H/4], K_heads[0:H/4], V_heads[0:H/4]
│                        GPU1: Q_heads[H/4:H/2], K_heads[H/4:H/2], ...
│
├── O Projection    ←── LinearOProj（行并行 + AllReduce）
│                        每 GPU 计算本地 partial output，AllReduce 求和
│
├── Gate Projection ←── LinearColParallelMerged（列并行，无需通信）
├── Up Projection   ←── （gate 和 up 合并到同一矩阵中）
│
└── Down Projection ←── LinearRowParallel（行并行 + AllReduce）
```

### 3.2 `LinearQKVMerged`：注意力头的均分

```python
class LinearQKVMerged(_LinearTPImpl):
    def __init__(self, hidden_size, head_dim, num_qo_heads, num_kv_heads, has_bias):
        tp_info = get_tp_info()
        GQA_ratio = div_even(num_qo_heads, num_kv_heads)  # Q 头与 KV 头的比例（GQA）
        local_num_kv = div_even(num_kv_heads, tp_info.size)  # 每 GPU 分配的 KV head 数
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        # local_isize = hidden_size（输入不切分，每 GPU 持有完整输入）
```

以 Llama-3 8B（32 个 Q heads，8 个 KV heads，GQA ratio=4）在 TP=4 时为例：

```
全量 QKV 权重维度：[hidden(4096), (32+8+8)×head_dim(128)] = [4096, 6144]

TP=4 切分后每 GPU：
  - KV heads per GPU = 8 / 4 = 2
  - Q heads per GPU  = GQA_ratio × 2 = 4 × 2 = 8
  - local_osize = (4+2) × 2 × 128 = 1536

GPU0 持有: Q[head 0-7],   K[head 0-1], V[head 0-1]  → weight[4096, 1536]
GPU1 持有: Q[head 8-15],  K[head 2-3], V[head 2-3]  → weight[4096, 1536]
GPU2 持有: Q[head 16-23], K[head 4-5], V[head 4-5]  → weight[4096, 1536]
GPU3 持有: Q[head 24-31], K[head 6-7], V[head 6-7]  → weight[4096, 1536]

每 GPU 只需全量 QKV 权重的 1/4 显存 ✓
```

### 3.3 `LinearOProj`：行并行 + AllReduce

```python
class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size, output_size, has_bias):
        tp_info = get_tp_info()
        local_isize = div_even(input_size, tp_info.size)  # 输入按 GPU 数切分
        local_osize = output_size                          # 输出保持全量
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)   # 本地矩阵乘，y 是"部分和"
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)           # AllReduce：各 GPU 部分和相加
        return y
```

**O Projection 的数据流**：

```
注意力计算后，每 GPU 持有本 GPU heads 的 attention 输出 o_local：

GPU0: o0 ∈ R[T, hidden/4]  →  乘以 W_O0 → partial_y0 ∈ R[T, hidden]
GPU1: o1 ∈ R[T, hidden/4]  →  乘以 W_O1 → partial_y1 ∈ R[T, hidden]
GPU2: o2 ∈ R[T, hidden/4]  →  乘以 W_O2 → partial_y2 ∈ R[T, hidden]
GPU3: o3 ∈ R[T, hidden/4]  →  乘以 W_O3 → partial_y3 ∈ R[T, hidden]
                                 ↓
              AllReduce（NCCL）: y = y0 + y1 + y2 + y3
              每 GPU 最终得到相同的完整 output y ∈ R[T, hidden] ✓
```

### 3.4 `LinearColParallelMerged` + `LinearRowParallel`：FFN 切分

FFN 使用**列并行（Column Parallel）接行并行（Row Parallel）**的经典模式，整个 FFN 只需一次通信：

```python
class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(   # 列并行，输出维度切分
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )
        self.down_proj = LinearRowParallel(             # 行并行，输入维度切分
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)   # 列并行矩阵乘，无需通信
        y = self.act_fn(gate_up)                  # SiLU/GELU 激活，本地执行
        return self.down_proj.forward(y)          # 行并行 + AllReduce（唯一通信点）
```

**FFN 数据流（TP=4，intermediate_size=14336）**：

```
输入：x ∈ R[T, 4096]，各 GPU 持有相同的 x（AllReduce 后所有 GPU 数据一致）

Step 1: 列并行 Gate/Up（无通信）
  GPU0: gate_up_local ∈ R[T, 14336/4=3584]   持有 intermediate 第 0-3583 列的权重
  GPU1: gate_up_local ∈ R[T, 3584]            持有第 3584-7167 列
  GPU2: gate_up_local ∈ R[T, 3584]            持有第 7168-10751 列
  GPU3: gate_up_local ∈ R[T, 3584]            持有第 10752-14335 列

Step 2: SiLU 激活（本地，无通信）
  GPU0: y0 = silu(gate0) * up0  ∈ R[T, 1792]  （3584 / 2，因 gate+up 各占一半）
  GPU1: y1 = ...                ∈ R[T, 1792]

Step 3: 行并行 Down + AllReduce（唯一通信）
  GPU0: partial_out0 = y0 @ W_down0  ∈ R[T, 4096]
  GPU1: partial_out1 = y1 @ W_down1  ∈ R[T, 4096]
  ...
       ↓ AllReduce（NCCL 求和）
  each GPU: out = partial0 + partial1 + partial2 + partial3  ∈ R[T, 4096] ✓
```

**关键设计**：列并行接行并行的结构将 FFN 内部通信次数从 2 次压缩到 **1 次**（只在 Down Projection 后 AllReduce）。

---

## 4. 注意力层的 TP 处理

### 4.1 `AttentionLayer` 中的 TP 感知（`layers/attention.py`）

```python
class AttentionLayer(StateLessOP):
    def __init__(self, layer_id, num_qo_heads, num_kv_heads, head_dim, ...):
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)  # 本 GPU 的 Q heads 数
        self.num_kv_heads = div_even(num_kv_heads, tp_size)  # 本 GPU 的 KV heads 数

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        q, k = self.rotary.forward(ctx.batch.positions, q, k)   # 位置编码，本地计算
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)  # 本地 Attention
        return o.view(-1, self.qo_attn_dim)  # 输出只包含本 GPU heads
```

注意力计算**完全在本 GPU 内完成**，不需要跨 GPU 通信。通信点在后续的 O Projection。

### 4.2 KV Cache 的 TP 分布

```
TP=4 时，每 GPU 只存储对应 heads 的 KV Cache：

GPU0 的 KV Cache：head 0-1 的 K、V 张量
GPU1 的 KV Cache：head 2-3 的 K、V 张量
GPU2 的 KV Cache：head 4-5 的 K、V 张量
GPU3 的 KV Cache：head 6-7 的 K、V 张量

每 GPU KV Cache 显存 = 全量 KV Cache / tp_size（自动缩减）
```

`CacheManager` 无需感知 TP——只需用正确的 `cache_per_page`（已除以 tp_size）初始化即可。

---

## 5. Tensor Parallelism 与 Paged Attention KV Cache 的分布式管理

### 5.1 核心设计原则：本地化管理

Tensor Parallelism 与 Paged Attention KV Cache Management 在 Mini-SGLang 中采用 **"计算分布式，管理本地化"** 的设计哲学：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 每个 GPU Rank 的独立内存空间                                                  │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ KV Cache 物理存储（Rank 0, Head 0-1 的 K/V 张量）                    │    │
│  │ Shape: [num_pages, 2, num_kv_heads/tp_size, page_size, head_dim]   │    │
│  │                                                                     │    │
│  │ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                                   │    │
│  │ │Page0│ │Page1│ │Page2│ │Page3│ ... ← 物理 page                    │    │
│  │ └─────┘ └─────┘ └─────┘ └─────┘                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Page Table（Rank 0 独立维护）                                        │    │
│  │ Shape: [max_running_req, aligned_max_seq_len]                      │    │
│  │                                                                     │    │
│  │ req 0: [0, 4, 8, 12, ...]  ← token 位置 → page index 映射           │    │
│  │ req 1: [16, 20, 24, ...]                                           │    │
│  │ req 2: [32, 36, 40, ...]                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ CacheManager（Rank 0 独立实例）                                      │    │
│  │                                                                     │    │
│  │ free_slots: [44, 48, 52, ...]  ← 可分配的 page 列表                │    │
│  │ prefix_cache: RadixCache()      ← 本地前缀缓存管理器                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘

Rank 1, 2, 3 同样独立维护各自的 KV Cache + Page Table + CacheManager
每个 rank 持有不同 heads 的 KV Cache（例如 Rank 1 持有 head 2-3）
```

**关键要点**：
- 每个 rank 的 `CacheManager` **完全独立**运行，互不干扰
- Page table 在每个 rank 上是相同的逻辑结构，但指向不同的物理 GPU 内存
- KV Cache 物理存储按 TP heads 切分，每个 rank 只存储 `num_kv_heads / tp_size` 个 heads

### 5.2 KV Cache 的 TP 切分与 Page 映射

```python
# engine/engine.py 中的 KV Cache 初始化
cache_per_page = (
    2  # key + value
    * config.model_config.head_dim
    * div_even(config.model_config.num_kv_heads, config.tp_info.size)  # ← TP 切分点
    * config.page_size
    * self.dtype.itemsize
    * config.model_config.num_layers
)

self.kv_cache = create_kvcache_pool(
    model_config=config.model_config,     # num_kv_heads 已被 TP 感知的层读取
    num_pages=self.num_pages + 1,         # 每个 rank 分配相同数量的 pages
    page_size=config.page_size,
    device=self.device,                   # 各 rank 绑定不同 GPU
    dtype=self.dtype,
)
```

**具体案例**（Llama-3 8B，8 个 KV heads，TP=4，page_size=16）：

```
单个 page 的存储需求（每 rank）：
  - TP=1: 2 × head_dim(128) × 8 heads × 16 tokens × 2bytes × num_layers = 8KB/layer
  - TP=4: 2 × head_dim(128) × 2 heads × 16 tokens × 2bytes × num_layers = 2KB/layer（1/4）

80GB 显存 A100 可分配的 page 数量（假设 32 层模型）：
  - TP=1: ~40GB / (8KB × 32) ≈ 160,000 pages
  - TP=4: ~40GB / (2KB × 32) ≈ 640,000 pages（4 倍提升！）

但注意：每个 rank 实际分配相同数量的 pages（如都是 160,000），
TP=4 时总容量 = 160,000 pages × 4 GPUs = 640,000 pages（总 KV Cache 容量不变）
```

### 5.3 Page Table 的分布式一致性

Page table 在所有 rank 上是 **逻辑同步** 的——相同的 `(req_id, token_pos)` 映射到相同的物理 page index，但每个 rank 的该 page 存储不同 heads 的 KV：

```
TP=2，request 0 的前 8 个 token，page_size=4：

Rank 0 Page Table[req=0]:
  token[0:4]   → page 0  ← 物理存储：head 0-3 的 K/V（Rank 0 GPU 内存）
  token[4:8]   → page 1

Rank 1 Page Table[req=0]:
  token[0:4]   → page 0  ← 物理存储：head 4-7 的 K/V（Rank 1 GPU 内存）
  token[4:8]   → page 1

两个 rank 的 page index 相同（都是 0, 1），但物理内存地址完全不同！

Attention 计算时（以 token[5] 为例）：
  Rank 0:
    - Q[head 0-3] @ K[head 0-3][token 0:5]  ← K 从 Rank 0 page table[0:2] 读取
    - 输出 o[head 0-3]
  Rank 1:
    - Q[head 4-7] @ K[head 4-7][token 0:5]  ← K 从 Rank 1 page table[0:2] 读取
    - 输出 o[head 4-7]

两个 rank 独立读取各自 GPU 的 KV Cache，完全无通信！
```

### 5.4 CacheManager 的本地分配逻辑

```python
# scheduler/cache.py 中的 allocate_paged
def allocate_paged(self, reqs: List[Req]) -> None:
    needed_pages = 0
    for req in reqs:
        first_page = div_ceil(req.cached_len, self.page_size)
        last_page = div_ceil(req.device_len, self.page_size)
        if last_page > first_page:
            needed_pages += last_page - first_page  # 计算增量 pages

    if needed_pages > 0:
        allocated = self._allocate(needed_pages)    # ← 从本地 free_slots 分配
        _write_page_table(...)                       # ← 写入本地 page table
```

**关键机制**：
- `allocate_paged()` 在每个 rank 上 **独立调用**，基于相同的 `reqs` 列表（rank 0 广播）
- 每个 rank 计算出相同的 `needed_pages`（因为请求列表一致）
- 每个 rank 从 **本地 `free_slots`** 分配相同数量的 pages
- 由于 page 分配逻辑是确定性的（FIFO），各 rank 分配的 page index **天然对齐**

```
示例：新请求需要 2 个新 page

Rank 0:
  free_slots: [8, 12, 16, ...]
  allocate_paged() → 分配 [8, 12]
  Page Table[req=5] 更新：[..., 8, 12]

Rank 1:
  free_slots: [8, 12, 16, ...]  ← 与 Rank 0 相同的 index（但物理内存不同）
  allocate_paged() → 分配 [8, 12]
  Page Table[req=5] 更新：[..., 8, 12]  ← 逻辑一致

两者 page index 完全相同（8, 12），但：
  - Rank 0 page 8 存储在 GPU 0 的物理地址
  - Rank 1 page 8 存储在 GPU 1 的物理地址
```

### 5.5 Prefix Cache 的独立管理

每个 rank 的 `RadixCache` 独立维护前缀树：

```
相同的输入前缀 "Hello world"：

Rank 0 的 RadixCache:
  "Hello world" → page indices [0, 4, 8]  ← head 0-3 的 KV Cache
                  物理存储在 GPU 0 的 page 0, 4, 8

Rank 1 的 RadixCache:
  "Hello world" → page indices [0, 4, 8]  ← head 4-7 的 KV Cache
                  物理存储在 GPU 1 的 page 0, 4, 8

两个 rank 的前缀树结构完全相同，page index 对齐，但物理内存分离
```

当新请求匹配前缀时：
```python
# 在 Rank 0 和 Rank 1 上独立调用
match_result = cache_manager.match_req(req)  # ← 返回本地前缀匹配
# Rank 0 返回：cached_len=11, pages=[0,4,8]（GPU 0 内存）
# Rank 1 返回：cached_len=11, pages=[0,4,8]（GPU 1 内存）

# 各 rank 锁定各自的 cache handle
cache_manager.lock(match_result.cuda_handle)
```

### 5.6 与 CUDA Graph 的集成

CUDA Graph 录制阶段，page table 在各 rank 上独立录制：

```python
# engine/graph.py
batch.padded_reqs = [req1, req2, ..., dummy]  # ← 各 rank 相同的逻辑 batch

# GraphCaptureBuffer 在各 rank 上独立分配固定 GPU 内存
self.buffer = GraphCaptureBuffer(
    input_ids=torch.zeros(..., device=self.device),  # ← 各 rank 不同 device
    out_loc=torch.zeros(..., device=self.device),    # ← 指向本地 page table indices
    ...
)

# 录制时，out_loc 指向的 page table indices 在各 rank 相同，但读取的 KV Cache 不同
```

回放时：
```python
# 各 rank 独立回放
g.replay()  # ← GPU 0 从 page table[8] 读取 head 0-3 的 KV
            # ← GPU 1 从 page table[8] 读取 head 4-7 的 KV
```

### 5.7 完整流程：一个请求的 Page 生命周期（TP=2）

```
Step 1: 请求到达（只在 Rank 0）
  Rank 0: 接收 ZMQ 请求，广播给 Rank 1
  Rank 1: 接收广播，两者现在持有相同的请求信息

Step 2: Prefix Cache 匹配（各 rank 独立）
  Rank 0: match_req() → cached_len=100, handle_0
  Rank 1: match_req() → cached_len=100, handle_1
  ⇒ cached_len 相同（确定性匹配），但 handle 指向不同 GPU 的物理内存

Step 3: Page 分配（各 rank 独立，但 index 对齐）
  Rank 0: allocate_paged() → 分配 pages [128, 132, 136]（GPU 0 内存）
  Rank 1: allocate_paged() → 分配 pages [128, 132, 136]（GPU 1 内存）
  ⇒ page indices 相同，物理地址不同

Step 4: Forward 计算（本地 + 通信）
  Rank 0:
    - QKV 投影：本地计算，输出 Q[head 0-1], K[head 0-1], V[head 0-1]
    - Attention：从 page table[128:137] 读取 K/V（GPU 0），计算 o[head 0-1]
    - O Projection：o[head 0-1] @ W_O0 → partial_y0
    - AllReduce: partial_y0 + partial_y1 → 完整输出（通信点）
  Rank 1: 同上，处理 head 2-3

Step 5: KV Cache 写入（各 rank 独立）
  通过 out_loc 张量（指向 page table indices），注意力层将新生成的 K/V 写入：
  Rank 0: store_kv() → 写入 page 136（GPU 0 物理内存，head 0-1）
  Rank 1: store_kv() → 写入 page 136（GPU 1 物理内存，head 2-3）

Step 6: Prefix Cache 插入（各 rank 独立）
  Rank 0: insert_prefix() → 插入本地前缀树，返回 new_handle_0
  Rank 1: insert_prefix() → 插入本地前缀树，返回 new_handle_1

Step 7: Page 释放（各 rank 独立）
  请求完成时：
  Rank 0: _free(page_indices[cached_len:]) → 释放到本地 free_slots
  Rank 1: _free(page_indices[cached_len:]) → 释放到本地 free_slots
```

### 5.8 核心优势与设计取舍

| 维度 | TP=1 | TP=4 |
|---|---|---|
| 单 GPU KV Cache 大小 | N pages | N pages（物理相同） |
| 总 KV Cache 大小 | N pages | N pages（非 4N！） |
| 单 GPU 可服务 token 数 | N × page_size | N × page_size |
| 总可服务 token 数 | N × page_size | N × page_size |
| Page 管理复杂度 | 简单 | 简单（本地化） |
| 通信开销 | 0 | 2 × AllReduce/layer（仅计算） |
| KV Cache 访问 | 本地 | 本地（无跨 GPU 访问） |

**关键洞察**：
1. **KV Cache 不增加规模**：TP 只切分计算，每个 GPU 仍存储完整序列的部分 heads KV，总容量不变
2. **本地化管理免通信**：Page 分配/释放完全独立，无需 AllReduce 同步 page table
3. **确定性保证一致性**：相同的请求顺序 + 相同的分配逻辑 → page index 自然对齐
4. **Attention 本地完成**：每个 GPU 的 Q heads 只访问本 GPU 的 KV heads，无跨 GPU KV 读取

---

## 6. 完整实例演示：TP=2 的一次 Transformer 层前向

以 TP=2、hidden_size=4、intermediate_size=8 的简化 Transformer 层为例：

### Step 1：QKV 投影（列并行，无通信）

```
输入: x ∈ R[1, 4]，两 GPU 持有相同的 x

GPU0 持有 W_QKV 的前半部分（2 个 heads）：
  qkv0 = x @ W_QKV0  →  [q0, k0, v0]    q0 ∈ R[1, head_dim], k0, v0 同

GPU1 持有 W_QKV 的后半部分（2 个 heads）：
  qkv1 = x @ W_QKV1  →  [q1, k1, v1]

← 两 GPU 独立计算，无通信 →
```

### Step 2：Attention 计算（本地，无通信）

```
GPU0: o0 = Attention(q0, k0, v0, KV_Cache_0)  ← 只访问本 GPU 存储的 KV Cache
GPU1: o1 = Attention(q1, k1, v1, KV_Cache_1)

← 完全独立，无通信 →
```

### Step 3：O Projection（行并行 + AllReduce）

```
GPU0: partial_attn0 = o0 @ W_O0  ∈ R[1, 4]   （部分和）
GPU1: partial_attn1 = o1 @ W_O1  ∈ R[1, 4]   （部分和）
         ↓ AllReduce（通信点 1）
GPU0 = GPU1 = partial_attn0 + partial_attn1  ∈ R[1, 4]   ← 完整 Attention 输出
```

### Step 4：FFN（Gate/Up 列并行 → SiLU → Down 行并行 + AllReduce）

```
GPU0: gate_up0 = x @ W_gate_up0  ∈ R[1, 4]   y0 = silu(gate0)*up0  ∈ R[1, 2]
GPU1: gate_up1 = x @ W_gate_up1  ∈ R[1, 4]   y1 = silu(gate1)*up1  ∈ R[1, 2]

← 无通信 →

GPU0: partial_ffn0 = y0 @ W_down0  ∈ R[1, 4]
GPU1: partial_ffn1 = y1 @ W_down1  ∈ R[1, 4]
         ↓ AllReduce（通信点 2）
GPU0 = GPU1 = partial_ffn0 + partial_ffn1  ∈ R[1, 4]   ← 完整 FFN 输出
```

### 时序图

```
时间轴 ──────────────────────────────────────────────────────────────────────►

GPU0: [QKV 列并行] → [Attn 本地] → [O_proj 行并行] → AllReduce → [FFN Gate/Up] → [FFN Down] → AllReduce
GPU1: [QKV 列并行] → [Attn 本地] → [O_proj 行并行] → AllReduce → [FFN Gate/Up] → [FFN Down] → AllReduce
                                                          ↑                                        ↑
                                                     通信点 1                               通信点 2
                                                  （O_proj 后）                          （Down_proj 后）

一层只有 2 次 AllReduce，其余计算全部并行无通信 →
吞吐收益：Attention + FFN 计算耗时减半（2 GPU），通信开销 < 3%（NVLink）
```

---

## 7. 多进程启动与通信初始化

### 7.1 进程启动（`server/launch.py`）

```python
def launch_server():
    world_size = server_args.tp_info.size   # = tp_size，每 GPU 一个进程

    for i in range(world_size):
        new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
        mp.Process(
            target=_run_scheduler,          # 每进程：Scheduler + Engine 完整栈
            args=(new_args, ack_queue),
            name=f"minisgl-TP{i}-scheduler",
        ).start()
    # 等待所有 scheduler + tokenizer 进程就绪（ack_queue）
```

### 7.2 通信初始化（`engine/engine.py`）

```python
def _init_communication(self, config: EngineConfig):
    if config.tp_info.size == 1 or config.use_pynccl:
        # CPU 侧 Gloo group：用于元数据同步（_sync_get_memory 等）
        torch.distributed.init_process_group(backend="gloo", ...)
        tp_cpu_group = torch.distributed.group.WORLD
        # GPU 侧 PyNCCL：绑定 engine stream，低延迟 AllReduce
        enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
    else:
        # 原生 NCCL（GPU 侧）
        torch.distributed.init_process_group(backend="nccl", ...)
        tp_cpu_group = torch.distributed.new_group(backend="gloo")
    return tp_cpu_group
```

**双 Group 设计**：

| Group | Backend | 用途 |
|---|---|---|
| `tp_cpu_group` | Gloo | CPU 元数据操作：`_sync_get_memory()`，内存不均衡检测，`sync_all_ranks()` |
| PyNCCL / NCCL | NCCL | GPU AllReduce / AllGather，绑定 engine stream，不阻塞 CPU |

### 7.3 rank 0 的特殊职责

```
用户请求 ──ZMQ──► rank 0（primary）──ZMQ Pub──► rank 1, 2, 3（广播相同批次）
                       ↓
              _process_one_msg()
              add_one_req() → prefill_manager
              decode_manager.schedule()

rank 1, 2, 3 只接收广播，执行相同的 batch → 保证 AllReduce 数据语义正确
```

---

## 8. NVLink 带宽对性能的影响分析

### 8.1 通信发生在哪里？

TP 推理中每层只有 **2 次 AllReduce**（O Projection 后 + Down Projection 后）。这两次通信是性能的主要开销来源。通信带宽决定了这 2 次 AllReduce 占整个前向传播耗时的比例，从而决定了 TP 的**扩展效率**。

```
一次 Transformer Layer 的时间组成：

┌──────────────────┬──────────┬──────────────────┬──────────┐
│  QKV + Attn      │AllReduce1│  FFN Gate/Up/Down │AllReduce2│
│  (本地计算)      │(通信)    │  (本地计算)       │(通信)    │
└──────────────────┴──────────┴──────────────────┴──────────┘
         T_compute_1          T_comm_1   T_compute_2          T_comm_2

扩展效率 = T_compute / (T_compute + T_comm)
当 T_comm << T_compute 时，扩展效率 → 1（接近线性加速）
当 T_comm >> T_compute 时，扩展效率 → 0（通信成为瓶颈）
```

### 8.2 AllReduce 的通信量公式

**每次 AllReduce 传输的数据量**（Ring-AllReduce）：

$$\text{通信量} = 2 \times \frac{tp\_size - 1}{tp\_size} \times B \times H \times \text{dtype\_size}$$

当 `tp_size` 较大时，系数 $\approx 2$，公式简化为：

$$\text{通信量} \approx 2 \times B \times H \times \text{dtype\_size}$$

其中：
- $B$ = batch size（前向 token 数）  
- $H$ = `hidden_size`（4096 / 8192 等）  
- `dtype_size` = 2（fp16/bf16）

在 Mini-SGLang 中，`max_forward_len` 对应最大 $B$（`engine/config.py` 中 `max_forward_len = max_seq_len`），AllReduce 缓冲区大小也据此分配：

```python
# engine/engine.py 通信初始化
max_bytes = (
    config.max_forward_len      # 最大 B
    * config.model_config.hidden_size   # H
    * self.dtype.itemsize               # dtype_size
)
enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
```

### 8.3 主流 NVLink 带宽规格

| GPU 型号 | NVLink 版本 | 双向带宽 | 单向带宽 |
|---|---|---|---|
| A100 SXM | NVLink 3.0 | 600 GB/s | 300 GB/s |
| H100 SXM | NVLink 4.0 | 900 GB/s | 450 GB/s |
| H200 SXM | NVLink 4.0 | 900 GB/s | 450 GB/s |
| B200 SXM | NVLink 5.0 | 1800 GB/s | 900 GB/s |
| PCIe（无 NVLink）| — | ~64 GB/s | 32 GB/s |

Ring-AllReduce 中每张 GPU 的实际吞吐利用**单向带宽**。理论通信时间：

$$T_{comm} = \frac{\text{通信量}}{\text{单向带宽}}$$

### 8.4 计算时间 vs 通信时间的估算

以 **Llama-3 70B**（hidden=8192，intermediate=28672，num_layers=80）为例：

**Prefill 阶段（batch_tokens B=2048，TP=4）：**

```
每次 AllReduce 通信量：
  ≈ 2 × 2048 × 8192 × 2bytes = 67 MB

A100 NVLink 通信时间：
  67 MB / 300 GB/s ≈ 0.22 ms / AllReduce

O_proj 本地计算（FP16，A100 312 TFLOPS）：
  FLOPs = 2 × 2048 × 8192 × (8192/4) = 68.7 GFLOPs
  时间 = 68.7 GFLOPs / 312 TFLOPS ≈ 0.22 ms

FFN Down_proj 本地计算：
  FLOPs = 2 × 2048 × (28672/4) × 8192 = 240 GFLOPs
  时间 = 240 GFLOPs / 312 TFLOPS ≈ 0.77 ms

AllReduce 时间 / 计算时间比：
  O_proj layer:  0.22ms / 0.22ms = 1.0x  ← 通信 ≈ 计算，已接近瓶颈！
  FFN layer:     0.22ms / 0.77ms = 0.28x ← FFN 计算主导，效率尚可
```

**Decode 阶段（batch_size B=32，TP=4）：**

```
每次 AllReduce 通信量：
  ≈ 2 × 32 × 8192 × 2bytes = 1.05 MB

A100 NVLink 通信时间：
  1.05 MB / 300 GB/s ≈ 3.5 μs / AllReduce

O_proj 本地计算（memory-bound，A100 2 TB/s HBM）：
  权重读取量 = 8192 × (8192/4) × 2bytes = 32 MB（每张卡）
  时间 ≈ 32 MB / 2000 GB/s ≈ 16 μs

AllReduce 时间 / 计算时间比：
  3.5 μs / 16 μs = 0.22x  ← 通信开销仅占计算的 22%，扩展效率好
```

### 8.5 NVLink 带宽对扩展效率的量化影响

定义扩展效率（parallel efficiency）：

$$\eta = \frac{T_1 / tp\_size}{T_{tp\_size}} = \frac{T_{compute}/tp\_size}{T_{compute}/tp\_size + T_{comm}}$$

**Decode 场景**（B=32，Llama-3 70B，每层）：

| 硬件配置 | 单向带宽 | T_comm (每 AllReduce) | T_compute/GPU (O+FFN) | 全层扩展效率 |
|---|---|---|---|---|
| A100 × 4（NVLink） | 300 GB/s | ~3.5 μs × 2 | ~48 μs | **93%** |
| H100 × 4（NVLink） | 450 GB/s | ~2.3 μs × 2 | ~27 μs（MFU↑） | **95%** |
| A100 × 8（NVLink） | 300 GB/s | ~3.9 μs × 2 | ~24 μs | **75%** |
| PCIe × 4（无NVLink）| 32 GB/s | ~33 μs × 2 | ~48 μs | **42%** |

**Prefill 场景**（B=2048，更大通信量）：

| 硬件配置 | 单向带宽 | T_comm (每 AllReduce) | 扩展效率 |
|---|---|---|---|
| A100 × 4（NVLink） | 300 GB/s | ~220 μs | **~64%** |
| H100 × 4（NVLink） | 450 GB/s | ~147 μs | **~72%** |
| PCIe × 4（无NVLink）| 32 GB/s | ~2100 μs | **<20%** |

```
NVLink 带宽对扩展效率的影响（Decode TP=4，A100）：

带宽（GB/s）  100    200    300    450    900
              │      │      │      │      │
效率(%)       │      │      │      │      │
  100 ├────────────────────────────────────────►
   90 ├──────────────────────────●────────●───
   80 ├──────────────────●────────────────────
   70 ├──────────●────────────────────────────
   60 ├──●────────────────────────────────────
   50 ├────────────────────────────────────────
              ↑             ↑
           PCIe        NVLink 3.0

NVLink vs PCIe：扩展效率从 ~60% 提升到 ~93%（decode TP=4）
NVLink 4.0 vs 3.0：效率从 ~93% 提升到 ~95%（decode 场景改善有限）
```

### 8.6 Mini-SGLang 的通信优化：PyNCCL + CUDA Stream 重叠

单纯带宽不是全部——Mini-SGLang 通过 PyNCCL 将通信绑定到 engine stream，消除了 stream 同步开销：

```
普通 NCCL（有 stream 同步）：

engine stream: ─[O_proj kernel]──sync──              ──sync──[下一层 QKV]──►
   NCCL stream:                        ──[AllReduce]──
                                 ↑                ↑
                            sync 等待        sync 等待
                        额外开销 ~5-10 μs     额外开销 ~5-10 μs

PyNCCL（绑定 engine stream）：

engine stream: ─[O_proj kernel]──[AllReduce]──[FFN 计算]──►
                                    ↑
                              直接在 engine stream 执行，无 sync 开销！
```

**PyNCCL 对延迟的影响**：对于 Decode 场景（AllReduce 单次 3-4 μs），节省掉 2 × 10 μs 的 stream sync 相当于将 AllReduce 开销减少 **~83%**，这对小 batch decode 尤为关键。

### 8.7 不同场景的 TP 选择建议

```
决策框架（基于 NVLink 带宽与场景）：

                     是否有 NVLink？
                    ┌─────┴─────┐
                   否            是
                   ▼             ▼
              PCIe 互联      继续分析
          扩展效率差，         ▼
          TP≤2 才合算    主要是 Prefill 还是 Decode？
                       ┌──────┴──────┐
                    Prefill主导    Decode主导
                    长序列B大       小batch
                       ▼              ▼
                 计算主导，       通信量小，
                 TP=4/8效率      TP=4/8均可
                ~72%(H100)     >90% 扩展效率
```

| 使用场景 | 推荐 TP | 原因 |
|---|---|---|
| 大 batch Prefill（B>1000），有 NVLink | TP=2 或 TP=4 | Prefill 通信量大，高 TP 效率下降 |
| 小 batch Decode（B<64），有 NVLink | TP=4~8 | 通信量小，近线性加速 |
| 超大模型（70B+），显存不足 | TP=4~8（必须） | 显存约束驱动，效率次要 |
| 无 NVLink（PCIe） | TP=1 或 TP=2 | 带宽不足，TP>2 几乎无收益 |

---

## 9. 总结

Tensor Parallelism 的核心是**切分权重，并行计算，集合通信合并结果**。

六条核心要点：

1. **四类线性层策略**：`LinearQKVMerged`（列并行，注意力头均分）、`LinearOProj`（行并行+AllReduce）、`LinearColParallelMerged`（列并行，FFN gate/up）、`LinearRowParallel`（行并行+AllReduce，FFN down）覆盖 Transformer 所有矩阵乘法的 TP 切分。

2. **每层只有 2 次 AllReduce**：O Projection 后一次，Down Projection 后一次。列并行接行并行的结构将 FFN 内部通信压缩到最少。

3. **每进程一 GPU**：`mp.Process` 独立进程模型，每进程持有完整 Scheduler + Engine 栈，rank 0 作为 primary 协调 ZMQ 通信和广播批次信息。

4. **双通信 Group**：CPU 侧 Gloo 处理元数据，GPU 侧 PyNCCL 绑定 engine stream，通信不阻塞 CPU 调度，与 Overlap Scheduling 天然兼容。

5. **KV Cache 本地化管理**：每个 rank 独立维护 `CacheManager` + `RadixCache` + `free_slots`，page table 逻辑同步（相同 index）但物理分离（不同 GPU 内存），确定性分配算法保证 page index 自然对齐，Attention 完全本地访问 KV Cache，无跨 GPU 通信。

6. **KV Cache 显存优势**：`cache_per_page` 除以 `tp_size`，单 GPU KV Cache 存储需求线性降低，但总容量不变（每 GPU 存储部分 heads 的完整序列 KV）。

| 特性 | TP=1 | TP=2 | TP=4 |
|---|---|---|---|
| 权重显存 / GPU | 100% | 50% | 25% |
| KV Cache 显存 / GPU | 100% | 50% | 25% |
| 每层通信次数 | 0 | 2 次 AllReduce | 2 次 AllReduce |
| KV Cache 管理 | 单机 | 本地化（无通信） | 本地化（无通信） |
| Page 分配一致性 | N/A | 确定性对齐 | 确定性对齐 |
| 理论计算吞吐（NVLink）| 1x | ~1.9x | ~3.7x |
| 支持最大模型规模 | ~单卡上限 | ~2× 单卡 | ~4× 单卡 |
