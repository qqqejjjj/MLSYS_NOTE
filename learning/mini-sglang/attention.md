# Mini-SGLang Attention 模块深度解析

本文档对 Mini-SGLang 的 `minisgl.attention` 模块进行系统性代码分析，涵盖抽象接口设计、FlashAttention 与 FlashInfer 两套 Backend 的实现细节、元数据生命周期管理，以及 CUDA Graph 集成机制。

---

## 0. 模块架构总览

`minisgl.attention` 是推理引擎中 **Softmax Attention 计算** 的执行层，处于调用链的核心位置：

```
AttentionLayer.forward(qkv)
        │  (来自 layers/attention.py)
        ▼
ctx.attn_backend.forward(q, k, v, layer_id, ctx.batch)
        │  (BaseAttnBackend 的具体实现，由 Engine 在初始化时注入)
        ├── store_kv(k, v, out_loc, layer_id)    ← 将 KV 写入 KV Cache
        └── attention_kernel(q, k_cache, v_cache, metadata)  ← 实际 Attention 计算
```

**三个已注册的 Backend**（通过 Registry 模式管理）：

```
SUPPORTED_ATTENTION_BACKENDS:
  "fa"     → FlashAttentionBackend  (fa.py)    ← 调用 sgl-kernel FA3/FA4
  "fi"     → FlashInferBackend      (fi.py)    ← 调用 FlashInfer FA2 (prefill+decode)
  "trtllm" → TensorRTLLMBackend     (trtllm.py)← 调用 FlashInfer TRTLLM kernel
```

**文件结构**：

```
attention/
├── __init__.py   ← Registry 注册、create_attention_backend()、HybridBackend
├── base.py       ← BaseAttnMetadata、BaseAttnBackend 抽象基类
├── utils.py      ← BaseCaptureData（CUDA Graph 捕获缓冲区基类）
├── fa.py         ← FlashAttentionBackend + FAMetadata
├── fi.py         ← FlashInferBackend + FIMetadata
└── trtllm.py     ← TensorRTLLMBackend + TRTLLMMetadata
```

---

## 1. 抽象接口设计（`base.py`）

### 1.1 `BaseAttnMetadata`

```python
@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...
```

`metadata` 对象由 `prepare_metadata()` 在每个 batch 调度前创建，并挂载到 `batch.attn_metadata`。

`get_last_indices(bs)` 返回每个请求**最后一个 Q token 的位置索引**（在 `cu_seqlens_q` 中），用于 Sampling 阶段提取最后一步的 logits：

```python
# 在 Sampler 中的调用：
last_indices = batch.attn_metadata.get_last_indices(batch.size)
logits = lm_head_output[last_indices]  # [bs, vocab_size]
```

对于 Decode（每请求 1 个 token）：`get_last_indices` 返回 `[0, 1, 2, ..., bs-1]`。  
对于 Prefill（每请求多个 token）：返回每个请求最后一个 token 在 flat 张量中的位置。

### 1.2 `BaseAttnBackend` — 五个必须实现的方法

```python
class BaseAttnBackend(ABC):
    def forward(q, k, v, layer_id, batch) -> torch.Tensor:
        ...  # Attention 主计算，含 store_kv

    def prepare_metadata(batch) -> None:
        ...  # 在每个 batch 调度前，构建 batch.attn_metadata

    def init_capture_graph(max_seq_len, bs_list) -> None:
        ...  # 初始化 CUDA Graph 捕获所需的静态缓冲区

    def prepare_for_capture(batch) -> None:
        ...  # CUDA Graph 录制前，将 batch 元数据写入静态缓冲区

    def prepare_for_replay(batch) -> None:
        ...  # CUDA Graph 回放前，将动态数据同步到静态缓冲区
```

**方法职责分离**设计：
- `prepare_metadata`：动态 batch → 动态 metadata（每次调度都调用）
- `prepare_for_capture`：动态 metadata → 静态录制缓冲区（CUDA Graph 录制时调用一次）
- `prepare_for_replay`：动态 metadata → 静态回放缓冲区（CUDA Graph 回放时每次调用）

### 1.3 `HybridBackend` — Prefill/Decode 分离

```python
class HybridBackend(BaseAttnBackend):
    def __init__(self, prefill_backend, decode_backend): ...

    def forward(self, q, k, v, layer_id, batch):
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def init_capture_graph(self, max_seq_len, bs_list):
        # CUDA Graph 只捕获 Decode（Prefill 形状不固定，无法捕获）
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)
```

通过 `"fi,fa"` 这样的字符串创建 Hybrid Backend：

```python
# 来自 __init__.py
if "," in backend:
    p_backend, d_backend = backend.split(",", 1)
    return HybridBackend(
        create_attention_backend(p_backend, config),   # Prefill: FlashInfer
        create_attention_backend(d_backend, config),   # Decode:  FlashAttention
    )
```

实际应用中 `"fa,fa"` 和 `"fi,fi"` 是最常见配置，`"fi,fa"` 可用于测试不同 kernel 的组合。

---

## 2. CUDA Graph 捕获缓冲区（`utils.py`）

```python
@dataclass
class BaseCaptureData:
    seq_lens:     torch.Tensor   # [max_bs]           int32, GPU
    positions:    torch.Tensor   # [max_bs]           int32, GPU
    cu_seqlens_k: torch.Tensor   # [max_bs + 1]       int32, GPU，Decode 用 [0, 1, 2, ..., bs]
    cu_seqlens_q: torch.Tensor   # [max_bs + 1]       int32, GPU，Decode 用 [0, 1, 2, ..., bs]
    page_table:   torch.Tensor   # [max_bs, max_seq]  int32, GPU

    @classmethod
    def create(cls, max_bs, max_seq_len, device, **kwargs):
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
        )
```

**关键约束**：CUDA Graph 捕获的是一组固定的 GPU 地址。`BaseCaptureData` 中的所有 Tensor 在整个捕获周期内地址不变，回放时只需将新数据 `.copy_()` 到这些固定缓冲区，CUDA Graph 内的 kernel 自然读到最新值。

---

## 3. FlashAttention Backend（`fa.py`）

### 3.1 使用的底层 Kernel

```python
from sgl_kernel.flash_attn import flash_attn_with_kvcache
```

`sgl_kernel` 是 SGLang 维护的专用 CUDA kernel 包。Flash Attention 版本由 GPU 架构决定：

```python
def __init__(self, config):
    from minisgl.utils import is_sm100_supported
    self.version = 4 if is_sm100_supported() else 3
    # sm100 = Blackwell (B200/B100) → FA4
    # sm90  = Hopper   (H100/H200) → FA3
```

| GPU 架构 | SM 版本 | FA 版本 | 特性 |
|---|---|---|---|
| Blackwell (B200) | sm_100 | FA4 | 专为 Blackwell 优化 |
| Hopper (H100/H200) | sm_90 | FA3 | 利用 Hopper TMA + wgmma |
| Ampere (A100) | sm_80 | FA3（降级） | 无 TMA 优化 |

### 3.2 FAMetadata 结构

```python
@dataclass
class FAMetadata(BaseAttnMetadata):
    cu_seqlens_k:   torch.Tensor  # [bs+1], GPU, cumulative KV seq lengths
    cu_seqlens_q:   torch.Tensor  # [bs+1], GPU, cumulative Q seq lengths
    cache_seqlens:  torch.Tensor  # [bs],   GPU, 每个请求的 KV 历史长度
    max_seqlen_k:   int           # batch 内最大 KV 序列长度
    max_seqlen_q:   int           # batch 内最大 Q 序列长度
    page_table:     torch.Tensor  # [bs, max_seqlen_k // page_size], GPU, 物理页面索引
```

`cu_seqlens_q` 和 `cu_seqlens_k` 是 **变长序列的标准表达**：

```
# 3个请求，sequence lengths = [5, 3, 7]
# cu_seqlens = [0, 5, 8, 15]  ← prefix sum（cumulative sum，前缀和）
# 第 i 个请求的 token 范围：[cu_seqlens[i], cu_seqlens[i+1])
```

### 3.3 `prepare_metadata` — 三种情况下 `cu_seqlens_q` 的构建

```python
def prepare_metadata(self, batch: Batch) -> None:
    seqlens_q = [req.extend_len for req in reqs]   # Q 长度 = 本次新增 token 数
    seqlens_k = [req.device_len for req in reqs]   # KV 长度 = 历史 + 新 token

    if max_seqlen_q == 1:
        # ① Decode 阶段：每请求恰好 1 个新 token
        cu_seqlens_q = torch.arange(0, bs+1, ...)   # [0, 1, 2, ..., bs]

    elif all(l == 0 for l in cached_lens):
        # ② 纯 Prefill（无缓存命中）：Q 长度 = KV 长度，共用同一个 cumsum
        cu_seqlens_q = cu_seqlens_k

    else:
        # ③ 扩展 Prefill（Extend，部分缓存命中）：Q < KV，需要单独计算
        cu_seqlens_q = torch.tensor([0] + seqlens_q, ...).cumsum_(dim=0)
```

**三种情况对应 Paged Attention 中的 Extend 操作**：

```
情况①（Decode）：  cached=5, device=6 → extend_len=1
情况②（Prefill）：cached=0, device=7 → extend_len=7, Q与KV对齐
情况③（Extend）：  cached=3, device=7 → extend_len=4, Q⊂KV区间
```

### 3.4 Page Table 的构建与 page_size 支持

`page_table` 将逻辑 KV 槽位映射到物理内存页：

```python
page_table = get_global_ctx().page_table   # 全局页表：[max_reqs, max_logical_slots]

new_page_table = torch.stack([
    page_table[req.table_idx, : max_seqlen_k : self.page_size]
    for req in reqs
])
# 切片 ::self.page_size 意味着：若 page_size=16，
# 全局页表每16个槽位对应一个物理页，取每页起始槽位的索引
if self.page_size > 1:
    new_page_table.div_(self.page_size, rounding_mode="floor")
    # 将槽位偏移转换为页面偏移（以页为单位的物理地址）
```

全局页表以 `page_size=1`（每槽一页）为基准存储，FA Backend 在准备元数据时按实际 `page_size` 重新采样和转换。

### 3.5 `forward` 与 `_fa_sgl_impl`

```python
def forward(self, q, k, v, layer_id, batch):
    metadata = batch.attn_metadata
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)   # ① 写入 KV Cache
    return _fa_sgl_impl(                                    # ② 计算 Attention
        q=q,
        k_cache=self.kvcache.k_cache(layer_id),
        v_cache=self.kvcache.v_cache(layer_id),
        page_table=metadata.page_table,
        cache_seqlens=metadata.cache_seqlens,
        ...
    )

def _fa_sgl_impl(...) -> torch.Tensor:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    return flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=True,
        ver=version,        # 3 or 4，控制使用 FA3/FA4
        num_splits=num_splits,   # 0 = 自动决定 split-K 并行度
        pack_gqa=pack_gqa,       # GQA 打包优化
        ...
    )
```

`ku_seqlens_k_new` 参数名的 `_new` 后缀提示：此 `cu_seqlens_k` 是"新 KV（含本次写入的 Q）"的累计长度，区别于纯历史 KV。FA3 内核通过 `cu_seqlens_k` 和 `page_table` 的配合，实现对 Paged Memory 的跨页访问。

---

## 4. FlashInfer Backend（`fi.py`）深度分析

FlashInfer 是针对 LLM 推理场景深度优化的 Attention 计算库，设计哲学与 Flash Attention 不同：

| 特性 | FlashAttention (sgl-kernel) | FlashInfer |
|---|---|---|
| 接口风格 | 单次函数调用，自动处理所有情况 | 两阶段：`plan()` + `run()` |
| 内核版本 | FA3 (Hopper), FA4 (Blackwell) | FA2（稳定），FA3（开发中，目前较慢） |
| Decode 优化 | 通用实现 | Tensor Core decode kernel（GQA 专项） |
| 内存布局 | page_table 索引 | paged KV + flat indices |
| CUDA Graph 支持 | 直接支持 | 专用 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper` |

### 4.1 FIMetadata — 双设备元数据

```python
@dataclass
class FIMetadata(BaseAttnMetadata):
    # CPU 上的数据（用于 plan() 调用，plan 会读取 CPU 数据）
    cu_seqlens_q_cpu:   torch.Tensor  # on CPU, pin_memory=True
    cu_seqlens_k_cpu:   torch.Tensor  # on CPU, pin_memory=True
    last_page_len_cpu:  torch.Tensor  # on CPU，每个请求最后一页的已用槽位数
    seq_lens_cpu:       torch.Tensor  # on CPU，每个请求的总 KV 长度

    # GPU 上的数据（用于 run() 调用）
    cu_seqlens_q_gpu:   torch.Tensor  # on GPU
    indices:            torch.Tensor  # on GPU，flat 格式的 KV Cache 物理槽位索引

    # 配置信息
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1]    # FlashInfer 当前只支持 page_size=1
    pos_encoding_mode:  str           # "NONE"（RoPE 已在 AttentionLayer 中预先计算）
    dtype:              torch.dtype

    # FlashInfer Wrapper 对象（包含预编译的计划）
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False  # 懒惰初始化标志，控制 plan() 只调用一次
```

**关键设计**：FlashInfer 需要 `plan()` 阶段（CPU 端参数准备 + GPU 端内部缓冲区初始化），然后才能 `run()`（实际 GPU 计算）。`initialized` flag 实现懒惰初始化，确保 `plan()` 只在 metadata 首次使用时调用一次。

**`page_size=1` 的约束**：FlashInfer 把 KV Cache 视为一维的物理槽位数组（`indices` 是 flat 列表），全局页表中的每个 slot 对应一个独立的物理地址。这与 FA Backend 不同（FA 可以有任意 page_size，通过 page_table 二级寻址）。

### 4.2 初始化与工作缓冲区管理

```python
class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config):
        # 128MB Float 工作缓冲区（用于 Attention 中间计算，如 log-sum-exp 等）
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        # Prefill Wrapper：使用 FA2 kernel，NHD 布局
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",        # (num_heads, head_dim) 布局
            backend="fa2",          # 明确使用 FA2（fa3 在 FlashInfer 中目前较慢）
        )
        # Decode Wrapper：可选 Tensor Core 路径
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",
        )
        # 关键：共享 int_workspace_buffer（约 8MB，存放 FlashInfer 内部的分页索引结构）
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer
```

**两个工作缓冲区的用途对比**：

| 缓冲区 | 大小 | 用途 |
|---|---|---|
| `float_workspace_buffer` | 128 MB | Attention 中间结果（partial softmax, partial output, rescaling factors） |
| `int_workspace_buffer` | ~8 MB | KV 页面索引的内部转换结构（FlashInfer 特有的 ragged-to-CSR 转换） |

Prefill 和 Decode wrapper 共享同一个 `int_workspace_buffer` 的原因：两者不同时使用（每个 batch 要么 prefill 要么 decode），避免了重复分配 8MB 内存。

### 4.3 Tensor Core Decode 路径的选择逻辑

```python
@cached_property
def use_tensor_cores(self) -> bool:
    if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
        logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
        return overriden_value

    # GQA 比例决定是否使用 Tensor Core 路径
    GQA = self.config.num_qo_heads // self.config.num_kv_heads
    return GQA >= 4
```

**背景**：FlashInfer 的 Decode 阶段有两套实现：

```
普通 decode kernel（use_tensor_cores=False）:
  每个 KV Head 独立计算 → 线程级并行
  适合 MHA（GQA=1）或低 GQA ratio
  
Tensor Core decode kernel（use_tensor_cores=True）:
  将多个 Q Head 合并打包，利用 Tensor Core 的矩阵乘加速
  适合高 GQA（如 LLaMA-3 70B: 64 Q heads / 8 KV heads = GQA=8）
  GQA < 4 时打包开销大于收益
```

| 模型 | Q heads | KV heads | GQA | Tensor Core |
|---|---|---|---|---|
| LLaMA-3 8B | 32 | 8 | 4 | ✓ |
| LLaMA-3 70B | 64 | 8 | 8 | ✓ |
| Qwen2.5 7B | 28 | 4 | 7 | ✓ |
| MHA 模型 | 32 | 32 | 1 | ✗ |

### 4.4 两阶段调用：`plan()` + `run()`

FlashInfer 的核心设计是将 **元数据准备**（plan）与**实际计算**（run）分离：

#### Phase 1：`plan()`（CPU 端参数解析 + GPU 缓冲区初始化）

```python
@staticmethod
def _initialize_metadata_once(metadata: FIMetadata) -> None:
    if metadata.initialized:
        return   # 懒惰初始化：同一个 metadata 对象只 plan 一次
    metadata.initialized = True

    if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
        # Decode Plan：传入 paged KV 的 CSR 格式描述
        metadata.wrapper.plan(
            indptr=metadata.cu_seqlens_k_cpu,         # CPU, [bs+1]，每请求 KV 的页面数前缀和
            indices=metadata.indices,                  # GPU, flat KV 槽位索引
            last_page_len=metadata.last_page_len_cpu,  # CPU，最后一页剩余 slot 数（page_size=1时全为1）
            num_qo_heads=metadata.num_qo_heads,
            num_kv_heads=metadata.num_kv_heads,
            head_dim=metadata.head_dim,
            page_size=metadata.page_size,              # 恒为 1
            pos_encoding_mode="NONE",                  # RoPE 已在 AttentionLayer 外部完成
            seq_lens=metadata.seq_lens_cpu,
            data_type=metadata.dtype,
            non_blocking=True,                         # CPU→GPU 异步传输
        )
    else:
        # Prefill Plan：传入 Q 和 KV 的分段信息
        metadata.wrapper.plan(
            qo_indptr=metadata.cu_seqlens_q_cpu,       # CPU, Q 序列前缀和
            paged_kv_indptr=metadata.cu_seqlens_k_cpu, # CPU, KV 页面数前缀和
            paged_kv_indices=metadata.indices,         # GPU, flat KV 槽位索引
            paged_kv_last_page_len=metadata.last_page_len_cpu,
            ...
            causal=True,                               # 自回归推理使用因果 mask
            non_blocking=True,
        )
```

**`non_blocking=True` 的意义**：`plan()` 中的 CPU→GPU 数据传输使用异步流，与后续的 GPU 计算流水线重叠。只要在 `run()` 被调用前完成传输即可（CUDA stream 依赖自动保证）。

#### Phase 2：`run()`（GPU 上的实际 Attention 计算）

```python
def forward(self, q, k, v, layer_id, batch):
    metadata = batch.attn_metadata
    assert isinstance(metadata, FIMetadata)

    # 懒初始化：首次调用时触发 plan()
    self._initialize_metadata_once(metadata)

    # 写入 KV Cache（与 FA Backend 相同）
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)

    # 将 KV Cache 视图展平以匹配 page_size=1 的需求
    kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
    def _flatten_cache(cache):
        # cache: [max_pages, page_size, num_kv_heads, head_dim]（page_size=1 时 shape[1]=1）
        return cache.view(-1, 1, cache.shape[2], cache.shape[3])
        # → [total_slots, 1, num_kv_heads, head_dim]
    kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))

    # FlashInfer run()：利用 plan() 缓存的内部状态直接计算
    return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)
```

### 4.5 `prepare_metadata` — FlashInfer vs FA 的 indices 构建差异

FA Backend 使用 **二维 page_table** `[bs, max_seqlen_k // page_size]`；  
FlashInfer 使用 **一维 flat indices**（CSR 行的 value array）：

```python
def prepare_metadata(self, batch: Batch) -> None:
    page_table = get_global_ctx().page_table  # [max_reqs, max_slots_per_req]

    # FA Backend 的方式（二维 page_table）：
    # page_table[req.table_idx, :device_len]  → 每请求的物理槽位列表

    # FlashInfer 的方式（拼接为一维 flat indices）：
    indices = torch.cat([
        page_table[req.table_idx, : req.device_len]
        for req in reqs
    ])
    # 对应的 indptr（cu_seqlens_k_cpu）指明每请求 KV 的槽位范围：
    # cu_seqlens_k_cpu = [0, len_0, len_0+len_1, ..., sum(all_lens)]
```

**FlashInfer 要求的数据格式（Paged KV 的 CSR 表示）**：

```
请求 0: device_len=5, 槽位 = [42, 7, 103, 19, 55]
请求 1: device_len=3, 槽位 = [200, 11, 88]
请求 2: device_len=6, 槽位 = [1, 2, 3, 4, 5, 99]

indices（flat）: [42, 7, 103, 19, 55, 200, 11, 88, 1, 2, 3, 4, 5, 99]
indptr:         [0, 5, 8, 14]

FlashInfer Decode kernel 在处理请求 i 时：
  访问 indices[indptr[i] : indptr[i+1]] 中的物理槽位 → 对应 KV Cache 中的物理行
```

### 4.6 CUDA Graph 集成

FlashInfer 的 CUDA Graph 支持通过专用的 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper` 实现。

#### 初始化阶段 (`init_capture_graph`)

```python
def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
    max_bs = max(bs_list)
    capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
    capture.page_table = capture.page_table.view(-1)   # 展平为 1D flat indices
    self.capture = capture
    self.capture_bs = sorted(bs_list)
```

注意 FlashInfer 将 `page_table` 展平为一维（与 FA Backend 的二维不同），因为 FlashInfer 的 CUDA Graph 版本需要预先绑定 flat `indices_buffer`。

#### 录制 (`prepare_for_capture`)

```python
def prepare_for_capture(self, batch: Batch) -> None:
    bs = batch.size
    # 为每个可能的 bs 值创建独立的 CUDA Graph Wrapper
    self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
        self.float_workspace_buffer,
        kv_layout="NHD",
        use_tensor_cores=self.use_tensor_cores,
        # 绑定静态缓冲区地址（CUDA Graph 录制后地址固定）
        indptr_buffer=capture.cu_seqlens_k[: bs + 1],
        indices_buffer=capture.indices,         # 1D flat indices
        last_page_len_buffer=capture.one_tensor[:bs],
    )
    self.graph_wrappers[bs]._backend = "fa2"
    # 共享 int_workspace_buffer（与非 Graph wrapper 复用）
    self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
    # plan() 录制前必须调用一次（创建内部转换结构）
    self.prepare_metadata(batch)
    self._initialize_metadata_once(batch.attn_metadata)
```

#### 回放 (`prepare_for_replay`)

```python
def prepare_for_replay(self, batch: Batch) -> None:
    metadata, bs = batch.attn_metadata, batch.padded_size
    # 将当前 batch 的动态元数据 copy 到静态录制缓冲区
    metadata.wrapper = self.graph_wrappers[bs]   # 切换到 Graph wrapper
    # initialized=False 时，_initialize_metadata_once 会重新调用 plan()
    # 这里 metadata 是全新构建的（每次 prepare_metadata 都创建新对象），
    # 因此 initialized=False，触发新的 plan() 调用（将新的 cu_seqlens/indices 上传）
    self._initialize_metadata_once(metadata)
```

**FlashInfer CUDA Graph 的关键区别**（与 FA Backend 相比）：
- FA Backend 在回放时只需 `.copy_()` 静态缓冲区；
- FlashInfer 每次回放前须重新调用 `plan()`（更新内部的 ragged→CSR 转换结构），但因使用 `non_blocking=True` 且复用了工作缓冲区，开销极低。

---

## 5. TensorRT-LLM Backend（`trtllm.py`）

TRT-LLM Backend 是通过 FlashInfer 暴露的 TensorRT-LLM Attention Kernel 封装：

```python
def forward(self, q, k, v, layer_id, batch):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache

    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))

    if batch.is_prefill:
        return trtllm_batch_context_with_kv_cache(
            query=q, kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,   # 128MB
            block_tables=metadata.page_table,          # 二维，与 FA 相同
            seq_lens=metadata.cache_seqlens,
            bmm1_scale=self.scale, bmm2_scale=1.0,     # QK^T / sqrt(d) 的两个缩放因子
            kv_layout="NHD",
            ...
        )
    else:
        return trtllm_batch_decode_with_kv_cache(...)
```

与 FlashInfer Backend 不同，TRT-LLM kernel 使用**二维 `page_table`**（与 FA 相同），接口更接近 Flash Attention。元数据结构 `TRTLLMMetadata` 与 `FAMetadata` 几乎完全相同。TRT-LLM Backend 主要用于与 TensorRT-LLM 生态对比测试。

---

## 6. AttentionLayer：连接模型层与 Attention Backend

`layers/attention.py` 中的 `AttentionLayer` 是 Transformer 层中调用 Attention Backend 的唯一入口：

```python
class AttentionLayer(StateLessOP):
    def __init__(self, layer_id, num_qo_heads, num_kv_heads, head_dim,
                 rotary_config, q_norm=None, k_norm=None):
        tp_size = get_tp_info().size
        # Tensor Parallelism：每个 GPU 只处理部分 heads
        self.num_qo_heads = div_even(num_qo_heads, tp_size)   # = num_qo_heads / tp
        self.num_kv_heads = div_even(num_kv_heads, tp_size)   # = num_kv_heads / tp
        # RoPE 在此层管理（Backend 使用 pos_encoding_mode="NONE"，不重复做 RoPE）
        self.rotary = get_rope(head_dim, rotary_dim, ...)

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()

        # ① 分割 QKV（由上游的 QKV Linear 层拼接输出）
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)

        # ② 可选：Q/K Norm（Qwen2 等模型使用）
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))

        # ③ RoPE 位置编码（在送入 Backend 之前完成，Backend 内不再做）
        q, k = self.rotary.forward(ctx.batch.positions, q, k)

        # ④ 调用 Attention Backend（含 store_kv + attention 计算）
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)

        # ⑤ 将输出 reshape 回 [-1, qo_attn_dim] 供后续 O-Projection 使用
        return o.view(-1, self.qo_attn_dim)
```

**`StateLessOP` 的含义**：`AttentionLayer` 本身不持有模型权重（`nn.Module` 风格），所有状态（KV Cache、Backend 配置）均通过 `get_global_ctx()` 访问全局上下文，这简化了 CUDA Graph 的使用（无需担心 layer 对象的状态变化）。

---

## 7. 元数据生命周期 — 完整时序

以一次 **Decode batch（4 个请求，各有 1 个新 token）** 为例，追踪 FlashInfer Backend 的完整调用链：

```
Scheduler.step()
  │
  ├─ 1. batch = Batch(reqs, phase="decode")
  │
  ├─ 2. ctx.attn_backend.prepare_metadata(batch)
  │        │
  │        └─ FIMetadata(
  │               cu_seqlens_q_cpu = [0, 1, 2, 3, 4]  (Decode: 每请求1个Q)
  │               cu_seqlens_k_cpu = [0, 12, 25, 38, 51]  (各请求的历史KV长度)
  │               indices = [page_ids_req0..., page_ids_req1..., ...]  GPU flat
  │               last_page_len_cpu = [1, 1, 1, 1]  (page_size=1, 最后页恒为1)
  │               wrapper = self.decode_wrappers  (BatchDecodeWrapper)
  │               initialized = False  ← 尚未 plan()
  │           )
  │           batch.attn_metadata = ↑
  │
Engine.step()
  │
  └─ for layer in model.layers:
         layer.forward(qkv)
           │
           AttentionLayer.forward(qkv)
             │
             ├─ split q, k, v
             ├─ q_norm / k_norm (可选)
             ├─ rotary.forward(positions, q, k)
             └─ ctx.attn_backend.forward(q, k, v, layer_id, batch)
                  │
                  FlashInferBackend.forward(...)
                    │
                    ├─ _initialize_metadata_once(metadata)   ← 首次调用
                    │    │
                    │    └─ decode_wrapper.plan(               ← 触发 plan()
                    │           indptr=[0,12,25,38,51],        CPU→GPU async
                    │           indices=[flat GPU tensor],
                    │           ...
                    │       )
                    │    metadata.initialized = True           ← 标记已 plan
                    │
                    ├─ kvcache.store_kv(k, v, out_loc, layer_id)  ← StoreKernel
                    │
                    └─ metadata.wrapper.run(q, paged_kv_cache)    ← FA2 GPU kernel
                         └─ output: [num_tokens, num_qo_heads, head_dim]
```

---

## 8. 三个 Backend 的选择建议

| 场景 | 推荐 Backend | 原因 |
|---|---|---|
| H100/H200, MHA | `"fa"` | FA3 在 Hopper 上利用 TMA 和 wgmma，Prefill 吞吐最优 |
| H100/H200, GQA≥4 + 高并发 Decode | `"fi"` | Tensor Core Decode kernel，GQA 场景优势明显 |
| H100, 混合场景 | `"fi,fa"` 或 `"fa,fa"` | Prefill 和 Decode 使用不同优化路径 |
| Blackwell (B200) | `"fa"` | FA4 专为 Blackwell 设计，目前最优 |
| 性能对比测试 | `"trtllm"` | 与 TRT-LLM 生态 baseline 对齐 |
| page_size > 1 | `"fa"` or `"trtllm"` | FlashInfer 目前仅支持 page_size=1 |

```
性能取舍总结：

FA3 (sgl-kernel):
  ✓ 支持任意 page_size
  ✓ Hopper TMA 优化（异步内存传输，隐藏 HBM 延迟）
  ✓ SM100 上有 FA4 进一步优化
  ✗ 无专用 Tensor Core Decode kernel

FlashInfer (FA2 backend):
  ✓ Tensor Core Decode（GQA≥4 时吞吐显著更高）
  ✓ plan/run 分离使 CUDA Graph 集成更干净
  ✗ page_size 恒为 1（不支持更大页面减少碎片）
  ✗ 每次 decode 回放须重新 plan（有少量 CPU 开销）
```

---

## 9. 各 Attention Kernel 实现原理深度解析

本节聚焦于 mini-sglang 中被调用的三类 Attention Kernel 的**内部算法与硬件实现**，而非调度层的用法。

### 9.1 标准（朴素）Attention 的性能问题

理解 Flash Attention 的前提，是看清楚朴素实现的瓶颈：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V$$

朴素实现（三个独立 CUDA kernel）：

```
Kernel 1: S = Q @ K^T          # [N, N]  O(N²d) 浮点运算，写 N² 元素到 HBM
Kernel 2: P = softmax(S)       # [N, N]  读 N² 元素，写 N² 元素
Kernel 3: O = P @ V            # [N, d]  O(N²d) 浮点运算，读 N² 元素
```

**关键问题**：序列长度 N=4096 时，S 矩阵大小 = N² × 2 bytes (fp16) = **32 MB**。  
H100 HBM 带宽 ~3.35 TB/s，读写 32 MB 耗时 ~10µs，而每个 Transformer 层都需这个开销。  
**HBM 带宽，而非算力，成为瓶颈。**

---

### 9.2 Flash Attention 核心算法 — 分块在线 Softmax

**核心思想**：不实体化 S 矩阵，通过**在线 Softmax**（online softmax）在扫描 K/V tiles 的过程中逐渐累积输出 O。

#### 在线 Softmax 数学推导

设 score 向量 $s = [s_1, s_2, \ldots, s_N]$，标准 Softmax 需要两次全扫描：  
- 第一次：求 $m = \max_i s_i$  
- 第二次：计算 $p_i = \exp(s_i - m)$，求 $l = \sum_i p_i$

**在线版本**：将 score 分成若干分块 $B_1, B_2, \ldots$，每次到来一个新分块时"修正"之前的累积量：

$$
\text{当新分块 } B_t \text{ 到来，已知 } (m_{t-1},\ l_{t-1},\ O_{t-1}):
$$

$$
m_t = \max(m_{t-1},\ \max_{s \in B_t} s)
$$

$$
l_t = l_{t-1} \cdot e^{m_{t-1} - m_t} + \sum_{s \in B_t} e^{s - m_t}
$$

$$
O_t = O_{t-1} \cdot \frac{l_{t-1} \cdot e^{m_{t-1}-m_t}}{l_t}\ +\ \frac{\sum_{s \in B_t} e^{s - m_t} \cdot V_t}{l_t}
$$

扫描完所有分块后，$O_T$ 即为精确等价于完整 Softmax 的结果。

#### 分块访存分析

设 SRAM 大小为 $M$，tile size $B_r, B_c \approx \sqrt{M/d}$：

```
外层循环（Q tiles）:  N / Br 次 → 每次从 HBM 读 Q_tile (Br×d)
  内层循环（KV tiles）: N / Bc 次 → 每次从 HBM 读 K/V tile (2×Bc×d)

HBM 读量 = O(N²d / M) × (per tile size)
         ≈ O(N² / SRAM_tiles) 次 HBM 读取
HBM 写量 = O(Nd)  (只写 O，不写 S/P)
```

**对比朴素实现**：HBM 读写量从 $O(N^2)$ 降至 $O(N^2 \cdot d / M)$，M >> d 时可减少~8–16×。

---

### 9.3 Flash Attention 1/2/3/4 演进

#### FA1（Dao et al., 2022）

外层 Q-tiling，内层 KV-tiling，维护 per-token `(m, l)` 累积器：

```
for q_block in Q:              ← 每次加载 Q_block 到 SRAM
  m, l, O = -inf, 0, zeros
  for kv_block in KV:          ← 每次加载 K/V_block 到 SRAM
    S = Q_block @ K_block^T    ← SRAM 内矩阵乘，不出 SRAM
    update (m, l, O) via online softmax
  O_final = O / l              ← 写回到 HBM
```

**限制**：
- 使用 `wmma`（单 warp，16×16 tile），SM 利用率低
- 内层循环每步都需 rescale O，非矩阵乘运算占比高
- 前向与后向均为单 kernel，无 producer/consumer 分离

#### FA2（Dao, 2023）

三项核心改进：

1. **循环顺序优化**：外层改为 KV 循环、内层 Q 循环，使 Q tile 驻留寄存器，减少 SRAM 压力
2. **减少 rescaling 频率**：不再每步 rescale，仅在 KV 循环结束后执行一次最终 rescale
3. **更好的 warp 级并行**：每个 thread block 内多个 warp 处理不重叠的 KV 范围，warp 间无 SRAM 竞争

```
for q_block (each warp handles distinct q_block):
  m, l = -inf, 0
  O_accum = zeros(Br, d)
  for kv_block:
    S = Q_block @ K_block^T    ← 积累，暂不 rescale O
    m_new = max(m, max(S))
    l_new = l * exp(m - m_new) + sum(exp(S - m_new))
    O_accum = O_accum * (l exp(m-m_new)/l_new) + exp(S-m_new)/l_new * V_block
    m, l = m_new, l_new
  write O_accum to HBM         ← 仅一次 rescale
```

#### FA3（Shah et al., 2024）— **mini-sglang 在 H100 上使用的版本**

FA3 将算法改进与 **Hopper 架构专用硬件特性**深度耦合：

**关键 Hopper 特性**：

| 特性 | 说明 |
|---|---|
| **TMA** (Tensor Memory Accelerator) | H100 专有 DMA 单元，支持多维张量异步拷贝 HBM→SMEM，无需占用 CUDA 线程 |
| **wgmma** (Warpgroup MMA) | 128 线程（4 warp）联合执行的矩阵乘，tile 大小如 64×256×16，直接从 SMEM 送入 Tensor Core |
| **Named Barriers** | 细粒度 warp 级同步原语，允许 producer/consumer 独立推进 |
| **Async Copy** | `cp.async` 指令族，在 SMEM 加载时不阻塞计算 warp |

**FA3 Warp 专精化（Warp Specialization）**：

```
SM 内的 warp 分组：
  Producer warps (0-1)：专门负责 TMA 异步加载 K/V tiles
  Consumer warps (2-7)：专门负责 wgmma 计算 + 在线 Softmax
```

**Pingpong 双缓冲流水线**：

```
时间线（→ 为 Consumer，⇒ 为 Producer）：

T0: ⇒ TMA-load K_0/V_0 → SMEM[0]
T1: → wgmma(Q, K_0) [用 SMEM[0]]
    ⇒ TMA-load K_1/V_1 → SMEM[1]   ← 与 T1 计算并行！
T2: → softmax(S_0), rescale O
    → wgmma(Q, K_1) [用 SMEM[1]]
    ⇒ TMA-load K_2/V_2 → SMEM[0]   ← 复用 SMEM[0]（pingpong）
T3: → softmax(S_1), rescale O
    → wgmma(Q, K_2) [用 SMEM[0]]
    ⇒ TMA-load K_3/V_3 → SMEM[1]
...
```

**效果**：TMA 加载 K/V 的 HBM 延迟被完全隐藏，计算和内存访问全程重叠。

**FA3 vs FA2 性能对比（H100 实测，序列长4096，d=128）**：

| 指标 | FA2 | FA3 | 提升 |
|---|---|---|---|
| Prefill 吞吐 (TFLOPS) | ~250 | ~370 | ~1.5× |
| MFU (Model FLOP Utilization) | ~62% | ~92% | |
| 原因 | 频繁 HMMA 空泡 | TMA 掩盖 HBM 延迟，wgmma 更大 tile | |

#### FA4（2025）— **Blackwell B200（SM100）使用的版本**

FA4 针对 Blackwell 的新硬件特性进一步优化：

- **UMMA** (Unified MMA)：SM100 的下一代矩阵指令，支持更大的 tile（如 128×256×16），比 wgmma 宽 2×
- **增强的 TMA**：支持 bulk tensor reshape，减少 stride 计算开销
- sm_100 有 ~2× 更多 Tensor Core 和 HBM 带宽（B200: 8 TB/s vs H100: 3.35 TB/s）

mini-sglang 通过 `is_sm100_supported()` 检测 B200，选择 `ver=4` 传给 `flash_attn_with_kvcache`。

---

### 9.4 Paged KV Cache 下的 FA3/FA4 内核

当启用 Paged KV Cache 时，内层 KV tile 的加载不再是连续的指针偏移，而需要通过 `page_table` 间接寻址：

```
连续 KV（标准 FA）：
  inner_loop k = 0, 1, 2, ...:
    K_tile = k_ptr + k * Bc * head_dim    ← 线性地址

Paged KV（FA3 in sgl-kernel）：
  inner_loop k = 0, 1, ..., num_pages-1:
    page_id = page_table[req_idx, k]      ← 额外一次全局内存读（indirection）
    K_tile = k_cache[page_id, :, :, :]   ← 非连续 HBM 地址
```

**开销来源**：每次 tile 需要额外一次 HBM 读（读 `page_table` 中一个 int32）。对于大 `page_size`（如 16）这个开销很小，这也是 FA 后端借助 `page_size > 1` 摊薄 page_table 读取开销的原因之一。

**`page_size` 对内存局部性的影响**：

```
page_size=1  → 每个 token 独立页，page_table 跳转最频繁  
page_size=16 → 每 16 个 token 共享一页，内存碎片少，连续读长度更长
```

**`num_splits` 参数（Split-K 并行）**：

```
不分割（num_splits=1）:
  1 个 thread block 处理全部 KV tiles
  对长序列（N>>Br）串行性强，一个 SM 忙而其他 SM 空闲

分割（num_splits=4，以4为例）:
  4 个 thread block 各自处理 1/4 的 KV tiles
  每块输出一个 partial (m_i, l_i, O_i)
  
归约阶段（额外 kernel）:
  将 4 组 partial softmax 合并为最终 O：
  m   = max(m_0, m_1, m_2, m_3)
  l   = Σ l_i * exp(m_i - m)
  O   = Σ O_i * (l_i * exp(m_i - m)) / l
```

`num_splits=0` 时 sgl-kernel 自动选择最优分割策略（根据 seq_len / batch_size 权衡）。

**`pack_gqa` 参数（GQA 打包优化）**：

```
GQA ratio G = num_qo_heads / num_kv_heads

不打包（pack_gqa=False）：
  outer: G 次迭代（每个 Q head 单独算）
  inner: 每次重新从 HBM 加载 K/V tile
  K/V 加载次数 = G × num_KV_tiles

打包（pack_gqa=True）：
  outer: 1 次迭代（同时处理同一 KV head 对应的所有 G 个 Q heads）
  inner: K/V tile 只加载一次，G 个 Q heads 复用
  K/V 加载次数 = num_KV_tiles  → 节省 G 倍 HBM 读量
```

对 LLaMA-3 8B（G=4），`pack_gqa=True` 可减少 K/V HBM 读量 4×，显著提升 Prefill/Decode 带宽利用率。

---

### 9.5 FlashInfer FA2 Prefill Kernel

FlashInfer 的 `BatchPrefillWithPagedKVCacheWrapper` 实现 FA2 算法，并额外支持 paged KV 和变长 batch。

#### plan() 阶段做了什么

```python
wrapper.plan(
    qo_indptr=[0, 5, 8, 12],       # CPU, Q 序列分段
    paged_kv_indptr=[0, 5, 8, 12], # CPU, KV 页面分段
    paged_kv_indices=[页面ID列表],  # GPU, flat page ids
    paged_kv_last_page_len=[1,1,1], # CPU, 最后页填充量
    causal=True,
    ...
)
```

`plan()` 的核心工作是将 **Python 级 CSR 描述**转换为 **CUDA kernel 可直接消费的内部数据结构**，存入 `int_workspace_buffer`（~8MB）：

```
int_workspace_buffer 内容（伪结构）：
  kv_tile_indices:  [num_tiles_total]  int32  ← 每个 Q-block 需要访问的 KV tile 列表
  kv_indptr:        [num_q_blocks+1]   int32  ← 每个 Q-block 的 tile 范围指针
  q_tile_to_req:    [num_q_blocks]     int32  ← 每个 Q-block 属于哪个 request
```

这种预计算避免了 kernel 执行时的 CSR 解析开销。

#### run() 阶段的 kernel 执行流

```
launch_config:
  grid: (num_q_blocks, num_kv_heads, 1)   ← 每个 block 处理一个 Q tile × KV head
  block: (128, 1, 1)                       ← 4 warps

per_block:
  q_block_idx = blockIdx.x
  kv_head_idx = blockIdx.y
  req_idx = q_tile_to_req[q_block_idx]

  load Q_tile from global memory (kv_layout=NHD → coalesced)
  m = -inf, l = 0, O = zeros(Br, head_dim)

  kv_tile_start = kv_indptr[q_block_idx]
  kv_tile_end   = kv_indptr[q_block_idx + 1]

  for t in range(kv_tile_start, kv_tile_end):
    page_id = paged_kv_indices[kv_tile_indices[t]]
    K_tile = k_cache[page_id, 0, kv_head_idx, :]  # page_size=1
    V_tile = v_cache[page_id, 0, kv_head_idx, :]

    S = Q_tile @ K_tile^T                          # [Br, Bc]
    apply causal mask (if needed)
    update (m, l, O)

  O = O / l
  write to output[cu_seqlens_q[req_idx]...][q_head_range][:]
```

**Causal Masking 处理**：仅在 Q 和 KV 序列位置范围有交叠的 tile 才应用 mask。对纯历史 KV tile（KV 位置 < Q 位置开始），无需 mask，节省 mask 计算。

---

### 9.6 FlashInfer FA2 标准 Decode Kernel（非 Tensor Core）

Decode 阶段每个请求只有 **1 个 Q token**，需要对整个 KV Context 做 Attention。

**参数规模**：Q = [1, head_dim]，K/V = [seq_len, head_dim]  
→ $S = Q \cdot K^T$ 是纯 **GEMV**（矩阵×向量），算术强度极低（约 1 FLOP/byte）。

#### Thread Block 映射

```
grid: (batch_size * num_kv_heads * num_splits, 1, 1)
  → 每个 block 负责一个 (request, KV_head, split) 的三元组

block: (128, 1, 1)  ← 4 warps

per_block 伪代码：
  req_idx    = blockIdx.x / (num_kv_heads * num_splits)
  kv_head    = (blockIdx.x / num_splits) % num_kv_heads
  split_id   = blockIdx.x % num_splits

  page_start = kv_indptr[req_idx]
  page_end   = kv_indptr[req_idx + 1]
  my_pages   = pages[page_start + split_id * chunk : ... + (split_id+1) * chunk]

  m, l = -inf, 0
  O = zeros(head_dim)

  for page_id in my_pages:
    K = k_cache[page_id, 0, kv_head, :]   # [1, head_dim] (page_size=1)
    V = v_cache[page_id, 0, kv_head, :]

    s = dot(Q[q_head, :], K)              # 标量，单次 GEMV 行
    m_new = max(m, s)
    ...                                   # online softmax accumulate
    O += exp(s - m_new) / l_new * V

  if num_splits > 1:
    write partial (m, l, O) to float_workspace_buffer[split_id]
    # 后续归约 kernel 合并
  else:
    write final O to output
```

**注意**：Q head 与 KV head 的对应关系：每个 KV head 对应 `GQA = qo_heads/kv_heads` 个 Q heads，kernel 内部会循环处理这 GQA 个 Q heads（非 Tensor Core 路径下是串行的）。

#### float_workspace_buffer 的存储布局

```
float_workspace_buffer (128 MB) 布局：

[request_0][kv_head_0][split_0]: partial_m (1 float32) + partial_l (1 float32) + partial_O (head_dim floats)
[request_0][kv_head_0][split_1]: ...
...
[request_bs][kv_head_last][split_last]: ...

总大小 ∝ batch_size × num_kv_heads × num_splits × (2 + head_dim) × 4 bytes
```

归约 kernel 读取所有 `num_splits` 组的 (m, l, O)，使用与 online softmax 相同的合并公式，写出最终结果。

---

### 9.7 FlashInfer Tensor Core Decode Kernel（GQA ≥ 4）

#### 问题：GEMV 的 Tensor Core 利用率

```
标准 decode GEMV（per Q head）：
  s = Q_vec [1 × 128] @ K_mat [seq_len × 128]^T
  → warp 内每个线程处理 head_dim/warp_size = 4 个 float16，
  → 不满足 Tensor Core 最小 tile 约束（16×16×16），无法使用 wgmma
```

#### 解决方案：GQA 打包形成 GEMM

将同一 KV head 对应的 GQA 个 Q vectors 打包为一个矩阵：

```
GQA = 8 时（LLaMA-3 70B）：

Q_packed = stacked 8 Q vectors = [8, 128]    ← 满足 Tensor Core 最小 tile 约束

S = Q_packed @ K^T = [8, seq_len]            ← 真正的 GEMM！
                    ↑ 16 × alignment → 可用 wgmma
```

**Kernel Thread Block 映射变化**：

```
非 Tensor Core：
  grid → (batch × kv_heads × splits)，每 block 处理 1 个 (req, kv_head, split)
  内部串行循环 GQA 个 Q heads

Tensor Core：
  grid → (batch × kv_heads_groups × splits)，每 block 处理 1 个 (req, kv_head_group, split)
  使用 wgmma 同时处理 GQA 个 Q heads（并行 GEMM tile）
```

**寄存器压力变化**：

```
非 Tensor Core：
  每线程维护 1 组 (m, l, O)           → O 大小 = head_dim/warp_size = 4 floats
  寄存器占用：低

Tensor Core：
  每线程维护 GQA 组 (m, l, O)         → O 大小 = GQA × head_dim/warp_size = 32 floats
  寄存器占用：高 ← GQA 过大（≥16）时可能导致 register spill，性能下降
```

这是 `GQA >= 4` 而非 `GQA >= 2` 作为阈值的深层原因：GQA=4 的收益（4× GEMM tile → 4× Tensor Core 利用率）已足够摊平更高寄存器压力的开销。

#### 计算流程对比

```
非 Tensor Core（GQA=4 为例）：
  for q_head_offset in [0, 1, 2, 3]:
    q_vec = Q[q_head_offset, :]         # [128]
    for page_id in my_pages:
      k_vec = K[page_id, :]
      s = dot(q_vec, k_vec)             # 1 GEMV row，scalar
      ...

Tensor Core（GQA=4）：
  q_mat = Q[0:4, :]                     # [4, 128]，一次性加载4个Q head
  for page_id in my_pages:
    k_vec = K[page_id, :]               # [1, 128]
    s_vec = q_mat @ k_vec^T             # [4, 1]，1次 wgmma 16×16×16 tile (含padding)
    # 同时更新4个head的 (m, l, O)
    ...
```

---

### 9.8 TensorRT-LLM Attention Kernels（`trtllm_batch_*`）

TRT-LLM 的 Attention Kernel 是 NVIDIA 面向生产环境的 **FMHA（Fused Multi-Head Attention）**实现，由 FlashInfer Python 包代理暴露。

#### Context Kernel（Prefill）：`trtllm_batch_context_with_kv_cache`

本质上是针对 **encoder 风格长序列** 精调的 FA2 变体：

```python
trtllm_batch_context_with_kv_cache(
    query=q,                    # [total_tokens, num_heads, head_dim]
    kv_cache=(k_cache, v_cache),# paged kv cache
    block_tables=page_table,    # 2D page table（与 FA Backend 格式相同）
    seq_lens=cache_seqlens,     # 每请求 KV 长度
    bmm1_scale=softmax_scale,   # QK^T 的缩放因子（= 1/sqrt(d)）
    bmm2_scale=1.0,             # PV 的附加缩放（量化场景用，通常为1.0）
    kv_layout="NHD",
    ...
)
```

`bmm1_scale` 和 `bmm2_scale` 将 Attention 分为两次有独立 scale 的 BMM：

$$
O = \underbrace{(\text{softmax}(\underbrace{QK^T}_{\times \text{bmm1\_scale}}))}_{\text{attention mask}} \times V \times \text{bmm2\_scale}
$$

这在 **Int8/FP8 量化场景**下特别有用：K/V 可以用 Int8 存储，bmm2_scale 用于反量化。

**Block Table 格式（2D，与 FA 相同）**：

```
page_table: [batch_size, max_pages_per_req]  int32
  → row i 存储 request i 的所有物理页面 ID
  → 与 FlashInfer 的 flat indices 不同，TRT-LLM 使用二维直查
```

内部实现使用 NVIDIA CUTLASS 的 GEMM kernel primitive，应用了 LDGSTS（异步全局内存到共享内存加载）等 Ampere/Hopper 特性。

#### Decode Kernel：`trtllm_batch_decode_with_kv_cache`

专为 **decode 步骤**（Q seq_len = 1）优化：

```
FMHA Decode 核心特征：
  1. Multi-Split-K: 自动将 KV 序列分成多块并行计算
  2. HMMA-layout 优化：KV Cache 内部以"按 head_dim 优先"排列，decode 时连续访问
  3. Warp 级流水线：加载下一页 KV 同时完成当前页计算（类 FA3 思路，但用 HMMA 而非 wgmma）
```

**与 FlashInfer 标准 Decode 的对比**：

| 特性 | FlashInfer FA2 Decode | TRT-LLM FMHA Decode |
|---|---|---|
| 算法基础 | FA2（online softmax） | FA2（online softmax） |
| Tensor Core | wgmma（pack_gqa） | HMMA（单步 wmma） |
| Split-K | 手动 `num_splits` | 自动内部分析 |
| 量化支持 | 有限 | 完整（Int8/FP8，bmm_scale） |
| page_table 格式 | 1D flat indices | 2D block table |
| 调优接口 | `use_tensor_cores` / `num_splits` | 自动（黑盒） |

---

### 9.9 各 Kernel 算法汇总

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                        mini-sglang Attention Kernel 算法全景图                           │
├───────────────────┬──────────────┬──────────────────────────────┬────────────────────────┤
│  Backend          │  场景         │  核心算法                    │  关键硬件特性           │
├───────────────────┼──────────────┼──────────────────────────────┼────────────────────────┤
│  FA (sgl-kernel)  │  Prefill     │  FA3: wgmma + TMA Pingpong   │  H100 wgmma, TMA       │
│  ver=3            │              │  + Paged KV 二级寻址          │  Named Barriers        │
├───────────────────┼──────────────┼──────────────────────────────┼────────────────────────┤
│  FA (sgl-kernel)  │  Decode      │  FA3 + Split-K + pack_gqa    │  H100 wgmma            │
│  ver=3            │  (num_splits)│  自动 Split-K 归约            │                        │
├───────────────────┼──────────────┼──────────────────────────────┼────────────────────────┤
│  FA (sgl-kernel)  │  Prefill/    │  FA4: UMMA + 增强 TMA        │  B200 UMMA             │
│  ver=4            │  Decode      │  + 更大 TileSize              │  SM100 8TB/s HBM       │
├───────────────────┼──────────────┼──────────────────────────────┼────────────────────────┤
│  FlashInfer FA2   │  Prefill     │  FA2 + plan() 预计算 CSR     │  A100/H100 通用         │
│  (BatchPrefill)   │              │  + causal自适应masking        │                        │
├───────────────────┼──────────────┼──────────────────────────────┼────────────────────────┤
│  FlashInfer FA2   │  Decode      │  FA2 GEMV + Split-K          │  A100/H100 通用         │
│  (非 TensorCore)  │  GQA < 4     │  每 KV head 串行 GQA loop    │                        │
├───────────────────┼──────────────┼──────────────────────────────┼────────────────────────┤
│  FlashInfer FA2   │  Decode      │  GQA Pack → GEMM             │  H100 wgmma (16×16×k)  │
│  (TensorCore)     │  GQA ≥ 4     │  GQA 个 Q head → wgmma tile  │  寄存器占用高           │
├───────────────────┼──────────────┼──────────────────────────────┼────────────────────────┤
│  TRT-LLM FMHA     │  Prefill     │  FA2 + CUTLASS GEMM prim     │  LDGSTS, HMMA           │
│                   │              │  bmm_scale 支持量化           │  自动 Split-K           │
├───────────────────┼──────────────┼──────────────────────────────┼────────────────────────┤
│  TRT-LLM FMHA     │  Decode      │  FA2 GEMV + HMMA             │  Warp 流水线            │
│                   │              │  auto Split-K + 量化          │  2D block_table        │
└───────────────────┴──────────────┴──────────────────────────────┴────────────────────────┘
```
