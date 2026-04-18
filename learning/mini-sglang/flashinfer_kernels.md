# FlashInfer Kernel 深度解析

本文档对 mini-sglang 通过 FlashInfer 调用的底层 CUDA Kernel 进行系统性分析，涵盖分页 KV Cache 数据结构设计、Prefill FA2 Kernel、Decode GEMV Kernel、Tensor Core Decode Kernel 以及 plan/run 两阶段分派机制的逐行解读。

FlashInfer 源码仓库：[https://github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)  
mini-sglang 使用的版本：**FA2 backend**（`backend="fa2"` 参数），对应 `csrc/` 及 `include/flashinfer/attention/` 下的实现。

---

## 0. FlashInfer 总体架构

### 0.1 两阶段设计：plan + run

FlashInfer 将 Attention 计算分成两个完全独立的阶段：

```
Python 调用链：

wrapper.plan(indptr, indices, ...)      ← Phase 1：CPU 端元数据准备
  │
  ├─ 将 CSR 格式的 KV 索引转换为 Kernel 内部格式
  ├─ 决定 grid 大小 / work partition
  └─ 将结果写入 int_workspace_buffer（GPU）

wrapper.run(q, paged_kv_cache)          ← Phase 2：GPU 计算
  │
  ├─ 从 int_workspace_buffer 读取 plan 结果
  └─ 启动真正的 Attention CUDA Kernel
```

这种设计的核心好处：
- **plan 可以 non_blocking=True**（异步 CPU→GPU 传输），与 Python 端调度重叠
- **run 的 kernel launch 参数固定**，不需要在 GPU 上做 CSR 解析

### 0.2 FlashInfer 仓库目录结构（与 mini-sglang 相关部分）

```
flashinfer/
├── include/flashinfer/
│   ├── attention/
│   │   ├── prefill.cuh          ← FA2 Prefill 核心 CUDA 模板
│   │   ├── decode.cuh           ← FA2 Decode (GEMV) 核心 CUDA 模板
│   │   └── handler.cuh          ← Wrapper 类、plan() 实现
│   ├── page.cuh                 ← PagedKVCache 数据结构
│   ├── layout.cuh               ← QKVLayout (NHD / HND) 枚举
│   ├── cp_async.cuh             ← Async copy 原语 (cp.async)
│   ├── vec_dtypes.cuh           ← 向量化数据类型 (vec_t<8>)
│   ├── math.cuh                 ← PTX 数学原语 (shfl_xor, etc.)
│   └── state.cuh                ← softmax 状态 (m, d, o) 累积器
├── csrc/
│   ├── batch_prefill.cu         ← Prefill kernel 模板实例化入口
│   ├── batch_decode.cu          ← Decode kernel 模板实例化入口
│   └── single_decode.cu         ← Single decode 入口
└── python/flashinfer/
    ├── prefill.py               ← BatchPrefillWithPagedKVCacheWrapper
    └── decode.py                ← BatchDecodeWithPagedKVCacheWrapper
```

### 0.3 C++ 模板元编程策略

FlashInfer 大量使用 **C++ 模板实例化**来在编译期确定 kernel 参数，避免运行时 if-else 分支：

```cpp
// 来自 csrc/batch_decode.cu（简化）
// 每种参数组合在编译时生成独立 kernel，零运行时 overhead

template <uint32_t HEAD_DIM,           // 编译期常量：128 / 64 / 256
          QKVLayout KV_LAYOUT,         // NHD 或 HND
          bool USE_TENSOR_CORES,       // 决定使用 GEMV 还是 GEMM 路径
          typename DType,              // bf16 / fp16
          typename IdType>             // int32_t
cudaError_t BatchDecodeWithPagedKVCacheDispatched(
    DType* q, PagedKVCache<...> paged_kv, ...);
```

mini-sglang 中触发的具体实例通常是：
- `HEAD_DIM=128, KV_LAYOUT=NHD, DType=bfloat16, IdType=int32_t`

---

## 1. PagedKVCache 数据结构（`page.cuh`）

### 1.1 核心 struct 定义

```cpp
// include/flashinfer/page.cuh

template <PageStorage PAGE_STORAGE,  // POINTER（基础）或 INDICES（间接寻址）
          QKVLayout KV_LAYOUT,       // NHD 或 HND
          typename DType,
          typename IdType>           // 通常是 int32_t
struct PagedKVCache {
    // ── 数据指针 ──────────────────────────────────────────────────────
    DType* data;
    //   如果 KV_LAYOUT == NHD，data 的 logical shape:
    //     [max_num_pages, 2, page_size, num_heads_kv, head_dim]
    //     其中 axis=1 分别是 K（0）和 V（1）
    //   如果 KV_LAYOUT == HND:
    //     [max_num_pages, 2, num_heads_kv, page_size, head_dim]

    // ── CSR 格式的 KV 索引 ────────────────────────────────────────────
    IdType* indices;
    //   长度 = Σ ceil(seq_len_i / page_size) for i in batch
    //   每个元素是一个物理页面 ID，即 data 的第 0 维索引

    IdType* indptr;
    //   长度 = batch_size + 1，前缀和
    //   request i 的页面范围：indices[indptr[i] : indptr[i+1]]

    IdType* last_page_len;
    //   长度 = batch_size
    //   每个 request 最后一页已填充的 token 数（1..page_size）
    //   当 page_size=1 时，全为 1

    // ── 描述符 ────────────────────────────────────────────────────────
    uint32_t num_heads_kv;
    uint32_t head_dim;
    uint32_t page_size;    // mini-sglang 的 FlashInfer backend 中固定为 1
    uint32_t batch_size;
};
```

### 1.2 `page_size=1` 时的退化形式

当 `page_size=1`（mini-sglang 中 FlashInfer backend 的约束）时：

```
indices 就是每个请求的所有 KV token 的物理行号列表（flat 1D 数组）
last_page_len 全为 [1, 1, 1, ..., 1]

举例（3 个请求，seqlens=[5, 3, 4]）：
  indices = [page_ids_req0 × 5, page_ids_req1 × 3, page_ids_req2 × 4]
           ≈ [42,7,103,19,55,  200,11,88,  1,2,3,4]   (长度12)
  indptr  = [0, 5, 8, 12]
```

### 1.3 设备端寻址函数

```cpp
// page.cuh 中的核心内联函数

// 获取第 page_idx 页中 token_in_page 位置的 K 向量指针
__device__ __forceinline__ DType* PagedKVCache::GetKPtr(
    uint32_t page_idx,       // 物理页面 ID（来自 indices 数组）
    uint32_t token_in_page,  // 页内偏移（page_size=1 时恒为 0）
    uint32_t head_idx) {

    if constexpr (KV_LAYOUT == QKVLayout::kNHD) {
        // layout: [max_pages][2][page_size][num_kv_heads][head_dim]
        return data
            + page_idx * 2 * page_size * num_heads_kv * head_dim
            + 0       * page_size * num_heads_kv * head_dim  // K 在 axis=1 的 0
            + token_in_page * num_heads_kv * head_dim
            + head_idx * head_dim;
    } else {
        // layout: [max_pages][2][num_kv_heads][page_size][head_dim]
        return data
            + page_idx * 2 * num_heads_kv * page_size * head_dim
            + 0       * num_heads_kv * page_size * head_dim
            + head_idx * page_size * head_dim
            + token_in_page * head_dim;
    }
}

// V 指针同理，只需将 K 的 axis=1 的 0 换成 1
```

---

## 2. 向量化数据类型与内存访问（`vec_dtypes.cuh`）

### 2.1 `vec_t<float, N>` 向量原语

FlashInfer 的所有内存访问都通过向量化类型进行，确保每次 load/store 是 128-bit 对齐的宽指令：

```cpp
// include/flashinfer/vec_dtypes.cuh

// N = 向量宽度（float: N=4 → 128bit, half: N=8 → 128bit）
template <typename T, uint32_t N>
struct vec_t;

// bf16 的 8 元素特化（128 bit）
template <>
struct vec_t<__nv_bfloat16, 8> {
    uint4 data;  // 原始存储：4×32bit = 128bit

    __device__ __forceinline__ static void load(
        const __nv_bfloat16* ptr,
        vec_t<__nv_bfloat16, 8>& vec) {
        // PTX 指令：ld.global.v4.u32（128-bit 向量加载）
        *reinterpret_cast<uint4*>(&vec.data) =
            *reinterpret_cast<const uint4*>(ptr);
    }

    __device__ __forceinline__ static void store(
        __nv_bfloat16* ptr,
        const vec_t<__nv_bfloat16, 8>& vec) {
        *reinterpret_cast<uint4*>(ptr) =
            *reinterpret_cast<const uint4*>(&vec.data);
    }
};
```

对 `head_dim=128, bf16` 的单个 head：
- 128 个 bf16 = 256 bytes
- 每线程处理 `VEC_SIZE=8` 个 bf16 = 16 bytes（128-bit）
- `bdx = head_dim / VEC_SIZE = 128 / 8 = 16` 个线程处理一个 head 的全部元素

### 2.2 Warp 内 Reduction（`math.cuh`）

Online Softmax 需要在 warp 内做 max 和 sum reduction，FlashInfer 使用 PTX shuffle 指令：

```cpp
// include/flashinfer/math.cuh

// warp 内 max reduction（所有线程得到全局 max）
__device__ __forceinline__ float warp_reduce_max(float val) {
    // __shfl_xor_sync 是一条 PTX 指令：shfl.sync.bfly.b32
    // butterfly reduce 需要 log2(32)=5 步，每步 latency ≈2 cycles
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));  // 跨 16 lane
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  8));  // 跨 8  lane
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  4));  // 跨 4  lane
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  2));  // 跨 2  lane
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  1));  // 跨 1  lane
    return val;
}

// warp 内 sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}
```

---

## 3. Online Softmax 状态累积器（`state.cuh`）

FA2 的核心数学结构是 `(m, d, o)` 三元组，FlashInfer 将其封装为 `WarpBlockState`：

```cpp
// include/flashinfer/state.cuh

// T_state: float（累积精度）; T_out: bf16（输出精度）
// num_rows: head 内并行处理的 Q 行数（Prefill 中 = BLOCK_SIZE_Q）
template <uint32_t VEC_SIZE, uint32_t HEAD_DIM, typename T_state>
struct alignas(16) WarpBlockState {
    // ── Online Softmax 统计量 ─────────────────────────────────────────
    float o[HEAD_DIM / VEC_SIZE][VEC_SIZE];
    //   ↑ 输出累积器，shape 等价于 [head_dim]
    float m;    // running max：当前处理过的所有 KV 块的最大 score
    float d;    // running denominator：Σ exp(score_j - m)

    __device__ __forceinline__ void init() {
        m = -INFINITY;
        d = 0.f;
        // 将 o 清零
        #pragma unroll
        for (int i = 0; i < HEAD_DIM / VEC_SIZE; ++i)
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j)
                o[i][j] = 0.f;
    }

    // ── 核心 merge 操作 ───────────────────────────────────────────────
    // 将另一个 state (m_other, d_other, o_other) 合并进 this：
    // 数学等价于合并两个不相交的 KV 段的 partial attention 结果
    __device__ __forceinline__ void merge(
        float m_other, float d_other,
        const float o_other[HEAD_DIM / VEC_SIZE][VEC_SIZE]) {

        // 新 max
        float m_new = fmaxf(m, m_other);

        // 旧的 d 需要用 exp(m_old - m_new) 重新缩放
        float scale_self  = __expf(m - m_new);       // this  的缩放因子
        float scale_other = __expf(m_other - m_new);  // other 的缩放因子

        // 新 d = old_d * scale_self + other_d * scale_other
        float d_new = d * scale_self + d_other * scale_other;

        // 更新 o：重新缩放两个 partial output 后加权
        #pragma unroll
        for (int i = 0; i < HEAD_DIM / VEC_SIZE; ++i) {
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j) {
                o[i][j] = (o[i][j] * scale_self + o_other[i][j] * scale_other);
                // 注意：这里还没有除以 d_new，softmax 的归一化在最后统一做
            }
        }

        m = m_new;
        d = d_new;
    }
};
```

**关键细节**：`o` 在整个 KV 循环过程中存储的是 **unnormalized** 的加权和（分子），最终输出时一次性除以 `d`（分母）。

---

## 4. Decode Kernel — 标准 GEMV 路径（`decode.cuh`）

### 4.1 Kernel 签名与 Thread Block 映射

```cpp
// include/flashinfer/attention/decode.cuh

template <uint32_t BDX,        // bdx: head 内并行的线程数 = head_dim / VEC_SIZE
          uint32_t BDY,        // bdy: 每个 block 内并行处理的 Q heads 数
          PageStorage PAGE_STORAGE,
          QKVLayout KV_LAYOUT,
          uint32_t NUM_STAGES_SMEM,   // 异步 cp.async 的 pipeline stages
          uint32_t VEC_SIZE,
          bool DETERMINISTIC,
          typename DType,
          typename IdType>
__global__ void
__launch_bounds__(BDX * BDY)        // ← 限制寄存器用量，提升 occupancy
BatchDecodeWithPagedKVCacheKernel(
    DType* __restrict__ q,          // [total_q_tokens, num_qo_heads, head_dim]
    PagedKVCache<PAGE_STORAGE, KV_LAYOUT, DType, IdType> paged_kv,
    // plan 阶段生成的工作分区描述符
    IdType* __restrict__ kv_indptr, // 每个 (batch, kv_head) 的 page 范围
    IdType* __restrict__ kv_indices,// 物理 page 索引（可能是 plan 重排后的）
    uint32_t* __restrict__ kv_chunk_size_ptr,  // 每个工作单元的 KV chunk 大小
    DType* __restrict__ o,          // 输出 [total_q_tokens, num_qo_heads, head_dim]
    float* __restrict__ tmp,        // Split-K 中间结果缓冲区（float workspace）
    float* __restrict__ lse,        // log-sum-exp 输出（可选）
    float sm_scale)                 // = 1.0 / sqrt(head_dim)
{
    auto batch_idx  = blockIdx.x;   // 每个 block 负责一个 request
    auto kv_head    = blockIdx.y;   // 每个 block 负责一个 KV head
    auto qo_local   = threadIdx.y;  // 当前线程处理组内第几个 Q head（0..BDY-1）
    auto tx         = threadIdx.x;  // head 内的线程位置（0..BDX-1）

    // GQA 映射：qo_head = kv_head * GQA_ratio + qo_local
    const uint32_t qo_head = kv_head * (gridDim.y == 1 ? 1 : BDY) + qo_local;
    // → qo_head 是当前线程对应的 Q head 全局编号
```

**Grid 配置**：`grid = (batch_size, num_kv_heads, num_splits)`

```
blockIdx.x = batch（request）索引      ← 不同 request 完全并行
blockIdx.y = KV head 索引              ← 不同 KV head 完全并行
blockIdx.z = Split-K 切片索引（若>1）  ← 同一 KV 序列的不同片段并行
```

---

### Grid / Block 维度与矩阵乘法映射示意图

> 以下以 LLaMA-3 8B 典型参数为数值示例：
> `batch_size=2, num_qo_heads=32, num_kv_heads=8, head_dim=128`
> `GQA=4, VEC_SIZE=8, BDX=16, BDY=4, num_splits=1`

#### ① 从三张全局张量出发

Decode 阶段每个 request 只有 **1 个 Q token**，三张张量的完整 shape：

```
全局张量 Q   [batch_size=2,  num_qo_heads=32, head_dim=128]   ← 来自 HBM
全局张量 K   [total_kv_tokens, num_kv_heads=8, head_dim=128]  ← PagedKVCache，分散在 HBM
全局张量 V   [total_kv_tokens, num_kv_heads=8, head_dim=128]  ← PagedKVCache，分散在 HBM
输出张量 O   [batch_size=2,  num_qo_heads=32, head_dim=128]   ← 写回 HBM
```

计算目标（对每个 request `b`，每个 qo_head `h`）：

$$O[b,h,:] = \text{softmax}\!\left(\frac{Q[b,h,:]\cdot K[b,:,\text{kv\_head}(h),:]^T}{\sqrt{d}}\right) \cdot V[b,:,\text{kv\_head}(h),:]$$

其中 $\text{kv\_head}(h) = \lfloor h / \text{GQA} \rfloor$，是 GQA 的 head 折叠映射。

#### ② 张量维度 → CUDA Grid/Block 映射

```
 Q 张量                         CUDA 并行层次
 ┌──────────────────────────┐
 │  axis-0: batch_size = 2  ├──────────────►  blockIdx.x  ∈ [0, batch_size)
 │                          │                 每个 block 处理 1 个 request
 ├──────────────────────────┤
 │  axis-1: num_qo_heads=32 ├──────────────►  blockIdx.y  ∈ [0, num_kv_heads)
 │   = GQA × num_kv_heads   │                 × threadIdx.y ∈ [0, GQA=BDY)
 │   = 4   ×   8            │
 │   ├─ kv_head  (粗粒度) ───┼──────────────►  blockIdx.y   ← block 间并行
 │   └─ qo_local (细粒度) ───┼──────────────►  threadIdx.y  ← block 内并行
 ├──────────────────────────┤
 │  axis-2: head_dim = 128  ├──────────────►  threadIdx.x  ∈ [0, BDX=16)
 │   每线程覆盖 VEC_SIZE=8 维 │                 每线程负责 8 个连续维度
 └──────────────────────────┘

 K/V 张量（分页存储，page_size=1）
 ┌──────────────────────────┐
 │  axis-0: kv_tokens(可变) ├──────────────►  block 内串行循环（for page in pages）
 │                          │                 KV seq 维度不并行→ 靠 Split-K 可选并行
 ├──────────────────────────┤
 │  blockIdx.z（可选）        │  Split-K 将 KV 序列切为 num_splits 段分给不同 block
 │  ∈ [0, num_splits)       │  各段独立算部分 softmax → MergeStates 归约
 ├──────────────────────────┤
 │  axis-1: num_kv_heads=8  ├──────────────►  blockIdx.y  ← 与 Q 的 kv_head 对齐
 ├──────────────────────────┤
 │  axis-2: head_dim = 128  ├──────────────►  threadIdx.x  ← 与 Q 的 head_dim 对齐
 └──────────────────────────┘
```

#### ③ 单个 Block 的计算视图（blockIdx.x=0, blockIdx.y=1）

该 block 负责 `request=0，kv_head=1`，对应的 GQA Q heads `= [4, 5, 6, 7]`：

```
      Q  [batch=0, qo_heads 4..7, :]             K  [batch=0, kv_head=1, :, :]
      ┌──────────────────────────────┐            ┌──────────────────────────────────┐
 h=4  │ q4[0..7] q4[8..15] … q4[d] │            │ k_tok0[0..7] … k_tok0[head_dim]  │
 h=5  │ q5[0..7] q5[8..15] … q5[d] │     @      │ k_tok1[0..7] … k_tok1[head_dim]  │
 h=6  │ q6[0..7] q6[8..15] … q6[d] │            │         ...                      │
 h=7  │ q7[0..7] q7[8..15] … q7[d] │            │ k_tokN[0..7] … k_tokN[head_dim]  │
      └──────┬───────────────────────┘            └──────────────────────┬───────────┘
      [4×128]│                                                  [N×128]  │ 逐页加载到SMEM
             │ threadIdx.y       threadIdx.x                             │
             │ ─────────────     ─────────────                          │
             │ ty=0 → q4行       tx=0  →  d[0..7]                      │
             │ ty=1 → q5行       tx=1  →  d[8..15]                     │
             │ ty=2 → q6行       tx=2  →  d[16..23]   q 常驻寄存器      │
             │ ty=3 → q7行       ...                   K 每页经 cp.async  │
             │                   tx=15 →  d[120..127]  从 HBM 流入 SMEM  │
             │                                                           │
             └──────────────── 点积 & warp_reduce_sum ──────────────────┘
                                        │
                                        ▼
                          score[4×N]  (4 个 Q head × N 个 KV token）
                                        │  online softmax (逐页更新 m, d)
                                        ▼
      V  [batch=0, kv_head=1, :, :]                    O  [batch=0, qo_heads 4..7, :]
      ┌───────────────────────────────────┐             ┌───────────────────────────┐
      │ v_tok0[0..7] … v_tok0[head_dim]  │             │ o4[0..7] … o4[head_dim]  │
      │ v_tok1[0..7] … v_tok1[head_dim]  │    ──►      │ o5[0..7] … o5[head_dim]  │
      │         ...                      │             │ o6  …                    │
      │ v_tokN[0..7] … v_tokN[head_dim]  │             │ o7  …                    │
      └───────────────────────────────────┘             └───────────────────────────┘
                   [N×128]（同 K，逐页加载）                         [4×128]写回 HBM
```

**线程到元素的精确映射**（以 ty=0, tx=3 为例，即 Q head 4，head 维度 `[24..31]`）：

```
 寄存器中常驻：  q4[24], q4[25], …, q4[31]       （8 个 bf16，VEC_SIZE=8）
 每次循环Load：  k_tok_i[24], …, k_tok_i[31]     （SMEM 中读取，8 个 bf16）
 局部乘加：      partial = q4[24]*k[24] + … + q4[31]*k[31]   （8 次 fmadd）
 warp reduce：   score = Σ(partial, over tx=0..15)           （5 次 shfl_xor）
```

#### ④ head_dim 轴的向量化切分

```
  head_dim = 128  等分给 BDX=16 个线程，每线程负责 VEC_SIZE=8 个元素

  ◄──────────────────── head_dim = 128 个 bf16 ─────────────────────►
  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
  │tx= 0   │tx= 1   │tx= 2   │tx= 3   │tx= 4   │  ...   │tx=14   │tx=15   │
  │d[0..7] │d[8..15]│d[16..23│d[24..31│d[32..39│        │d[112..9│d[120.7]│
  └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
  每格 = 1 次 128-bit 向量加载（ld.global.v4.u32 指令）
  ↑
  Q 的每行（每个 Q head）恰好由一横行的 BDX=16 个线程完整覆盖
  K/V 的每行（每个 KV token）同样由相同 16 个线程覆盖（tx 对齐，同维度内积）

  所有 BDY=4 个 threadIdx.y 复用同一份 threadIdx.x 分配方案，
  各自独立持有自己 Q head 的 q_vec，互不干扰（寄存器私有）
```

---

### 4.2 共享内存 Pipeline：`cp.async` 多级缓冲

```cpp
// 在 BDX * BDY 个线程中，为 K/V pipeline 分配 SMEM
__shared__ DType kv_smem[NUM_STAGES_SMEM * 2 * BDY * head_dim];
//                        ↑ stages        ↑ K,V  ↑ BDY heads

// 使用 cp.async 异步加载（不占用 CUDA core，由 LSU 单元完成）
// 与 mma 计算重叠，隐藏 HBM→SMEM 延迟
if constexpr (NUM_STAGES_SMEM > 1) {
    // 预取 stage 0 的 K/V page 到 SMEM
    async_load_kv_page(kv_smem, 0, page_id_0, ...);
    __pipeline_commit();   // 提交 async 请求

    async_load_kv_page(kv_smem, 1, page_id_1, ...);
    __pipeline_commit();   // 再提交一个 stage

    for (uint32_t page = 0; page < num_pages; ++page) {
        // 等待当前 stage 的加载完成
        __pipeline_wait_prior(NUM_STAGES_SMEM - 1);

        // 从 SMEM 读 K/V 并计算 dot product（计算与下一轮加载重叠）
        compute_attention_step(state, kv_smem[stage % NUM_STAGES_SMEM], ...);

        // 预取下下轮
        if (page + NUM_STAGES_SMEM < num_pages) {
            async_load_kv_page(kv_smem, (stage+1) % NUM_STAGES_SMEM, ...);
            __pipeline_commit();
        }
    }
}
```

`NUM_STAGES_SMEM` 通常为 2（双缓冲）—— 计算 stage N 时，stage N+1 的 KV 数据已经在异步加载中。

### 4.3 Q·K^T 内积 — 逐 token 的 GEMV 行

```cpp
// 每个线程负责 VEC_SIZE 个 K 元素（向量化读取）
// tx 线程处理 head 中的 [tx*VEC_SIZE .. tx*VEC_SIZE + VEC_SIZE) 维度

// 从 Q 寄存器（整个 K 循环期间常驻寄存器）
vec_t<DType, VEC_SIZE> q_vec;
q_vec.load(q + qo_head * head_dim + tx * VEC_SIZE);

// 内层：遍历当前 page 的所有 KV token
for (uint32_t token_in_page = 0; token_in_page < last_page_len; ++token_in_page) {
    // 从 SMEM 读 K_token 对应 [tx*VEC_SIZE..+VEC_SIZE] 的部分
    vec_t<DType, VEC_SIZE> k_vec;
    k_vec.load(k_smem + token_in_page * BDY * head_dim
                      + qo_local * head_dim + tx * VEC_SIZE);

    // fp32 累加（避免 fp16/bf16 精度损失）
    float score = 0.f;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
        score += float(q_vec[i]) * float(k_vec[i]);

    // warp 内 reduce: 把 BDX 个线程各自算的 VEC_SIZE 个分量汇总
    score = warp_reduce_sum(score);  // shfl butterfly，最终所有线程都有全局sum
    score *= sm_scale;               // / sqrt(head_dim)

    // Optional: softcap (Gemma/Gemma2)
    // score = sm_scale_log2e * tanhf(score / softcap) * softcap;

    // 更新 online softmax 状态
    float m_old = state.m;
    state.m = fmaxf(state.m, score);
    float exp_m = __expf(m_old - state.m);   // rescale 因子
    state.d = state.d * exp_m + __expf(score - state.m);

    // 用 V token 更新输出累积器
    vec_t<DType, VEC_SIZE> v_vec;
    v_vec.load(v_smem + token_in_page * BDY * head_dim
                      + qo_local * head_dim + tx * VEC_SIZE);

    float p = __expf(score - state.m);
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
        state.o[tx][i] = state.o[tx][i] * exp_m + p * float(v_vec[i]);
}
```

**score 与 V 不是外积，而是逐 token 的 AXPY 累加**

从代数角度，完整 Attention 输出为：

$$O = \text{softmax}(Q K^T) \cdot V = \frac{\sum_{i} e^{s_i - m} \cdot v_i}{\sum_{i} e^{s_i - m}}$$

朴素实现需要先把所有 $s_i$ 存成向量 `score[seq_len]`，再乘 V：

```
score[seq_len] ← Q @ K^T       // 先物化全部 score
p[seq_len]     ← softmax(score) // softmax
O              ← p @ V          // 矩阵-向量乘
```

这需要 `seq_len` 大小的额外内存，且 K/V 需要读两遍（一遍算 score，一遍乘 V）。

FlashInfer 采用 **Online Softmax** 的做法，在 **同一个 token 循环里**即时消费 score，score 始终只是一个标量：

```
对 token i（循环体内，伪代码）：

  score_i = dot(q, k_i)                   // ① 标量，立即得到

  // ② 用 score_i 更新 running max m
  m_new  = max(m_old, score_i)
  exp_m  = exp(m_old - m_new)             //   旧状态需要 rescale 的因子

  // ③ 更新 running denominator d
  d      = d * exp_m + exp(score_i - m_new)

  // ④ AXPY：用 v_i 向量做一次 rank-1 更新
  p_i    = exp(score_i - m_new)           //   当前 token 的非归一化权重（标量）
  o_acc  = o_acc * exp_m + p_i * v_i      //   ← 这就是 state.o[tx][i] 那行
  //       ↑ 重新缩放旧结果      ↑ 加入新 token 的贡献
```

本质是每轮做一次 **scaled AXPY**（标量乘向量再加到向量上），而非外积：

```
                标量               向量
               ┌────┐          ┌──────────────────────────┐
  step i:  p_i │0.3 │  ×  v_i │v[0] v[1] … v[127]        │
               └────┘          └──────────────────────────┘
                                             │ 加到
                                             ▼
               o_acc            ┌──────────────────────────┐
          (寄存器，常驻)  ──────►│o[0] o[1] … o[127]  (更新) │
                                └──────────────────────────┘

  step i+1:  p_{i+1} × v_{i+1} 继续加到同一个 o_acc
  ...
  最后：  o_final = o_acc / d      （统一除以分母，一次性归一化）
```

**为什么不是外积**：外积的结果是一个矩阵（每对元素的乘积），这里 `p_i`（标量）× `v_i`（向量）的结果仍是向量，然后直接原地累加进 `o_acc`，O 的 shape 始终是 `[head_dim]`，不存在任何 `[seq_len, head_dim]` 的中间矩阵。

**为什么 score 向量从不被物化**：因为 online softmax 在看到下一个 token 之前无法确定当前 token 的最终归一化权重 $p_i / Z$（分母 $Z$ 到序列末尾才确定）。在线算法的妙处就是把"未来的归一化"推迟到最后统一做，中间只需要维护`(m, d, o_acc)` 三个状态量。

**关键性能要点**：
- `q_vec` 在整个 KV 循环期间**常驻寄存器**（不重新加载），这是 decode 性能的关键
- score 不需要任何额外存储（`seq_len` 大小的缓冲区），只占 1 个 float 寄存器
- `score = warp_reduce_sum(score)` 通过 5 次 `__shfl_xor_sync` 完成
- `state.o` 保存在寄存器数组中（`head_dim/VEC_SIZE * VEC_SIZE` = `head_dim` 个 float）
- K 和 V 各只被读取**一遍**（K 算完 score 后同一页 V 紧接着读，均在 SMEM 中）

### 4.4 最终 Softmax 归一化与写回

```cpp
// K 循环结束后统一归一化
float d_rcp = 1.f / state.d;
#pragma unroll
for (int v_it = 0; v_it < HEAD_DIM / VEC_SIZE; ++v_it) {
    vec_t<DType, VEC_SIZE> o_vec;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i)
        o_vec[i] = DType(state.o[v_it][i] * d_rcp);  // 转换到输出精度 bf16

    // 128-bit 向量写回 HBM
    o_vec.store(o + qo_head * head_dim + v_it * VEC_SIZE);
}

// 若需要 log-sum-exp（用于 split-k merge）
if (lse != nullptr)
    lse[batch_idx * num_qo_heads + qo_head] = state.m + __logf(state.d);
```

---

## 5. Decode Kernel — Tensor Core 路径

### 5.1 为什么 GEMV 无法用 Tensor Core

单个 Q token 的 decode 本质是：

$$s_i = Q_{[1 \times d]} \cdot K_{[d \times N]}^T$$

这是 **GEMV**（matrix-vector product），其 tile 约束为 $1 \times d \times d$，不满足 Tensor Core 的最小 tile（Hopper wgmma 最小 $16 \times 16 \times 16$）。

### 5.2 GQA Packing — 把 GEMV 变 GEMM

当 `GQA_ratio = num_qo_heads / num_kv_heads >= 4` 时，FlashInfer 将同一 KV head 对应的多个 Q heads 打包：

```
Q_packed = [q_0, q_1, q_2, q_3]  shape [4, head_dim] ← GQA=4

S = Q_packed @ K^T   # [4, seq_len]  → 真正的 GEMM！
```

#### 具体数值例子（head_dim=4, seq_len=3, GQA=4）

用小数字直观演示为什么打包后从 GEMV 变成了 GEMM。

**场景**：kv_head=0，对应 4 个 Q heads（q0, q1, q2, q3），KV 序列 3 个 token。

```
K 矩阵（3个KV token，每行是一个key向量，head_dim=4）：

        d0   d1   d2   d3
k_tok0 [ 1    0    1    0 ]
k_tok1 [ 0    1    0    1 ]    K shape: [3, 4]
k_tok2 [ 1    1    0    0 ]
```

**不打包（GQA=1 视角，3 次独立 GEMV）**：

每个 Q head 单独算，每次都是 [1×4] @ [4×3] = [1×3]：

```
q0 = [1, 0, 0, 1]   →  q0 @ K^T = [1·1+0·0+0·1+1·0, 1·0+0·1+0·0+1·1, 1·1+0·1+0·0+1·0]
                                  = [1, 1, 1]        ← score for q0

q1 = [0, 1, 1, 0]   →  q1 @ K^T = [0·1+1·0+1·1+0·0, 0·0+1·1+1·0+0·1, 0·1+1·1+1·0+0·0]
                                  = [1, 1, 1]        ← score for q1

q2 = [1, 1, 0, 0]   →  q2 @ K^T = [1·1+1·0+0·1+0·0, 1·0+1·1+0·0+0·1, 1·1+1·1+0·0+0·0]
                                  = [1, 1, 2]        ← score for q2

q3 = [0, 0, 1, 1]   →  q3 @ K^T = [0·1+0·0+1·1+1·0, 0·0+0·1+1·0+1·1, 0·1+0·1+1·0+1·0]
                                  = [1, 1, 0]        ← score for q3

4 次独立 GEMV，K 没有被复用，被读了 4 遍
```

**打包后（GQA Packing，1 次 GEMM）**：

把 4 个 Q head 堆叠成矩阵 Q_packed [4×4]，K^T [4×3]，做一次矩阵乘法：

```
Q_packed  @ K^T
─────────────────────────────────────────────────────────
 [1, 0, 0, 1]       [1, 0, 1]        [1, 1, 1]  ← q0 的结果
 [0, 1, 1, 0]  @    [0, 1, 1]   =    [1, 1, 1]  ← q1 的结果
 [1, 1, 0, 0]       [1, 0, 0]        [1, 1, 2]  ← q2 的结果
 [0, 0, 1, 1]       [0, 1, 0]        [1, 1, 0]  ← q3 的结果

Q_packed shape: [4, 4]   K^T shape: [4, 3]   S shape: [4, 3]
```

结果完全一致。K 只被读了 **1 次**，4 个 Q head 的计算复用了同一份 K。

**为什么这能用 Tensor Core**：

```
GEMV（单 Q head）：                GEMM（打包 4 个 Q head）：

Q [1 × 4]                          Q_packed [4 × 4]
K^T [4 × 3]                        K^T      [4 × 3]

M=1，不满足 wgmma 要求 M≥16         M=4（pad到16）→ M=16，满足！
→ 只能用 CUDA core 做 shfl reduce  → 可以调用 wgmma 指令，
                                     Tensor Core 全速运行
```

Tensor Core 硬件要求输入矩阵的行数（M）至少达到 16，才能让 128 个线程（4 warps）各自分到足够的工作量——M=1 时只有 1 行结果，128 个线程绝大多数都是冗余的：

```
M=1 时 Tensor Core 利用率：
  128 线程协作，但只产出 1×N 的结果
  → 127/128 = 99% 的线程在做重复工作或空转
  → 利用率约 1/16 ≈ 6%

M=16（GQA=16 或 pad 到 16）时：
  128 线程每人分到 16×N/128 个输出元素
  → Tensor Core 利用率 100%
```

**数学推导**：GQA 的 G 个 Q heads 共享同一个 KV head，G 个独立 GEMV 拼成一个 GEMM：

$$\underbrace{Q_{\text{packed}}}_{G \times d} \cdot \underbrace{K^T}_{d \times N} = \underbrace{S}_{G \times N}$$

其中 $G$ = `GQA_GROUP_SIZE`，$d$ = `head_dim`，$N$ = `seq_len_per_kv_head`（当前 KV block 大小 = `BLOCK_SIZE_KV`）。

wgmma 的最小 tile 约束是 $M \geq 16, N \geq 8, K = 16$（bf16）。当 $G < 16$ 时，FlashInfer 将 Q 矩阵沿 M 维**填充到 16 的倍数**（padding 行的权重为 0，不影响有效输出行）；当 $G \geq 16$ 时，tile 可以完整铺满。因此，**GQA ≥ 4 就值得切换到 Tensor Core 路径**——即使 $G=4$ 需要 pad 到 16，Tensor Core 的吞吐依然优于 GEMV 的 `shfl` 路径，因为 SMEM→寄存器的带宽被充分利用。

对应的 Kernel 签名不同：

```cpp
// include/flashinfer/attention/decode.cuh (Tensor Core 分支)

template <uint32_t NUM_WARPS,   // 每 block 的 warp 数（通常 4）
          uint32_t NUM_STAGES,
          uint32_t HEAD_DIM,
          uint32_t GQA_GROUP_SIZE,  // = num_qo_heads / num_kv_heads，编译期常量
          QKVLayout KV_LAYOUT,
          typename DType,
          typename IdType>
__global__ void
__launch_bounds__(NUM_WARPS * 32)
BatchDecodeWithPagedKVCacheKernelTensorCore(
    DType* __restrict__ q,
    PagedKVCache<...> paged_kv,
    ...
```

### 5.3 wgmma（Hopper Warp Group MMA）指令

```cpp
// 在 Tensor Core 路径中，使用 CUTLASS wgmma wrapper

// Q 矩阵从 SMEM 加载（wgmma 从 SMEM 读 A 操作数）
smem_copy_q(q_smem, q_ptr, GQA_GROUP_SIZE, head_dim);

// K/V 矩阵通过 TMA 或 cp.async 加载到 SMEM
async_load_kv_smem(k_smem, v_smem, page_id, ...);

// wgmma 指令（需要 CUTLASS Hopper API）：
// GQA_GROUP_SIZE >= 16 时可以用满一个完整 wgmma tile
cute::wgmma::wgmma_async<GQA_GROUP_SIZE, head_dim, BLOCK_SIZE_KV>(
    acc_s,     // [GQA_GROUP_SIZE, BLOCK_SIZE_KV] fp32 累加器（寄存器）
    q_smem,    // [GQA_GROUP_SIZE, head_dim]      bf16 (SMEM)
    k_smem     // [BLOCK_SIZE_KV,  head_dim]      bf16 (SMEM, transposed)
);
// 指令语义：acc_s += q_smem @ k_smem^T
// 硬件上：128 线程（4 warp）协作计算，每次消费 16×16×16 的 bf16 tile

cute::wgmma::commit_group();
cute::wgmma::wait_group<NUM_STAGES - 1>();  // 等待上一轮完成
```

**wgmma 如何分 tile 计算 $S = Q_{\text{packed}} \cdot K^T$**：

整个 GEMM 沿 K 维（`head_dim`）按 $k=16$ 步进，每步一条 wgmma 指令，循环 $d/16$ 次：

$$S \mathrel{+}= Q_{\text{packed}}[:, k{\cdot}16:(k{+}1){\cdot}16] \cdot K^T[k{\cdot}16:(k{+}1){\cdot}16, :]$$

```
K 维循环（共 head_dim/16 = 128/16 = 8 次）：

  iteration 0:  Q[:,  0:16] @ K^T[ 0:16, :]  → acc_s 部分和
  iteration 1:  Q[:,16:32] @ K^T[16:32, :]  → acc_s 累加
  iteration 2:  Q[:,32:48] @ K^T[32:48, :]  → acc_s 累加
  ...
  iteration 7:  Q[:,112:128] @ K^T[112:128, :] → acc_s 最终值

每次 wgmma.mma_async 指令：
  输入 A：SMEM 中的 [G × 16] bf16 tile（Q 的当前 16 列）
  输入 B：SMEM 中的 [BLOCK_SIZE_KV × 16] bf16 tile（K 的当前 16 列）
  输出 acc_s：寄存器中的 [G × BLOCK_SIZE_KV] fp32 累加器（不清零，持续累加）
```

- wgmma 是**异步**指令（投递即返回），`commit_group` 标记一批指令边界，`wait_group<S-1>` 确保最多 S-1 批在飞行中，实现计算-加载流水线重叠。

### 5.4 寄存器 fragmentation 与边界

wgmma 的输出累加器 `acc_s` 在寄存器中按照 MMA fragmentation 分布：

```
4 warps × 32 threads = 128 threads 共同持有 acc_s [GQA_GROUP_SIZE × BLOCK_SIZE_KV]

每 32×32 的 tile：
  warp 0 持有 [0:16, 0:16] 的部分 → rows 0..15, cols 0..15
  warp 1 持有 [0:16, 16:32] 的部分
  warp 2 持有 [16:32, 0:16] 的部分
  warp 3 持有 [16:32, 16:32] 的部分

每个 thread 持有 2 行 × 4 列（fp32）= 8 个 float 寄存器
```

**为什么每线程恰好持有 2 行 × 4 列（8 个 float）**：

以 wgmma tile $M=16, N=16$ 为例，128 个线程要分摊 $16 \times 16 = 256$ 个 fp32 元素：

$$\frac{256 \text{ elements}}{128 \text{ threads}} = 2 \text{ elements/thread}$$

但实际上 N 维通常是 16 的倍数，每个 16 列分组给每线程 2 个，BLOCK_SIZE_KV=32 则每线程持有 $2 \times 2 = 4$ 组，共 $2 \times 4 = 8$ 个 float。一般规律：

$$\text{regs per thread} = 2_{\text{rows}} \times \frac{N_{\text{tile}}}{16} \times 2_{\text{cols-per-group}} = \frac{N_{\text{tile}}}{4}$$

```
以 GQA=64（M=64），BLOCK_SIZE_KV=64（N=64）为例：

acc_s[64 × 64] = 4096 个 fp32
128 线程均摊 → 每线程 32 个 float

寄存器分布（warp level，M=64 = 4×16-row block）：
┌───────────────────────┬───────────────────────┬───────┬───────┐
│ warp 0                │ warp 1                │warp 2 │warp 3 │
│ rows  0..15           │ rows  0..15           │ 16..31│ 16..31│
│ cols  0..31           │ cols 32..63           │ 0..31 │32..63 │
│ (16×32/32th = 16 regs)│ (16 regs)             │(16 reg│(16 reg│
└───────────────────────┴───────────────────────┴───────┴───────┘
                                                  再乘4个16-row block → ×4
                                                  每 warp 64 regs
                                                  每 thread 64/32 = 2 regs... 
实际上 M=64 需要 4 个 16-row 分组，每 thread 寄存器 = 2 × 4 × (64/16/2) = 32
```

GQA=4 vs GQA=64 的寄存器对比：

| GQA | acc_s shape | 总 fp32 | 每线程 float | 额外 SMEM（Q tile） |
|-----|------------|---------|------------|--------------------|
| 1 (GEMV) | — | — | ~16（仅 o_acc） | 0 |
| 4   | 4×BKV      | 4×BKV   | BKV/8      | 4×d bf16 |
| 16  | 16×BKV     | 16×BKV  | BKV/2      | 16×d bf16 |
| 64  | 64×BKV     | 64×BKV  | 2×BKV      | 64×d bf16 |

这就是 **GQA 越大，寄存器和 SMEM 压力越高**的原因——Q_packed 矩阵尺寸正比于 GQA，acc_s 尺寸也正比于 GQA，两者都消耗线程私有资源，最终限制 SM occupancy。

---

## 6. Prefill Kernel（`prefill.cuh`）

### 6.1 Kernel 签名与 Grid 布局

```cpp
// include/flashinfer/attention/prefill.cuh

template <bool CAUSAL,            // 是否使用因果 mask
          SharedMemFillMode FILL_MODE,
          uint32_t NUM_WARPS,
          uint32_t NUM_STAGES,
          uint32_t VEC_SIZE,
          uint32_t NUM_MMA_D,     // head_dim / MMA_d_stride
          uint32_t NUM_MMA_Q,     // BLOCK_SIZE_Q / MMA_q_stride
          uint32_t NUM_MMA_KV,    // BLOCK_SIZE_KV / MMA_kv_stride
          PageStorage PAGE_STORAGE,
          QKVLayout KV_LAYOUT,
          PosEncodingMode POS_ENCODING_MODE,
          typename DType,
          typename IdType>
__global__ void
__launch_bounds__(NUM_WARPS * 32)
BatchPrefillWithPagedKVCacheKernel(
    DType* __restrict__ q,        // [total_q_tokens, num_qo_heads, head_dim]
    PagedKVCache<...>  paged_kv,
    // plan() 生成的索引结构（存在 int_workspace_buffer）
    IdType* __restrict__ q_indptr,          // Q 序列前缀和 [bs+1]
    IdType* __restrict__ kv_tile_indices,   // 每个 Q-block 需访问的 KV tile 列表
    IdType* __restrict__ kv_indptr_sorted,  // KV tile 列表的分段指针
    IdType* __restrict__ q_tile_to_req,     // Q-block 属于哪个 request
    DType* __restrict__ o,        // [total_q_tokens, num_qo_heads, head_dim]
    float* __restrict__ lse,      // log-sum-exp（可选）
    ...
```

**Grid 维度**：`dim3 grid(num_Q_blocks, num_qo_heads, 1)`

**Block 维度**：`dim3 block(NUM_WARPS * 32, 1, 1)`（即 128 线程 / 4 warps）

```
blockIdx.x = Q tile 的全局编号（跨 request 连续编号）
blockIdx.y = Q head（qo_head）编号，范围 [0, num_qo_heads)
blockIdx.z = 1（未使用）

threadIdx.x = 线程在 block 内的线性编号 [0, NUM_WARPS*32)
warp_id    = threadIdx.x / 32      // 0..3
lane_id    = threadIdx.x % 32      // 0..31
```

#### Grid 维度详细推导

Prefill 阶段是 **variable-length batching**，每个 request 的 Q 长度不同。Grid 的 `blockIdx.x` 需要跨越所有 request 的所有 Q tiles：

$$\text{num\_Q\_blocks} = \sum_{i=0}^{\text{batch\_size}-1} \left\lceil \frac{\text{seq\_len}_i}{\text{BLOCK\_SIZE\_Q}} \right\rceil$$

> 具体数值示例：`batch_size=3, seq_lens=[70, 40, 50], BLOCK_SIZE_Q=32, num_qo_heads=32`

```
Request 0:  seq_len=70  → ceil(70/32) = 3 个 Q tiles（tile 0, 1, 2）
Request 1:  seq_len=40  → ceil(40/32) = 2 个 Q tiles（tile 3, 4）
Request 2:  seq_len=50  → ceil(50/32) = 2 个 Q tiles（tile 5, 6）

num_Q_blocks = 3 + 2 + 2 = 7
```

```
Grid 全局视图（num_Q_blocks=7, num_qo_heads=32）：

                blockIdx.y（qo_head 编号）
                0        1        2       ...      31
blockIdx.x ┌─────────┬─────────┬─────────┬───┬─────────┐
  0 (R0,T0)│ block   │ block   │ block   │...│ block   │
            │(0,0)   │(0,1)   │(0,2)   │   │(0,31)  │
  1 (R0,T1)├─────────┼─────────┼─────────┼───┼─────────┤
            │(1,0)   │(1,1)   │ ...     │   │(1,31)  │
  2 (R0,T2)├─────────┼─────────┼─────────┼───┼─────────┤
            │(2,0)   │ ...     │         │   │         │
  3 (R1,T0)├─────────┼─────────┼─────────┼───┼─────────┤  ← request 1 起始
            │(3,0)   │(3,1)   │ ...     │   │(3,31)  │
  4 (R1,T1)├─────────┼─────────┼─────────┼───┼─────────┤
            │(4,0)   │ ...     │         │   │         │
  5 (R2,T0)├─────────┼─────────┼─────────┼───┼─────────┤  ← request 2 起始
            │(5,0)   │ ...     │         │   │         │
  6 (R2,T1)├─────────┼─────────┼─────────┼───┼─────────┤
            │(6,0)   │(6,1)   │ ...     │   │(6,31)  │
            └─────────┴─────────┴─────────┴───┴─────────┘

总 block 数 = 7 × 32 = 224 个 CUDA block
```

**每个 block 的工作**：
- 通过 `q_tile_to_req[blockIdx.x]` 查找自己属于哪个 request
- 计算自己负责的 Q 行范围：`[tile_start, tile_start + BLOCK_SIZE_Q)`
- 将这些 Q 行加载到 SMEM，然后遍历所有需要的 KV tiles（由 plan() 预计算）

#### Q tile 到 request 的映射（q_tile_to_req）

```
q_tile_to_req 数组内容（上例）：

  tile_idx:       0   1   2   3   4   5   6
  q_tile_to_req:  0   0   0   1   1   2   2
                  ╰── R0 ──╯  ╰ R1 ╯  ╰ R2 ╯

q_indptr（每个 request 的第一个 Q tile 编号）：
  [0, 3, 5, 7]

所以 blockIdx.x=4 → req_idx=1 → Q 行范围 = [32, 40)（补足到 BLOCK_SIZE_Q 时 pad 0）
```

#### 边界 tile 的 padding 处理

最后一个 Q tile 通常不满 BLOCK_SIZE_Q 行：

```
Request 0, tile 2（blockIdx.x=2）：

  逻辑 Q 行 = [64, 70)，只有 6 行有效，BLOCK_SIZE_Q=32 中剩余 26 行

  ┌──────────────────────┐
  │ Q row 64 ← 有效      │
  │ Q row 65 ← 有效      │
  │ Q row 66 ← 有效      │
  │ Q row 67 ← 有效      │
  │ Q row 68 ← 有效      │
  │ Q row 69 ← 有效      │
  │ (pad 0) row 70..95   │  ← 加载时写 0 到 SMEM，softmax 贡献为 0
  └──────────────────────┘

  Causal mask 进一步约束：对于 pad 行，Q·K^T score → -∞，exp(-∞)=0
  不影响有效行的 softmax 计算结果
```

#### Block 内部结构（128 线程 / 4 warps）

```
Block（128 线程）
┌────────────────────────────────────────────────────────────────┐
│  warp 0 (thread 0..31)   │  warp 1 (thread 32..63)           │
│  负责 QK^T 的             │  负责 QK^T 的                      │
│  S[0:16, 0:16]           │  S[0:16, 16:32]                   │
├──────────────────────────┼────────────────────────────────────┤
│  warp 2 (thread 64..95)  │  warp 3 (thread 96..127)          │
│  负责 QK^T 的             │  负责 QK^T 的                      │
│  S[16:32, 0:16]          │  S[16:32, 16:32]                  │
└──────────────────────────┴────────────────────────────────────┘

共享内存（SMEM）：
  q_smem    [BLOCK_SIZE_Q × head_dim]              = [32 × 128] bf16 = 8 KB   ← Q tile 常驻，全 KV 循环不变
  k_smem    [NUM_STAGES × BLOCK_SIZE_KV × head_dim] = [2 × 32 × 128] bf16 = 16 KB ← 双缓冲 stage 0/1
  v_smem    [NUM_STAGES × BLOCK_SIZE_KV × head_dim] = [2 × 32 × 128] bf16 = 16 KB ← 双缓冲 stage 0/1
  reduce_buf [NUM_WARPS × BLOCK_SIZE_Q] float        = [4 × 32] float = 512 B  ← cross-warp softmax 归约
  ──────────────────────────────────────────────────────────────────────────
  NUM_STAGES=2 含义：k/v_smem 各自分成 stage 0 和 stage 1 两个槽（ping-pong）
  计算 stage (j%2) 的 GEMM 时，cp.async 同步填充 stage ((j+1)%2) → HBM 延迟完全隐藏
  ──────────────────────────────────────────────────────────────────────────
  总计 ≈ 40.5 KB / block

寄存器（每 warp / 每 thread）：
  acc_s:  [16 × 16] fp32 fragmented → 每线程 8 个 float
  acc_o:  [16 × (head_dim/NUM_MMA_KV列)] fp32 → 每线程 head_dim 相关
  state:  m (1 float), d (1 float) × BLOCK_SIZE_Q/NUM_WARPS 行
```

### 6.2 plan() 阶段：KV tile 预计算

plan() 在 CPU 端（Python）预计算每个 Q-block 需要访问的 KV tile 集合：

```python
# python/flashinfer/prefill.py  （简化）

def plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices,
         num_qo_heads, num_kv_heads, head_dim, page_size, causal, ...):

    # 1. 构建 q_tile_to_req: 每个 Q-block → request 映射
    q_tile_to_req = compute_q_tile_to_req(qo_indptr, BLOCK_SIZE_Q)
    # 长度 = num_q_blocks = sum(ceil(seq_q_i / BLOCK_SIZE_Q))

    # 2. 对每个 Q-block，计算它需要访问的 KV tiles
    #    Causal 时，Q-block 只需要访问 KV 中 <= 当前 Q 位置的 tile
    kv_tile_list = []
    kv_indptr    = [0]
    for q_block_idx, req_idx in enumerate(q_tile_to_req):
        q_pos_start = q_block_idx * BLOCK_SIZE_Q
        q_pos_end   = min((q_block_idx+1) * BLOCK_SIZE_Q, qo_indptr[req_idx+1])

        # causal=True 时：KV 位置 <= q_pos_end (本 Q block 的最后一行)
        kv_max = kv_pos_end_of_req if not causal else q_pos_end
        kv_tiles_for_this_q = list(range(0, ceil(kv_max / BLOCK_SIZE_KV)))
        kv_tile_list.extend(kv_tiles_for_this_q)
        kv_indptr.append(len(kv_tile_list))

    # 3. 上传到 int_workspace_buffer（GPU）
    self._kv_tile_indices = torch.tensor(kv_tile_list).pin_memory().cuda(non_blocking=True)
    self._kv_indptr       = torch.tensor(kv_indptr ).pin_memory().cuda(non_blocking=True)
    self._q_tile_to_req   = torch.tensor(q_tile_to_req).pin_memory().cuda(non_blocking=True)
```

这些数组存入 `int_workspace_buffer`，run() 时 Kernel 直接读取，无需在 GPU 上做 CSR 解析。

### 6.3 Flash Attention 算法：Online Softmax + cp.async 流水线

#### NUM_STAGES 双缓冲原理

`NUM_STAGES=2` 使 K/V 的 HBM→SMEM 传输与 GEMM 计算完全重叠：

```
KV SMEM 物理布局（NUM_STAGES=2，两个 ping-pong slot）：

  k_smem[0]   stage 0 的 K tile  [BKV × head_dim]  ← 偶数轮写入
  k_smem[1]   stage 1 的 K tile  [BKV × head_dim]  ← 奇数轮写入
  v_smem[0]   stage 0 的 V tile  [BKV × head_dim]
  v_smem[1]   stage 1 的 V tile  [BKV × head_dim]

KV tile 循环时间线（HBM 延迟被 GEMM 完全掩盖）：

  iter:     j=0             j=1             j=2             j=3
  ──────────────────────────────────────────────────────────────────
  cp.async: [load tile 1]   [load tile 2]   [load tile 3]   [load t4]
            LSU单元 异步     LSU单元 异步     LSU单元 异步
  GEMM:     [compute tile 0][compute tile 1][compute tile 2][cpt t3]
            CUDA Core/TC    CUDA Core/TC
  ──────────────────────────────────────────────────────────────────
  stage使用:  stage 0         stage 1         stage 0         stage 1
              tile 0 算完后    tile 1 算完后
              stage 0 槽可复用 stage 1 槽可复用

  stage = kv_it % NUM_STAGES          ← 当前 iter 读哪个 SMEM 槽
  next_stage = (kv_it+1) % NUM_STAGES ← 同时预取哪个 SMEM 槽
```

`__pipeline_wait_prior(NUM_STAGES - 1)` = 等到「最多还有 1 个未完成的异步传输」，
即只等当前 iter 的数据就绪，允许下一个 tile 的预取继续在飞行中。

**没有双缓冲**：每次 KV tile 都必须等 HBM→SMEM 传输（~200 周期）完成才能开始 GEMM，
Tensor Core 和 CUDA Core 全部空转等待数据，带宽利用率接近 0%。

---

#### Flash Attention 算法 —— Prefill Tiled 版本

**Online Softmax 维护的三个状态**（每行 $i$ 独立，寄存器中，全 KV 循环不落 SMEM）：

$$m_i \in \mathbb{R}, \quad d_i \in \mathbb{R}, \quad O_{i,:} \in \mathbb{R}^{d}$$

初始化：$m_i = -\infty$，$d_i = 0$，$O_{i,:} = \mathbf{0}$

**对每个 KV tile $j$（共 $\lceil N_{KV}/B_{KV} \rceil$ 次）**：

$$\text{Step 1（GEMM）}\quad S_j = \frac{Q_{\text{tile}} \cdot K_j^T}{\sqrt{d}} \in \mathbb{R}^{B_Q \times B_{KV}}$$

$$\text{Step 2（行 max）}\quad \tilde{m}_j = \text{rowmax}(S_j) \in \mathbb{R}^{B_Q}$$

$$\text{Step 3（更新 max）}\quad m^{\text{new}} = \max\!\left(m,\; \tilde{m}_j\right)$$

$$\text{Step 4（更新分母）}\quad d^{\text{new}} = d \cdot e^{m - m^{\text{new}}} + \text{rowsum}\!\left(e^{S_j - m^{\text{new}}}\right)$$

$$\text{Step 5（更新输出）}\quad O^{\text{new}} = \underbrace{O \cdot e^{m - m^{\text{new}}}}_{\text{rescale 旧累加器}} + \underbrace{e^{S_j - m^{\text{new}}} \cdot V_j}_{\text{加入新 tile 的贡献}}$$

更新 $m \leftarrow m^{\text{new}}$，$d \leftarrow d^{\text{new}}$，$O \leftarrow O^{\text{new}}$

所有 KV tile 遍历完后归一化：$O_{\text{final}} = O / d$

**注意 rescale 步骤**：每次 KV tile 迭代后，旧的 $O$ 需要乘以 $e^{m_{\text{old}} - m_{\text{new}}}$。
当 $m$ 没有更新（当前 tile 的 max 不超过历史 max）时，$e^0 = 1$，rescale 是 no-op。
这就是 FA2 相比 FA1 的改进：FA1 需要两遍扫描，FA2 单遍在线完成。

**含 cp.async 双缓冲的完整 Kernel 主循环**（伪代码，对应 `prefill.cuh`）：

```cpp
// ─── 初始化 ────────────────────────────────────────────────────────────
WarpBlockState state;  // 寄存器：m[BQ], d[BQ], o_acc[BQ × head_dim]
state.init();          // m = -inf, d = 0, o_acc = 0

// Q tile 加载到 SMEM（整个 KV 循环仅此一次，不再更新）
load_q_smem(q_smem, q, q_tile_start, qo_head);  // 分配各 warp 搬运不同行
__syncthreads();

// ─── 流水线启动：预取第 0 个 KV tile ─────────────────────────────────
issue_cp_async_kv(k_smem[0], v_smem[0], kv_tile_indices[kv_start]);
__pipeline_commit();

// ─── 主循环 ──────────────────────────────────────────────────────────
for (uint32_t kv_it = kv_start; kv_it < kv_end; ++kv_it) {
    uint32_t stage = kv_it % NUM_STAGES;   // 当前计算使用哪个 SMEM 槽

    // Step ①：等待当前 stage KV 数据就绪（允许下一 tile 的预取在飞行中）
    __pipeline_wait_prior(NUM_STAGES - 1);
    __syncthreads();

    // Step ②：预取下一个 KV tile（双缓冲，计算与加载重叠）
    if (kv_it + 1 < kv_end) {
        uint32_t ns = (kv_it + 1) % NUM_STAGES;
        issue_cp_async_kv(k_smem[ns], v_smem[ns], kv_tile_indices[kv_it+1]);
        __pipeline_commit();
    }

    // Step ③：QK^T GEMM —— S[BQ×BKV] = Q_smem @ K_smem[stage]^T / sqrt(d)
    //    NUM_MMA_D=8 次 HMMA 16×16×16，每次消费 head_dim 的 16 列（SMEM→TC）
    float acc_s[…] = {};   // 寄存器（warp fragmentation 分布）
    for (int d_it = 0; d_it < NUM_MMA_D; ++d_it)
        wmma_mma(acc_s, q_smem + d_it*16, k_smem[stage] + d_it*16);

    // Step ④：跨 warp Softmax 归约（Prefill 独有，Decode 不需要）
    //    acc_s 的同一行 S[row, :] 散布在多个 warp 的寄存器中
    //    a. 每 warp 先做 warp 内 max（shfl_xor），得 partial_max
    //    b. 写入 smem_reduce_buf[warp_id][row]，__syncthreads()
    //    c. 每 warp 读全部 warp 的 partial_max，取全局 m_new
    //    d. 同理做 sum(exp(S - m_new)) 得 sum_exp
    cross_warp_softmax_reduce(smem_reduce_buf, acc_s, &m_new, &sum_exp);

    // Step ⑤：更新 Online Softmax 状态（寄存器内就地更新）
    float alpha = expf(state.m - m_new);   // rescale 系数，每行独立
    state.o_acc *= alpha;                   // 旧输出整体缩放
    state.m  = m_new;
    state.d  = state.d * alpha + sum_exp;  // 分母更新
    // acc_s 就地变为 P = exp(S - m_new)，供 PV GEMM 使用
    for each element: acc_s[i] = expf(acc_s[i] - m_new);

    // Step ⑥：PV GEMM —— o_acc[BQ×d] += P[BQ×BKV] @ V[BKV×d]
    //    NUM_MMA_KV=2 次 HMMA 指令，每次处理 BKV 的 16 行
    for (int kv_mma = 0; kv_mma < NUM_MMA_KV; ++kv_mma)
        wmma_mma(state.o_acc, acc_s + kv_mma*16, v_smem[stage] + kv_mma*16);
}

// ─── 归一化并写回 HBM ──────────────────────────────────────────────────
for each q_row:
    state.o_acc[q_row, :] /= state.d[q_row];   // 一次性除以分母
store_bf16(o + q_global_base, state.o_acc);     // 转 bf16 写回
if (lse) lse[q_global_idx] = state.m + logf(state.d);  // 供 split-K merge
```

**关键数据流总结**：

```
数据       存放位置       存活周期             作用
─────────────────────────────────────────────────────────────────────
Q tile     SMEM (8 KB)    整个 KV 循环         QK^T 的 A 操作数（常驻）
K tile_j   SMEM (8 KB)    1 个 iter（ping-pong 复用）  QK^T 的 B 操作数
V tile_j   SMEM (8 KB)    1 个 iter（ping-pong 复用）  PV 的 B 操作数
acc_s      寄存器          1 个 iter（每轮清零再累积）  中间 S 矩阵→P 矩阵
state.o    寄存器          整个 KV 循环（累加）         输出累加器（每轮 rescale）
state.m/d  寄存器          整个 KV 循环（递推）         online softmax 分子/分母
reduce_buf SMEM (512 B)   1 次跨 warp 归约用完即弃      cross-warp max/sum 中转
```

### 6.4 线程分工与矩阵计算对应关系

> 以下用具体参数说明：
> `BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=32, head_dim=128, NUM_WARPS=4`
> 等价于 `NUM_MMA_Q=2, NUM_MMA_KV=2, NUM_MMA_D=8`，block 共 128 线程

#### ① 一个 Block 负责哪些矩阵区域

```
全局张量视角（单 request，单 qo_head）：

 Q [total_q_tokens × head_dim]      K [total_kv_tokens × head_dim]
 ┌────────────────────────────┐      ┌────────────────────────────┐
 │                            │      │                            │
 │   ← 本 block 负责这一行段 → │      │   ← 每次循环取一 KV tile → │
 │   q_tile [BQ=32 × 128]     │      │   kv_tile [BKV=32 × 128]  │
 │   ████████████████████████ │      │   ████████████████████████ │
 │                            │      │                            │
 └────────────────────────────┘      └────────────────────────────┘
          ↓ 加载到 SMEM（一次）                ↓ 每轮循环 cp.async

          Q_smem [32 × 128]                  K_smem [32 × 128]
          V_smem [32 × 128]（同一 page）

 每次 KV tile 循环计算：

   S     =  Q_smem  @  K_smem^T          P  @  V_smem  →  O_acc（寄存器）
 [32×32]    [32×128]    [128×32]       [32×32]  [32×128]   [32×128]
```

#### ② Warp → 输出 tile 映射（两阶段 GEMM 分配不同）

**阶段一：QK^T，输出 S [32 × 32]**

S 按 MMA tile [16 × 16] 切成 [2 × 2] = 4 块，4 个 warp 各取一块：

```
         KV 位置 →     col 0..15      col 16..31
         Q 位置 ↓   ┌─────────────┬────────────┐
         row 0..15  │  warp 0     │  warp 1    │  ← 这两个 warp 共享 Q 行 0..15
                    ├─────────────┼────────────┤
         row 16..31 │  warp 2     │  warp 3    │  ← 这两个 warp 共享 Q 行 16..31
                    └─────────────┴────────────┘
                      ↑                ↑
               计算 Q[0:16,:]@K[0:16,:]^T  Q[0:16,:]@K[16:32,:]^T
               (两者都用同一段 Q 行，K 列不同)

  每个 warp 内：沿 K 维（head_dim=128）迭代 NUM_MMA_D=8 次 HMMA：
    loop d in [0,16,32,...,112]:
      warp 0: acc_s[0:16, 0:16] += Q_smem[0:16, d:d+16] @ K_smem[0:16, d:d+16]^T
```

**阶段二：PV，输出 O [32 × 128]**

O 按 [16 × 16] 切成 [2 × 8] = 16 块，4 个 warp 均分为每 warp 4 块：

```
       head_dim 方向 →  d=0..15  d=16..31  d=32..47  ...  d=112..127
       Q 位置 ↓       ┌────────┬─────────┬─────────┬──────────────┐
       row 0..15      │ warp 0 │ warp 0  │ warp 0  │   warp 0     │← 4块
                      ├────────┼─────────┼─────────┼──────────────┤
       row 16..31     │ warp 2 │ warp 2  │ warp 2  │   warp 2     │← 4块
                      └────────┴─────────┴─────────┴──────────────┘
       （warp 1/3 取另外 4 个 d 列段，warp 0 和 warp 1 分别负责不重叠的 d 列）

  注意：PV 阶段 warp 分配方式与 QK^T 不同——
  QK^T 中 warp 按 KV 列切分（同一 Q 行跨多 warp），
  PV 中 warp 按 head_dim 列切分（O 的列方向）
```

#### ③ 线程在 warp 内的 HMMA Fragment 分布

一个 warp 负责 [16 × 16] 的累加器 tile，32 个线程的 fragmentation：

```
HMMA 16×16×16（Ampere/Ada HMMA_16816）的寄存器布局：

  ← 16 列（KV 维度，score 值） →
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
0 │t0│t1│t2│t3│t0│t1│t2│t3│t8│t9│tA│tB│t8│t9│tA│tB│  每 4 线程一组覆盖 2 列
1 │t0│t1│t2│t3│t0│t1│t2│t3│t8│t9│tA│tB│t8│t9│tA│tB│
  ├──┤                                              ├──┤
8 │t4│t5│t6│t7│t4│t5│t6│t7│tC│tD│tE│tF│tC│tD│tE│tF│
9 │t4│t5│t6│t7│  ...                               │
  ├──┤                                              ├──┤
  │t16..t19 ...                                    │
  │t20..t23 ...                                    │
  │t24..t27 ...                                    │
  │t28..t31 ...                                    │
  └──────────────────────────────────────────────────┘
  ↑ 16 行（Q 维度，query 位置）

  每线程持有 4 个 fp32：对应 2 行 × 2 列的输出元素
  （纵轴每 8 行一组，横轴每 8 列一组，线程 id 交织排列）
```

精确规律：thread `t` 在 warp 内持有 acc_s 的元素位置：

```
row_group = t / 4           // 0..7，决定属于哪 8 行组
col_pair  = t % 4           // 0..3，决定持有哪对列

持有的 4 个元素（fp32）：
  (row_group * 2 + 0,  col_pair * 2 + 0)
  (row_group * 2 + 0,  col_pair * 2 + 8)   ← +8 源于 HMMA 的 zigzag 布局
  (row_group * 2 + 1,  col_pair * 2 + 0)
  (row_group * 2 + 1,  col_pair * 2 + 8)
```

#### ④ 跨 warp 的 Softmax 归约

QK^T 后，Q 行 `q=5`（在 row 0..15 中）的完整 score 向量 `S[5, 0:31]` **分散在 warp 0 和 warp 1** 的寄存器中：

```
warp 0 寄存器：S[5, 0],  S[5, 1],  ..., S[5, 15]    （warp 0 负责 KV col 0..15）
warp 1 寄存器：S[5, 16], S[5, 17], ..., S[5, 31]    （warp 1 负责 KV col 16..31）

求 softmax(S[5, :]) 需要 max(S[5, 0:31])
→ 需要跨 warp 通信！
```

FlashInfer 通过 SMEM 完成跨 warp 归约（简化）：

```cpp
// 每个 warp 先做 warp 内 reduce（shfl_xor，得到本 warp 这 16 列的 max）
float partial_max = warp_reduce_max_over_kv_cols(acc_s, q_row_in_tile);

// 将 partial_max 写入 SMEM 的归约区域
smem_reduce_buf[warp_id * BQ + q_row] = partial_max;
__syncthreads();

// 由负责该 Q 行的 warp 读取全部 NUM_MMA_KV 个 partial，取最终 max
float m_new = -INFINITY;
for (int w = 0; w < NUM_WARPS_KV; ++w)
    m_new = fmaxf(m_new, smem_reduce_buf[w * BQ + q_row]);

// 用 m_new 更新 online softmax 状态，并 rescale 自己的 acc_s 列
```

这是 Prefill kernel 比 Decode kernel **复杂得多**的原因：每处理一个 KV tile，S 矩阵的每行都需要一次跨 warp 的 max+sum 归约，然后才能计算 exp 并与 V 矩阵相乘。

#### ⑤ 完整的单次 KV tile 处理流程（线程视角）

```
所有 128 线程并行执行：

Step 1  Load KV tile（cp.async，不阻塞）
        → L2→SMEM 异步搬运 k_smem[BKV×d], v_smem[BKV×d]

Step 2  __pipeline_wait_prior()
        → 确认 SMEM 数据就绪

Step 3  QK^T GEMM（각 warp 处理自己的 [16×16] 输出 tile）
        loop d = 0, 16, ..., 112:           // NUM_MMA_D = 8 步
          wmma::load q_frag  from q_smem    // 从 SMEM 读当前 16 列的 Q 片段
          wmma::load k_frag  from k_smem    // 从 SMEM 读当前 16 列的 K 片段
          wmma::mma_sync(acc_s, q_frag, k_frag)  // 累加器更新

Step 4  跨 warp softmax 归约（__syncthreads() + SMEM）
        → 每行 Q 的 m_new = max(S[row, :]) 全局可见
        → 每线程 rescale acc_s，计算 exp(acc_s - m_new)
        → 更新 online state: m, d

Step 5  PV GEMM（各 warp 处理自己的 [16×16] O 输出 tile）
        loop kv = 0, 16:                    // NUM_MMA_KV = 2 步
          wmma::load p_frag  from 寄存器 acc_s（此时 acc_s = exp(S-m)）
          wmma::load v_frag  from v_smem
          wmma::mma_sync(acc_o, p_frag, v_frag)
        → 注意：acc_o 使用不同的 warp-to-tile 分配（按 head_dim 列切分）

Step 6  下一轮 KV tile 循环，acc_o 持续累加（online softmax rescale）

KV 循环结束后：
  → acc_o /= d（统一归一化）
  → 转换为 bf16 写回 HBM（O[q_tile_start:q_tile_start+BQ, qo_head, :]）
```

#### ⑥ 矩阵与计算资源的完整对应

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Prefill Kernel 计算资源分配总览                     │
├────────────────┬─────────────────────────────────────────────────────┤
│ 粒度           │  负责的计算                                          │
├────────────────┼─────────────────────────────────────────────────────┤
│ Grid block     │  1 个 Q tile [BQ × d] × 整个 KV 序列                │
│ (blockIdx.x/y) │  = 固定 Q 的 BQ 行，串行遍历所有 KV tile             │
├────────────────┼─────────────────────────────────────────────────────┤
│ Warp           │  QK^T 阶段：S 的 [16 × 16] 子块（按 Q行/KV列 切分） │
│ (warp_id 0..3) │  PV 阶段：O 的 [16 × 16] 子块（按 Q行/head_dim 切） │
│                │  online softmax 状态 (m, d, o) 按 Q 行由 warp 维护  │
├────────────────┼─────────────────────────────────────────────────────┤
│ Thread         │  warp 内 HMMA fragment 的 4 个 fp32 元素             │
│ (lane_id 0..31)│  精确位置由 (lane_id/4, lane_id%4) 的 zigzag 决定   │
│                │  Q 加载：tx 负责 head_dim 的 VEC_SIZE 个连续维度     │
└────────────────┴─────────────────────────────────────────────────────┘
```

### 6.5 外层循环：Q tile 在 SMEM 中驻留

```cpp
// 每个 block 负责一个 Q tile：[BLOCK_SIZE_Q, head_dim]
// block 内 NUM_WARPS 个 warp 共同处理

// Step 1：将 Q tile 加载到 SMEM（整个 KV 循环期间 Q 不再重新加载）
__shared__ DType q_smem[BLOCK_SIZE_Q * head_dim];

// 分配给每个 warp 加载 Q tile 的一部分行
load_q_to_smem(q_smem, q, q_tile_start, blockIdx.y /*qo_head*/, ...);
__syncthreads();

// 初始化 WarpBlockState（寄存器，每个 warp 一份）
WarpBlockState<VEC_SIZE, head_dim, float> state;
state.init();

// Step 2：遍历该 Q tile 需要访问的所有 KV tiles（来自 plan 结果）
uint32_t kv_start  = kv_indptr_sorted[blockIdx.x];
uint32_t kv_end    = kv_indptr_sorted[blockIdx.x + 1];

for (uint32_t kv_it = kv_start; kv_it < kv_end; ++kv_it) {
    uint32_t kv_tile_idx = kv_tile_indices[kv_it];

    // 将 KV tile 通过 cp.async 异步加载到 SMEM
    // page_size=1 时：每个 token 独立 page，需要 paged_kv.indices 间接取地址
    async_load_paged_kv_tile(k_smem, v_smem, paged_kv,
                             kv_tile_idx, BLOCK_SIZE_KV, ...);
    __pipeline_commit();
    __pipeline_wait_prior(NUM_STAGES - 1);

    // 计算当前 KV tile 的 attention
    compute_qk_and_update_state(state, q_smem, k_smem, v_smem, ...);
}
```

### 6.6 因果 Mask 的高效处理

```cpp
// compute_qk_and_update_state 内部（简化）

// 对于 CAUSAL=true，需要 mask：
// Q token 位置 q_pos 只能看到 KV token 位置 kv_pos <= q_pos

// 关键优化：只在 Q 和 KV 位置有交叠的 tile 才应用 mask
// ┌──────────────────────────────────────────────────────┐
// │ 三类 tile：                                          │
// │  (A) KV tile 完全在 Q 之前  → 无需 mask，全部有效    │
// │  (B) KV tile 与 Q 重叠     → 需要逐元素检查 mask    │
// │  (C) KV tile 完全在 Q 之后  → plan 阶段已排除        │
// └──────────────────────────────────────────────────────┘
bool is_diagonal_tile = (kv_tile_idx * BLOCK_SIZE_KV > q_tile_start);
// is_diagonal_tile 为 true → 该 tile 在对角线附近，需要 mask

if constexpr (CAUSAL) {
    if (is_diagonal_tile) {
        // 为每个 (Q row, KV col) 对计算 causal mask
        // Q row 绝对位置 = q_tile_start + q_row_in_tile
        // KV col 绝对位置 = kv_tile_idx * BLOCK_SIZE_KV + kv_col_in_tile
        bool causal_ok = (q_abs_pos >= kv_abs_pos);
        score = causal_ok ? score : -INFINITY;
    }
    // 非对角线 tile：所有 KV 位置都满足 causal，不加 mask
}
```

### 6.7 fa2 的 wgmma vs 标准 HMMA

在 FlashInfer 的 FA2 backend（mini-sglang 使用的）中：

```cpp
// FA2 使用 MMA_16x8x16 tile（WMMA API 的 Ampere/Hopper 通用版本）
// 不同于 FA3 的 wgmma（只在 FA3/Hopper 专用 backend 中使用）

// QK^T 计算：使用 CUTLASS mma::Mma<...> 或直接 PTX wmma 指令
// tile shape: [16, 8] × [8, 16] = [16, 16]（最小 MMA tile）

wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
wmma::fill_fragment(acc, 0.0f);

wmma::load_matrix_sync(q_frag, q_smem_ptr, head_dim);  // 从 SMEM 加载
wmma::load_matrix_sync(k_frag, k_smem_ptr, head_dim);  // 从 SMEM 加载
wmma::mma_sync(acc, q_frag, k_frag, acc);              // 执行 MMA
// 结果在寄存器中，shape [16, 16]（以 warp fragmentation 分布）
```

---

## 7. HMMA Fragment 布局深度解析

### 7.1 为什么需要理解 Fragment 布局

Prefill kernel 的 QK^T 和 PV 两阶段 GEMM 都用 HMMA 指令完成，softmax 归约发生在 HMMA 指令之间的"间隙"中。如果不清楚每个线程持有哪些元素，就无法理解：

- cross-warp softmax 归约中 S 矩阵的行如何"散落"在多个 warp
- PV GEMM 中 P 矩阵（即 exp(S - m)）是如何被各线程就地修改的
- 寄存器压力的来源（每线程持有多少个 fp32 累加器）

---

### 7.2 HMMA 指令快速回顾

FlashInfer FA2 backend 使用的是 **`m16n8k16`**（Ampere/Ada 架构）或 **`m16n16k16`** 的 WMMA 指令，这里以 Prefill kernel 实际使用的 **`m16n16k16` bf16** 为基准讲解。

```
HMMA m16n16k16 指令（单条）：

  A 操作数：[16 × 16] bf16，每 warp 的 32 线程共同持有（来自 SMEM Q tile）
  B 操作数：[16 × 16] bf16，每 warp 的 32 线程共同持有（来自 SMEM K tile）
  C/D 操作数：[16 × 16] fp32 累加器，每 warp 的 32 线程共同持有（寄存器）

  执行：  D = A × B + C    （fp32 累加，bf16 运算）
  латентность：~16 cycles（Ampere），之后结果在线程寄存器里
```

对应 CUDA 代码（WMMA API）：

```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::load_matrix_sync(a_frag, smem_a_ptr, ldA);  // 32 线程协作从 SMEM 加载
wmma::load_matrix_sync(b_frag, smem_b_ptr, ldB);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);   // 计算 c += a × b
// 执行后，c_frag 以 fragment layout 分布在 32 线程的寄存器中
```

---

### 7.3 累加器 Fragment 布局：每个线程持有哪些元素

HMMA `m16n16k16` 的累加器是 **[16 × 16] fp32** 矩阵，32 个线程共同持有，每线程持有 8 个 fp32 元素。

#### 完整位置表

```
矩阵坐标 (row, col) → 持有线程（lane_id）+ 寄存器槽

  col:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
row 0: [ 0][ 0][ 1][ 1][ 2][ 2][ 3][ 3][ 0][ 0][ 1][ 1][ 2][ 2][ 3][ 3]
row 1: [ 0][ 0][ 1][ 1][ 2][ 2][ 3][ 3][ 0][ 0][ 1][ 1][ 2][ 2][ 3][ 3]
row 2: [ 4][ 4][ 5][ 5][ 6][ 6][ 7][ 7][ 4][ 4][ 5][ 5][ 6][ 6][ 7][ 7]
row 3: [ 4][ 4][ 5][ 5][ 6][ 6][ 7][ 7][ 4][ 4][ 5][ 5][ 6][ 6][ 7][ 7]
row 4: [ 8][ 8][ 9][ 9][10][10][11][11][ 8][ 8][ 9][ 9][10][10][11][11]
row 5: [ 8][ 8][ 9][ 9][10][10][11][11][ 8][ 8][ 9][ 9][10][10][11][11]
row 6: [12][12][13][13][14][14][15][15][12][12][13][13][14][14][15][15]
row 7: [12][12][13][13][14][14][15][15][12][12][13][13][14][14][15][15]
row 8: [16][16][17][17][18][18][19][19][16][16][17][17][18][18][19][19]
row 9: [16][16][17][17][18][18][19][19][16][16][17][17][18][18][19][19]
row10: [20][20][21][21][22][22][23][23][20][20][21][21][22][22][23][23]
row11: [20][20][21][21][22][22][23][23][20][20][21][21][22][22][23][23]
row12: [24][24][25][25][26][26][27][27][24][24][25][25][26][26][27][27]
row13: [24][24][25][25][26][26][27][27][24][24][25][25][26][26][27][27]
row14: [28][28][29][29][30][30][31][31][28][28][29][29][30][30][31][31]
row15: [28][28][29][29][30][30][31][31][28][28][29][29][30][30][31][31]

每格中的数字 = lane_id（0..31）
```

#### 规律总结

```
对于 lane_id = t（0..31）：

  row_base = (t / 4) * 2          // 步长 2，范围 0,2,4,6,...,14
  col_base_left  = (t % 4) * 2    // 步长 2，范围 0,2,4,6
  col_base_right = col_base_left + 8

  线程 t 持有的 8 个元素的矩阵坐标：

  寄存器槽 0: (row_base + 0,  col_base_left  + 0)   →  c_frag.x[0]
  寄存器槽 1: (row_base + 0,  col_base_left  + 1)   →  c_frag.x[1]
  寄存器槽 2: (row_base + 1,  col_base_left  + 0)   →  c_frag.x[2]
  寄存器槽 3: (row_base + 1,  col_base_left  + 1)   →  c_frag.x[3]
  寄存器槽 4: (row_base + 0,  col_base_right + 0)   →  c_frag.x[4]
  寄存器槽 5: (row_base + 0,  col_base_right + 1)   →  c_frag.x[5]
  寄存器槽 6: (row_base + 1,  col_base_right + 0)   →  c_frag.x[6]
  寄存器槽 7: (row_base + 1,  col_base_right + 1)   →  c_frag.x[7]
```

**直觉记忆**：每个线程负责一个 **"2行 × 8列（分左右两半 × 4列)"** 的 L 形区域，在整个 16×16 tile 里形成 zigzag 交织分布。

---

### 7.4 具体示例：线程 t=5 持有哪些元素

```
t = 5：
  row_base      = (5 / 4) * 2 = 1 * 2 = 2       → 行 2 和行 3
  col_base_left  = (5 % 4) * 2 = 1 * 2 = 2       → 列 2, 3
  col_base_right = 2 + 8 = 10                     → 列 10, 11

  持有元素：
  ┌───────────────────────────────────────────────────────────────────┐
  │  (2, 2)  (2, 3)  (2,10)  (2,11)                                  │
  │  (3, 2)  (3, 3)  (3,10)  (3,11)                                  │
  └───────────────────────────────────────────────────────────────────┘

  在 16×16 矩阵中的位置（× 标记）：

  col: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
  row 0:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
  row 1:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
  row 2:  .  .  ×  ×  .  .  .  .  .  .  ×  ×  .  .  .  .   ← t=5
  row 3:  .  .  ×  ×  .  .  .  .  .  .  ×  ×  .  .  .  .   ← t=5
  row 4:  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
  ...
```

---

### 7.5 为什么 S 矩阵的同一行散落在多个 warp

Prefill kernel 中 `BLOCK_SIZE_Q=32, BLOCK_SIZE_KV=32`，S 矩阵是 [32 × 32]。

用 4 个 [16×16] 的 HMMA tile 覆盖整个 S：

```
  S [32 × 32]  被 4 个 warp 各自持有一个 [16×16] tile：

         KV 列 0..15        KV 列 16..31
  Q行  ┌──────────────────┬──────────────────┐
  0..15│   warp 0 持有    │   warp 1 持有    │
       │  acc_s [16×16]  │  acc_s [16×16]  │
  ─────├──────────────────┼──────────────────┤
 16..31│   warp 2 持有    │   warp 3 持有    │
       │  acc_s [16×16]  │  acc_s [16×16]  │
       └──────────────────┴──────────────────┘

  S 的第 5 行（Q 行 5）完整向量 = [S[5,0..15], S[5,16..31]]
                                     ↑ 在 warp 0 的寄存器中    ↑ 在 warp 1 的寄存器中
```

对 warp 0 内的线程，第 5 行（row_base=4，即 t=8..11）的元素分布：

```
  warp 0 中，S[5, *] 由哪些线程持有？
    row 5 = row_base + 1 where row_base = 4 → t = 8,9,10,11（4 个线程）

  t=8:  持有 (4,0)(4,1)(5,0)(5,1)(4,8)(4,9)(5,8)(5,9)
              ↑   ↑   ↑   ↑                ↑   ↑  ← 第5行的4个元素
  t=9:  持有 (4,2)(4,3)(5,2)(5,3)(4,10)(4,11)(5,10)(5,11)
  t=10: 持有 (4,4)(4,5)(5,4)(5,5)(4,12)(4,13)(5,12)(5,13)
  t=11: 持有 (4,6)(4,7)(5,6)(5,7)(4,14)(4,15)(5,14)(5,15)

  → S[5, 0..15] 分散在 warp 0 的 t=8..11 的寄存器中（每人持有 4 个）
  → S[5,16..31] 分散在 warp 1 的 t=8..11 的寄存器中（每人持有 4 个）

  求 max(S[5, 0..31]) 需要：
   Step a) warp 0 内：t=8..11 互相 shfl_xor → 得到 S[5, 0..15] 的 max
   Step b) warp 1 内：t=8..11 互相 shfl_xor → 得到 S[5,16..31] 的 max
   Step c) 两个 partial max 写入 smem_reduce_buf，__syncthreads()
   Step d) 任一 warp 读取两个值，取全局 max(S[5, 0..31])
```

这就是 §6.4④ 中 cross-warp reduction 的硬件根因：**同一 Q 行的 score 向量在 MMA fragmentation 下跨越了多个 warp 的寄存器**，无法在 warp 内单独完成归约。

---

### 7.6 数值示例：HMMA tile 的 Q·K^T 计算

以 `head_dim=16`（便于展示，实际为 128），`BLOCK_SIZE_Q=16, BLOCK_SIZE_KV=16`, 1 个 warp 为例：

```
Q_tile  [16 × 16] bf16（来自 SMEM）
K_tile^T [16 × 16] bf16（KV tile 转置后，来自 SMEM）

单条 HMMA m16n16k16：
  acc_s [16 × 16] fp32 += Q_tile @ K_tile^T

实际 head_dim=128 时，需要 128/16 = 8 次 HMMA 指令才能完成 Q·K^T：

  iter k=0: acc_s += Q[:, 0:16]  @ K[:,  0:16]^T     ← 使用 head_dim 的前 16 维
  iter k=1: acc_s += Q[:, 16:32] @ K[:,16:32]^T
  ...
  iter k=7: acc_s += Q[:,112:128]@ K[:,112:128]^T

每次 iter k，Q 和 K 各从 SMEM 的不同列段取 [16×16] 的子矩阵，
acc_s 在 8 次迭代中持续累加（不清零），最终得到完整的 S = Q·K^T
这 8 次 HMMA 对应模板参数 NUM_MMA_D = head_dim / 16 = 8
```

---

### 7.7 A/B 操作数（输入矩阵）的 Fragment 布局

输入矩阵的 fragment 布局与累加器不同（bf16，每线程持有更少字节）：

```
A fragment（m16n16k16，row_major bf16）：
  16×16 bf16 = 512 bytes，32 线程各持有 16 bytes（8 个 bf16）

  线程 t 持有 A 的哪些元素：
    row = (t % 16)                  // 0..15，每线程负责固定的行
    col_pair = (t / 16) * 8         // 0 或 8（高低 8 列各由 16 个线程负责）
    持有列：col_pair + 0 .. col_pair + 7

  简化视图：
    t=0..15  持有 A[t%16, 0:8]   （左 8 列）
    t=16..31 持有 A[t%16, 8:16]  （右 8 列）

B fragment（m16n16k16，col_major bf16，即 K^T）：
  与 A 类似，但按 col_major 读取：
    t=0..15  持有 B[0:8,  t%16]  （上 8 行 → 对应 K 的 8 个 head_dim 维度）
    t=16..31 持有 B[8:16, t%16]  （下 8 行）
```

对 `cp.async` 加载到 SMEM 后的布局：Q tile 以 row_major（行优先）存储，K tile 以 col_major（列优先）存储，正好与 A/B fragment 的读取模式匹配，避免额外的 SMEM bank conflict。

---

### 7.8 Fragment 布局对 Online Softmax 代码的影响

理解这一布局后，softmax 归约代码就自然而然了：

```cpp
// acc_s 是刚完成 NUM_MMA_D 次累加后的 fp32 fragment [16×16]
// 每线程持有 8 个 fp32

// ─── Step 1：warp 内 reduce，得到本 warp 这 16 列的行 max ───
// 线程 t 持有 row_base 和 row_base+1 两行各 4 个元素（跨 8 列）
// 先对同行的 4 个元素做 warp-level reduce（利用 shfl_xor mask = 0b0111）

float row_max_left  = fmaxf(c_frag.x[0], fmaxf(c_frag.x[1],
                      fmaxf(c_frag.x[4], c_frag.x[5])));  // row_base 行，左+右 4 对
float row_max_right = fmaxf(c_frag.x[2], fmaxf(c_frag.x[3],
                      fmaxf(c_frag.x[6], c_frag.x[7])));  // row_base+1 行

// 跨同行不同 col_pair 的线程做 warp shfl_xor（mask = 0b0100, 0b0010, 0b0001）
// 最终每个线程得到本 warp 负责的 16 列中，自己那行的 max

// ─── Step 2：写 SMEM + __syncthreads() ───
// (见 §6.4④ 跨 warp 归约)

// ─── Step 3：用 m_new 就地更新 acc_s（exp 操作，准备 PV GEMM 的 P 矩阵）───
// c_frag.x[i] = expf(c_frag.x[i] - m_new[对应行]);
// 注意：同一线程持有两行（row_base 和 row_base+1），m_new 要取对应行的值
#pragma unroll
for (int i = 0; i < 8; ++i) {
    float m_row = (i < 4) ? m_new[row_base] : m_new[row_base + 1];
    c_frag.x[i] = expf(c_frag.x[i] - m_row);  // exp(S - m) = P（未归一化）
}
// 现在 c_frag 就是 P 矩阵的 fragment，直接用于 PV GEMM 的 A 操作数
```

**关键点**：`c_frag.x[0..3]` 属于 `row_base` 行，`c_frag.x[4..7]` 属于 `row_base+1` 行，softmax 的 exp 操作需要**按寄存器槽选取对应行的 `m_new`**，这完全由 fragment 布局决定。

---

### 7.9 SMEM Bank Conflict 分析

HMMA 指令从 SMEM 读 A/B 操作数时，32 个线程的访问模式决定是否有 bank conflict：

```
SMEM bank 数：32（128 字节对齐，每 bank 4 字节）

A fragment 读取模式（Q tile，row_major，16×16 bf16）：
  线程 t=0..15  读 row t%16，col 0..7（连续 4×bf16 = 8 bytes = 2 banks/线程）
  线程 t=16..31 读 row t%16，col 8..15（同理）

  同一 warp 的 32 线程访问 16 个不同行：
    t=0 和 t=16 都读 row 0，但读不同列段（col 0..7 vs col 8..15）
    → 每行 32 字节，跨 8 个 bank（stride = 2），32 线程分布在 8 行
    → bank 分布均匀，无 conflict

Q tile 加载时使用 __ldmatrix 指令（PTX ldmatrix.sync.aligned）：
  专门为 HMMA 操作数设计的向量化 SMEM 读取指令
  一条指令让 warp 的 32 线程协作加载整个 fragment，硬件自动避免 conflict
  比普通 ld.shared 快约 2×
```

---

### 7.10 小结：Fragment 布局速查

| 参数 | 值 | 说明 |
|------|-----|------|
| HMMA tile | 16 × 16 × 16 | M, N, K 维度（bf16 输入，fp32 累加） |
| 参与线程 | 32（1 warp） | 必须整 warp 同步执行 |
| 累加器每线程 | 8 个 fp32（32 bytes） | = M×N/线程数 = 256/32 |
| 覆盖行（每线程） | 连续 2 行 | row_base = (lane/4)*2 |
| 覆盖列（每线程） | 左 4 + 右 4（两段） | col = (lane%4)*2, (lane%4)*2+8 |
| A/B 每线程 | 8 个 bf16（16 bytes） | 单行 8 个连续元素 |
| SMEM 读指令 | `ldmatrix.x4` | 4×4×bf16 = 32 bytes/线程 |
| Online softmax 影响 | 同行元素跨 warp 分布 → 需跨 warp 归约 | 见 §6.4④ |

---

## 8. Split-K 归约 Kernel

当 `num_splits > 1` 时（Decode 的并行化策略），每个 (request, kv_head) 被切成多段并行计算，最后需要合并：

```cpp
// csrc/batch_decode.cu -> MergeStateInPlace / MergeStatesIntoResultsKernel

template <uint32_t VEC_SIZE, uint32_t BDX, uint32_t BDY,
          typename DType>
__global__ void MergeStatesIntoResultsKernel(
    const float* __restrict__ partial_o,  // [batch * kv_heads * splits, qo_heads, head_dim]
    const float* __restrict__ partial_lse,// [batch * kv_heads * splits, qo_heads]
    DType* __restrict__ merged_o,         // [batch * qo_heads, head_dim]
    float* __restrict__ merged_lse,
    uint32_t num_heads, uint32_t num_splits, uint32_t head_dim) {

    // 每个 block 处理一个 (batch, qo_head) 的归约
    uint32_t batch_head_idx = blockIdx.x;

    // 初始化合并状态
    float m = -INFINITY, d = 0.f;
    float o_acc[head_dim];  // = 0.f，寄存器

    // 遍历所有 splits，使用与 online softmax 相同的合并公式
    for (uint32_t split = 0; split < num_splits; ++split) {
        float m_s = partial_lse[...split...];  // 读取第 split 段的 m (log-sum-exp)
        float d_s = __expf(m_s);               // d_s = exp(m_s) = Z_s（归一化常数）

        // 与当前累积状态合并
        float m_new = fmaxf(m, m_s);
        float scale_cur   = __expf(m - m_new);
        float scale_split = __expf(m_s - m_new);

        d = d * scale_cur + d_s * scale_split;

        // 更新输出累积器
        for (int i = 0; i < head_dim / VEC_SIZE; ++i) {
            // 读取 partial_o[split] 对应的 VEC_SIZE 个元素
            float o_split[VEC_SIZE] = load_vec(partial_o, split, i);
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j)
                o_acc[i*VEC_SIZE+j] = o_acc[i*VEC_SIZE+j] * scale_cur
                                    + o_split[j] * scale_split;
        }
        m = m_new;
    }

    // 最终归一化
    float d_rcp = 1.f / d;
    for (int i = 0; i < head_dim; ++i)
        merged_o[batch_head_idx * head_dim + i] = DType(o_acc[i] * d_rcp);
}
```

**数学等价性**：这个 merge 操作与 Online Softmax 的增量更新完全等价，是 FA2 算法的核心恒等式：

$$
m_{\text{new}} = \max(m_1, m_2)
$$
$$
d_{\text{new}} = d_1 e^{m_1 - m_{\text{new}}} + d_2 e^{m_2 - m_{\text{new}}}
$$
$$
O_{\text{new}} = \frac{d_1 e^{m_1 - m_{\text{new}}} O_1 + d_2 e^{m_2 - m_{\text{new}}} O_2}{d_{\text{new}}}
$$

---

## 9. plan() 的 CPU-GPU 异步流水线

mini-sglang 中 `non_blocking=True` 的含义：

```
CPU 时间线：
  t=0: prepare_metadata()
       → 构造 cu_seqlens_k_cpu (pinned memory)
       → 构造 indices (GPU tensor，直接 cat)
  t=1: _initialize_metadata_once()
       → wrapper.plan(indptr=cu_seqlens_k_cpu, ..., non_blocking=True)
          ├─ CPU 端：在 pinned memory 上做 CSR 转换计算
          └─ H2D copy: pinned → GPU int_workspace_buffer
             使用单独 CUDA stream，non_blocking

  t=2: CPU 立即返回，开始下一层的 Python 逻辑

GPU 时间线：
  ...正在执行上一层的 Attention kernel...
  ...H2D copy（pinned → int_workspace_buffer）并行进行...
  ...
  等 run() 被调用时，H2D copy 已完成（CUDA stream 保证顺序）

wrapper.run(q, paged_kv_cache)
  → CUDA 检查 stream 依赖：H2D copy 已完成
  → 启动 BatchDecodeWithPagedKVCacheKernel
```

**为什么 `non_blocking=True` 是安全的**：`plan()` 使用的 non_blocking H2D copy 和 `run()` 的 kernel launch 在同一个 CUDA stream 中，CUDA runtime 保证同一 stream 内的操作按提交顺序串行执行，H2D copy 一定先于 kernel 完成。

---

## 10. CUDA Graph 专用 Wrapper 的实现

`CUDAGraphBatchDecodeWithPagedKVCacheWrapper` 区别于普通 wrapper 的关键点：

```cpp
// 普通 Wrapper 的 plan() → GPU 分配新 buffer
class BatchDecodeWithPagedKVCacheWrapper {
    void plan(...) {
        // 每次 plan 都可能 malloc 不同大小的 int_workspace
        this->int_workspace_buffer = alloc_or_reuse_buffer(required_size);
        ...
    }
};

// CUDA Graph Wrapper：所有 buffer 地址在创建时由外部绑定，不再变化
class CUDAGraphBatchDecodeWithPagedKVCacheWrapper {
    // 构造时一次性绑定所有 buffer 地址
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
        float* float_workspace,
        QKVLayout kv_layout,
        bool use_tensor_cores,
        int32_t* indptr_buffer,       // 外部预分配，地址固定
        int32_t* indices_buffer,      // 外部预分配，地址固定
        int32_t* last_page_len_buffer // 外部预分配，地址固定
    ) {
        this->indptr_buf_   = indptr_buffer;   // 记录外部 buffer 地址
        this->indices_buf_  = indices_buffer;
        this->last_plen_buf_ = last_page_len_buffer;
    }

    void plan(...) {
        // 将新的 indptr/indices 值 copy 到固定地址的 buffer
        cudaMemcpyAsync(this->indptr_buf_, new_indptr, ...);    // 目标地址不变！
        cudaMemcpyAsync(this->indices_buf_, new_indices, ...);
        // CUDA Graph 录制时这些地址已被 capture，回放时自动使用
    }
};
```

mini-sglang 中的对应关系（来自 `fi.py`）：

```python
# 构造时绑定 capture data 的固定地址
self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
    self.float_workspace_buffer,
    indptr_buffer   = capture.cu_seqlens_k[: bs + 1],  # ← GPU tensor，地址固定
    indices_buffer  = capture.indices,                   # ← GPU tensor，地址固定
    last_page_len_buffer = capture.one_tensor[:bs],      # ← GPU tensor，地址固定
)

# 每次回放前 copy 新数据到固定地址
# （由 _initialize_metadata_once 内的 plan() 触发 cudaMemcpyAsync）
```

---

## 11. float_workspace_buffer 内存布局

128 MB 的 `float_workspace_buffer` 在 Split-K 路径下的使用：

```
float_workspace_buffer 布局（Decode，num_splits=4 为例）：

offset 0:
  partial_o:  [batch_size * num_kv_heads * num_splits, num_qo_heads_per_kv, head_dim]
              float32（因为是中间结果，必须高精度）

offset partial_o_size:
  partial_lse: [batch_size * num_kv_heads * num_splits, num_qo_heads_per_kv]
               float32（log-sum-exp 值）

具体大小（LLaMA-3 8B，bs=1，8KV heads，4splits，GQA=4，d=128）：
  partial_o: 1 * 8 * 4 * 4 * 128 * 4 bytes = 65536 bytes = 64 KB
  partial_lse: 1 * 8 * 4 * 4 * 4 bytes    = 512 bytes
  总计: ~65 KB，远小于 128 MB，所以 128 MB 绰绰有余
```

Prefill 路径下，`float_workspace_buffer` 的用途不同：

```
Prefill（variable-length batching）：
  lse_buffer: [total_q_tokens, num_qo_heads] float32
  // plan 时根据 total_q_tokens 动态决定使用多少，但不超过 128MB
```

---

## 12. 关键设计对比总结

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  FlashInfer FA2 Kernel 关键参数一览                      │
├───────────────┬────────────────────────────────────────────────────────┤
│  Kernel       │  BatchDecodeWithPagedKVCache                           │
│  (GEMV 路径)   │                                                        │
├───────────────┼────────────────────────────────────────────────────────┤
│ Grid          │ (batch_size * num_kv_heads * num_splits, 1, 1)         │
│ Block         │ (BDX, BDY, 1)  BDX=head_dim/VEC_SIZE, BDY=GQA_group   │
│ SMEM          │ K/V double-buffer: NUM_STAGES * 2 * BDY * head_dim     │
│ 寄存器         │ Q 向量(BDX) + online state(head_dim floats + 2 floats) │
│ 内存访问       │ Q: 1次 HBM read; KV: 每page 2次 cp.async (ping-pong)   │
├───────────────┼────────────────────────────────────────────────────────┤
│  Kernel       │  BatchDecodeWithPagedKVCache (Tensor Core)             │
│  (wgmma 路径)  │                                                        │
├───────────────┼────────────────────────────────────────────────────────┤
│ Grid          │ (batch_size * num_kv_heads_groups * num_splits, 1, 1)  │
│ Block         │ (NUM_WARPS * 32, 1, 1)                                 │
│ SMEM          │ Q smem: GQA * head_dim; KV pipeline: NUM_STAGES * ...  │
│ 寄存器         │ acc_s: GQA * BLOCK_KV floats (fragmentation 分布)      │
│ 关键约束       │ GQA_GROUP_SIZE 需是 wgmma tile 高度的倍数 (≥16 最优)    │
├───────────────┼────────────────────────────────────────────────────────┤
│  Kernel       │  BatchPrefillWithPagedKVCache                          │
├───────────────┼────────────────────────────────────────────────────────┤
│ Grid          │ (num_q_blocks, num_qo_heads, 1)                        │
│ Block         │ (NUM_WARPS * 32, 1, 1)                                 │
│ SMEM          │ Q tile驻留 + KV double-buffer                          │
│ 关键优化       │ plan() 预计算 Q→KV tile 映射；causal mask 仅在对角tile  │
│ Causal mask   │ 非对角 tile (KV全在Q之前) 跳过 mask 检查               │
└───────────────┴────────────────────────────────────────────────────────┘
```

---

## 13. mini-sglang 调用路径与 Kernel 对照

```
fi.py: FlashInferBackend.forward(q, k, v, layer_id, batch)
  │
  ├─ _initialize_metadata_once(metadata)
  │    └─ wrapper.plan(...)          ← 触发 plan，(已在 prepare_metadata 时完成)
  │
  ├─ kvcache.store_kv(k, v, ...)    ← 写入 paged KV cache（StoreKernel）
  │
  └─ metadata.wrapper.run(q, paged_kv_cache)
       │
       ├─ [Prefill] BatchPrefillWithPagedKVCacheKernel
       │    grid: (num_q_blocks, num_qo_heads)
       │    每 block: Q tile 驻 SMEM，循环 KV tiles（paged 间接寻址）
       │    online FA2 softmax，写 o, lse
       │
       └─ [Decode] BatchDecodeWithPagedKVCacheKernel
            grid: (batch * kv_heads * splits, 1)
            │
            ├─ [GEMV, GQA<4]  每 block: BDY=GQA Q heads 并行，内积累加
            │                  cp.async 双缓冲 KV，shfl reduce
            │
            └─ [wgmma, GQA≥4] 每 block: GQA heads 打包成矩阵，wgmma MMA
                               多个 splits 并行 → MergeStatesKernel 归约
```
