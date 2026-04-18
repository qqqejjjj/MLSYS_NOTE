# Mini-SGLang Kernel 模块深度解析

本文档对 Mini-SGLang 的 `minisgl.kernel` 模块进行系统性的代码分析。该模块是整个推理引擎的性能核心，包含了专为高吞吐量 LLM 推理设计的全套底层算子：CUDA JIT/AOT 内核、Triton 内核以及 CPU 高速比较算法。

## 0. 模块总览

`minisgl.kernel` 对外暴露以下 7 个核心接口（来自 `kernel/__init__.py`）：

```python
from .index   import indexing                            # 词表 Embedding 查找（CUDA JIT）
from .store   import store_cache                         # KV Cache 分散写入（CUDA JIT）
from .radix   import fast_compare_key                    # RadixCache 前缀比较（AOT C++）
from .tensor  import test_tensor                         # 张量相等测试（AOT C++）
from .pynccl  import PyNCCLCommunicator, init_pynccl     # NCCL 通信器（AOT CUDA）
from .moe_impl import fused_moe_kernel_triton            # 融合 MoE 矩阵乘（Triton JIT）
from .moe_impl import moe_sum_reduce_triton              # MoE TopK 加权求和（Triton JIT）
```

这些内核按照编译方式可以分为三类：

```
┌────────────────────────────────────────────────────────────────────────┐
│                        minisgl.kernel 编译策略                         │
├───────────────────────┬─────────────────────┬──────────────────────────┤
│        JIT (模板实例化) │      AOT (预编译)    │     Triton JIT          │
│    csrc/jit/*.cu       │   csrc/src/*.cpp/cu  │   triton/*.py           │
├───────────────────────┼─────────────────────┼──────────────────────────┤
│  index.cu (IndexKernel)│  radix.cpp          │  fused_moe_kernel        │
│  store.cu (StoreKernel)│  pynccl.cu          │  moe_sum_reduce_kernel   │
│                        │  tensor.cpp         │                          │
├───────────────────────┼─────────────────────┼──────────────────────────┤
│ load_inline() 首次调用  │ load() 安装时编译   │ @triton.jit 装饰器       │
│ @functools.cache 缓存  │ 无运行时模板参数     │ autotuning 自动调优      │
└───────────────────────┴─────────────────────┴──────────────────────────┘
```

**文件结构**：

```
kernel/
├── __init__.py          ← 公共接口导出
├── __main__.py          ← 开发工具：生成 .clangd 配置
├── index.py             ← indexing() Python 包装器
├── store.py             ← store_cache() Python 包装器
├── radix.py             ← fast_compare_key() Python 包装器
├── tensor.py            ← test_tensor() Python 包装器
├── pynccl.py            ← init_pynccl() / PyNCCLCommunicator 包装器
├── moe_impl.py          ← fused_moe_kernel_triton() / moe_sum_reduce_triton()
├── utils.py             ← 编译基础设施：KernelConfig, load_aot, load_jit
├── triton/
│   └── fused_moe.py     ← Triton 内核源码
└── csrc/
    ├── jit/
    │   ├── index.cu     ← IndexKernel CUDA 模板（JIT）
    │   └── store.cu     ← StoreKernel CUDA 模板（JIT）
    ├── src/
    │   ├── radix.cpp    ← fast_compare_key 实现（AOT）
    │   ├── tensor.cpp   ← test_tensor 实现（AOT）
    │   └── pynccl.cu    ← NCCLWrapper 实现（AOT CUDA）
    └── include/minisgl/
        ├── warp.cuh     ← warp::copy<N> / warp::reset<N> 原语
        ├── utils.cuh    ← PDL::wait/launch, pointer::offset, LaunchKernel
        ├── utils.h      ← host::RuntimeCheck, panic
        ├── tensor.h     ← Tensor 视图工具
        └── nccl227.h    ← NCCL 2.27 API 头文件
```

---

## 1. 编译基础设施（`utils.py`）

理解整个 kernel 模块的关键，在于理解 Mini-SGLang 如何通过 **TVM FFI** 桥接 Python 与 C++/CUDA。

### 1.1 KernelConfig — 内核配置描述符

```python
class KernelConfig(NamedTuple):
    num_threads:    int   # 每个 CUDA block 的线程数（默认 128）
    max_occupancy:  int   # 最大并发 block 数量（默认 1）
    use_pdl:        bool  # 是否启用 PDL（H100+ Programmatic Dependent Launch）

    @property
    def template_args(self) -> str:
        # 生成 C++ 模板实例化参数字符串，供 JIT 编译使用
        return f"{self.num_threads},{self.max_occupancy},{'true' if self.use_pdl else 'false'}"
```

`template_args` 的输出会直接拼接进 C++ 模板实例化语句，例如：

```cpp
// template_args = "128,1,false" 对应：
IndexKernel<16384, 4, 128, 1, false>::run(...)
```

### 1.2 AOT 编译（`load_aot`）

```python
def load_aot(*args, cpp_files=(), cuda_files=(), ...):
    from tvm_ffi.cpp import load
    # 在包安装时（pip install）完成编译
    # 产物为共享库 .so，运行时直接 dlopen
    return load(*args, cpp_files=cpp_files, cuda_files=cuda_files, ...)
```

**特点**：
- 编译发生在安装阶段，第一次 import 时开销为零（只是 dlopen）
- 不支持 C++ 模板参数在运行时变化——模板参数必须在编译时固定
- 适用于：`radix.cpp`（`fast_compare_key`）、`tensor.cpp`（`test_tensor`）、`pynccl.cu`（`NCCLWrapper`）

### 1.3 JIT 编译（`load_jit`）

```python
def load_jit(*args, cuda_files, cuda_wrappers, ...):
    from tvm_ffi.cpp import load_inline
    # 第一次调用时触发编译，@functools.cache 确保同一参数只编译一次
    code = _make_wrapper(cuda_wrappers, ...)
    return load_inline(*args, cuda_files=cuda_files, extra_cuda_code=code, ...)
```

其中 `_make_wrapper` 负责生成导出包装函数：

```cpp
// cuda_wrappers=[("launch", "StoreKernel<16384, 128, 1, false>::run")]
// 生成：
TVM_FFI_DLL_EXPORT_TYPED_FUNC(launch, (StoreKernel<16384, 128, 1, false>::run));
```

这样，Python 就可以通过 `module.launch(...)` 调用已实例化的 C++ 模板函数。

**特点**：
- `@functools.cache` 装饰器确保同一 `element_size` 的内核全进程生命周期内只编译一次
- 支持运行时决定模板参数（根据张量形状按需实例化不同特化版本）
- 适用于：`index.cu`（`IndexKernel`）、`store.cu`（`StoreKernel`）

### 1.4 编译路径总结

```
Python 调用 indexing(weight, indices, output)
        │
        ▼
index.py: _jit_index_module(element_size, num_splits)
        │
        ├─ cache hit？→ 返回已编译模块 (functools.cache)
        │
        └─ cache miss？→ load_jit(
               cuda_files=["jit/index.cu"],
               cuda_wrappers=[("launch", "IndexKernel<16384,4,128,1,false>::run")],
               ...
           )
           │
           ▼
       tvm_ffi.cpp.load_inline() → nvcc 编译 → .so 加载
           │
           ▼
       module.launch(weight, indices, output, num_tokens)
```

---

## 2. IndexKernel — 词表 Embedding 查找

### 2.1 功能与场景

`indexing()` 实现了 **词表 Embedding 查找（Vocabulary Embedding Lookup / Gather）**：给定 token ID 列表 `indices`，从词表权重矩阵 `weight` 中逐行 gather 对应的 embedding 向量，写入输出缓冲区。

在 LLM 推理流程中，这是 Transformer 的第一步——将离散的整数 token ID 序列转换为连续的浮点向量序列，供后续注意力层使用：

```
weight:  [vocab_size, hidden_dim]   ← 词表权重矩阵（存放在 GPU HBM）
indices: [num_tokens]               ← token ID 序列（GPU 上的 int32/int64）
output:  [num_tokens, hidden_dim]   ← gather 后的 embedding（GPU HBM）

indexing(weight, indices, output)
```

---

### 2.2 核心问题：Embedding 查找是矩阵乘法吗？

这是理解 IndexKernel 设计动机的关键问题。

#### 2.2.1 数学上的等价性

从线性代数角度，Embedding 查找确实**等价于**一次矩阵乘：

设词表权重 $W \in \mathbb{R}^{V \times D}$（$V$ = vocab_size，$D$ = hidden_dim），对 token ID 为 $i$ 的查找，可以用 **one-hot 向量** $\mathbf{e}_i \in \mathbb{R}^{V}$ 表示：

$$\text{embed}(i) = \mathbf{e}_i \cdot W = W[i, :] \in \mathbb{R}^D$$

其中 $\mathbf{e}_i$ 只有第 $i$ 维为 1，其余全为 0。对一批 $N$ 个 token，设指标矩阵 $E \in \{0,1\}^{N \times V}$（每行恰好一个 1），则：

$$\text{Output} = E \cdot W \in \mathbb{R}^{N \times D}$$

这在形式上就是一次 GEMM（General Matrix Multiplication）。

#### 2.2.2 为什么实现上绝对不能用 GEMM

虽然数学等价，但**用 GEMM 实现 Embedding 查找是灾难性的**。以 LLaMA-3 8B 为例：

| 参数 | 值 |
|---|---|
| vocab_size $V$ | 128,000 |
| hidden_dim $D$ | 4,096 |
| batch token 数 $N$ | 32 |
| dtype | bfloat16（2 字节） |

**GEMM 的实际工作量**：
- 矩阵 $E$ 的形状：$[32, 128000]$，每行仅 1 个非零元素
- 矩阵乘 FLOPs：$N \times V \times D = 32 \times 128000 \times 4096 \approx 1.68 \times 10^{10}$
- 其中有意义的计算：仅 $N \times D = 32 \times 4096 = 131{,}072$ 次乘法（全部 ×1）
- **无效运算占比：$\approx 99.999\%$（其余全是乘以 0）**

**Gather（IndexKernel）的工作量**：
- 无任何乘法，只有内存读写
- 读取字节数：$N \times D \times 2 = 32 \times 4096 \times 2 = 262{,}144$ 字节（256 KB）
- 写入字节数：相同

对比如下：

```
╔══════════════════════════════════════════════════════════════════╗
║  方法         │ FLOPs              │ 内存读写    │ 有效计算率    ║
╠══════════════════════════════════════════════════════════════════╣
║  GEMM         │ ~1.68×10¹⁰         │ 读 W + 读 E │ 0.001%       ║
║  IndexKernel  │ 0（纯内存复制）     │ 只读目标行  │ 100%         ║
╚══════════════════════════════════════════════════════════════════╝
```

#### 2.2.3 正确的实现：纯内存 Gather（零浮点运算）

IndexKernel 的精妙之处在于它直接利用了 "one-hot × dense = row selection" 的数学事实：

> **当左矩阵是 one-hot 矩阵时，矩阵乘退化为行选取操作，不需要任何乘法，只需内存寻址和复制。**

因此 IndexKernel 是一个 **100% memory bandwidth-bound 内核**，内核体内不含任何浮点指令，只有：
1. 计算源地址：`weight + indices[warp_id] * element_size`
2. 计算目标地址：`output + warp_id * element_size`
3. 协作复制字节：`warp::copy<kSizePerWarp>(dst, src)`

性能瓶颈完全在于 GPU HBM（High Bandwidth Memory）的带宽，而不是 CUDA core 或 Tensor Core 的算力。

---

### 2.3 Python 接口详解（`index.py`）

```python
# ──────────────────────────────────────────────────────────────────
# 默认内核配置：128 线程/block，1 个 block/SM，不启用 PDL
DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128,
                                           max_occupancy=1,
                                           use_pdl=False)

@functools.cache                          # 进程级缓存，同参数只编译一次
def _jit_index_module(element_size: int,
                      *,
                      num_splits: int = 1,
                      config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG) -> Module:
    # make_cpp_args 将参数拼成 "16384,4,128,1,false" 这样的字符串
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        # 告诉 JIT 需要生成两个导出函数：
        #   module.launch(...)        → IndexKernel<args>::run
        #   module.launch_masked(...) → IndexKernel<args>::run（masked 版）
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )

def indexing(weights, indices, *, output=None, vocab_range=None):
    # output=None 时自动分配，避免调用方管理缓冲区
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()

    # 根据行大小选择 num_splits，决定每行分配多少个 warp
    if element_size % 2048 == 0:   num_splits = 4
    elif element_size % 1024 == 0: num_splits = 2
    else:                          num_splits = 1

    module = _jit_index_module(element_size, num_splits=num_splits)

    # vocab_range=None：普通 gather（index_kernel）
    # vocab_range=(start, length)：TP 词表分段（masked_index_kernel）
    module.launch(weights, indices, output, vocab_range)
    return output
```

注意 `module.launch` 调用的实际语义：Python 传入两个 Tensor 和一个 `Optional[Tuple[int,int]]`，TVM FFI 会在 C++ 侧根据 `mask_opts.has_value()` 决定分派给 `index_kernel` 还是 `masked_index_kernel`。

---

### 2.4 完整 CUDA 代码逐行走读

#### 2.4.1 参数结构体

```cpp
// 普通 gather 的参数包
struct IndexKernelParams {
  void *__restrict__ output;         // 输出缓冲，__restrict__ 声明无别名
  const void *__restrict__ weight;   // 词表权重矩阵，只读
  const void *__restrict__ indice;   // token ID 数组（int32/int64 多态）
  std::size_t num_warps;             // 总 warp 数 = num_tokens × num_splits
};

// Masked gather 增加词表分段信息
struct MaskedKernelParams {
  IndexKernelParams params;
  std::size_t start;    // 本 GPU 负责的词表起始 ID
  std::size_t length;   // 本 GPU 负责的词表长度
};
```

**设计点**：将所有参数打包成一个 struct，原因是：
- 减少传参寄存器占用（单个 struct 指针 vs. 多个零散参数）
- 配合 `__grid_constant__` 修饰（见 2.5.2），使整个 struct 从寄存器传递升级为常量内存广播

#### 2.4.2 `index_kernel` 逐行解析

```cpp
template <std::size_t kNumThreads,    // 编译期常量：block 内线程数（=128）
          std::size_t kMaxOccupancy,  // 编译期常量：每 SM 最大并发 block 数（=1）
          bool kUsePDL,               // 编译期常量：是否启用 PDL
          std::size_t kElementSize,   // 编译期常量：每行字节数
          std::size_t kNumSplits,     // 编译期常量：每行的 warp 数
          std::integral T>            // 模板类型参数：int32_t 或 int64_t（索引类型）
__global__
__launch_bounds__(kNumThreads, kMaxOccupancy)  // ← CUDA 特性 (见 2.5.1)
void index_kernel(
    const __grid_constant__ IndexKernelParams params)  // ← CUDA/Hopper 特性 (见 2.5.2)
{
  using namespace device;

  // 编译期计算的常量（全部在寄存器文件之前被 nvcc 展开为立即数）
  constexpr auto kSize        = kElementSize;
  constexpr auto kSizePerWarp = kSize / kNumSplits;   // 每个 warp 负责的字节数
  constexpr auto kWarpPerBlock = (unsigned)(kNumThreads / 32); // = 4

  static_assert(kNumThreads % 32 == 0);               // block 必须是整数个 warp
  static_assert(std::has_single_bit(kNumSplits));      // num_splits 必须是 2 的幂
  static_assert(kElementSize % kNumSplits == 0);       // 行必须能被整数分割

  // C++17 结构化绑定：从 struct 中解构字段，可读性好且开销为零
  const auto &[output, weight, indices_, num_warps] = params;

  // indices_ 是 void*，需强转为具体的整数类型（由模板参数 T 决定）
  const auto indices = static_cast<const T *>(indices_);

  // 计算当前线程所在的全局 warp id
  // threadIdx.x / 32 → warp 在 block 内的编号（0..kWarpPerBlock-1）
  // blockIdx.x * kWarpPerBlock → 本 block 的起始 warp 偏移
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;

  PDL::wait<kUsePDL>();  // H100+ PDL：等待上游内核完成（见 2.8）

  if (warp_id < num_warps) {  // 边界保护（最后一个 block 可能有空余 warp）

    // 由于 num_splits 个 warp 共享同一行，通过整除映射回 token id
    // 示例：num_splits=4, warp_id=7 → token id = 7/4 = 1（第2个token）
    const auto pos = indices[warp_id / kNumSplits];

    // 目标地址：output[token_id * element_size + split_offset * kSizePerWarp]
    //          = output + warp_id * kSizePerWarp（利用 num_splits 分段连续）
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);

    // 来源地址：weight[pos * element_size + split_offset * kSizePerWarp]
    //   pos * kSize → 跳到词表第 pos 行的起始位置
    //   (warp_id % kNumSplits) * kSizePerWarp → 本 warp 负责该行的哪个分段
    const auto src = pointer::offset(
        weight,
        pos * kSize,                          // 行起始
        (warp_id % kNumSplits) * kSizePerWarp // 列偏移
    );

    // 32 线程协作复制 kSizePerWarp 字节（见 Section 7 的 warp::copy 详解）
    warp::copy<kSizePerWarp>(dst, src);
  }

  PDL::launch<kUsePDL>();  // H100+ PDL：通知下游内核可以启动
}
```

#### 2.4.3 `IndexKernel::run` — 启动参数计算

```cpp
template <std::size_t element_size, std::size_t num_splits = 1,
          std::size_t num_threads = 128, std::size_t max_concurrency = 1,
          bool use_pdl = false>
struct IndexKernel {
  static void run(TensorView weights, TensorView indices, TensorView output,
                  Optional<Tuple<int,int>> mask_opts) {
    // ── 类型与形状验证 ──────────────────────────────
    // TensorMatcher 验证 weights: [-1, D], output: [L, D], indices: [L]
    // SymbolicSize{"D"} 和 SymbolicSize{"L"} 在验证时自动推断实际值

    // ── 启动参数计算 ────────────────────────────────
    constexpr auto kWarpPerBlock = num_threads / 32;    // = 4（每 block 4 个 warp）

    // 总工作量 = num_token × num_splits 个 warp 任务
    const auto num_warps  = num_splits * num_indices;
    // block 数 = ceil(总 warp 数 / 每 block warp 数)
    const auto num_blocks = div_ceil(num_warps, kWarpPerBlock);

    // ── 参数打包 ────────────────────────────────────
    const auto params = IndexKernelParams{
        .output  = output.data_ptr(),
        .weight  = weights.data_ptr(),
        .indice  = indices.data_ptr(),
        .num_warps = num_warps,
    };

    // ── 内核选择与启动 ──────────────────────────────
    // 根据 indices 的 dtype（int32 or int64）在编译期选择不同的模板特化
    const auto kernel = use_int32
        ? index_kernel<num_threads, max_concurrency, use_pdl,
                       element_size, num_splits, int32_t>
        : index_kernel<num_threads, max_concurrency, use_pdl,
                       element_size, num_splits, int64_t>;

    // cudaLaunchKernelEx 支持 PDL 属性（见 2.5.4）
    LaunchKernel(num_blocks, num_threads, device)
        .with_attr(use_pdl)(kernel, params);
  }
};
```

---

### 2.5 CUDA 关键特性深度分析

#### 2.5.1 `__launch_bounds__` — 编译期 Occupancy 控制

```cpp
__global__
__launch_bounds__(kNumThreads, kMaxOccupancy)
void index_kernel(...)
```

`__launch_bounds__(maxTPB, minBPS)` 向 nvcc **编译器**声明两件事：
- `maxTPB`（= `kNumThreads` = 128）：该内核每个 block **最多**有 128 个线程
- `minBPS`（= `kMaxOccupancy` = 1）：至少需要每 SM 同时运行 1 个 block

**对寄存器分配的影响**：

H100 的 SM 有 65536 个 32-bit 寄存器。若 nvcc 不知道线程上限，它会保守地为每线程分配较少寄存器（防止超出 SM 寄存器池）。`__launch_bounds__` 告知上限后：

| 场景 | 每线程可用寄存器 | 寄存器溢出（spill） |
|---|---|---|
| 无 `__launch_bounds__`（256 线程保守估算） | ~128 | 可能 |
| `__launch_bounds__(128, 1)`（min 1 block/SM）| ~255 | 无 |

对于 IndexKernel，由于内核简单、寄存器用量少，主要收益是**消除编译器保守性**，确保 nvcc 可以内联和展开 `warp::copy` 中的 `#pragma unroll kLoopCount`。

#### 2.5.2 `__grid_constant__` — Hopper 常量参数优化

```cpp
void index_kernel(const __grid_constant__ IndexKernelParams params)
```

`__grid_constant__` 是 CUDA 12 / sm_90（H100）引入的修饰符，意义为：

> "这个参数在整个 grid 的生命周期内是常量，且所有线程读到的值相同。"

**底层实现差异**：

```
传统参数传递（无 __grid_constant__）：
  每个 SM 将参数从寄存器文件中持有
  SM 间没有共享 → 每个 warp 启动时各自从参数内存加载一次

__grid_constant__（H100）：
  参数存放在每个 SM 的 constant memory（专用缓存，64KB/SM）
  所有 warp 广播读取，L1 constant cache 命中
  → 消除每 warp 的参数加载开销
```

对于 `IndexKernelParams`（含 3 个指针 + 1 个 size_t，共 32 字节），`__grid_constant__` 意味着这 32 字节只上传一次并广播给所有线程，而不是每个线程从 launch 参数栈重复加载。

#### 2.5.3 `__restrict__` — 无别名假设与内存加载优化

```cpp
void *__restrict__ output;
const void *__restrict__ weight;
const void *__restrict__ indice;
```

`__restrict__` 告诉编译器：**这些指针指向的内存区域互不重叠（no aliasing）**。

对 GPU 代码的影响：
- 编译器可以对 `weight` 的加载生成 `LDG.E` 指令（Global Load with Eviction hint），而非默认的 `LD.E`
- 对 `const __restrict__` 指针可以进一步推断为 read-only，nvcc 可安排 **Texture Cache** 路径（`LDG.E.CI` = Cache Invalidate，不污染 L1$）

在 IndexKernel 中，由于 `weight` 只读而 `output` 只写，这帮助编译器优化 L1 缓存策略，避免写入覆盖后续读取的 `weight` 数据。

#### 2.5.4 `cudaLaunchKernelEx` + `cudaLaunchConfig_t` — 新式启动 API

```cpp
// LaunchKernel::with_attr(use_pdl) 的实现
auto with_attr(bool use_pdl) -> LaunchKernel & {
    if (use_pdl) {
        // 设置 PDL 属性：告知 CUDA runtime 该内核支持程序化流序列化
        m_attr_cache.id  = cudaLaunchAttributeProgrammaticStreamSerialization;
        m_attr_cache.val.programmaticStreamSerializationAllowed = 1;
        m_config.attrs    = &m_attr_cache;
        m_config.numAttrs = 1;
    } else {
        m_config.numAttrs = 0;  // 无额外属性，走普通路径
    }
    return *this;
}

// 最终调用
template <typename T, typename... Args>
auto operator()(T &&kernel, Args &&...args) const -> void {
    // cudaLaunchKernelEx 是 CUDA 11.6+ 新 API，替代 <<<grid, block, smem, stream>>>
    // 支持通过 cudaLaunchConfig_t 传递任意扩展属性（如 PDL、cooperative launch 等）
    cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...);
}
```

`cudaLaunchKernelEx` 是 `<<<>>>` 语法的扩展替代，支持：
- `cudaLaunchAttributeProgrammaticStreamSerialization`（PDL）
- `cudaLaunchAttributeCooperative`（Cooperative Groups）
- `cudaLaunchAttributeMemSyncDomainRemote`（NVLink 内存同步域）

---

### 2.6 访存模式：为何能接近峰值带宽

IndexKernel 的访存设计确保了以下三个条件同时满足，从而能够接近 H100 HBM 的理论峰值带宽（3.35 TB/s）：

#### 条件一：Coalesced Memory Access（合并访存）

一个 warp 的 32 个线程在 `warp::copy<kSizePerWarp>` 中的访问模式（以 `kUnit=16`，即 uint4 为例）：

```
warp::copy 的每一轮（kBytesPerLoop = 16 × 32 = 512 字节）：

线程 0  → 加载/存储地址: base + 0   × 16  (16字节)
线程 1  → 加载/存储地址: base + 1   × 16
线程 2  → 加载/存储地址: base + 2   × 16
...
线程 31 → 加载/存储地址: base + 31  × 16

32 × 16 = 512 字节，恰好是 L1 cache line 的整数倍
→ 触发 1 次 512-byte 事务，无任何带宽浪费（100% 合并效率）
```

#### 条件二：128-bit 向量化加载

选择 `uint4`（128-bit / 16 字节）作为加载单元，对应 PTX 指令 `LDG.E.128`。这是 H100 HBM 单线程每拍能发出的最大单次加载粒度，在线程级别实现了最大带宽利用率。

#### 条件三：避免 Bank Conflict（无共享内存）

IndexKernel 完全不使用共享内存（`__shared__`），数据直接从 HBM 读到寄存器再写回 HBM，消除了 Shared Memory Bank Conflict 的顾虑。

**实际带宽利用率估算**（H100 SXM5）：

```
H100 HBM3 理论峰值：3,350 GB/s
每个 warp 每轮：512 字节读 + 512 字节写 = 1024 字节
设 SM 数 = 132，warp 占用率 = 4（max_occupancy=1 × warps/block=4）
活跃 warp 数：132 × 1 block × 4 warp/block = 528 warp

理论吞吐（单轮）：528 × 1024 B / ~10ns ≈ 54 TB/s  (bandwidth-limit: 3.35 TB/s)
实际限制 = HBM 带宽，而非计算能力
```

---

### 2.7 `num_splits` 并行策略详解

`num_splits` 的引入解决了**单 warp 带宽利用率**与**大行宽（large hidden_dim）**之间的矛盾。

**问题**：一个 warp（32 线程，每次 16 字节）每次能复制 512 字节。若 `element_size = 16384`（如 LLaMA 8K 或 Qwen2-72B），单 warp 需要 32 轮才能复制一行，串行瓶颈明显。

**解法**：将一行在列方向切成 `num_splits` 段，每段分配给一个独立 warp **并行**处理：

```
num_splits = 4 时，一行切成 4 段：

Warp 0 → weight[pos, 0     : 4096]   → output[token, 0     : 4096]
Warp 1 → weight[pos, 4096  : 8192]   → output[token, 4096  : 8192]
Warp 2 → weight[pos, 8192  : 12288]  → output[token, 8192  : 12288]
Warp 3 → weight[pos, 12288 : 16384]  → output[token, 12288 : 16384]
（全部并行，4 个 warp 同时执行）
```

`warp_id` 到 `(token_id, split_id)` 的映射推导：
```
warp_id = token_id × num_splits + split_id

→ token_id = warp_id / num_splits   (整除)
→ split_id = warp_id % num_splits   (取余)

源地址 = weight + token_id × element_size + split_id × (element_size / num_splits)
       = weight + pos × kSize + (warp_id % kNumSplits) × kSizePerWarp    ✓（与代码一致）
```

**选择策略**：

```python
if element_size % 2048 == 0:   num_splits = 4  # 16384+ 字节行：最大并行
elif element_size % 1024 == 0: num_splits = 2  # 1024+ 字节行：适度并行
else:                          num_splits = 1  # 小行：单 warp 足够
```

这里的判断阈值 2048 和 1024 并非随意选择：
- 单 warp 每次 512 字节，`num_splits=1` 时复制 `element_size` 字节需 `element_size/512` 轮
- H100 的 warp scheduler 每个 SM 可以 issue 4 个 warp/cycle（Quad-warp issuer）
- 当行大小超过 ~1KB，4 个 warp 并行处理带来的实际吞吐提升超过调度开销

---

### 2.8 H100 专属特性：PDL（Programmatic Dependent Launch）

#### 2.8.1 传统内核依赖管理的问题

假设有两个内核 Kernel A → Kernel B（B 依赖 A 的输出），在传统 CUDA 流中：

```
Stream: |── Kernel A (全部完成) ──|── Kernel B (等A完毕才启动) ──|
         ^                        ^
         launch                   launch（等stream上的事件）
```

Kernel B 必须等待 Kernel A **整体完成**（所有 block 都退出）才能开始。这产生一个**同步气泡（sync bubble）**：Kernel A 最后几个 block 在运行时，GPU 的其余 SM 处于空闲。

#### 2.8.2 PDL 的解决方案

PDL（Programmatic Dependent Launch，H100 GH100 引入）允许：

> Kernel A 在 **完成主要工作但尚未退出** 时，主动通知 CUDA runtime 启动 Kernel B，从而让 A 的尾部执行与 B 的启动重叠。

```
Stream（有PDL）:
|── Kernel A ─────────────────────────────|
              ├─ griddepcontrol.launch_dependents
              │    ↓
              |── Kernel B 启动（A 未完全退出）─|
Timeline:      ↑
           A 的大部分工作已完成，少量尾部 block 仍在运行
```

#### 2.8.3 PTX 指令实现

Mini-SGLang 通过内联汇编直接使用 PTX 指令：

```cpp
namespace PDL {
// Kernel A 内调用：等待上游（链式时可选）
template <bool kUsePDL>
__always_inline __device__ void wait() {
    if constexpr (kUsePDL) {
        // griddepcontrol.wait：挂起当前 grid，直到上游信号到达
        // 用于多级 PDL 链（A→B→C）中 B 等待 A 的信号
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
}

// Kernel A 内调用：发出信号，通知下游可以启动
template <bool kUsePDL>
__always_inline __device__ void launch() {
    if constexpr (kUsePDL) {
        // griddepcontrol.launch_dependents：通知 runtime 下游内核可以启动
        // 此时当前 kernel 仍在运行，实现 overlap
        asm volatile("griddepcontrol.launch_dependents;" :::);
    }
}
}
```

注意两条指令在 `index_kernel` 中的位置：

```cpp
PDL::wait<kUsePDL>();               // ① 内核开头：等待（若有上游依赖）
if (warp_id < num_warps) {
    // ... 主体工作（warp::copy）...
}
PDL::launch<kUsePDL>();             // ② 内核结尾：主体工作完成后发信号
                                    //    此时 block 还未退出，但关键数据已写入
```

#### 2.8.4 `with_attr(use_pdl)` 的设置路径

要使 PDL 生效，下游内核也必须通过同样的 `cudaLaunchAttributeProgrammaticStreamSerialization` 属性标记：

```
stream.submit(index_kernel, use_pdl=True)
           ↓
   cudaLaunchKernelEx(&config) 中：
   config.attrs[0] = {
     .id  = cudaLaunchAttributeProgrammaticStreamSerialization,
     .val.programmaticStreamSerializationAllowed = 1
   }
           ↓
   CUDA runtime 在 index_kernel 执行时侦听 griddepcontrol.launch_dependents
   → 触发下游内核（如 store_kv_cache）的早期启动
```

#### 2.8.5 当前为何默认关闭（`use_pdl = False`）

- **设备兼容性**：PDL 仅支持 sm_90（H100/H200），在 A100（sm_80）或更早设备上编译会失败
- **收益场景有限**：IndexKernel 本身极轻量，主要瓶颈是 HBM 带宽。若后续内核不受 IndexKernel 限制（如 LM Head 的矩阵乘），PDL 的流水线收益不明显
- **配置成本**：需要同时修改上游和下游内核的 launch 属性，且需要对 stream 的依赖图有精确了解

预计在 H100 上对 `IndexKernel → Transformer 第一层 Attn` 的场景中启用 PDL 有意义。

---

### 2.9 Masked Variant：Tensor Parallelism 词表切分

在 Tensor Parallelism（TP）下，每个 GPU 只持有词表权重的一个分段 `W_local ∈ ℝ^{(V/tp) × D}`，覆盖 token ID 范围 `[start, start + V/tp)`。

**AllReduce 保证正确性的数学原理**：

```
GPU 0 的视角（负责 token ID [0, V/2)）：
masked_embed_0(i) = W_0[i, :]  if 0 ≤ i < V/2
                  = 0           otherwise

GPU 1 的视角（负责 token ID [V/2, V)）：
masked_embed_1(i) = W_1[i-V/2, :]  if V/2 ≤ i < V
                  = 0                otherwise

AllReduce(sum):
embed(i) = masked_embed_0(i) + masked_embed_1(i) = W[i, :]  ✓
（恰好等于全量词表查找结果，因为两分段互补，不会双重计数）
```

`masked_index_kernel` 的实现：

```cpp
template <..., std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy)
void masked_index_kernel(const __grid_constant__ MaskedKernelParams mask_params) {
  const auto &[params, start, length] = mask_params;     // 结构化绑定
  const auto &[output, weight, indices_, num_warps] = params;
  const auto indices = static_cast<const T *>(indices_);
  const auto warp_id = (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;

  PDL::wait<kUsePDL>();

  if (warp_id < num_warps) {
    // pos 先减去 start，得到在本 GPU 本地权重矩阵中的行偏移
    const auto pos = indices[warp_id / kNumSplits] - start;
    const auto dst = pointer::offset(output, warp_id * kSizePerWarp);

    if (pos < length) {
      // token 属于本 GPU 范围：正常 gather（使用本地行偏移）
      const auto src = pointer::offset(weight,
                                       pos * kSize,
                                       (warp_id % kNumSplits) * kSizePerWarp);
      warp::copy<kSizePerWarp>(dst, src);   // 复制有效数据
    } else {
      // token 不属于本 GPU 范围：零填充（AllReduce 求和后不影响结果）
      warp::reset<kSizePerWarp>(dst);       // 写入 0
    }
  }

  PDL::launch<kUsePDL>();
}
```

**注意**：当 `indices[i] - start` 溢出（无符号整数下溢，变为巨大正数）时，`pos < length` 自然为 `false`，触发 `warp::reset`，符合预期。这是一个利用无符号整数下溢行为的隐式边界检查技巧。

---

### 2.10 完整数值示例

以 **LLaMA-3 8B 模型的 Embedding 层** 为例，逐步追踪 `indexing()` 的执行：

**参数**：
- `vocab_size = 128000`，`hidden_dim = 4096`，dtype = bfloat16（2 字节）
- `num_tokens = 8`（一次 decode 的 8 个请求各有 1 个新 token）
- `indices = [5432, 102, 31000, 7, 65430, 200, 99999, 1024]`（8 个 token ID）

**Step 1：Python 层参数计算**

```python
element_size = 4096 × 2 = 8192    # 每行字节数
# 8192 % 2048 == 0 → num_splits = 4
num_splits = 4
```

**Step 2：JIT 编译（首次调用触发）**

```
_jit_index_module(element_size=8192, num_splits=4) 触发 nvcc 编译：
IndexKernel<8192, 4, 128, 1, false>::run
↓
index_kernel<128, 1, false, 8192, 4, int32_t>
```

**Step 3：启动参数计算**

```
kWarpPerBlock = 128 / 32 = 4  （每 block 4 个 warp）
num_warps     = 8 × 4 = 32    （8 个 token，每 token 4 个 warp）
num_blocks    = ceil(32 / 4) = 8
kSizePerWarp  = 8192 / 4 = 2048  （每 warp 负责 2048 字节 = 1024 个 bf16）

launch: <<<8 blocks, 128 threads>>>
```

**Step 4：GPU 上的 warp 分配**

```
warp_id = blockIdx.x × 4 + threadIdx.x / 32

┌─────────┬──────────┬───────────┬──────────────────────────────────────────────┐
│ warp_id │ token_id │  split_id │ 负责的字节范围（在 output 和 weight 中）      │
│         │ (÷4)     │  (%4)     │                                              │
├─────────┼──────────┼───────────┼──────────────────────────────────────────────┤
│    0    │    0     │     0     │ weight[5432, 0:1024]   → output[0, 0:1024]   │
│    1    │    0     │     1     │ weight[5432, 1024:2048] → output[0, 1024:2048]│
│    2    │    0     │     2     │ weight[5432, 2048:3072] → output[0, 2048:3072]│
│    3    │    0     │     3     │ weight[5432, 3072:4096] → output[0, 3072:4096]│
│    4    │    1     │     0     │ weight[102,  0:1024]   → output[1, 0:1024]   │
│    5    │    1     │     1     │ weight[102,  1024:2048] → output[1, 1024:2048]│
│   ...   │   ...    │    ...    │ ...                                          │
│   31    │    7     │     3     │ weight[1024, 3072:4096] → output[7, 3072:4096]│
└─────────┴──────────┴───────────┴──────────────────────────────────────────────┘
```

所有 32 个 warp（= 8 blocks × 4 warp/block）**完全并行**执行，无任何数据依赖。

**Step 5：单 warp 内的访存（以 warp_id=0 为例）**

```
kSizePerWarp = 2048 字节，kUnit = 16（uint4，因 2048 % (16×32) = 0）
kBytesPerLoop = 16 × 32 = 512 字节，kLoopCount = 2048 / 512 = 4

src = weight + 5432 × 8192 + 0  = weight 中第 5432 行的起始地址
dst = output + 0 × 2048          = output 起始地址

Round 0: 32 线程各加载 src[lane*16 : lane*16+16]，写入 dst[lane*16 : ...]
Round 1: 32 线程各加载 src[512 + lane*16 : ...]，写入 dst[512 + ...]
Round 2: ...（类同）
Round 3: ...（类同）
总计：4 × 32 × 16 = 2048 字节，恰好是 weight 第 5432 行的前 1/4
```

**最终结果**：`output[0, :] = weight[5432, :]`，8 行同时完成，耗时约 **~5 μs**（H100，受 HBM 带宽限制）。

---

## 3. StoreKernel — KV Cache 分散写入

### 3.1 功能与场景

`store_cache()` 实现了 **Paged KV Cache 的分散写入**：将 Attention 计算得到的 Key/Value 张量按照 `out_loc`（物理页槽位索引）写入 KV Cache 的对应位置。

```
k:       [num_tokens, num_kv_heads/tp, head_dim]   ← 当前 batch 的 K 张量
v:       [num_tokens, num_kv_heads/tp, head_dim]   ← 当前 batch 的 V 张量
indices: [num_tokens]                              ← 每个 token 对应的物理槽位 ID
k_cache: [max_num_pages, per_page_kv_size]         ← 全量 KV Cache
v_cache: [max_num_pages, per_page_kv_size]         ← 全量 KV Cache

store_cache(k_cache, v_cache, indices, k, v)
# 效果：k_cache[indices[i]] = k[i], v_cache[indices[i]] = v[i]
```

### 3.2 Python 接口（`store.py`）

```python
def store_cache(k_cache, v_cache, indices, k, v):
    # element_size：每个物理槽位存储的字节数 = (num_kv_heads/tp × head_dim) × dtype_bytes
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)  # @functools.cache 缓存

    # 底层调用 StoreKernel<element_size, 128, 1, false>::run(...)
    module.launch(k_cache, v_cache, indices, k, v)
```

### 3.3 CUDA 内核实现（`csrc/jit/store.cu`）

```cpp
template<
  std::size_t element_size,  // 每个 KV 槽位的字节数
  std::size_t num_threads = 128,
  std::size_t max_concurrency = 1,
  bool use_pdl = false>
struct StoreKernel {
  static void run(k_cache, v_cache, indices, k, v, num_tokens,
                  kv_cache_stride, kv_input_stride) {
    // 一个 warp 处理一个 token 的 K+V 写入
    // grid = ceil(num_tokens / kNumWarps) 个 block
    store_kv_cache<...><<<grid, num_threads, 0, stream>>>(...);
  }
};

template <bool use_int64_t, ...>
__global__ void store_kv_cache(...) {
  const auto warp_id = ... ; // 每个 warp 对应一个 token

  // 从 indices 中读取物理槽位 ID（支持 int32 和 int64）
  using IndexT = std::conditional_t<use_int64_t, int64_t, int32_t>;
  const auto pos = static_cast<const IndexT*>(indices)[warp_id];

  // K Cache：从物理槽位 pos 开始写
  const auto dst_k = pointer::offset(k_cache, pos * kv_cache_stride);
  const auto src_k = pointer::offset(k, warp_id * kv_input_stride);
  warp::copy<kElementSize>(dst_k, src_k);  // 32 线程协作写 K

  // V Cache：类似地写 V
  const auto dst_v = pointer::offset(v_cache, pos * kv_cache_stride);
  const auto src_v = pointer::offset(v, warp_id * kv_input_stride);
  warp::copy<kElementSize>(dst_v, src_v);  // 32 线程协作写 V
}
```

### 3.4 与 Paged Attention 的交互

```
Prefill / Decode 一步：

1. [调度器 Scheduler] 分配物理页，生成 out_loc 数组
   out_loc = [42, 7, 103, 19, ...]   # 每个新 token 的物理槽位

2. [Attention 计算] Flash Attention 输出当前 batch 的 K, V
   k.shape = [num_tokens, kv_heads/tp, head_dim]

3. [store_cache] 按照 out_loc 分散写入 KV Cache
   k_cache[42] = k[0]
   k_cache[7]  = k[1]
   ...

4. 下一步解码时，Paged Attention 直接通过槽位 ID 读取历史 KV
```

`kv_cache_stride` 和 `kv_input_stride` 允许处理非连续的张量布局，增加了灵活性。

---

## 4. fast_compare_key — RadixCache 前缀比较

### 4.1 功能与场景

`fast_compare_key()` 为 **RadixCache 的前缀树节点键值比较** 而设计。RadixCache 的核心操作是在前缀树中查找最长公共前缀，而每个节点的键（key）是一段 token ID 序列（CPU 上的 int32/int64 Tensor）。

该函数返回两个序列**从开头算起的最长公共元素数**：

```python
fast_compare_key(a, b) -> int
# a = [1, 2, 3, 4, 5]
# b = [1, 2, 3, 9, 5]
# 返回 3（前 3 个元素相同）
```

### 4.2 C++ 实现（`csrc/src/radix.cpp`）

```cpp
auto fast_compare_key(const TensorView a, const TensorView b) -> size_t {
    // 仅接受 1D CPU int32/int64 连续张量
    RuntimeCheck(_is_1d_cpu_int_tensor(a) && _is_1d_cpu_int_tensor(b), ...);
    RuntimeCheck(a.dtype() == b.dtype());

    const auto common_len = std::min(a.size(0), b.size(0));

    if (a.dtype().bits == 64) {
        const auto a_ptr = static_cast<const int64_t*>(a.data_ptr());
        const auto b_ptr = static_cast<const int64_t*>(b.data_ptr());
        // std::mismatch: 找到第一个不同的位置，O(n) 但 SIMD 向量化极快
        const auto diff = std::mismatch(a_ptr, a_ptr + common_len, b_ptr);
        return static_cast<size_t>(diff.first - a_ptr);  // 返回公共前缀长度
    } else {
        // int32 分支类似
        ...
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_compare_key, fast_compare_key);
```

**性能说明**：`std::mismatch` 在现代编译器（GCC/Clang with `-O3`）下会自动向量化为 SIMD 指令（SSE/AVX），比 Python 循环快 10-100x。由于 RadixCache 工作在 CPU 上（前缀树本身是 CPU 数据结构），这里不需要 GPU。

### 4.3 在 RadixCache 中的使用

在 `kvcache/radix_cache.py` 中的前缀查找路径：

```python
def match_prefix(self, key: list[int]) -> ...:
    key_tensor = torch.tensor(key, dtype=torch.int32)  # 转 Tensor（for fast_compare_key）
    node = self.root
    while node.children:
        child = ...
        # 使用 fast_compare_key 替代 Python 层逐元素比较
        match_len = fast_compare_key(child.key_tensor, key_tensor[offset:])
        if match_len > 0:
            offset += match_len
            node = child
```

---

## 5. PyNCCLCommunicator — 高性能 NCCL 通信器

### 5.1 为什么自定义 NCCL

PyTorch 的 `torch.distributed.all_reduce` 在调用时会将操作提交到一个**独立的 NCCL 通信流**，并与计算流进行同步。这引入了额外的流同步开销。

Mini-SGLang 的 `PyNCCLCommunicator` 直接将 NCCL 操作绑定到**引擎的主 CUDA 流**，避免了流切换，并支持与 Tensor Parallelism 计算的流水线重叠。

### 5.2 初始化流程（`pynccl.py`）

```python
def init_pynccl(*, tp_rank, tp_size, tp_cpu_group, max_size_bytes=0):
    # 限制内部对称内存缓冲区大小
    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()    # AOT 编译的 pynccl.cu

    # 第一步：Rank 0 生成 NCCL 唯一 ID（128 字节随机标识符）
    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]  # ncclGetUniqueId()
        torch.distributed.broadcast_object_list(
            id_list, src=0, group=tp_cpu_group  # 通过 Gloo（CPU）广播给所有 rank
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(id_list, src=0, group=tp_cpu_group)

    # 第二步：所有 rank 用相同的 ID 初始化 NCCL 通信器
    nccl_id = id_list[0]
    communicator = PyNCCLWrapper(tp_rank, tp_size, max_size_bytes, nccl_id)
    return communicator
```

**为什么用 Gloo 广播 NCCL ID**：
- NCCL 本身没有内置的 bootstrap 机制来分发 `ncclUniqueId`
- 初始化阶段利用 PyTorch Gloo 后端（CPU 通信），只需一次很小的广播（128 字节）
- 之后所有 GPU-GPU 通信完全走 NCCL，不再经过 Gloo

### 5.3 NCCLWrapper 实现（`csrc/src/pynccl.cu`）

```cpp
struct NCCLWrapper : public tvm::ffi::Object {
    NCCLWrapper(int rank, int world_size, size_t max_bytes, NCCLIDList uid) {
        // 初始化 NCCL 通信器
        ncclUniqueId id = get_uid(uid);
        ncclComm_t comm;
        ncclCommInitRank(&comm, world_size, id, rank);
        m_comm = {comm, ncclCommDestroy};  // RAII shared_ptr

        // 分配 NCCL 对称内存（NVLink 直接访问的共享缓冲区）
        void *buf;
        ncclMemAlloc(&buf, max_bytes);     // → NCCL 对称内存分配
        m_sym_mem = {buf, ncclMemFree};

        // 注册 NCCL Window（用于对称内存集合操作）
        ncclWindow_t win;
        ncclCommWindowRegister(comm, buf, max_bytes, &win, NCCL_WIN_COLL_SYMMETRIC);
        m_win = {win, [comm](ncclWindow_t w) { ncclCommWindowDeregister(comm, w); }};
    }

    void all_reduce(TensorView t, std::string op) const {
        const auto stream = LaunchKernel::resolve_device(t.device());  // 引擎主流

        if (size_bytes <= m_max_bytes) {
            // 小张量：复制到对称内存，用对称 AllReduce（NVLink 直接访问，延迟更低）
            cudaMemcpyAsync(buf_ptr, data_ptr, size_bytes, DeviceToDevice, stream);
            ncclAllReduce(buf_ptr, buf_ptr, count, dtype, op, comm, stream);
            cudaMemcpyAsync(data_ptr, buf_ptr, size_bytes, DeviceToDevice, stream);
        } else {
            // 大张量：原地 AllReduce（直接使用输入/输出张量的显存地址）
            ncclAllReduce(data_ptr, data_ptr, count, dtype, op, comm, stream);
        }
    }

    void all_gather(TensorView dst, TensorView src) const {
        // 直接 AllGather，不使用对称内存缓冲
        ncclAllGather(src_ptr, dst_ptr, count, dtype, comm, stream);
    }
};
```

### 5.4 关键设计：NCCL 对称内存（Symmetric Memory）

```
不使用对称内存（传统 AllReduce）：
  GPU 0 VRAM → [ncclAllReduce 需要 P2P 读写] → GPU 1 VRAM

使用对称内存（NCCL 2.27+）：
  所有 GPU 共同注册同一块虚拟地址空间（通过 NVLink fabric）
  NCCL 可以直接寻址对方的缓冲区，避免 GPU 内部的额外拷贝
```

`ncclMemAlloc` + `ncclCommWindowRegister` 启用后，小张量的 AllReduce 走对称内存路径，延迟更低。

### 5.5 在 Tensor Parallelism 中的使用

```
Transformer 层的 TP 推理流程（TP=2）：

GPU 0 计算 → 部分 hidden_states (half)
GPU 1 计算 → 部分 hidden_states (half)

↓ PyNCCLCommunicator.all_reduce(hidden_states, op="sum")

GPU 0 + GPU 1 → 各自持有完整 hidden_states（sum 后）
（全程在各 GPU 的引擎流上异步执行，无 CPU 介入）
```

---

## 6. Fused MoE Kernels — 融合混合专家矩阵计算

MoE（Mixture of Experts）模型（如 Mixtral-8x7B、Qwen-MoE）在 FFN 层使用多个专家网络，推理时动态路由到 top-k 个专家。Mini-SGLang 提供了两个 Triton 内核来加速这一过程。

### 6.1 数据流概述

```
输入：A [num_tokens, hidden_dim]   ← token 特征向量
     B [num_experts, hidden_dim, inter_dim]  ← 所有专家的权重矩阵

路由：topk_ids [num_tokens, top_k]  ← 每个 token 选中的专家 ID
    topk_weights [num_tokens, top_k] ← 对应的路由权重

第一步：fused_moe_kernel_triton(A, B, C, ...)
  C [num_tokens * top_k, inter_dim] ← 每个 token x 每个选中专家的输出

第二步：moe_sum_reduce_triton(C_3d, output, ...)
  C_3d [num_tokens, top_k, inter_dim] ← reshape
  output [num_tokens, inter_dim]      ← 加权求和后的最终输出
```

### 6.2 `fused_moe_kernel_triton` — 融合专家 GEMM（`moe_impl.py`）

```python
def fused_moe_kernel_triton(A, B, C, topk_weights, topk_ids,
                             sorted_token_ids, expert_ids,
                             num_tokens_post_padded,
                             mul_routed_weight, top_k, config, compute_type):
    # sorted_token_ids: 按专家排序后的 token ID（含 padding）
    # expert_ids: 每个瓦片对应哪个专家
    # num_tokens_post_padded: padding 后的总"处理量"

    K = B.shape[2]  # 输入 hidden_dim
    N = B.shape[1]  # 输出 inter_dim（每个专家的输出维度）
    M = sorted_token_ids.shape[0]  # = num_tokens * top_k（含 padding）

    even_Ks = (K % config["BLOCK_SIZE_K"] == 0)  # K 对齐时可消除边界 mask

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) *   # M 维度瓦片数
        triton.cdiv(N, config["BLOCK_SIZE_N"]),     # N 维度瓦片数
    )

    fused_moe_kernel[grid](
        A, B, C, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, M, len(topk_ids),
        A.stride(0), A.stride(1),             # 各维 stride（保证 coalesced）
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2),
        mul_routed_weight=mul_routed_weight,
        top_k=top_k, compute_type=compute_type,
        even_Ks=even_Ks,
        **config,
    )
```

### 6.3 `fused_moe_kernel` Triton 内核（`triton/fused_moe.py`）

```python
@triton.jit
def fused_moe_kernel(a_ptr, b_ptr, c_ptr, topk_weights_ptr,
                     sorted_token_ids_ptr, expert_ids_ptr, ...,
                     BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
                     MUL_ROUTED_WEIGHT, top_k, compute_type, even_Ks):
    # ─── L2 Cache 友好的分组排序映射 ───
    pid = tl.program_id(0)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # 这种分组方式让相邻 pid 复用 B 矩阵的缓存（L2 reuse）

    # ─── 读取当前 block 负责的 token IDs 和专家 ID ───
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)  # 实际 token 索引
    off_expert = tl.load(expert_ids_ptr + pid_m)               # 当前专家 ID

    # A 指针：按 token 索引和 top_k 解码出原始 token（offs_token // top_k）
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + ...)
    # B 指针：按专家 ID 选择对应权重矩阵
    b_ptrs = b_ptr + off_expert * stride_be + ...

    # ─── FP32 累加的 GEMM ───
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None])  # 无 K 边界 mask
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (k_mask))
            b = tl.load(b_ptrs, mask=k_mask)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # ─── 可选：乘以路由权重 ───
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask)
        accumulator *= moe_weight[:, None]

    # ─── 写回 C，转换到目标精度（bf16/fp16）───
    accumulator = accumulator.to(compute_type)
    tl.store(c_ptrs, accumulator, mask=c_mask)
```

**L2 复用优化**：通过 `GROUP_SIZE_M` 分组，让同一组内的瓦片共享对同一专家权重矩阵（B）的访问，提升 L2 缓存命中率。

### 6.4 `moe_sum_reduce_triton` — TopK 加权求和

```python
def moe_sum_reduce_triton(input, output):
    # input:  [num_tokens, top_k, hidden_dim]  ← 每个 token 每个专家的输出
    # output: [num_tokens, hidden_dim]         ← 加权求和结果

    token_num, topk_num, hidden_dim = input.shape
    BLOCK_M = 1        # 每个 block 处理 1 个 token（简化逻辑）
    BLOCK_DIM = 2048   # 每个 block 处理 2048 个 hidden 维度
    NUM_STAGE = 1      # 流水线深度
    num_warps = 8      # 256 线程

    grid = (triton.cdiv(token_num, BLOCK_M), triton.cdiv(hidden_dim, BLOCK_DIM))
    moe_sum_reduce_kernel[grid](input, *input.stride(), output, *output.stride(),
                                 token_num, topk_num, hidden_dim,
                                 BLOCK_M=BLOCK_M, BLOCK_DIM=BLOCK_DIM,
                                 NUM_STAGE=NUM_STAGE, num_warps=num_warps)
```

```python
@triton.jit
def moe_sum_reduce_kernel(input_ptr, ..., output_ptr, ...,
                           token_num, topk_num, hidden_dim,
                           BLOCK_M, BLOCK_DIM, NUM_STAGE):
    token_block_id = tl.program_id(0)  # 处理哪些 token
    dim_block_id   = tl.program_id(1)  # 处理哪些 hidden 维度

    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    for token_index in range(token_start, token_end):  # BLOCK_M=1 时只循环一次
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        input_t_ptr = input_ptr + token_index * stride_0 + offs_dim

        # 对 top_k 个专家的输出累加（不含权重，权重已在 fused_moe_kernel 中乘好）
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            tmp = tl.load(input_t_ptr + i * stride_1, mask=offs_dim < dim_end, other=0.0)
            accumulator += tmp

        tl.store(output_ptr + token_index * output_stride_0 + offs_dim,
                 accumulator.to(input_ptr.dtype.element_ty),
                 mask=offs_dim < dim_end)
```

---

## 7. Warp 级内存原语（`include/minisgl/warp.cuh`）

`warp::copy` 和 `warp::reset` 是整个 kernel 模块的**核心构建块**，被 `IndexKernel` 和 `StoreKernel` 广泛使用。

### 7.1 内存访问单元选择

```cpp
namespace details {
// 根据复制字节数自动选择最大的对齐访问单元：
// 16 字节 → uint4 (128-bit SIMD load/store)
// 8  字节 → uint2 (64-bit)
// 4  字节 → uint1 (32-bit)
inline constexpr auto resolve_unit_size(std::size_t x) -> std::size_t {
    if (x % (16 * kWarpThreads) == 0) return 16;  // 需被 16*32=512 整除
    if (x % (8  * kWarpThreads) == 0) return 8;   // 需被 8*32=256 整除
    if (x % (4  * kWarpThreads) == 0) return 4;   // 需被 4*32=128 整除
    return 0;
}
}
```

### 7.2 `warp::copy<kBytes>` — 协作复制

```cpp
template <std::size_t kBytes, std::size_t kUnit = details::resolve_unit_size(kBytes)>
__always_inline __device__ void copy(void* dst, const void* src) {
    using Package = mem_package_t<kBytes, kUnit>;  // uint4/uint2/uint1
    constexpr auto kBytesPerLoop = sizeof(Package) * kWarpThreads; // 每轮复制字节
    constexpr auto kLoopCount    = kBytes / kBytesPerLoop;         // 循环次数

    // kLoopCount 展开（静态展开，编译期已知）
    const auto lane_id = threadIdx.x % kWarpThreads;
    auto* dst_p = static_cast<Package*>(dst);
    auto* src_p = static_cast<const Package*>(src);

    #pragma unroll kLoopCount
    for (std::size_t i = 0; i < kLoopCount; ++i) {
        dst_p[i * kWarpThreads + lane_id] = src_p[i * kWarpThreads + lane_id];
    }
}
```

**举例**：`warp::copy<4096>(dst, src)` with `kUnit=16`：
- `kBytesPerLoop = 16 × 32 = 512`
- `kLoopCount = 4096 / 512 = 8`
- 每个线程每轮执行一次 128-bit load + 128-bit store
- 共 8 轮，32 线程，一次 warp::copy 处理 4096 字节 = 2048 个 bfloat16

```
warp::copy<4096> 时序（以 kUnit=16 为例）：

Round 0: Thread 0: dst[0]  =src[0];  Thread 1: dst[1]  =src[1];  ... Thread31: dst[31] =src[31]
Round 1: Thread 0: dst[32] =src[32]; Thread 1: dst[33] =src[33]; ... Thread31: dst[63] =src[63]
...
Round 7: Thread 0: dst[224]=src[224];                             ... Thread31: dst[255]=src[255]
（每个索引单位 = 16 字节 = 1 个 uint4）
```

所有访问完全 coalesced（连续 32 线程访问连续 16×32=512 字节），充分利用 GPU 显存带宽。

### 7.3 `warp::reset<kBytes>` — 协作清零

逻辑与 `copy` 完全对称，只是写入固定的零值 `Package{}`，用于 `masked_index_kernel` 中词表边界外 token 的零填充。

### 7.4 PDL（Programmatic Dependent Launch）支持

```cpp
namespace PDL {
template <bool kUsePDL> __always_inline __device__ void wait() {
    if constexpr (kUsePDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");  // H100+
    }
}
template <bool kUsePDL> __always_inline __device__ void launch() {
    if constexpr (kUsePDL) {
        asm volatile("griddepcontrol.launch_dependents;" :::);
    }
}
}
```

在 H100 及以上的 GPU 上，PDL 允许当前内核在完成一定量的工作后通知下游内核**立即启动**（而不是等待当前内核完全结束），实现内核级流水线。这对于 `index_kernel` → Attention → `store_kv_cache` 这样的顺序场景有潜在性能收益。目前默认 `use_pdl=False`，需显式启用。

---

## 8. 系统集成视图

### 8.1 各内核在推理流水线中的位置

```
一次 Decode 步骤的完整流水线：

┌─────────────────────────────────────────────┐
│  CPU Scheduler                              │
│  1. RadixCache.match_prefix()               │
│     └─ fast_compare_key(node.key, query_key)│  ← radix.cpp (CPU AOT)
│  2. 分配物理 KV 页面，生成 out_loc 列表      │
└──────────────────┬──────────────────────────┘
                   │ out_loc, input_ids 传入 GPU
┌──────────────────▼──────────────────────────┐
│  GPU Engine（引擎流 stream_0）               │
│                                             │
│  3. Token Embedding                         │
│     indexing(embedding_weight, input_ids,   │  ← index.cu (CUDA JIT)
│              hidden_states)                 │
│                                             │
│  4. Transformer Layers（循环 N 层）          │
│     4a. Attention                           │
│         [Flash Attention 计算 Q,K,V]        │
│                                             │
│     4b. store_cache(k_cache, v_cache,       │  ← store.cu (CUDA JIT)
│                     out_loc, k, v)          │
│                                             │
│     4c. AllReduce（TP > 1 时）              │
│         pynccl.all_reduce(attn_output)      │  ← pynccl.cu (CUDA AOT)
│                                             │
│     4d. MoE FFN（仅 MoE 模型）              │
│         fused_moe_kernel_triton(...)        │  ← triton/fused_moe.py
│         moe_sum_reduce_triton(...)          │  ← triton/fused_moe.py
│                                             │
│     4e. AllReduce（TP > 1 时）              │
│         pynccl.all_reduce(ffn_output)       │  ← pynccl.cu (CUDA AOT)
│                                             │
│  5. Sampling（lm_head + argmax / top-p）    │
└─────────────────────────────────────────────┘
```

### 8.2 编译时机总结

| 内核 | 编译时机 | 触发条件 | 缓存 |
|---|---|---|---|
| `IndexKernel` | 首次 `indexing()` 调用 | `element_size` 变化时 | `@functools.cache` |
| `StoreKernel` | 首次 `store_cache()` 调用 | `element_size` 变化时 | `@functools.cache` |
| `NCCLWrapper` | `pip install` 安装时 | 无（固定） | `.so` 文件 |
| `fast_compare_key` | `pip install` 安装时 | 无（固定） | `.so` 文件 |
| `test_tensor` | `pip install` 安装时 | 无（固定） | `.so` 文件 |
| `fused_moe_kernel` | 首次调用时 | Triton autotuning | Triton cache |
| `moe_sum_reduce` | 首次调用时 | 形状变化时 | Triton cache |

### 8.3 内存所有权

```
CPU 内存：
├── RadixCache 树节点（Python 对象 + torch.Tensor keys）
├── NCCL UID (NCCLIDList，初始化后丢弃)
└── KernelConfig / 编译缓存（functools.cache）

GPU 显存（HBM）：
├── KV Cache 物理页池（由 PagedMemoryPool 管理）
│   ├── k_cache [max_pages, kv_slot_size]
│   └── v_cache [max_pages, kv_slot_size]
├── Embedding 权重 [vocab_size/tp, hidden_dim]
├── MoE 专家权重 [num_experts, hidden_dim, inter_dim]
└── NCCL 对称内存缓冲（由 ncclMemAlloc 管理）
    └── 小张量 AllReduce 的中间缓冲（≤ max_size_bytes）
```

---

## 9. 总结

| 内核 | 文件位置 | 编译策略 | 核心技术 | 高性能关键 |
|---|---|---|---|---|
| `indexing` | `index.py` + `jit/index.cu` | CUDA JIT | Warp-level gather | `num_splits` 并行、coalesced 128-bit load |
| `store_cache` | `store.py` + `jit/store.cu` | CUDA JIT | Warp-level scatter | 一 warp/token、支持非连续 stride |
| `fast_compare_key` | `radix.py` + `src/radix.cpp` | AOT C++ | `std::mismatch` | SIMD 自动向量化（AVX2） |
| `test_tensor` | `tensor.py` + `src/tensor.cpp` | AOT C++ | 张量等值检查 | 仅测试用途 |
| `init_pynccl` | `pynccl.py` + `src/pynccl.cu` | AOT CUDA | NCCL + 对称内存 | 绑定引擎流、NVLink 对称内存 |
| `fused_moe_kernel_triton` | `moe_impl.py` + `triton/fused_moe.py` | Triton JIT | 分组 GEMM | L2 分组复用、`even_Ks` 展开 |
| `moe_sum_reduce_triton` | `moe_impl.py` + `triton/fused_moe.py` | Triton JIT | FP32 累加规约 | `BLOCK_DIM=2048`, `num_warps=8` |

**设计哲学总结**：

1. **最小化主机-设备同步**：所有 GPU 内核通过 TVM FFI 直接绑定到引擎 CUDA 流，避免隐式同步点。

2. **编译时特化 vs 运行时灵活性**：JIT 内核（index、store）通过 `element_size` 在运行时实例化模板，既保持了 C++ 模板的零成本抽象，又支持不同模型配置；AOT 内核（pynccl、radix）参数固定，彻底消除运行时编译开销。

3. **Warp 即处理单元**：GPU 内核设计以 warp（32 线程）为基本处理粒度，每个 warp 处理一个逻辑单位（一个 token 的 embedding、一个 KV 槽位的写入），实现了简洁的线程映射和高效的 coalesced 访存。

4. **PDL 与 Triton 流水线**：预留了 H100+ PDL 支持（用于内核级流水线）以及 Triton `num_stages` pipeline（用于软件流水线，隐藏 HBM 延迟）两种未来加速方向。
