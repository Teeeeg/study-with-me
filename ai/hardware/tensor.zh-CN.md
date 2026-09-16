---
title: CUDA Core 与 Tensor Core
description: 深入理解NVIDIA GPU中的CUDA Core和Tensor Core的架构、工作原理及其关系
lang: zh
ref: tensor
nav_order: 2
math: true
---

# CUDA Core 与 Tensor Core

在NVIDIA的GPU架构中，主要存在三种核心类型：**CUDA Core**、**Tensor Core** 以及 **RT Core**。CUDA Core 是通用计算核心，Tensor Core 是专为深度学习设计的矩阵运算加速单元，RT Core 则专注于光线追踪。本文重点介绍 CUDA Core 和 Tensor Core 的架构、工作原理及其关系。

## GPU 核心架构概览

![GPU core architecture showing CUDA Core, Tensor Core and their organization](../../.asset/gpu/gpu-core-architecture.svg)

## CUDA Core：通用并行计算核心

### 演进历史

- **Stream Processor (2006年前)**：早期的并行处理单元
- **CUDA Core (2010 Fermi+)**：引入统一架构，强调 CUDA 编程模型集成
  - 每个 SM 包含 32 个 CUDA Core（Fermi）
  - 每个 CUDA Core = 1 个 FPU（浮点单元）+ 1 个 ALU（整数单元）

### 工作方式：SM、Warp 与 SIMT

![SM, Warp and SIMT execution model](../../.asset/gpu/sm-warp-simt.svg)

CUDA Core 通过 **Warp** 组织并行执行：
- **1 Warp = 32 个线程**：这是GPU硬件调度的基本单位
- **SIMT（Single Instruction, Multiple Threads）模型**：
  - Warp 内的32个线程同步执行相同指令
  - 每个线程操作不同的数据
  - 所有线程在同一时钟周期执行同一条指令

**执行层次**：
```
Grid (整个 Kernel)
├── Thread Block (如 256 线程)
│   ├── Warp 0 (线程 0-31)
│   ├── Warp 1 (线程 32-63)
│   └── ... (共 8 个 Warp)
└── 映射到 SM 执行
```

**SM（Streaming Multiprocessor）**：
- 物理执行单元，包含多个 CUDA Core、Tensor Core、共享内存等
- 一个 SM 可以同时管理多个 Warp（如 32-64 个）
- Warp 调度器选择就绪的 Warp 执行，隐藏内存延迟

**性能考虑**：
- **Warp Divergence**：Warp 内线程走不同分支时，需串行执行所有路径
- **Occupancy**：SM 上活跃 Warp 越多，越能隐藏延迟
- **Block Size**：应为 32 的倍数（常用 128、256、512）

### 适用场景

- 通用并行计算、科学模拟
- 图形渲染、视频编解码
- 深度学习中的非矩阵运算（激活函数、normalization）

## Tensor Core：深度学习专用加速器

![Tensor Core matrix multiplication operation](../../.asset/gpu/tensor-core-operation.svg)

### 核心能力

Tensor Core 执行融合矩阵乘加（FMA）运算：

$$
D = A \times B + C
$$

- 单个 Tensor Core 每周期执行 **4×4×4** 矩阵乘法
- CUDA 通过 Warp 对外提供 **16×16×16** GEMM API
- 支持混合精度：FP16 输入，FP32 累加

### 演进历程

| 代次 | 架构 | 年份 | 关键特性 |
|------|------|------|---------|
| 第一代 | Volta | 2017 | FP16 输入，FP32 累加 |
| 第二代 | Turing | 2018 | 增加 INT8/INT4 支持 |
| 第三代 | Ampere | 2020 | TF32, BF16, 结构化稀疏性 |
| 第四代 | Hopper | 2022 | FP8, Transformer Engine |
| 第五代 | Blackwell | 2024 | FP4, 万亿参数优化 |

### 工作原理

**硬件层面**：
```
单个 Tensor Core: 4×4×4 矩阵乘法
├── 输入: A[4×4] (FP16), B[4×4] (FP16)
├── 累加: C[4×4] (FP32)
└── 输出: D[4×4] = A×B + C (FP32)
```

**编程层面**：
```
Warp 级别: 16×16×16 GEMM
├── 通过 WMMA API 调用
├── 一个 Warp (32线程) 协同执行
└── 自动分配到多个 Tensor Core
```

**大规模矩阵**：
```
2048×2048 矩阵
├── 分解为 Tiles (128×128)
├── Tiles 分解为 Fragments (16×16)
└── Fragments 映射到 Tensor Core (4×4×4)
```

## 混合精度训练

![Mixed precision training workflow with Tensor Core](../../.asset/gpu/mixed-precision-training.svg)

### 训练流程

1. **权重转换**：FP32 → FP16（前向），保留 FP32 副本（更新）
2. **前向传播**：FP16 计算，Tensor Core 加速
3. **Loss Scaling**：FP16 loss 放大（避免下溢）
4. **反向传播**：使用放大的 loss 计算 FP16 梯度
5. **Gradient Unscaling**：FP16 梯度 → FP32，反缩放
6. **参数更新**：使用 FP32 梯度更新 FP32 权重

### 为什么需要硬件支持

- Tensor Core 专门设计：FP16 计算 + FP32 累加
- 在保持精度的同时获得 8-20× 性能提升
- 降低内存带宽需求（FP16 vs FP32）

## CUDA Core 与 Tensor Core 的关系

![CUDA Core and Tensor Core collaboration in deep learning](../../.asset/gpu/cuda-tensor-collaboration.svg)

### 在 GPU 中的位置

```
GPU
├── SM (Streaming Multiprocessor) ×N
│   ├── CUDA Core ×M          # 通用计算
│   ├── Tensor Core ×K        # 矩阵乘法（Volta+）
│   ├── RT Core ×L            # 光线追踪（Turing+）
│   ├── Shared Memory
│   ├── Register File
│   └── L1 Cache
├── L2 Cache
└── HBM/GDDR Memory
```

### 功能对比

| 特性 | CUDA Core | Tensor Core |
|------|-----------|-------------|
| **引入时间** | 2006 | 2017 |
| **运算类型** | 标量 FMA | 矩阵乘法 (4×4×4) |
| **精度** | FP32, FP64, INT32 | FP16, TF32, BF16, FP8, FP4 |
| **性能** | 基准 | 8-30× (矩阵运算) |
| **适用** | 通用计算 | 深度学习矩阵运算 |

### 协同工作

在深度学习中的分工：

**Tensor Core 负责**：
- 全连接层矩阵乘法
- 卷积（通过 Im2Col 转为 GEMM）
- Attention 中的 Q×K^T、Attention×V

**CUDA Core 负责**：
- 激活函数（ReLU, GELU, Sigmoid）
- Normalization（LayerNorm, BatchNorm）
- Elementwise 操作
- 损失函数计算

**Transformer 前向传播示例**：
```
Input Embedding       → [CUDA Core]
Q/K/V = X × W        → [Tensor Core: GEMM]
Attention = Q × K^T  → [Tensor Core: GEMM]
Softmax              → [CUDA Core]
Output = Attn × V    → [Tensor Core: GEMM]
LayerNorm            → [CUDA Core]
FFN矩阵乘法          → [Tensor Core: GEMM]
ReLU                 → [CUDA Core]
```

## 卷积与 Tensor Core

### Im2Col：卷积转矩阵乘法

卷积通过 **Im2Col** 算法转换为矩阵乘法，从而使用 Tensor Core 加速：

```
原始卷积: [N, C_in, H, W] ⊗ [C_out, C_in, K, K]

Im2Col 转换:
1. 输入展开: [N×H'×W', C_in×K×K]
2. 卷积核重排: [C_out, C_in×K×K]
3. 矩阵乘法: [N×H'×W', C_out] = 输入展开 × 卷积核^T
4. Reshape 为输出

→ 可使用 Tensor Core 加速
```

## 编程接口

### CUDA Core 编程

标准 CUDA：
```cuda
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];  // CUDA Core
    }
}
```

### Tensor Core 编程

WMMA API：
```cuda
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::load_matrix_sync(a_frag, A, 16);
wmma::load_matrix_sync(b_frag, B, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // Tensor Core
wmma::store_matrix_sync(C, c_frag, 16);
```

### 深度学习框架自动优化

PyTorch 混合精度示例：
```python
model = MyModel().cuda()
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():  # 自动使用 Tensor Core
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 性能对比

### 训练加速比（相对纯 CUDA Core）

- **Volta (V100)**：8-12× 
- **Ampere (A100)**：15-20×
- **Hopper (H100)**：25-30×

### 性能提升来源

1. Tensor Core 对矩阵乘法加速（深度学习 70-90% 计算）
2. 混合精度降低内存带宽（2× 带宽节省）
3. 更高的算力利用率

## 总结

| 维度 | CUDA Core | Tensor Core |
|------|-----------|-------------|
| **本质** | 通用并行计算核心 | 专用矩阵乘法加速器 |
| **功能** | 标量运算 | 4×4×4 矩阵乘法 |
| **出现** | 所有 CUDA GPU (2006+) | Volta+ (2017+) |
| **适用** | 通用计算、非矩阵运算 | 深度学习矩阵运算 |
| **关系** | GPU 基础计算单元 | 与 CUDA Core 互补协同 |

### 使用建议

1. **深度学习训练**：启用混合精度（AMP），充分利用 Tensor Core
2. **推理部署**：使用 INT8/FP8 量化，利用 Tensor Core 低精度支持
3. **通用计算**：直接使用 CUDA Core 编程
4. **混合负载**：识别矩阵运算，让框架自动调度

## 参考资料

- [NVIDIA Tensor Core Architecture](https://www.nvidia.com/en-us/data-center/tensor-cores/)
- [CUDA C++ Programming Guide - WMMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [AI System: Tensor Core 基本原理](https://infrasys-ai.github.io/aisystem-docs/02Hardware04NVIDIA/01BasicTC.html)
