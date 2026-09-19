---
title: Attention Backend 原理简介
description: 区分 Attention 数学定义、Kernel 与 Backend，并比较 FlashAttention、FlashInfer、xFormers、cuDNN、FlexAttention 等主流实现的贡献与适用场景。
lang: zh
ref: attention-backend
nav_order: 5
math: true
---

# Attention Backend 原理简介

Attention 是 Transformer 中计算量最大、也最依赖硬件特性的算子之一。同一个数学公式，可以由完全不同的 GPU Kernel 执行；它们在长短序列、Prefill、Decode、KV Cache 布局、数据类型和硬件代际上的性能差异很大。

**Attention Backend 就是框架或推理引擎用来执行 Attention 的具体实现。** 它不改变模型“应该算什么”，主要决定“如何分块、如何搬运数据、如何并行，以及调用哪组 Kernel”。

---

## 1. 从数学公式到 Backend

标准 Scaled Dot-Product Attention 为：

$$
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}+A\right)V
$$

其中 $A$ 可以表达因果掩码、Padding Mask 或其他注意力约束。这个公式只描述结果，没有规定中间矩阵是否写回显存、线程如何分工、KV Cache 如何存放。

一次实际调用通常经过四层：

| 层次 | 回答的问题 | 示例 |
| --- | --- | --- |
| Attention 语义 | 要计算哪种注意力 | Causal、Cross Attention、Sliding Window、Block Sparse |
| 前端 API | 模型代码如何发起计算 | PyTorch SDPA、vLLM Attention Layer、TensorRT-LLM Plugin |
| Backend | 当前输入应该选哪套实现 | FlashAttention、FlashInfer、cuDNN、xFormers |
| Kernel | GPU 最终执行什么程序 | CUDA、Triton、CUTLASS、CK 或手写汇编生成的 Kernel |

因此，**Backend 不是一种新的 Attention 公式，也不一定只对应一个 Kernel**。一个 Backend 往往包含多组 Kernel，再根据 GPU、数据类型、序列长度、Head Dimension、Mask 和 Prefill／Decode 阶段进行选择。

---

## 2. 为什么需要多个 Backend

不存在对所有输入都最快的单一 Kernel，因为 Attention 的工作形态会变化：

| 场景 | Query 长度 | KV 长度 | 主要特点 |
| --- | ---: | ---: | --- |
| 训练／Prefill | 通常较长 | 通常较长 | 大块矩阵乘法，并行度高，适合 FlashAttention 一类融合 Kernel |
| 普通 Decode | 通常为 1 | 持续增长 | 计算量小但要读取大量 KV，常受显存带宽和调度开销限制 |
| Chunked Prefill | 中等或不规则 | 已缓存前缀加新 Token | 要同时处理 KV Cache、分块边界和不等长请求 |
| Speculative Decode | 一次多个 | 持续增长 | Query 不再固定为 1，需要高效处理短 Query 与长 KV |
| Sliding Window／稀疏注意力 | 不规则 | 只访问部分 KV | 通用稠密 Kernel 可能做无效工作，需要可编程 Mask 或专用 Kernel |

Backend 选择还受以下条件约束：

- **硬件**：NVIDIA Ampere、Hopper、Blackwell，AMD GPU 和 CPU 的最佳实现不同；
- **数据类型**：FP32、FP16、BF16、FP8 以及量化 KV Cache 支持范围不同；
- **Head 结构**：MHA、MQA、GQA 的 KV Head 数和数据复用方式不同；
- **内存布局**：连续 KV、Paged KV、Ragged Tensor 和 Prefix Sharing 需要不同寻址方式；
- **功能**：Dropout、任意 Mask、Attention Bias、Logits Soft Cap 和返回 Attention Score 并非都受支持。

框架通常先过滤掉不满足功能和硬件约束的实现，再用固定规则、启发式策略或自动调优选择 Backend。最快实现若不支持当前 Mask 或数据类型，就不能使用。

---

## 3. 主流 Attention Backend

下表同时列出独立 Kernel 库、框架内置 Backend 和推理引擎专用实现。它们所处层级略有不同，但都可能成为一次 Attention 调用最终采用的执行路径。

| Backend／项目 | 主要贡献 | 更适合的场景 | 关键限制与边界 |
| --- | --- | --- | --- |
| **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** | 提出 IO-aware 的精确 Attention：通过 Tiling 和 Online Softmax，不在 HBM 中物化完整 $N\times N$ 分数与概率矩阵。FlashAttention-2 改进线程块与 Warp 的工作划分；FlashAttention-3 针对 Hopper 引入异步流水线和更深入的硬件优化。 | 训练和长 Prompt Prefill；连续 Q/K/V 上的大块计算。 | 仍执行稠密注意力的二次计算，不会消除 KV Cache；支持范围取决于版本、GPU 架构、Head Dimension、Mask 和数据类型。 |
| **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** | 面向 LLM Serving 提供 Attention、Sampling、MoE 等 Kernel；重点优化 Paged KV Cache、Ragged Batch、Prefill、Decode 和 Speculative Decoding，并通过可组合接口适配推理引擎。 | 在线推理、动态批处理、长 KV Decode、Paged KV 和不等长请求。 | 是推理 Kernel 库，不是模型服务调度器；性能依赖批次形态、KV 布局和引擎集成。 |
| **[xFormers Memory-Efficient Attention](https://facebookresearch.github.io/xformers/components/ops.html#memory-efficient-attention)** | 较早提供可直接调用的 Memory-Efficient Attention，并通过 Operator Dispatch 在多种 CUDA 实现间选择；支持多类 Attention Bias，推动框架避免物化完整注意力矩阵。 | PyTorch 训练、研究模型及需要多种 Bias／Mask 的通用场景。 | “xFormers”是算子库，不是一种单独算法；某些路径内部可调用 CUTLASS 或 FlashAttention，实现与性能随环境变化。 |
| **[cuDNN Scaled Dot-Product Attention](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html)** | NVIDIA 在 cuDNN 中提供融合 SDPA，并利用底层硬件、图优化和启发式选择；统一支持训练前向／反向及部分推理场景，便于 PyTorch 等框架接入。 | NVIDIA GPU 上需要官方库支持、稳定集成和训练反向传播的场景。 | 闭源厂商库；具体功能与最快执行计划受 cuDNN、CUDA、GPU 型号和输入形状限制。 |
| **[PyTorch FlexAttention](https://pytorch.org/blog/flexattention/)** | 用 `score_mod` 与 `block_mask` 表达 Attention 变体，再由 `torch.compile` 生成融合 Triton Kernel；目标是在不手写 CUDA 的前提下兼顾可编程性与性能。 | Sliding Window、Document Mask、Prefix-LM、Soft Cap 等非标准 Attention，以及模型研究。 | 它是可编程 Attention API 与编译路径，不是固定的预编译 Kernel；编译开销、形状和规则能否块稀疏化会影响收益。 |
| **[Triton Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)** | 用接近 Python 的 GPU DSL 展示并实现融合 Attention，让开发者能控制 Tiling、数据搬运与自动调优；降低定制 Kernel 的门槛。 | 原型验证、研究新 Attention 变体、针对特定形状定制 Kernel。 | Triton 是 Kernel 编程与编译工具，不等于某个统一 Backend；生产表现取决于具体实现、调优配置和硬件支持。 |
| **[vLLM PagedAttention](https://docs.vllm.ai/en/latest/design/paged_attention/)** | 将操作系统分页思想用于 KV Cache：用固定大小的物理 Block 存储非连续 KV，通过 Block Table 寻址，减少外部碎片并支持共享前缀；配套 Kernel 直接消费分页 KV。 | 高并发 LLM Serving、Continuous Batching、Prefix Sharing 和可变长度 Decode。 | 核心贡献首先是 KV Cache 内存管理与对应 Kernel；vLLM 新版本还可调度 FlashAttention、FlashInfer 等 Backend，不能把 vLLM 的所有 Attention 都等同于最初的 PagedAttention Kernel。 |
| **[TensorRT-LLM Attention](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html)** | 将 Attention Kernel、Paged KV Cache、In-flight Batching、量化 KV、Context FMHA 与 Generation FMHA 集成到 NVIDIA 推理运行时，并按模型和硬件选择实现。 | NVIDIA GPU 上追求端到端部署性能、量化和多 GPU 推理的生产环境。 | 属于完整推理栈中的专用路径，和 TensorRT-LLM 的模型构建、插件及运行时紧密绑定，移植性低于独立算子库。 |
| **[ROCm Composable Kernel／AITER](https://github.com/ROCm/aiter)** | 为 AMD GPU 提供 Attention、Paged Attention、FlashAttention 等算子，并利用 Composable Kernel、Triton 或汇编级实现优化 ROCm 推理。 | AMD Instinct GPU 上的训练与 LLM Serving。 | 面向 AMD／ROCm 生态；支持矩阵、最佳 Kernel 和成熟度会随 GPU 世代及版本变化，不能直接套用 CUDA Backend 的结论。 |

还有 Intel oneDNN／IPEX、Apple Metal、Google TPU 和各云厂商编译器内的 Attention 实现。它们同样重要，但通常服务于特定硬件或编译栈；判断是否“主流”应结合实际部署平台，而不是只看项目热度。

---

## 4. FlashAttention、PagedAttention 与 FlashInfer 的区别

这三个名字经常一起出现，但解决的问题不同：

| 名称 | 首要问题 | 核心思路 |
| --- | --- | --- |
| FlashAttention | 单次 Attention 内部如何少搬数据 | 分块、融合与 Online Softmax，不物化完整 S/P |
| PagedAttention | 多请求的 KV Cache 如何高效分配和访问 | KV 分页、Block Table、按需分配与共享 |
| FlashInfer | Serving 中不同 Attention 形态如何高效执行 | 为 Prefill、Decode、Paged KV、Ragged Batch 等提供专用 Kernel 和调度接口 |

它们不是互斥关系。推理引擎可以用分页方式管理 KV Cache，在 Prefill 采用 FlashAttention 风格 Kernel，在 Decode 改用更适合短 Query 和 Paged KV 的 FlashInfer 或专用 Paged Attention Kernel。

---

## 5. PyTorch 中的 Backend 选择

PyTorch 的 `scaled_dot_product_attention` 是统一前端 API。根据设备和输入，它可能选择 Flash Attention、Memory-Efficient Attention、cuDNN Attention 或数学实现。概念上可以理解为：

```python
output = torch.nn.functional.scaled_dot_product_attention(
	query,
	key,
	value,
	attn_mask=mask,
	is_causal=True,
)
```

调用统一 API 不代表每次都运行同一个 Kernel。以下情况可能触发回退：

- 当前 GPU 或数据类型不受融合实现支持；
- Head Dimension、Stride 或 Tensor Layout 不满足约束；
- 同时传入的 Mask、Dropout、GQA 或其他功能不受该路径支持；
- 需要返回融合 Backend 不会保留的完整 Attention 权重；
- 确定性、数值精度或反向传播要求限制了候选实现。

因此，“代码里调用了 SDPA”只能说明使用统一接口，不能证明实际运行的是 FlashAttention。要确认执行路径，应查看框架日志、Profiler 中的 Kernel 名称，或强制不同 Backend 后分别验证。

---

## 6. 推理引擎如何选择 Backend

一个简化的选择流程如下：

1. **识别平台**：CUDA、ROCm、CPU 或其他加速器；
2. **识别阶段**：Prefill、Decode、Chunked Prefill 或 Speculative Decode；
3. **检查输入**：数据类型、Head Dimension、MHA／GQA／MQA、Mask、最大序列长度；
4. **检查 KV Cache**：连续或分页布局、Block Size、KV 数据类型、是否共享前缀；
5. **过滤不兼容实现**：功能正确性优先于理论峰值；
6. **按规则或基准选择**：在剩余实现中比较延迟、吞吐、显存和编译成本。

同一批请求内部也不一定永远只用一个 Backend。例如 Prefill 与 Decode 的形状差异极大，分别选择实现通常更合理。运行时升级 CUDA、PyTorch、vLLM 或驱动后，默认选择也可能变化，所以 Backend 名称应和版本一起记录。

---

## 7. 如何判断哪个 Backend 更快

不要只测一个固定形状，也不要只看平均 Token/s。至少要覆盖真实流量中的：

| 维度 | 建议记录的变量 |
| --- | --- |
| 模型结构 | Head 数、KV Head 数、Head Dimension、层数、Attention 类型 |
| 请求形态 | Batch Size、Prompt 长度、输出长度、长度分布 |
| 执行阶段 | Prefill、Decode、Chunked Prefill、Speculative Decode |
| KV Cache | 连续／分页、Block Size、数据类型、Prefix Cache 命中率 |
| 系统环境 | GPU 型号、驱动、CUDA／ROCm、框架和 Backend 版本 |
| 指标 | TTFT、TPOT／ITL、P50／P99 延迟、吞吐、显存峰值和功耗 |

验证顺序应该是：

1. **先验证正确性**：相同输入下比较输出误差、Mask、边界长度和 GQA 行为；
2. **预热并同步**：排除首次编译、缓存建立和异步计时误差；
3. **分开测 Prefill 与 Decode**：总吞吐无法解释瓶颈来自哪个阶段；
4. **覆盖长度分布**：短序列上的赢家不一定适合长上下文；
5. **再做端到端压力测试**：Kernel 更快不保证调度、通信和采样后的服务整体更快。

**Backend 的价值不是拥有最响亮的算法名，而是在当前硬件和工作负载下，以正确的功能语义完成更少的数据搬运、更高的并行利用率和更低的端到端延迟。**
