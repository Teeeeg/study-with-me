---
title: 大模型权重格式与数据类型
description: SafeTensors 与 GGUF、BF16 与 FP16，以及如何估算检查点体积。
lang: zh
ref: weights-and-types
nav_order: 1
math: true
---

# 大模型权重格式与数据类型

一个大模型的发行版有三个互相独立的属性：

- **文件格式**：张量和元数据如何存储，例如 SafeTensors 或 GGUF。
- **权重表示**：每个数值占多少比特，例如 BF16、FP16、FP8 或 INT4。
- **量化方法**：低比特权重如何产生、如何解读，例如 AWQ、GPTQ，或 GGUF 的
	`Q4_K_M` 编码。

因此一个模型可以是 **SafeTensors + BF16**、**SafeTensors + AWQ INT4**，也可以是
**GGUF + Q4_K_M**。这些名字描述的是不同层面，不能混为一谈。

## 主流选择

> **一句话结论：** 原始开放权重检查点的主流格式是 SafeTensors。现代加速器上稠密
> 权重的主流类型是 BF16。本地推理则常用 GGUF 搭配 `Q4_K_M` 这类 4 bit 编码。

| 目标 | 常见选择 | 原因 |
| --- | --- | --- |
| 发布、归档或微调模型 | **SafeTensors + 原始 dtype**，通常是 BF16 | 加载安全、框架支持广泛、可分片，且无需多余转换 |
| 在现代 GPU / TPU 上做稠密推理 | **SafeTensors + BF16** | 数值范围接近 FP32，存储只有 FP32 的一半；加速器原生支持 |
| 在较旧 GPU 上做稠密推理 | **SafeTensors + FP16** | 对旧硬件和旧算子的兼容性更好 |
| 本地 CPU、Apple Silicon 或 CPU/GPU 混合推理 | **GGUF + `Q4_K_M`** | 单文件部署、内存映射、后端可移植，体积与质量平衡好 |
| 显存受限的 GPU 推理 | **SafeTensors + AWQ/GPTQ INT4** | 仅权重压缩，且有优化过的 GPU 算子 |
| 新一代数据中心 GPU 上的高吞吐服务 | **FP8 检查点或编译后的引擎** | 显存带宽更低，且矩阵运算更快（需硬件支持） |

最稳妥的默认做法：除非目标运行时或硬件要求转换，否则保持模型作者提供的格式与
dtype。

## 文件格式

| 格式 | 现状 | 使用原因 | 主要限制 |
| --- | --- | --- | --- |
| **SafeTensors**（`.safetensors`） | 主流检查点与交换格式 | 存储张量时不含可执行的 pickle 代码；支持快速、惰性加载与分片检查点 | 配置与分词器文件通常仍要单独放在权重旁边 |
| **GGUF**（`.gguf`） | 主流本地推理格式 | 自描述、可内存映射、常为单文件，支持多种量化编码 | 面向推理，且绑定 GGML 系运行时 |
| **PyTorch 检查点**（`.bin`、`.pt`、`.pth`） | 分发上的遗留格式；训练状态仍在用 | 可序列化张量、优化器状态和 Python 对象 | 基于 pickle 的变体可移植性差，只应从可信来源加载 |
| **ONNX / TensorRT**（`.onnx`、`.engine`、`.plan`） | 派生的部署产物 | 为特定推理运行时存储或编译计算图 | 不是权威的源检查点；导出结果可能与硬件或运行时绑定 |

Hugging Face 模型仓库是一个**软件包**，而不是一种权重格式。它通常包含
SafeTensors 分片，加上 `config.json`、分词器文件、生成配置和分片索引。

## 权重数据类型与体积

对于 $P$ 十亿（billion）参数、每参数 $b$ 比特的模型，原始权重体积（十进制 GB）
约为：

$$
\text{体积 (GB)} \approx P \times \frac{b}{8}
$$

| 存储表示 | 每参数比特 | 8B 模型 | 70B 模型 | 主要用途 |
| --- | ---: | ---: | ---: | --- |
| **BF16** | 16 | 16 GB | 140 GB | 现代加速器上的主流稠密训练与推理 |
| **FP16** | 16 | 16 GB | 140 GB | 稠密推理，以及对旧 GPU 的兼容 |
| **FP8** | 8 加缩放因子 | 约 8 GB | 约 70 GB | 支持该类型的加速器上的数据中心训练或服务 |
| **INT8** | 8 加缩放因子 | 约 8 GB | 约 70 GB | 算子生态成熟的量化推理 |
| **4 bit 量化** | 通常等效 4-5 | 约 4-5 GB | 约 35-44 GB | 本地推理或显存受限的 GPU 服务 |
| **FP32** | 32 | 32 GB | 280 GB | 主权重、优化器状态和数值敏感的运算 |

以上都是**仅权重**的估算。量化文件还包含缩放因子、零点、码本，有时还有更高精度的
张量。运行时内存还要加上 KV 缓存、激活值和工作区。对于混合专家（MoE）模型，要用
总参数量，而不是每 token 激活的参数量。

## 为什么 BF16 是稠密模型的默认选择

BF16 和 FP16 都占 16 bit。FP16 把更多比特分给尾数，BF16 把更多比特分给指数：

![FP16 用更多尾数位换精度，BF16 用更多指数位换范围](../../.asset/fp16-vs-bf16.drawio.svg)

现代大模型通常更偏好 BF16，因为它：

- 数值范围与 FP32 大致相同，从而减少上溢和下溢；
- 与 FP16 一样每参数只占 2 字节；
- 通常无需 FP16 训练所必需的动态损失缩放（loss scaling）；
- 能在当前的 AI 加速器上直接运行，且通常配合 FP32 累加。

BF16 并不更精确：它只有 7 位尾数，而 FP16 有 10 位。在 FP16 较窄的范围内，它能表示
更接近的数值。仅就存储权重而言，应保持发布方给定的 dtype；把一个已验证的 FP16
模型转成 BF16 并不会自动让它变好。

## 参考资料

- [SafeTensors 格式](https://github.com/huggingface/safetensors)
- [Hugging Face 量化总览](https://huggingface.co/docs/transformers/en/quantization/overview)
- [GGUF 规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [NVIDIA 混合精度训练指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
