---
title: 从 Token 到注意力机制
description: 从分词开始，理解 Transformer、自注意力、QKV、多头注意力、因果掩码、RoPE 与稀疏注意力。
lang: zh
ref: attention
nav_order: 4
math: true
---

# 从 Token 到注意力机制

注意力机制解决的核心问题是：**处理序列中的某个位置时，模型应该从其他位置读取多少信息？**
Transformer 把这个过程实现为可并行的矩阵运算，使每个 token 都能根据当前上下文得到新的表示。

这篇笔记沿着一条完整的数据路径展开：文本先被切成 token，再转换为向量；向量进入
Transformer，通过自注意力交换上下文信息，最后用于生成下一个 token 或完成分类等任务。

## 从文本到 Token

语言模型不直接读取字符串。分词器（tokenizer）先调用 `encode`，把文本切成 token，并把
每个 token 映射到词表中的整数 ID。模型接收的是这些 ID，输出的也是词表中某个 token 的
ID 或所有候选 token 的概率分布。

![分词器把输入文本转换为 token ID，语言模型输出下一个 token ID](../../.asset/attention/tokenize_1.png)

token 不一定等于一个完整单词。常见粒度包括：

- **词级（word）**：直观，但词表很大，难以处理未登录词和词形变化；
- **子词级（subword）**：在词表大小和序列长度之间折中，是现代大模型的常见选择；
- **字符级（character）**：词表小，但序列明显变长；
- **字节级（byte）**：几乎可表示任意输入，但通常需要更多位置。

![同一段文本可以按词、子词、字符或字节切分](../../.asset/attention/tokenize_2.png)

因此，模型所说的“上下文长度”是 token 数，不是字数或单词数。不同分词器处理同一句话，
得到的 token 数也可能不同。

## Transformer 的位置

原始 Transformer 是编码器-解码器（encoder-decoder）架构。编码器读取完整输入序列，
通过自注意力为每个位置生成带上下文的表示；解码器一边读取编码器输出，一边根据已经生成的
内容预测下一个 token。

![原始 Transformer 由编码器和解码器组成](../../.asset/attention/attention_1.png)

后来常见的预训练模型通常只保留其中一侧：

| 架构 | 代表模型 | 注意力可见范围 | 典型用途 |
| --- | --- | --- | --- |
| 仅解码器 | GPT 系列 | 只能看当前位置及其之前的位置 | 文本生成、对话、代码补全 |
| 仅编码器 | BERT | 每个位置都能看完整输入 | 分类、检索、序列标注 |
| 编码器-解码器 | T5、原始 Transformer | 编码器双向；解码器因果；两者间有交叉注意力 | 翻译、摘要、条件生成 |

![仅解码器的生成模型与仅编码器的表征模型](../../.asset/attention/generative_models.png)

“编码器”和“解码器”描述的是信息流与训练目标，并不意味着只有解码器才能产生向量表示，
也不意味着编码器不能参与生成任务。

## 自注意力在做什么

词嵌入最初只表示 token 本身。自注意力（self-attention）让当前位置检查序列中的其他位置，
计算它们与当前位置的相关程度，再按相关程度汇总信息。于是同一个词在不同句子中可以得到
不同的上下文化表示。

![自注意力先计算相关性，再组合其他位置的信息](../../.asset/attention/self_attention.png)

例如，处理代词“它”时，注意力可能更多地读取前文中的名词；处理动词时，某些注意力头可能
关注主语或宾语。这些模式是训练中学到的，而不是人工写入的语法规则。

注意力矩阵可以把这种关系可视化：行代表当前正在更新的位置，列代表它可以读取的位置，颜色
越深表示权重越高。编码器的自注意力通常能看到整张矩阵。

![编码器中的每个位置都可以关注整个输入序列](../../.asset/attention/self_attention_detailed_4.png)

## Q、K、V：查询、键和值

每个输入向量 $x_i$ 会经过三组可学习的线性投影，得到 Query、Key 和 Value：

$$
q_i = x_i W_Q, \qquad k_i = x_i W_K, \qquad v_i = x_i W_V
$$

可以用“检索”来理解三者：

- **Query（查询）**：当前位置想找什么信息；
- **Key（键）**：每个位置提供什么匹配线索；
- **Value（值）**：匹配后真正被读取的内容。

当前位置的 Query 与所有可见位置的 Key 做点积。点积越大，表示两者越相关。除以
$\sqrt{d_k}$ 是为了避免维度增大后点积幅度过大，再用 softmax 把分数归一化为总和为 1
的权重：

$$
S = \frac{QK^\mathsf{T}}{\sqrt{d_k}}, \qquad
A = \operatorname{softmax}(S)
$$

![Query 与各位置的 Key 比较，得到相关性分数](../../.asset/attention/self_attention_detailed_1.png)

最后用注意力权重对 Value 加权求和：

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V
$$

![相关性权重乘以各位置的 Value，再求和形成上下文表示](../../.asset/attention/self_attention_detailed_2.png)

这里容易混淆的一点是：注意力输出不是“选中一个词”，而是所有可见 Value 的加权组合。
不同位置都可能贡献信息，只是贡献大小不同。

## 多头注意力

单个注意力模式很难同时表达多种关系。多头注意力（multi-head attention）把表示拆到多个
子空间，每个头拥有自己的 $W_Q$、$W_K$、$W_V$，独立计算注意力。各头结果拼接后，再经过
输出投影 $W_O$：

$$
\operatorname{head}_h = \operatorname{Attention}(Q_h,K_h,V_h)
$$

$$
\operatorname{MHA}(X)
= \operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W_O
$$

![多个注意力头独立读取上下文，再合并各头的信息](../../.asset/attention/self_attention_detailed_3.png)

多个头可以学习不同类型的关联，例如局部搭配、长距离依赖、位置关系或指代关系。头的作用
通常不是预先指定的，而且并非每个头都能对应一个清晰的人类概念。

### 分组查询注意力

标准多头注意力中，每个 Query 头都有自己的 Key 和 Value 头。分组查询注意力（Grouped
Query Attention，GQA）让一组 Query 头共享同一组 Key 和 Value：

![GQA 中多个 Query 头共享一组 Key 和 Value](../../.asset/attention/group_head.png)

设 Query 头数为 $H_q$，Key/Value 头数为 $H_{kv}$：

- $H_{kv}=H_q$ 时是标准多头注意力（MHA）；
- $H_{kv}=1$ 时是多查询注意力（MQA）；
- $1 < H_{kv} < H_q$ 时是分组查询注意力（GQA）。

自回归推理需要缓存历史 token 的 Key 和 Value。减少 $H_{kv}$ 能显著缩小 KV 缓存与内存
带宽开销，同时通常比完全共享的 MQA 保留更多表达能力。

## 因果掩码

生成模型训练时可以并行处理整段文本，但预测第 $i$ 个位置时不能偷看未来 token。因果掩码
（causal mask）把未来位置的注意力分数设为 $-\infty$，使它们经过 softmax 后权重为 0：

$$
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

$$
\operatorname{CausalAttention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}+M\right)V
$$

![因果掩码使解码器只能关注当前位置和已经出现的位置](../../.asset/attention/masked_attention.png)

注意力矩阵因此呈下三角形。训练阶段的并行计算不会破坏自回归约束；推理阶段则逐 token
生成，并复用 KV 缓存避免重复计算全部历史位置。

## 位置信息与 RoPE

只看 $QK^\mathsf{T}$ 时，注意力本身无法区分 token 的先后顺序。因此 Transformer 必须注入
位置信息。方法包括固定正弦位置编码、可学习绝对位置嵌入、相对位置偏置，以及现代大模型
常用的旋转位置编码（Rotary Positional Embedding，RoPE）。

### 给每个 Token 一组时钟指针

可以把 RoPE 想象成给每个 token 的 Query 和 Key 装上一组“时钟指针”。第 0 个位置的
指针指向初始方向；位置每向后移动一步，指针就转动一定角度。两个 token 做注意力计算时，
模型不仅比较它们的内容，还能从指针的夹角中感知它们相隔多远。

实际上，RoPE 会把向量的维度两两分组，例如 $(x_1,x_2)$、$(x_3,x_4)$，把每一组看成
平面上的一根箭头，再按照 token 的位置旋转。不同维度组的旋转速度不同：转得快的像秒针，
适合区分较短的距离；转得慢的像分针和时针，可以覆盖更长的距离。多组不同速度的“指针”
合在一起，就能表达多种距离尺度。

位置为 $i$ 的 Query 旋转 $i$ 对应的角度，位置为 $j$ 的 Key 旋转 $j$ 对应的角度：

$$
q_i' = R_i q_i, \qquad k_j' = R_j k_j
$$

计算点积时，两者共同经历的旋转会抵消，结果取决于旋转角度之差，也就是相对位置 $j-i$：

$$
{q_i'}^\mathsf{T}k_j'
= q_i^\mathsf{T}R_{j-i}k_j
$$

因此，第 5 个与第 7 个 token、第 100 个与第 102 个 token 虽然绝对位置不同，但都相隔
2 个位置，RoPE 能让它们呈现相似的位置关系。它并不是把位置编号直接塞进词向量，而是让
Query 和 Key 之间的夹角携带相对距离。

![RoPE 应用于 Query 和 Key，使相关性计算包含位置信息](../../.asset/attention/rotary_embedding.png)

RoPE 通常只作用于 Query 和 Key，因为两者负责决定“去哪里找”；Value 负责“找到后取回
什么”，一般不需要旋转。简而言之：**RoPE 用向量旋转表示位置，用旋转角度之差表示相对
距离。**

## 从全注意力到稀疏注意力

长度为 $n$ 的序列会产生 $n\times n$ 个注意力分数，因此标准全注意力的计算量和注意力
矩阵内存通常随序列长度按 $O(n^2)$ 增长。上下文变长时，这会成为主要瓶颈。

一种直接的办法是局部注意力：每个 token 只读取附近固定窗口内的位置，将复杂度降到约
$O(nw)$，其中 $w$ 是窗口大小。

![全局自回归注意力与局部窗口自回归注意力](../../.asset/attention/sparse_attention.png)

更一般的稀疏注意力会设计结构化连接模式，例如：

- **滑动窗口**：关注附近 token，适合局部依赖；
- **跨步连接**：按固定间隔关注远处位置；
- **固定或分块连接**：在局部块之外保留少量跨块通道；
- **全局 token**：让少数特殊位置连接整个序列。

![全注意力、跨步稀疏注意力与固定模式稀疏注意力](../../.asset/attention/sparse_attention_1.png)

稀疏化减少计算和显存，但也可能切断重要的长距离依赖。实际模型往往组合局部窗口、少量
全局连接与跨层信息传递，在效率和信息覆盖范围之间折中。

## 完整数据流

把以上概念串起来，一层自注意力的主要过程是：

1. 分词器把文本转换为 token ID；
2. 嵌入层把 token ID 映射成向量，并注入位置信息；
3. 线性投影产生 Query、Key、Value；
4. Query 与 Key 计算相关性，并应用缩放与可选的因果/稀疏掩码；
5. softmax 得到注意力权重，对 Value 加权求和；
6. 多个头的输出拼接并投影，形成带上下文的信息；
7. 输出继续经过残差连接、归一化和前馈网络，再送入下一层。

需要注意，自注意力只是 Transformer 层的一部分。完整层还包含前馈网络、残差连接和归一化；
模型通过堆叠许多层，让信息逐层组合成更复杂的表示。

## 要点回顾

- token 是模型读写的离散单位，粒度由分词器决定；
- 自注意力用 Query 查找 Key，并按相关性汇总 Value；
- 多头注意力让模型在不同表示子空间中并行学习关系；
- GQA 通过共享 Key/Value 头降低 KV 缓存成本；
- 仅解码器模型使用因果掩码，保证预测时看不到未来；
- RoPE 在 Query 和 Key 中编码相对位置信息；
- 全注意力是 $O(n^2)$，局部或结构化稀疏注意力可降低长上下文成本。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
