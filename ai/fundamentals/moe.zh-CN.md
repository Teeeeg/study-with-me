---
title: 图解混合专家模型（MoE）
description: 从专家、路由和 Top-K 出发，理解负载均衡、专家容量、共享专家，以及总参数与激活参数的区别。
lang: zh
ref: moe
nav_order: 5
math: true
---

# 图解混合专家模型（MoE）

混合专家模型（Mixture of Experts，MoE）的核心目标是：**增加模型容量，但不让每个 token
都使用全部参数。**

普通稠密模型让每个 token 经过同一个前馈神经网络；MoE 则准备多个前馈网络作为“专家”，
再由路由器为每个 token 选择少数几个专家。模型可以拥有更多参数来学习不同模式，而单次前向
传播只激活其中一部分，因此 MoE 也被称为一种**条件计算**（conditional computation）。

> **先记住：** MoE 通常替换的是 Transformer 层中的 FFN，不是注意力层；路由也通常按
> token、按层独立发生，而不是为整段问题挑选一个完整的领域模型。

## 从一个 FFN 到多个专家

在标准 Transformer 块中，自注意力负责让 token 交换上下文信息，FFN 则逐 token 地变换
这些信息。FFN 往往占据模型参数的很大一部分，因此最适合扩展成多个分支。

MoE 层把原来的一个 FFN 替换成 $E$ 个结构相同、参数各自独立的 FFN：

![稠密 Transformer 的单个 FFN 被替换为多个专家 FFN](../../.asset/moe/dense-to-moe.png)

对于某个 token $x$，稠密层始终计算同一个函数：

$$
y = \operatorname{FFN}(x)
$$

MoE 层则从专家集合 $\{E_1, E_2, \ldots, E_E\}$ 中选择一小部分：

$$
y = \sum_{i \in \operatorname{TopK}(x)} g_i(x)E_i(x)
$$

其中 $g_i(x)$ 是路由器分配给专家 $i$ 的权重。若模型有 64 个专家而每个 token 只选择 2
个，那么专家参数很多，但每个 token 只计算其中 $2/64$ 的分支。

### “专家”究竟擅长什么

“专家”这个名称容易让人联想到数学、医学或编程专家，但实际情况通常更细粒度：

- 某些专家偏好标点、数字、专有名词或特定语言；
- 某些专家响应特定句法结构或上下文模式；
- 不同层的专家可能承担不同功能；
- 解码器模型中的专家未必具有清晰、稳定的人类可解释领域。

专家的分工不是人工指定的，而是训练过程中由数据、路由和优化共同形成的。同一个句子的
不同 token 会进入不同专家；同一个 token 在不同层也可能走不同路径。

## 路由器如何选择专家

路由器（router，也叫 gate）通常是一个很小的线性层。它接收 token 的隐藏状态 $x$，为
每个专家产生一个路由分数：

$$
s = xW_r
$$

对分数做 softmax，得到专家概率：

$$
p_i(x) = \frac{\exp(s_i)}{\sum_{j=1}^{E}\exp(s_j)}
$$

然后选择概率最高的 $K$ 个专家，只计算这些专家，并按保留下来的门控权重合并输出。

![路由器为专家计算概率，激活被选中的专家并加权输出](../../.asset/moe/router-selection.png)

完整的数据流可以概括为：

1. token 隐藏状态进入路由器；
2. 路由器计算每个专家的分数和概率；
3. Top-K 筛掉未选中的专家；
4. token 被分发给入选专家；
5. 专家各自执行 FFN；
6. 专家输出乘以门控权重后求和。

![一个 token 从路由打分到专家加权输出的完整流程](../../.asset/moe/routing-flow.png)

路由器虽然参数很少，却同时决定两件事：推理时计算哪些专家，以及训练时哪些专家收到梯度。
因此，路由质量会直接影响模型能力和系统效率。

## Top-1 与 Top-2 路由

**Token Choice** 是最常见的路由方式：每个 token 主动选择得分最高的专家。

Top-1 路由只激活一个专家，计算量和通信量最低：

![Top-1 路由为每个 token 选择一个专家](../../.asset/moe/top-1-routing.png)

Top-2 路由激活两个专家，再将两份输出按门控权重合并：

![Top-2 路由为每个 token 选择两个专家并合并输出](../../.asset/moe/top-2-routing.png)

两者的取舍是：

| 路由方式 | 优点 | 代价 |
| --- | --- | --- |
| Top-1 | 计算和跨设备通信更少，实现更简单 | 每个 token 的专家组合能力较弱，更依赖路由准确性 |
| Top-2 | 可以融合两个专家，通常更稳健 | 专家计算、KV 之外的激活传输和通信更多 |

Top-K 不是“取出概率后就结束”。实现中通常还会把入选专家的权重重新归一化，使它们的和为
1。训练时也可能给路由分数加入噪声，鼓励探索并减少早期固定选择。

## 最大难题：路由坍缩

如果某个专家在训练早期稍占优势，路由器会给它更多 token；它因此获得更多梯度、学得更快，
随后又吸引更多 token。这种正反馈会造成**路由坍缩**：少数专家过载，其余专家几乎没有
训练机会。

![路由不均衡时，大多数 token 会涌向同一个专家](../../.asset/moe/routing-imbalance.png)

后果不仅是“分工不好看”：

- 冷门专家训练不足，额外参数没有转化为有效容量；
- 热门专家成为计算与通信瓶颈，其他设备却处于空闲；
- 批次吞吐下降，甚至出现 token 被丢弃；
- 路由越来越偏，训练可能变得不稳定。

### 负载均衡辅助损失

主语言建模损失只关心预测是否正确，并不会主动要求专家均匀工作。因此训练通常增加一个较小
的辅助损失，惩罚过度集中的路由。

以 Switch Transformer 的形式为例，在一个包含 $T$ 个 token、$E$ 个专家的批次中：

$$
f_i = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}
\left[\operatorname*{argmax}_j p_j(x_t)=i\right]
$$

$f_i$ 是实际发给专家 $i$ 的 token 比例；$P_i$ 是路由器给该专家的平均概率：

$$
P_i = \frac{1}{T}\sum_{t=1}^{T}p_i(x_t)
$$

负载均衡项可以写成：

$$
L_{\text{balance}} = \alpha E\sum_{i=1}^{E}f_iP_i
$$

![辅助损失同时考虑路由概率和实际分发给各专家的 token 比例](../../.asset/moe/auxiliary-loss.png)

当 $f_i$ 和 $P_i$ 都接近 $1/E$ 时，专家使用较均匀。系数 $\alpha$ 必须适中：太小起不到
均衡作用，太大则会压过语言建模目标，强迫本应不同的 token 平均分配。

现代 MoE 还有无辅助损失的负载均衡、给专家分数添加动态偏置等方法。具体公式会变化，但目标
相同：**既允许专家形成分工，又不让少数专家垄断流量。**

## 专家容量与 Token 溢出

即使平均使用率接近，单个批次中的 token 仍可能集中到某个专家。为了让张量形状和设备负载
可控，系统会规定每个专家一次最多处理多少 token，称为**专家容量**（expert capacity）。

对于 Top-1 路由，一个常见定义是：

$$
C = \left\lceil
\frac{T}{E}\times \text{capacity factor}
\right\rceil
$$

Top-K 路由通常还需把 $K$ 计入分发量。容量因子大于 1，表示在理想平均负载之上预留余量。

![专家达到容量后，token 会尝试进入下一个候选专家](../../.asset/moe/expert-capacity.png)

当首选专家已满时，token 可以转给下一个候选专家。如果所有候选专家都满了，传统实现可能
让该 token 跳过 MoE 计算，仅沿残差路径进入下一层，这就是 **token overflow**。

![候选专家均已满时会发生 token overflow](../../.asset/moe/token-overflow.png)

容量因子体现了直接的系统取舍：

![容量因子越大，溢出越少，但未使用的容量越多](../../.asset/moe/capacity-factor.png)

- 容量太小：溢出较多，信息处理受损；
- 容量太大：预留槽位和显存被浪费，计算效率下降；
- 路由越均衡：相同容量下越不容易溢出。

不少现代实现通过更好的路由、动态形状或不丢 token 的调度减轻这个问题，但负载不均仍然是
分布式 MoE 的核心工程挑战。

## Switch Transformer 的简化

Switch Transformer 把 Transformer 中的 FFN 替换成 **Switch Layer**，并采用 Top-1
路由。每个 token 只进入一个专家，显著降低了相较 Top-2 路由的计算与通信复杂度。

![Switch Transformer 使用 Top-1 的稀疏专家层](../../.asset/moe/switch-transformer.png)

它的重要意义不只是“少选一个专家”，而是展示了扩大专家数量、保持每 token 计算量相对可控
的可行路径。与此同时，Top-1 更依赖负载均衡、容量设置和训练稳定性。

## 共享专家与路由专家

并非所有知识都需要专家分工。语法、通用事实或基础变换可能被大多数 token 使用。如果这些
公共模式重复存进每个路由专家，会浪费容量。

因此，一些现代架构把专家分为：

- **共享专家（shared experts）**：每个 token 都会经过，负责通用能力；
- **路由专家（routed experts）**：由路由器选择少数几个，负责更有差异的模式。

可把输出简化写成：

$$
y = \sum_{s=1}^{S}E_s^{\text{shared}}(x)
+ \sum_{i \in \operatorname{TopK}(x)}g_i(x)E_i^{\text{routed}}(x)
$$

共享专家提高了公共知识复用，也给所有 token 提供稳定路径；代价是它们始终参与计算。它与
负载均衡是两个不同问题：共享专家决定哪些能力不参与竞争，负载均衡决定路由专家之间如何
分配流量。

## 总参数不等于激活参数

阅读 MoE 模型规格时，至少要区分两个数字：

- **总参数（total parameters，也常被称为 sparse parameters）**：加载模型需要容纳的
  全部参数，包括所有专家；
- **激活参数（active parameters）**：处理一个 token 时实际参与前向计算的参数。

![MoE 加载全部专家，但每个 token 只激活其中一部分](../../.asset/moe/sparse-vs-active-parameters.png)

因此，“每次只激活少量参数”不等于“只需加载少量参数”。如果所有专家都驻留在 GPU 上，
模型权重仍按总参数消耗显存；MoE 主要节省的是每 token 的专家计算，而不是天然节省权重
存储。

实际速度也不会简单地按激活比例提升，因为路由会引入 token 排序、跨设备 all-to-all 通信、
负载不均和较小矩阵乘法等开销。MoE 的优势最容易在大批量、良好并行和高效专家内核下发挥。

### Mixtral 8x7B 的参数账

Mixtral 8x7B 每个 MoE 层有 8 个专家，每个 token 选择 2 个。名字中的“7B”不能直接理解为
每个完整专家都是一个独立的 7B 模型：注意力、嵌入和输出头由专家共享，只有 FFN 部分复制
为多个专家。

![Mixtral 8x7B 中共享组件、路由器和八个 FFN 专家](../../.asset/moe/mixtral-architecture.png)

该模型约有 46.7B 总参数，但处理一个 token 时约激活 12.9B 参数。激活量不只是
$2/8$ 的总参数，因为共享的注意力、嵌入和输出层始终参与计算。

![Mixtral 8x7B 的总参数与激活参数构成](../../.asset/moe/mixtral-parameter-count.png)

这组数字准确体现了 MoE 的价值与代价：它用接近 46.7B 参数的存储容量承载知识，但把单
token 的主要计算控制在约 12.9B 激活参数的量级。

## 训练与推理中的完整流程

一个 token 经过 MoE Transformer 层时，流程可以总结为：

1. 自注意力先聚合序列上下文；
2. 路由器根据当前 token 的隐藏状态计算专家分数；
3. Top-K 选择路由专家，并检查专家容量；
4. 系统按专家重排 token，必要时跨设备通信；
5. 各专家并行执行自己的 FFN；
6. 系统把结果送回 token 原位置，并按门控权重合并；
7. 训练时额外计算负载均衡损失，并更新被选中的专家。

这也解释了为什么 MoE 的难点不仅是模型结构，还包括分布式系统：专家通常分布在不同设备，
路由决策会直接转化为网络通信和设备负载。

## 要点回顾

- MoE 通常用多个专家 FFN 替换 Transformer 中的单个稠密 FFN；
- 路由器按 token、按层计算专家概率，Top-K 只激活少数专家；
- 专家更常专门处理细粒度 token 模式，而非完整的人类知识领域；
- 辅助损失和容量限制用于缓解路由坍缩与设备负载不均；
- 共享专家处理通用模式，路由专家提供条件化容量；
- 总参数决定权重存储，激活参数更接近每 token 的计算规模；
- MoE 以更多内存和通信换取更大的模型容量与相对可控的计算量。

## 参考资料

- [A Visual Guide to Mixture of Experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)
- [MoE 图解指南（中文翻译与补充）](https://blog.csdn.net/qq_36667170/article/details/148955499)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)