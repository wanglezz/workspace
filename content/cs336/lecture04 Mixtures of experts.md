---
title: CS336 lecture04 Mixtures of experts
tags:
  - cs336
  - 技术
author: tomato
share: "true"
date: 2025-11-28
dir: cs336
math: "true"
---
# 什么是MoE
MoE 架构就是在传统Dense Model 架构上，将单一 FFN 层替换成多个 FFN 层+门控，也就是分为多个专家加上选择器。{{< figure src="/images/Pasted image 20251128085651.png">}}这样的架构在保持单次FLOPS 不变的情况下增大了模型的总参数，被证明能大幅降低 loss 和提升训练速度。{{< figure src="/images/Pasted image 20251128091537.png">}}其中的 128e 代表 128 位 expert。可以看到随着专家的增加，模型的 loss 和 perplexity 显著优化。{{< figure src="/images/Pasted image 20251128091604.png">}}
##  MoE流行的原因
### 专家并行
可以将专家分到不同的设备上，在训练时将 token 路由到对应专家的设备上，实现系统上的并行。
## MoE还没有更普及的原因
- 基础设施非常复杂
- 路由设计复杂
# MoE设计
## Routing function 路由算法
路由算法可以大致分为以下几类：
- tokens choose expert
- expert choose tokens
- global assignment
实际上，目前几乎所有的 MoEs 都使用了**tokens choice topk**的路由算法，即对每一个 token 选取匹配最好的前 K 个专家。当然，这个划分不是平均的，会先根据 token 计算出每个专家的得分，根据得分计算每个专家对应权重，在输出时将经过每个专家的结果加权求和得到最终结果。(一个反直觉的现象，即使不采用路由算法，使用一个简单的哈希函数将 token 映射到一个专家上也会获取性能提升，通常将其作为 Baseline。)
### 其他路由算法
- RL（强化学习）：早期一些 MoE 工作尝试使用 RL 学习路由行为，但计算成本非常高，现在已经无人使用。
- Solve a matching problem：通过解决线性分配问题或最优传输类问题，同样成本远高于好处。
### Top-k routing details
#### 打分层
<div>$$
s_{i,t} = \text{Softmax}_i \left( \mathbf{u}_t^{l^T} \mathbf{e}_i^l \right)
$$</div>
##### 🔍 物理含义
计算当前 Token 与每个专家的“匹配度”或“亲密度”，并将其归一化为概率分布。
##### 📝 符号拆解
- **$\mathbf{u}_t^l$**：第 $l$ 层、第 $t$ 个 Token 的**输入向量**。
    - 上标 $T$ 表示**转置 (Transpose)**，用于进行向量点积运算。
- **$\mathbf{e}_i^l$**：第 $i$ 个专家的**路由嵌入向量 (Router/Centroid Embedding)**。
    - 这是路由网络（Router）中可学习的参数。你可以把它想象成该专家所擅长领域的“特征指纹”。
- **点积 ($\mathbf{u}^T \mathbf{e}$)**：衡量相似度。
    - 如果 Token 的向量方向与专家的特征向量方向一致，点积结果就大，说明该专家适合处理这个 Token。
#### 门控机制
<div>$$
g_{i,t} = \begin{cases} 
s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t} \mid 1 \le j \le N\}, K), \\ 
0, & \text{otherwise}, 
\end{cases}
$$</div>
##### 🔍 物理含义
这是 MoE 实现**稀疏性 (Sparsity)** 的关键步骤。它是一个分段函数，决定了谁“入选”，谁“淘汰”。
##### 📝 符号拆解
- **$g_{i,t}$**：最终用于计算的门控值。
- **$s_{i,t}$**：上一步计算出的**原始打分 (Score)**。
- **$\text{Topk}(\dots, K)$**：这是一个集合操作。
    - 它查看所有 $N$ 个专家的分数 $\{s_{1,t}, s_{2,t}, \dots, s_{N,t}\}$。
    - 选出数值最大的 $K$ 个分数（例如 $K=2$）。
- **逻辑判断**：
    - **Case 1 (入选)**：如果当前专家 $i$ 的分数属于前 $K$ 名，则 $g_{i,t}$ 继承其原始分数 $s_{i,t}$。
    - **Case 2 (落选)**：否则，$g_{i,t}$ 被强制置为 $0$。这意味着该专家在反向传播时不会收到梯度更新。
可以注意到这一步得到的$g_{i,t}$的和通常是小于1的，可以理解为是一个缩放系数，表示对专家选择的置信度。当然也有模型选择在之后重新归一化。
#### 最终输出层
<div>$$
\mathbf{h}_t^l = \sum_{i=1}^N \left( g_{i,t} \text{FFN}_i \left( \mathbf{u}_t^l \right) \right) + \mathbf{u}_t^l
$$</div>
##### 🔍 物理含义
这是 MoE 层最终的输出计算步骤。它描述了如何将被选中专家的计算结果进行加权组合，并通过残差连接融合原始输入信息。
##### 📝 符号拆解
- **$\mathbf{h}_t^l$**：第 $l$ 层、第 $t$ 个 Token 的**最终输出向量**（Hidden State）。这是这一层处理完后传递给下一层的数据。
- **$N$**：专家的总数量（例如 64 个或 8 个）。
- **$g_{i,t}$**：第 $i$ 个专家的**门控权重 (Gating Weight)**。
    - 如果是被选中的专家，这是一个非零的概率值。
    - 如果是未选中的专家，这个值为 $0$。
    - _注：因为未选中专家的权重为 0，所以虽然公式写的是 $1$ 到 $N$ 的求和，实际计算量只有 Top-K 个。_
- **$\text{FFN}_i(\cdot)$**：第 $i$ 个专家的**前馈神经网络**（Feed-Forward Network）。每个专家都有自己独立的权重参数。
- **$\mathbf{u}_t^l$**：**残差连接 (Residual Connection)**。
    - 直接把输入加到输出上，用于防止梯度消失，并允许模型保留原始信息，只学习“需要修改”的部分。
## Expert sizes
### Deepseek等采用的MoE变体
{{< figure src="/images/Pasted image 20251128202811.png">}}
下面这张图片很好的展示了增加专家细粒度对模型性能的影响{{< figure src="/images/Pasted image 20251128204434.png">}}一些消融实验{{< figure src="/images/Pasted image 20251128211638.png">}}展示了增加共享专家对性能没有提升而增加专家细粒度有明显提升。
## Training objectives
### 主要挑战
- 需要稀疏性来提升训练效率。
- 门控决策是离散的，不可微。
### Solutions
#### RL for MoEs
使用基于强化学习的方法优化门控决策，实际优化效果并不显著。
#### Stochastic perturbation(随机扰动)
经典实现
<div>$$
H(x) = \text{TopK}(\underbrace{x \cdot W_g}_{\text{部分1: 原始分数}} + \underbrace{\text{StandardNormal}() \cdot \text{Softplus}(x \cdot W_{noise})}_{\text{部分2: 自适应噪声}})
$$</div>
通过在原始分数的基础上增加一个噪声，将离散的门控决策变成一个连续的概率分布问题。让每个专家都有可能进入Top-K训练，避免未选中的专家梯度断裂。
#### Heuristic balancing losses
为了避免所有tokens最后都集中到一两个专家，引入了一个辅助loss，强迫 Router 把tokens尽量平均地分配给所有的专家，不要让某几个专家累死，也不要让其他专家闲死。
经典的 Switch Transformer 中的 Balancing Loss 公式大致如下：

<div>$$
Loss_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} (f_i \cdot P_i)
$$</div>
- **$f_i$ (Fraction)**: 实际上有多少比例 Token 被分给了专家 $i$。
- **$P_i$ (Probability)**: Router 给专家 $i$ 打出的平均 Softmax 概率（连续的置信度）。
很明显，$f_i$和$P_i$是正相关的，可以将最小化loss简化为最小化平方和问题。
假设有 2 个专家，总概率是 1.0。
- **情况 A：极度偏科（赢者通吃）**
    - 专家 1：0.9
    - 专家 2：0.1
    - 平方和 Loss $\approx 0.9^2 + 0.1^2 = 0.81 + 0.01 = \mathbf{0.82}$ （**很大**，惩罚重）
- **情况 B：完全平均（理想状态）**
    - 专家 1：0.5
    - 专家 2：0.5
    - 平方和 Loss $\approx 0.5^2 + 0.5^2 = 0.25 + 0.25 = \mathbf{0.50}$ （**最小**，惩罚轻）