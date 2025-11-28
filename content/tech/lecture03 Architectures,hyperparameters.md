---
title: CS336
date: 2025-10-28
dir: tech
author: tomato
share: "true"
---

# 回顾标准transformer
## original transformer
{{< figure src="/images/2025 Lecture 3 - architecture_03.png">}}
# 模型架构和训练过程中差异
## Pre-vs-post norm
## Layer vs RMSnorm

## activations
### ReLU
传统激活函数$FF(x) = \max(0, xW_1) W_2$,简单的将negetive的输出变为 0
### \*GLU(gate linear unit)
门激活函数，以$GeGLU$为例$FFN_{GEGLU}(x, W, V, W_2) = (GELU(xW) \otimes xV)W_2$引入了新的权重 $V$作为门，对经过$GELU$的输出进行动态的筛选，实现更精细的流量控制。
不同激活函数效果的对比
{{< figure src="/images/2025 Lecture 3 - architecture_24.png">}}
## serial vs parallel layers
{{< figure src="/images/2025 Lecture 3 - architecture_27.png">}}
上图展示了串行layer 和并行 layer 输出公式的区别，并行方式下MLP 和 attention 的计算可以同时进行，在大规模模型上可以获得 15% 的性能提升且不会损失模型质量。
## position embeddings
### RoPE：rotary position embeddings
考虑到内积对任意（同步）旋转是是不变的，RoPE 在这个基础上做了一个变形，它不是将两个向量旋转相同的角度，而是将每个向量按照其绝对位置$i$进行旋转，这样只要两个向量相对位置相同，它们的夹角也一定相同。
{{< figure src="/images/2025 Lecture 3 - architecture_31.png">}}
左边是 we 和 know 的初始角度，右边是在不同语境下we和 know的相对角度，注意到中间的图 we 和 know 的绝对位置分别是 0 和 1，所以根据绝对位置旋转 0 和 1 个 position 后相对位置为 1，右图同理。可以发现，RoPE 抛弃了保持原始语义夹角，获得了一个更好反应相对位置的夹角。
RoPE 的实际应用
<div>$$
f_{\{q,k\}}(x_m, m) = \boldsymbol{R}_{\Theta, m}^d \boldsymbol{W}_{\{q,k\}} \boldsymbol{x}_m
$$</div>
这个公式展示了如何计算带有位置 $m$ 信息的 Query 向量或 Key 向量。
* $\boldsymbol{x}_m$：位于位置 $m$ 的原始词嵌入 (token embedding)。
* $\boldsymbol{W}_{\{q,k\}}$：将词嵌入 $\boldsymbol{x}_m$ 转换为 Query 向量 (如果用 $W_q$) 或 Key 向量 (如果用 $W_k$) 的标准权重矩阵。$\boldsymbol{W}_{\{q,k\}} \boldsymbol{x}_m$ 是**不带位置信息**的 $q$ 或 $k$ 向量。
* $\boldsymbol{R}_{\Theta, m}^d$：这就是 RoPE 的核心——**旋转矩阵**。它是一个 $d \times d$ 的矩阵（$d$ 是 $q/k$ 向量的维度），其数值取决于位置 $m$。
* $f_{\{q,k\}}(...)$：最终得到的、**包含了位置 $m$ 信息**的 $q$ 或 $k$ 向量。

**简单来说**：上述公式表示，要得到位置 $m$ 的 $q$ 或 $k$ 向量，就先计算出普通的 $q/k$ 向量 ($\boldsymbol{W} \boldsymbol{x}_m$)，然后再用特定于位置 $m$ 的旋转矩阵 $\boldsymbol{R}$ 去乘以它（即旋转它）。
旋转矩阵的定义，$m$是词的原始位置，矩阵由$d/2$个2x2的标准二维旋转矩阵组成，可以看出，RoPE 并不把 d 维向量当做一个整体看待，而是两两分组，然后对每组进行一个 2D 旋转。
<div>$$
\boldsymbol{R}_{\Theta, m}^d = 
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix}
$$</div>
## hyperparameters
### feedforward-model dim ratio
ffn 中间层的维度和模型主维度通常应该满足以下关系：
<div>$$
d_{ff} = 4 \times d_{model}
$$</div>
模型在流经过 FFN 层时，会经过升维->激活->降维的过程，如下(ReLU)
<div>$$
FFN(x) = \max(0, xW_1+b_1) W_2 + b_2
$$</div>
而在使用GLU的情况下通常是这样:
<div>$$
d_{ff} = \frac{8}{3}d_{model}
$$</div>
### head-dim * num-heads to model-dim ratio
通常 head-dim * num-heads == model-dim
### Aspect ratio
{{< figure src="/images/2025 Lecture 3 - architecture_44.png">}}
## dropout and regularization
{{< figure src="/images/2025 Lecture 3 - architecture_49.png">}}
上图展示了许多较新的、超大规模的模型（如 LLaMA, PaLM）在“预训练” (Pretraining) 阶段放弃了使用 Dropout，转而只依赖 Weight Decay (权重衰减)。
{{< figure src="/images/2025 Lecture 3 - architecture_50.png">}}
左图展示了一个反直觉的结果，即 weight decay 对过拟合影响不大。中间的 图展示了，当 Weight Decay 与衰减的学习率相互作用时，可以获得更低的训练损失。
## stability tricks
### Q-K norms
## attention heads
### GQA/MQA
{{< figure src="/images/2025 Lecture 3 - architecture_58.png">}}
上图展示了一次标准的注意力计算过程的操作次数和空间读取数。
我们推导一下该计算过程，首先定义计算中变量如下：
* `b`: **Batch Size** (批处理大小) 
- `n`: **Sequence Length** (序列长度，即 token 数量) 
- `d`: **Model Dimension** (模型的隐藏维度，例如 $d=512$) 
- `h`: **Number of Heads** (注意力头的数量，例如 $h=8$)
**Total Arithmetic Operations (总计算量)**
* 在 MHA 中，我们需要 4 个权重矩阵：$W_Q, W_K, W_V, W_O$。
* 这 4 个矩阵的维度**都是** `[d, d]`。 
* 我们的输入 $X$ 的维度是 `[b, n, d]`。 
* 我们以计算 $Q = X \cdot W_Q$ 为例： 
	* 这是一个 `[b, n, d]` 矩阵与 `[d, d]` 矩阵的批量乘法。 
	* 我们先看一个批次 (batch)：`[n, d]` $\times$ `[d, d]`。 
	* 根据矩阵乘法规则 (nd $\times$ dm = ndm)，这个计算需要 $n \times d \times d = nd^2$ 次算术运算。 
	* 因为有 `b` 个批次，所以总共是 $b \times nd^2 = bnd^2$ 次运算。 
	* 我们有 4 个这样的投影（$Q, K, V, O$），总计算量为 $4 \times (bnd^2)$。
* 在大O表示法中，我们忽略常数 4，所以这部分的计算复杂度为：$\mathcal{O}(bnd^2)$
**Total Memory Accesses (总内存访问量)**
总内存访问量有三个部分组成，我们这里只看 softmax 部分：
* $S = QK^T$ 产生的分数矩阵 $S$ 的维度是 `[b, h, n, n]`。 
* **写**$S$ 矩阵到内存 $\rightarrow$ 访问量 $\mathcal{O}(bhn^2)$。 
* **读** $S$ 矩阵以计算 Softmax $\rightarrow$ 访问量 $\mathcal{O}(bhn^2)$。 
* **写** $P = \text{softmax}(S)$ 矩阵 $\rightarrow$ 访问量 $\mathcal{O}(bhn^2)$。 
* **读** $P$ 矩阵以计算 $O = PV$ $\rightarrow$ 访问量 $\mathcal{O}(bhn^2)$。
所以这一部分的复杂度是$\mathcal{O}(bhn^2)$ : 访问“注意力矩阵”
{{< figure src="/images/2025 Lecture 3 - architecture_59.png">}}
上图展示了使用 kv-cache可以很大程度上减少推理阶段的计算。因为在推理阶段，每生成一个新 token 都需要前面所有已生成 token 的 KV 向量，而前面 token 的 KV 向量在后续推理过程中是固定的，所以用kv-cache 存储已经生成 token 的 KV 向量能够很大程度减少计算的复杂度。相应的，这种情况下 arthimetic intensity 会降低，因为内存访问量加大了。
#### MQA(Multi-Query Attention)
即让所有的查询头共享同一套 K 和 V。传统的 MHA 每一个查询头都有一对 K 和 V。MQA可以省去大量的内存，但是迫使全部头关注同样的信息，会造成模型表达能力下降，造成一定程度的性能损失。
#### GQA(Grouped-Query Attention)
即将 Q 头分组，组里的头共享 K 和 V。既保持了 MHA 的模型质量也获得了 MQA 的性能优化。
### Sparse/sliding window attention
由于 transformer 在长序列上太昂贵了。为了解决这个问题，**稀疏注意力**被提出了，它们通过只计算一部分“最重要”的注意力（局部的 + 某种长距离的）来**大幅降低计算成本**。
{{< figure src="/images/2025 Lecture 3 - architecture_66.png">}}
现在的标准做法