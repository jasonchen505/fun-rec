# LLM & Agent & 生成式推荐（GR）面试深度准备指南

> 基于 fun-rec 项目（深度推荐算法实践）系统整理，面向正在寻找 LLM 算法实习的 MS 在读学生。

---

## 目录

1. [项目框架速览](#1-项目框架速览)
2. [推荐系统基础：从判别式到生成式](#2-推荐系统基础从判别式到生成式)
3. [LLM 三阶段训练范式深度解析](#3-llm-三阶段训练范式深度解析)
4. [生成式推荐核心技术](#4-生成式推荐核心技术)
5. [语义 ID 与 Tokenization](#5-语义-id-与-tokenization)
6. [端到端生成式推荐架构](#6-端到端生成式推荐架构)
7. [会思考的推荐模型：OneRec-Think](#7-会思考的推荐模型onerec-think)
8. [强化学习在推荐中的应用](#8-强化学习在推荐中的应用)
9. [Scaling Law 与工程实践](#9-scaling-law-与工程实践)
10. [高频面试问题与深度回答](#10-高频面试问题与深度回答)
11. [代码实现要点](#11-代码实现要点)
12. [面试者自我介绍模板](#12-面试者自我介绍模板)

---

## 1. 项目框架速览

### 1.1 项目结构

```
fun-rec/
├── src/funrec/           # 核心代码库
│   ├── models/           # 41个推荐模型实现
│   │   ├── hstu.py       # HSTU (生成式推荐核心架构)
│   │   ├── sasrec.py     # 序列推荐基准模型
│   │   ├── dssm.py       # 双塔召回模型
│   │   ├── din.py        # 注意力机制排序模型
│   │   └── ...           # FM, DeepFM, DCN, DIEN等
│   ├── config/           # 模型配置
│   ├── training/         # 训练器与损失函数
│   ├── evaluation/       # 评估模块
│   └── experiment.py     # 实验流程封装
├── docs/                 # 完整技术文档
│   ├── chapter_5_gr_basic/    # 生成式推荐基础
│   ├── chapter_6_scaling/     # Scaling Law架构
│   ├── chapter_7_gr_e2e/      # 端到端生成式建模
│   ├── chapter_8_thinking/    # 会思考的推荐模型
│   └── chapter_9_diffusion/   # 扩散模型推荐
└── web_project/          # 生产级推荐系统实战
```

### 1.2 核心设计模式

```python
# fun-rec 的统一实验接口
import funrec
metrics = funrec.run_experiment('wide_deep', return_metrics=True)

# 或者多模型对比
results, table = funrec.compare_models(
    ['wide_deep', 'deepfm', 'dssm'],
    return_results=True
)
```

**面试考点**：理解推荐系统的模块化设计（数据加载→特征工程→模型训练→评估）

---

## 2. 推荐系统基础：从判别式到生成式

### 2.1 判别式推荐的核心范式

**核心公式**：学习条件概率 $p(y=1|u, i, c)$，预测用户 $u$ 在上下文 $c$ 下对物品 $i$ 的兴趣。

**"Embedding & MLP" 范式**：
1. 将用户ID、物品ID及各类特征通过 Embedding 层映射为稠密向量
2. 通过 MLP 或特征交互模块处理
3. 输出标量分数

**关键模型演进**（项目中已实现）：

| 模型 | 核心思想 | 代码位置 |
|------|----------|----------|
| FM | 二阶特征交叉，$0.5 \cdot ((\sum v)^2 - \sum v^2)$ | `src/funrec/models/fm.py` |
| DeepFM | FM + DNN 并行结构 | `src/funrec/models/deepfm.py` |
| DCN | 显式高阶交叉，$x_{l+1} = x_0 \cdot x_l^T \cdot w_l + b_l + x_l$ | `src/funrec/models/dcn.py` |
| DIN | 注意力机制建模用户兴趣演化 | `src/funrec/models/din.py` |
| SASRec | 自注意力序列推荐 | `src/funrec/models/sasrec.py` |
| HSTU | 生成式推荐基础架构 | `src/funrec/models/hstu.py` |

### 2.2 判别式范式的三大局限

1. **参数效率问题**：Embedding 层占 90%+ 参数，稀疏低效
2. **语义建模缺失**：物品 ID 无语义关联，冷启动困难
3. **多阶段级联困境**：召回→粗排→精排→重排，目标不一致、误差累积

### 2.3 生成式推荐的核心思想

**范式转换**：将推荐重新定义为**序列生成任务**

$$p_\theta(i_{1:T} | u, c) = \prod_{t=1}^{T} p_\theta(i_t | i_{<t}, u, c)$$

**核心优势**：
- 自回归建模捕捉序列依赖
- 端到端优化消除级联误差
- 通过语义 ID 实现物品的语义表示
- 统一架构简化系统设计

**面试深挖点**：为什么生成式推荐能解决判别式范式的局限？关键在于它将"对候选集打分"转变为"从全量物品空间直接生成"。

---

## 3. LLM 三阶段训练范式深度解析

### 3.1 预训练（Pre-training）

**训练目标**：因果语言建模（CLM）

$$\mathcal{L}_{\text{CLM}} = -\sum_{i=1}^{n} \log p_\theta(x_i | x_{<i})$$

**关键细节**：
- **架构选择**：Decoder-Only 为主流（GPT、LLaMA）
- **规模效应**：模型性能随参数量、数据量、计算量幂律提升
- **涌现能力**：零样本学习、少样本学习在规模达到阈值后出现

**面试考点**：解释 Scaling Law 的数学形式 $\hat{L}(N) = E + \frac{A}{N^\alpha}$ 及其在推荐系统中的验证

### 3.2 指令微调（SFT）

**训练数据格式**：指令-输入-输出三元组

```
指令：总结下面这段文字的主要内容。
输入：[一段关于人工智能发展历史的文字]
输出：[人工智能从1950年代诞生至今，经历了...]
```

**损失函数**：只对输出部分计算

$$\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{m} \log p_\theta(y_i | y_{<i}, \boldsymbol{c})$$

**关键理解**：
- SFT 让模型学会"理解指令"这一元能力
- 数据量通常数万到数十万，远小于预训练
- 支持全参数微调或参数高效微调（LoRA）

### 3.3 偏好对齐（RLHF/DPO）

**RLHF 三步骤**：

1. **收集偏好数据**：$(c, y_w, y_l)$，$y_w$ 为更优输出
2. **训练奖励模型**：

$$\mathcal{L}_{\text{RM}} = -\mathbb{E} \left[ \log \sigma(r_\phi(c, y_w) - r_\phi(c, y_l)) \right]$$

3. **策略优化（PPO）**：

$$\mathcal{L}_{\text{RL}} = \mathbb{E} [r_\phi(c, y)] - \beta \cdot D_{\text{KL}}(p_\theta \| p_{\text{ref}})$$

**DPO 简化方案**：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{p_\theta(y_w|c)}{p_{\text{ref}}(y_w|c)} - \beta \log \frac{p_\theta(y_l|c)}{p_{\text{ref}}(y_l|c)} \right) \right]$$

**面试深挖点**：
- RLHF 中 KL 散度约束的作用是什么？（防止 reward hacking，保持生成质量稳定）
- DPO 为什么能替代 RLHF？（数学推导证明奖励模型可隐式表示）

---

## 4. 生成式推荐核心技术

### 4.1 从 LLM 到生成式推荐的映射

| LLM 阶段 | 推荐系统适配 | 核心挑战 |
|----------|-------------|----------|
| 预训练 | 用户行为序列预训练、多模态内容预训练 | 如何平衡语言能力和推荐能力？ |
| 指令微调 | 推荐任务指令化、多任务联合训练 | 如何处理 ID 化物品？ |
| 偏好对齐 | 隐式反馈对齐、业务指标优化 | 如何从隐式反馈构造偏好数据？ |

### 4.2 推荐场景的特殊挑战

1. **物品 Tokenization**：ID 无语义，需设计语义 ID
2. **协同信号融合**："买了 A 的人也买 B"无法从内容描述直接获得
3. **冷启动问题**：新物品/新用户缺乏交互历史
4. **实时性要求**：数十毫秒内完成推理 vs 自回归逐 token 生成

---

## 5. 语义 ID 与 Tokenization

### 5.1 核心问题

传统推荐系统的物品用原子 ID 表示（如 `item_12345`），对 LLM 来说是 OOV 符号。

### 5.2 RQ-VAE 层次化量化

**核心思想**：将物品的连续表示离散化为层次化索引序列

**量化过程**：

$$c_i = \arg\min_k \|\boldsymbol{r}_i - \boldsymbol{v}^i_k\|^2, \quad \boldsymbol{r}_{i+1} = \boldsymbol{r}_i - \boldsymbol{v}^i_{c_i}$$

其中 $c_i$ 为第 $i$ 层选择的码本索引，$\boldsymbol{r}_i$ 为残差向量。

**关键特性**：
- **语义逐层细化**：从粗粒度类别到精细个体特征
- **前缀共享**：内容相似物品共享更多前缀索引
- **索引示例**：`<a_5><b_2><c_6><d_7>`

### 5.3 RQ-Kmeans 工业级方案

OneRec 采用的方案（相比 RQ-VAE 更简单高效）：

$$\mathcal{R}^{(1)} = \{\tilde{\boldsymbol{M}}_i\}, \quad \mathcal{C}^{(l)} = \text{K-means}(\mathcal{R}^{(l)}, N_t)$$

**面试考点**：RQ-VAE vs RQ-Kmeans 的区别？RQ-VAE 端到端训练码本，RQ-Kmeans 直接在残差上聚类。

### 5.4 LC-Rec 的均匀语义映射

解决索引冲突问题，建模为最优传输问题：

$$\min \sum_{\boldsymbol{r}_H \in \mathcal{B}} \sum_{k=1}^K q(c_H=k|\boldsymbol{r}_H) \|\boldsymbol{r}_H - \boldsymbol{v}^H_k\|^2$$

通过 Sinkhorn-Knopp 算法求解，确保物品在码本向量上的分配尽可能均衡。

### 5.5 PLUM 工业级方案（YouTube）

**多模态嵌入拼接**：

$$\boldsymbol{e}_{\text{final}} = [\boldsymbol{e}_{\text{text}} \oplus \boldsymbol{e}_{\text{visual}} \oplus \boldsymbol{e}_{\text{audio}} \oplus \boldsymbol{e}_{\text{cf}}]$$

**关键创新**：
- 引入协同过滤嵌入 $\boldsymbol{e}_{\text{cf}}$ 编码"谁看了这个还看了什么"
- 多分辨率码本：不同层使用不同大小（128→256→512→1024）
- 渐进式掩码训练：强制每层索引携带增量语义信息

---

## 6. 端到端生成式推荐架构

### 6.1 OneRec-V1 架构

**Encoder-Decoder 设计**：

**编码器四通路**：
1. **用户静态特征通路**：处理用户画像
2. **短期行为通路**：最近 20 次交互
3. **正反馈行为通路**：256 次高参与度交互
4. **超长期历史通路**：通过 QFormer 压缩 10 万条历史

**解码器设计**：
$$\boldsymbol{d}_m^{(i+1)} = \boldsymbol{d}_m^{(i)} + \text{CausalSelfAttn}(\boldsymbol{d}_m^{(i)})$$
$$\boldsymbol{d}_m^{(i+1)} = \boldsymbol{d}_m^{(i+1)} + \text{CrossAttn}(\boldsymbol{d}_m^{(i+1)}, \boldsymbol{z}_{enc}, \boldsymbol{z}_{enc})$$
$$\boldsymbol{d}_m^{(i+1)} = \boldsymbol{d}_m^{(i+1)} + \text{MoE}(\text{RMSNorm}(\boldsymbol{d}_m^{(i+1)}))$$

### 6.2 OneRec-V2 Lazy Decoder-Only 架构

**核心设计哲学**：将计算资源集中到真正对损失贡献梯度的目标物品 Token 上

**效率提升**：

| 架构 | 参数量 | 计算量 (GFLOPs) | 收敛损失 |
|------|--------|-----------------|----------|
| Encoder-Decoder | 1B | 296.36 | 3.28 |
| Lazy Decoder-Only | 1B | 18.89 | 3.27 |

**计算开销降低 94%，训练资源节约 90%！**

**Context Processor 设计**：
$$d_{context} = S_{kv} \cdot L_{kv} \cdot G_{kv} \cdot d_{head}$$

键值对在整个前向传播过程中保持不变，可被多个解码器层共享。

**Lazy Decoder Block**：
1. **Lazy Cross-Attention**：无键值投影，直接使用预计算键值对
2. **Causal Self-Attention**：目标物品语义 ID Token 间的自回归建模
3. **Feed-Forward Network**：深层可用 MoE 替代

**面试深挖点**：
- 为什么 Lazy Decoder-Only 能实现接近 100% 的计算集中在目标 Token？
- 与传统 Encoder-Decoder 的本质区别是什么？

---

## 7. 会思考的推荐模型：OneRec-Think

### 7.1 核心创新

将对话、推理与推荐统一在同一个框架中：
- **自然语言交互**：理解用户即时需求
- **显式推理生成**：生成结构化推理路径
- **端到端推荐**：直接生成推荐物品语义 ID

### 7.2 三阶段训练框架

#### 阶段一：物品对齐（Itemic Alignment）

**逐层内容描述生成**：

```
输入: <item_a_8121>
输出: 这是一个街头美食类视频

输入: <item_a_8121><item_b_3259>  
输出: 这是一个展示繁华街头市场的美食视频，包含各种小吃摊位
```

**目标函数**：
$$\mathcal{L}_{\text{align}} = \mathcal{L}_{\text{ID2Text}} + \mathcal{L}_{\text{Text2ID}}$$

#### 阶段二：推理激活（Reasoning Activation）

**推理脚手架三个层次**：

1. **用户画像推理**：从历史行为生成兴趣总结
   - 训练归纳推理能力：具体实例→一般模式

2. **候选评估推理**：评估候选物品匹配度
   - 训练演绎推理能力：一般偏好→具体预期

3. **端到端推理式推荐**：无候选集直接生成推荐
   - 整合反事实推理和多目标权衡

#### 阶段三：推理增强（Reasoning Enhancement）

**推荐特定奖励函数**：

$$r(s, a) = \lambda_1 \cdot r_{\text{cf}} + \lambda_2 \cdot r_{\text{sem}} + \lambda_3 \cdot r_{\text{coh}} + \lambda_4 \cdot r_{\text{feedback}}$$

- $r_{\text{cf}}$：协同过滤相似度
- $r_{\text{sem}}$：内容语义相关性
- $r_{\text{coh}}$：推理连贯性（通过 NLI 模型判断）
- $r_{\text{feedback}}$：用户反馈信号

**GRPO（Group Relative Policy Optimization）**：

$$\hat{A}_k = r(s_k, a_k) - \frac{1}{K}\sum_{j=1}^{K}r(s_j, a_j)$$

**关键理解**：模型不需要知道"绝对正确"的推荐是什么，只需学会识别哪些推理路径和推荐结果相对更好。

### 7.3 Think-Ahead 架构

**核心理念**：推理过程可以提前在用户行为更新时异步计算

**工作流程**：
1. **异步推理预计算**：用户新交互触发后台推理（容忍 500ms）
2. **轻量级在线选择**：从预计算候选集快速选择（10-20ms）
3. **增量推理更新**：仅在用户画像显著变化时重新推理

**性能指标**：
- P50 延迟降低 73%（320ms→86ms）
- P99 延迟降低 68%（480ms→153ms）
- 推理质量保持率 98.5%

---

## 8. 强化学习在推荐中的应用

### 8.1 奖励系统设计（OneRec-V1）

**三个层次**：

1. **用户偏好对齐（P-Score）**：
   - 使用神经网络学习个性化偏好分数
   - 基于 SIM 架构，每个目标构建独立塔
   - 可动态调整各目标权重

2. **生成格式规范化**：
   - 解决"挤压效应"：RL 导致合法 Token 概率被挤压到非法 Token 水平
   - 格式奖励：对合法样本设置优势为 1，非法样本直接丢弃

3. **工业场景对齐**：
   - 端到端特性使得只需将优化目标融入奖励系统
   - 例如：病毒内容农场降权

### 8.2 ECPO 算法

$$\mathcal{J}_{ECPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min\left(\frac{\pi_{\theta}(o_i|u)}{\pi_{\theta_{old}}'(o_i|u)}A_i, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)A_i\right)\right]$$

**关键改进**：对负优势样本的策略比率进行预先裁剪，防止梯度爆炸。

### 8.3 GBPO 算法（OneRec-V2）

**核心思想**：用 BCE 损失的稳定梯度来界定 RL 梯度

$$\pi_{\theta_{old}}'(o_i|u) = \begin{cases} \max(\pi_{\theta_{old}}, \text{sg}(\pi_\theta)), & A_i \ge 0 \\ \max(\pi_{\theta_{old}}, 1 - \text{sg}(\pi_\theta)), & A_i < 0 \end{cases}$$

**优势**：
- 完整样本利用：保留所有样本的梯度
- 有界梯度稳定化：用 BCE 梯度界定 RL 梯度

### 8.4 时长感知奖励塑形

**解决长视频偏差**：

1. **对数分桶**：$\mathcal{F}(d) = \lfloor \log_{\beta}(d + \epsilon) \rfloor$
2. **分桶内百分位计算**：$q_i = \frac{|\{p_j \in P_{u,b} | p_j \le p_i\}|}{|P_{u,b}|}$
3. **优势值分配**：前 25% 为正样本（+1），明确负反馈为负样本（-1）

---

## 9. Scaling Law 与工程实践

### 9.1 推荐系统中的 Scaling Law

OneRec-V2 验证的幂律关系：

$$\hat{L}(N) = E + \frac{A}{N^\alpha}$$

拟合参数：$E=3.13, A=3660, \alpha=0.489$

**实验验证**：

| 模型规模 | 参数量 | 收敛损失 |
|---------|--------|----------|
| Dense | 0.1B | 3.57 |
| Dense | 1B | 3.27 |
| Dense | 8B | 3.19 |
| MoE | 4B (0.5B 激活) | 3.22 |

**关键洞察**：MoE 模型（4B 总参数，0.5B 激活）优于 2B 密集模型，计算开销与 0.5B 密集模型相当。

### 9.2 HSTU 架构细节

```python
# src/funrec/models/hstu.py 核心实现
class HstuLayer(tf.keras.layers.Layer):
    """多头注意力机制，支持：
    - dot_product: 标准点积注意力
    - relative_position_bias: 相对位置偏置
    - time_interval_bias: 时间间隔偏置
    """
    
    def call(self, inputs, input_interval=None, training=True):
        # Q, K, V 线性投影
        Q = self.query_dense(self.queries)
        K = self.key_dense(self.keys)
        V = self.value_dense(self.keys) if self.value_projection else self.keys
        
        # 多头分割
        Q_split = tf.reshape(Q, [batch_size, -1, self.num_heads, 
                                 self.num_units // self.num_heads])
        
        # 应用注意力（支持多种类型）
        if "dot_product" in self.attention_type:
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = self.apply_attention(K_, V_, outputs)
        
        # U 投影（可选）
        if self.u_projection:
            U = self.silu(self.u_dense(self.queries))
            outputs = U * self.normalize(outputs)
        
        # 残差连接
        outputs += self.queries
        return outputs
```

**面试考点**：HSTU 相比标准 Transformer 的关键改进是什么？
- U 投影机制：$\text{output} = U \cdot \text{normalize}(\text{attention})$
- 支持时间间隔偏置建模
- 硬件感知设计优化 MFU

---

## 10. 高频面试问题与深度回答

### Q1: 生成式推荐与判别式推荐的本质区别是什么？

**深度回答**：

本质区别在于**建模哲学**：

- **判别式**：学习 $p(y=1|u,i,c)$，在给定候选集下做出最优选择，是"自上而下"的工程化思路
- **生成式**：学习 $p_\theta(i_{1:T}|u,c)$，学习用户行为的生成过程，是"自下而上"的建模思路

**具体差异**：
1. **目标函数**：局部决策边界 vs 完整概率分布
2. **信息流动**：前馈网络独立打分 vs 自回归循环流动
3. **架构要求**：多阶段异构模块 vs 统一 Transformer 架构

### Q2: 语义 ID 是如何同时编码语言语义和协同语义的？

**深度回答**：

以 PLUM 为例：

1. **多模态嵌入拼接**：$\boldsymbol{e}_{\text{final}} = [\boldsymbol{e}_{\text{text}} \oplus \boldsymbol{e}_{\text{visual}} \oplus \boldsymbol{e}_{\text{audio}} \oplus \boldsymbol{e}_{\text{cf}}]$
   - 文本/视觉/音频编码器提取内容语义
   - 协同过滤嵌入编码"谁看了这个还看了什么"

2. **RQ-VAE 量化**：将融合表示离散化为层次化索引
   - 码本学习到的是内容-行为联合空间的聚类

3. **持续预训练**：通过 ID-文本交错序列建立双向映射
   - 50% 用户行为数据（纯 ID 序列）
   - 50% 视频元数据语料（领域文本+ID-文本交错）

### Q3: OneRec-Think 的推理脚手架是如何激活 LLM 推理能力的？

**深度回答**：

采用**渐进式训练**，类似教育学中的"脚手架教学法"：

1. **用户画像推理**（归纳推理）：
   - 从离散 ID 序列识别内容主题
   - 统计分析不同主题占比
   - 用自然语言组织成连贯画像

2. **候选评估推理**（演绎推理）：
   - 建立"用户兴趣→物品内容→匹配判断"三段论
   - 不是简单的二分类，而是生成详细推理过程

3. **端到端推理式推荐**（综合推理）：
   - 整合反事实推理（用户表达与历史不同的需求）
   - 多目标权衡（内容相关性与情感需求的平衡）

**关键创新**：通过强化学习（GRPO）精炼推理路径，使用推荐特定奖励函数而非简单的 0-1 标签。

### Q4: Lazy Decoder-Only 架构为什么能实现 94% 的计算节省？

**深度回答**：

传统 Encoder-Decoder 的问题：
- 编码器处理完整用户上下文序列（数百个 Token）
- 解码器对每个目标 Token 都需要交叉注意力
- 大量计算花在上下文编码上，而真正产生梯度的目标 Token 解码占比极低

Lazy Decoder-Only 的解决方案：

1. **Context Processor**：将用户特征统一转换为键值对
   - $d_{context} = S_{kv} \cdot L_{kv} \cdot G_{kv} \cdot d_{head}$
   - 键值对在整个前向传播中保持不变，可被多层共享

2. **Lazy Decoder Block**：
   - 不将上下文作为序列一部分，而是静态条件信息
   - 训练时输入仅 3 个 Token：[BOS, s^1, s^2]
   - 通过交叉注意力访问预计算的键值对

3. **分组查询注意力（GQA）**：多个查询头共享同一组键值头

### Q5: 推荐系统中的强化学习面临哪些独特挑战？

**深度回答**：

1. **多有效性（Multi-Validity）问题**：
   - 数学推理有唯一正确答案，推荐可能有数十个"正确"选择
   - 简单监督学习会惩罚不在标签中的合理推荐

   **解决方案**：GRPO 基于相对优势而非绝对标准，允许多种有效推理模式

2. **隐式反馈的噪声**：
   - 用户反馈是隐式的（点击、观看时长），非明确好坏评判
   - 长视频天然倾向于累积更长播放时长

   **解决方案**：时长感知奖励塑形，分桶内百分位计算

3. **Reward Hacking**：
   - 模型可能学会"欺骗"奖励模型
   - 挤压效应导致合法 Token 概率被挤压

   **解决方案**：
   - KL 散度约束防止偏离参考模型过远
   - 格式奖励解决挤压效应
   - GBPO 用 BCE 梯度界定 RL 梯度

### Q6: 如何在实时推荐系统中部署"会思考"的模型？

**深度回答**：

采用 **Think-Ahead 架构**：

1. **异步推理预计算**：
   - 用户新交互触发后台推理（500ms 预算）
   - 生成多条推理路径，每条对应一个候选集
   - 缓存在用户的实时特征存储中

2. **轻量级在线选择**：
   - 推荐请求到来时，从预计算候选集快速选择
   - 基于实时上下文（时间、设备、会话状态）打分排序
   - 10-20ms 内完成

3. **增量推理更新**：
   - 新行为与已有推理一致时，仅追加简短更新
   - 仅在用户画像显著变化时触发完整重新推理

**效果**：P99 延迟 < 150ms，推理质量保持率 98.5%

---

## 11. 代码实现要点

### 11.1 统一实验接口

```python
# src/funrec/experiment.py
def run_experiment(model_name: str, verbose: bool = True, 
                   return_metrics: bool = False) -> Dict[str, Any]:
    config = load_config(model_name)
    train_data, test_data = load_data(config.data)
    feature_columns, processed_data = prepare_features(
        config.features, train_data, test_data
    )
    models = train_model(config.training, feature_columns, processed_data)
    metrics = evaluate_model(models, processed_data, config.evaluation, feature_columns)
    return metrics
```

### 11.2 模型构建模式

```python
# 统一的模型构建函数签名
def build_xxx_model(feature_columns, model_config):
    """
    返回: (main_model, user_model, item_model)
    - main_model: 训练用完整模型
    - user_model: 推理时用户塔
    - item_model: 推理时物品塔
    """
    # 1. 构建输入层
    input_layer_dict = build_input_layer(feature_columns)
    
    # 2. 构建 Embedding 表
    embedding_table_dict = build_embedding_table_dict(feature_columns)
    
    # 3. 特征交叉/序列建模
    # ...
    
    # 4. 构建模型
    model = tf.keras.Model(inputs=..., outputs=...)
    
    # 5. 设置推理接口
    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_embedding)
    
    return model, user_model, item_model
```

### 11.3 关键层实现

**FM 二阶交叉**：
```python
# src/funrec/models/layers.py
class FM(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        # 0.5 * ((sum(v))^2 - sum(v^2))
        square_of_sum = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(inputs * inputs, axis=1, keepdims=True)
        cross_term = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=2)
        return cross_term
```

**DIN 注意力**：
```python
class DinAttentionLayer(tf.keras.layers.Layer):
    def call(self, inputs, mask=None, **kwargs):
        query, keys = inputs
        # 拼接 [query, keys, query-keys, query*keys]
        att_inputs = tf.concat([tf.tile(query, [1, length, 1]),
                                keys, query - keys, query * keys], axis=-1)
        hidden_layer = self.ffn_layer(att_inputs)
        scores = self.dense(hidden_layer)
        scores = tf.nn.softmax(scores)
        return tf.squeeze(tf.matmul(scores, keys), axis=1)
```

### 11.4 训练器关键逻辑

```python
# src/funrec/training/trainer.py
def train_model(training_config, feature_columns, processed_data):
    # 动态导入模型构建函数
    build_function_path = training_config.get("build_function")
    module_path, function_name = build_function_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    build_function = getattr(module, function_name)
    
    # 构建模型
    model, user_model, item_model = build_function(feature_columns, model_params)
    
    # 处理多输出模型
    if isinstance(loss, list) and len(loss) > 1:
        labels_for_fit = [labels_for_fit] * len(loss)
    
    # 训练
    history = model.fit(train_features, labels_for_fit, ...)
    return model, user_model, item_model
```

---

## 12. 面试者自我介绍模板

### 版本一：技术深度型

> 面试官您好，我是 XXX，XX 大学 MS 在读，专注于 LLM 与推荐系统交叉领域的研究。
>
> 在技术积累方面，我系统学习了推荐系统从传统级联架构到生成式范式的完整演进。深入理解了 HSTU、OneRec 等生成式推荐架构，特别是 Lazy Decoder-Only 设计如何实现 94% 的计算节省。对 LLM 三阶段训练范式（预训练-SFT-RLHF）有扎实掌握，并研究了其在推荐场景的适配方案，包括语义 ID 构建（RQ-VAE/RQ-Kmeans）、协同语义与语言语义的统一（LC-Rec/PLUM 框架）。
>
> 在实践方面，我基于 fun-rec 框架复现了多种推荐模型（SASRec、DSSM、HSTU 等），理解其模块化设计和统一实验接口。对 OneRec-Think 的推理脚手架、GRPO 强化学习、Think-Ahead 部署架构有深入研究，能够清晰阐述如何让推荐系统"先思考再推荐"。
>
> 我对贵公司的 XX 方向非常感兴趣，特别是 [结合具体业务场景]。我相信我的技术积累和学习能力能够为团队带来价值。

### 版本二：问题导向型

> 面试官您好，我是 XXX。最近我在思考一个问题：如何让推荐系统像 ChatGPT 一样具备推理能力？
>
> 为了解答这个问题，我深入研究了生成式推荐的最新进展。我发现核心挑战在于三个层面：首先是如何让 LLM "认识"推荐物品——通过语义 ID 和多层次对齐训练（如 LC-Rec 的索引到文本双向映射）；其次是如何让 LLM "思考"推荐理由——OneRec-Think 的推理脚手架通过用户画像推理、候选评估推理、端到端推理三个层次逐步激活推理能力；最后是如何让 LLM 的思考"落地"——Think-Ahead 架构通过异步预计算实现 150ms 内的实时推荐。
>
> 这个研究过程让我对 LLM 在推荐系统中的应用有了系统性的理解。我期待在贵公司能够将这些理论知识转化为实际的业务价值。

---

## 附录：关键论文与参考

1. **InstructGPT**：三阶段训练范式的奠基工作
2. **LC-Rec**：Language-Collaborative Recommendation，语义 ID 与对齐训练
3. **PLUM**：YouTube 工业级语义对齐框架
4. **OneRec-V1/V2**：快手端到端生成式推荐，Lazy Decoder-Only 架构
5. **OneRec-Think**：会思考的推荐模型，推理脚手架与 GRPO
6. **HSTU**：Meta 生成式推荐架构，Scaling Law 首次探索
7. **DPO**：Direct Preference Optimization，简化 RLHF

---

> **文档生成时间**：2026-06-27
> 
> **基于项目**：fun-rec（深度推荐算法实践）
> 
> **适用场景**：LLM 算法实习面试准备，特别是 LLM & Agent & GR 应用或后训练方向
