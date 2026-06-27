# Fun-Rec 项目完整复现计划

> **硬件资源**：8卡 NVIDIA RTX 3090（每卡 24GB 显存，总计 192GB）
> 
> **目标**：完整复现项目中的核心模型，深入理解 LLM & GR & Agent & RL 后训练技术栈

---

## 一、资源评估与可行性分析

### 1.1 项目模型资源需求总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Fun-Rec 模型资源需求                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  模型类型        │  参数量      │  显存需求   │  训练时间(3090)  │  并行策略  │
├─────────────────────────────────────────────────────────────────────────────┤
│  【召回模型】                                                           │
│  ItemCF/UserCF   │  无参数      │  CPU only   │  分钟级          │  -        │
│  Swing           │  无参数      │  CPU only   │  分钟级          │  -        │
│  Item2Vec        │  ~1M         │  <1GB       │  分钟级          │  单卡     │
│  FM_Recall       │  ~1M         │  <1GB       │  分钟级          │  单卡     │
│  YoutubeDNN      │  ~5M         │  ~2GB       │  5-10分钟        │  单卡     │
│  DSSM            │  ~5M         │  ~2GB       │  5-10分钟        │  单卡     │
│  EGES            │  ~5M         │  ~2GB       │  10-15分钟       │  单卡     │
│  MIND            │  ~10M        │  ~4GB       │  10-15分钟       │  单卡     │
│  SDM             │  ~10M        │  ~4GB       │  10-15分钟       │  单卡     │
├─────────────────────────────────────────────────────────────────────────────┤
│  【排序模型】                                                           │
│  FM              │  ~1M         │  <1GB       │  分钟级          │  单卡     │
│  DeepFM          │  ~5M         │  ~2GB       │  5-10分钟        │  单卡     │
│  DCN             │  ~5M         │  ~2GB       │  5-10分钟        │  单卡     │
│  WDL             │  ~5M         │  ~2GB       │  5-10分钟        │  单卡     │
│  PNN/NFM/AFM     │  ~5M         │  ~2GB       │  5-10分钟        │  单卡     │
│  AutoInt         │  ~10M        │  ~4GB       │  10-15分钟       │  单卡     │
│  xDeepFM         │  ~10M        │  ~4GB       │  10-15分钟       │  单卡     │
│  FiBiNET         │  ~10M        │  ~4GB       │  10-15分钟       │  单卡     │
├─────────────────────────────────────────────────────────────────────────────┤
│  【序列模型】                                                           │
│  SASRec          │  ~5M         │  ~2GB       │  10-15分钟       │  单卡     │
│  DIN             │  ~5M         │  ~2GB       │  5-10分钟        │  单卡     │
│  DIEN            │  ~10M        │  ~4GB       │  10-15分钟       │  单卡     │
│  DSIN            │  ~10M        │  ~4GB       │  10-15分钟       │  单卡     │
├─────────────────────────────────────────────────────────────────────────────┤
│  【多任务模型】                                                          │
│  Shared_Bottom   │  ~10M        │  ~4GB       │  10-15分钟       │  单卡     │
│  MMOE/HMoe       │  ~15M        │  ~6GB       │  15-20分钟       │  单卡     │
│  PLE             │  ~20M        │  ~8GB       │  15-20分钟       │  单卡     │
│  PEPNet          │  ~15M        │  ~6GB       │  15-20分钟       │  单卡     │
│  PRM/PRS         │  ~15M        │  ~6GB       │  15-20分钟       │  单卡     │
│  STAR/APG        │  ~15M        │  ~6GB       │  15-20分钟       │  单卡     │
├─────────────────────────────────────────────────────────────────────────────┤
│  【生成式推荐】                                                          │
│  HSTU            │  ~50M        │  ~8GB       │  30-60分钟       │  单卡     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 资源分配策略

```
8卡3090资源分配方案：

方案A：串行复现（推荐，深入理解每个模型）
├── 每次只训练一个模型
├── 专注理解模型原理和实现细节
└── 总时间：约2-3天

方案B：并行加速（快速验证）
├── 同时训练8个模型
├── 快速验证所有模型效果
└── 总时间：约1天

方案C：混合策略（平衡）
├── 阶段1：串行学习核心模型（HSTU/SASRec/DSSM/DeepFM）
├── 阶段2：并行验证其他模型
└── 总时间：约1.5天
```

### 1.3 关键发现

```
✅ 项目所有模型都可以在单卡3090上训练
✅ 8卡3090完全可以支撑完整复现
✅ 主要瓶颈是数据下载和预处理，而非GPU算力
✅ HSTU是唯一接近GPU瓶颈的模型（~8GB显存）
```

---

## 二、复现阶段规划

### 阶段0：环境搭建（预计时间：2-3小时）

```bash
# Step 1: 创建conda环境
conda create -n funrec python=3.10 -y
conda activate funrec

# Step 2: 安装依赖
cd /home/chenyizhou/fun-rec
pip install -r requirements.txt

# Step 3: 设置环境变量
export FUNREC_RAW_DATA_PATH=/home/chenyizhou/fun-rec/data/raw
export FUNREC_PROCESSED_DATA_PATH=/home/chenyizhou/fun-rec/data/processed

# Step 4: 验证GPU可用
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 阶段1：基础召回模型（预计时间：3-4小时）

**目标**：理解协同过滤和向量召回的基本原理

#### 1.1 ItemCF/UserCF（CPU模型，理解原理）

```python
# 运行ItemCF
import funrec
metrics = funrec.run_experiment('item_cf', return_metrics=True)
print(f"ItemCF指标: {metrics}")

# 运行UserCF
metrics = funrec.run_experiment('user_cf', return_metrics=True)
print(f"UserCF指标: {metrics}")
```

**学习重点**：
- 协同过滤的核心思想：相似用户/相似物品
- 余弦相似度/皮尔逊相关系数的计算
- 冷启动问题的处理

#### 1.2 YoutubeDNN（向量召回基础）

```python
# 运行YoutubeDNN
metrics = funrec.run_experiment('youtubednn', return_metrics=True)
print(f"YoutubeDNN指标: {metrics}")
```

**学习重点**：
- 双塔结构：用户塔 + 物品塔
- Sampled Softmax Loss的原理
- 负采样策略

#### 1.3 DSSM（对比学习召回）

```python
# 运行DSSM
metrics = funrec.run_experiment('dssm', return_metrics=True)
print(f"DSSM指标: {metrics}")
```

**学习重点**：
- 对比学习损失函数（InfoNCE）
- 温度参数的作用
- 向量归一化的重要性

### 阶段2：排序模型（预计时间：4-5小时）

**目标**：理解特征交叉和CTR预估的核心技术

#### 2.1 FM（二阶特征交叉）

```python
# 运行FM
metrics = funrec.run_experiment('fm', return_metrics=True)
print(f"FM指标: {metrics}")
```

**学习重点**：
- FM的数学原理：$0.5 \cdot ((\sum v)^2 - \sum v^2)$
- 隐向量的作用
- 时间复杂度优化

#### 2.2 DeepFM（FM + DNN）

```python
# 运行DeepFM
metrics = funrec.run_experiment('deepfm', return_metrics=True)
print(f"DeepFM指标: {metrics}")
```

**学习重点**：
- Wide & Deep架构
- FM部分和DNN部分的分工
- 特征工程的重要性

#### 2.3 DCN（显式高阶交叉）

```python
# 运行DCN
metrics = funrec.run_experiment('dcn', return_metrics=True)
print(f"DCN指标: {metrics}")
```

**学习重点**：
- Cross Network的数学原理
- 高阶特征交叉的显式建模
- 与FM的区别和联系

### 阶段3：序列模型（预计时间：4-5小时）

**目标**：理解用户行为序列建模的核心技术

#### 3.1 SASRec（自注意力序列推荐）

```python
# 运行SASRec
metrics = funrec.run_experiment('sasrec', return_metrics=True)
print(f"SASRec指标: {metrics}")
```

**学习重点**：
- Transformer在推荐中的应用
- 因果掩码的作用
- 位置编码的选择

#### 3.2 DIN（注意力机制排序）

```python
# 运行DIN
metrics = funrec.run_experiment('din', return_metrics=True)
print(f"DIN指标: {metrics}")
```

**学习重点**：
- 注意力机制在推荐中的作用
- 目标物品与历史行为的交互
- Dice激活函数

#### 3.3 HSTU（生成式推荐基础）

```python
# 运行HSTU
metrics = funrec.run_experiment('hstu', return_metrics=True)
print(f"HSTU指标: {metrics}")
```

**学习重点**：
- HSTU与SASRec的区别
- 时间间隔偏置的作用
- U投影机制

### 阶段4：多任务模型（预计时间：3-4小时）

**目标**：理解多目标优化和任务关系建模

#### 4.1 MMOE（多门混合专家）

```python
# 运行MMOE
metrics = funrec.run_experiment('mmoe', return_metrics=True)
print(f"MMOE指标: {metrics}")
```

**学习重点**：
- 混合专家（MoE）架构
- 门控机制的作用
- 任务冲突的处理

#### 4.2 PLE（渐进式分层抽取）

```python
# 运行PLE
metrics = funrec.run_experiment('ple', return_metrics=True)
print(f"PLE指标: {metrics}")
```

**学习重点**：
- 共享专家和任务专家的分离
- 渐进式多层抽取
- 任务漂移问题

### 阶段5：生成式推荐深入（预计时间：6-8小时）

**目标**：深入理解生成式推荐的核心技术

#### 5.1 HSTU深度分析

```python
# 分析HSTU的注意力机制
from funrec.models.hstu import HstuLayer, build_hstu_model

# 查看模型结构
model, user_model, item_model = build_hstu_model(feature_columns, model_config)
model.summary()

# 分析注意力权重
# ...
```

**学习重点**：
- 相对位置偏置
- 时间间隔偏置
- 因果掩码
- U投影机制

#### 5.2 语义ID构建（RQ-VAE/RQ-Kmeans）

```python
# 语义ID构建示例
import numpy as np
from sklearn.cluster import KMeans

class RQKmeans:
    def __init__(self, num_layers=3, codebook_size=256):
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.codebooks = []
    
    def fit(self, embeddings):
        residuals = embeddings.copy()
        for layer in range(self.num_layers):
            kmeans = KMeans(n_clusters=self.codebook_size)
            kmeans.fit(residuals)
            self.codebooks.append(kmeans.cluster_centers_)
            residuals = residuals - kmeans.cluster_centers_[kmeans.labels_]
    
    def encode(self, embeddings):
        indices = []
        residuals = embeddings.copy()
        for layer in range(self.num_layers):
            kmeans = KMeans(n_clusters=self.codebook_size)
            kmeans.cluster_centers_ = self.codebooks[layer]
            labels = kmeans.predict(residuals)
            indices.append(labels)
            residuals = residuals - self.codebooks[layer][labels]
        return np.stack(indices, axis=1)
```

**学习重点**：
- 残差量化原理
- 码本学习方法
- 索引冲突问题

#### 5.3 LLM三阶段训练范式

```python
# 模拟SFT训练
def sft_training_step(model, batch):
    """SFT训练步骤"""
    input_ids = batch['input_ids']
    labels = batch['labels']
    
    # 只对输出部分计算损失
    outputs = model(input_ids)
    loss = cross_entropy(outputs, labels, ignore_index=-100)
    
    return loss

# 模拟DPO训练
def dpo_training_step(model, batch):
    """DPO训练步骤"""
    chosen = batch['chosen']
    rejected = batch['rejected']
    
    # 计算策略概率
    chosen_logprobs = model.log_prob(chosen)
    rejected_logprobs = model.log_prob(rejected)
    
    # DPO损失
    logits = beta * (chosen_logprobs - rejected_logprobs)
    loss = -torch.log(torch.sigmoid(logits)).mean()
    
    return loss
```

**学习重点**：
- 预训练-指令微调-偏好对齐三阶段
- RLHF vs DPO的区别
- KL散度约束的作用

### 阶段6：强化学习与推荐对齐（预计时间：4-5小时）

**目标**：理解强化学习在推荐中的应用

#### 6.1 GRPO算法理解

```python
# GRPO实现示例
def grpo_loss(policy, ref_policy, prompts, responses, rewards):
    """Group Relative Policy Optimization"""
    # 计算策略概率
    policy_logprobs = policy.log_prob(prompts, responses)
    ref_logprobs = ref_policy.log_prob(prompts, responses)
    
    # 计算相对优势
    advantages = rewards - rewards.mean()
    advantages = advantages / (rewards.std() + 1e-8)
    
    # 策略比率
    ratio = torch.exp(policy_logprobs - ref_logprobs)
    
    # GRPO损失
    loss = -torch.min(
        ratio * advantages,
        torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
    ).mean()
    
    return loss
```

**学习重点**：
- 相对优势 vs 绝对优势
- 裁剪机制的作用
- 多有效性问题

#### 6.2 奖励函数设计

```python
# 推荐系统奖励函数
def recommendation_reward(rec_item, user_history, user_feedback):
    """综合奖励函数"""
    # 协同过滤相似度
    r_cf = compute_cf_similarity(rec_item, user_history)
    
    # 内容语义相关性
    r_sem = compute_semantic_similarity(rec_item, user_history)
    
    # 推理连贯性
    r_coh = compute_coherence(rec_item, reasoning_path)
    
    # 用户反馈
    r_feedback = user_feedback  # 点击=0.5, 完播=1.0, 快速划过=-0.5
    
    # 综合奖励
    reward = (lambda_1 * r_cf + 
              lambda_2 * r_sem + 
              lambda_3 * r_coh + 
              lambda_4 * r_feedback)
    
    return reward
```

**学习重点**：
- 奖励函数的多维度设计
- 时长感知奖励塑形
- 格式奖励

### 阶段7：工程落地实践（预计时间：4-5小时）

**目标**：理解推荐系统的工程落地

#### 7.1 离线流水线

```bash
# 运行离线流水线
cd /home/chenyizhou/fun-rec/web_project
python -m backend.offline.pipeline --steps all
```

**学习重点**：
- 特征工程流程
- 模型训练流程
- 特征上线和模型部署

#### 7.2 在线服务

```bash
# 启动在线服务
cd /home/chenyizhou/fun-rec/web_project
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**学习重点**：
- 多路召回策略
- 排序模型推理
- 重排多样性策略

#### 7.3 系统监控

```python
# 健康检查
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "recall": {"status": "ready", "latency_p99": "15ms"},
            "ranking": {"status": "ready", "latency_p99": "30ms"},
            "reranking": {"status": "ready", "latency_p99": "5ms"}
        }
    }
```

**学习重点**：
- 降级策略
- 缓存优化
- 监控告警

---

## 三、每日执行计划

### Day 1：环境搭建 + 基础召回模型

```
上午（3小时）：
├── 环境搭建（2小时）
│   ├── conda环境创建
│   ├── 依赖安装
│   └── GPU验证
└── ItemCF/UserCF复现（1小时）
    ├── 运行实验
    └── 理解原理

下午（4小时）：
├── YoutubeDNN复现（2小时）
│   ├── 运行实验
│   ├── 分析模型结构
│   └── 理解双塔架构
├── DSSM复现（1.5小时）
│   ├── 运行实验
│   └── 理解对比学习
└── 总结与笔记（0.5小时）
```

### Day 2：排序模型 + 序列模型

```
上午（3小时）：
├── FM复现（1小时）
├── DeepFM复现（1小时）
└── DCN复现（1小时）

下午（4小时）：
├── SASRec复现（2小时）
│   ├── 运行实验
│   ├── 分析注意力机制
│   └── 理解因果掩码
├── DIN复现（1.5小时）
└── 总结与笔记（0.5小时）
```

### Day 3：HSTU + 多任务模型

```
上午（3小时）：
├── HSTU深度分析（2小时）
│   ├── 运行实验
│   ├── 分析时间间隔偏置
│   ├── 分析U投影机制
│   └── 对比SASRec
└── MMOE复现（1小时）

下午（4小时）：
├── PLE复现（1.5小时）
├── PEPNet复现（1.5小时）
├── 语义ID构建实践（1小时）
└── 总结与笔记（0.5小时）
```

### Day 4：生成式推荐深入

```
上午（3小时）：
├── 语义ID构建（RQ-VAE/RQ-Kmeans）（2小时）
│   ├── 实现RQ-Kmeans
│   ├── 在MovieLens上测试
│   └── 分析索引质量
└── LLM三阶段训练理解（1小时）

下午（4小时）：
├── LC-Rec/PLUM论文精读（2小时）
├── OneRec-V1/V2架构分析（1.5小时）
└── 总结与笔记（0.5小时）
```

### Day 5：强化学习与推荐对齐

```
上午（3小时）：
├── GRPO算法实现（2小时）
│   ├── 实现GRPO损失函数
│   ├── 对比PPO/DPO
│   └── 分析相对优势
└── 奖励函数设计（1小时）

下午（4小时）：
├── OneRec-Think论文精读（2小时）
├── 推理脚手架设计分析（1.5小时）
└── 总结与笔记（0.5小时）
```

### Day 6：工程落地实践

```
上午（3小时）：
├── 离线流水线运行（2小时）
│   ├── 特征工程
│   ├── 模型训练
│   └── 特征上线
└── 模型部署（1小时）

下午（4小时）：
├── 在线服务搭建（2小时）
│   ├── 多路召回
│   ├── 排序推理
│   └── 重排策略
├── 系统监控（1.5小时）
└── 总结与笔记（0.5小时）
```

### Day 7：总结与面试准备

```
上午（3小时）：
├── 项目整体回顾（1.5小时）
├── 关键技术梳理（1.5小时）

下午（4小时）：
├── 面试问题准备（2小时）
├── 自我介绍练习（1小时）
└── 查漏补缺（1小时）
```

---

## 四、关键代码实践

### 4.1 快速验证脚本

```python
#!/usr/bin/env python3
"""快速验证所有模型"""

import funrec
from tabulate import tabulate

MODELS = [
    # 召回模型
    'item_cf', 'user_cf', 'swing',
    'youtubednn', 'dssm', 'mind',
    # 排序模型
    'fm', 'deepfm', 'dcn', 'wide_deep',
    'nfm', 'afm', 'autoint', 'xdeepfm',
    # 序列模型
    'sasrec', 'din', 'dien',
    # 多任务模型
    'mmoe', 'ple', 'pepnet',
    # 生成式推荐
    'hstu'
]

def run_all_models():
    results = []
    for model_name in MODELS:
        try:
            print(f"\n{'='*50}")
            print(f"运行模型: {model_name}")
            print(f"{'='*50}")
            
            metrics = funrec.run_experiment(model_name, return_metrics=True)
            results.append({
                'model': model_name,
                'metrics': metrics,
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'model': model_name,
                'metrics': str(e),
                'status': 'failed'
            })
    
    # 打印结果表格
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    
    table_data = []
    for r in results:
        if r['status'] == 'success':
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in r['metrics'].items()])
        else:
            metrics_str = r['metrics']
        table_data.append([r['model'], r['status'], metrics_str])
    
    print(tabulate(table_data, headers=['模型', '状态', '指标'], tablefmt='grid'))

if __name__ == '__main__':
    run_all_models()
```

### 4.2 HSTU深度分析脚本

```python
#!/usr/bin/env python3
"""HSTU深度分析"""

import tensorflow as tf
from funrec.models.hstu import HstuLayer, build_hstu_model
from funrec.config.config_hstu import CONFIG
from funrec.data.loaders import load_data
from funrec.features.processors import prepare_features

def analyze_hstu():
    # 加载数据
    config = CONFIG
    train_data, test_data = load_data(config['data'])
    feature_columns, processed_data = prepare_features(
        config['features'], train_data, test_data
    )
    
    # 构建模型
    model, user_model, item_model = build_hstu_model(
        feature_columns, config['training']['model_params']
    )
    
    # 打印模型结构
    print("\n" + "="*50)
    print("HSTU模型结构")
    print("="*50)
    model.summary()
    
    # 分析注意力层
    print("\n" + "="*50)
    print("注意力层参数")
    print("="*50)
    for layer in model.layers:
        if 'hstu' in layer.name.lower() or 'attention' in layer.name.lower():
            print(f"\n层名称: {layer.name}")
            print(f"参数数量: {layer.count_params()}")
            if hasattr(layer, 'attention_type'):
                print(f"注意力类型: {layer.attention_type}")
    
    # 分析时间间隔偏置
    print("\n" + "="*50)
    print("时间间隔偏置分析")
    print("="*50)
    for layer in model.layers:
        if hasattr(layer, 'time_interval_bias'):
            bias = layer.time_interval_bias.numpy()
            print(f"时间间隔偏置形状: {bias.shape}")
            print(f"偏置值范围: [{bias.min():.4f}, {bias.max():.4f}]")
    
    return model

if __name__ == '__main__':
    analyze_hstu()
```

### 4.3 语义ID构建实践

```python
#!/usr/bin/env python3
"""语义ID构建实践"""

import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf

class RQKmeans:
    """残差量化K-means"""
    
    def __init__(self, num_layers=3, codebook_size=256):
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.codebooks = []
    
    def fit(self, embeddings):
        """训练码本"""
        residuals = embeddings.copy()
        
        for layer in range(self.num_layers):
            print(f"训练第{layer+1}层码本...")
            
            kmeans = KMeans(
                n_clusters=self.codebook_size,
                random_state=42,
                n_init=10
            )
            kmeans.fit(residuals)
            
            self.codebooks.append(kmeans.cluster_centers_)
            
            # 计算残差
            residuals = residuals - kmeans.cluster_centers_[kmeans.labels_]
            
            # 打印量化误差
            quantization_error = np.mean(np.sum(residuals**2, axis=1))
            print(f"  量化误差: {quantization_error:.4f}")
    
    def encode(self, embeddings):
        """编码为语义ID"""
        indices = []
        residuals = embeddings.copy()
        
        for layer in range(self.num_layers):
            kmeans = KMeans(n_clusters=self.codebook_size)
            kmeans.cluster_centers_ = self.codebooks[layer]
            labels = kmeans.predict(residuals)
            indices.append(labels)
            residuals = residuals - self.codebooks[layer][labels]
        
        return np.stack(indices, axis=1)
    
    def decode(self, indices):
        """解码为嵌入"""
        embeddings = np.zeros((len(indices), self.codebooks[0].shape[1]))
        for layer in range(self.num_layers):
            embeddings += self.codebooks[layer][indices[:, layer]]
        return embeddings

def demo_semantic_id():
    """语义ID构建演示"""
    # 生成模拟物品嵌入
    num_items = 1000
    embedding_dim = 128
    embeddings = np.random.randn(num_items, embedding_dim)
    
    # 构建语义ID
    rq_kmeans = RQKmeans(num_layers=3, codebook_size=64)
    rq_kmeans.fit(embeddings)
    
    # 编码
    semantic_ids = rq_kmeans.encode(embeddings)
    print(f"\n语义ID形状: {semantic_ids.shape}")
    print(f"示例语义ID: {semantic_ids[:5]}")
    
    # 解码
    reconstructed = rq_kmeans.decode(semantic_ids)
    
    # 计算重建误差
    reconstruction_error = np.mean(np.sum((embeddings - reconstructed)**2, axis=1))
    print(f"重建误差: {reconstruction_error:.4f}")
    
    # 分析索引唯一性
    unique_ids = set(tuple(sid) for sid in semantic_ids)
    print(f"唯一语义ID数量: {len(unique_ids)}")
    print(f"索引冲突率: {1 - len(unique_ids)/num_items:.2%}")

if __name__ == '__main__':
    demo_semantic_id()
```

---

## 五、学习成果检验

### 5.1 每日检查清单

```
Day 1 检查：
□ 能否解释ItemCF和UserCF的区别？
□ 能否解释YoutubeDNN的双塔结构？
□ 能否解释Sampled Softmax Loss的原理？
□ 能否运行并解释DSSM的对比学习损失？

Day 2 检查：
□ 能否推导FM的时间复杂度优化？
□ 能否解释DeepFM中FM和DNN的分工？
□ 能否解释SASRec的因果掩码作用？
□ 能否解释DIN的注意力机制？

Day 3 检查：
□ 能否解释HSTU的时间间隔偏置？
□ 能否解释HSTU的U投影机制？
□ 能否解释MMOE的门控机制？
□ 能否解释PLE的共享/任务专家分离？

Day 4 检查：
□ 能否实现RQ-Kmeans？
□ 能否解释LLM三阶段训练范式？
□ 能否解释LC-Rec的语义对齐？
□ 能否解释OneRec的Lazy Decoder-Only？

Day 5 检查：
□ 能否实现GRPO损失函数？
□ 能否解释相对优势的作用？
□ 能否设计推荐系统的奖励函数？
□ 能否解释OneRec-Think的推理脚手架？

Day 6 检查：
□ 能否运行离线流水线？
□ 能否搭建在线服务？
□ 能否解释多路召回策略？
□ 能否解释降级策略？

Day 7 检查：
□ 能否清晰介绍项目？
□ 能否回答五类面试问题？
□ 能否解释技术选型的原因？
□ 能否讨论局限性和改进方向？
```

### 5.2 面试模拟问题

```
基础问题：
1. 请解释生成式推荐与判别式推荐的区别
2. 请解释HSTU与SASRec的区别
3. 请解释语义ID的构建方法

深入问题：
1. OneRec-Think的推理脚手架为什么需要三阶段？
2. GRPO相比PPO的优势是什么？
3. 如何设计推荐系统的奖励函数？

工程问题：
1. 如何保证离线/线上特征一致性？
2. 模型上线后效果下降怎么排查？
3. 如何设计降级策略？

业务问题：
1. 生成式推荐适合什么场景？
2. 资源有限时优先优化什么？
3. 如何评估推荐系统的业务价值？
```

---

## 六、参考资源

### 6.1 论文推荐

```
必读论文：
├── [InstructGPT] Training language models to follow instructions with human feedback
├── [DPO] Direct Preference Optimization
├── [LC-Rec] Adapting Large Language Models for Recommender Systems
├── [PLUM] Pre-trained Language Models for Recommendations
├── [OneRec] OneRec: Unidirectional Generative Recommendation
├── [OneRec-Think] OneRec-Think: Thinking Before You Recommend
└── [HSTU] Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers

扩展论文：
├── [ChatRec] Towards Conversational Recommender Systems
├── [TALLRec] Recommendation as Language Processing
├── [InstructRec] Recommendation as Language Processing
└── [RecLLM] Large Language Models for Recommendation
```

### 6.2 代码参考

```
项目代码：
├── src/funrec/models/          # 模型实现
├── src/funrec/config/          # 配置文件
├── src/funrec/training/        # 训练流程
├── src/funrec/evaluation/      # 评估指标
└── web_project/                # 工程实践

关键文件：
├── hstu.py                     # HSTU模型实现
├── sasrec.py                   # SASRec模型实现
├── dssm.py                     # DSSM模型实现
├── trainer.py                  # 训练器
└── metrics.py                  # 评估指标
```

---

> **计划制定时间**：2026-06-27
> 
> **硬件资源**：8卡 RTX 3090（24GB × 8 = 192GB）
> 
> **预计完成时间**：7天
