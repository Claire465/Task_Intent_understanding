# Task Intent Understanding

用于重现评估任务意图理解（Task Intent Understanding）论文实验的代码库。

本项目针对医疗问答场景，通过人工和 LLM 自动生成两类评分标准，并基于语义熵（Semantic Entropy）量化模型在任务意图层面的不确定性。

---

## 目录

- [项目背景](#项目背景)
- [整体流程](#整体流程)
- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [分步说明](#分步说明)
  - [数据处理](#数据处理)
  - [评估流程](#评估流程)
  - [优化流程](#优化流程)
- [输出文件说明](#输出文件说明)

---

## 项目背景

传统 LLM 评估依赖人工编写的固定标准，难以捕捉模型对用户意图的理解深度。本项目提出：

1. 让人类和模型自身解读查询意图，生成两类评分问题（Scoring Questions）
2. 将评分问题按医疗维度分类（Completeness、Diagnostic Relevance 等 10 类）
3. 多轮独立评估 + 语义聚类，计算语义熵以衡量评估不确定性

---

## 整体流程

```
原始查询 (selected_query.txt)
    │
    ▼ [01_interpretation] LLM 解读意图，生成评分问题
    │
    ▼ [02_data_change] 格式标准化
    │
    ▼ [03_classification] 评分问题分类（A–J 10个医疗维度）
    │
    ▼ [04_generation] 目标模型生成回答
    │
    ├──────────────────────────────────────────────────┐
    ▼ [step1 × N次] LLM 逐条打分 (Yes/No)             │
    │                                                   │
    ▼ [step1_p_true] 聚合多轮，计算 p_true             │ 优化分支
    │                                                   │
    ▼ [step2_entailment] 语义聚类（NLI 蕴含关系）     │ [01_constraints] 生成语言约束
    │                                                   │
    ▼ [step3_bincount] 计算语义熵 (Shannon Entropy)   │ [02_recombination] 约束重组查询
    │                                                   │
    ▼ [step4] 汇总最终结果 ◄─────────────────────────┘
```

---

## 目录结构

```
Task_Intent_understanding/
├── run_pipeline.py              # 一键启动脚本
├── utils.py                     # 公共工具（读取 Prompt 模板）
│
├── Data_processing/             # 数据处理流程（步骤 1–4）
│   ├── 01_interpretation.py     # 查询意图解读
│   ├── 02_data_change.py        # 格式转换
│   ├── 03_classification.py     # 评分维度分类
│   ├── 04_generation.py         # 目标模型生成回答
│   ├── prompts/
│   │   └── 01_interpretation.txt
│   └── Processed_data/          # 中间数据目录（自动创建）
│
├── Evaluation_processing/       # 评估流程（步骤 1–4）
│   ├── step1.py                 # LLM 逐条评分
│   ├── step1_p_true.py          # 多轮聚合，计算 p_true
│   ├── step2_entailment.py      # 语义聚类
│   ├── step3_bincount.py        # 语义熵计算
│   ├── step4.py                 # 结果汇总
│   ├── prompts/
│   │   └── RAL_evaluator.py     # 评估 Prompt
│   └── evaluation_data/         # 评估结果目录（自动创建）
│
└── Optimization/                # 查询优化分支（独立）
    ├── 01_contraints_generate.py  # 语言约束生成
    ├── 02_recombination.py        # 约束重组查询
    ├── prompts/
    │   ├── 01_constraints.txt
    │   └── 02_recombination.txt
    └── data_precision/            # 优化结果目录（自动创建）
```

---

## 环境配置

**Python 版本：** 3.8+

**安装依赖：**

```bash
pip install requests tqdm numpy
```

**配置 API：**

在各脚本中找到以下占位符并替换为实际值：

| 占位符 | 说明 |
|--------|------|
| `"Baseurl"` | API 服务地址，如 `https://api.openai.com` |
| `"Skey"` | API 密钥（Bearer Token） |
| `"Model_Name"` | 调用的模型名称，如 `gpt-4o`、`gemini-pro` |

涉及文件：

- `Data_processing/01_interpretation.py`
- `Data_processing/03_classification.py`
- `Data_processing/04_generation.py`
- `Evaluation_processing/step1.py`
- `Evaluation_processing/step2_entailment.py`
- `Optimization/01_contraints_generate.py`
- `Optimization/02_recombination.py`

**准备输入数据：**

将待评估的查询逐行写入：

```
Data_processing/Processed_data/selected_query.txt
```

每行一条医疗查询，例如：

```
My child has a fever of 38.5°C for two days. Should I go to the ER?
Is it safe to take ibuprofen and paracetamol together?
```

---

## 快速开始

```bash
cd Task_Intent_understanding

# 完整流程（数据处理 + 5 轮评估 + 优化）
python run_pipeline.py

# 常用组合
python run_pipeline.py --skip-opt              # 跳过优化分支
python run_pipeline.py --skip-data             # 数据已处理，只跑评估
python run_pipeline.py --skip-eval --skip-opt  # 只跑数据处理

# 性能调整
python run_pipeline.py --num-pool 8 --num-eval-runs 3
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--skip-data` | False | 跳过数据处理步骤（01–04） |
| `--skip-eval` | False | 跳过评估步骤 |
| `--skip-opt` | False | 跳过优化步骤 |
| `--num-eval-runs` | 5 | 独立评估轮次（用于计算 p_true） |
| `--num-pool` | 4 | 并发线程数 |

---

## 分步说明

### 数据处理

#### Step 01 — 查询意图解读

```bash
python Data_processing/01_interpretation.py --dir Data_processing/Processed_data
```

读取 `selected_query.txt`，调用 LLM 分析每条查询，输出：
- `task_type`：任务类型（Diagnosis / Treatment / Prevention / Consultation）
- `interpretation`：关键要素与理想回答结构
- `scoring_questions`：评分问题列表（每条查询至少 5 个）

输出：`Processed_data/01_gemini_interpretation.json`

#### Step 02 — 格式转换

```bash
python Data_processing/02_data_change.py
```

将 Step 01 输出整理为标准结构，添加 `main_id`、`point_id` 等字段。

输出：`Processed_data/02_gemini_data.json`

#### Step 03 — 评分维度分类

```bash
python Data_processing/03_classification.py --dir Processed_data --num_pool 4
```

将每条评分问题分类到以下 10 个医疗评估维度之一：

| 编码 | 维度 | 说明 |
|------|------|------|
| A | Completeness | 回答是否涵盖所有相关要素 |
| B | Diagnostic Relevance | 是否聚焦核心病理机制 |
| C | Treatment Appropriateness | 治疗建议是否合理安全 |
| D | Clinical Safety | 是否识别紧急情况 |
| E | Differentiation | 是否区分不同可能原因 |
| F | Actionability | 建议是否具体可执行 |
| G | Tone/Empathy/Clarity | 语言是否专业、清晰 |
| H | Caution/Uncertainty | 是否说明不确定性边界 |
| I | Pathophysiology | 是否解释背后机制 |
| J | Personalization | 是否结合患者情况个性化 |

输出：`Processed_data/03_gemini_category_data.json`

#### Step 04 — 目标模型生成回答

```bash
python Data_processing/04_generation.py --dir Processed_data --worker 4
```

对每条原始查询调用目标模型生成回答。

输出：`Processed_data/04_gemini_generation.jsonl`

---

### 评估流程

#### Step 1 — 逐条评分（多轮）

```bash
# 运行 N 次，每次产生独立评估文件
python Evaluation_processing/step1.py --run_id 1
python Evaluation_processing/step1.py --run_id 2
# ...
```

LLM 作为裁判，对每个评分问题判断目标模型的回答是否满足要求（Yes/No）。

输出：`evaluation_data/step1_referee_gemini{1..N}.jsonl`

#### Step 1b — 聚合 p_true

```bash
python Evaluation_processing/step1_p_true.py --eval_dir evaluation_data --num_runs 5
```

统计多轮评估中 `Yes` 的比例，得到每个评分点的 `p_true`。

输出：`evaluation_data/step1_referee_gemini_integration.json`

#### Step 2 — 语义聚类

```bash
python Evaluation_processing/step2_entailment.py
```

通过双向 NLI 蕴含判断，将多次评估的文本解释聚类，得到语义 ID。

- 双向蕴含 → 同一语义簇
- 存在矛盾 → 不同语义簇

输出：`evaluation_data/step2_cluster_gemini_result.json`

#### Step 3 — 语义熵计算

```bash
python Evaluation_processing/step3_bincount.py
```

基于语义簇的频率分布，计算 Shannon 熵：

$$H = -\sum_c p_c \log p_c$$

熵越高，表示模型在该评分点的不确定性越大。

输出：`evaluation_precision/step3_gemini_entropy.json`

#### Step 4 — 结果汇总

```bash
python Evaluation_processing/step4.py
```

将所有中间结果合并，生成最终评估文件。

输出：`evaluation_data/step4_gemini.json`

**最终结果结构：**

```json
{
  "main_id": 0,
  "instruction": "原始查询",
  "task_types": "Diagnosis",
  "constraint_dimensions": ["A", "B", "F"],
  "scoring_questions": [
    {
      "point_id": 0,
      "question": "评分问题文本",
      "constraint_dimensions": ["A"],
      "model_output": {
        "output": "模型回答",
        "generations": ["评估解释1", "评估解释2", ...],
        "p_true": 0.8,
        "semantic_ids": [0, 0, 1, 0, 1],
        "semantic_entropy": 0.673
      }
    }
  ]
}
```

---

### 优化流程

优化流程独立于评估，目的是通过为查询添加语言约束，生成更精确的受控查询。

#### Opt Step 1 — 约束生成

```bash
python Optimization/01_contraints_generate.py \
  --inputdir Data_processing/Processed_data \
  --outputdir Optimization/data_precision
```

使用 LLM 为每条查询提取形容词（Adjectives）、副词（Adverbs）、介词（Prepositions）约束。

输出：`data_precision/01_gemini_constraints.json`

#### Opt Step 2 — 约束重组

```bash
python Optimization/02_recombination.py --dir Optimization/data_precision
```

将提取的约束自然地嵌入原始查询，生成更具体精确的重写版本。

输出：`data_precision/02_gemini_recombination.json`

---

## 输出文件说明

| 文件 | 位置 | 说明 |
|------|------|------|
| `01_gemini_interpretation.json` | `Processed_data/` | LLM 意图解读结果 |
| `02_gemini_data.json` | `Processed_data/` | 格式化后的评分数据 |
| `03_gemini_category_data.json` | `Processed_data/` | 带维度分类的评分数据 |
| `04_gemini_generation.jsonl` | `Processed_data/` | 目标模型生成的回答 |
| `step1_referee_gemini{N}.jsonl` | `evaluation_data/` | 第 N 轮逐条评分结果 |
| `step1_referee_gemini_integration.json` | `evaluation_data/` | 聚合后的 p_true |
| `step2_cluster_gemini_result.json` | `evaluation_data/` | 语义聚类结果 |
| `step3_gemini_entropy.json` | `evaluation_precision/` | 语义熵 |
| `step4_gemini.json` | `evaluation_data/` | **最终汇总结果** |
| `01_gemini_constraints.json` | `data_precision/` | 约束生成结果 |
| `02_gemini_recombination.json` | `data_precision/` | 约束重组结果 |
