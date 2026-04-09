# Task Intent Understanding

用于对比 **LLM 生成评分标准** 与 **人工评分标准** 在模型回答评估上差异的实验代码库，旨在量化大模型与人类在医疗问答任务意图理解上的偏差。

---

## 研究问题

给定同一批医疗查询和同一个目标模型的回答，分别用两套评分标准进行评估：

| | LLM 评分标准 | 人工评分标准 |
|---|---|---|
| **来源** | 由 LLM 自动解读查询意图并生成 | 由人工标注 |
| **格式转换** | Step 02 | Step 02（同） |
| **维度分类** | Step 03 | Step 03（同） |
| **目标模型生成回答** | Step 04（共用） | Step 04（共用） |
| **评估流程** | Step 1–4（共用） | Step 1–4（共用） |
| **优化分支** | 有 | 无 |

通过对比两套标准下的 `p_true`、语义熵等指标，分析 LLM 与人类对任务意图的理解偏差。

---

## 目录

- [整体流程](#整体流程)
- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [分步说明](#分步说明)
  - [数据处理](#数据处理)
  - [评估流程](#评估流程)
  - [优化流程（仅 LLM 分支）](#优化流程仅-llm-分支)
- [输出文件说明](#输出文件说明)

---

## 整体流程

```
原始查询 (selected_query.txt)
        │
        ├─────────────────────────┬──────────────────────────────────┐
        │                         │                                  │
        ▼                         ▼                                  │
  ┌─────────────┐          ┌─────────────┐                          │
  │ LLM 分支    │          │ 人工分支    │                          │
  │             │          │             │                          │
  │ [Step 01]   │          │  人工标注   │                          │
  │ LLM 解读    │          │  评分标准   │                          │
  │ 意图，生成  │          │  （已有）   │                          │
  │ 评分问题    │          │             │                          │
  └──────┬──────┘          └──────┬──────┘                          │
         │                        │                                  │
         ▼                        ▼                                  │
  [Step 02] 格式转换       [Step 02] 格式转换                       │
         │                        │                                  │
         ▼                        ▼                                  │
  [Step 03] 评分维度分类   [Step 03] 评分维度分类                   │
         │                        │                                  │
         └────────────┬───────────┘                                  │
                      │ 共用同一套目标模型回答                       │
                      ▼                                              │
             [Step 04] 目标模型生成回答                              │
                      │                                              │
                      ▼                                              │
             [Step 1 × N] LLM 裁判逐条评分 (Yes/No)                 │
                      │                                              │
                      ▼                                              │
             [Step 1b] 聚合多轮，计算 p_true                        │
                      │                                              │
                      ▼                                              │
             [Step 2] 语义聚类（NLI 蕴含关系）                      │
                      │                                              │
                      ▼                                              │
             [Step 3] 计算语义熵                                     │
                      │                                              │
                      ▼                                              │
             [Step 4] 汇总结果                                       │
                      │                                              │
                      ▼                                              │
         ┌────────────────────────┐                                  │
         │  对比分析              │                                  │
         │  LLM 标准 vs 人工标准  │                                  │
         │  p_true / 语义熵差异   │         [优化分支（仅 LLM）]◄───┘
         └────────────────────────┘         约束生成 → 重组查询
```

---

## 目录结构

```
Task_Intent_understanding/
├── run_pipeline.py                  # 一键启动脚本
├── utils.py                         # 公共工具（读取 Prompt 模板）
│
├── Data_processing/                 # 数据处理（步骤 01–04，两条分支共用）
│   ├── 01_interpretation.py         # Step 01：LLM 解读查询意图，生成评分问题
│   ├── 02_data_change.py            # Step 02：格式转换（LLM 分支 / 人工分支各跑一次）
│   ├── 03_classification.py         # Step 03：评分维度分类（A–J）
│   ├── 04_generation.py             # Step 04：目标模型生成回答（两条分支共用）
│   ├── prompts/
│   │   └── 01_interpretation.txt
│   └── Processed_data/              # 中间数据目录（自动创建）
│
├── Evaluation_processing/           # 评估流程（两条分支共用同一套代码）
│   ├── step1.py                     # LLM 裁判逐条评分
│   ├── step1_p_true.py              # 多轮聚合，计算 p_true
│   ├── step2_entailment.py          # 语义聚类
│   ├── step3_bincount.py            # 语义熵计算
│   ├── step4.py                     # 结果汇总
│   ├── prompts/
│   │   └── RAL_evaluator.py         # 评估 Prompt
│   └── evaluation_data/             # 评估结果目录（自动创建）
│
└── Optimization/                    # 优化分支（仅 LLM 分支使用）
    ├── 01_contraints_generate.py    # 语言约束生成
    ├── 02_recombination.py          # 约束重组查询
    ├── prompts/
    │   ├── 01_constraints.txt
    │   └── 02_recombination.txt
    └── data_precision/              # 优化结果目录（自动创建）
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
| `"Model_Name"` | 模型名称，如 `gpt-4o`、`gemini-pro` |

涉及文件：`01_interpretation.py`、`03_classification.py`、`04_generation.py`、`step1.py`、`step2_entailment.py`、`01_contraints_generate.py`、`02_recombination.py`

**准备输入数据：**

```
Data_processing/Processed_data/selected_query.txt    # 原始查询，每行一条
```

两条分支的起点不同：

| 分支 | 起点 |
|------|------|
| LLM 分支 | `selected_query.txt` → Step 01 自动生成评分问题 |
| 人工分支 | 人工标注的评分标准文件 → 直接进入 Step 02 |

---

## 快速开始

```bash
cd Task_Intent_understanding

# 完整跑 LLM 分支（含优化）
python run_pipeline.py

# 只跑数据处理（LLM 生成评分标准 + 格式化 + 分类 + 模型回答生成）
python run_pipeline.py --skip-eval --skip-opt

# 人工分支：数据已格式化，只跑分类 + 评估
# 先单独运行 Step 03：
python Data_processing/03_classification.py \
  --dir Data_processing/Processed_data \
  --output_path Data_processing/Processed_data
# 再跑评估：
python run_pipeline.py --skip-data --skip-opt

# 调整并发数和评估轮次
python run_pipeline.py --num-pool 8 --num-eval-runs 3
```

**`run_pipeline.py` 参数说明：**

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

两条分支在数据处理阶段的差异仅在 **Step 01**：LLM 分支由模型自动生成评分问题，人工分支直接使用人工标注文件。**Step 02 起两条分支流程完全一致**。

#### Step 01 — LLM 解读查询意图（仅 LLM 分支）

```bash
python Data_processing/01_interpretation.py --dir Data_processing/Processed_data
```

调用 LLM 分析每条查询，从模型视角描述理想回答应包含的要素，并生成对应评分问题（每条查询至少 5 个）。

输出：`Processed_data/01_gemini_interpretation.json`

> **人工分支**：跳过此步，直接使用人工标注的评分标准文件，格式与此输出相同。

#### Step 02 — 格式转换

```bash
python Data_processing/02_data_change.py
```

将评分标准整理为标准结构，添加 `main_id`、`point_id` 等字段，供后续步骤统一处理。

输出：`Processed_data/02_gemini_data.json`

#### Step 03 — 评分维度分类

```bash
python Data_processing/03_classification.py \
  --dir Data_processing/Processed_data \
  --output_path Data_processing/Processed_data \
  --num_pool 4
```

将每条评分问题归入以下 10 个医疗评估维度之一：

| 编码 | 维度 | 说明 |
|------|------|------|
| A | Completeness | 是否涵盖所有相关病因、症状或病史 |
| B | Diagnostic Relevance | 是否聚焦核心病理机制或临床表现 |
| C | Treatment Appropriateness | 治疗建议是否合理安全 |
| D | Clinical Safety | 是否识别紧急情况并建议就医 |
| E | Differentiation | 是否区分不同可能原因（如病毒性 vs 细菌性） |
| F | Actionability | 建议是否具体可执行或个性化 |
| G | Tone/Empathy/Clarity | 语言是否专业、冷静、不引发恐慌 |
| H | Caution/Uncertainty | 是否说明不确定性边界，避免过度自我诊断 |
| I | Pathophysiology | 是否解释背后的病理机制 |
| J | Personalization | 是否结合患者情境给出实际建议 |

输出：`Processed_data/03_gemini_category_data.json`

#### Step 04 — 目标模型生成回答（两条分支共用）

```bash
python Data_processing/04_generation.py \
  --dir Data_processing/Processed_data \
  --output_dir Data_processing/Processed_data \
  --worker 4
```

对每条原始查询调用目标模型生成回答。**两条分支评估的是同一批模型回答**，差异仅来自评分标准本身。

输出：`Processed_data/04_gemini_generation.jsonl`

---

### 评估流程

以下步骤对 LLM 分支和人工分支完全相同，分别运行两次即可得到两套评估结果用于对比。

#### Step 1 — 逐条评分（多轮独立运行）

```bash
python Evaluation_processing/step1.py --run_id 1
python Evaluation_processing/step1.py --run_id 2
# ... 默认共 5 轮
```

LLM 裁判对每个评分问题独立判断目标模型的回答是否满足要求（Yes / No），并给出说明。多轮独立运行用于估计评估本身的不确定性。

输出：`evaluation_data/step1_referee_gemini{1..N}.jsonl`

#### Step 1b — 聚合 p_true

```bash
python Evaluation_processing/step1_p_true.py \
  --eval_dir Evaluation_processing/evaluation_data \
  --num_runs 5
```

统计多轮评估中 Yes 的比例，得到每个评分点的满足概率 `p_true`。

输出：`evaluation_data/step1_referee_gemini_integration.json`

#### Step 2 — 语义聚类

```bash
python Evaluation_processing/step2_entailment.py
```

通过双向 NLI 蕴含判断，将多轮评估的文本解释聚类，判断不同轮次的说明是否表达相同语义：

- 双向蕴含 → 同一语义簇
- 存在矛盾或中立 → 不同语义簇

输出：`evaluation_data/step2_cluster_gemini_result.json`

#### Step 3 — 语义熵计算

```bash
python Evaluation_processing/step3_bincount.py
```

基于语义簇的频率分布，计算 Shannon 熵：

$$H = -\sum_c p_c \log p_c$$

熵越高，代表裁判在该评分点的判断越不一致，即该维度的评估不确定性越大。通过对比 LLM 标准与人工标准下的语义熵分布，可以分析两者在哪些评估维度上存在系统性分歧。

输出：`evaluation_precision/step3_gemini_entropy.json`

#### Step 4 — 结果汇总

```bash
python Evaluation_processing/step4.py
```

将所有中间结果合并为最终文件，包含每条查询、每个评分点的完整评估信息。

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
        "output": "目标模型回答",
        "generations": ["第1轮裁判说明", "第2轮裁判说明", "..."],
        "p_true": 0.8,
        "semantic_ids": [0, 0, 1, 0, 1],
        "semantic_entropy": 0.673
      }
    }
  ]
}
```

---

### 优化流程（仅 LLM 分支）

优化流程独立于评估主流程，目的是通过为查询注入语言约束，生成更精确的受控查询版本，用于进一步分析约束对评估结果的影响。**人工分支无此步骤。**

#### Opt Step 1 — 约束生成

```bash
python Optimization/01_contraints_generate.py \
  --inputdir Data_processing/Processed_data \
  --outputdir Optimization/data_precision
```

使用 LLM 为每条查询提取形容词（Adjectives）、副词（Adverbs）、介词（Prepositions）三类语言约束。

输出：`data_precision/01_gemini_constraints.json`

#### Opt Step 2 — 约束重组

```bash
python Optimization/02_recombination.py \
  --dir Optimization/data_precision \
  --worker 4
```

将提取的约束自然地嵌入原始查询，生成表述更精确、意图更明确的重写版本。

输出：`data_precision/02_gemini_recombination.json`

---

## 输出文件说明

### LLM 分支

| 文件 | 位置 | 说明 |
|------|------|------|
| `01_gemini_interpretation.json` | `Processed_data/` | LLM 生成的意图解读与评分问题 |
| `02_gemini_data.json` | `Processed_data/` | 格式化后的评分数据 |
| `03_gemini_category_data.json` | `Processed_data/` | 含维度分类的评分数据 |
| `04_gemini_generation.jsonl` | `Processed_data/` | 目标模型回答（两分支共用） |
| `step1_referee_gemini{N}.jsonl` | `evaluation_data/` | 第 N 轮逐条评分结果 |
| `step1_referee_gemini_integration.json` | `evaluation_data/` | 聚合后的 p_true |
| `step2_cluster_gemini_result.json` | `evaluation_data/` | 语义聚类结果 |
| `step3_gemini_entropy.json` | `evaluation_precision/` | 语义熵 |
| `step4_gemini.json` | `evaluation_data/` | **LLM 分支最终结果** |
| `01_gemini_constraints.json` | `data_precision/` | 语言约束生成结果 |
| `02_gemini_recombination.json` | `data_precision/` | 约束重组结果 |

### 人工分支

| 文件 | 位置 | 说明 |
|------|------|------|
| 人工标注评分标准文件 | `Processed_data/` | 替代 Step 01 输出，格式相同 |
| `02_human_data.json` | `Processed_data/` | 格式化后的人工评分数据 |
| `03_human_category_data.json` | `Processed_data/` | 含维度分类的人工评分数据 |
| `step4_human.json` | `evaluation_data/` | **人工分支最终结果** |

> 建议对两条分支的输出文件使用不同前缀（`gemini_` vs `human_`）以便区分，再用 `step4_gemini.json` 与 `step4_human.json` 进行对比分析。
