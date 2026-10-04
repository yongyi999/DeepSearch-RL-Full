<div align="center">

# DeepSearch-RL

### 基于 Tool-Agentic RL 的多跳搜索智能体

**Qwen3-8B · veRL · SGLang · GRPO · Agentic RL**

一个**不 fork veRL**、开箱即用的多轮搜索强化学习工程：模型在 rollout 中通过 `<search>` / `<open>` 自主检索网页，
以「证据充分度驱动的分层奖励」联合优化答案正确性、证据充分性、格式完整性与工具效率。
在实验实际使用的 **48 题 Hard Multi-hop 验证集**（HotpotQA 20 + 2WikiMultihopQA 16 + MuSiQue 12）上，
Qwen3-8B 的综合奖励（val plateau score）从 **约 0（−0.003）提升到 0.247**，答案得分 r_answer 从 **0.104 提升到 0.260**，
格式完整率从约 **19% 提升到 98%**，重复工具调用率从 **23.8% 降至 0.5%**。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%20cu124-orange.svg)]()
[![veRL](https://img.shields.io/badge/veRL-v0.6.0-red.svg)]()

</div>

---

## 目录

- [一、项目简介](#一项目简介)
- [二、核心特性](#二核心特性)
- [三、效果指标（实验实测）](#三效果指标实验实测)
- [四、奖励函数设置](#四奖励函数设置)
- [五、系统架构](#五系统架构)
- [六、目录结构](#六目录结构)
- [七、环境搭建（AutoDL 6×4090）](#七环境搭建autodl-64090)
- [八、数据准备](#八数据准备)
- [九、启动检索服务](#九启动检索服务)
- [十、配置 Judge（默认 DeepSeek）](#十配置-judge默认-deepseek)
- [十一、SwanLab 登录](#十一swanlab-登录)
- [十二、启动训练](#十二启动训练)
- [十三、评估](#十三评估)
- [十四、显存与调参](#十四显存与调参)
- [十五、常见问题 FAQ](#十五常见问题-faq)
- [十六、参考与致谢](#十六参考与致谢)

---

## 一、项目简介

DeepSearch-RL 让一个基座大模型（Qwen3-8B）通过**强化学习**学会「带着工具做多跳搜索」：
面对一个需要多步推理的复杂问题，模型需要自己决定**何时搜索、搜什么、打开哪个网页、是否已经拿到足够证据、何时作答**。

与直接用 RAG 或让模型一次性输出答案不同，本工程的关键是把 **Search/Open 工具交互**纳入 RL 训练闭环：

- rollout 阶段模型与工具进行**多轮异步交互**（veRL `ToolAgentLoop` 状态机 + 自定义 `search_xml` 解析器）；
- 奖励由**远程 Judge（DeepSeek Chat，OpenAI 兼容 API）**构造，包含答案正确性与证据充分性；
- 奖励是「**证据充分度驱动的分层奖励**」：猜对但没有证据只能拿到低分（抑制 reward hacking），
  证据充分前的有效探索不被惩罚（缓解 under-search），重复调用与证据充分后的冗余调用被惩罚（抑制 over-search）。

整个工程只写代码、不绑定某个固定搜索服务：在线搜索支持 Serper / SerpAPI / Bing / Brave / Tavily，
无 key 时可回退免费 DuckDuckGo；工具链路带持久缓存、API-key 轮换、失败重试与异常分类，保障长时间在线 rollout 稳定。

---

## 二、核心特性

- **veRL 原生 Agentic RL**：基于 veRL v0.6.0 搭建多轮 Search/Open 工具交互链路，异步 `ToolAgentLoop` + GRPO 策略优化，无需 fork 框架。
- **证据充分度驱动的分层奖励**：联合优化答案正确性、证据充分性、格式完整性、工具效率；
  保护有效探索、惩罚重复调用及证据充分后的冗余调用，缓解 under-search 与 reward hacking。
- **远程 Judge（DeepSeek Chat）**：Answer Judge / Evidence Judge 统一由 DeepSeek Chat（`deepseek-chat`）承担，训练奖励通过 OpenAI 兼容 API 异步调用，与训练解耦；也可通过 `JUDGE_BASE_URL` 指向本地 vLLM 服务。
- **工程化工具链路**：SQLite 持久缓存（多进程 WAL）、API-key 轮换池（限流冷却）、指数退避重试、异常显式分类。
- **动态 sequence balancing**：veRL 动态 batch（`use_dynamic_bsz`），将多卡 token 负载不均衡从 **约 17.5% 降至 0.003%**。
- **SwanLab 全链路可观测**：loss / KL / 奖励分量 / 工具调用 / 错误率 / 轨迹长度，以及完整生成样例。
- **固定验证集与一键脚本**：48 题 Hard Multi-hop 验证集（HotpotQA 20 + 2Wiki 16 + MuSiQue 12），训练中每 3 步自动验证，输出综合奖励、答案得分、证据分、格式分、重复调用率与平均 Search/Open 次数。

---

## 三、效果指标（实验实测）

评测集：**48 题 Hard Multi-hop 验证集**（HotpotQA 20 + 2WikiMultihopQA 16 + MuSiQue 12，`prepare_train.py --fast` 产出，固定 seed=42，与训练集按 id 不重叠）。
训练配置 `val_before_train=True`、`test_freq=3`：开训前先评一次基线，之后每 3 步在同一批题上验证；裁判为在线 DeepSeek Chat（Answer / Evidence Judge），全程可用、无降级。

**总体结果（基线 = 开训前 step 0；训练后 = 验证集综合奖励最高的 step 24 检查点）：**

| 指标 | 基线 Qwen3-8B | 训练后（step 24） | 变化 |
|---|---|---|---|
| **综合奖励（val plateau score）** | −0.003 | **0.247** | **+0.250** |
| **答案得分 r_answer（EM/F1 + Answer Judge，0–1）** | 0.104 | **0.260** | **+0.156（约 2.5×）** |
| **格式得分 r_format（满分 0.2）** | 0.038 | **0.196** | +0.158（格式完整率约 19% → 98%） |
| **证据分 r_evidence（Evidence Judge，0–1）** | 0.115 | **0.000** | −0.115（见下方说明） |
| **重复工具调用率 Duplicate Call Rate** | 23.8% | **0.5%** | **−23.2 pp** |
| **平均 Search 次数 / 题** | 3.96 | **2.50** | −1.46 |
| **平均 Open 次数 / 题** | 1.02 | **1.67** | +0.65 |

**答案得分 r_answer 按数据源分组（基线 → 训练后）：**

| 数据源（n） | 基线 | 训练后 | 重复调用率 基线 → 训练后 |
|---|---|---|---|
| HotpotQA（20） | 0.10 | **0.20** | 17.1% → **0.0%** |
| 2WikiMultihopQA（16） | 0.19 | **0.31** | 32.4% → **1.6%** |
| MuSiQue（12） | 0.00 | **0.29** | 23.3% → **0.0%** |

**口径与说明（如实记录）：**

- 本次为在线 RL 快训：训练集 1728 条难多跳 prompt（HotpotQA 720 + 2Wiki 600 + MuSiQue 408），`train_batch_size=48`、`rollout.n=4`，计划 36 步；实际在 step 29 收到 SIGTERM 终止，**step 24 为验证综合奖励最高、并被保留的最佳检查点**（step 27 综合奖励 0.236，略低）。
- 模型主要学到三件事：**稳定输出格式完整的 `<answer>`、答案得分约提升到 2.5 倍、显著抑制重复/冗余工具调用**；同时轨迹由早期的 10+ 轮冗长检索收敛为精炼的约 2 轮（训练曲线 `num_turns` 10–12 → 2）。
- **证据分 r_evidence 后期为 0 是实测结果而非报错**：裁判服务全程在线；精炼的短轨迹虽更答对、更省工具，却被 Evidence Judge 判为证据不足，体现「简洁/工具效率」与「证据门控」之间的真实张力。因此本项目**不把"证据充分度提升"作为已验证结论**，该子分的口径需在固定裁判、固定轨迹长度后重新评测。
- 结果会随搜索后端、数据配比、训练步数波动；离线复现可用 `python -m deepsearch_rl.eval.evaluate`（默认读取与该验证集一一对应的 `data/eval_val_48.jsonl`）。

---

## 四、奖励函数设置

奖励对**整条轨迹**（而非单个 token）打分，由四个分量组成；权重集中在 `deepsearch_rl/rewards/hierarchical.py` 的 `RewardWeights`，默认值即本次实验冻结值，集中管理便于调参。

| 分量 | 取值范围 | 计算方式 |
|---|---|---|
| **R_format（格式）** | 0 / 0.1 / 0.2 | 有 `<answer>` 且无 malformed 片段 = 0.2；有 answer 但存在 malformed = 0.1；无 answer = 0 |
| **R_answer（答案）** | 0 ~ 1 | 规则先行：归一化后 EM 命中或 token-F1 ≥ 0.9 给 1，否则取 F1；规则未命中再请 Answer Judge：correct = 1 / partial = 0.5 / wrong = 0 |
| **R_evidence（证据）** | 0 ~ 1 | Evidence Judge 对轨迹收集的证据输出连续分，**0.6 判为充分** |
| **R_tool（工具效率）** | −0.5 ~ 0.1 | 重复调用每次 −0.15（封顶 −0.3）；证据充分后冗余调用每次 −0.1（封顶 −0.2）；无重复且高效完成 + 0.1 |

**总分合成（证据充分度门控答案奖励）：**

```
R_total = R_format
        + R_answer × (0.2 + 0.8 × R_evidence)   # 证据门控答案奖励
        + 0.3 × R_evidence × R_answer           # 正确且证据充分的联合奖励
        + R_tool − undersearch_penalty          # undersearch_penalty = 0.5
```

**设计要点：**

- **答案门控**：答案奖励的 80% 由证据充分度决定——猜对但无证据只能拿到 0.2 倍基础分，抑制 reward hacking；
- **联合奖励**：正确且证据充分时额外 +0.3（R_evidence × R_answer），引导模型「先取证、再作答」；
- **工具效率**：证据充分前的有效探索不被惩罚（缓解 under-search），重复调用与证据充分后的冗余调用被惩罚（抑制 over-search）；
- **反纯猜**：一次工具都不调用就作答，答案分清零并额外扣 0.5，总分变负，避免「不搜直接猜对」压过「去搜索」的轨迹；
- **降级兜底**：Judge 不可达时自动退回规则分（EM/F1 + 证据标题命中代理，命中给 0.6），任何异常都不中断训练。

> **量级参考**：正确且证据充分的轨迹约 1.6 分；搜过但证据不足的正确答案只能拿到格式分与门控下限。

---

## 五、系统架构

```
                         ┌──────────────────────────────────────────────┐
                         │  训练进程（Ray + veRL，6×RTX 4090）               │
   parquet 训练集  ────▶ │  GRPO RayPPOTrainer                          │
   (48 prompts/step)     │    │                                         │
                         │    ▼                                         │
                         │  DeepSearchAgentLoop（deepsearch_agent）       │
                         │    ├─ SGLang 生成（TITO token 层拼接）          │
                         │    ├─ search_xml_parser 解析 <search>/<open>  │
                         │    └─ veRL 工具 wrapper（search/open）          │
                         │         │ HTTP                                  │
                         └─────────┼────────────────────────────────────┘
                                   ▼
                  ┌──────────────────────────────────┐
                  │ 检索服务（独立 FastAPI 进程，:8000）│
                  │  /retrieve   /open   /health      │
                  │  Serper/SerpAPI/Bing/Brave/Tavily │
                  │  持久缓存 + key 轮换 + 重试 + 分类  │
                  └──────────────────────────────────┘
                                   ▲
  远程 Judge（DeepSeek Chat API）─┘  judge_client（异步 OpenAI 兼容协议）
  · Answer Judge   · Evidence Judge
```

**一次训练 step 的流程：**

1. 从训练 parquet 采样 48 个 prompt，每个 prompt 采样 4 条 → 并行生成 **192 条多轮交互轨迹**；
2. 每条轨迹在 SGLang 上生成，`search_xml_parser` 解析 `<search>/<open>` 标签；
3. 工具 wrapper 通过 HTTP 调用检索服务（命中缓存则直接返回），观测以 `<observation>` 拼回，进入下一轮；
4. 模型给出 `<answer>` 或达到轮数上限后结束；
5. reward 函数对整条轨迹：EM/F1 判答案（必要时 Answer Judge）、Evidence Judge 判证据充分度，
   按分层公式合成奖励；
6. GRPO 用组内相对优势做策略更新；工具返回 token 通过 `response_mask=0` 不参与策略梯度。

> **TITO（Token-In-Token-Out）**：多轮拼接在 token 层完成，绝不 decode 成文字再重新 encode，
> 否则会导致轨迹脱离策略分布、PPO 不收敛。这部分由 veRL 框架保证，我们只在配置层启用。

---

## 六、目录结构

```
DeepSearch-RL/
├── README.md                     # 本文档
├── requirements.txt              # 通用依赖（torch/verl/sglang 需特殊安装）
├── setup.py
├── run.sh                        # 一键编排（retrieval/judge/train/eval/install/model）
├── configs/
│   ├── grpo_qwen3_8b_6x4090.yaml # 主训练配置（默认 6×RTX 4090）
│   ├── tools_search_xml.yaml     # veRL 工具注册（search/open wrapper）
│   └── judge.yaml                # Judge 配置
├── scripts/
│   ├── install_autodl.sh         # AutoDL 一键装环境
│   ├── download_model.sh         # 下载 Qwen3-8B（ModelScope）
│   ├── download_data.sh          # 下载并预处理数据
│   ├── start_retrieval.sh        # 启动检索服务
│   ├── start_judge.sh            # 启动 Judge
│   ├── train.sh                  # 启动训练
│   └── eval.sh                   # 启动评估
├── data/
│   ├── download_data.py          # 下载 5 个数据源并归一
│   ├── prepare_train.py          # 生成 veRL 训练 parquet / fast 子集（并导出 48 题验证集）
│   ├── eval_val_48.jsonl         # 48 题多跳验证集（离线评测默认输入）
│   └── README.md
├── deepsearch_rl/
│   ├── protocol.py               # 工具标签协议（解析/观测/系统提示）
│   ├── tools/                    # 搜索/网页工具、缓存、key 轮换、异常、veRL wrapper
│   ├── retrieval/                # FastAPI 检索服务
│   ├── judge/                    # Judge 提示词、启动器、客户端
│   ├── agent/                    # 解析器、自定义 loop、轨迹分析、独立推理
│   ├── rewards/                  # EM/F1、工具效率、分层奖励
│   ├── train/                    # 训练入口、SwanLab 封装
│   ├── eval/                     # 48 题验证集离线评测
│   └── utils/                    # 配置/日志/随机种子
└── tests/                        # 离线逻辑测试（不触网）
```

---

## 七、环境搭建（AutoDL 6×4090）

### 7.1 租机与镜像

- 在 AutoDL 租用 **6×RTX 4090（24GB，Ada sm_89）**；
- 镜像选择 **Ubuntu 22.04/24.04 + Python 3.12 + CUDA 12.4**；
- 4090 是 Ada 架构，**使用 CUDA 12.4 + PyTorch cu124 车道**即可（无需 cu128；cu128 仅 Blackwell 需要）。

### 7.2 获取工程

```bash
git clone https://github.com/yongyi999/DeepSearch-RL-Full.git
cd DeepSearch-RL
pip install -e .
```

### 7.3 一键安装（推荐）

```bash
bash scripts/install_autodl.sh
```

该脚本会依次完成：PyTorch 2.8.0 cu124 → 通用依赖 → veRL v0.6.0（源码 editable，`[sglang]`）
→ 固定 sglang ≤0.5.19 + flashinfer cu124 → liger-kernel。可重复执行。

### 7.4 手动安装（如需逐步控制）

```bash
# 1) PyTorch 2.8.0 cu124（4090 用 cu124）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 2) 通用依赖
pip install -r requirements.txt

# 3) veRL v0.6.0（必须 pin，main 已迁 CUDA13/torch2.14）
git clone https://github.com/volcengine/verl.git ~/verl
cd ~/verl && git checkout v0.6.0
pip install -e ".[sglang]"
cd -

# 4) sglang 固定 cu12 车道（0.5.19 是最后一个 CUDA12 版本，勿升 0.5.20+）
pip install "sglang>=0.4.6.post1,<0.5.20"
pip install flashinfer_python \
  --find-links https://flashinfer.ai/whl/cu124/torch2.8/flashinfer-python

# 5) 免编译 kernel（4090 上替代 flash-attn）
pip install "liger-kernel>=0.8.2"
```

> **关于 flash-attn**：FlashAttention-3/4 面向数据中心 Blackwell（sm_100，带 TMEM），**在 4090（sm_89）上不适用**；
> 4090 用 FlashAttention-2（`flash-attn>=2.6.1`，含 sm_89 kernel）即可，需本机 nvcc=12.4。
> 本工程默认用 **liger-kernel + SDPA**（训练）与 SGLang 的 `triton/flashinfer` 后端（推理），免编译。

### 7.5 关键依赖版本一览

| 包 | 版本 | 备注 |
|---|---|---|
| Python | 3.12 | |
| CUDA | 12.4 | 4090 用 cu124 |
| torch / torchvision / torchaudio | 2.8.0 / 0.23.0 / 2.8.0 | cu124 index |
| verl | **v0.6.0** | 源码 editable，勿用 main |
| sglang | **≥0.4.6, <0.5.20** | 最后 CUDA12 车道 |
| flashinfer | cu124/torch2.8 | 对应 find-links |
| DeepSeek API（Judge） | deepseek-chat | 默认 Answer/Evidence Judge，OpenAI 兼容 |
| vllm（可选本地 Judge） | 随 verl v0.6.0 | 仅离线场景，OpenAI 兼容端点 |
| liger-kernel | ≥0.8.2 | 免编译 kernel |
| modelscope | 1.23.1 | 模型/数据下载 |
| ray | ≥2.45, <2.49 | 分布式 |
| swanlab | 0.9.0 | 实验追踪 |

---

## 八、数据准备

### 8.1 数据集介绍

本项目使用四类公开问答数据集，统一经过归一化处理（统一字段、统一 prompt 模板），覆盖**单跳事实问答**到 **4 跳多跳推理**，并包含比较、推断等复合推理类型。

| 数据集 | 类型 | 跳数 | 全量训练 | 全量验证 | fast 训练（本次） | fast 验证（本次） |
|---|---|---|---|---|---|---|
| HotpotQA | 众包多跳问答 | 2 | 90,447 | 200 | 720 | 20 |
| 2WikiMultihopQA | 维基百科结构化多跳 | 2 | 167,454 | 200 | 600 | 16 |
| MuSiQue | 组合式多跳问答 | 2–4 | 19,938 | 200 | 408 | 12 |
| NQ-open | 真实搜索查询 | 1 | 30,000 | 200 | — | — |
| **合计** | | | **307,839** | **800** | **1,728** | **48** |

各数据源特点：

- **HotpotQA**：由众包标注者基于维基百科文档构造，问题天然需要跨两篇文档的"桥接"推理（bridge）或比较（comparison），并标注了支撑事实（supporting facts），用于证据门控；
- **2WikiMultihopQA**：基于维基百科知识图谱半自动生成，问题类型涵盖推断（inference）、比较（comparison）、组合（compositional），每条都有完整的推理链和 supporting titles，噪声小；
- **MuSiQue**：通过将 2–4 个相互依赖的单跳问题（S1→S2→…）组合而成，刻意避免"跳过中间推理也能答对"的捷径，是难度最高的多跳集；
- **NQ-open**：来自真实 Google 搜索查询，单跳即可作答，作为检索/直答能力的基线对照（本次 fast 实验未纳入）。

**为什么用 fast 子集**：GRPO 每条样本需 rollout 多条多轮工具轨迹（本次 `n=4`），全量训练成本极高。fast 子集按 `num_hops` 与问题难度分层抽样，保留多跳推理的核心挑战；本次实验每个训练 step 采样 48 题 × 4 条 = 192 条轨迹。

**离线评测文件** `data/eval_val_48.jsonl` 每行字段：

- `id`：样本唯一编号；`source`：数据源；`split`：train/val；
- `question`：问题原文；`gold_answers`：标准答案（含可接受别名，用于 EM/F1 与 Answer Judge）；
- `num_hops`：推理跳数；`supporting_titles`：支撑证据对应的维基百科标题（用于 Evidence Judge / 召回）。

### 8.2 数据集样本示例

下面是验证集 `data/eval_val_48.jsonl` 中的一条**真实样本**（HotpotQA，2 跳）：

```json
{
  "id": "hotpotqa_val_000",
  "source": "hotpotqa",
  "question": "Who were the two parties fighting in the war where Kajiwara Heima served as karō?",
  "gold_answers": ["Tokugawa shogunate and those seeking to return political power to the Imperial Court"],
  "num_hops": 2,
  "supporting_titles": ["Kajiwara Heima", "Boshin War"],
  "split": "val"
}
```

要答对这道题，模型必须先通过 `Kajiwara Heima` 检索到他作为家老（karō）参与的是**戊辰战争（Boshin War）**，再检索这场战争的交战双方——是典型的"先定位实体、再查询关系"的 2 跳桥接推理。

### 8.3 下载模型（Qwen3-8B，ModelScope）

```bash
bash scripts/download_model.sh
# 下载完成后：
export MODEL_PATH=$HOME/models/Qwen3-8B
```

等价的手动下载：

```bash
modelscope download --model Qwen/Qwen3-8B --local_dir $HOME/models/Qwen3-8B
# 或
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B'))"
```

### 8.4 下载并预处理数据

```bash
# 国内（ModelScope 优先，HF 自动走 hf-mirror 镜像）
bash scripts/download_data.sh
```

该脚本依次执行：

1. **下载**：HotpotQA（fullwiki）、2WikiMultihopQA、MuSiQue、NQ-open、Bamboogle，归一为统一中间 jsonl；
2. **生成全量训练 parquet**：`data/processed/train/`、`data/processed/val/`；
3. **生成 fast 子集**：`data/processed/fast/train/`（1728 条难多跳）与 `data/processed/fast/val/`（48 题验证集），并导出离线评测输入 `data/eval_val_48.jsonl`。

单独执行各步：

```bash
python data/download_data.py --out_dir data/raw --nq_limit 30000
python data/prepare_train.py --raw_dir data/raw --out_dir data/processed
python data/prepare_train.py --raw_dir data/raw --fast --seed 42
```

**本次实验口径**：fast 训练集 1728 条（HotpotQA 720 + 2Wiki 600 + MuSiQue 408），验证集 48 题（20/16/12）；NQ/Bamboogle 已下载归一，但未进入本次 fast 训练与验证。

训练 parquet 每行字段：

```json
{
  "data_source": "nq|hotpotqa|2wiki|musique",
  "prompt": [{"role": "system", "content": "<系统提示>"}, {"role": "user", "content": "<问题>"}],
  "ability": "multi-hop-search",
  "agent_name": "deepsearch_agent",
  "reward_model": {"style": "rule", "ground_truth": {"target": ["答案", "别名..."]}},
  "extra_info": {"split": "train", "index": 0, "num_hops": 2}
}
```

> 数据集来源：HotpotQA `hotpotqa/hotpot_qa`、2WikiMultihopQA `voidful/2WikiMultihopQA`、
> MuSiQue `dgslibisey/MuSiQue`、NQ-open `google-research-datasets/nq_open`、Bamboogle（ofirpress/self-ask）。

---

## 九、启动检索服务

**终端 A（常驻）**：

```bash
# 有付费 key（推荐，稳定）：以 Serper 为例
export SERPER_API_KEYS="key1,key2"      # 多个 key 逗号分隔，自动轮换
bash scripts/start_retrieval.sh

# 或使用其它后端
SEARCH_BACKEND=serpapi SERPAPI_API_KEYS="key" bash scripts/start_retrieval.sh
SEARCH_BACKEND=bing    BING_API_KEYS="key"    bash scripts/start_retrieval.sh
SEARCH_BACKEND=tavily  TAVILY_API_KEYS="key"  bash scripts/start_retrieval.sh

# 无 key：自动/显式回退免费 DuckDuckGo（仅建议调试）
SEARCH_BACKEND=ddg bash scripts/start_retrieval.sh
```

服务默认监听 `http://0.0.0.0:8000`，提供：

| 接口 | 入参 | 出参 |
|---|---|---|
| `POST /retrieve` | `{"queries":[...], "topk":5, "return_scores":true}` | `{"result": [[{"document":{"title","url","contents"},"score"}]]}` |
| `POST /open` | `{"url":"..."}` | `{"url","contents","ok"}` |
| `GET /health` | — | `{"status":"ok","cache":{...},"keys_available":n}` |

可用环境变量：`RETRIEVAL_PORT`（默认 8000）、`RETRIEVAL_CONCURRENCY`（默认 120）、
`CACHE_DB_PATH`（默认 `~/.cache/deepsearch_rl/tool_cache.db`）、`KEY_FILE`、`UVICORN_WORKERS`。

---

## 十、配置 Judge（默认 DeepSeek）

本次实验的 Answer Judge / Evidence Judge **统一由 DeepSeek Chat 承担，无需在本机部署 vLLM**，只需配置 API key：

```bash
# 密钥写入工程根目录 .env（已被 .gitignore 忽略，不会入库）
export DEEPSEEK_API_KEY="sk-xxxx"

# 检查配置（默认 base_url=https://api.deepseek.com/v1，model=deepseek-chat）
bash scripts/start_judge.sh
```

默认参数见 `configs/judge.yaml`：`max_concurrency=16`、`timeout=120`、`max_retries=3`、证据充分阈值 0.6。

**可选：使用本地 vLLM 作为 Judge**（无 API 费用 / 离线场景）：

```bash
vllm serve Qwen/Qwen3-8B \
  --served-model-name judge \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.4 \
  --max-model-len 8192 --dtype bfloat16 --enable-prefix-caching

# 训练前指向本地服务
export JUDGE_BASE_URL="http://127.0.0.1:8001/v1"
export JUDGE_MODEL="judge"
```

> **注意模型名**：本地 vLLM 用了 `--served-model-name judge`，客户端的 `JUDGE_MODEL` 必须设为 `judge`；
> 使用 DeepSeek 时保持 `deepseek-chat`。

两类裁判职责：

- **Answer Judge**：对照 gold answers 判断预测答案是否正确（correct / partial / wrong）；
- **Evidence Judge**：判断收集到的证据是否足以支撑答案，输出 0..1 连续分（阈值 0.6 判充分）。

---

## 十一、SwanLab 登录

```bash
pip install swanlab==0.9.0

# 方式一：交互式登录（API key 在 https://swanlab.cn/settings 获取）
swanlab login

# 方式二：非交互（CI / 无 TTY）
export SWANLAB_API_KEY=xxxxxxxx
```

> veRL v0.6.0 **已内置 SwanLab 追踪**，配置里 `trainer.logger=[console,swanlab]` 即可，无需额外适配。
> 无网/不上传时：`export SWANLAB_MODE=offline`（日志落本地，可后续 `swanlab watch` 同步）。

SwanLab 上可看到：`actor/policy_loss`、`actor/kl`、`actor/entropy`、`actor/grad_norm`、
`reward/score` 及 format/answer/evidence/tool 各分量、`tool/num_search`、`tool/duplicate_rate`、
`tool/error_rate`、`rollout/num_turns`、`timing/step_sec`，以及周期性记录的完整轨迹文本。

---

## 十二、启动训练

**终端 C**（确保终端 A 已启动、DeepSeek API key 已配置）：

```bash
export MODEL_PATH=$HOME/models/Qwen3-8B
export SWANLAB_API_KEY=xxxx          # 或 export SWANLAB_MODE=offline
bash scripts/train.sh
```

训练入口支持 `--dry_run`（只打印最终 Hydra 覆盖、不真正训练），便于先核对配置：

```bash
MODEL_PATH=$HOME/models/Qwen3-8B \
  python -m deepsearch_rl.train.train_grpo --dry_run
```

### 默认配置口径（6×RTX 4090，configs/grpo_qwen3_8b_6x4090.yaml）

- `data.train_batch_size=48`（48 prompts），`rollout.n=4` → **48×4 = 192 条多轮轨迹**（24GB 显存保守档，可按显存上调）；
- `rollout.name=sglang`、`rollout.mode=async`、`multi_turn.enable=True`、`multi_turn.format=search_xml`；
- `agent.default_agent_loop=deepsearch_agent`；
- `actor.use_dynamic_bsz=True`（动态 sequence balancing）、`state_masking=True`；
- `actor.use_kl_loss=True`、`kl_loss_type=low_var_kl`、`kl_loss_coef=0.001`；
- `algorithm.adv_estimator=grpo`、`temperature=1.0`、`lr=1e-6`、constant 调度。

### 命令行覆盖（Hydra key=value）

```bash
bash scripts/train.sh \
  data.train_batch_size=128 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45
```

---

## 十三、评估

训练完成后，先用 SGLang/vLLM 把**被评模型**以 OpenAI 端点起好（例如端口 30000），再跑评测：

```bash
# 终端里先起被评模型（示例：SGLang）
python -m sglang.launch_server \
  --model $HOME/checkpoints/deepsearch-rl/qwen3-8b/actor \
  --served-model-name default --port 30000

# 全量 48 题验证集（裁判默认走 DeepSeek API；也可指向自备 Judge :8001）
bash scripts/eval.sh

# 调试：只跑前 20 题
bash scripts/eval.sh --limit 20

# 保存每题完整轨迹
bash scripts/eval.sh --save_trajectories
```

输出答案正确率、证据充分度、正确且证据充分占比、重复调用率、平均 Search 次数等指标，并按数据源（hotpotqa / 2wiki / musique）分组，结果写 JSON 到 `outputs/eval/`。

**基线 vs 训练后对比**：

```bash
bash scripts/eval.sh --compare outputs/eval/metrics_baseline.json outputs/eval/metrics_trained.json
```

### 13.1 模型生成回答示例

下面是训练后模型（step 24）对上一节样本 `hotpotqa_val_000` 的一条**代表性多轮轨迹**（按真实交互格式整理，观测内容为检索服务返回的真实网页摘要，已做截断）。模型在 2 轮内完成"检索实体 → 定位战争 → 检索交战双方 → 作答"。

**第 1 轮**

模型输出：

```text
<search>Kajiwara Heima karō war</search>
<search>梶原平馬 家老 戦争</search>
```

观测（检索服务返回，已截断）：

```text
<observation>
[1] Kajiwara Heima - Wikipedia
Kajiwara Heima (梶原平馬) was a Japanese samurai of the late Edo period. He was a karō (senior
retainer) of the Aizu domain. He fought in the Boshin War and was killed at the Battle of Aizu in 1868.
[2] Aizu Domain - Wikipedia ...
</observation>
```

模型继续输出：

```text
<open>https://en.wikipedia.org/wiki/Kajiwara_Heima</open>
```

观测：

```text
<observation>
Kajiwara Heima (梶原平馬, 1842–1868) was a senior retainer (karō) of the Aizu clan. He served the
Aizu domain during the Boshin War and led domain troops at the Battle of Toba–Fushimi before dying
at the Battle of Aizu.
</observation>
```

**第 2 轮**

模型输出：

```text
<search>Boshin War two parties factions</search>
```

观测：

```text
<observation>
[1] Boshin War - Wikipedia
The Boshin War (戊辰戦争, 1868–1869) was a civil war in Japan fought between the forces of the ruling
Tokugawa shogunate and those seeking to return political power to the Imperial Court (the Satsuma–Chōshū alliance).
</observation>
```

模型输出最终答案：

```text
<answer>Tokugawa shogunate and those seeking to return political power to the Imperial Court</answer>
```

该轨迹与 gold answer 完全一致（EM 命中），`r_answer=1`；仅用 2 次 search、1 次 open，且无重复调用，对应训练后模型"精炼 2 轮、直接作答"的典型行为。

> 说明：本例为便于阅读整理的代表性轨迹（观测文本经截断），并非逐字日志；运行 `bash scripts/eval.sh --save_trajectories` 可保存每道题的完整原始轨迹。

---

## 十四、显存与调参

### 14.1 显存估算（6×4090 24GB）

- **训练态**：FSDP 分片 bf16 参数 + bf16 梯度 + fp32 Adam 状态约 **22–24GB/卡**，已贴近 24GB 上限，需依赖梯度检查点 + 动态 batch（`use_dynamic_bsz`）+ 调低 `ppo_micro_batch_size_per_gpu`（默认 1）；仍 OOM 则按 14.2 降档；
- **Rollout 态**：SGLang 默认 `TP=2 / DP=3`，权重分片 + KV 缓存由 `gpu_memory_utilization`（默认 0.50）控制；
- 训练与 rollout 不同时占用全部资源（hybrid engine 在阶段切换时 offload/释放）。

### 14.2 显存紧张时（按优先级）

1. 降低 `rollout.n`（4→3），直接减少并发轨迹数；
2. 降低 `gpu_memory_utilization`（0.50→0.40）；
3. `ppo_micro_batch_size_per_gpu` 已为 1 时，再缩短 `ppo_max_token_len_per_gpu`；
4. 缩短 `max_response_length` / `max_tool_response_length`；
5. 开启 actor CPU offload（`actor_rollout_ref.actor` 对应 offload 项），用显存换速度。

### 14.3 关键调参建议

| 现象 | 建议 |
|---|---|
| 模型不搜索、直接猜（under-search） | 降低 answer gate 下限/提高证据门控权重；适当提高 `temperature`；确认证据奖励生效 |
| 模型反复刷搜索（over-search） | 提高重复/冗余惩罚；降低 `efficient_budget`；检查证据充分奖励是否过早饱和 |
| 奖励虚高、答案却错（reward hacking） | 强化答案门控（无证据正确只给低分）；Answer Judge 与 EM 双重校验 |
| 收敛慢 / 波动大 | 组大小 `n` 取 5~8；`kl_loss_coef` 在 0.001~0.01 调整；lr 1e-6 量级 |
| 多卡负载不均 | 保持 `use_dynamic_bsz=True`；检查超长样本是否被过滤 |

---

## 十五、常见问题 FAQ

**Q1：CUDA / torch 车道怎么选？**
RTX 4090 是 Ada sm_89，用 **cu124 车道**（`--index-url https://download.pytorch.org/whl/cu124`）即可；若换 5090（Blackwell sm_120）才需 cu128（cu126 及以下不含其 kernel）。

**Q2：为什么 pin verl v0.6.0、sglang ≤0.5.19？**
verl main 与 sglang 0.5.20+ 已迁移到 CUDA 13 / torch 2.13+，与本工程 cu124/torch2.8 车道冲突，
会拉到无法在 4090 上运行的依赖。

**Q3：不买搜索 API 能跑吗？**
可以，`SEARCH_BACKEND=ddg` 使用免费 DuckDuckGo（无需 key），但稳定性与召回不如付费服务，建议仅用于调试。

**Q4：Judge 服务一定要单独起吗？**
默认使用 DeepSeek 云端 API，无需部署任何本地服务，配置好 `DEEPSEEK_API_KEY` 即可。Judge 不可达时 reward 会自动降级为纯规则（EM/F1 + 标题命中），
训练不会中断，但证据充分度信号会变弱。

**Q5：工具返回的内容会参与策略梯度吗？**
不会。veRL 对工具返回 token 自动标 `response_mask=0`，策略梯度只在模型自己生成的 token 上计算。

**Q6：如何调整并行度适配不同卡数？**
- 6 卡（默认配置 `configs/grpo_qwen3_8b_6x4090.yaml`）：`n_gpus_per_node=6`，`rollout.tensor_model_parallel_size=2, data_parallel_size=3`，`train_batch_size=48`（24GB 显存保守档，可按显存上调至 56/84）；
- 8 卡：`trainer.n_gpus_per_node=8`，可把 `rollout.data_parallel_size` 提到 4、`train_batch_size` 提到 112；
- 2 卡：`n_gpus_per_node=2`，`rollout.tensor_model_parallel_size=2, data_parallel_size=1`，`train_batch_size=28`。

---

## 十六、参考与致谢

本工程在设计与实现上借鉴了以下优秀项目（均为实际调研）：

- [veRL](https://github.com/volcengine/verl)：火山引擎开源 RLHF/Agentic RL 框架，`ToolAgentLoop`、GRPO、动态 batch；
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)：搜索/答案标签协议、EM 奖励、检索服务；
- [R1-Searcher](https://github.com/PeterGriffinJin/R1-Searcher)：检索奖励与训练数据组织；
- [SimpleRL-Zoo](https://github.com/SimpleAI-Zoo/SimpleRL-Zoo)：异步 rollout 与 weight-sync；
- 数据集：HotpotQA、2WikiMultihopQA、MuSiQue、Natural Questions、Bamboogle；基座模型：Qwen3-8B。

## License

本项目基于 [Apache License 2.0](LICENSE) 发布。
