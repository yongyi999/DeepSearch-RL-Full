# data/ 数据目录说明

DeepSearch-RL 的数据下载、统一中间格式、训练 parquet 与 48 题多跳验证集均在本目录生成。

## 1. 数据源

| source    | HuggingFace ID / 来源                          | config / split            | 单跳/多跳 | 说明 |
|-----------|------------------------------------------------|---------------------------|-----------|------|
| nq        | `google-research-datasets/nq_open`             | train / validation        | 单跳      | 真实 Google query，answer 为 list[str] |
| hotpotqa  | `hotpotqa/hotpot_qa`                           | `fullwiki`，train / validation | 2 跳 | answer 为 str；type=bridge/comparison；level=easy/medium/hard |
| 2wiki     | `voidful/2WikiMultihopQA`                      | train / validation        | 2 跳      | answer 为 str；带 evidences/supporting_facts |
| musique   | `dgslibisey/MuSiQue`                           | train / validation        | 2-4 跳    | answer=str + answer_aliases=list；answerable=bool |
| bamboogle | [官方 json](https://raw.githubusercontent.com/ofirpress/self-ask/master/data/bamboogle_2hop.json) | 仅 validation（125 题） | 2 跳 | 对抗题，golden_answers=list；不进训练集 |

## 2. 统一中间 jsonl 字段（download_data.py 产出到 `data/raw/`）

每个源每个 split 一个文件：`{source}_{split}.jsonl`，每行字段：

```json
{
  "id": "hotpot_xxx | nq_xxx | 2wiki_xxx | musique_xxx | bamboogle_0000",
  "source": "nq | hotpotqa | 2wiki | musique | bamboogle",
  "question": "...",
  "gold_answers": ["主答案", "别名..."],
  "type": "single-hop | bridge | comparison | compositional ...",
  "level": "easy | medium | hard",
  "num_hops": 1,
  "supporting_titles": ["标题1", "标题2"],
  "split": "train | validation"
}
```

> MuSiQue 行额外带 `answerable: bool`，供验证集按 answerable 过滤。

答案归一规则：HotpotQA `[answer]`；NQ-open 直接用 `answer` 列表；MuSiQue `[answer] + answer_aliases`；2Wiki `[answer]`；Bamboogle 用 `golden_answers`。

## 3. 训练 parquet（prepare_train.py 产出到 `data/processed/`）

严格按 SPEC 4.1，每行：

```json
{
  "data_source": "nq | hotpotqa | 2wiki | musique",
  "prompt": [
    {"role": "system", "content": "<SYSTEM_PROMPT>"},
    {"role": "user",   "content": "<build_user_prompt(question)>"}
  ],
  "ability": "multi-hop-search",
  "agent_name": "deepsearch_agent",
  "reward_model": {"style": "rule", "ground_truth": {"target": ["答案", "别名..."]}},
  "extra_info": {"split": "train|val", "index": 0, "num_hops": 2}
}
```

- prompt 是 `list[dict]`（veRL `return_raw_chat=True` 要求）。
- 训练混合：NQ-open 降采样到 `--nq_limit`（默认 30000，seed=42）+ HotpotQA/2Wiki/MuSiQue train 全量；Bamboogle 排除。
- 从每个 train 源固定抽 200 条做训练期 val（seed=42）。
- 输出 shard：`data/processed/train/{source}.parquet`、`data/processed/val/{source}.parquet`。

## 4. 48 题多跳验证集（实验实际使用的评测集）

实验在训练过程中用一个固定的小型多跳验证集做周期性评测（veRL `val_before_train=True`、`test_freq=3`）。
它由 `prepare_train.py --fast` 生成到 `data/processed/fast/val/`，只取各源 train 中难多跳样本，
与同目录 `fast/train/` 按 id 不重叠（seed=42）：

| 来源                       | 验证集条数 | 筛选条件                         |
|----------------------------|-----------:|----------------------------------|
| HotpotQA fullwiki          | 20         | 优先 `level==hard` / `type==comparison` |
| 2WikiMultihopQA            | 16         | compositional/comparison/bridge/inference 均衡 |
| MuSiQue                    | 12         | 优先 3–4 跳、`answerable==True`  |
| **合计**                   | **48**     | 恰好 1 个 val batch（val_batch_size=48） |

为支持离线评测，这 48 题同时导出为 `data/eval_val_48.jsonl`，字段：
`id, source, question, gold_answers, num_hops, supporting_titles, split`，与 `fast/val` 一一对应。
离线评测入口：`python -m deepsearch_rl.eval.evaluate`（默认即读取该文件）。

## 5. 命令

```bash
# 一键：下载 -> 全量 parquet -> 快训子集（含 48 题验证集）
bash scripts/download_data.sh

# 或分步：
python data/download_data.py --out_dir data/raw \
    --sources nq,hotpotqa,2wiki,musique,bamboogle \
    --nq_limit 30000 --backend auto

python data/prepare_train.py --raw_dir data/raw \
    --out_dir data/processed --nq_limit 30000 --seed 42

# 快训子集：data/processed/fast/{train,val}（48 题验证集同时导出为 data/eval_val_48.jsonl）
python data/prepare_train.py --raw_dir data/raw --fast --seed 42
```

模型下载：

```bash
bash scripts/download_model.sh   # 默认落盘 ~/models/Qwen3-8B
```

## 6. 后端 / 网络提示

- `--backend auto`：优先 modelscope，失败回退 huggingface datasets。
- 国内 HF 加速：脚本自动设置 `HF_ENDPOINT=https://hf-mirror.com`，可手动覆盖。
- 依赖：`pip install -U datasets huggingface_hub modelscope pyarrow`。
