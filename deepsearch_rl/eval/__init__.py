# -*- coding: utf-8 -*-
"""
多跳搜索验证集评测子模块
========================

对被评模型（SGLang/vLLM OpenAI 端点）在 ``data/eval_val_48.jsonl``——实验实际
使用的 48 题多跳验证集（HotpotQA 20 + 2WikiMultihopQA 16 + MuSiQue 12，与
``data/processed/fast/val`` 对应，seed=42）——上做离线多轮检索评测，输出核心指标：

- Accuracy             = mean(answer_correct)
- Evidence Sufficiency = mean(evidence_sufficiency)
- Correct & Sufficient = mean(correct_and_sufficient)
- Duplicate Call Rate  = mean(每题 duplicate_rate)
- Avg Search/query     = mean(num_search)

入口：``python -m deepsearch_rl.eval.evaluate``（见 evaluate.py 的 argparse）。
另提供 ``compare`` 函数对比两份 metrics JSON（baseline vs trained，含 +pp 差值）。

注意：本包顶层不 import openai / torch / verl；工具与 Judge 客户端在函数内懒加载，
Judge 不可达时自动降级为 EM/F1 + supporting-title 命中代理，绝不中断评测。
"""

from .evaluate import (
    aggregate_metrics,
    build_markdown_table,
    compare_metrics,
    evaluate_main,
)

__all__ = [
    "aggregate_metrics",
    "build_markdown_table",
    "compare_metrics",
    "evaluate_main",
]
