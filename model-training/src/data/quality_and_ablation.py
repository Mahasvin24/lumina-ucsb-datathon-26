from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .common import write_json


def _collect_valid_step_rows(path: Path) -> list[dict[str, float]]:
    step_rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            window = json.loads(line)
            length = int(window["sequence_length"])
            for idx in range(max(0, length - 1)):
                if window["target_mask"][idx] != 1:
                    continue
                step_rows.append(
                    {
                        "y": float(window["target_next_response"][idx]),
                        "question_prior": float(window["question_difficulty_prior"][idx]),
                        "concept_prior": float(window["concept_difficulty_prior"][idx]),
                        "hist_accuracy": float(window["hist_accuracy"][idx]),
                        "response_feature": float(window["responses"][idx]),
                    }
                )
    return step_rows


def _accuracy(y_true: list[float], y_prob: list[float], threshold: float = 0.5) -> float:
    if not y_true:
        return 0.0
    hits = 0
    for truth, prob in zip(y_true, y_prob, strict=True):
        pred = 1.0 if prob >= threshold else 0.0
        if pred == truth:
            hits += 1
    return hits / len(y_true)


def _auc(y_true: list[float], y_prob: list[float]) -> float:
    pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0])
    n = len(pairs)
    if n == 0:
        return 0.5
    n_pos = sum(1 for _, y in pairs if y == 1.0)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank_sum_pos = 0.0
    idx = 0
    while idx < n:
        j = idx
        while j + 1 < n and pairs[j + 1][0] == pairs[idx][0]:
            j += 1
        avg_rank = (idx + j + 2) / 2.0
        pos_count = sum(1 for k in range(idx, j + 1) if pairs[k][1] == 1.0)
        rank_sum_pos += avg_rank * pos_count
        idx = j + 1

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def run_quality_checks_and_ablation(processed_dir: Path, reports_dir: Path, random_seed: int) -> dict[str, Any]:
    random.seed(random_seed)
    fold_dirs = sorted([path for path in processed_dir.iterdir() if path.is_dir() and path.name.startswith("fold_")])

    fold_metrics: dict[str, Any] = {}
    auc_values: list[float] = []
    acc_values: list[float] = []
    leakage_values: list[float] = []

    for fold_dir in fold_dirs:
        steps = _collect_valid_step_rows(fold_dir / "valid_windows.jsonl")
        y_true = [row["y"] for row in steps]
        if not y_true:
            continue

        global_mean = sum(y_true) / len(y_true)
        pred_global = [global_mean] * len(y_true)
        pred_question = [row["question_prior"] for row in steps]
        pred_qc = [0.7 * row["question_prior"] + 0.3 * row["concept_prior"] for row in steps]
        pred_full = [
            min(1.0, max(0.0, 0.55 * row["question_prior"] + 0.25 * row["concept_prior"] + 0.20 * row["hist_accuracy"]))
            for row in steps
        ]

        random_baseline = pred_full[:]
        random.shuffle(random_baseline)
        leakage_probe = _accuracy(y_true, [row["response_feature"] for row in steps])

        fold_auc = _auc(y_true, pred_full)
        fold_acc = _accuracy(y_true, pred_full)
        auc_values.append(fold_auc)
        acc_values.append(fold_acc)
        leakage_values.append(leakage_probe)

        fold_metrics[fold_dir.name] = {
            "n_steps": len(y_true),
            "global_baseline": {"auc": _auc(y_true, pred_global), "acc": _accuracy(y_true, pred_global)},
            "question_prior_baseline": {"auc": _auc(y_true, pred_question), "acc": _accuracy(y_true, pred_question)},
            "question_concept_baseline": {"auc": _auc(y_true, pred_qc), "acc": _accuracy(y_true, pred_qc)},
            "full_feature_baseline": {"auc": fold_auc, "acc": fold_acc},
            "shuffled_time_probe": {"auc": _auc(y_true, random_baseline), "acc": _accuracy(y_true, random_baseline)},
            "response_feature_leakage_probe": {"acc": leakage_probe},
        }

    summary = {
        "mean_full_feature_auc": sum(auc_values) / len(auc_values) if auc_values else 0.0,
        "mean_full_feature_acc": sum(acc_values) / len(acc_values) if acc_values else 0.0,
        "mean_response_feature_leakage_probe_acc": (
            sum(leakage_values) / len(leakage_values) if leakage_values else 0.0
        ),
        "note": (
            "Leakage probe should not be near 1.0 for next-step targets. "
            "If it is very high, verify that targets are y_{t+1} and not y_t."
        ),
        "fold_metrics": fold_metrics,
    }
    write_json(reports_dir / "ablation_qlevel_v1.json", summary)
    return summary
