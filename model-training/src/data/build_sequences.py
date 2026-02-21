from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import stable_hash, write_json


def _window_user_sequence(
    rows: list[dict[str, Any]],
    window_size: int,
    stride: int,
) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    windows: list[list[dict[str, Any]]] = []
    start = 0
    while start < len(rows):
        chunk = rows[start : start + window_size]
        if len(chunk) < 2:
            break
        windows.append(chunk)
        if start + window_size >= len(rows):
            break
        start += stride
    return windows


def _pack_window(window_rows: list[dict[str, Any]], window_size: int) -> dict[str, Any]:
    length = len(window_rows)
    pad_len = window_size - length

    question_idx = [row["question_idx"] for row in window_rows] + [0] * pad_len
    responses = [row["response"] for row in window_rows] + [0] * pad_len
    delta_t_log_bin = [row["delta_t_log_bin"] for row in window_rows] + [0] * pad_len
    num_concepts = [row["num_concepts"] for row in window_rows] + [0] * pad_len
    concept_indices = [row["concept_indices"] for row in window_rows] + [[] for _ in range(pad_len)]
    concept_rarity_bucket = [row["concept_rarity_bucket"] for row in window_rows] + [0] * pad_len
    hist_accuracy = [row["hist_accuracy"] for row in window_rows] + [0.0] * pad_len
    question_prior = [row["question_difficulty_prior"] for row in window_rows] + [0.0] * pad_len
    concept_prior = [row["concept_difficulty_prior"] for row in window_rows] + [0.0] * pad_len
    question_type = [row["question_type"] for row in window_rows] + [-1] * pad_len
    content_missing = [row["content_embedding_missing"] for row in window_rows] + [1] * pad_len
    analysis_missing = [row["analysis_embedding_missing"] for row in window_rows] + [1] * pad_len
    content_embedding_qid = [row["content_embedding_qid"] for row in window_rows] + [-1] * pad_len
    analysis_embedding_qid = [row["analysis_embedding_qid"] for row in window_rows] + [-1] * pad_len

    # Next-step target for each timestep. Last valid timestep has no target.
    next_response = responses[1:length] + [0] * (window_size - length + 1)
    target_mask = [1] * max(0, length - 1) + [0] * (window_size - max(0, length - 1))
    input_mask = [1] * length + [0] * pad_len

    return {
        "user_id": window_rows[0]["user_id"],
        "fold": window_rows[0]["fold"],
        "sequence_length": length,
        "question_idx": question_idx,
        "responses": responses,
        "delta_t_log_bin": delta_t_log_bin,
        "num_concepts": num_concepts,
        "concept_indices": concept_indices,
        "concept_rarity_bucket": concept_rarity_bucket,
        "hist_accuracy": hist_accuracy,
        "question_difficulty_prior": question_prior,
        "concept_difficulty_prior": concept_prior,
        "question_type": question_type,
        "content_embedding_missing": content_missing,
        "analysis_embedding_missing": analysis_missing,
        "content_embedding_qid": content_embedding_qid,
        "analysis_embedding_qid": analysis_embedding_qid,
        "target_next_response": next_response,
        "target_mask": target_mask,
        "input_mask": input_mask,
    }


def package_sequences(
    processed_dir: Path,
    window_size: int,
    stride: int,
) -> dict[str, Any]:
    fold_dirs = sorted([p for p in processed_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    fold_summary: dict[str, Any] = {}

    for fold_dir in fold_dirs:
        print(f"      [windows] fold={fold_dir.name}: packaging splits...", flush=True)
        feature_files = sorted(fold_dir.glob("*_features.jsonl"))
        for feature_path in feature_files:
            split_name = feature_path.name.replace("_features.jsonl", "")
            window_path = fold_dir / f"{split_name}_windows.jsonl"

            windows_written = 0
            user_count = 0
            current_user: int | None = None
            user_rows: list[dict[str, Any]] = []

            with feature_path.open("r", encoding="utf-8") as feature_handle, window_path.open(
                "w", encoding="utf-8"
            ) as window_handle:
                for line in feature_handle:
                    row = json.loads(line)
                    row_user = int(row["user_id"])

                    if current_user is None:
                        current_user = row_user
                    if row_user != current_user:
                        user_count += 1
                        for window in _window_user_sequence(user_rows, window_size=window_size, stride=stride):
                            window_handle.write(json.dumps(_pack_window(window, window_size=window_size), ensure_ascii=True))
                            window_handle.write("\n")
                            windows_written += 1
                        user_rows = []
                        current_user = row_user
                    user_rows.append(row)

                if user_rows:
                    user_count += 1
                    for window in _window_user_sequence(user_rows, window_size=window_size, stride=stride):
                        window_handle.write(json.dumps(_pack_window(window, window_size=window_size), ensure_ascii=True))
                        window_handle.write("\n")
                        windows_written += 1

            fold_id = fold_dir.name.replace("fold_", "")
            fold_summary.setdefault(fold_id, {})
            fold_summary[fold_id][f"{split_name}_windows"] = windows_written
            fold_summary[fold_id][f"{split_name}_users"] = user_count
            print(
                f"      [windows] fold={fold_id}, split={split_name}: "
                f"users={user_count}, windows={windows_written}",
                flush=True,
            )

    schema = {
        "window_size": window_size,
        "stride": stride,
        "features": [
            "question_idx",
            "responses",
            "delta_t_log_bin",
            "num_concepts",
            "concept_indices",
            "concept_rarity_bucket",
            "hist_accuracy",
            "question_difficulty_prior",
            "concept_difficulty_prior",
            "question_type",
            "content_embedding_missing",
            "analysis_embedding_missing",
            "content_embedding_qid",
            "analysis_embedding_qid",
        ],
        "targets": ["target_next_response", "target_mask"],
    }
    config_hash = stable_hash({"window_size": window_size, "stride": stride, "schema": schema})
    manifest = {
        "version": "dkt_qlevel_v1",
        "config_hash": config_hash,
        "fold_summary": fold_summary,
        "schema": schema,
    }
    write_json(processed_dir / "manifest.json", manifest)
    return manifest
