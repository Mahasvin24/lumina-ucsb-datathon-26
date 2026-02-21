from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_json, write_json, write_jsonl


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def build_cv_splits(
    cleaned_rows: list[dict[str, Any]],
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    folds = sorted({row["fold"] for row in cleaned_rows})
    split_map: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for fold_id in folds:
        valid = [row for row in cleaned_rows if row["fold"] == fold_id]
        train = [row for row in cleaned_rows if row["fold"] != fold_id]
        train_users = {row["user_id"] for row in train}
        valid_users = {row["user_id"] for row in valid}
        overlap = train_users & valid_users
        if overlap:
            raise ValueError(
                f"Leakage check failed for fold {fold_id}: "
                f"{len(overlap)} users overlap across train/valid."
            )
        split_map[fold_id] = {"train": train, "valid": valid}
    return split_map


def _build_vocab(values: list[int]) -> dict[int, int]:
    vocab = {-1: 0}
    for token in sorted(set(values)):
        if token in vocab:
            continue
        vocab[token] = len(vocab)
    return vocab


def _question_priors(train_rows: list[dict[str, Any]], alpha: float) -> tuple[dict[int, float], float]:
    totals: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    for row in train_rows:
        qid = row["question_id"]
        totals[qid] += 1
        correct[qid] += row["response"]

    global_total = len(train_rows)
    global_correct = sum(row["response"] for row in train_rows)
    global_mean = (global_correct / global_total) if global_total else 0.5

    priors: dict[int, float] = {}
    for qid, count in totals.items():
        priors[qid] = (correct[qid] + alpha * global_mean) / (count + alpha)
    return priors, global_mean


def _concept_priors(train_rows: list[dict[str, Any]], alpha: float, global_mean: float) -> dict[int, float]:
    totals: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    for row in train_rows:
        for concept_id in row["concept_ids"]:
            totals[concept_id] += 1
            correct[concept_id] += row["response"]

    priors: dict[int, float] = {}
    for concept_id, count in totals.items():
        priors[concept_id] = (correct[concept_id] + alpha * global_mean) / (count + alpha)
    return priors


def _project_embedding(vector: list[float], out_dim: int) -> list[float]:
    if out_dim <= 0:
        raise ValueError("out_dim must be > 0")
    if not vector:
        return [0.0] * out_dim
    if len(vector) <= out_dim:
        projected = [float(value) for value in vector] + [0.0] * (out_dim - len(vector))
        return _l2_normalize(projected)
    projected = [0.0] * out_dim
    segment = len(vector) / float(out_dim)
    for out_idx in range(out_dim):
        start = int(out_idx * segment)
        end = int((out_idx + 1) * segment)
        if end <= start:
            end = min(len(vector), start + 1)
        bucket = vector[start:end]
        projected[out_idx] = sum(float(value) for value in bucket) / max(1, len(bucket))
    return _l2_normalize(projected)


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _to_log_bin(delta_ms: int) -> int:
    if delta_ms <= 0:
        return 0
    seconds = max(1.0, delta_ms / 1000.0)
    return min(20, int(math.log2(seconds)))


def _load_question_metadata(metadata_root: Path) -> dict[int, dict[str, Any]]:
    questions_path = metadata_root / "questions.json"
    questions_payload = read_json(questions_path)
    metadata: dict[int, dict[str, Any]] = {}
    for key, value in questions_payload.items():
        qid = int(key)
        question_type_raw = value.get("type")
        question_type = _normalize_question_type(question_type_raw)
        metadata[qid] = {
            "question_type": question_type,
        }
    return metadata


def _normalize_question_type(question_type: Any) -> int:
    if question_type is None:
        return -1
    if isinstance(question_type, int):
        return question_type
    text = str(question_type).strip().lower()
    if text in {"0", "fill_blank", "fill-in-the-blank", "fill in the blank", "填空"}:
        return 0
    if text in {"1", "multi_choice", "multiple_choice", "multiple-choice", "choice", "选择"}:
        return 1
    return -1


def _load_embedding_file(path: Path, out_dim: int) -> dict[int, list[float]]:
    payload = read_json(path)
    result: dict[int, list[float]] = {}
    for key, value in payload.items():
        if not isinstance(value, list):
            continue
        result[int(key)] = _project_embedding(value, out_dim)
    return result


def _augment_split_rows(
    rows: list[dict[str, Any]],
    question_vocab: dict[int, int],
    concept_vocab: dict[int, int],
    question_priors: dict[int, float],
    concept_priors: dict[int, float],
    global_mean: float,
    question_meta: dict[int, dict[str, Any]],
    q_content_emb: dict[int, list[float]],
    q_analysis_emb: dict[int, list[float]],
) -> list[dict[str, Any]]:
    rows_sorted = sorted(rows, key=lambda r: (r["user_id"], r["timestamp"], r["sequence_row"], r["t_idx"]))

    user_total_attempts: dict[int, int] = defaultdict(int)
    user_total_correct: dict[int, int] = defaultdict(int)
    user_question_attempts: dict[tuple[int, int], int] = defaultdict(int)
    user_concept_attempts: dict[tuple[int, int], int] = defaultdict(int)
    user_last_timestamp: dict[int, int] = {}

    concept_counts = Counter(cid for row in rows for cid in row["concept_ids"])

    enriched: list[dict[str, Any]] = []
    for row in rows_sorted:
        uid = row["user_id"]
        qid = row["question_id"]
        concept_ids = row["concept_ids"]
        prev_ts = user_last_timestamp.get(uid, row["timestamp"])
        delta_ms = max(0, row["timestamp"] - prev_ts)

        hist_attempts = user_total_attempts[uid]
        hist_correct = user_total_correct[uid]
        hist_acc = (hist_correct / hist_attempts) if hist_attempts else global_mean

        q_key = (uid, qid)
        user_q_attempts = user_question_attempts[q_key]

        concept_attempt_values = [user_concept_attempts[(uid, cid)] for cid in concept_ids]
        avg_user_concept_attempts = (
            sum(concept_attempt_values) / len(concept_attempt_values) if concept_attempt_values else 0.0
        )
        concept_prior_values = [concept_priors.get(cid, global_mean) for cid in concept_ids]
        concept_prior = sum(concept_prior_values) / len(concept_prior_values) if concept_prior_values else global_mean

        concept_rarity = sum(concept_counts.get(cid, 0) for cid in concept_ids)
        if concept_ids:
            concept_rarity /= len(concept_ids)
        rarity_bucket = 0
        if concept_rarity > 10:
            rarity_bucket = 1
        if concept_rarity > 100:
            rarity_bucket = 2
        if concept_rarity > 1000:
            rarity_bucket = 3

        meta = question_meta.get(qid, {"question_type": -1})
        has_content_emb = qid in q_content_emb
        has_analysis_emb = qid in q_analysis_emb

        enriched_row = {
            **row,
            "question_idx": question_vocab.get(qid, 0),
            "concept_indices": [concept_vocab.get(cid, 0) for cid in concept_ids],
            "num_concepts": len(concept_ids),
            "delta_t_ms": delta_ms,
            "delta_t_log_bin": _to_log_bin(delta_ms),
            "hist_attempts": hist_attempts,
            "hist_accuracy": hist_acc,
            "user_question_attempts": user_q_attempts,
            "user_concept_attempts_avg": avg_user_concept_attempts,
            "question_difficulty_prior": question_priors.get(qid, global_mean),
            "concept_difficulty_prior": concept_prior,
            "concept_rarity_bucket": rarity_bucket,
            "question_type": meta["question_type"],
            # Store embedding references to avoid duplicating vectors per interaction row.
            "content_embedding_qid": qid if has_content_emb else -1,
            "analysis_embedding_qid": qid if has_analysis_emb else -1,
            "content_embedding_missing": 0 if has_content_emb else 1,
            "analysis_embedding_missing": 0 if has_analysis_emb else 1,
        }
        enriched.append(enriched_row)

        user_total_attempts[uid] += 1
        user_total_correct[uid] += row["response"]
        user_question_attempts[q_key] += 1
        for cid in concept_ids:
            user_concept_attempts[(uid, cid)] += 1
        user_last_timestamp[uid] = row["timestamp"]

    return enriched


def build_features_for_all_folds(
    interim_dir: Path,
    processed_dir: Path,
    metadata_root: Path,
    emb_dim: int,
    smoothing_alpha: float,
    cleaned_filename: str = "question_level_long_clean.jsonl",
    extra_split_files: dict[str, Path] | None = None,
) -> dict[str, Any]:
    cleaned_rows = _read_jsonl(interim_dir / cleaned_filename)
    split_map = build_cv_splits(cleaned_rows)
    extra_split_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_path in (extra_split_files or {}).items():
        extra_split_rows[split_name] = _read_jsonl(split_path)

    question_meta = _load_question_metadata(metadata_root)
    q_content_emb = _load_embedding_file(metadata_root / "embeddings" / "qid2content_emb.json", emb_dim)
    q_analysis_emb = _load_embedding_file(metadata_root / "embeddings" / "qid2analysis_emb.json", emb_dim)
    write_json(
        processed_dir / "semantic_embedding_tables.json",
        {
            "projection_dim": emb_dim,
            "qid2content_emb": {str(k): v for k, v in q_content_emb.items()},
            "qid2analysis_emb": {str(k): v for k, v in q_analysis_emb.items()},
        },
    )

    fold_summaries: dict[str, Any] = {}
    for fold_id, split in split_map.items():
        print(f"      [features] fold={fold_id}: preparing train/valid splits...", flush=True)
        train_rows = split["train"]
        valid_rows = split["valid"]
        question_vocab = _build_vocab([row["question_id"] for row in train_rows])
        concept_vocab = _build_vocab([cid for row in train_rows for cid in row["concept_ids"]])
        question_priors, global_mean = _question_priors(train_rows, smoothing_alpha)
        concept_priors = _concept_priors(train_rows, smoothing_alpha, global_mean)

        train_features = _augment_split_rows(
            rows=train_rows,
            question_vocab=question_vocab,
            concept_vocab=concept_vocab,
            question_priors=question_priors,
            concept_priors=concept_priors,
            global_mean=global_mean,
            question_meta=question_meta,
            q_content_emb=q_content_emb,
            q_analysis_emb=q_analysis_emb,
        )
        valid_features = _augment_split_rows(
            rows=valid_rows,
            question_vocab=question_vocab,
            concept_vocab=concept_vocab,
            question_priors=question_priors,
            concept_priors=concept_priors,
            global_mean=global_mean,
            question_meta=question_meta,
            q_content_emb=q_content_emb,
            q_analysis_emb=q_analysis_emb,
        )

        fold_dir = processed_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(fold_dir / "train_features.jsonl", train_features)
        write_jsonl(fold_dir / "valid_features.jsonl", valid_features)
        for extra_name, rows in extra_split_rows.items():
            extra_features = _augment_split_rows(
                rows=rows,
                question_vocab=question_vocab,
                concept_vocab=concept_vocab,
                question_priors=question_priors,
                concept_priors=concept_priors,
                global_mean=global_mean,
                question_meta=question_meta,
                q_content_emb=q_content_emb,
                q_analysis_emb=q_analysis_emb,
            )
            write_jsonl(fold_dir / f"{extra_name}_features.jsonl", extra_features)
        write_json(
            fold_dir / "feature_state.json",
            {
                "fold_id": fold_id,
                "question_vocab_size": len(question_vocab),
                "concept_vocab_size": len(concept_vocab),
                "global_mean_response": global_mean,
                "smoothing_alpha": smoothing_alpha,
            },
        )

        fold_summaries[str(fold_id)] = {
            "train_rows": len(train_features),
            "valid_rows": len(valid_features),
            "question_vocab_size": len(question_vocab),
            "concept_vocab_size": len(concept_vocab),
            "global_mean_response": global_mean,
        }
        print(
            f"      [features] fold={fold_id}: done "
            f"(train_rows={len(train_features)}, valid_rows={len(valid_features)})",
            flush=True,
        )

    summary = {
        "folds": fold_summaries,
        "embedding_projection_dim": emb_dim,
        "embedding_projection_method": "contiguous_mean_pool_l2",
        "metadata_root": str(metadata_root),
    }
    write_json(processed_dir / "feature_summary.json", summary)
    return summary
