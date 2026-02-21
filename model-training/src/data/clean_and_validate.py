from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import read_json, write_json, write_jsonl


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(read_json_from_line(line))
    return rows


def read_json_from_line(line: str) -> dict[str, Any]:
    import json

    return json.loads(line)


def clean_and_validate_long_table(
    interim_dir: Path,
    reports_dir: Path,
    min_user_history: int,
    input_basename: str = "question_level_long",
    cleaned_basename: str = "question_level_long_clean",
    short_basename: str = "question_level_long_short_histories",
    write_quality_report: bool = True,
    filter_short_histories: bool = True,
) -> dict[str, Any]:
    long_path = interim_dir / f"{input_basename}.jsonl"
    rows = _read_jsonl(long_path)
    input_rows = len(rows)

    rows.sort(key=lambda r: (r["user_id"], r["timestamp"], r["sequence_row"], r["t_idx"]))

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, int, int, tuple[int, ...]]] = set()
    duplicate_rows = 0
    for row in rows:
        # Only drop true exact duplicates to preserve valid repeated attempts.
        key = (
            row["user_id"],
            row["question_id"],
            row["timestamp"],
            row["response"],
            row["sequence_row"],
            row["t_idx"],
            tuple(row["concept_ids"]),
        )
        if key in seen:
            duplicate_rows += 1
            continue
        seen.add(key)
        deduped.append(row)

    user_counts = Counter(row["user_id"] for row in deduped)
    short_history_users = {uid for uid, count in user_counts.items() if count < min_user_history}

    kept_rows: list[dict[str, Any]] = []
    short_rows: list[dict[str, Any]] = []
    for row in deduped:
        if filter_short_histories and row["user_id"] in short_history_users:
            short_rows.append(row)
        else:
            kept_rows.append(row)

    kept_rows.sort(key=lambda r: (r["fold"], r["user_id"], r["timestamp"], r["sequence_row"], r["t_idx"]))
    short_rows.sort(key=lambda r: (r["fold"], r["user_id"], r["timestamp"], r["sequence_row"], r["t_idx"]))

    kept_path = interim_dir / f"{cleaned_basename}.jsonl"
    short_path = interim_dir / f"{short_basename}.jsonl"
    write_jsonl(kept_path, kept_rows)
    write_jsonl(short_path, short_rows)

    fold_sizes = Counter(row["fold"] for row in kept_rows)
    concept_lengths = Counter(len(row["concept_ids"]) for row in kept_rows)
    missing_concept_rows = concept_lengths.get(0, 0)

    per_user_lengths: dict[int, int] = defaultdict(int)
    for row in kept_rows:
        per_user_lengths[row["user_id"]] += 1

    quality_report = {
        "input_basename": input_basename,
        "input_rows": input_rows,
        "duplicate_rows_dropped": duplicate_rows,
        "kept_rows": len(kept_rows),
        "short_history_rows": len(short_rows),
        "short_history_users": len(short_history_users),
        "fold_sizes": {str(k): v for k, v in sorted(fold_sizes.items())},
        "missing_concept_rows": missing_concept_rows,
        "concept_count_distribution": {str(k): v for k, v in sorted(concept_lengths.items())},
        "min_user_length": min(per_user_lengths.values()) if per_user_lengths else 0,
        "median_user_length": _median(list(per_user_lengths.values())),
        "max_user_length": max(per_user_lengths.values()) if per_user_lengths else 0,
        "ingest_summary": read_json(interim_dir / f"{input_basename}_ingest_summary.json"),
        "cleaned_path": str(kept_path),
        "short_history_path": str(short_path),
    }
    if write_quality_report:
        write_json(reports_dir / "data_quality_qlevel_v1.json", quality_report)
    return quality_report


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_values[mid])
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
