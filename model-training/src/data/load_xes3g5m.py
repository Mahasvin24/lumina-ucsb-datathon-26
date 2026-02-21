from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import ensure_dirs, write_json, write_jsonl


SEQUENCE_COLUMNS = ("questions", "concepts", "responses", "timestamps", "selectmasks")


@dataclass
class IngestResult:
    interactions: list[dict[str, Any]]
    rejected_rows: int
    rejected_interactions: int
    parse_failures: int
    source_rows: int


def _parse_int_sequence(raw: str) -> list[int]:
    cleaned = raw.strip()
    if not cleaned:
        return []
    return [int(token) for token in cleaned.split(",")]


def _parse_concept_sequence(raw: str) -> list[list[int]]:
    cleaned = raw.strip()
    if not cleaned:
        return []
    concept_groups: list[list[int]] = []
    for token in cleaned.split(","):
        if token == "-1":
            concept_groups.append([])
            continue
        concept_groups.append([int(part) for part in token.split("_") if part])
    return concept_groups


def ingest_question_level_sequences(csv_path: Path) -> IngestResult:
    interactions: list[dict[str, Any]] = []
    rejected_rows = 0
    rejected_interactions = 0
    parse_failures = 0
    source_rows = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            source_rows += 1
            try:
                fold = int(row["fold"])
                user_id = int(row["uid"])
                questions = _parse_int_sequence(row["questions"])
                responses = _parse_int_sequence(row["responses"])
                timestamps = _parse_int_sequence(row["timestamps"])
                masks = _parse_int_sequence(row.get("selectmasks", ""))
                concepts = _parse_concept_sequence(row["concepts"])
            except (ValueError, KeyError):
                parse_failures += 1
                rejected_rows += 1
                continue

            if not masks:
                masks = [1] * len(questions)

            lengths = {len(questions), len(responses), len(timestamps), len(masks), len(concepts)}
            if len(lengths) != 1:
                rejected_rows += 1
                continue

            for t_idx, (question_id, response, timestamp, mask, concept_ids) in enumerate(
                zip(questions, responses, timestamps, masks, concepts, strict=True)
            ):
                # Masked, padded or malformed interactions are dropped.
                if mask != 1:
                    rejected_interactions += 1
                    continue
                if question_id < 0 or timestamp < 0 or response not in (0, 1):
                    rejected_interactions += 1
                    continue

                interactions.append(
                    {
                        "fold": fold,
                        "user_id": user_id,
                        "sequence_row": row_idx,
                        "t_idx": t_idx,
                        "question_id": question_id,
                        "response": response,
                        "timestamp": timestamp,
                        "concept_ids": concept_ids,
                    }
                )

    return IngestResult(
        interactions=interactions,
        rejected_rows=rejected_rows,
        rejected_interactions=rejected_interactions,
        parse_failures=parse_failures,
        source_rows=source_rows,
    )


def build_question_level_long_table(
    raw_root: Path,
    output_root: Path,
    source_filename: str = "train_valid_sequences_quelevel.csv",
    output_basename: str = "question_level_long",
) -> dict[str, Any]:
    csv_path = raw_root / "question_level" / source_filename
    ingest = ingest_question_level_sequences(csv_path)

    ensure_dirs(output_root)
    long_table_path = output_root / f"{output_basename}.jsonl"
    write_jsonl(long_table_path, ingest.interactions)

    summary = {
        "source_file": str(csv_path),
        "source_rows": ingest.source_rows,
        "parsed_interactions": len(ingest.interactions),
        "rejected_rows": ingest.rejected_rows,
        "rejected_interactions": ingest.rejected_interactions,
        "parse_failures": ingest.parse_failures,
        "long_table_path": str(long_table_path),
    }
    write_json(output_root / f"{output_basename}_ingest_summary.json", summary)
    return summary
