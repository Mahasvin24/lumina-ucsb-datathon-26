from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _required_columns() -> set[str]:
    return {
        "interaction_id",
        "question_idx",
        "concept_indices",
        "concept_mask",
        "delta_t_log_bin",
        "question_type",
        "num_concepts",
        "concept_rarity_bucket",
        "hist_accuracy",
        "question_difficulty_prior",
        "concept_difficulty_prior",
        "content_embedding_missing",
        "analysis_embedding_missing",
        "content_vec_0",
        "content_vec_31",
        "analysis_vec_0",
        "analysis_vec_31",
    }


def _load_summary(weights_dir: Path) -> list[dict[str, Any]]:
    summary_path = weights_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {summary_path}, got {type(payload).__name__}")
    return payload


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate student CSV and memory-weight artifacts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root path (defaults to parent of students/).",
    )
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required. Run with model-training/.venv/bin/python students/validate_students.py"
        ) from exc

    root = args.root.resolve()
    students_dir = root / "students"
    weights_dir = students_dir / "student_weights"
    summary = _load_summary(weights_dir)
    _assert(len(summary) == 5, f"Expected 5 students in summary, found {len(summary)}.")

    required_cols = _required_columns()
    memory_signatures: list[tuple[float, float]] = []
    report_rows: list[str] = []

    for idx, item in enumerate(summary, start=1):
        csv_path = students_dir / f"student_{idx}.csv"
        memory_path = Path(str(item.get("memory_state_path", "")))
        model_with_memory_path = Path(str(item.get("model_with_memory_path", "")))

        _assert(csv_path.exists(), f"Missing CSV: {csv_path}")
        _assert(memory_path.exists(), f"Missing memory-state file: {memory_path}")
        _assert(model_with_memory_path.exists(), f"Missing model-with-memory file: {model_with_memory_path}")

        model_checkpoint = torch.load(model_with_memory_path, map_location="cpu")
        _assert("num_questions" in model_checkpoint, f"{model_with_memory_path}: missing num_questions")
        num_questions = int(model_checkpoint["num_questions"])

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing_columns = sorted(required_cols - fieldnames)
            _assert(not missing_columns, f"{csv_path.name} missing columns: {missing_columns}")

            rows = list(reader)
            expected_rows = int(item["feature_rows"])
            _assert(
                len(rows) == expected_rows,
                f"{csv_path.name} row mismatch: csv={len(rows)}, summary={expected_rows}",
            )

            # Spot check interaction_id formula used by the DKT pipeline.
            for row in rows[:50]:
                qidx = int(row["question_idx"])
                resp = int(row["response"])
                interaction_id = int(row["interaction_id"])
                expected = qidx + (resp * num_questions)
                _assert(
                    interaction_id == expected,
                    f"{csv_path.name} interaction_id mismatch: got={interaction_id}, expected={expected}",
                )

        memory_payload = torch.load(memory_path, map_location="cpu")
        h_n = memory_payload.get("short_term_hidden_h_n")
        c_n = memory_payload.get("long_term_cell_c_n")
        _assert(h_n is not None and c_n is not None, f"{memory_path.name} missing h_n/c_n")
        _assert(
            isinstance(h_n, torch.Tensor) and isinstance(c_n, torch.Tensor),
            f"{memory_path.name} h_n/c_n must be tensors",
        )
        _assert(h_n.shape == c_n.shape, f"{memory_path.name} h_n/c_n shape mismatch: {h_n.shape} vs {c_n.shape}")
        _assert(
            h_n.ndim == 3 and h_n.shape[1] == 1,
            f"{memory_path.name} unexpected memory shape: {h_n.shape}",
        )

        _assert("model_state_dict" in model_checkpoint, f"{model_with_memory_path} missing model_state_dict")
        _assert(
            "student_short_term_hidden_h_n" in model_checkpoint and "student_long_term_cell_c_n" in model_checkpoint,
            f"{model_with_memory_path} missing persisted student memory tensors",
        )
        _assert(
            torch.allclose(model_checkpoint["student_short_term_hidden_h_n"], h_n),
            f"{model_with_memory_path.name} stored h_n does not match standalone memory file",
        )
        _assert(
            torch.allclose(model_checkpoint["student_long_term_cell_c_n"], c_n),
            f"{model_with_memory_path.name} stored c_n does not match standalone memory file",
        )

        memory_signatures.append((round(float(h_n.abs().sum().item()), 6), round(float(c_n.abs().sum().item()), 6)))
        report_rows.append(
            f"- student_{idx}.csv: rows={len(rows)}, fold={item.get('fold_id')}, memory_shape={tuple(h_n.shape)}"
        )

    _assert(len(set(memory_signatures)) > 1, "All student memory states are identical.")

    print("PASS: student artifact validation completed.")
    for line in report_rows:
        print(line)
    print(f"Unique memory signatures: {len(set(memory_signatures))}")


if __name__ == "__main__":
    main()
