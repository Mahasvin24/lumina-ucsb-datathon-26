from __future__ import annotations

import csv
import statistics
from io import StringIO
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from backend.app.analytics import build_dashboard_data
from backend.app.inference import load_model, predict_probabilities, resolve_student_artifact
from backend.app.parsing import parse_enriched_csv


router = APIRouter()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _discover_student_ids(students_dir: Path) -> list[int]:
    ids: list[int] = []
    for csv_path in sorted(students_dir.glob("student_*.csv")):
        suffix = csv_path.stem.replace("student_", "", 1)
        try:
            student_id = int(suffix)
        except ValueError:
            continue
        if student_id > 0:
            ids.append(student_id)
    return sorted(set(ids))


def _load_test_question_concepts() -> dict[int, set[str]]:
    test_path = REPO_ROOT / "students/test_questions.csv"
    mapping: dict[int, set[str]] = {}
    if not test_path.exists():
        return mapping
    with test_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                qid = int(str(row.get("question_id", "")).strip())
            except (TypeError, ValueError):
                continue
            raw = str(row.get("concept_ids", "") or "")
            concepts = {part.strip() for part in raw.split("|") if part.strip()}
            mapping[qid] = concepts
    return mapping


def _load_test_question_aliases() -> dict[int, set[int]]:
    test_path = REPO_ROOT / "students/test_questions.csv"
    aliases: dict[int, set[int]] = {}
    if not test_path.exists():
        return aliases
    with test_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                qid = int(str(row.get("question_id", "")).strip())
            except (TypeError, ValueError):
                continue
            alias_set = aliases.setdefault(qid, {qid})
            try:
                qidx = int(str(row.get("question_idx", "")).strip())
                if qidx > 0:
                    alias_set.add(qidx)
            except (TypeError, ValueError):
                pass
    return aliases


class ProcessQuestionsRequest(BaseModel):
    question_ids: list[int]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict")
async def predict(student_id: int = Form(...), file: UploadFile = File(...)) -> dict[str, object]:
    content = await file.read()
    batch = parse_enriched_csv(content)
    artifact_path, summary_row = resolve_student_artifact(REPO_ROOT, student_id)
    loaded = load_model(REPO_ROOT, artifact_path)
    probabilities = predict_probabilities(loaded, batch)
    return {
        "student_id": student_id,
        "user_id": summary_row.get("user_id"),
        "probabilities": probabilities,
    }


@router.get("/dashboard-data")
def dashboard_data(
    student_threshold: float = Query(default=70.0, ge=0.0, le=100.0),
    question_threshold: float = Query(default=60.0, ge=0.0, le=100.0),
    exclude_unanswered: bool = Query(default=False),
    top_n_hardest: int = Query(default=15, ge=1, le=200),
) -> dict[str, object]:
    try:
        return build_dashboard_data(
            repo_root=REPO_ROOT,
            student_threshold=student_threshold,
            question_threshold=question_threshold,
            exclude_unanswered=exclude_unanswered,
            top_n_hardest=top_n_hardest,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/process-selected-questions")
def process_selected_questions(payload: ProcessQuestionsRequest) -> dict[str, object]:
    selected = sorted({int(question_id) for question_id in payload.question_ids if int(question_id) > 0})
    if not selected:
        raise HTTPException(status_code=400, detail="Provide at least one valid question_id.")

    students_dir = REPO_ROOT / "students"
    student_ids = _discover_student_ids(students_dir)
    if not student_ids:
        raise HTTPException(status_code=404, detail=f"No student CSV files found in {students_dir}.")

    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    total_rows = 0
    test_question_concepts = _load_test_question_concepts()
    test_question_aliases = _load_test_question_aliases()

    for student_id in student_ids:
        csv_path = students_dir / f"student_{student_id}.csv"
        try:
            df = pd.read_csv(csv_path)
            has_question_id = "question_id" in df.columns
            has_question_idx = "question_idx" in df.columns
            if not has_question_id and not has_question_idx:
                raise HTTPException(
                    status_code=400,
                    detail=f"{csv_path.name} must include question_id or question_idx.",
                )

            # Always run inference on the full student CSV so each student can
            # return a score for every selected test question.
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            batch = parse_enriched_csv(csv_buffer.getvalue().encode("utf-8"))
            artifact_path, summary_row = resolve_student_artifact(REPO_ROOT, student_id)
            loaded = load_model(REPO_ROOT, artifact_path)
            probabilities = predict_probabilities(loaded, batch)
            if not probabilities:
                raise HTTPException(status_code=400, detail=f"No probability outputs for {csv_path.name}.")

            overall_mean = float(statistics.mean(probabilities))
            question_probs: dict[int, list[float]] = {qid: [] for qid in selected}
            concept_probs: dict[int, list[float]] = {qid: [] for qid in selected}
            selected_aliases = {qid: test_question_aliases.get(qid, {qid}) for qid in selected}
            for (_, row), prob in zip(df.iterrows(), probabilities):
                qid = int(row["question_id"]) if has_question_id and not pd.isna(row["question_id"]) else -1
                qidx = int(row["question_idx"]) if has_question_idx and not pd.isna(row["question_idx"]) else -1
                p = float(prob)
                for target_qid, aliases in selected_aliases.items():
                    if qid in aliases or qidx in aliases:
                        question_probs[target_qid].append(p)

                row_concepts = {part.strip() for part in str(row.get("concept_ids", "")).split("|") if part.strip()}
                if not row_concepts:
                    continue
                for target_qid in selected:
                    target_concepts = test_question_concepts.get(target_qid, set())
                    if target_concepts and (row_concepts & target_concepts):
                        concept_probs[target_qid].append(p)

            question_summaries = [
                {
                    "question_id": qid,
                    "rows": len(values),
                    "avg_probability": (
                        round(statistics.mean(values), 4)
                        if values
                        else (
                            round(statistics.mean(concept_probs.get(qid, [])), 4)
                            if concept_probs.get(qid)
                            else round(overall_mean, 4)
                        )
                    ),
                    "used_fallback": len(values) == 0 and not concept_probs.get(qid),
                    "used_concept_proxy": len(values) == 0 and bool(concept_probs.get(qid)),
                }
                for qid, values in question_probs.items()
            ]
            processed_rows = len(selected)
            total_rows += processed_rows

            results.append(
                {
                    "student_id": student_id,
                    "user_id": summary_row.get("user_id"),
                    "processed_rows": processed_rows,
                    "matched_rows": int(
                        sum(
                            1
                            for item in question_summaries
                            if (not item["used_fallback"]) and (not item["used_concept_proxy"])
                        )
                    ),
                    "proxy_rows": int(sum(1 for item in question_summaries if item["used_concept_proxy"])),
                    "question_summaries": question_summaries,
                }
            )
        except HTTPException as exc:
            errors.append({"student_id": student_id, "error": str(exc.detail)})
        except Exception as exc:  # noqa: BLE001
            errors.append({"student_id": student_id, "error": f"Unexpected error: {exc}"})

    if not results and errors:
        raise HTTPException(status_code=500, detail={"message": "All student processing failed.", "errors": errors})

    return {
        "selected_question_ids": selected,
        "students": results,
        "total_processed_rows": total_rows,
        "errors": errors,
    }

