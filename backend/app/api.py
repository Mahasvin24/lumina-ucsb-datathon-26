from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.app.analytics import build_dashboard_data
from backend.app.inference import load_model, predict_probabilities, resolve_student_artifact
from backend.app.parsing import parse_enriched_csv


router = APIRouter()
REPO_ROOT = Path(__file__).resolve().parents[2]


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

