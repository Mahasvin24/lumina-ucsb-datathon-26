from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

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

