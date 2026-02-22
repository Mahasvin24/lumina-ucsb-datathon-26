from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import HTTPException


@dataclass
class LoadedModel:
    model: Any
    num_questions: int
    artifact_path: Path


_MODEL_CACHE: dict[str, LoadedModel] = {}


def _load_train_module(repo_root: Path) -> Any:
    module_path = repo_root / "model-training/src/modeling/train_dkt_lstm.py"
    module_name = "train_dkt_lstm_backend_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_student_summary(repo_root: Path) -> list[dict[str, Any]]:
    summary_path = repo_root / "students/student_weights/summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing summary file: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="Invalid student summary format.")
    return payload


def resolve_student_artifact(repo_root: Path, student_id: int) -> tuple[Path, dict[str, Any]]:
    summary = _read_student_summary(repo_root)
    selected: dict[str, Any] | None = None
    for item in summary:
        student_file = str(item.get("student_file", ""))
        if student_file == f"student_{student_id}.csv":
            selected = item
            break
    if selected is None:
        raise HTTPException(status_code=404, detail=f"Unknown student_id={student_id}.")

    artifact_raw = selected.get("model_with_memory_path")
    if not isinstance(artifact_raw, str) or not artifact_raw:
        raise HTTPException(status_code=500, detail="Student summary missing model_with_memory_path.")
    artifact_path = Path(artifact_raw)
    if not artifact_path.is_absolute():
        artifact_path = repo_root / artifact_path
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f"Model artifact missing: {artifact_path}")
    return artifact_path, selected


def load_model(repo_root: Path, artifact_path: Path) -> LoadedModel:
    cache_key = str(artifact_path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    checkpoint = torch.load(artifact_path, map_location="cpu")
    train_module = _load_train_module(repo_root)
    model_config = checkpoint.get("model_config", {})
    model = train_module.DKTLSTM(
        num_questions=int(checkpoint["num_questions"]),
        concept_vocab_size=int(checkpoint.get("concept_vocab_size", 1)),
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        concept_embedding_dim=int(model_config.get("concept_embedding_dim", 16)),
        delta_t_embedding_dim=int(model_config.get("delta_t_embedding_dim", 8)),
        question_type_embedding_dim=int(model_config.get("question_type_embedding_dim", 4)),
        numeric_projection_dim=int(model_config.get("numeric_projection_dim", 16)),
        semantic_projection_dim=int(model_config.get("semantic_projection_dim", 32)),
        semantic_vector_dim=int(model_config.get("semantic_vector_dim", 32)),
        use_tags=bool(model_config.get("use_tags", True)),
        use_semantic_vectors=bool(model_config.get("use_semantic_vectors", True)),
        use_numeric_features=bool(model_config.get("use_numeric_features", True)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loaded = LoadedModel(model=model, num_questions=int(checkpoint["num_questions"]), artifact_path=artifact_path)
    _MODEL_CACHE[cache_key] = loaded
    return loaded


def predict_probabilities(loaded: LoadedModel, batch: dict[str, torch.Tensor]) -> list[float]:
    model = loaded.model
    with torch.no_grad():
        logits = model(
            interaction_ids=batch["interaction_ids"],
            concept_indices=batch["concept_indices"],
            concept_mask=batch["concept_mask"],
            delta_t_log_bin=batch["delta_t_log_bin"],
            question_type=batch["question_type"],
            num_concepts=batch["num_concepts"],
            concept_rarity_bucket=batch["concept_rarity_bucket"],
            hist_accuracy=batch["hist_accuracy"],
            question_difficulty_prior=batch["question_difficulty_prior"],
            concept_difficulty_prior=batch["concept_difficulty_prior"],
            content_embedding_missing=batch["content_embedding_missing"],
            analysis_embedding_missing=batch["analysis_embedding_missing"],
            content_vectors=batch["content_vectors"],
            analysis_vectors=batch["analysis_vectors"],
        )
        question_idx = batch["question_idx"].clamp(min=0, max=loaded.num_questions - 1)
        gathered = logits.gather(dim=2, index=question_idx.unsqueeze(-1)).squeeze(-1)
        probs = torch.sigmoid(gathered).squeeze(0)
    return probs.cpu().numpy().tolist()

