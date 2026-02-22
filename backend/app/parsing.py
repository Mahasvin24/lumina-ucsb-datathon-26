from __future__ import annotations

from io import BytesIO

import pandas as pd
import torch
from fastapi import HTTPException


def _split_int_list(value: object) -> list[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(part) for part in text.split("|") if part]


def _split_float_list(value: object) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [float(part) for part in text.split("|") if part]


def parse_enriched_csv(content: bytes) -> dict[str, torch.Tensor]:
    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows.")

    required_cols = {
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
    }
    content_vec_cols = sorted([col for col in df.columns if col.startswith("content_vec_")])
    analysis_vec_cols = sorted([col for col in df.columns if col.startswith("analysis_vec_")])
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")
    if not content_vec_cols or not analysis_vec_cols:
        raise HTTPException(status_code=400, detail="Missing semantic vector columns (content_vec_* / analysis_vec_*).")
    if len(content_vec_cols) != len(analysis_vec_cols):
        raise HTTPException(status_code=400, detail="content_vec_* and analysis_vec_* dimensions do not match.")

    interaction_ids = torch.tensor(df["interaction_id"].astype(int).to_numpy(), dtype=torch.long).unsqueeze(0)
    question_idx = torch.tensor(df["question_idx"].astype(int).to_numpy(), dtype=torch.long).unsqueeze(0)
    delta_t_log_bin = torch.tensor(df["delta_t_log_bin"].astype(int).to_numpy(), dtype=torch.long).unsqueeze(0)
    question_type = torch.tensor(df["question_type"].astype(int).to_numpy(), dtype=torch.long).unsqueeze(0)
    num_concepts = torch.tensor(df["num_concepts"].astype(float).to_numpy(), dtype=torch.float32).unsqueeze(0)
    concept_rarity_bucket = (
        torch.tensor(df["concept_rarity_bucket"].astype(float).to_numpy(), dtype=torch.float32).unsqueeze(0)
    )
    hist_accuracy = torch.tensor(df["hist_accuracy"].astype(float).to_numpy(), dtype=torch.float32).unsqueeze(0)
    question_difficulty_prior = (
        torch.tensor(df["question_difficulty_prior"].astype(float).to_numpy(), dtype=torch.float32).unsqueeze(0)
    )
    concept_difficulty_prior = (
        torch.tensor(df["concept_difficulty_prior"].astype(float).to_numpy(), dtype=torch.float32).unsqueeze(0)
    )
    content_embedding_missing = (
        torch.tensor(df["content_embedding_missing"].astype(float).to_numpy(), dtype=torch.float32).unsqueeze(0)
    )
    analysis_embedding_missing = (
        torch.tensor(df["analysis_embedding_missing"].astype(float).to_numpy(), dtype=torch.float32).unsqueeze(0)
    )

    concept_lists = [_split_int_list(value) for value in df["concept_indices"].tolist()]
    mask_lists = [_split_float_list(value) for value in df["concept_mask"].tolist()]
    max_concepts = max((len(values) for values in concept_lists), default=1)
    max_concepts = max(1, max_concepts)
    concept_indices_values: list[list[int]] = []
    concept_mask_values: list[list[float]] = []
    for idx, values in enumerate(concept_lists):
        mask_values = mask_lists[idx]
        if mask_values and len(mask_values) != len(values):
            raise HTTPException(
                status_code=400,
                detail=f"concept_mask length mismatch at row {idx}: {len(mask_values)} != {len(values)}",
            )
        padded_values = values[:max_concepts] + [0] * max(0, max_concepts - len(values))
        if mask_values:
            padded_mask = mask_values[:max_concepts] + [0.0] * max(0, max_concepts - len(mask_values))
        else:
            padded_mask = [1.0] * min(len(values), max_concepts) + [0.0] * max(0, max_concepts - len(values))
        concept_indices_values.append(padded_values)
        concept_mask_values.append(padded_mask)

    concept_indices = torch.tensor(concept_indices_values, dtype=torch.long).unsqueeze(0)
    concept_mask = torch.tensor(concept_mask_values, dtype=torch.float32).unsqueeze(0)

    content_vectors = torch.tensor(df[content_vec_cols].astype(float).to_numpy(), dtype=torch.float32).unsqueeze(0)
    analysis_vectors = torch.tensor(df[analysis_vec_cols].astype(float).to_numpy(), dtype=torch.float32).unsqueeze(0)

    return {
        "interaction_ids": interaction_ids,
        "question_idx": question_idx,
        "concept_indices": concept_indices,
        "concept_mask": concept_mask,
        "delta_t_log_bin": delta_t_log_bin,
        "question_type": question_type,
        "num_concepts": num_concepts,
        "concept_rarity_bucket": concept_rarity_bucket,
        "hist_accuracy": hist_accuracy,
        "question_difficulty_prior": question_difficulty_prior,
        "concept_difficulty_prior": concept_difficulty_prior,
        "content_embedding_missing": content_embedding_missing,
        "analysis_embedding_missing": analysis_embedding_missing,
        "content_vectors": content_vectors,
        "analysis_vectors": analysis_vectors,
    }

