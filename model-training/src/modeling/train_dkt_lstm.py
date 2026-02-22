from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path(".cache") / "matplotlib").resolve())

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def load_semantic_embedding_tables(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]:
    if not path.exists():
        return {}, {}, 0
    payload = read_json(path)
    semantic_dim = int(payload.get("projection_dim", 0))
    content_payload = payload.get("qid2content_emb", {})
    analysis_payload = payload.get("qid2analysis_emb", {})

    content_embeddings: dict[int, np.ndarray] = {}
    analysis_embeddings: dict[int, np.ndarray] = {}
    if isinstance(content_payload, dict):
        for key, value in content_payload.items():
            if not isinstance(value, list):
                continue
            content_embeddings[int(key)] = np.asarray(value, dtype=np.float32)
    if isinstance(analysis_payload, dict):
        for key, value in analysis_payload.items():
            if not isinstance(value, list):
                continue
            analysis_embeddings[int(key)] = np.asarray(value, dtype=np.float32)
    if semantic_dim <= 0:
        if content_embeddings:
            semantic_dim = int(len(next(iter(content_embeddings.values()))))
        elif analysis_embeddings:
            semantic_dim = int(len(next(iter(analysis_embeddings.values()))))
    return content_embeddings, analysis_embeddings, semantic_dim


class DKTWindowDataset(Dataset):
    def __init__(
        self,
        window_path: Path,
        num_questions: int,
        semantic_content_embeddings: dict[int, np.ndarray] | None = None,
        semantic_analysis_embeddings: dict[int, np.ndarray] | None = None,
        semantic_dim: int = 0,
    ) -> None:
        self.num_questions = num_questions
        self.rows = read_jsonl(window_path)
        if not self.rows:
            raise ValueError(f"No rows found in {window_path}")

        max_concepts_per_timestep = 1
        for row in self.rows:
            seq_len = len(row["question_idx"])
            concept_indices_raw = row.get("concept_indices", [[] for _ in range(seq_len)])
            if len(concept_indices_raw) != seq_len:
                concept_indices_raw = (concept_indices_raw[:seq_len]) + ([[]] * max(0, seq_len - len(concept_indices_raw)))
            max_concepts_per_timestep = max(
                max_concepts_per_timestep,
                max((len(concepts) for concepts in concept_indices_raw), default=0),
            )
        self.max_concepts_per_timestep = max_concepts_per_timestep

        self.semantic_dim = max(
            int(semantic_dim),
            len(next(iter(semantic_content_embeddings.values()))) if semantic_content_embeddings else 0,
            len(next(iter(semantic_analysis_embeddings.values()))) if semantic_analysis_embeddings else 0,
        )
        max_qid = -1
        if semantic_content_embeddings:
            max_qid = max(max_qid, max(semantic_content_embeddings.keys()))
        if semantic_analysis_embeddings:
            max_qid = max(max_qid, max(semantic_analysis_embeddings.keys()))
        lookup_size = max(0, max_qid + 1)
        content_lookup = np.zeros((lookup_size, self.semantic_dim), dtype=np.float32)
        analysis_lookup = np.zeros((lookup_size, self.semantic_dim), dtype=np.float32)
        if semantic_content_embeddings:
            for qid, vec in semantic_content_embeddings.items():
                if 0 <= qid < lookup_size:
                    content_lookup[qid] = vec[: self.semantic_dim]
        if semantic_analysis_embeddings:
            for qid, vec in semantic_analysis_embeddings.items():
                if 0 <= qid < lookup_size:
                    analysis_lookup[qid] = vec[: self.semantic_dim]

        self.content_embedding_lookup = torch.tensor(content_lookup, dtype=torch.float32)
        self.analysis_embedding_lookup = torch.tensor(analysis_lookup, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        q = torch.tensor(row["question_idx"], dtype=torch.long)
        seq_len = int(q.shape[0])
        r = torch.tensor(row.get("responses", [0] * seq_len), dtype=torch.long)
        in_mask = torch.tensor(row.get("input_mask", [1] * seq_len), dtype=torch.long)
        interaction_ids = q + (r * self.num_questions)
        interaction_ids[in_mask == 0] = 0
        next_question_idx = torch.zeros_like(q)
        next_question_idx[:-1] = q[1:]
        targets = torch.tensor(row.get("target_next_response", [0.0] * seq_len), dtype=torch.float32)
        target_masks = torch.tensor(row.get("target_mask", [0] * seq_len), dtype=torch.float32)

        delta_t_log_bin = torch.tensor(row.get("delta_t_log_bin", [0] * seq_len), dtype=torch.long)
        num_concepts = torch.tensor(row.get("num_concepts", [0] * seq_len), dtype=torch.float32)
        concept_rarity_bucket = torch.tensor(row.get("concept_rarity_bucket", [0] * seq_len), dtype=torch.float32)
        hist_accuracy = torch.tensor(row.get("hist_accuracy", [0.0] * seq_len), dtype=torch.float32)
        question_difficulty_prior = torch.tensor(
            row.get("question_difficulty_prior", [0.0] * seq_len),
            dtype=torch.float32,
        )
        concept_difficulty_prior = torch.tensor(
            row.get("concept_difficulty_prior", [0.0] * seq_len),
            dtype=torch.float32,
        )
        question_type = torch.tensor(row.get("question_type", [-1] * seq_len), dtype=torch.long)
        content_missing = torch.tensor(row.get("content_embedding_missing", [1] * seq_len), dtype=torch.float32)
        analysis_missing = torch.tensor(row.get("analysis_embedding_missing", [1] * seq_len), dtype=torch.float32)
        content_qid = torch.tensor(row.get("content_embedding_qid", [-1] * seq_len), dtype=torch.long)
        analysis_qid = torch.tensor(row.get("analysis_embedding_qid", [-1] * seq_len), dtype=torch.long)

        concept_indices_raw = row.get("concept_indices", [[] for _ in range(seq_len)])
        if len(concept_indices_raw) != seq_len:
            concept_indices_raw = (concept_indices_raw[:seq_len]) + ([[]] * max(0, seq_len - len(concept_indices_raw)))
        concept_indices = torch.zeros((seq_len, self.max_concepts_per_timestep), dtype=torch.long)
        concept_mask = torch.zeros((seq_len, self.max_concepts_per_timestep), dtype=torch.float32)
        for ts_idx, concepts in enumerate(concept_indices_raw):
            if not concepts:
                continue
            limit = min(self.max_concepts_per_timestep, len(concepts))
            concept_indices[ts_idx, :limit] = torch.tensor(concepts[:limit], dtype=torch.long)
            concept_mask[ts_idx, :limit] = 1.0

        content_vec = torch.zeros((seq_len, self.semantic_dim), dtype=torch.float32)
        analysis_vec = torch.zeros((seq_len, self.semantic_dim), dtype=torch.float32)
        if self.content_embedding_lookup.shape[0] > 0 and self.semantic_dim > 0:
            valid_content = (content_qid >= 0) & (content_qid < self.content_embedding_lookup.shape[0])
            if valid_content.any():
                content_vec[valid_content] = self.content_embedding_lookup[content_qid[valid_content]]
        if self.analysis_embedding_lookup.shape[0] > 0 and self.semantic_dim > 0:
            valid_analysis = (analysis_qid >= 0) & (analysis_qid < self.analysis_embedding_lookup.shape[0])
            if valid_analysis.any():
                analysis_vec[valid_analysis] = self.analysis_embedding_lookup[analysis_qid[valid_analysis]]

        return {
            "interaction_ids": interaction_ids,
            "next_question_idx": next_question_idx,
            "targets": targets,
            "target_masks": target_masks,
            "delta_t_log_bin": delta_t_log_bin,
            "num_concepts": num_concepts,
            "concept_rarity_bucket": concept_rarity_bucket,
            "hist_accuracy": hist_accuracy,
            "question_difficulty_prior": question_difficulty_prior,
            "concept_difficulty_prior": concept_difficulty_prior,
            "question_type": question_type,
            "content_embedding_missing": content_missing,
            "analysis_embedding_missing": analysis_missing,
            "content_vectors": content_vec,
            "analysis_vectors": analysis_vec,
            "concept_indices": concept_indices,
            "concept_mask": concept_mask,
        }


class DKTLSTM(nn.Module):
    def __init__(
        self,
        num_questions: int,
        concept_vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        concept_embedding_dim: int,
        delta_t_embedding_dim: int,
        question_type_embedding_dim: int,
        numeric_projection_dim: int,
        semantic_projection_dim: int,
        semantic_vector_dim: int,
        use_tags: bool,
        use_semantic_vectors: bool,
        use_numeric_features: bool,
    ) -> None:
        super().__init__()
        self.num_questions = num_questions
        self.use_tags = use_tags
        self.use_semantic_vectors = use_semantic_vectors
        self.use_numeric_features = use_numeric_features
        self.interaction_embedding = nn.Embedding(num_questions * 2, embedding_dim, padding_idx=0)

        if self.use_tags:
            self.concept_embedding = nn.Embedding(max(1, concept_vocab_size), concept_embedding_dim, padding_idx=0)

        if self.use_numeric_features:
            self.delta_t_embedding = nn.Embedding(21, delta_t_embedding_dim, padding_idx=0)
            self.question_type_embedding = nn.Embedding(3, question_type_embedding_dim)
            # num_concepts, concept_rarity_bucket, hist_accuracy, q_prior, c_prior, content_missing, analysis_missing
            self.numeric_projection = nn.Linear(7, numeric_projection_dim)

        if self.use_semantic_vectors:
            semantic_input_dim = (semantic_vector_dim * 2) + 2
            self.semantic_projection = nn.Linear(semantic_input_dim, semantic_projection_dim)

        lstm_input_size = embedding_dim
        if self.use_tags:
            lstm_input_size += concept_embedding_dim
        if self.use_numeric_features:
            lstm_input_size += delta_t_embedding_dim + question_type_embedding_dim + numeric_projection_dim
        if self.use_semantic_vectors:
            lstm_input_size += semantic_projection_dim

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, num_questions)

    def forward(
        self,
        interaction_ids: torch.Tensor,
        concept_indices: torch.Tensor,
        concept_mask: torch.Tensor,
        delta_t_log_bin: torch.Tensor,
        question_type: torch.Tensor,
        num_concepts: torch.Tensor,
        concept_rarity_bucket: torch.Tensor,
        hist_accuracy: torch.Tensor,
        question_difficulty_prior: torch.Tensor,
        concept_difficulty_prior: torch.Tensor,
        content_embedding_missing: torch.Tensor,
        analysis_embedding_missing: torch.Tensor,
        content_vectors: torch.Tensor,
        analysis_vectors: torch.Tensor,
    ) -> torch.Tensor:
        x_parts: list[torch.Tensor] = [self.interaction_embedding(interaction_ids)]

        if self.use_tags:
            concept_emb = self.concept_embedding(concept_indices)
            concept_mask_exp = concept_mask.unsqueeze(-1)
            concept_sum = (concept_emb * concept_mask_exp).sum(dim=2)
            concept_count = concept_mask_exp.sum(dim=2).clamp(min=1.0)
            concept_pooled = concept_sum / concept_count
            x_parts.append(concept_pooled)

        if self.use_numeric_features:
            delta_idx = delta_t_log_bin.clamp(min=0, max=20)
            delta_emb = self.delta_t_embedding(delta_idx)
            question_type_idx = (question_type + 1).clamp(min=0, max=2)
            question_type_emb = self.question_type_embedding(question_type_idx)
            numeric_raw = torch.stack(
                [
                    num_concepts,
                    concept_rarity_bucket,
                    hist_accuracy,
                    question_difficulty_prior,
                    concept_difficulty_prior,
                    content_embedding_missing,
                    analysis_embedding_missing,
                ],
                dim=-1,
            )
            numeric_proj = self.numeric_projection(numeric_raw)
            x_parts.extend([delta_emb, question_type_emb, numeric_proj])

        if self.use_semantic_vectors:
            semantic_input = torch.cat(
                [
                    content_vectors,
                    analysis_vectors,
                    content_embedding_missing.unsqueeze(-1),
                    analysis_embedding_missing.unsqueeze(-1),
                ],
                dim=-1,
            )
            semantic_proj = self.semantic_projection(semantic_input)
            x_parts.append(semantic_proj)

        x = torch.cat(x_parts, dim=-1)
        h, _ = self.lstm(x)
        h = self.dropout(h)
        return self.output(h)


def masked_bce_loss(
    gathered_logits: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    criterion: nn.BCEWithLogitsLoss,
) -> torch.Tensor:
    loss_per_step = criterion(gathered_logits, targets)
    masked = loss_per_step * masks
    denom = masks.sum().clamp(min=1.0)
    return masked.sum() / denom


def gather_next_question_logits(logits: torch.Tensor, next_question_idx: torch.Tensor) -> torch.Tensor:
    return logits.gather(dim=2, index=next_question_idx.unsqueeze(-1)).squeeze(-1)


@dataclass
class EvalResult:
    loss: float
    auc: float
    y_true: np.ndarray
    y_prob: np.ndarray


def evaluate(
    model: DKTLSTM,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    probs_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            interaction_ids = batch["interaction_ids"].to(device)
            next_question_idx = batch["next_question_idx"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["target_masks"].to(device)
            logits = model(
                interaction_ids=interaction_ids,
                concept_indices=batch["concept_indices"].to(device),
                concept_mask=batch["concept_mask"].to(device),
                delta_t_log_bin=batch["delta_t_log_bin"].to(device),
                question_type=batch["question_type"].to(device),
                num_concepts=batch["num_concepts"].to(device),
                concept_rarity_bucket=batch["concept_rarity_bucket"].to(device),
                hist_accuracy=batch["hist_accuracy"].to(device),
                question_difficulty_prior=batch["question_difficulty_prior"].to(device),
                concept_difficulty_prior=batch["concept_difficulty_prior"].to(device),
                content_embedding_missing=batch["content_embedding_missing"].to(device),
                analysis_embedding_missing=batch["analysis_embedding_missing"].to(device),
                content_vectors=batch["content_vectors"].to(device),
                analysis_vectors=batch["analysis_vectors"].to(device),
            )
            gathered_logits = gather_next_question_logits(logits, next_question_idx)
            loss = masked_bce_loss(gathered_logits, targets, masks, criterion)

            total_loss += float(loss.item())
            total_batches += 1

            prob = torch.sigmoid(gathered_logits)
            mask_bool = masks > 0
            probs_all.append(prob[mask_bool].detach().cpu().numpy())
            targets_all.append(targets[mask_bool].detach().cpu().numpy())

    y_prob = np.concatenate(probs_all) if probs_all else np.array([], dtype=np.float32)
    y_true = np.concatenate(targets_all) if targets_all else np.array([], dtype=np.float32)
    mean_loss = total_loss / max(1, total_batches)
    return EvalResult(loss=mean_loss, auc=safe_auc(y_true, y_prob), y_true=y_true, y_prob=y_prob)


def plot_loss_curve(train_loss: list[float], valid_loss: list[float], out_path: Path) -> None:
    epochs = list(range(1, len(train_loss) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="train_loss")
    ax.plot(epochs, valid_loss, label="valid_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_auc_curve(valid_auc: list[float], out_path: Path) -> None:
    epochs = list(range(1, len(valid_auc) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, valid_auc, label="valid_auc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC Curve")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    if y_true.size and np.unique(y_true).size >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_prediction_hist(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if y_true.size:
        ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="label_0")
        ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="label_1")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def list_available_folds(data_dir: Path) -> list[int]:
    folds: list[int] = []
    for path in sorted(data_dir.glob("fold_*")):
        if not path.is_dir():
            continue
        suffix = path.name.replace("fold_", "")
        if suffix.isdigit():
            folds.append(int(suffix))
    return folds


def load_feature_state(fold_dir: Path) -> dict[str, Any]:
    return read_json(fold_dir / "feature_state.json")


def build_loader(
    path: Path,
    num_questions: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    semantic_tables_path: Path,
    semantic_vector_dim: int,
) -> DataLoader:
    content_embeddings, analysis_embeddings, table_dim = load_semantic_embedding_tables(semantic_tables_path)
    resolved_semantic_dim = semantic_vector_dim if semantic_vector_dim > 0 else table_dim
    dataset = DKTWindowDataset(
        path,
        num_questions=num_questions,
        semantic_content_embeddings=content_embeddings,
        semantic_analysis_embeddings=analysis_embeddings,
        semantic_dim=resolved_semantic_dim,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_fold(args: argparse.Namespace, fold_id: int, run_dir: Path, device: torch.device) -> dict[str, Any]:
    fold_dir = args.data_dir / f"fold_{fold_id}"
    train_path = fold_dir / "train_windows.jsonl"
    valid_path = fold_dir / "valid_windows.jsonl"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(f"Missing train/valid windows for fold_{fold_id} in {fold_dir}")

    feature_state = load_feature_state(fold_dir)
    num_questions = int(feature_state["question_vocab_size"])
    concept_vocab_size = int(feature_state.get("concept_vocab_size", 1))
    semantic_tables_path = args.data_dir / "semantic_embedding_tables.json"
    train_loader = build_loader(
        train_path,
        num_questions=num_questions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        semantic_tables_path=semantic_tables_path,
        semantic_vector_dim=args.semantic_vector_dim,
    )
    valid_loader = build_loader(
        valid_path,
        num_questions=num_questions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        semantic_tables_path=semantic_tables_path,
        semantic_vector_dim=args.semantic_vector_dim,
    )

    model = DKTLSTM(
        num_questions=num_questions,
        concept_vocab_size=concept_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        concept_embedding_dim=args.concept_embedding_dim,
        delta_t_embedding_dim=args.delta_t_embedding_dim,
        question_type_embedding_dim=args.question_type_embedding_dim,
        numeric_projection_dim=args.numeric_projection_dim,
        semantic_projection_dim=args.semantic_projection_dim,
        semantic_vector_dim=args.semantic_vector_dim,
        use_tags=not args.disable_tags,
        use_semantic_vectors=not args.disable_semantic_vectors,
        use_numeric_features=not args.disable_numeric_features,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    fold_artifact_dir = run_dir / f"fold_{fold_id}"
    fold_artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_artifact_dir / "best_model.pt"

    history_train_loss: list[float] = []
    history_valid_loss: list[float] = []
    history_valid_auc: list[float] = []
    best_auc = -math.inf
    best_epoch = -1
    best_loss = math.inf
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_running = 0.0
        train_batches = 0

        progress = tqdm(train_loader, desc=f"fold_{fold_id} epoch_{epoch}", leave=False)
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)

            interaction_ids = batch["interaction_ids"].to(device)
            next_question_idx = batch["next_question_idx"].to(device)
            targets = batch["targets"].to(device)
            masks = batch["target_masks"].to(device)

            logits = model(
                interaction_ids=interaction_ids,
                concept_indices=batch["concept_indices"].to(device),
                concept_mask=batch["concept_mask"].to(device),
                delta_t_log_bin=batch["delta_t_log_bin"].to(device),
                question_type=batch["question_type"].to(device),
                num_concepts=batch["num_concepts"].to(device),
                concept_rarity_bucket=batch["concept_rarity_bucket"].to(device),
                hist_accuracy=batch["hist_accuracy"].to(device),
                question_difficulty_prior=batch["question_difficulty_prior"].to(device),
                concept_difficulty_prior=batch["concept_difficulty_prior"].to(device),
                content_embedding_missing=batch["content_embedding_missing"].to(device),
                analysis_embedding_missing=batch["analysis_embedding_missing"].to(device),
                content_vectors=batch["content_vectors"].to(device),
                analysis_vectors=batch["analysis_vectors"].to(device),
            )
            gathered_logits = gather_next_question_logits(logits, next_question_idx)
            loss = masked_bce_loss(gathered_logits, targets, masks, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss_running += float(loss.item())
            train_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = train_loss_running / max(1, train_batches)
        valid_eval = evaluate(model, valid_loader, criterion, device)
        valid_loss = valid_eval.loss
        valid_auc = valid_eval.auc

        history_train_loss.append(train_loss)
        history_valid_loss.append(valid_loss)
        history_valid_auc.append(valid_auc if not math.isnan(valid_auc) else 0.0)

        current_score = valid_auc if not math.isnan(valid_auc) else -math.inf
        improved = current_score > best_auc
        if not improved and math.isinf(best_auc):
            improved = valid_loss < best_loss

        if improved:
            best_auc = current_score
            best_epoch = epoch
            best_loss = valid_loss
            no_improve_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_questions": num_questions,
                    "embedding_dim": args.embedding_dim,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "concept_vocab_size": concept_vocab_size,
                    "model_config": {
                        "concept_embedding_dim": args.concept_embedding_dim,
                        "delta_t_embedding_dim": args.delta_t_embedding_dim,
                        "question_type_embedding_dim": args.question_type_embedding_dim,
                        "numeric_projection_dim": args.numeric_projection_dim,
                        "semantic_projection_dim": args.semantic_projection_dim,
                        "semantic_vector_dim": args.semantic_vector_dim,
                        "use_tags": not args.disable_tags,
                        "use_semantic_vectors": not args.disable_semantic_vectors,
                        "use_numeric_features": not args.disable_numeric_features,
                    },
                    "best_epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            no_improve_epochs += 1

        print(
            f"[fold_{fold_id}] epoch={epoch} train_loss={train_loss:.5f} "
            f"valid_loss={valid_loss:.5f} valid_auc={valid_auc:.5f}"
        )

        if args.patience > 0 and no_improve_epochs >= args.patience:
            print(f"[fold_{fold_id}] Early stopping at epoch {epoch}.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_eval = evaluate(model, valid_loader, criterion, device)

    plot_loss_curve(history_train_loss, history_valid_loss, fold_artifact_dir / "loss_curve.png")
    plot_auc_curve(history_valid_auc, fold_artifact_dir / "auc_curve.png")
    plot_roc_curve(final_eval.y_true, final_eval.y_prob, fold_artifact_dir / "roc_curve.png")
    plot_prediction_hist(final_eval.y_true, final_eval.y_prob, fold_artifact_dir / "prediction_histogram.png")

    metrics = {
        "fold_id": fold_id,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_valid_auc": float(final_eval.auc),
        "best_valid_loss": float(final_eval.loss),
        "epochs_trained": len(history_train_loss),
        "history": {
            "train_loss": history_train_loss,
            "valid_loss": history_valid_loss,
            "valid_auc": history_valid_auc,
        },
        "checkpoint_path": str(checkpoint_path),
    }
    write_json(fold_artifact_dir / "metrics.json", metrics)
    return metrics


def resolve_checkpoint_path(args: argparse.Namespace, fold_id: int, run_dir: Path) -> Path:
    if args.checkpoint and not args.all_folds:
        return args.checkpoint
    if args.checkpoint_root is not None:
        return args.checkpoint_root / f"fold_{fold_id}" / "best_model.pt"
    return run_dir / f"fold_{fold_id}" / "best_model.pt"


def evaluate_fold(args: argparse.Namespace, fold_id: int, run_dir: Path, device: torch.device) -> dict[str, Any]:
    fold_dir = args.data_dir / f"fold_{fold_id}"
    split_name = "valid" if args.eval_split == "valid" else "train"
    split_path = fold_dir / f"{split_name}_windows.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_name}_windows.jsonl for fold_{fold_id} in {fold_dir}")

    ckpt_path = resolve_checkpoint_path(args, fold_id, run_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for fold_{fold_id}: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    num_questions = int(checkpoint["num_questions"])
    model_config = checkpoint.get("model_config", {})
    semantic_tables_path = args.data_dir / "semantic_embedding_tables.json"
    semantic_vector_dim = int(model_config.get("semantic_vector_dim", args.semantic_vector_dim))
    loader = build_loader(
        split_path,
        num_questions=num_questions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        semantic_tables_path=semantic_tables_path,
        semantic_vector_dim=semantic_vector_dim,
    )

    model = DKTLSTM(
        num_questions=num_questions,
        concept_vocab_size=int(checkpoint.get("concept_vocab_size", 1)),
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        concept_embedding_dim=int(model_config.get("concept_embedding_dim", args.concept_embedding_dim)),
        delta_t_embedding_dim=int(model_config.get("delta_t_embedding_dim", args.delta_t_embedding_dim)),
        question_type_embedding_dim=int(model_config.get("question_type_embedding_dim", args.question_type_embedding_dim)),
        numeric_projection_dim=int(model_config.get("numeric_projection_dim", args.numeric_projection_dim)),
        semantic_projection_dim=int(model_config.get("semantic_projection_dim", args.semantic_projection_dim)),
        semantic_vector_dim=semantic_vector_dim,
        use_tags=bool(model_config.get("use_tags", False)),
        use_semantic_vectors=bool(model_config.get("use_semantic_vectors", False)),
        use_numeric_features=bool(model_config.get("use_numeric_features", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    result = evaluate(model, loader, criterion, device)

    fold_artifact_dir = run_dir / f"fold_{fold_id}"
    fold_artifact_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(result.y_true, result.y_prob, fold_artifact_dir / f"{split_name}_roc_curve.png")
    plot_prediction_hist(
        result.y_true,
        result.y_prob,
        fold_artifact_dir / f"{split_name}_prediction_histogram.png",
    )

    metrics = {
        "fold_id": fold_id,
        "eval_split": split_name,
        "auc": float(result.auc),
        "loss": float(result.loss),
        "checkpoint_path": str(ckpt_path),
    }
    write_json(fold_artifact_dir / f"{split_name}_eval_metrics.json", metrics)
    print(f"[fold_{fold_id}] eval_split={split_name} auc={result.auc:.5f} loss={result.loss:.5f}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM-based DKT model on processed windows.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/dkt_qlevel_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dkt_lstm"))
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--fold", type=int, default=0, help="Fold id for single-fold mode.")
    parser.add_argument("--all-folds", action="store_true", help="Run across all fold_* directories.")

    parser.add_argument("--eval-only", action="store_true", help="Skip training and only report AUC/loss.")
    parser.add_argument("--eval-split", choices=["train", "valid"], default="valid")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path for single-fold eval-only.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Directory containing fold_*/best_model.pt for all-fold eval-only.",
    )

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--concept-embedding-dim", type=int, default=32)
    parser.add_argument("--delta-t-embedding-dim", type=int, default=8)
    parser.add_argument("--question-type-embedding-dim", type=int, default=4)
    parser.add_argument("--numeric-projection-dim", type=int, default=16)
    parser.add_argument("--semantic-projection-dim", type=int, default=32)
    parser.add_argument("--semantic-vector-dim", type=int, default=32)
    parser.add_argument("--disable-tags", action="store_true")
    parser.add_argument("--disable-semantic-vectors", action="store_true")
    parser.add_argument("--disable-numeric-features", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available. Check your PyTorch install and macOS GPU support.")
        device = torch.device("mps")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this system.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    available_folds = list_available_folds(args.data_dir)
    if not available_folds:
        raise ValueError(f"No fold_* directories found under {args.data_dir}")

    fold_ids = available_folds if args.all_folds else [args.fold]
    missing = sorted(set(fold_ids) - set(available_folds))
    if missing:
        raise ValueError(f"Requested folds not found: {missing}; available={available_folds}")

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_reports: list[dict[str, Any]] = []
    if args.eval_only:
        for fold_id in fold_ids:
            fold_reports.append(evaluate_fold(args, fold_id, run_dir, device))
    else:
        for fold_id in fold_ids:
            fold_reports.append(train_fold(args, fold_id, run_dir, device))

    auc_key = "auc" if args.eval_only else "best_valid_auc"
    auc_values = np.array([float(report[auc_key]) for report in fold_reports], dtype=np.float64)
    finite_mask = np.isfinite(auc_values)
    summary = {
        "mode": "eval_only" if args.eval_only else "train",
        "run_name": args.run_name,
        "data_dir": str(args.data_dir),
        "output_dir": str(run_dir),
        "folds": fold_reports,
        "mean_auc": float(np.mean(auc_values[finite_mask])) if finite_mask.any() else float("nan"),
        "std_auc": float(np.std(auc_values[finite_mask])) if finite_mask.any() else float("nan"),
    }
    write_json(run_dir / "cv_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
