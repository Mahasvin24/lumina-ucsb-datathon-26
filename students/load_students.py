import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/Users/mahasvin/Github/lumina-ucsb-datathon-26-v2")
SRC = ROOT / "model-training/data/interim/question_level_long_clean.jsonl"
OUT_DIR = ROOT / "students"
PROCESSED_DIR = ROOT / "model-training/data/processed/dkt_qlevel_v1"
ARTIFACTS_ROOT = ROOT / "model-training/artifacts/dkt_lstm"
WEIGHTS_DIR = OUT_DIR / "student_weights"
PREFERRED_RUNS = ["full_train_all_folds_v2", "full_train_all_folds"]


def _read_first_user_ids(student_dir: Path) -> list[int]:
    student_files = [student_dir / f"student_{i}.csv" for i in range(1, 6)]
    user_ids: list[int] = []
    for file_path in student_files:
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
            if not first:
                raise RuntimeError(f"{file_path} has no data rows")
            user_ids.append(int(first["user_id"]))
    return user_ids


def _write_full_histories(user_ids: list[int]) -> dict[int, list[dict]]:
    uid_set = set(user_ids)
    rows_by_uid = {uid: [] for uid in uid_set}

    with SRC.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            uid = int(row["user_id"])
            if uid in uid_set:
                rows_by_uid[uid].append(row)

    fields = [
        "fold",
        "user_id",
        "sequence_row",
        "t_idx",
        "question_id",
        "response",
        "timestamp",
        "concept_ids",
    ]
    for idx, uid in enumerate(user_ids, start=1):
        out = OUT_DIR / f"student_{idx}.csv"
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows_by_uid[uid]:
                writer.writerow(
                    {
                        "fold": row["fold"],
                        "user_id": row["user_id"],
                        "sequence_row": row["sequence_row"],
                        "t_idx": row["t_idx"],
                        "question_id": row["question_id"],
                        "response": row["response"],
                        "timestamp": row["timestamp"],
                        "concept_ids": "_".join(str(x) for x in row.get("concept_ids", [])),
                    }
                )
    return rows_by_uid


def _load_rows_for_uid(path: Path, uid: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["user_id"]) == uid:
                rows.append(row)
    return rows


def _serialize_list(values: list[int]) -> str:
    return "|".join(str(v) for v in values)


def _load_train_module() -> Any:
    module_path = ROOT / "model-training/src/modeling/train_dkt_lstm.py"
    module_name = "train_dkt_lstm_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_semantic_tables() -> tuple[int, dict[str, list[float]], dict[str, list[float]]]:
    semantic_path = PROCESSED_DIR / "semantic_embedding_tables.json"
    with semantic_path.open("r", encoding="utf-8") as handle:
        semantic_payload = json.load(handle)
    emb_dim = int(semantic_payload.get("projection_dim", 0))
    content_lookup = semantic_payload.get("qid2content_emb", {})
    analysis_lookup = semantic_payload.get("qid2analysis_emb", {})
    return emb_dim, content_lookup, analysis_lookup


def _load_student_feature_rows(uid: int, fold_id: int) -> list[dict]:
    feature_state_path = PROCESSED_DIR / f"fold_{fold_id}" / "feature_state.json"
    with feature_state_path.open("r", encoding="utf-8") as handle:
        feature_state = json.load(handle)
    if int(feature_state["fold_id"]) != fold_id:
        raise RuntimeError(f"feature_state fold mismatch for fold_{fold_id}")

    fold_dir = PROCESSED_DIR / f"fold_{fold_id}"
    candidate_files = [fold_dir / "valid_features.jsonl", fold_dir / "train_features.jsonl"]
    feature_rows: list[dict] = []
    for file_path in candidate_files:
        feature_rows = _load_rows_for_uid(file_path, uid)
        if feature_rows:
            break
    if not feature_rows:
        raise RuntimeError(f"No feature rows found for user_id={uid} in fold_{fold_id}.")
    return feature_rows


def _resolve_checkpoint_path(fold_id: int) -> Path:
    for run_name in PREFERRED_RUNS:
        path = ARTIFACTS_ROOT / run_name / f"fold_{fold_id}" / "best_model.pt"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No checkpoint found for fold_{fold_id}. Checked runs: {', '.join(PREFERRED_RUNS)}"
    )


def _write_student_lstm_inputs_csv(
    student_index: int,
    feature_rows: list[dict],
    num_questions: int,
    emb_dim: int,
    content_lookup: dict[str, list[float]],
    analysis_lookup: dict[str, list[float]],
) -> int:
    zero_vec = [0.0] * emb_dim

    content_cols = [f"content_vec_{i}" for i in range(emb_dim)]
    analysis_cols = [f"analysis_vec_{i}" for i in range(emb_dim)]
    fields = [
        "fold",
        "user_id",
        "sequence_row",
        "t_idx",
        "question_id",
        "response",
        "timestamp",
        "concept_ids",
        "question_idx",
        "interaction_id",
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
    ] + content_cols + analysis_cols

    out = OUT_DIR / f"student_{student_index}.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in feature_rows:
            qid = int(row["question_id"])
            content_qid = int(row.get("content_embedding_qid", -1))
            analysis_qid = int(row.get("analysis_embedding_qid", -1))
            content_vec = content_lookup.get(str(content_qid), zero_vec) if content_qid >= 0 else zero_vec
            analysis_vec = analysis_lookup.get(str(analysis_qid), zero_vec) if analysis_qid >= 0 else zero_vec
            interaction_id = int(row["question_idx"]) + int(row["response"]) * num_questions

            concept_indices = row.get("concept_indices", [])
            concept_mask = [1] * len(concept_indices)
            out_row = {
                "fold": row["fold"],
                "user_id": row["user_id"],
                "sequence_row": row["sequence_row"],
                "t_idx": row["t_idx"],
                "question_id": qid,
                "response": row["response"],
                "timestamp": row["timestamp"],
                "concept_ids": _serialize_list(row.get("concept_ids", [])),
                "question_idx": row["question_idx"],
                "interaction_id": interaction_id,
                "concept_indices": _serialize_list(concept_indices),
                "concept_mask": _serialize_list(concept_mask),
                "delta_t_log_bin": row["delta_t_log_bin"],
                "question_type": row["question_type"],
                "num_concepts": row["num_concepts"],
                "concept_rarity_bucket": row["concept_rarity_bucket"],
                "hist_accuracy": row["hist_accuracy"],
                "question_difficulty_prior": row["question_difficulty_prior"],
                "concept_difficulty_prior": row["concept_difficulty_prior"],
                "content_embedding_missing": row["content_embedding_missing"],
                "analysis_embedding_missing": row["analysis_embedding_missing"],
            }
            for i in range(emb_dim):
                out_row[f"content_vec_{i}"] = content_vec[i]
                out_row[f"analysis_vec_{i}"] = analysis_vec[i]
            writer.writerow(out_row)
    return len(feature_rows)


def _rows_to_lstm_tensors(
    feature_rows: list[dict],
    num_questions: int,
    emb_dim: int,
    content_lookup: dict[str, list[float]],
    analysis_lookup: dict[str, list[float]],
    torch: Any,
) -> dict[str, Any]:
    seq_len = len(feature_rows)
    max_concepts = max((len(row.get("concept_indices", [])) for row in feature_rows), default=1)
    max_concepts = max(1, max_concepts)
    zero_vec = [0.0] * emb_dim

    interaction_ids = []
    delta_t_log_bin = []
    question_type = []
    num_concepts = []
    concept_rarity_bucket = []
    hist_accuracy = []
    question_difficulty_prior = []
    concept_difficulty_prior = []
    content_embedding_missing = []
    analysis_embedding_missing = []
    content_vectors = []
    analysis_vectors = []
    concept_indices = []
    concept_mask = []

    for row in feature_rows:
        qidx = int(row["question_idx"])
        response = int(row["response"])
        interaction_ids.append(qidx + (response * num_questions))
        delta_t_log_bin.append(int(row["delta_t_log_bin"]))
        question_type.append(int(row["question_type"]))
        num_concepts.append(float(row["num_concepts"]))
        concept_rarity_bucket.append(float(row["concept_rarity_bucket"]))
        hist_accuracy.append(float(row["hist_accuracy"]))
        question_difficulty_prior.append(float(row["question_difficulty_prior"]))
        concept_difficulty_prior.append(float(row["concept_difficulty_prior"]))
        content_embedding_missing.append(float(row["content_embedding_missing"]))
        analysis_embedding_missing.append(float(row["analysis_embedding_missing"]))

        content_qid = int(row.get("content_embedding_qid", -1))
        analysis_qid = int(row.get("analysis_embedding_qid", -1))
        cvec = content_lookup.get(str(content_qid), zero_vec) if content_qid >= 0 else zero_vec
        avec = analysis_lookup.get(str(analysis_qid), zero_vec) if analysis_qid >= 0 else zero_vec
        content_vectors.append(cvec)
        analysis_vectors.append(avec)

        concepts = [int(v) for v in row.get("concept_indices", [])]
        padded = concepts[:max_concepts] + [0] * max(0, max_concepts - len(concepts))
        mask = [1.0] * min(max_concepts, len(concepts)) + [0.0] * max(0, max_concepts - len(concepts))
        concept_indices.append(padded)
        concept_mask.append(mask)

    return {
        "interaction_ids": torch.tensor(interaction_ids, dtype=torch.long).unsqueeze(0),
        "concept_indices": torch.tensor(concept_indices, dtype=torch.long).unsqueeze(0),
        "concept_mask": torch.tensor(concept_mask, dtype=torch.float32).unsqueeze(0),
        "delta_t_log_bin": torch.tensor(delta_t_log_bin, dtype=torch.long).unsqueeze(0),
        "question_type": torch.tensor(question_type, dtype=torch.long).unsqueeze(0),
        "num_concepts": torch.tensor(num_concepts, dtype=torch.float32).unsqueeze(0),
        "concept_rarity_bucket": torch.tensor(concept_rarity_bucket, dtype=torch.float32).unsqueeze(0),
        "hist_accuracy": torch.tensor(hist_accuracy, dtype=torch.float32).unsqueeze(0),
        "question_difficulty_prior": torch.tensor(question_difficulty_prior, dtype=torch.float32).unsqueeze(0),
        "concept_difficulty_prior": torch.tensor(concept_difficulty_prior, dtype=torch.float32).unsqueeze(0),
        "content_embedding_missing": torch.tensor(content_embedding_missing, dtype=torch.float32).unsqueeze(0),
        "analysis_embedding_missing": torch.tensor(analysis_embedding_missing, dtype=torch.float32).unsqueeze(0),
        "content_vectors": torch.tensor(content_vectors, dtype=torch.float32).reshape(1, seq_len, emb_dim),
        "analysis_vectors": torch.tensor(analysis_vectors, dtype=torch.float32).reshape(1, seq_len, emb_dim),
    }


def _extract_student_memory_state(model: Any, batch: dict[str, Any], torch: Any) -> tuple[Any, Any]:
    model.eval()
    with torch.no_grad():
        x_parts: list[Any] = [model.interaction_embedding(batch["interaction_ids"])]

        if model.use_tags:
            concept_emb = model.concept_embedding(batch["concept_indices"])
            concept_mask_exp = batch["concept_mask"].unsqueeze(-1)
            concept_sum = (concept_emb * concept_mask_exp).sum(dim=2)
            concept_count = concept_mask_exp.sum(dim=2).clamp(min=1.0)
            concept_pooled = concept_sum / concept_count
            x_parts.append(concept_pooled)

        if model.use_numeric_features:
            delta_idx = batch["delta_t_log_bin"].clamp(min=0, max=20)
            delta_emb = model.delta_t_embedding(delta_idx)
            question_type_idx = (batch["question_type"] + 1).clamp(min=0, max=2)
            question_type_emb = model.question_type_embedding(question_type_idx)
            numeric_raw = torch.stack(
                [
                    batch["num_concepts"],
                    batch["concept_rarity_bucket"],
                    batch["hist_accuracy"],
                    batch["question_difficulty_prior"],
                    batch["concept_difficulty_prior"],
                    batch["content_embedding_missing"],
                    batch["analysis_embedding_missing"],
                ],
                dim=-1,
            )
            numeric_proj = model.numeric_projection(numeric_raw)
            x_parts.extend([delta_emb, question_type_emb, numeric_proj])

        if model.use_semantic_vectors:
            semantic_input = torch.cat(
                [
                    batch["content_vectors"],
                    batch["analysis_vectors"],
                    batch["content_embedding_missing"].unsqueeze(-1),
                    batch["analysis_embedding_missing"].unsqueeze(-1),
                ],
                dim=-1,
            )
            semantic_proj = model.semantic_projection(semantic_input)
            x_parts.append(semantic_proj)

        x = torch.cat(x_parts, dim=-1)
        _, (h_n, c_n) = model.lstm(x)
    return h_n.cpu(), c_n.cpu()


def _build_model_from_checkpoint(
    train_module: Any,
    checkpoint: dict[str, Any],
    torch: Any,
) -> Any:
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
    model.to(torch.device("cpu"))
    model.eval()
    return model


def _save_student_weight_artifacts(
    student_index: int,
    uid: int,
    fold_id: int,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    h_n: Any,
    c_n: Any,
    torch: Any,
) -> tuple[Path, Path]:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    memory_path = WEIGHTS_DIR / f"student_{student_index}_memory_state.pt"
    checkpoint_out = WEIGHTS_DIR / f"student_{student_index}_model_with_memory.pt"

    torch.save(
        {
            "student_index": student_index,
            "user_id": uid,
            "fold_id": fold_id,
            "base_checkpoint": str(checkpoint_path),
            "short_term_hidden_h_n": h_n,
            "long_term_cell_c_n": c_n,
        },
        memory_path,
    )

    checkpoint_payload = dict(checkpoint)
    checkpoint_payload["student_index"] = student_index
    checkpoint_payload["student_user_id"] = uid
    checkpoint_payload["student_fold_id"] = fold_id
    checkpoint_payload["base_checkpoint"] = str(checkpoint_path)
    checkpoint_payload["student_short_term_hidden_h_n"] = h_n
    checkpoint_payload["student_long_term_cell_c_n"] = c_n
    torch.save(checkpoint_payload, checkpoint_out)

    return memory_path, checkpoint_out


uids = _read_first_user_ids(OUT_DIR)
rows_by_uid = _write_full_histories(uids)
emb_dim, content_lookup, analysis_lookup = _load_semantic_tables()

try:
    import torch
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "PyTorch is required. Run with model-training/.venv/bin/python students/load_students.py"
    ) from exc

train_module = _load_train_module()
model_cache: dict[int, tuple[Any, dict[str, Any], Path]] = {}
summary_rows: list[dict[str, Any]] = []

for idx, uid in enumerate(uids, start=1):
    fold_id = int(rows_by_uid[uid][0]["fold"])
    feature_rows = _load_student_feature_rows(uid, fold_id)

    if fold_id not in model_cache:
        checkpoint_path = _resolve_checkpoint_path(fold_id)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = _build_model_from_checkpoint(train_module, checkpoint, torch)
        model_cache[fold_id] = (model, checkpoint, checkpoint_path)
    model, checkpoint, checkpoint_path = model_cache[fold_id]
    num_questions = int(checkpoint["num_questions"])

    csv_rows = _write_student_lstm_inputs_csv(
        student_index=idx,
        feature_rows=feature_rows,
        num_questions=num_questions,
        emb_dim=emb_dim,
        content_lookup=content_lookup,
        analysis_lookup=analysis_lookup,
    )

    batch = _rows_to_lstm_tensors(
        feature_rows=feature_rows,
        num_questions=num_questions,
        emb_dim=emb_dim,
        content_lookup=content_lookup,
        analysis_lookup=analysis_lookup,
        torch=torch,
    )
    h_n, c_n = _extract_student_memory_state(model=model, batch=batch, torch=torch)
    memory_path, checkpoint_out = _save_student_weight_artifacts(
        student_index=idx,
        uid=uid,
        fold_id=fold_id,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        h_n=h_n,
        c_n=c_n,
        torch=torch,
    )

    summary_rows.append(
        {
            "student_file": f"student_{idx}.csv",
            "user_id": uid,
            "fold_id": fold_id,
            "feature_rows": csv_rows,
            "memory_state_path": str(memory_path),
            "model_with_memory_path": str(checkpoint_out),
        }
    )

summary_path = WEIGHTS_DIR / "summary.json"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary_rows, handle, ensure_ascii=True, indent=2)

print("Wrote full histories for:", uids)
print("Rebuilt student_1..student_5.csv with concrete LSTM input features.")
print(f"Saved student-specific memory and checkpoint artifacts in: {WEIGHTS_DIR}")