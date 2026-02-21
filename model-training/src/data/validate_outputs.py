from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import read_json, write_json


def _check_file(path: Path, errors: list[str]) -> bool:
    if not path.exists():
        errors.append(f"Missing file: {path}")
        return False
    if path.stat().st_size == 0:
        errors.append(f"Empty file: {path}")
        return False
    return True


def validate_outputs(
    processed_dir: Path,
    reports_dir: Path,
    include_test_files: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = processed_dir / "manifest.json"
    feature_summary_path = processed_dir / "feature_summary.json"
    quality_path = reports_dir / "data_quality_qlevel_v1.json"
    ablation_path = reports_dir / "ablation_qlevel_v1.json"
    pipeline_summary_path = reports_dir / "pipeline_run_summary_qlevel_v1.json"
    validation_path = reports_dir / "output_validation_summary_qlevel_v1.json"

    required_paths = [
        manifest_path,
        feature_summary_path,
        quality_path,
        ablation_path,
        pipeline_summary_path,
    ]
    for path in required_paths:
        _check_file(path, errors)

    fold_output_summary: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        fold_summary = manifest.get("fold_summary", {})
        for fold_id in sorted(fold_summary.keys(), key=lambda x: int(x)):
            fold_dir = processed_dir / f"fold_{fold_id}"
            split_files = {
                "train": fold_dir / "train_windows.jsonl",
                "valid": fold_dir / "valid_windows.jsonl",
            }
            if include_test_files:
                split_files["test_window"] = fold_dir / "test_window_windows.jsonl"
                split_files["test_quelevel"] = fold_dir / "test_quelevel_windows.jsonl"

            split_status: dict[str, str] = {}
            for split_name, path in split_files.items():
                ok = _check_file(path, errors)
                split_status[split_name] = "ok" if ok else "missing_or_empty"

            fold_output_summary[fold_id] = split_status

        features = manifest.get("schema", {}).get("features", [])
        if "concept_indices" not in features:
            errors.append("Manifest schema is missing required feature: concept_indices")
        if include_test_files:
            any_test_counts = any("test_window_windows" in stats for stats in fold_summary.values())
            if not any_test_counts:
                errors.append("Expected test window counts in manifest, but none found.")

    if quality_path.exists():
        quality = read_json(quality_path)
        if quality.get("kept_rows", 0) <= 0:
            errors.append("Quality report indicates no kept rows.")
        if quality.get("duplicate_rows_dropped", 0) > 0:
            warnings.append(
                f"Duplicate rows dropped: {quality['duplicate_rows_dropped']}. "
                "Verify these were true duplicates."
            )

    if ablation_path.exists():
        ablation = read_json(ablation_path)
        auc = float(ablation.get("mean_full_feature_auc", 0.0))
        acc = float(ablation.get("mean_full_feature_acc", 0.0))
        if auc <= 0.0 or acc <= 0.0:
            errors.append("Ablation summary has non-positive AUC/ACC.")
        leakage_probe = float(ablation.get("mean_response_feature_leakage_probe_acc", 0.0))
        if leakage_probe > 0.95:
            warnings.append(
                "Leakage probe is very high (>0.95). Re-verify next-step target alignment."
            )

    summary = {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "include_test_files": include_test_files,
        "fold_output_summary": fold_output_summary,
        "required_paths_checked": [str(path) for path in required_paths],
    }
    write_json(validation_path, summary)
    return summary
