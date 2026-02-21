from __future__ import annotations

import argparse
from pathlib import Path

from .build_features import build_features_for_all_folds
from .build_sequences import package_sequences
from .clean_and_validate import clean_and_validate_long_table
from .common import PipelineConfig, ensure_dirs, write_json
from .load_xes3g5m import build_question_level_long_table
from .quality_and_ablation import run_quality_checks_and_ablation
from .validate_outputs import validate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XES3G5M question-level DKT preprocessing pipeline.")
    parser.add_argument("--raw-root", type=Path, default=Path("XES3G5M"), help="Path to XES3G5M root directory.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Output root for interim/processed/reports artifacts.",
    )
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--min-user-history", type=int, default=3)
    parser.add_argument("--embedding-projection-dim", type=int, default=32)
    parser.add_argument("--smoothing-alpha", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-test-files",
        action="store_true",
        help="Also preprocess question_level test files using fold-specific train-fitted states.",
    )
    return parser.parse_args()


def run_pipeline(config: PipelineConfig, include_test_files: bool = False) -> dict[str, object]:
    ensure_dirs(config.interim_dir, config.processed_dir, config.reports_dir)
    print("[1/7] Ingesting train/valid question-level sequences...", flush=True)

    ingest_summary = build_question_level_long_table(
        raw_root=config.raw_root,
        output_root=config.interim_dir,
        source_filename="train_valid_sequences_quelevel.csv",
        output_basename="question_level_long",
    )
    print(
        f"      Parsed interactions: {ingest_summary['parsed_interactions']}, "
        f"rejected interactions: {ingest_summary['rejected_interactions']}",
        flush=True,
    )
    print("[2/7] Cleaning and validating train/valid interactions...", flush=True)
    quality_report = clean_and_validate_long_table(
        interim_dir=config.interim_dir,
        reports_dir=config.reports_dir,
        min_user_history=config.min_user_history,
        input_basename="question_level_long",
        cleaned_basename="question_level_long_clean",
        short_basename="question_level_long_short_histories",
        write_quality_report=True,
        filter_short_histories=True,
    )
    print(
        f"      Kept rows: {quality_report['kept_rows']}, "
        f"duplicates dropped: {quality_report['duplicate_rows_dropped']}",
        flush=True,
    )

    extra_split_files: dict[str, Path] = {}
    if include_test_files:
        print("[3/7] Ingesting/cleaning question-level test files...", flush=True)
        build_question_level_long_table(
            raw_root=config.raw_root,
            output_root=config.interim_dir,
            source_filename="test_window_sequences_quelevel.csv",
            output_basename="test_window_long",
        )
        clean_and_validate_long_table(
            interim_dir=config.interim_dir,
            reports_dir=config.reports_dir,
            min_user_history=config.min_user_history,
            input_basename="test_window_long",
            cleaned_basename="test_window_long_clean",
            short_basename="test_window_long_short_histories",
            write_quality_report=False,
            filter_short_histories=False,
        )
        build_question_level_long_table(
            raw_root=config.raw_root,
            output_root=config.interim_dir,
            source_filename="test_quelevel.csv",
            output_basename="test_quelevel_long",
        )
        clean_and_validate_long_table(
            interim_dir=config.interim_dir,
            reports_dir=config.reports_dir,
            min_user_history=config.min_user_history,
            input_basename="test_quelevel_long",
            cleaned_basename="test_quelevel_long_clean",
            short_basename="test_quelevel_long_short_histories",
            write_quality_report=False,
            filter_short_histories=False,
        )
        extra_split_files = {
            "test_window": config.interim_dir / "test_window_long_clean.jsonl",
            "test_quelevel": config.interim_dir / "test_quelevel_long_clean.jsonl",
        }
        print("      Added extra splits: test_window, test_quelevel", flush=True)
    else:
        print("[3/7] Skipping optional test-file preprocessing.", flush=True)

    print("[4/7] Building fold-wise features and priors...", flush=True)
    feature_summary = build_features_for_all_folds(
        interim_dir=config.interim_dir,
        processed_dir=config.processed_dir,
        metadata_root=config.raw_root / "metadata",
        emb_dim=config.embedding_projection_dim,
        smoothing_alpha=config.smoothing_alpha,
        cleaned_filename="question_level_long_clean.jsonl",
        extra_split_files=extra_split_files,
    )
    print(
        f"      Completed features for folds: {', '.join(sorted(feature_summary['folds'].keys()))}",
        flush=True,
    )
    print("[5/7] Packaging sequence windows...", flush=True)
    manifest = package_sequences(
        processed_dir=config.processed_dir,
        window_size=config.window_size,
        stride=config.stride,
    )
    print(
        f"      Manifest config hash: {manifest['config_hash']}, "
        f"window_size={manifest['schema']['window_size']}, stride={manifest['schema']['stride']}",
        flush=True,
    )
    print("[6/7] Running quality checks and ablation summary...", flush=True)
    ablation_report = run_quality_checks_and_ablation(
        processed_dir=config.processed_dir,
        reports_dir=config.reports_dir,
        random_seed=config.random_seed,
    )
    print(
        f"      Mean AUC={ablation_report['mean_full_feature_auc']:.4f}, "
        f"Mean ACC={ablation_report['mean_full_feature_acc']:.4f}",
        flush=True,
    )

    pipeline_summary = {
        "ingest_summary": ingest_summary,
        "quality_report_path": str(config.reports_dir / "data_quality_qlevel_v1.json"),
        "feature_summary": feature_summary,
        "manifest_path": str(config.processed_dir / "manifest.json"),
        "ablation_report_path": str(config.reports_dir / "ablation_qlevel_v1.json"),
        "manifest_config_hash": manifest["config_hash"],
        "mean_full_feature_auc": ablation_report["mean_full_feature_auc"],
        "mean_full_feature_acc": ablation_report["mean_full_feature_acc"],
        "include_test_files": include_test_files,
    }
    write_json(config.reports_dir / "pipeline_run_summary_qlevel_v1.json", pipeline_summary)
    print("[7/7] Validating generated outputs...", flush=True)
    validation_summary = validate_outputs(
        processed_dir=config.processed_dir,
        reports_dir=config.reports_dir,
        include_test_files=include_test_files,
    )
    pipeline_summary["validation_summary_path"] = str(
        config.reports_dir / "output_validation_summary_qlevel_v1.json"
    )
    pipeline_summary["validation_status"] = validation_summary["status"]
    write_json(config.reports_dir / "pipeline_run_summary_qlevel_v1.json", pipeline_summary)
    return pipeline_summary


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        raw_root=args.raw_root,
        output_root=args.output_root,
        window_size=args.window_size,
        stride=args.stride,
        min_user_history=args.min_user_history,
        embedding_projection_dim=args.embedding_projection_dim,
        smoothing_alpha=args.smoothing_alpha,
        random_seed=args.seed,
    )
    summary = run_pipeline(config, include_test_files=args.include_test_files)
    print("Pipeline complete.")
    print(f"Manifest: {summary['manifest_path']}")
    print(f"Ablation report: {summary['ablation_report_path']}")
    print(f"Output validation: {summary['validation_status']}")


if __name__ == "__main__":
    main()
