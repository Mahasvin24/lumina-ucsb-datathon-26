from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    raw_root: Path
    output_root: Path
    window_size: int = 200
    stride: int = 50
    min_user_history: int = 3
    embedding_projection_dim: int = 32
    smoothing_alpha: float = 5.0
    random_seed: int = 42

    @property
    def interim_dir(self) -> Path:
        return self.output_root / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.output_root / "processed" / "dkt_qlevel_v1"

    @property
    def reports_dir(self) -> Path:
        return self.output_root / "reports"


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def stable_hash(payload: dict[str, Any]) -> str:
    message = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")
