from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_run_dir(output_dir: str | Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in run_name)
    run_dir = Path(output_dir) / f"{stamp}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir
