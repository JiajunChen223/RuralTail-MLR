# Contributing

Thank you for helping improve RuralTail-MLR. This repository is primarily a
research-code release, so the most useful contributions are reproducibility
reports, focused bug fixes, and small documentation improvements.

## Good Issue Reports

When reporting a reproduction problem, please include:

- Dataset protocol: China-MAS-50k or Agriculture-Vision-2021.
- Carrier: for example `RN50`, `PVT-B2`, or `MOut-S`.
- Score source: `BCE`, `Focal`, `ASL`, or `TALC`.
- Operating rule: fixed `0.5` (`-F`) or H/M/T group (`-G`).
- Exact command and relevant config path.
- Python, PyTorch, CUDA, and GPU details.
- The shortest relevant traceback or log excerpt.

## Pull Requests

Please keep pull requests narrow and reproducible:

- Preserve the public paper protocol unless the change explicitly documents why
  a protocol update is needed.
- Do not commit raw datasets, generated outputs, checkpoints, or private paths.
- Add or update tests when changing data handling, metrics, thresholds, losses,
  checkpoint loading, or protocol tooling.
- Run `pytest -q` before opening a pull request.
- Run `ruff check .` when touching Python code.

## Data And Checkpoints

Large files should stay outside Git. The repository keeps placeholder
directories under `data/`, `outputs/`, and `artifacts/` so the expected layout is
visible without publishing datasets, model weights, or generated reports.
