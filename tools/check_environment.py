from __future__ import annotations

import argparse
import importlib.util

from _bootstrap import bootstrap

bootstrap()

from ruraltail_mlr.utils.run_meta import collect_environment_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether the environment is ready for RuralTail-MLR experiments.")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--require_timm", action="store_true")
    parser.add_argument("--require_mamba_ssm", action="store_true")
    parser.add_argument("--require_causal_conv1d", action="store_true")
    args = parser.parse_args()

    info = collect_environment_info(".")
    problems = []
    if args.require_cuda and not info["cuda_available"]:
        problems.append("CUDA is not available. Install a CUDA-enabled PyTorch build for 384 main experiments.")
    if args.require_timm and info["packages"].get("timm") is None:
        problems.append("timm is not installed. Install requirements.txt before timm backbone experiments.")
    if args.require_mamba_ssm and importlib.util.find_spec("mamba_ssm") is None:
        problems.append("mamba-ssm is not installed. MLMamba in the formal benchmark will fail.")
    if args.require_causal_conv1d and importlib.util.find_spec("causal_conv1d") is None:
        problems.append("causal-conv1d is not installed. MLMamba in the formal benchmark will fail.")
    print(info)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
