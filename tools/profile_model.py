from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

import torch
from omegaconf import OmegaConf

from ruraltail_mlr.metrics.efficiency import count_parameters, measure_inference_time
from ruraltail_mlr.models.factory import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile model parameter count and inference time.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    model = build_model(cfg.model, cfg.data.class_mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print({"params": count_parameters(model), "inference_time_s": measure_inference_time(model, cfg.train.input_size, device)})


if __name__ == "__main__":
    main()
