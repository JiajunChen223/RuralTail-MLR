from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ruraltail_mlr.engine.amp import autocast_context, make_grad_scaler
from ruraltail_mlr.engine.checkpoint import load_checkpoint, save_checkpoint
from ruraltail_mlr.engine.distributed import DistributedState, barrier, broadcast_object, reduce_mean
from ruraltail_mlr.engine.evaluator import Evaluator
from ruraltail_mlr.metrics.efficiency import count_parameters
from ruraltail_mlr.metrics.threshold import clone_threshold_spec, plain_threshold_grid, summarize_threshold_spec, tune_thresholds_on_val
from ruraltail_mlr.utils.io import save_json
from ruraltail_mlr.utils.logging import JsonlLogger


def _get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


class Trainer:
    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        run_dir: str | Path,
        class_names: list[str],
        class_groups: dict,
        class_mapping_hash: str | None = None,
        device: torch.device | None = None,
        distributed_state: DistributedState | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_dir = Path(run_dir)
        self.class_names = class_names
        self.class_groups = class_groups
        self.class_mapping_hash = class_mapping_hash
        self.distributed_state = distributed_state or DistributedState(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.is_main_process = self.distributed_state.is_main_process
        self.device = device or torch.device(
            cfg.train.device if cfg.train.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if hasattr(self.model, "module"):
            self.raw_model = self.model.module
        else:
            self.model.to(self.device)
            self.raw_model = self.model
        self.criterion = criterion.to(self.device)
        self.scaler = make_grad_scaler(enabled=bool(cfg.train.amp) and self.device.type == "cuda")
        self.gradient_accumulation_steps = int(_get(cfg.train, "gradient_accumulation_steps", 1))
        if self.gradient_accumulation_steps < 1:
            raise ValueError("train.gradient_accumulation_steps must be >= 1")
        self.max_grad_norm = float(_get(cfg.train, "max_grad_norm", 0.0))
        self.logger = JsonlLogger(self.run_dir / "train_log.jsonl") if self.is_main_process else None
        self.monitor_name = str(_get(cfg.run, "monitor", "mAP"))
        self.monitor_mode = str(_get(cfg.run, "monitor_mode", "max")).lower()
        if self.monitor_mode not in {"max", "min"}:
            raise ValueError("run.monitor_mode must be 'max' or 'min'")
        self.best_metric = -float("inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch = 0
        early_cfg = _get(cfg.train, "early_stopping", None)
        self.early_stopping_enabled = bool(_get(early_cfg, "enabled", False))
        self.early_stopping_monitor = str(_get(early_cfg, "monitor", self.monitor_name))
        self.early_stopping_mode = str(_get(early_cfg, "mode", self.monitor_mode)).lower()
        if self.early_stopping_mode not in {"max", "min"}:
            raise ValueError("train.early_stopping.mode must be 'max' or 'min'")
        self.early_stopping_patience = int(_get(early_cfg, "patience", 25))
        self.early_stopping_min_delta = float(_get(early_cfg, "min_delta", 0.0))
        self.early_stopping_min_epochs = int(_get(early_cfg, "min_epochs", 1))
        self.early_stopping_best = (
            -float("inf") if self.early_stopping_mode == "max" else float("inf")
        )
        self.early_stopping_wait = 0
        self.eval_threshold = clone_threshold_spec(_get(cfg.eval, "threshold", 0.5))
        self.tune_threshold_on_val = bool(_get(cfg.eval, "tune_threshold_on_val", False))
        self.threshold_strategy = str(_get(cfg.eval, "threshold_strategy", "global")).lower()
        self.threshold_grid = plain_threshold_grid(_get(cfg.eval, "threshold_grid", None))
        self.best_eval_threshold = clone_threshold_spec(self.eval_threshold)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = int(self.cfg.train.epochs)
        for epoch in range(1, epochs + 1):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            train_loss = self.train_one_epoch(train_loader, epoch)
            if self.is_main_process:
                val_metrics = self.evaluate_loader(val_loader, split="val")
                metric = self._metric_from(val_metrics, self.monitor_name)
                is_best = self._is_improved(metric, self.best_metric, self.monitor_mode, min_delta=0.0)
                self.eval_threshold = clone_threshold_spec(val_metrics.get("threshold_spec", self.eval_threshold))
                if is_best:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    self.best_eval_threshold = clone_threshold_spec(self.eval_threshold)
                self.save(epoch=epoch, is_best=is_best)
                early_state = self._update_early_stopping(epoch, val_metrics)
                assert self.logger is not None
                self.logger.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val": val_metrics,
                        "monitor": {
                            "name": self.monitor_name,
                            "value": metric,
                            "best": self.best_metric,
                            "best_epoch": self.best_epoch,
                        },
                        "eval": {
                            "threshold": self.eval_threshold,
                            "tune_threshold_on_val": self.tune_threshold_on_val,
                            "threshold_strategy": self.threshold_strategy,
                        },
                        "optimizer": {
                            "lr": [float(group["lr"]) for group in self.optimizer.param_groups],
                            "scheduler": type(self.scheduler).__name__ if self.scheduler is not None else None,
                        },
                        "early_stopping": early_state,
                        "distributed": {
                            "enabled": self.distributed_state.enabled,
                            "world_size": self.distributed_state.world_size,
                            "global_batch_size": int(_get(self.cfg.train, "batch_size", 0)),
                            "gradient_accumulation_steps": self.gradient_accumulation_steps,
                            "effective_batch_size": int(_get(self.cfg.train, "batch_size", 0))
                            * self.gradient_accumulation_steps,
                            "max_grad_norm": self.max_grad_norm,
                        },
                    }
                )
                if early_state["should_stop"]:
                    save_json(early_state, self.run_dir / "early_stopping.json")
            else:
                early_state = None
            early_state = broadcast_object(early_state, self.distributed_state)
            if self.scheduler is not None:
                self.scheduler.step()
            if early_state["should_stop"]:
                break

    def _metric_from(self, metrics: dict, name: str) -> float:
        return float(metrics.get(name, metrics.get("mAP", 0.0)))

    @staticmethod
    def _is_improved(metric: float, reference: float, mode: str, min_delta: float = 0.0) -> bool:
        if mode == "max":
            return metric > reference + min_delta
        return metric < reference - min_delta

    def _update_early_stopping(self, epoch: int, val_metrics: dict) -> dict:
        metric = self._metric_from(val_metrics, self.early_stopping_monitor)
        significant = self._is_improved(
            metric,
            self.early_stopping_best,
            self.early_stopping_mode,
            min_delta=self.early_stopping_min_delta,
        )
        if significant:
            self.early_stopping_best = metric
            self.early_stopping_wait = 0
        else:
            self.early_stopping_wait += 1

        should_stop = (
            self.early_stopping_enabled
            and epoch >= self.early_stopping_min_epochs
            and self.early_stopping_wait >= self.early_stopping_patience
        )
        return {
            "enabled": self.early_stopping_enabled,
            "monitor": self.early_stopping_monitor,
            "mode": self.early_stopping_mode,
            "metric": metric,
            "best_significant_metric": self.early_stopping_best,
            "wait": self.early_stopping_wait,
            "patience": self.early_stopping_patience,
            "min_delta": self.early_stopping_min_delta,
            "min_epochs": self.early_stopping_min_epochs,
            "should_stop": should_stop,
        }

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        losses = []
        progress = tqdm(loader, desc=f"epoch {epoch}", leave=False, disable=not self.is_main_process)
        self.optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(progress, start=1):
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            with autocast_context(bool(self.cfg.train.amp), self.device):
                logits = self.model(images)
                loss_pack = self.criterion(logits, targets)
                raw_loss = loss_pack["loss"]
                if not torch.isfinite(raw_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch}, step={step_idx}: {float(raw_loss.detach().cpu())}"
                    )
                loss = raw_loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            should_step = step_idx % self.gradient_accumulation_steps == 0 or step_idx == len(loader)
            if should_step:
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            losses.append(float(raw_loss.detach().cpu()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
        local_loss = float(sum(losses) / max(len(losses), 1))
        return reduce_mean(local_loss, self.distributed_state)

    def evaluate_loader(self, loader: DataLoader, split: str) -> dict:
        evaluator = Evaluator(self.class_names, self.class_groups, self.device)
        logits, targets, ids = evaluator.predict(self.raw_model, loader)
        threshold = clone_threshold_spec(_get(self.cfg.eval, "threshold", 0.5))
        threshold_tuning = None
        if split == "val" and self.tune_threshold_on_val:
            probs = torch.sigmoid(logits).numpy()
            y_true = targets.numpy().astype(int)
            threshold, best_score = tune_thresholds_on_val(
                y_true,
                probs,
                strategy=self.threshold_strategy,
                class_names=self.class_names,
                class_groups=self.class_groups,
                grid=self.threshold_grid,
                fallback_threshold=float(_get(self.cfg.eval, "threshold", 0.5)),
            )
            threshold_meta = summarize_threshold_spec(threshold, self.class_names, self.class_groups)
            threshold_tuning = {
                "split": split,
                **threshold_meta,
                "macro_F1": float(best_score),
                "grid": self.threshold_grid,
            }
            save_json(threshold_tuning, self.run_dir / "threshold_val.json")
        metrics, per_class, hmt, preds = evaluator.evaluate(logits, targets, ids, threshold=threshold)
        save_json(metrics, self.run_dir / f"metrics_{split}.json")
        if split == "test":
            pd.DataFrame(per_class).to_csv(self.run_dir / "per_class_ap_test.csv", index=False)
            save_json(hmt, self.run_dir / "hmt_metrics_test.json")
            preds.insert(1, "split", split)
            preds.to_csv(self.run_dir / "predictions_test.csv", index=False)
        elif threshold_tuning is not None:
            pd.DataFrame(per_class).to_csv(self.run_dir / "per_class_ap_val.csv", index=False)
        return metrics

    def _tune_threshold_from_loader(self, loader: DataLoader):
        evaluator = Evaluator(self.class_names, self.class_groups, self.device)
        logits, targets, _ = evaluator.predict(self.raw_model, loader)
        probs = torch.sigmoid(logits).numpy()
        y_true = targets.numpy().astype(int)
        threshold, best_score = tune_thresholds_on_val(
            y_true,
            probs,
            strategy=self.threshold_strategy,
            class_names=self.class_names,
            class_groups=self.class_groups,
            grid=self.threshold_grid,
            fallback_threshold=float(_get(self.cfg.eval, "threshold", 0.5)),
        )
        threshold_meta = summarize_threshold_spec(threshold, self.class_names, self.class_groups)
        save_json(
            {
                "split": "val",
                **threshold_meta,
                "macro_F1": float(best_score),
                "grid": self.threshold_grid,
            },
            self.run_dir / "threshold_val.json",
        )
        return threshold

    def test(self, test_loader: DataLoader, checkpoint: str = "best", val_loader: DataLoader | None = None) -> dict:
        if not self.is_main_process:
            barrier(self.distributed_state)
            return {}
        path = self.run_dir / ("checkpoint_best.pth" if checkpoint == "best" else checkpoint)
        eval_threshold = clone_threshold_spec(_get(self.cfg.eval, "threshold", 0.5))
        if path.exists():
            ckpt = load_checkpoint(path, map_location=self.device)
            self.raw_model.load_state_dict(ckpt["model_state"])
            eval_threshold = clone_threshold_spec(ckpt.get("eval_threshold", eval_threshold))
        if self.tune_threshold_on_val and "ckpt" in locals() and "eval_threshold" not in ckpt and val_loader is not None:
            eval_threshold = self._tune_threshold_from_loader(val_loader)
        evaluator = Evaluator(self.class_names, self.class_groups, self.device)
        logits, targets, ids = evaluator.predict(self.raw_model, test_loader)
        metrics, per_class, hmt, preds = evaluator.evaluate(logits, targets, ids, threshold=eval_threshold)
        save_json(metrics, self.run_dir / "metrics_test.json")
        pd.DataFrame(per_class).to_csv(self.run_dir / "per_class_ap_test.csv", index=False)
        save_json(hmt, self.run_dir / "hmt_metrics_test.json")
        preds.insert(1, "split", "test")
        preds.to_csv(self.run_dir / "predictions_test.csv", index=False)
        save_json({"params": count_parameters(self.model), "flops": None}, self.run_dir / "efficiency.json")
        barrier(self.distributed_state)
        return metrics

    def save(self, epoch: int, is_best: bool) -> None:
        if not self.is_main_process:
            return
        payload = {
            "model_state": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "best_val_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "eval_threshold": self.eval_threshold,
            "best_eval_threshold": self.best_eval_threshold,
            "class_mapping_hash": self.class_mapping_hash,
            "config": str(self.cfg),
        }
        save_checkpoint(self.run_dir / "checkpoint_last.pth", payload)
        if is_best:
            save_checkpoint(self.run_dir / "checkpoint_best.pth", payload)
