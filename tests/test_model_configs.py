from pathlib import Path
import importlib.util
import sys

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from ruraltail_mlr.data.label_schema import default_class_mapping, write_class_mapping
from ruraltail_mlr.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_formal_model_configs_are_exactly_the_selected_six():
    model_dir = ROOT / "configs" / "model"
    assert {p.name for p in model_dir.glob("*.yaml")} == {
        "resnet50_linear.yaml",
        "efficientnetv2_s_linear.yaml",
        "pvtv2_b2_linear.yaml",
        "mambaout_s_linear.yaml",
        "sfin_resnet18.yaml",
        "mlmamba_resnet18.yaml",
    }
    expected = {
        "resnet50_linear.yaml": ("resnet50", "linear"),
        "efficientnetv2_s_linear.yaml": ("tf_efficientnetv2_s", "linear"),
        "pvtv2_b2_linear.yaml": ("pvt_v2_b2", "linear"),
        "mambaout_s_linear.yaml": ("mambaout_small", "linear"),
        "sfin_resnet18.yaml": ("resnet18", "sfin"),
        "mlmamba_resnet18.yaml": ("resnet18", "mlmamba"),
    }
    for filename, (backbone, head) in expected.items():
        cfg = _read_yaml(model_dir / filename)
        assert cfg["backbone"] == backbone
        assert cfg["head"] == head
        assert cfg["pretrained"] is True


def test_recent_method_default_capacity_is_pinned():
    sfin_cfg = _read_yaml(ROOT / "configs" / "model" / "sfin_resnet18.yaml")
    assert sfin_cfg["heads"] == 4

    mlmamba_cfg = _read_yaml(ROOT / "configs" / "model" / "mlmamba_resnet18.yaml")
    assert mlmamba_cfg["hidden_dim"] == 128


def test_mambaout_config_uses_forward_feature_dim_override():
    cfg = _read_yaml(ROOT / "configs" / "model" / "mambaout_s_linear.yaml")
    assert cfg["feature_dim"] == 2304
    assert cfg["embedding_dim"] == 2304


def test_formal_experiment_configs_are_exactly_the_six_model_bases():
    experiment_dir = ROOT / "configs" / "experiment"
    assert {p.name for p in experiment_dir.glob("*.yaml")} == {
        "main_resnet50.yaml",
        "main_efficientnetv2_s.yaml",
        "main_pvtv2_b2.yaml",
        "main_mambaout_s.yaml",
        "main_sfin_resnet18.yaml",
        "main_mlmamba_resnet18.yaml",
    }
    expected = {
        "main_resnet50.yaml": "resnet50_linear",
        "main_efficientnetv2_s.yaml": "efficientnetv2_s_linear",
        "main_pvtv2_b2.yaml": "pvtv2_b2_linear",
        "main_mambaout_s.yaml": "mambaout_s_linear",
        "main_sfin_resnet18.yaml": "sfin",
        "main_mlmamba_resnet18.yaml": "mlmamba",
    }
    for filename, model_name in expected.items():
        cfg = _read_yaml(experiment_dir / filename)
        assert cfg["model"]["name"] == model_name
        assert ("gr" + "aph") not in cfg
        assert cfg.get("loss", {}).get("cls", {}).get("name", "asl") == "asl"


@pytest.mark.parametrize(
    "filename,input_size",
    [
        ("main_resnet50.yaml", 384),
        ("main_efficientnetv2_s.yaml", 384),
        ("main_pvtv2_b2.yaml", 384),
        ("main_mambaout_s.yaml", 384),
        ("main_sfin_resnet18.yaml", 256),
        ("main_mlmamba_resnet18.yaml", 256),
    ],
)
def test_formal_experiments_share_30_epoch_protocol(filename: str, input_size: int):
    cfg = _read_yaml(ROOT / "configs" / "experiment" / filename)
    assert cfg["train"]["input_size"] == input_size
    assert cfg["train"]["batch_size"] == 32
    assert cfg["train"]["gradient_accumulation_steps"] == 2
    assert cfg["train"]["epochs"] == 30
    assert cfg["train"]["optimizer"] == "adamw"
    assert float(cfg["train"]["lr"]) == pytest.approx(5e-5)
    assert float(cfg["train"]["weight_decay"]) == pytest.approx(0.05)
    assert cfg["train"]["amp"] is False
    assert cfg["train"]["scheduler"]["name"] == "warmup_cosine"
    assert cfg["train"]["scheduler"]["warmup_epochs"] == 2
    assert float(cfg["train"]["scheduler"]["warmup_start_factor"]) == pytest.approx(0.1)
    assert float(cfg["train"]["scheduler"]["eta_min"]) == pytest.approx(1e-6)
    assert cfg["train"]["early_stopping"]["enabled"] is True
    assert cfg["train"]["early_stopping"]["monitor"] == "mAP"
    assert cfg["train"]["early_stopping"]["mode"] == "max"
    assert cfg["train"]["early_stopping"]["patience"] == 6
    assert float(cfg["train"]["early_stopping"]["min_delta"]) == pytest.approx(0.0002)
    assert cfg["train"]["early_stopping"]["min_epochs"] == 12


def test_default_config_matches_paper_training_protocol():
    cfg = _read_yaml(ROOT / "configs" / "default.yaml")
    train = cfg["train"]
    assert train["input_size"] == 384
    assert train["batch_size"] == 32
    assert train["gradient_accumulation_steps"] == 2
    assert train["epochs"] == 30
    assert train["optimizer"] == "adamw"
    assert float(train["lr"]) == pytest.approx(5e-5)
    assert float(train["weight_decay"]) == pytest.approx(0.05)
    assert train["scheduler"]["name"] == "warmup_cosine"
    assert train["scheduler"]["warmup_epochs"] == 2
    assert cfg["run"]["monitor"] == "mAP"
    assert train["early_stopping"]["enabled"] is True
    assert train["early_stopping"]["monitor"] == "mAP"
    assert train["early_stopping"]["mode"] == "max"
    assert train["early_stopping"]["patience"] == 6
    assert float(train["early_stopping"]["min_delta"]) == pytest.approx(0.0002)
    assert train["early_stopping"]["min_epochs"] == 12


def test_paper_protocol_uses_four_score_sources_and_formal_grid():
    protocol = _read_yaml(ROOT / "configs" / "protocol" / "paper_talc.yaml")
    assert set(protocol["score_sources"]) == {"bce", "focal", "asl", "talc"}
    assert protocol["public_method_policy"]["main_score_sources"] == ["bce", "focal", "asl", "talc"]
    assert protocol["public_method_policy"]["formal_operating_rules"] == ["fixed", "group"]
    assert protocol["public_method_policy"]["public_display_rows"] == [
        "BCE-F",
        "BCE-G",
        "Focal-F",
        "Focal-G",
        "ASL-F",
        "ASL-G",
        "TALC-F",
        "TALC-G",
    ]
    assert protocol["public_method_policy"]["diagnostic_only_methods"] == []
    assert protocol["datasets"]["china_mas_50k"]["split_seed"] == 20260425
    assert protocol["datasets"]["agriculture_vision_2021"]["split_seed"] == 20260501
    early = protocol["training"]["early_stopping"]
    assert early == {
        "enabled": True,
        "monitor": "mAP",
        "mode": "max",
        "patience": 6,
        "min_delta": 0.0002,
        "min_epochs": 12,
    }
    grid = protocol["operating_rules"]["group"]["grid"]
    assert grid == [round(0.10 + 0.05 * idx, 2) for idx in range(13)]


def test_public_tree_has_no_legacy_alias_surface():
    forbidden = [
        "logit" + "_cal" + "ibration",
        "t_aware" + "_asl",
        "decision_aware" + "_asl",
        "asl_t" + "_soft_f1",
        "legacy diagnostic " + "launchers",
        "Calibrated score " + "learners",
    ]
    roots = [ROOT / "configs", ROOT / "src", ROOT / "tools", ROOT / "README.md", ROOT / "REPRODUCE.md"]
    paths = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        else:
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md", ".sh"}
            )
    haystack = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for token in forbidden:
        assert token not in haystack


def test_talc_method_config_keeps_training_evaluation_fixed():
    cfg = _read_yaml(ROOT / "configs" / "method" / "asl_softf1_t.yaml")
    assert cfg["eval"]["threshold"] == 0.5
    assert cfg["eval"]["tune_threshold_on_val"] is False
    assert cfg["eval"]["threshold_strategy"] == "global"
    loss = cfg["loss"]["cls"]
    assert loss["name"] == "asl_soft_f1"
    assert float(loss["lambda_soft_f1"]) == pytest.approx(0.2)
    assert float(loss["lambda_tail_recall"]) == pytest.approx(0.05)
    assert float(loss["soft_temperature"]) == pytest.approx(0.5)
    assert float(loss["tail_recall_target"]) == pytest.approx(0.65)
    assert loss["threshold_by_group"] == {"head": 0.5, "medium": 0.35, "tail": 0.25, "default": 0.5}


def test_run_matrix_points_to_paper_protocol():
    matrix = _read_yaml(ROOT / "configs" / "run_matrix" / "formal_runs.yaml")
    paper = matrix["paper_protocol"]
    assert paper["protocol"] == "configs/protocol/paper_talc.yaml"
    assert paper["datasets"]["china_mas_50k"]["score_sources"] == ["bce", "focal", "asl", "talc"]
    assert paper["datasets"]["agriculture_vision_2021"]["score_sources"] == ["bce", "focal", "asl", "talc"]
    assert paper["operating_rules"]["diagnostic_only"] == ["global", "classwise"]
    assert paper["public_display"]["asl"] == ["ASL-F", "ASL-G"]


def test_sfin_forward_shape(tmp_path: Path):
    mapping = tmp_path / "class_mapping.json"
    write_class_mapping(default_class_mapping(), mapping)
    cfg = OmegaConf.create({"name": "sfin", "backbone": "resnet18", "pretrained": False, "heads": 4})
    model = build_model(cfg, str(mapping))
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 256, 256))

    assert logits.shape == (1, 18)


def test_mlmamba_dependency_error_is_explicit(tmp_path: Path):
    if importlib.util.find_spec("mamba_ssm"):
        pytest.skip("mamba_ssm is installed; runtime forward is covered on the training machine.")
    mapping = tmp_path / "class_mapping.json"
    write_class_mapping(default_class_mapping(), mapping)
    cfg = OmegaConf.create({"name": "mlmamba", "backbone": "resnet18", "pretrained": False, "hidden_dim": 128})
    with pytest.raises(ImportError, match="mamba-ssm"):
        build_model(cfg, str(mapping))


def test_warmup_cosine_scheduler_builder():
    tools_dir = ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    spec = importlib.util.spec_from_file_location("train_tool", tools_dir / "train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    cfg = OmegaConf.create(
        {
            "train": {
                "optimizer": "adamw",
                "lr": 1e-3,
                "weight_decay": 0.01,
                "epochs": 20,
                "scheduler": {
                    "name": "warmup_cosine",
                    "warmup_epochs": 5,
                    "warmup_start_factor": 0.1,
                    "eta_min": 1e-6,
                },
            }
        }
    )
    model = torch.nn.Linear(2, 1)
    optimizer = module.build_optimizer(cfg, model)
    scheduler = module.build_scheduler(cfg, optimizer)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)


@pytest.mark.parametrize(
    "cfg_name",
    [
        "efficientnetv2_s_linear",
        "pvtv2_b2_linear",
        "mambaout_s_linear",
    ],
)
def test_real_timm_model_forward_shapes(tmp_path: Path, cfg_name: str):
    pytest.importorskip("timm")
    mapping = tmp_path / "class_mapping.json"
    write_class_mapping(default_class_mapping(), mapping)

    cfg = OmegaConf.load(ROOT / "configs" / "model" / f"{cfg_name}.yaml")
    cfg.pretrained = False
    model = build_model(cfg, str(mapping))
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 384, 384))

    assert logits.shape == (1, 18)


def test_required_recent_method_dependencies_are_declared():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    optional = (ROOT / "requirements-mamba.txt").read_text(encoding="utf-8")
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    for token in ["mamba-ssm", "causal-conv1d"]:
        assert token not in requirements
        assert token in optional
        assert token in environment
