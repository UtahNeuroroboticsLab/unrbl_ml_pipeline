"""Select model defaults before merging configuration, without loading a backend."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unrbl_ml_pipeline.cli import parse_training_args, show_catalog, training_overrides
from unrbl_ml_pipeline.training.config import resolve_config, resolve_model_name

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "settings.jsonc"
    path.write_text(json.dumps({"model_name": "mamba_decoder",
                               "model_overrides": {"mamba_decoder": {"epochs": 7},
                                                   "kf_decoder": {"epochs": 9}}}))
    return path


def test_cli_file_python_and_explicit_precedence(config_file):
    assert parse_training_args([]).model == "clstm_decoder"
    assert parse_training_args(["--config", str(config_file)]).model == "mamba_decoder"
    assert parse_training_args([], default_config=config_file,
                               default_model="caleb_decoder").model == "mamba_decoder"
    assert parse_training_args(["--config", str(config_file)],
                               python_overrides={"model_name": "kf_decoder"}).model == "kf_decoder"
    args = parse_training_args(["--config", str(config_file), "--model", "CALEB_DECODER"],
                               python_overrides={"model_name": "kf_decoder"})
    assert args.model == "caleb_decoder"
    resolved = resolve_config(args.model, user_config=training_overrides(args), config_path=config_file)
    assert resolved["model_name"] == "caleb_decoder"
    assert resolved["optimizer"] == "sgd"
    selected = resolve_model_name(config_path=config_file)
    assert resolve_config(selected, config_path=config_file)["epochs"] == 7


def test_catalogs_do_not_require_a_config(config_file, capsys):
    for flag in ("--list-models", "--list-steps"):
        args = parse_training_args([flag], default_config="/does/not/exist.jsonc")
        assert show_catalog(args)
    args = parse_training_args(["--list-losses", "--config", str(config_file)])
    assert args.model == "mamba_decoder"
    assert show_catalog(args)
    assert "mse_pinn_jacobian" in capsys.readouterr().out


@pytest.mark.parametrize("name,error", [("unknown_decoder", KeyError), ("", ValueError), (42, ValueError)])
def test_invalid_model_selection_fails_early(tmp_path, name, error):
    path = tmp_path / "bad.jsonc"
    path.write_text(json.dumps({"model_name": name}))
    with pytest.raises(error):
        parse_training_args(["--config", str(path)])


@pytest.mark.parametrize("filename,wrapped", [("testfile.py", True), ("testfile_train_api.py", False)])
@pytest.mark.parametrize("explicit", [None, "kf_decoder"])
def test_examples_select_from_file(monkeypatch, tmp_path, config_file, filename, wrapped, explicit):
    from unrbl_ml_pipeline import training

    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return object(), {"history": SimpleNamespace(history={"loss": [1]}),
                          "test_loss": 1, "model_path": tmp_path / "model.keras",
                          "state": SimpleNamespace(files=[]), "Y_pred": np.ones((2, 6))}

    monkeypatch.setattr(training, "run_training" if wrapped else "train", fake)
    spec = importlib.util.spec_from_file_location("config_example", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = ["--config", str(config_file), "--output-dir", str(tmp_path)]
    if explicit:
        argv += ["--model", explicit]
    assert module.main(argv) == 0
    chosen = calls[0][0][0]
    assert (chosen if wrapped else chosen.name) == (explicit or "mamba_decoder")


def test_installed_cli_uses_config_model(monkeypatch, config_file):
    from unrbl_ml_pipeline import cli, training

    calls = []
    monkeypatch.setattr(training, "run_training", lambda *a, **kw: (calls.append(a), {}))
    assert cli.main(["--config", str(config_file)]) == 0
    assert calls == [("mamba_decoder",)]


@pytest.mark.parametrize("explicit,overrides,expected", [
    (None, {}, "mamba_decoder"),
    (None, {"model_name": "kf_decoder"}, "kf_decoder"),
    ("caleb_decoder", {"model_name": "kf_decoder"}, "caleb_decoder"),
])
def test_runner_resolves_before_training(monkeypatch, config_file, tmp_path, explicit, overrides, expected):
    from unrbl_ml_pipeline.training import runner

    def capture(mod, **kwargs):
        assert mod.name == expected
        assert kwargs["config"]["model_name"] == expected
        raise RuntimeError("captured before training")

    monkeypatch.setattr(runner, "train", capture)
    with pytest.raises(RuntimeError, match="captured before training"):
        runner.run_training(explicit, config_path=config_file, config=overrides,
                            output_dir=tmp_path / "runs")
    saved = next((tmp_path / "runs").glob("*/config.json"))
    assert json.loads(saved.read_text())["model_name"] == expected


def test_low_level_explicit_model_is_authoritative(monkeypatch, config_file):
    import importlib
    module = importlib.import_module("unrbl_ml_pipeline.training.train")

    def capture(mod, cfg):
        assert mod.name == cfg["model_name"] == "caleb_decoder"
        assert cfg["optimizer"] == "sgd"
        raise RuntimeError("captured before preprocessing")

    monkeypatch.setattr("unrbl_ml_pipeline.training.compatibility.validate_loss", capture)
    with pytest.raises(RuntimeError, match="captured before preprocessing"):
        module.train("caleb_decoder", config_path=config_file)
