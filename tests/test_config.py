import pytest

from firerisk.config import load_config


def test_overrides_validated():
    config = load_config(overrides=["model.mode=lora", "training.epochs=2"])
    assert config["model"]["mode"] == "lora"
    assert config["training"]["epochs"] == 2
    with pytest.raises(ValueError, match="Unknown"):
        load_config(overrides=["training.epochz=1"])
    with pytest.raises(ValueError, match="positive"):
        load_config(overrides=["training.accumulation_steps=0"])


def test_inheritance_and_cycles(tmp_path):
    (tmp_path / "base.yaml").write_text("training:\n  epochs: 5\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nmodel:\n  mode: linear\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["training"]["epochs"] == 5
    assert cfg["model"]["mode"] == "linear"
    (tmp_path / "base.yaml").write_text("extends: child.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "child.yaml")
