"""Export an inference checkpoint and public provenance without training data.

Install FireRisk Bench first. This command writes a local directory and never
creates or uploads a Hugging Face repository.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import validate_repo_id

from firerisk.config import validate_config
from firerisk.models import _timm_config


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_config(config):
    config = copy.deepcopy(config)
    config["storage"]["root"] = None
    validate_config(config)
    return config


def assert_portable(value):
    if isinstance(value, dict):
        for nested in value.values():
            assert_portable(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_portable(nested)
    elif isinstance(value, str) and Path(value).is_absolute():
        raise ValueError("Public metadata contains an absolute local filesystem path")


def model_card(repo_id, config, summary, manifest, base_repo, environment):
    mode = config["model"]["mode"]
    adaptation = {"full": "full fine-tuning", "linear": "frozen encoder probe", "lora": "LoRA"}[
        mode
    ]
    raw = summary["evaluation"]["validation"]["raw"]
    model = summary["model"]
    base_revision = config["model"]["revision"]
    backbone_title = (
        "SigLIP2 ViT-B/16"
        if config["model"]["name"] == "vit_base_patch16_siglip_224.v2_webli"
        else config["model"]["name"]
    )
    backbone_name = config["model"]["name"].lower()
    if "siglip2" in backbone_name or ("siglip" in backbone_name and ".v2" in backbone_name):
        family_tag = "siglip2"
    elif "siglip" in backbone_name:
        family_tag = "siglip"
    else:
        family_tag = "vision"
    source_commit = environment.get("git_commit")
    code_provenance = (
        f"The recorded training code commit is `{source_commit}`."
        if source_commit
        else "The recorded artifacts do not identify a recoverable training Git commit."
    )
    if environment.get("git_dirty"):
        code_provenance += " Training used a dirty working tree."
    content = (
        "The checkpoint contains the fine-tuned vision encoder and classifier."
        if mode == "full"
        else "The compact checkpoint contains trainable weights and buffers. "
        "It requires the pinned pretrained backbone, downloaded by FireRisk Bench."
    )
    return f"""---
language:
  - en
pipeline_tag: image-classification
tags:
  - firerisk
  - remote-sensing
  - {family_tag}
  - pytorch
datasets:
  - blanchon/FireRisk
base_model: {base_repo}
license: gpl-3.0
---

# FireRisk {backbone_title} {adaptation}

This model assigns one of seven aerial hazard labels to an RGB image. It uses
the {backbone_title} image encoder and the FireRisk Bench classifier wrapper.
{content}

These are initial development weights from training seed {summary['seed']}.
The selected checkpoint is epoch {summary['best_epoch']}. It was selected by
validation macro F1. The test partition has not been evaluated.

| Validation metric | Value |
|---|---:|
| Accuracy | {raw['accuracy'] * 100:.2f}% |
| Macro F1 | {raw['macro_f1'] * 100:.2f}% |
| Balanced accuracy | {raw['balanced_accuracy'] * 100:.2f}% |

The validation partition also selected the checkpoint and fitted the saved
temperature. Its metrics do not estimate independent test performance.
Single-seed results do not establish variation across training seeds. The
scores of releases with different recipe settings do not form a controlled
causal ablation of adaptation mode.

## Loading

Use [FireRisk Bench](https://github.com/yunusserhat/firerisk). These files use
its `VisionClassifier` architecture. They do not provide a Transformers
`AutoModel` or `pipeline` checkpoint.

```bash
git clone https://github.com/yunusserhat/firerisk.git
cd firerisk
export FIRERISK_HOME="$PWD/.artifacts"
bash scripts/bootstrap.sh
source .venv/bin/activate
hf download {repo_id} --local-dir ./downloaded-model
firerisk predict --run-dir ./downloaded-model --image /path/to/aerial_image.png
```

For CPU installation, follow the repository's `docs/installation.md`.
Prediction does not download FireRisk. It downloads the pinned backbone on
the first load, even for the full release, because the current builder starts
from the pretrained architecture. Subsequent loads can reuse that cache.
`best.pt` loads with `torch.load(..., weights_only=True)`.

## Data and training

The input is `blanchon/FireRisk` revision `{config['data']['revision']}`.
Its mirror contains {manifest['num_rows']:,} images and only the source training
partition. This project defines a new image-level split with
{manifest['split_sizes']['train']:,} training, {manifest['split_sizes']['validation']:,}
validation and {manifest['split_sizes']['test']:,} test examples.
It is not the original paper's benchmark split.

The split seed is {config['data']['split_seed']}. Pixel duplicate auditing
found {manifest['audit']['duplicate_rows']} exact duplicate rows. The split hash is
`{manifest['split_hash']}`.

The backbone is `{base_repo}` revision `{base_revision}`.
The classifier has {model['total_parameters']:,} total parameters and
{model['trainable_parameters']:,} trainable parameters. Input size is
{config['model']['image_size']} pixels. `config.json` and `preprocessing.json`
retain the recipe, normalization, resize policy and augmentation settings.
`provenance.json` preserves the recorded training environment.
{code_provenance}

`calibration.json` holds a temperature fitted on validation logits. The CLI
uses it for its probability output. Class order is stored in `labels.json`
and `summary.json`. `validation_metrics.json` includes per-class scores,
confusion matrices and exploratory bootstrap intervals.

## Intended use

The model supports research on WHP-derived aerial hazard classes. Its class
probabilities do not represent a calibrated probability of future wildfire
occurrence. It is not an active-fire detector.

The mirror has no coordinates, timestamps or scene identifiers. Exact pixel
duplicate auditing does not exclude nearby or overlapping imagery across
partitions. Geographic or temporal transfer has not been established.
Label ambiguity, image acquisition changes and unknown pretraining overlap
also limit interpretation.

## Files and licenses

The release contains an inference checkpoint, portable configuration, class
mapping, preprocessing, validation results and source provenance. It excludes
source images, dataset shards, prediction arrays, optimizer state and credentials.
`SHA256SUMS` lists file checksums.

This release's fine-tuning additions and classifier use GNU GPL v3.
The original pretrained model retains its Apache 2.0 terms and attribution.
See `LICENSE`, `UPSTREAM_LICENSE`, `UPSTREAM_NOTICE.md` and the unmodified
`upstream_model_card.md`. The frozen release does not redistribute its
unchanged backbone weights. The dataset card lists its license as unknown.
The code or model release does not license the training imagery.
"""


def export(args):
    validate_repo_id(args.repo_id)
    run, output = Path(args.run_dir).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be empty to prevent accidental overwrites")
    config = public_config(read_json(run / "config.json"))
    summary = read_json(run / "summary.json")
    if summary.get("status") != "complete" or summary.get("smoke_test", False):
        raise ValueError("Publication requires a completed non-smoke training run")
    if "test" in summary.get("evaluation", {}):
        raise ValueError("This initial-release template is restricted to validation-only runs")
    if "validation" not in summary.get("evaluation", {}):
        raise ValueError("Saved validation metrics are required")
    revision = config["model"].get("revision")
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Backbone revision must be an immutable 40-character commit")
    if config["model"]["backend"] == "timm":
        base_repo = _timm_config(config["model"]["name"])["hf_hub_id"].split("@")[0]
    else:
        base_repo = config["model"]["name"]
    info = HfApi().model_info(base_repo, revision=revision)
    if info.card_data.get("license") != "apache-2.0":
        raise ValueError("Review the upstream license before adapting this export template")
    source_card = hf_hub_download(
        base_repo, "README.md", revision=revision, cache_dir=args.cache_dir
    )
    upstream_notices = [
        sibling.rfilename
        for sibling in info.siblings
        if any(
            term in Path(sibling.rfilename).name.upper() for term in ["LICENSE", "NOTICE", "COPYING"]
        )
    ]
    if upstream_notices:
        raise ValueError("Retain the snapshot's additional upstream license notices before export")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != summary["class_names"]:
        raise ValueError("Checkpoint and summary class mappings disagree")
    if checkpoint["config"]["model"] != config["model"]:
        raise ValueError("Checkpoint and saved model configuration disagree")
    if checkpoint["epoch"] + 1 != summary["best_epoch"]:
        raise ValueError("Checkpoint and selected epoch disagree")
    checkpoint = {
        key: checkpoint[key]
        for key in ["model", "protocol_hash", "class_names", "epoch", "best_score", "world_size"]
    }
    checkpoint["config"] = config
    manifest = read_json(run / "data_manifest.json")
    for key in ["storage_root", "packed_images"]:
        manifest.pop(key, None)
    manifest["shards"] = [
        {key: value for key, value in shard.items() if key != "local_path"}
        for shard in manifest["shards"]
    ]
    summary["config"] = config
    summary["release_kind"] = "initial single-seed validation checkpoint"
    source_environment = read_json(run / "environment.json")
    environment = {
        key: source_environment.get(key)
        for key in ["python", "packages", "cuda_runtime", "gpus", "git_commit", "git_dirty"]
    }
    metadata = {
        "config.json": config,
        "summary.json": summary,
        "preprocessing.json": read_json(run / "preprocessing.json"),
        "data_manifest.json": manifest,
        "history.json": read_json(run / "history.json"),
        "calibration.json": read_json(run / "calibration.json"),
        "validation_metrics.json": summary["evaluation"]["validation"],
        "labels.json": {
            "id2label": dict(enumerate(summary["class_names"])),
            "label2id": {label: index for index, label in enumerate(summary["class_names"])},
        },
        "provenance.json": {
            "model_repo_id": args.repo_id,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_environment": environment,
            "source_best_sha256": sha256(run / "best.pt"),
            "source_split_hash": manifest["split_hash"],
            "source_protocol_hash": summary["protocol_hash"],
            "base_model": base_repo,
            "base_model_revision": revision,
            "base_model_license": "apache-2.0",
            "code_repository": "https://github.com/yunusserhat/firerisk",
            "inference_format": "FireRisk Bench VisionClassifier, weights-only PyTorch payload",
        },
    }
    for value in metadata.values():
        assert_portable(value)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in metadata.items():
        write_json(output / name, value)
    torch.save(checkpoint, output / "best.pt")
    if args.safetensors:
        from safetensors.torch import save_file

        save_file(
            {name: tensor.contiguous() for name, tensor in checkpoint["model"].items()},
            output / "model.safetensors",
            metadata={"architecture": "FireRisk Bench VisionClassifier"},
        )
    shutil.copyfile(args.license_file, output / "LICENSE")
    shutil.copyfile(source_card, output / "upstream_model_card.md")
    request = urllib.request.Request(
        "https://www.apache.org/licenses/LICENSE-2.0.txt", headers={"User-Agent": "FireRiskBench"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        (output / "UPSTREAM_LICENSE").write_bytes(response.read())
    (output / "UPSTREAM_NOTICE.md").write_text(
        f"# Pretrained model attribution\n\n"
        f"Source https://huggingface.co/{base_repo}/tree/{revision}\n\n"
        "The source model card declares Apache 2.0. Its unmodified text is\n"
        "retained in upstream_model_card.md. The selected snapshot supplies\n"
        "no separate LICENSE or NOTICE file. UPSTREAM_LICENSE reproduces the\n"
        "Apache 2.0 license from https://www.apache.org/licenses/LICENSE-2.0.txt.\n\n"
        "The full release modifies the pretrained vision weights and adds a\n"
        "trained normalization layer and classifier. The frozen release only\n"
        "distributes trained normalization and classifier parameters, with\n"
        "the unchanged backbone fetched separately at the recorded revision.\n\n"
        "Refer to the retained upstream model card for architecture and\n"
        "citations. The original source terms and attribution remain\n"
        "applicable to upstream material.\n"
    )
    (output / "README.md").write_text(
        model_card(args.repo_id, config, summary, manifest, base_repo, environment)
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(output.iterdir()))
    )
    safe_reload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    if set(safe_reload) != set(checkpoint):
        raise RuntimeError("Exported checkpoint failed weights-only reload")
    print(json.dumps({"output": str(output), "files": len(list(output.iterdir()))}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--license-file", default=str(Path(__file__).resolve().parents[1] / "LICENSE"))
    parser.add_argument("--safetensors", action="store_true", help="Also export wrapper state tensors")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
