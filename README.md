# FireRisk Bench

A reproducible training and evaluation toolkit for seven-class aerial fire
hazard classification using [FireRisk](https://huggingface.co/datasets/blanchon/FireRisk).
The default model is SigLIP2 ViT-B/16. Presets also support LoRA, a frozen
encoder, DINOv2, ConvNeXt and ResNet50.

The repository runs on your own workstation or server. Dataset downloads,
model caches and checkpoints use a directory you choose. No institutional
machine, scheduler or pre-existing data is required.

## Start training

Clone this repository and run the commands from its root. For Linux with an
NVIDIA GPU, install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and use the locked Python 3.12 and CUDA 12.8 environment.

```bash
git clone https://github.com/yunusserhat/firerisk.git
cd firerisk
export FIRERISK_HOME="$PWD/.artifacts"
bash scripts/bootstrap.sh
source .venv/bin/activate
firerisk doctor
firerisk prepare
firerisk train --config configs/siglip2_base.yaml
```

The driver must support the CUDA version shipped with PyTorch. A separate
CUDA toolkit is not needed. For CPU installation, other environments and
moving the environment to a larger volume, see [installation](docs/installation.md).

Packed data access uses about **23 GB** for source shards and the image archive,
before environments, pretrained weights, checkpoints and reserved free space.
Set `FIRERISK_HOME` to a larger writable directory before preparation if needed.
The CLI otherwise uses `$XDG_CACHE_HOME/firerisk` or `~/.cache/firerisk`.
[Runtime and storage](docs/runtime.md) covers smaller disks and multiple GPUs.

Training selects `best.pt` using validation macro F1. `firerisk status` lists
saved progress. Resume interrupted training from the most recent saved epoch.

```bash
firerisk train --config configs/siglip2_base.yaml \
  --resume "$FIRERISK_HOME/runs/siglip2-base-full-seed42/last.pt"
```

After choosing and freezing the recipe, evaluate the held-out test partition.
Single-image prediction uses the same saved preprocessing and class mapping.

```bash
firerisk evaluate --run-dir "$FIRERISK_HOME/runs/siglip2-base-full-seed42"
firerisk predict --run-dir "$FIRERISK_HOME/runs/siglip2-base-full-seed42" \
  --image /path/to/aerial_image.png
```

## Compare models

| Preset | Backbone | Adaptation | Image size |
|---|---|---|---:|
| `siglip2_base.yaml` | SigLIP2 ViT-B/16 | Full | 224 |
| `siglip2_linear.yaml` | SigLIP2 ViT-B/16 | Frozen encoder with trained classifier | 224 |
| `siglip2_lora.yaml` | SigLIP2 ViT-B/16 | LoRA and classifier | 224 |
| `siglip2_384.yaml` | SigLIP2 ViT-B/16 | Full | 384 |
| `siglip2_so400m.yaml` | SigLIP2 So400m/14 | Full | 384 |
| `siglip2_so400m_lora.yaml` | SigLIP2 So400m/14 | LoRA and classifier | 384 |
| `siglip2_hf.yaml` | Google SigLIP2 FixRes | Full, Transformers | 224 |
| `dinov2.yaml` | DINOv2 ViT-B/14 | Full | 224 |
| `convnext.yaml` | ConvNeXt-Tiny | Full | 224 |
| `resnet50.yaml` | ResNet50 | Full | 224 |

The dataset and pretrained model revisions are pinned. Training includes
AdamW, separate backbone and classifier rates, warmup and cosine decay,
mixed precision, gradient accumulation and optional gradient checkpointing.
The presets are starting points whose performance should be established with
validation experiments.

```bash
python scripts/tune.py --config configs/siglip2_base.yaml \
  --lr 0.00001 0.00003 --weight-decay 0.01 0.05

python scripts/run_suite.py --configs configs/siglip2_base.yaml \
  configs/siglip2_linear.yaml configs/siglip2_lora.yaml configs/dinov2.yaml \
  configs/convnext.yaml configs/resnet50.yaml --seeds 42 100 2026
```

Freeze the comparison matrix before adding `--evaluate` to the suite command.
Use `--resume` to skip completed experiments and resume saved epochs. Then
export compatible results across independent seeds.

```bash
firerisk compare --runs "$FIRERISK_HOME/runs" \
  --output "$FIRERISK_HOME/reports/comparison"
```

The reports include classification metrics, confusion matrices, calibration,
bootstrap intervals and variation across seeds. Each run records its
configuration, environment, preprocessing, data manifest, progress and
checkpoints. [The research protocol](docs/protocol.md) explains fair tuning,
uncertainty, ensemble evaluation and the limits of the labels.

## Data and initial results

The pinned mirror contains 70,331 RGB images with seven labels and only one
source split. This toolkit creates a new reproducible 70/15/15 split after
checking decoded pixels for exact duplicates. The split seed is fixed
independently of training seeds. These scores therefore use a different
protocol from the original FireRisk paper.

| Initial experiment | Validation accuracy | Validation macro F1 |
|---|---:|---:|
| Frozen SigLIP2 encoder | 55.95% | 50.19% |
| Full SigLIP2 fine-tuning | 63.05% | 58.94% |

These are single-seed development results. The test partition has not been
evaluated. Details and figures appear in [initial results](docs/initial_results.md).
The mirror lacks coordinates and scene identifiers, so a random image split
cannot establish geographic generalization. The model predicts the dataset's
hazard labels rather than future wildfire occurrence.

## Published checkpoints

The initial [full fine-tuning checkpoint](https://huggingface.co/yunusserhat/firerisk-siglip2-base)
and [frozen encoder checkpoint](https://huggingface.co/yunusserhat/firerisk-siglip2-base-frozen)
are available on Hugging Face. Each release includes its configuration,
preprocessing, class mapping, calibration, validation results and checksums.
The frozen release stores the trained classifier and loads the pinned
pretrained encoder separately. Both use this toolkit's inference command.

After installing the toolkit and setting `FIRERISK_HOME`, download a release
and predict without downloading FireRisk.

```bash
hf download yunusserhat/firerisk-siglip2-base \
  --local-dir "$FIRERISK_HOME/releases/siglip2-base"
firerisk predict --run-dir "$FIRERISK_HOME/releases/siglip2-base" \
  --image /path/to/aerial_image.png
```

To prepare a release from your own completed validation run, export an
inference checkpoint to a separate directory. This command does not upload it.

```bash
python scripts/export_model.py \
  --run-dir "$FIRERISK_HOME/runs/siglip2-base-full-seed42" \
  --output "$FIRERISK_HOME/releases/my-siglip2-model" \
  --repo-id your-account/your-model
```

## Checks and license

```bash
pytest -q
ruff check src tests scripts
firerisk train --config configs/smoke.yaml
```

CI exercises tiny offline CPU models. The smoke preset downloads actual data
and pretrained weights for an intentionally bounded integration check.
[Validation](docs/validation.md) records the completed implementation checks.
A [model card template](docs/model_card_template.md) and
[bibliographic references](docs/references.bib) support reporting.

Code is licensed under the [GNU General Public License v3](LICENSE).
Dataset and pretrained model terms remain separate. The FireRisk Hub card
lists the dataset license as unknown. Source images, caches and checkpoints
are excluded from Git and can be fetched or generated using the commands above.
