# Installation

Run setup from a cloned checkout of this repository. The source package
supports Python 3.11 through 3.13. The locked training environment uses
Python 3.12, PyTorch 2.9.1, torchvision 0.24.1 and CUDA 12.8 wheels.
Linux is the validated training platform.

## NVIDIA GPU with the locked environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository root.

```bash
export FIRERISK_HOME="$PWD/.artifacts"
bash scripts/bootstrap.sh
source .venv/bin/activate
firerisk doctor
```

Bootstrap creates `.venv`, installs `uv.lock` without changing it and prints
the artifact directory and activation command. It keeps the uv download cache
under `FIRERISK_HOME/uv-cache` unless `UV_CACHE_DIR` is already set. It does not
change your parent shell, so export `FIRERISK_HOME` before running it and retain
that setting in each training shell.

The NVIDIA driver must support CUDA 12.8. PyTorch ships the runtime libraries,
so a separate CUDA toolkit is unnecessary for these commands. `firerisk doctor`
reports detected devices and free space. If CUDA is unavailable, training
selects CPU. The initial GPU checks used RTX 5090 hardware.

Choose a larger volume for artifacts and the Python environment by exporting
explicit paths before bootstrap.

```bash
export FIRERISK_HOME=/path/to/firerisk-artifacts
export UV_PROJECT_ENVIRONMENT="$FIRERISK_HOME/venv"
bash scripts/bootstrap.sh
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

The paths above are examples. Use directories writable by your account.
Storage defaults and disk estimates are documented in [runtime.md](runtime.md).

## CPU or a platform without the locked CUDA wheel

Use pip in an isolated environment. This preserves the pinned model-library
versions but does not install the entire uv dependency lock.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
export FIRERISK_HOME="$PWD/.artifacts"
firerisk doctor
pytest -q
```

On platforms where the CPU index does not provide a compatible wheel, install
the same PyTorch and torchvision versions from the default pip index instead.
The full training benchmark has only been validated on Linux. CPU execution
is useful for checks, development and prediction. Full fine-tuning on CPU
can be substantially slower.

For an explicit CPU run on a machine that also has an NVIDIA GPU, hide CUDA
from the process and use full precision.

```bash
CUDA_VISIBLE_DEVICES="" firerisk train --config configs/smoke.yaml \
  --set training.precision=fp32 --set training.workers=0
```

A machine with an older GPU may need `training.precision=fp16` or `fp32`.
Reduce the batch size and increase accumulation to fit available memory.
Retain the saved environment record when changing dependencies or hardware.

## Reproduce an experiment

The CLI reads presets relative to the current directory. Run the examples
from the checkout root, or pass an absolute path to `--config`.

```bash
firerisk prepare --config configs/siglip2_base.yaml
firerisk train --config configs/siglip2_base.yaml
firerisk status
```

Preparation downloads the pinned dataset revision and creates audited splits.
The first training run downloads the pinned backbone. Internet access is
needed until those files are cached. Offline reuse is supported after all
required files exist in the selected artifact directory.

`--set KEY=VALUE` overrides a YAML option and may be repeated. For example,
this halves the physical batch and preserves the single-GPU effective batch
of 64 images.

```bash
firerisk train --config configs/siglip2_base.yaml \
  --set training.batch_size=16 --set training.accumulation_steps=4
```

The saved `config.json` records the effective experiment settings. Use the
same artifact directory for preparation, training and evaluation. Changing
the split settings defines a new protocol and requires a new comparison.
