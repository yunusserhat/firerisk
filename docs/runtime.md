# Runtime and storage

Artifact storage is independent of the source checkout. Its selection order
is `storage.root` in the configuration, `FIRERISK_HOME`, then the user's cache
directory. The default cache is `$XDG_CACHE_HOME/firerisk` when XDG_CACHE_HOME
is an absolute path, otherwise `~/.cache/firerisk`. An unwritable chosen path
raises an error without silently moving the experiment.

The [quick start](../README.md) explicitly uses the checkout's ignored
`.artifacts` directory. Select another disk before downloading data if needed.

```bash
export FIRERISK_HOME=/path/to/firerisk-artifacts
firerisk doctor
```

All Hugging Face and torch caches, prepared data and default run directories
use this root. The data layer avoids a separate Datasets Arrow cache and an
extracted image directory. A run can override its output directory with
`firerisk train --run-dir /path/to/run`.

## Disk estimates

| Artifact | Approximate GiB |
|---|---:|
| Original dataset Parquet files | 10.78 |
| Optional indexed archive of original image bytes | 10.8 |
| CUDA environment and uv cache | 7 |
| SigLIP2 base image encoder cache | 0.35 |
| Full base best checkpoint | 0.35 |
| Full base resumable checkpoint with Adam moments | 1.05 |
| Base LoRA best checkpoint | 0.01 |

The environment estimate assumes uv can reuse files on the same filesystem.
Separate filesystems or a different dependency stack can increase the total.
An atomic checkpoint save temporarily keeps the old and new file. The default
free-space reserve is 5 GiB and is enforced before downloads, packing and
checkpoint writes. Filesystem free space does not include account or project
quotas. Existing user files are never deleted automatically.

`packed` mode stores unchanged compressed image bytes in a memory-mapped file
for shuffled reads with a bounded memory footprint. The original Parquet
files remain available for provenance. To avoid the extra archive, choose
`parquet` in both preparation and training.

```bash
firerisk prepare --set data.access_mode=parquet
firerisk train --set data.access_mode=parquet --set training.workers=0 \
  --set data.cache_row_groups=720
```

Random Parquet access with a small row-group cache is slow. The example cache
can hold a substantial part of the encoded corpus in RAM. Adjust it to your
memory budget and avoid replicating large caches across workers or GPUs.
Packed mode is recommended when its additional disk allocation fits. Audit
caches and packed bytes are shared across split seeds, while every run stores
its immutable manifest.

## GPU memory and distributed training

The default single-GPU effective batch is 64 images from a physical batch of
32 and two accumulation steps. For a smaller GPU, try a batch of 8 and eight
accumulation steps. LoRA and the frozen encoder can also reduce training memory.
Pretrained weights are omitted from compact adapter and probe checkpoints,
so their recorded backbone revision must remain available in the Hub cache.

Prepare the full data once before launching multiple GPU processes.

```bash
firerisk prepare
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m firerisk train --config configs/siglip2_base.yaml \
  --set training.batch_size=16 --set experiment.name=siglip2-base-ddp
```

This preserves effective batch 64 with 16 images, two accumulation steps and
two GPUs. Distributed evaluation assigns each example once without padding.
Training's distributed sampler may pad a few examples to give each rank the
same number of batches. World size is recorded as part of the recipe.

Resume `last.pt` using the same scientific settings and world size. Epoch
boundaries restore optimizer, scheduler, scaler, loader generator and each
rank's random state. Determinism across a different GPU or dependency stack
is not implied by reusing a seed.

Keep data, caches and weight files out of Git. Back up chosen checkpoints and
result tables if the selected storage volume has an automatic cleanup policy.
