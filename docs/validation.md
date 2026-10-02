# Implementation validation

Checks performed on 2026-10-02 using the locked
PyTorch 2.9.1 CUDA 12.8 stack:

- 115 offline CPU tests passed. They cover portable storage defaults and
  bootstrap paths with spaces, input/split integrity, disk guards, preprocessing,
  trainable parameters, frozen BatchNorm buffers, weighted accumulation,
  partial accumulation windows, best checkpoint selection, exact epoch
  continuation, calibration, report compatibility and recipe isolation.
- Real pretrained SigLIP2 base completed full and LoRA bf16 optimizer updates
  on an RTX 5090. The LoRA configuration trains 596,743 parameters.
- An offline two-GPU `torchrun` integration completed and verified unique
  evaluation indices and two saved RNG states.
- Pinned pretrained DINOv2, ConvNeXt, ResNet50 and SigLIP2 base384 presets
  passed CPU loading, preprocessing, finite seven-class logits and backward
  checks. Enabled checkpointing and DINOv2 positional interpolation worked.
- The official Transformers SigLIP2 base also passed CPU inference/backward
  on a real training image. Loaded vision/pooling tensors matched the joint
  source checkpoint. Strict weight-download checks cover exact missing files,
  sharded checkpoints, offline reuse and the configured filesystem reserve.
- All 70,331 FireRisk images were audited by decoded RGB pixel SHA-256.
  No exact duplicates were found. A separate check compared 64 random and
  shard-boundary packed images with the audit fingerprints.
- The full pinned data protocol contains 49,231 train, 10,552 validation and
  10,548 test examples, with split hash
  `0ab5619091f80f73f8229634a38194ad31eeddf9dfe70978d1b664fe8fb598cd`.
- A real-data GPU smoke run exercised training and compact/best/resumable
  serialization. Validation evaluation generated bootstrap statistics,
  temperature scaling, confusion/reliability plots and a learning curve.
  Single-image CPU prediction also completed.
- Wheel and source distribution built successfully. Ruff checks and offline
  tests run in GitHub CI without dataset/model downloads.
- Inference release exports for both completed SigLIP2 recipes retain every
  source tensor exactly and reproduce the original CLI prediction on a
  training image. Weights-only checkpoint loading and all release checksums
  passed. The public Hugging Face files match the local exports; the large
  checkpoint was verified against its remote LFS SHA-256.
- Dataset figures sample one training image per class with a declared seed
  and verify the split hash, class counts and cached pixel fingerprints.
  Headers of all 70,331 source images were confirmed as 320 by 320 RGB PNGs.

The initial full-data SigLIP2 full and frozen-encoder experiments are separate
from smoke validation. They retain their actual results and status under
`FIRERISK_HOME/runs`; use `firerisk status` for current progress. The initial
comparison fixes seed 42 and the default recipes. These runs do not replace
the validation search and three-seed protocol required for final claims.
Test evaluation is a separate command after locking the recipe.

The source repository and experiment artifacts are separate. Initial development
runs may record a dirty or unavailable Git commit; use a committed release
for final reproducibility and publish the exact `environment.json`, config,
split hash and prediction artifacts with results.
