---
language:
  - en
pipeline_tag: image-classification
tags:
  - firerisk
  - remote-sensing
  - siglip2
datasets:
  - blanchon/FireRisk
base_model: timm/vit_base_patch16_siglip_224.v2_webli
license: "<verify the derived-checkpoint release license>"
---

# <Model name and version>

Replace every angle-bracket placeholder using saved run artifacts. This is a template, not a claim of completed training.

## Model details

This model assigns one of seven WHP-derived hazard labels to an RGB aerial image: `high`, `low`, `moderate`, `non-burnable`, `very_high`, `very_low`, and `water`.

| Field | Value |
|---|---|
| Base model and immutable revision | <from config.json and summary.json> |
| Vision backend and pooling | <timm/Transformers; exact pretrained pooling> |
| Adaptation mode | <frozen-backbone classifier probe / LoRA / full> |
| Total and trainable parameters | <from summary.json> |
| Input size and preprocessing | <from preprocessing.json> |
| Training seed | <integer> |
| Code commit and environment | <from environment.json> |
| Training organization/contact | <responsible maintainer> |

The model uses pretrained visual representations followed by a seven-class classification head. Describe the trainable LayerNorm when reporting a frozen-backbone probe. For adapters, disclose rank, alpha, dropout and actual target modules; fused `qkv` adapters are not query/value-only adapters.

## Data and protocol

- Dataset source: `blanchon/FireRisk`, revision `<immutable commit>`.
- Source examples/shards: `<from data_manifest.json>`.
- Observed and retained class counts: `<include all seven classes>`.
- Split seed, actual train/validation/test sizes and split hash: `<from data_manifest.json>`.
- Duplicate hash mode, removed rows and conflicting-label handling: `<from audit>`.
- Protocol and scientific recipe hashes: `<from summary and comparison report>`.

The referenced Hub mirror provides the 70,331-example training portion of the original dataset. Any train/validation/test partitions created here define a new benchmark. Scores must not be presented as results on the original FireRisk validation set.

The mirror exposes image and label without coordinates, time or scene groups. Exact duplicate removal does not establish independence between nearby or overlapping images. Document any additional similarity review or geographic holdout separately.

## Training

Record optimizer, learning rates for head/backbone, weight decay rules, batch size, gradient accumulation, effective batch size, warmup, scheduler, precision, clipping, augmentation, loss and class-imbalance treatment. State the validation metric, checkpoint selection, patience, maximum epochs and selected epoch. Include GPU type, training time and peak allocated memory.

State whether this release is an exploratory bounded run or part of the final locked experiment. Identify all validation-only tuning and its budget. Test data must not have selected this model's hyperparameters.

## Evaluation

| Metric | Held-out value | Uncertainty and averaging domain |
|---|---:|---|
| Accuracy | <value> | <example bootstrap interval> |
| Macro F1 | <value> | <declared seven-class schema; example interval> |
| Balanced accuracy | <value> | <classes present> |
| High/very-high recall | <value> | <supports and separate class recalls> |
| One-vs-rest macro AUROC | <value or undefined> | <classes with positives and negatives> |
| Raw NLL / Brier / ECE | <values> | <Brier sum across classes; ECE 15 bins> |
| Calibrated NLL / Brier / ECE | <values> | <validation-only temperature and bounds> |

Attach each class's precision, recall, F1, support and confusion matrix. If reporting ordinal metrics, include only the five hazard classes and publish valid-pair coverage plus the count of hazard examples predicted as water/non-burnable. Do not assign those two classes an ordinal rank.

For a multi-seed release, list individual seeds/results and mean ± sample standard deviation. Distinguish example-bootstrap confidence intervals from Student-t intervals for the mean across training seeds. A single checkpoint cannot establish variance across training seeds.

## Intended use and limitations

The model supports research on image-based transfer to WHP-derived hazard classes. Its probability vector describes model confidence over these labels and is not a calibrated probability of future wildfire occurrence.

Document that temporal transfer, geographic transfer and alternative image acquisition conditions have <been evaluated / not been evaluated>. Explicitly discuss label ambiguity, spatial similarity across random splits, missing geospatial metadata, possible pretraining overlap and the effect of cropping on whole-tile labels. State the evidence available for any operational use rather than inferring it from benchmark performance.

## Reproduction and files

Provide the complete configuration, source code commit, package versions, immutable base-model and dataset revisions, split artifact/hash, evaluation command and result summaries. The required backbone checkpoint is `<base model revision>` when distributing a classifier probe or adapter without frozen backbone weights.

List the checksums of released files and indicate whether each file contains a full model, trainable delta, optimizer state or result artifact. Keep source images outside the repository unless their redistribution terms have been established.

## Licenses and citations

The repository code uses GNU GPL v3. Pretrained model licenses, derived-checkpoint release terms and dataset terms are separate. The Hub dataset card lists its license as unknown. Specify the reviewed release terms for this checkpoint and preserve applicable notices; do not infer dataset rights from the paper or code license.

Cite FireRisk, the exact pretrained encoder, LoRA when applicable, temperature scaling when applicable, and this repository/version. Bibliographic entries are supplied in `docs/references.bib`.
