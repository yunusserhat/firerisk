# FireRisk research protocol

This repository studies transfer from pretrained vision encoders to seven-class RGB aerial hazard classification. It provides a reproducible experiment framework; a smoke test or a short exploratory run is not evidence that a model is state of the art. The default starting point is the SigLIP2 Base vision encoder. Final claims require completed runs, a fixed evaluation protocol, and repeated training seeds.

## Dataset identity and the new benchmark

The pinned input is `blanchon/FireRisk` at revision `234b2e7fe6be2da773472e83bd4d42cc9815a630`. The Hub metadata declares **70,331 images in one `train` split**, seven labels, and 24 Parquet shards totaling 11,575,727,336 bytes. Use the generated manifest to obtain actual observed class counts. Viewer statistics can be partial and should not be used as the complete class distribution. [Dataset card](https://huggingface.co/datasets/blanchon/FireRisk/blob/234b2e7fe6be2da773472e83bd4d42cc9815a630/README.md).

The original paper describes 91,872 examples, comprising 70,331 training and 21,541 validation examples. The referenced Hub mirror exposes only the former. Its imagery comes from NAIP, with hazard labels derived from WHP 2020 and imagery collected over 2019–2020; the construction section describes RGB images cropped to 320 × 320. The published benchmark selected the highest accuracy over 100 epochs. [FireRisk paper](https://arxiv.org/html/2303.07035v1).

The supplied protocol therefore creates a **new benchmark within the Hub training mirror**. It does not reconstruct the original held-out validation set, and a score on this protocol is not directly comparable with the original paper's results. Report the source revision, split sizes, class counts, duplicate audit, and split hash in the paper. Never describe the Hub's single `train` split as an official test set.

The class IDs remain:

| ID | Label |
|---:|---|
| 0 | high |
| 1 | low |
| 2 | moderate |
| 3 | non-burnable |
| 4 | very_high |
| 5 | very_low |
| 6 | water |

The dataset card lists its license as `unknown`, and the original project does not supply a dataset license in its repository. The repository's code license does not license the input imagery. Record the source terms and clarify dataset redistribution rights before bundling images with a release. The original paper's document license does not establish a license for the dataset. [Dataset card](https://huggingface.co/datasets/blanchon/FireRisk), [original project](https://github.com/CharmonyShen/FireRisk).

## Splitting and leakage controls

The default preparation uses one fixed split seed, 70% training, 15% validation, and 15% test, allocated by class. Exact duplicate groups are formed before allocation. Pixel hashing decodes RGB pixels, corrects EXIF orientation, and hashes the pixel content and dimensions; images with the same pixels but different encodings therefore remain one group. The default removes redundant members while preserving one example. Conflicting labels within an exact duplicate group stop preparation for explicit investigation.

The resulting manifest records realized fractions, class counts, removed duplicate rows, and group overlap. `splits.npz` stores the immutable source-row indices. Its content hash includes both the index arrays and labels, independent of ZIP timestamps. Training seeds must change model optimization randomness while retaining the same split seed and split artifact.

Exact duplicate control does **not** establish geographic independence. The mirror's public schema contains only image and label; it does not expose coordinates, acquisition dates, or scene-group identifiers. Neighboring locations, overlapping imagery, visually similar tiles, or different acquisitions of the same location can remain across partitions. This limits claims about geographic transfer.

For a stronger follow-up study, obtain reliable coordinates or original scene identifiers and define spatial blocks before inspecting results. Hold out entire regions with a stated buffer distance, assess overlap, and use the same blocks for all models. A perceptual or embedding similarity audit can flag candidates for review, but similarity alone is not a geographic label. Any newly identified groups that change splits create a new protocol and invalidate previously selected test results for that protocol.

## Models and fine-tuning conditions

The default `vit_base_patch16_siglip_224.v2_webli` checkpoint is the SigLIP2 image tower distributed by timm. It keeps the pretrained attention pooling and uses the checkpoint's preprocessing. The timm card reports 92.9M parameters and Apache-2.0 licensing. [Model card](https://huggingface.co/timm/vit_base_patch16_siglip_224.v2_webli).

Google's fixed-resolution SigLIP2 checkpoints use the Transformers `SiglipVisionModel` implementation. NaFlex uses a distinct patch-input representation and is outside this implementation's supported model family. [Official Transformers documentation](https://huggingface.co/docs/transformers/en/model_doc/siglip2).

Evaluate three transfer conditions:

| Condition | Updated weights | Purpose |
|---|---|---|
| `linear` | LayerNorm and linear classifier; frozen backbone in evaluation mode | Measure separability of pretrained representations with a small trainable head |
| `lora` | Attention adapters and classifier | Measure adaptation under a small trainable-parameter budget |
| `full` | Entire vision backbone and classifier | Measure full adaptation |

The `linear` condition includes a trainable LayerNorm and should be described as a **frozen-backbone classifier probe**, rather than a strictly unconstrained linear regression on raw features. Dropout applies only during head training. LoRA targets are explicit: fused `qkv` projections for supported timm ViTs, and `q_proj`/`v_proj` for the Transformers SigLIP backbone. A fused `qkv` adapter updates a combined projection and should not be described as an adapter on only query and value. [PEFT custom-model guidance](https://huggingface.co/docs/peft/en/developer_guides/custom_models), [LoRA paper](https://arxiv.org/abs/2106.09685).

Checkpoint-specific resizing, interpolation, mean and standard deviation must remain consistent with the pretrained encoder. The training transform uses mild crops, flips and rotations in multiples of 90 degrees. Record any augmentation ablations. Whole-tile hazard labels can become less representative of an aggressively cropped region; stronger augmentation should earn its inclusion through validation experiments.

The starting recipe uses AdamW, separate classifier/backbone learning rates, cosine decay with warmup, gradient clipping, and configurable mixed precision. These defaults are hypotheses for validation, not a guaranteed optimal recipe. The two class-imbalance options, weighted loss and weighted sampling, are mutually exclusive. Compare ordinary cross-entropy first, then one imbalance treatment at a time. Keep any MixUp experiment separate, with its own saved recipe.

## Selection and fair comparison

1. Prepare and freeze the dataset split once. Keep test images out of hyperparameter selection.
2. Select learning rates, regularization, resolution and training duration using training/validation only. Give every baseline a stated, comparable tuning budget.
3. Select the epoch by validation macro F1 and record the early-stopping rule. The saved `best.pt` is the selected model; `last.pt` is for resuming optimization.
4. Lock one recipe per condition and run at least three independent training seeds on the same split. Record unsuccessful runs, memory failures, and changes required to fit hardware.
5. Evaluate the locked models on test data and publish all seeds. Comparing successive settings using test scores turns the test partition into a validation partition.

Equal image size and effective batch size help comparisons, but they do not make different pretraining datasets or compute budgets equivalent. Report the pretrained model revision, parameter counts, trainable counts, optimization budget, actual selected epoch, wall time, peak allocated GPU memory, GPU model, dependency versions and git commit. Resolution scaling deserves a separate ablation because it changes both compute and image detail.

## Three-seed execution recipe

The commands below assume installation is complete and `FIRERISK_HOME` identifies the chosen artifact directory. The split seed remains the preset value. The loop is the **final locked comparison**, following validation-only tuning; tailor each mode's frozen recipe if tuning selected different learning rates.

```bash
firerisk doctor --config configs/siglip2_base.yaml
firerisk prepare --config configs/siglip2_base.yaml

for transfer_mode in linear lora full; do
  for training_seed in 42 100 2026; do
    firerisk train --config configs/siglip2_base.yaml \
      --set "model.mode=$transfer_mode" \
      --set "experiment.seed=$training_seed" \
      --set "experiment.name=siglip2-$transfer_mode"
  done
done

# Run this evaluation stage after all recipes have been locked.
for transfer_mode in linear lora full; do
  for training_seed in 42 100 2026; do
    firerisk evaluate \
      --run-dir "$FIRERISK_HOME/runs/siglip2-$transfer_mode-seed$training_seed" \
      --split test --bootstrap-samples 1000
  done
done

firerisk compare --runs "$FIRERISK_HOME/runs" --output reports/final-comparison
```

Prepare the full data once before starting multiple GPU processes with `torchrun`. A configuration using `max_shards`, `max_train_batches`, `max_eval_batches`, or random initialization is marked as a smoke experiment and omitted from comparison tables by default. A bounded run over all data can still be exploratory: its completed execution does not establish that the model reached its best achievable performance.

## Metrics, calibration and uncertainty

The primary selection metric is macro F1 over every declared class. Also report accuracy, balanced accuracy, weighted F1, each class's precision/recall/F1/support, and the confusion matrix. Balanced accuracy averages recall over classes present in the evaluated partition; macro F1 retains the declared schema. Missing classes therefore require explicit discussion.

One-vs-rest AUROC is defined only where both positive and negative examples exist. The output records which classes entered the macro AUROC average. Do not compare two macro AUROCs with different averaging domains without saying so. The combined `high`/`very_high` recall summarizes screening of the highest hazard classes, while the individual recalls remain necessary.

Only five labels have an ordinal order: `very_low < low < moderate < high < very_high`. Ordinal MAE, within-one-level accuracy, severe underestimation and quadratic weighted kappa apply only where both target and prediction are in these classes. The report includes the number of valid pairs, coverage, and hazard examples predicted outside the ordinal classes. Always show coverage with an ordinal score; excluding water/non-burnable predictions can otherwise hide serious mistakes.

NLL uses probabilities clipped at 1e-15. The multiclass Brier score sums squared errors across classes per example. ECE uses 15 equal-width bins of maximum confidence; state this binning because ECE depends on it. Inspect reliability plots alongside the scalar value.

Temperature scaling fits one positive scalar on validation logits to reduce validation NLL and then applies it unchanged to test logits. It preserves class ranking and argmax predictions. Raw classification results remain the primary comparison; publish raw and calibrated NLL/Brier/ECE separately. The validation data also selected the checkpoint, so its calibrated scores are not independent calibration-test estimates. For a dedicated calibration study, split validation into selection and calibration subsets in a new protocol. [Calibration reference](https://proceedings.mlr.press/v70/guo17a.html).

The example bootstrap resamples within each observed class, preserves its count, and reports percentile 95% intervals for the principal scalar metrics. These intervals condition on the observed class mixture. They do not capture spatial dependence, future distribution shift or optimization-seed variation. Spatial blocks require a block bootstrap if coordinates become available.

The comparison report separately shows the mean and sample standard deviation across training seeds, plus a 95% Student-t interval for the seed mean when at least two seeds exist. With three seeds that interval is usually imprecise; report the individual values too. Different optimization/model settings remain separate recipe groups. Duplicate seeds are rejected, and mismatched protocol/split hashes are rejected unless explicitly allowed as separate groups. These intervals are not a substitute for a preregistered paired significance test when making a superiority claim.

## Fixed ensembles

Once member recipes have been fixed and evaluated, combine their saved
probabilities without additional GPU inference.

```bash
python scripts/ensemble.py --runs \
  "$FIRERISK_HOME/runs/siglip2-base-full-seed42" \
  "$FIRERISK_HOME/runs/siglip2-base-lora-seed42" \
  --output "$FIRERISK_HOME/runs/siglip2-ensemble-seed42" --plots
```

Members must have the same source observations, labels and split hash. The
ensemble uses equal fixed weights and fits its temperature on validation.
Do not choose members or weights using test scores. Across-seed ensemble
comparisons require disjoint sets of member checkpoints. Changing a bootstrap
seed does not create an independent training replication.

## Artifacts and interpretation

Each run records configuration, environment, preprocessing, data manifest, training history, best and resumable checkpoints, prediction arrays and final metrics. A prediction NPZ contains original row indices, labels, logits and raw probabilities. Its logits and the saved validation-fitted temperature allow reproduction of calibrated probabilities. Artifact paths are machine-specific; publish portable configurations and result tables, and describe how to reproduce the source-row indices from the pinned input.

The target is agreement with the dataset's WHP-derived hazard classes. It is not an observed future-fire label, an active-fire detector or a calibrated probability of wildfire occurrence. Generalization beyond the sampled imagery, time period and geographic setting requires additional validation. Pretraining overlap with aerial images cannot be ruled out from the model cards alone and should be treated as unknown.

Use [references.bib](references.bib) for the dataset, SigLIP2, calibration and adapter citations. Fill [model_card_template.md](model_card_template.md) from actual artifacts before publishing weights; do not replace missing experimental evidence with anticipated results.
