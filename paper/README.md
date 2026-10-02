# Manuscript

[main.tex](main.tex) contains a five-page English manuscript. It describes the
completed frozen encoder probe and full SigLIP2 adaptation runs as preliminary
validation experiments. No test scores or uncompleted model comparisons are
reported. The prose avoids dashes, semicolons and colons. Bibliographic titles,
URLs, mathematical signs and command syntax retain necessary characters.

The compiled [PDF](main.pdf) has five pages including references, two result
tables and one comparison figure. The
[source archive](firerisk-arxiv-source.tar.gz) contains `main.tex`,
`references.bib`, the resolved `main.bbl` and the required figure PDF.
The accompanying code is at
[yunusserhat/firerisk](https://github.com/yunusserhat/firerisk).

The author is Yunus Serhat Bıçakçı, Department of Artificial Intelligence and
Machine Learning, Faculty of Applied Sciences, Marmara University, Istanbul,
Türkiye. The byline links [ORCID 0000-0002-7288-9959](https://orcid.org/0000-0002-7288-9959).
These details were supplied by the author. The manuscript is a draft for
review before submission. The paper has not been
submitted to arXiv. The code license does not change the upstream dataset or
pretrained model terms.

## Build

Install a TeX distribution with `latexmk` and BibTeX, then run these commands
from this directory.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The output is `main.pdf`. `main.bbl` contains the resolved bibliography.
For an arXiv source submission, include `main.tex`, `references.bib` and
`main.bbl` and `figures/validation-confusion-comparison.pdf` in a source archive
after review. The source does not require custom document classes, remote
resources or shell escape.

## References and analysis

The related work cites the author's ATTransUNet article for aerial segmentation
and the street imagery geolocalization preprint for SigLIP retrieval.
[citation_audit.md](citation_audit.md) records checked primary sources,
publication status and bibliographic metadata. These are context citations,
not evidence of FireRisk performance.

[figures/analysis.json](figures/analysis.json) contains the confusion counts,
class metrics and grouped error analysis reconstructed from both saved
validation prediction files. The script verifies agreement with the original
summaries, split identity and class supports before writing the artifacts.
To regenerate them from completed runs, execute from the repository root.

```bash
python scripts/manuscript_results.py \
  --frozen-run "$FIRERISK_HOME/runs/siglip2-base-linear-seed42" \
  --full-run "$FIRERISK_HOME/runs/siglip2-base-full-seed42" \
  --output paper/figures
```

This reads existing validation predictions and does not train or evaluate a
checkpoint. Both recipes have the same 7,598 true hazard examples. Exact hazard
category accuracy includes predictions of water or nonburnable as errors.

## Provenance

The draft uses the saved summaries, histories, preprocessing and data
manifests of the two completed seed 42 runs. The reported classification
metrics are uncalibrated validation metrics. Runtime and peak memory come
from the training summaries. Times include the training loop, validation and
checkpoint writing. Peak memory is PyTorch allocated GPU memory, not total
device memory.

The runs executed concurrently on separate RTX 5090 GPUs. Their wall times
are contextual measurements rather than a controlled speed comparison.
The initial environment records an unavailable source commit and a modified
development tree. Future experiments should use a committed release and
record its revision.

The frozen encoder probe updates LayerNorm and a linear head. It is not a
strictly linear function of the raw encoder features. Both runs select epoch
7 and stop after epoch 14. Their configured schedule lengths, head learning
rates and dropout differ. The paper therefore avoids interpreting the
comparison as a controlled ablation of encoder adaptation alone.

The protocol identifiers are preserved below for exact reconstruction.

| Artifact | Identifier |
| --- | --- |
| Dataset | `blanchon/FireRisk` |
| Dataset revision | `234b2e7fe6be2da773472e83bd4d42cc9815a630` |
| Model | `timm/vit_base_patch16_siglip_224.v2_webli` |
| Model revision | `4c3661e5ac879a276ddc5ddc6d3f0ecc78fd5d82` |
| Split seed | `2026` |
| Split hash | `0ab5619091f80f73f8229634a38194ad31eeddf9dfe70978d1b664fe8fb598cd` |
| Protocol hash | `80a789b6b355c19780466eabf280e65534f86d030a21758f2505e1d31aa65251` |
| Training seed | `42` |

The data audit contains 70,331 unique pixel groups and zero exact duplicate
groups. Realized split sizes are 49,231 training, 10,552 validation and 10,548
test images. No test predictions have been used to write this draft.

References were checked against the original
[FireRisk paper](https://arxiv.org/abs/2303.07035), the original
[SigLIP2 paper](https://arxiv.org/abs/2502.14786) and the
[encoder model card](https://huggingface.co/timm/vit_base_patch16_siglip_224.v2_webli).
