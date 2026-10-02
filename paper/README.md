# Manuscript

[main.tex](main.tex) contains a short English manuscript. It describes the
completed frozen encoder probe and full SigLIP2 adaptation runs as preliminary
validation experiments. No test scores or uncompleted model comparisons are
reported. The prose avoids dashes, semicolons and colons. Bibliographic titles,
URLs, mathematical signs and command syntax retain necessary characters.

The compiled [PDF](main.pdf) has three pages. The
[source archive](firerisk-arxiv-source.tar.gz) contains only `main.tex`,
`references.bib` and the resolved `main.bbl`, ready for the author review.
The accompanying code is at
[yunusserhat/firerisk](https://github.com/yunusserhat/firerisk).

The manuscript is a draft for author review. Confirm the author list and
affiliations before submission. The paper has not been
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
`main.bbl` in a source archive after the author review. The source does not
require custom document classes, external figures or shell escape.

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
