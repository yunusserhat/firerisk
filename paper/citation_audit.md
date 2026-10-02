# Verification of relevant author publications

Checked on 2 October 2026. The author supplied the ORCID record and the current Marmara University affiliation. The public ORCID API was read at [the works endpoint](https://pub.orcid.org/v3.0/0000-0002-7288-9959/works). The two publications below have a direct methodological or imagery connection to this manuscript.

## ATTransUNet

Yunus Serhat Bıçakçı and Beytullah Sarıca. *ATTransUNet: Semantic Segmentation Model for Building Segmentation from Aerial Image and Laser Data*. Nordic Machine Intelligence, 2(3), 7–9, 2023. DOI [10.5617/nmi.10039](https://doi.org/10.5617/nmi.10039).

The [publisher article page](https://journals.uio.no/NMI/article/view/10039), [publisher PDF](https://journals.uio.no/NMI/article/download/10039/8476), [Crossref record](https://api.crossref.org/works/10.5617/nmi.10039), and [Marmara institutional record](https://avesis.marmara.edu.tr/yayin/66befbeb-1f19-4e00-9849-f1571f3ac482/attransunet-semantic-segmentation-model-for-building-segmentation-from-aerial-image-and-laser-data) were checked. Crossref and ORCID give 27 March 2023. The PDF header and publisher HTML contain an inconsistent 2022 date. The PDF volume heading, copyright, and Crossref agree on 2023, which is used here.

Suitable supporting sentence

> Bıçakçı and Sarıca combined attention gates and a Transformer within a U-Net for building segmentation from aerial imagery and laser data. Their work concerns dense building masks, whereas the present study assigns a single fire risk category to an RGB image.

This citation supports the aerial imagery and representation context. It does not establish a FireRisk result or a comparable accuracy.

```bibtex
@article{bicakci2023attransunet,
  title = {{ATTransUNet}: Semantic Segmentation Model for Building Segmentation from Aerial Image and Laser Data},
  author = {B{\i}{\c c}ak{\c c}{\i}, Yunus Serhat and Sar{\i}ca, Beytullah},
  journal = {Nordic Machine Intelligence},
  volume = {2},
  number = {3},
  pages = {7--9},
  year = {2023},
  doi = {10.5617/nmi.10039},
  url = {https://journals.uio.no/NMI/article/view/10039}
}
```

## SigLIP and street imagery

Yunus Serhat Bıçakçı, Joseph Shingleton, and Anahid Basiri. *Street-Level Geolocalization Using Multimodal Large Language Models and Retrieval-Augmented Generation*. arXiv preprint 2509.01341, 2025. DOI [10.48550/arXiv.2509.01341](https://doi.org/10.48550/arXiv.2509.01341).

The [arXiv metadata](https://arxiv.org/abs/2509.01341), [full paper](https://arxiv.org/html/2509.01341v1), and [University of Glasgow repository](https://eprints.gla.ac.uk/367593/) were checked. Submitted 1 September 2025. The Glasgow record describes it as a nonrefereed preprint. Its downloadable PDF is called a published version in the repository, which does not imply journal publication.

Suitable supporting sentence

> Bıçakçı and colleagues used SigLIP image embeddings to retrieve gallery geolocations for multimodal language model prompts. Their approach retained the pretrained encoder, while the present study examines both a frozen encoder and supervised adaptation on aerial images.

The gallery uses EMP-16 and OSV-5M. Retrieved similar and dissimilar image locations are included in the prompt. This is a useful encoder connection, but no geolocalization accuracy should be represented as evidence for FireRisk performance.

```bibtex
@misc{bicakci2025geolocalization,
  title = {Street-Level Geolocalization Using Multimodal Large Language Models and Retrieval-Augmented Generation},
  author = {B{\i}{\c c}ak{\c c}{\i}, Yunus Serhat and Shingleton, Joseph and Basiri, Anahid},
  year = {2025},
  eprint = {2509.01341},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi = {10.48550/arXiv.2509.01341},
  url = {https://arxiv.org/abs/2509.01341}
}
```

## Optional task specific benchmarking citation

*Performance Comparison of Multimodal Vision-Language Models in Classifying Turkish Dishes* by Yunus Serhat Bıçakçı is now a published journal article, rather than an in review manuscript. The [publisher page](https://dergipark.org.tr/en/pub/saucis/article/1727583) and [Crossref record](https://api.crossref.org/works/10.35377/saucis...1727583) confirm publication on 16 March 2026 in Sakarya University Journal of Computer and Information Sciences, 9(1), 119–133.

The DOI `10.35377/saucis...1727583` contains three literal dots. This is the registered DOI. The shorter one dot variation appearing in some automated citation formats returns a Crossref 404 response and should not replace it.

This study compares seven multimodal models without additional training on two Turkish food image datasets and reports macro and weighted classification metrics alongside inference time. It can support a short discussion of task specific benchmarking. The two geospatial works above provide a closer connection and are sufficient for the short FireRisk manuscript. No citation is recommended merely to increase the number of self citations.
