"""Plot prepared dataset counts and deterministic training image examples.

All inputs and the output directory are explicit. The command uses cached
data only and does not train, load a model or evaluate predictions. Image
figures retain the upstream imagery terms separately from the code license.
"""

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from firerisk.data import image_fingerprint, split_content_hash  # noqa: E402

CLASS_ORDER = [
    "very_low", "low", "moderate", "high", "very_high", "non-burnable", "water"
]
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high", "Nonburnable", "Water"]
SPLIT_ORDER = ["train", "validation", "test"]


def save_figure(figure, output, basename, title):
    figure.savefig(output / f"{basename}.pdf", metadata={"Title": title, "CreationDate": None})
    figure.savefig(output / f"{basename}.png", dpi=300)
    plt.close(figure)


def plot_counts(counts, output):
    figure, axis = plt.subplots(figsize=(6.9, 3.6), layout="constrained")
    positions = np.arange(len(CLASS_ORDER))
    left = np.zeros(len(CLASS_ORDER), dtype=int)
    colors = ["#0072B2", "#E69F00", "#999999"]
    for split, color in zip(SPLIT_ORDER, colors):
        axis.barh(positions, counts[split], left=left, label=split.capitalize(), color=color)
        left += counts[split]
    for position, total in zip(positions, left):
        axis.text(total + 250, position, f"{total:,}", va="center", fontsize=8)
    axis.set_yticks(positions, labels=CLASS_LABELS)
    axis.invert_yaxis()
    axis.set_xlabel("Number of images")
    axis.set_xlim(0, max(left) * 1.15)
    axis.ticklabel_format(axis="x", style="plain")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    save_figure(figure, output, "dataset-class-distribution", "FireRisk class and split counts")


def plot_examples(images, output):
    figure, axes = plt.subplots(2, 4, figsize=(6.9, 3.9), layout="constrained")
    for axis, name, image in zip(axes.flat, CLASS_LABELS, images):
        axis.imshow(image, interpolation="nearest")
        axis.set_title(name, pad=4, fontsize=9)
        axis.set_axis_off()
    axes.flat[-1].set_axis_off()
    save_figure(figure, output, "dataset-training-examples", "Deterministic FireRisk training examples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--packed-images", type=Path, help="Override an archive path after relocation")
    parser.add_argument("--packed-offsets", type=Path, help="Override an offset path after relocation")
    parser.add_argument("--pixel-hashes", type=Path, help="Compare selected pixels with the audit cache")
    parser.add_argument("--profile-all", action="store_true", help="Read image headers for all source rows")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if set(manifest["class_names"]) != set(CLASS_ORDER):
        raise ValueError("The declared class mapping differs from FireRisk")
    if manifest.get("subset_is_smoke_only"):
        raise ValueError("Dataset publication figures require the complete source dataset")
    with np.load(args.splits, allow_pickle=False) as arrays:
        labels = arrays["labels"]
        splits = {split: arrays[split] for split in SPLIT_ORDER}
    if split_content_hash(splits, labels) != manifest["split_hash"]:
        raise ValueError("The split file does not match the manifest hash")
    permutation = [manifest["class_names"].index(name) for name in CLASS_ORDER]
    counts = {}
    for split, indices in splits.items():
        all_counts = np.bincount(labels[indices], minlength=len(CLASS_ORDER))
        if not np.array_equal(all_counts, manifest["split_class_counts"][split]):
            raise ValueError(f"The {split} class counts differ from the manifest")
        counts[split] = all_counts[permutation]
    packed = manifest["packed_images"]
    archive_path = args.packed_images or Path(packed["path"])
    offset_path = args.packed_offsets or Path(packed["offsets_path"])
    archive = np.memmap(archive_path, dtype=np.uint8, mode="r")
    offsets = np.load(offset_path, mmap_mode="r", allow_pickle=False)
    if (len(offsets) != manifest["num_rows"] + 1 or offsets[0] != 0
            or offsets[-1] != len(archive) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("The packed image offsets are inconsistent")
    audit_hashes = np.load(args.pixel_hashes, mmap_mode="r", allow_pickle=False) if args.pixel_hashes else None
    if audit_hashes is not None and len(audit_hashes) != manifest["num_rows"]:
        raise ValueError("The cached pixel hash array has a different row count")
    rng = np.random.default_rng(args.seed)
    selections = []
    images = []
    for name, label in zip(CLASS_ORDER, permutation):
        candidates = np.sort(splits["train"][labels[splits["train"]] == label])
        index = int(rng.choice(candidates))
        encoded = archive[int(offsets[index]):int(offsets[index + 1])].tobytes()
        fingerprint = image_fingerprint(encoded, "pixels")
        if audit_hashes is not None and fingerprint != str(audit_hashes[index]):
            raise ValueError(f"Selected row {index} differs from its cached pixel fingerprint")
        with Image.open(io.BytesIO(encoded)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            images.append(image.copy())
            selections.append({
                "class": name, "source_row_index": index, "split": "train",
                "width": image.width, "height": image.height,
                "mode": image.mode, "pixel_sha256": fingerprint,
                "audit_cache_verified": audit_hashes is not None,
            })
    profile = None
    if args.profile_all:
        dimensions = Counter()
        modes = Counter()
        formats = Counter()
        for index in range(manifest["num_rows"]):
            encoded = archive[int(offsets[index]):int(offsets[index + 1])].tobytes()
            with Image.open(io.BytesIO(encoded)) as image:
                width, height = image.size
                if image.getexif().get(274) in {5, 6, 7, 8}:
                    width, height = height, width
                dimensions[f"{width}x{height}"] += 1
                modes[image.mode] += 1
                formats[image.format] += 1
        profile = {
            "rows_checked": manifest["num_rows"],
            "method": "Pillow image headers and EXIF orientation, without model predictions",
            "dimensions": dict(sorted(dimensions.items())),
            "source_modes": dict(sorted(modes.items())),
            "source_formats": dict(sorted(formats.items())),
        }
    provenance = {
        "schema_version": 1,
        "dataset": manifest["repo_id"],
        "dataset_revision": manifest["resolved_revision"],
        "split_hash": manifest["split_hash"],
        "split_seed": manifest["seed"],
        "class_order": CLASS_ORDER,
        "counts": {name: values.tolist() for name, values in counts.items()},
        "totals": np.sum(list(counts.values()), axis=0).tolist(),
        "sample_selection": {
            "partition": "train", "seed": args.seed, "generator": "numpy.default_rng PCG64",
            "method": "One uniform choice per class from sorted source row indices, in class_order",
            "curation": "No visual screening or replacement of sampled rows",
            "rows": selections,
        },
        "image_profile": profile,
        "imagery_credit": "USDA National Agriculture Imagery Program. FireRisk by Shen, "
                          "Seneviratne, Wanyan and Kirley. Hub mirror by Julien Blanchon.",
        "imagery_terms": "USDA acquired aerial photography and digital orthophotography are "
                         "public domain. The Hub mirror compilation license remains unknown. "
                         "The repository code license does not license source imagery.",
        "public_domain_sources": [
            "https://www.fpacbc.usda.gov/geospatial-services/customer-services",
            "https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery",
        ],
        "label_interpretation": "Titles are provided dataset hazard categories derived from WHP, "
                                "not image-only descriptions or future fire observations.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    plot_counts(counts, args.output)
    plot_examples(images, args.output)
    (args.output / "dataset-figure-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved figures for {sum(map(np.sum, counts.values())):,} images to {args.output}")
    print(json.dumps({"totals": provenance["totals"], "profile": profile}, indent=2))


if __name__ == "__main__":
    main()
