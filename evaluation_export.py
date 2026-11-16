"""Persist evaluation scores and visual diagnostics without changing predictions."""

import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import imsave
import numpy as np
from PIL import Image


def _rgb_image(tensor):
    """Convert an unnormalized CHW image to an RGB PNG representation."""
    array = tensor.detach().cpu().numpy()
    if array.ndim != 3 or array.shape[0] != 3 or not np.isfinite(array).all():
        raise ValueError("Expected a finite RGB tensor with shape (3, H, W)")
    array = np.moveaxis(array, 0, -1)
    return Image.fromarray((np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8))


def _comparison(path, original, reference, score_map, mask, title):
    figure = Figure(figsize=(12, 3.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 4)
    try:
        axes[0].imshow(original)
        axes[0].set_title("Input")
        axes[1].imshow(reference)
        axes[1].set_title("Rendered reference")
        heatmap = axes[2].imshow(score_map, cmap="magma", vmin=0, vmax=1)
        axes[2].set_title("Anomaly score")
        axes[3].imshow(mask > 0, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Ground truth")
        for axis in axes:
            axis.axis("off")
        figure.colorbar(heatmap, ax=axes[2], shrink=0.75)
        figure.suptitle(title, fontsize=10)
        figure.savefig(path, dpi=120)
    finally:
        figure.clear()


def save_evaluation(
    result_dir, *, config, image_paths, test_images, reference_images,
    labels, masks, scores, pose_times_ms, metrics, normalization,
):
    """Export one completed evaluation into a new directory under result_dir.

    scores and masks are the exact arrays used for metrics. RGB PNGs are for
    visual inspection; score arrays retain their original dtype and precision.
    This is an export, not a cache for resuming pose estimation.
    """
    scores = np.asarray(scores)
    masks = np.asarray(masks)
    labels = np.asarray(labels)
    count = len(image_paths)
    if (
        count == 0 or scores.ndim != 3 or scores.shape[0] != count
        or masks.shape != scores.shape or labels.shape != (count,)
        or any(len(items) != count for items in (test_images, reference_images, pose_times_ms))
    ):
        raise ValueError("Evaluation arrays, paths, labels and timings must have matching lengths/shapes")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Expected finite, globally normalized anomaly scores in [0, 1]")

    created_at = datetime.now(timezone.utc)
    output_dir = Path(result_dir) / ("evaluation_" + created_at.strftime("%Y%m%dT%H%M%S_%fZ"))
    output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = output_dir / "images"
    image_dir.mkdir()
    rows = []

    for index, source in enumerate(image_paths):
        source = Path(source)
        item_name = f"{index:04d}_{source.parent.name}_{source.stem}"
        item_dir = image_dir / item_name
        item_dir.mkdir()
        original = _rgb_image(test_images[index])
        reference = _rgb_image(reference_images[index])
        score_map = scores[index]
        mask = masks[index]
        image_score = float(score_map.max())
        dataset_label = int(labels[index] != 0)
        mask_label = int(np.any(mask > 0))

        original.save(item_dir / "input.png")
        reference.save(item_dir / "reference.png")
        np.save(item_dir / "anomaly_map.npy", score_map, allow_pickle=False)
        imsave(item_dir / "heatmap.png", score_map, cmap="magma", vmin=0, vmax=1)
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(item_dir / "mask.png")
        _comparison(
            item_dir / "comparison.png", original, reference, score_map, mask,
            f"{source.parent.name}/{source.name} | "
            f"dataset label: {dataset_label} | max score: {image_score:.6f}",
        )

        rows.append({
            "index": index,
            "image_path": str(source.resolve()),
            "category": source.parent.name,
            "dataset_class_id": int(labels[index]),
            "is_anomaly_dataset": dataset_label,
            "is_anomaly_mask": mask_label,
            "label_disagreement": int(dataset_label != mask_label),
            "positive_mask_pixels": int(np.count_nonzero(mask)),
            "image_score": image_score,
            "pose_refinement_ms": float(pose_times_ms[index]),
            "artifacts_dir": item_dir.relative_to(output_dir).as_posix(),
        })

    with (output_dir / "scores.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": created_at.isoformat(),
        "model_directory": str(Path(result_dir).resolve()),
        "config": dict(config),
        "metrics": {name: float(value) for name, value in metrics.items()},
        "normalization": {
            "method": "global min-max over all score maps in this evaluation",
            "raw_min": float(normalization["raw_min"]),
            "raw_max": float(normalization["raw_max"]),
        },
        "sample_count": count,
        "normal_images": sum(row["is_anomaly_dataset"] == 0 for row in rows),
        "anomalous_images": sum(row["is_anomaly_dataset"] != 0 for row in rows),
        "label_disagreements": sum(row["label_disagreement"] for row in rows),
        "score_map_shape": list(scores.shape[1:]),
        "score_map_dtype": str(scores.dtype),
        "definitions": {
            "image_score": "maximum of the globally normalized, smoothed anomaly map; not a probability",
            "image_auroc": "image labels taken from dataset folders",
            "image_auroc_from_masks": "image labels derived from nonempty evaluation masks",
            "pose_refinement_ms": "existing CUDA timer; excludes coarse LoFTR retrieval",
            "heatmap": "magma with common limits [0, 1] across this evaluation; no per-image rescaling",
            "rgb_pngs": "unnormalized input/reference tensors clipped to [0, 1] and quantized to 8-bit for inspection",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            **{name: version(name) for name in ("torch", "torchvision", "scipy", "Pillow")},
        },
        "files": {"scores": "scores.csv", "images": "images"},
    }
    # The summary is written last so an interrupted export is not marked complete.
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return output_dir
