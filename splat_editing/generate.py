"""Difetti di Unicorn. Per la bava sulla zampa: python -m splat_editing.generate --burr burr_001"""

import argparse
from filecmp import cmp
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torchvision.transforms.functional import to_pil_image

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.dataset_readers import readCamerasFromTransforms
from gaussian_splatting.utils.camera_utils import loadCam
from pose_estimation import checkpoint_path, load_render_settings
from project_paths import project_path
from .edits import apply_burr, apply_missing, apply_stain, select_patch


# Coordinate della superficie di Unicorn. Colore e raggio sono uguali per le due macchie.
STAIN_CENTERS = {
    "stain_001": (0.26, -0.17, -0.21),  # fianco
    "stain_002": (0.255, 0.295, 0.242),  # collo
}
VIEWS = (92, 98, 100)


def select_unicorn_horn(splat):
    """Cilindro attorno al corno, dalle gaussiane dell'attacco alla punta (coordinate di Unicorn)."""
    x, y, z = splat.get_xyz.unbind(dim=1)
    inside_radius = (x - 0.096).square() + (y - 0.271).square() <= 0.125**2
    return inside_radius & (z >= 0.65) & (z <= 1.08)


def publish_samples(staging, output):
    """Aggiunge i nuovi file e conserva quelli già presenti se identici."""
    new_files = []
    for source in sorted(staging.rglob("*")):
        if source.is_file():
            target = output / source.relative_to(staging)
            if not target.exists():
                new_files.append((source, target))
            elif not cmp(source, target, shallow=False):
                raise ValueError(f"{target} contiene già dati diversi: verifica la variante e il checkpoint.")
    for source, target in new_files:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)


def render_masks(splat, selected, cameras, pipeline, black):
    """Proietta le gaussiane del difetto in bianco mantenendo le occlusioni di tutta la scena."""
    colors = torch.zeros_like(splat.get_xyz)
    colors[selected] = 1
    masks = {}
    for idx, camera in cameras.items():
        coverage = render(camera, splat, pipeline, black, override_color=colors)["render"][0]
        masks[idx] = coverage >= 0.5
    return masks


@torch.no_grad()
def generate_defects(model_dir, data_dir, output_dir, variant_ids):
    dataset, pipeline, background = load_render_settings(str(model_dir))
    clean = GaussianModel(dataset.sh_degree)
    clean.load_ply(checkpoint_path(model_dir, 30000))
    infos = readCamerasFromTransforms(str(data_dir.resolve()), "transforms_train.json", dataset.white_background)
    cameras = {idx: loadCam(dataset, idx, infos[idx], 1.0) for idx in VIEWS}
    black = torch.zeros_like(background)

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temp:
        staging = Path(temp)
        good = staging / "train" / "good"
        good.mkdir(parents=True)
        # Il loader MAD legge le pose nello stesso ordine alfabetico delle immagini sane.
        transforms = json.loads((data_dir / "transforms_train.json").read_text(encoding="utf-8"))
        frames = []
        for idx, camera in cameras.items():
            name = f"view_{idx:03d}.png"
            reference = render(camera, clean, pipeline, background)["render"]
            to_pil_image(reference.clamp(0, 1).cpu()).save(good / name)
            frames.append({**transforms["frames"][idx], "file_path": f"./train/good/{name}"})
        transforms["frames"] = frames
        (staging / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")

        for variant_id in variant_ids:
            if variant_id == "missing_001":
                category = "Missing"
                selected = select_unicorn_horn(clean)
                # La sagoma del corno va misurata prima di rimuoverlo.
                masks = render_masks(clean, selected, cameras, pipeline, black)
                edited = apply_missing(clean, selected)
            elif variant_id == "burr_001":
                category = "Burrs"
                # Bordo esterno della zampa anteriore, nel sistema 3D di Unicorn.
                patch = select_patch(clean, (0.286, 0.466, -0.470), radius=0.015)
                donors = patch & (clean.get_opacity[:, 0] >= 0.5)
                edited = apply_burr(clean, donors, offset=(0.020, 0.053, 0.031), base_width=0.011)
                added = slice(len(clean.get_xyz), None)
                masks = render_masks(edited, added, cameras, pipeline, black)
            else:
                category = "Stains"
                selected = select_patch(clean, STAIN_CENTERS[variant_id], radius=0.09)
                masks = render_masks(clean, selected, cameras, pipeline, black)
                edited = apply_stain(clean, selected, color=(0.08, 0.25, 0.8))
            for idx, camera in cameras.items():
                stem = f"{variant_id}_view_{idx:03d}"
                defect = render(camera, edited, pipeline, background)["render"]
                mask = masks[idx]
                pixels = mask.sum().item()
                if pixels:
                    image_dir = Path("test") / category
                    mask_dir = Path("ground_truth") / category
                    opposite_dir = Path("qualitative/occluded") / category
                else:
                    image_dir = mask_dir = Path("qualitative/occluded") / category
                    opposite_dir = Path("test") / category
                if (output_dir / opposite_dir / f"{stem}.png").exists():
                    raise ValueError(f"{stem} esiste già con una diversa visibilità.")
                (staging / image_dir).mkdir(parents=True, exist_ok=True)
                (staging / mask_dir).mkdir(parents=True, exist_ok=True)
                to_pil_image(defect.clamp(0, 1).cpu()).save(staging / image_dir / f"{stem}.png")
                to_pil_image(mask.byte().mul(255).cpu()).save(staging / mask_dir / f"{stem}_mask.png")
                print(f"{image_dir / stem}: {pixels} pixel nella maschera")

        publish_samples(staging, output_dir)


def generate_stains(model_dir, data_dir, output_dir, stain_ids):
    """Genera le varianti Stains richieste con la stessa interfaccia del primo prototipo."""
    generate_defects(model_dir, data_dir, output_dir, stain_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera immagini, riferimenti sani e maschere di Unicorn.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stain", choices=STAIN_CENTERS, help="genera una macchia (default: entrambe)")
    group.add_argument("--missing", choices=["missing_001"], help="genera il corno mancante")
    group.add_argument("--burr", choices=["burr_001"], help="genera la protuberanza sulla zampa")
    args = parser.parse_args()
    root = Path(project_path("MAD-Sim_3dgs"))
    selected_id = args.stain or args.missing or args.burr
    generate_defects(
        model_dir=root / "results_baseline_windows_0" / "02Unicorn",
        data_dir=root / "02Unicorn",
        output_dir=root / "synthetic" / "02Unicorn",
        variant_ids=[selected_id] if selected_id else STAIN_CENTERS,
    )
