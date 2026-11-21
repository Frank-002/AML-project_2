"""Primo stain su Unicorn. Dalla radice del repository: python -m splat_editing.generate"""

from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.dataset_readers import readCamerasFromTransforms
from gaussian_splatting.utils.camera_utils import loadCam
from pose_estimation import checkpoint_path, load_render_settings
from project_paths import project_path
from .edits import apply_stain, select_patch


@torch.no_grad()
def generate_stain(model_dir, data_dir, output_dir, *, iteration, center, radius, color, views):
    dataset, pipeline, background = load_render_settings(str(model_dir))
    clean = GaussianModel(dataset.sh_degree)
    clean.load_ply(checkpoint_path(model_dir, iteration))
    cameras = readCamerasFromTransforms(str(data_dir.resolve()), "transforms_train.json", dataset.white_background)

    selected = select_patch(clean, center, radius)
    stained = apply_stain(clean, selected, color)
    print(f"Gaussiane selezionate: {selected.sum().item()} su {len(selected)}")

    # Tutte le gaussiane partecipano al rendering, anche quelle davanti alla macchia.
    mask_colors = torch.zeros_like(clean.get_xyz)
    mask_colors[selected] = 1
    black = torch.zeros_like(background)
    for index in views:
        camera = loadCam(dataset, index, cameras[index], 1.0)
        reference = render(camera, clean, pipeline, background)["render"]
        defect = render(camera, stained, pipeline, background)["render"]
        coverage = render(camera, stained, pipeline, black, override_color=mask_colors)["render"][0]
        mask = coverage >= 0.5

        view_dir = output_dir / camera.image_name
        view_dir.mkdir(parents=True, exist_ok=True)
        to_pil_image(reference.clamp(0, 1).cpu()).save(view_dir / "clean.png")
        to_pil_image(defect.clamp(0, 1).cpu()).save(view_dir / "stain.png")
        to_pil_image(mask.byte().mul(255).cpu()).save(view_dir / "mask.png")
        print(f"{view_dir}: {mask.sum().item()} pixel nella maschera")


if __name__ == "__main__":
    root = Path(project_path("MAD-Sim_3dgs"))
    generate_stain(
        model_dir=root / "results_baseline_windows_0" / "02Unicorn",
        data_dir=root / "02Unicorn",
        output_dir=root / "synthetic" / "02Unicorn" / "stain_001",
        iteration=30000,
        center=(0.26, -0.17, -0.21),  # punto sul fianco, nel sistema 3D di Unicorn
        radius=0.09,
        color=(0.08, 0.25, 0.8),
        views=(100, 98, 92),  # laterale, obliqua, lato opposto
    )
