"""Modifiche locali a uno splat addestrato."""

from copy import deepcopy
import math

import torch

from gaussian_splatting.utils.sh_utils import RGB2SH


@torch.no_grad()
def select_patch(splat, center, radius):
    """Seleziona le gaussiane con il centro entro il raggio dato, in unità della scena."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Il raggio deve essere positivo e finito.")
    xyz = splat.get_xyz
    center = xyz.new_tensor(center)
    if center.shape != (3,) or not center.isfinite().all():
        raise ValueError("Il centro deve contenere tre coordinate finite.")
    selected = torch.linalg.vector_norm(xyz - center, dim=1) <= radius
    if not selected.any():
        raise ValueError("La zona scelta non contiene centri di gaussiane.")
    return selected


@torch.no_grad()
def apply_stain(splat, selected, color):
    """Restituisce una copia dello splat con la zona selezionata colorata in RGB [0, 1]."""
    rgb = splat.get_xyz.new_tensor(color)
    if rgb.shape != (3,) or not ((rgb >= 0) & (rgb <= 1)).all():
        raise ValueError("Il colore deve avere tre componenti RGB fra 0 e 1.")
    stained = deepcopy(splat)
    stained._features_dc[selected, 0, :] = RGB2SH(rgb)
    # Un colore uniforme è indipendente dalla direzione di osservazione.
    # Per gli splat della baseline (grado 0) questo tensore è già vuoto.
    stained._features_rest[selected] = 0
    return stained
