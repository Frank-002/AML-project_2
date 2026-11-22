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


@torch.no_grad()
def apply_missing(splat, selected):
    """Nasconde le gaussiane selezionate su una copia dello splat."""
    if not selected.any():
        raise ValueError("La parte da rimuovere non contiene gaussiane.")
    edited = deepcopy(splat)
    # Il modello memorizza il logit dell'opacità: sigmoid(-inf) = 0.
    edited._opacity[selected] = -torch.inf
    return edited


@torch.no_grad()
def apply_burr(splat, selected, offset, base_width):
    """Aggiunge copie delle gaussiane locali lungo una piccola sporgenza, su uno splat da renderizzare."""
    xyz = splat.get_xyz
    offset = xyz.new_tensor(offset)
    if offset.shape != (3,) or not offset.isfinite().all() or not offset.any():
        raise ValueError("La direzione e la lunghezza della bava richiedono uno spostamento finito non nullo.")
    if not math.isfinite(base_width) or base_width <= 0:
        raise ValueError("Lo spessore alla base deve essere positivo e finito.")
    indices = selected.nonzero(as_tuple=True)[0]
    if not len(indices):
        raise ValueError("La base della bava non contiene gaussiane.")

    # Le gaussiane più esterne forniscono le copie più vicine alla punta.
    indices = indices[(xyz[indices] @ offset).argsort()]
    base = xyz[indices]
    tip = base.mean(dim=0) + offset
    count = len(indices)
    t = torch.arange(1, count + 1, device=xyz.device, dtype=xyz.dtype)[:, None] / count
    copies = {name: getattr(splat, name)[indices]
              for name in ("_features_dc", "_features_rest", "_opacity", "_rotation")}
    copies["_xyz"] = torch.lerp(base, tip, t)
    # Le gaussiane della superficie possono essere quasi piatte: diamo volume
    # alle copie e ne riduciamo le scale fino a metà in punta. Il PLY usa log-scale.
    copies["_scaling"] = (splat.get_scaling[indices].clamp_min(base_width) * (1 - 0.5 * t)).log()

    edited = deepcopy(splat)
    for name, extra in copies.items():
        original = getattr(splat, name)
        values = torch.cat((original, extra), dim=0)
        setattr(edited, name, torch.nn.Parameter(values, requires_grad=original.requires_grad))
    return edited
