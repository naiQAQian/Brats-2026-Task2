"""Idea 1 - Scale-aware deep supervision for small subregions (CC).

See ``docs/nnunet_3d_fullres_idea_plan.md`` (Idea 1).

nnU-Net supervises the decoder at several resolutions ("deep supervision"). To
build a low-resolution target the ground-truth label map is *downsampled*. A tiny
cystic-component (CC) blob can shrink to nothing at 1/4 or 1/8 resolution, so the
coarse heads effectively teach the model "CC is absent" - biasing it against the
very region we care about.

This trainer keeps nnU-Net's per-*scale* scalar weighting unchanged, but adds a
per-*class* weight at each scale, so CC is supervised fully at full resolution
and down-weighted (-> 0) at the coarsest scales where it has vanished anyway. ED
tapers more gently; the big regions stay fully supervised everywhere.

    output scale | CC weight | ED weight | other foreground
    -------------+-----------+-----------+-----------------
    full         |    1.0    |    1.0    |       1.0
    1/2          |    0.7    |    1.0    |       1.0
    1/4          |    0.2    |    0.8    |       1.0
    1/8 (+coarser)|   0.0    |    0.5    |       1.0

This is a *loss-only* change: the network architecture, inference, and export are
all identical to the baseline, so predictions are produced exactly as before.
"""
from typing import List

import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

# Per-scale (CC, ED) weights from the idea-plan table. Index 0 == full resolution.
# Scales beyond the table reuse the last (coarsest) row.
_SCALE_TABLE = [(1.0, 1.0), (0.7, 1.0), (0.2, 0.8), (0.0, 0.5)]


def _cc_ed_weights_for_scale(scale_idx: int):
    return _SCALE_TABLE[min(scale_idx, len(_SCALE_TABLE) - 1)]


class WeightedMemoryEfficientSoftDiceLoss(MemoryEfficientSoftDiceLoss):
    """MemoryEfficientSoftDiceLoss with a per-class weight on the class average.

    The parent averages the per-class soft-Dice uniformly. Here we replace that
    with a *normalized weighted* mean ``sum(w_c * dc_c) / sum(w_c)`` so a class
    can be down-weighted (or removed, ``w_c = 0``) without changing the overall
    loss scale of the remaining classes.

    ``class_weights`` is ordered like the Dice class axis: with ``do_bg=False``
    (the nnU-Net default for label training) that is the foreground labels in
    ascending order, e.g. ``[ET, NET, CC, ED]``.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is None:
            raise ValueError("class_weights must be provided")
        # registered as a buffer so `.to(device)` moves it with the loss module
        self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, x, y, loss_mask=None):
        # --- identical to MemoryEfficientSoftDiceLoss.forward up to the reduction ---
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None \
                else (y_onehot * loss_mask).sum(axes, dtype=torch.float32)

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                from nnunetv2.utilities.ddp_allgather import AllGatherGrad
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)
            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)

        dc = (2 * intersect + self.smooth) / (sum_gt + sum_pred + float(self.smooth)).clamp_min(1e-8)

        # --- the only change: normalized weighted mean over the class axis ---
        w = self.class_weights.to(dc.dtype)
        dc = (dc * w).sum(-1) / w.sum().clamp_min(1e-8)
        return -dc.mean()


class PerScaleDeepSupervisionWrapper(nn.Module):
    """Like nnU-Net's DeepSupervisionWrapper, but with a *separate* loss per scale.

    ``losses[i]`` is applied to deep-supervision output ``i`` and scaled by the
    usual scalar ``weight_factors[i]``. This lets each scale carry its own
    per-class weighting. Scales whose scalar weight is 0 are skipped (matching the
    baseline, which drops the two coarsest outputs).
    """

    def __init__(self, losses: List[nn.Module], weight_factors):
        super().__init__()
        assert any(w != 0 for w in weight_factors), "At least one weight factor should be != 0"
        self.losses = nn.ModuleList(losses)
        self.weight_factors = tuple(weight_factors)

    def forward(self, *args):
        assert all(isinstance(i, (tuple, list)) for i in args), \
            f"all args must be tuple or list, got {[type(i) for i in args]}"
        weights = self.weight_factors
        return sum(
            weights[i] * self.losses[i](*inputs)
            for i, inputs in enumerate(zip(*args))
            if weights[i] != 0.0
        )


class nnUNetTrainerScaleAwareDS(nnUNetTrainer):
    """nnU-Net trainer with scale-aware (per-class) deep supervision. Idea 1."""

    def _build_loss(self):
        assert self.enable_deep_supervision, \
            "nnUNetTrainerScaleAwareDS requires deep supervision to be enabled."
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerScaleAwareDS targets plain label training (no region-based labels)."

        labels = {str(k).upper(): v for k, v in self.dataset_json["labels"].items()}
        num_classes = self.label_manager.num_segmentation_heads  # incl. background
        # foreground labels in ascending integer order -> Dice class axis (do_bg=False)
        fg_labels = sorted(
            (name for name in self.dataset_json["labels"] if str(name).upper() != "BACKGROUND"),
            key=lambda n: int(self.dataset_json["labels"][n]),
        )

        cc_int = labels.get("CC")
        ed_int = labels.get("ED")
        if cc_int is None:
            self.print_to_log_file("[ScaleAwareDS] WARNING: no 'CC' label found; CC weighting is a no-op.")
        if ed_int is None:
            self.print_to_log_file("[ScaleAwareDS] WARNING: no 'ED' label found; ED weighting is a no-op.")

        deep_supervision_scales = self._get_deep_supervision_scales()
        # scalar per-scale weights: identical to the nnU-Net baseline
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float64)
        if self.is_ddp and not self._do_i_compile():
            weights[-1] = 1e-6
        else:
            weights[-1] = 0
        weights = weights / weights.sum()

        per_scale_losses = []
        for s in range(len(deep_supervision_scales)):
            cc_w, ed_w = _cc_ed_weights_for_scale(s)

            # CE class weights, indexed by label integer (length = num_classes incl. bg)
            ce_w = torch.ones(num_classes, dtype=torch.float32)
            if cc_int is not None:
                ce_w[int(cc_int)] = cc_w
            if ed_int is not None:
                ce_w[int(ed_int)] = ed_w

            # Dice class weights, ordered like the foreground class axis (do_bg=False)
            dice_w = []
            for name in fg_labels:
                u = str(name).upper()
                dice_w.append(cc_w if u == "CC" else ed_w if u == "ED" else 1.0)

            loss = DC_and_CE_loss(
                {"batch_dice": self.configuration_manager.batch_dice,
                 "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp},
                {"weight": ce_w},
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=lambda apply_nonlin=None, _dw=dice_w, **kw:
                    WeightedMemoryEfficientSoftDiceLoss(apply_nonlin=apply_nonlin, class_weights=_dw, **kw),
            )
            per_scale_losses.append(loss)
            self.print_to_log_file(
                f"[ScaleAwareDS] scale {s} (factor {deep_supervision_scales[s]}): "
                f"CC weight={cc_w}, ED weight={ed_w}"
            )

        wrapped = PerScaleDeepSupervisionWrapper(per_scale_losses, weights)
        # move CE weight + Dice class-weight buffers onto the training device
        wrapped = wrapped.to(self.device)
        return wrapped
