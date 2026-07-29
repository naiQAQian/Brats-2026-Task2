"""Idea 2 - ED boundary / distance auxiliary head (ED).

See ``docs/nnunet_3d_fullres_idea_plan.md`` (Idea 2).

Edema (ED) errors are usually *boundary* errors, not detection errors: the model
finds the edema but its contour bleeds into nearby FLAIR brightness or stops
short. Plain Dice+CE is not very sensitive to boundary quality.

This trainer adds a small **auxiliary head** on the decoder's full-resolution
feature map that predicts an ED **boundary band** (``dilate(ED) - erode(ED)``)
*during training only*. The extra BCE term nudges the shared features to respect
edema edges, improving surface metrics (NSD / HD95). At inference the auxiliary
head is ignored - the segmentation output and the export pipeline are identical
to the baseline.

Implementation:
  * ``build_network_architecture`` wraps the standard nnU-Net in ``EDBoundaryNet``,
    which taps the last decoder stage with a forward hook and feeds it to a 1x1
    conv aux head (built at construction time so its params join the optimizer).
  * ``train_step`` adds ``boundary_weight * BCE(aux_logits, ED boundary band)``.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast, nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


class EDBoundaryNet(nn.Module):
    """Wraps a standard nnU-Net, adding a training-only ED boundary head.

    ``forward`` returns exactly what the wrapped network returns (a list of
    deep-supervision logits when DS is on, otherwise a single tensor), so nnU-Net's
    validation and inference code paths are unaffected. When ``self.training`` is
    True the boundary logits are stashed on ``self.aux_logits`` for ``train_step``.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        # expose encoder/decoder so nnUNetTrainer.set_deep_supervision_enabled works
        self.encoder = base.encoder
        self.decoder = base.decoder

        # the full-resolution segmentation layer tells us the conv dim + feature width
        last_seg = base.decoder.seg_layers[-1]
        conv_op = type(last_seg)          # nn.Conv2d or nn.Conv3d
        in_channels = last_seg.in_channels
        self.aux_head = conv_op(in_channels, 1, kernel_size=1)

        self.aux_logits = None
        self._feat = None
        # capture the last decoder stage's output (the full-res feature map)
        base.decoder.stages[-1].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._feat = output

    def forward(self, x):
        self._feat = None
        out = self.base(x)
        if self.training and self._feat is not None:
            self.aux_logits = self.aux_head(self._feat)
        else:
            self.aux_logits = None
        return out


class nnUNetTrainerEDBoundary(nnUNetTrainer):
    """nnU-Net trainer with an ED boundary auxiliary head. Idea 2."""

    boundary_weight = 0.3          # lambda_boundary, idea-plan suggests 0.1 - 0.5
    boundary_iterations = 2        # morphological band half-width (voxels)

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        base = nnUNetTrainer.build_network_architecture(
            plans_manager, configuration_manager, num_input_channels,
            num_output_channels, enable_deep_supervision)
        return EDBoundaryNet(base)

    def _unwrap_network(self):
        net = self.network.module if self.is_ddp else self.network
        if hasattr(net, "_orig_mod"):      # torch.compile OptimizedModule
            net = net._orig_mod
        return net

    @property
    def _ed_label(self):
        labels = {str(k).upper(): v for k, v in self.dataset_json["labels"].items()}
        return labels.get("ED")

    def _ed_boundary_band(self, seg_full: torch.Tensor) -> torch.Tensor:
        """Binary boundary band of the ED label: dilate(ED) - erode(ED).

        ``seg_full`` is the full-resolution label map, shape (b, 1, *spatial).
        Returns a float tensor of the same shape in {0, 1}.
        """
        ed = self._ed_label
        with torch.no_grad():
            mask = (seg_full == int(ed)).float()
            spatial = mask.ndim - 2
            pool = F.max_pool3d if spatial == 3 else F.max_pool2d
            dil = mask
            ero = mask
            for _ in range(self.boundary_iterations):
                dil = pool(dil, kernel_size=3, stride=1, padding=1)
                ero = -pool(-ero, kernel_size=3, stride=1, padding=1)
            band = (dil - ero).clamp_(0.0, 1.0)
        return band

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

            # --- auxiliary ED boundary loss (training only) ---
            if self._ed_label is not None and self.boundary_weight > 0:
                aux_logits = self._unwrap_network().aux_logits
                if aux_logits is not None:
                    seg_full = target[0] if isinstance(target, list) else target
                    band = self._ed_boundary_band(seg_full).to(aux_logits.dtype)
                    l = l + self.boundary_weight * F.binary_cross_entropy_with_logits(aux_logits, band)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}
