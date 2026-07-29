"""Idea 3 - Single-network hierarchical logits (CC + ED).

See ``docs/nnunet_3d_fullres_idea_plan.md`` (Idea 3).

A flat softmax makes every class compete at once: a rare CC voxel must out-vote
background, ET, NET *and* ED - so it loses. But the labels nest naturally::

    WT  = { TC, ED }
    TC  = { ET, NET, CC }

This trainer keeps a single nnU-Net, but replaces each flat segmentation layer in
the decoder with a small set of **hierarchical heads** and composes the label
probabilities along the tree:

    p_wt = sigmoid(wt)                     # tumor vs background
    p_tc = sigmoid(tc)                     # core vs ED, inside WT
    q    = softmax([et, net, cc])          # core subtype, inside TC

    P(bg)  = 1 - p_wt
    P(ED)  = p_wt * (1 - p_tc)
    P(ET)  = p_wt * p_tc * q_ET
    P(NET) = p_wt * p_tc * q_NET
    P(CC)  = p_wt * p_tc * q_CC

CC therefore only competes *inside the tumor core* (through the softmax ``q``),
not against the whole image - which is exactly the competition the idea-plan wants
to remove.

The head returns ``log(P)`` in the standard class-channel order. Because the five
probabilities sum to 1, ``softmax(log P) == P``: the training loss (DC + CE, which
applies a softmax) and nnU-Net's inference/export (which also applies a softmax)
both behave correctly with **no changes to the prediction pipeline**. Deep
supervision is preserved - every decoder scale gets its own hierarchical head.

Note: this composition is specific to the BraTS-PED label convention used by
``Dataset50x`` (ET=1, NET=2, CC=3, ED=4); ``build_network_architecture`` asserts
the 5-class layout.
"""
import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# BraTS-PED label integers (must match Dataset50x dataset.json)
LBL_BG, LBL_ET, LBL_NET, LBL_CC, LBL_ED = 0, 1, 2, 3, 4
NUM_CLASSES = 5
_EPS = 1e-6


class HierarchicalSegHead(nn.Module):
    """Drop-in replacement for a decoder ``seg_layer``: feature -> log-prob logits.

    Maps a decoder feature map (``in_channels``) to ``NUM_CLASSES`` channels of
    log-probabilities composed along the WT -> {TC, ED} -> {ET, NET, CC} tree.
    Accepts the same call signature as the conv it replaces, so the unchanged
    nnU-Net decoder forward (and deep supervision) just works.
    """

    def __init__(self, in_channels: int, conv_op):
        super().__init__()
        self.wt = conv_op(in_channels, 1, kernel_size=1)   # tumor vs background
        self.tc = conv_op(in_channels, 1, kernel_size=1)   # core vs ED (within WT)
        self.core = conv_op(in_channels, 3, kernel_size=1)  # ET / NET / CC (within TC)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # compute probabilities in float32 for numerical stability under autocast
        wt = self.wt(feat).float()
        tc = self.tc(feat).float()
        core = self.core(feat).float()

        p_wt = torch.sigmoid(wt)                  # (b,1,...)
        p_tc = torch.sigmoid(tc)                  # (b,1,...)
        q = torch.softmax(core, dim=1)            # (b,3,...)

        p_core = p_wt * p_tc                       # voxel in tumor core
        probs = [None] * NUM_CLASSES
        probs[LBL_BG] = 1.0 - p_wt
        probs[LBL_ED] = p_wt * (1.0 - p_tc)
        probs[LBL_ET] = p_core * q[:, 0:1]
        probs[LBL_NET] = p_core * q[:, 1:2]
        probs[LBL_CC] = p_core * q[:, 2:3]

        out = torch.cat(probs, dim=1)              # (b, 5, ...), sums to 1 over dim 1
        # return log-probs: softmax(log P) == P, so downstream loss/inference are exact
        return torch.log(out.clamp_min(_EPS))


class nnUNetTrainerHierarchical(nnUNetTrainer):
    """nnU-Net trainer with hierarchical composition heads. Idea 3."""

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        assert num_output_channels == NUM_CLASSES, (
            f"nnUNetTrainerHierarchical expects {NUM_CLASSES} classes "
            f"(BraTS-PED: bg, ET, NET, CC, ED), got {num_output_channels}."
        )
        net = nnUNetTrainer.build_network_architecture(
            plans_manager, configuration_manager, num_input_channels,
            num_output_channels, enable_deep_supervision)

        # replace every (deep-supervision) seg layer with a hierarchical head
        seg_layers = net.decoder.seg_layers
        for i in range(len(seg_layers)):
            conv = seg_layers[i]
            seg_layers[i] = HierarchicalSegHead(conv.in_channels, type(conv))
        return net
