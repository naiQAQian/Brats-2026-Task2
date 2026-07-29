"""Custom nnU-Net v2 trainers for the Week-6 CC/ED improvement plan.

These trainers implement the three lightweight modifications to nnU-Net 3D
full-resolution proposed in ``docs/nnunet_3d_fullres_idea_plan.md`` and walked
through in ``notebooks/week6/week6_cc_ed_improvement_plan.ipynb``:

    nnUNetTrainerScaleAwareDS   Idea 1 - scale-aware deep supervision (CC)
    nnUNetTrainerEDBoundary     Idea 2 - ED boundary auxiliary head (ED)
    nnUNetTrainerHierarchical   Idea 3 - hierarchical logits (CC + ED)

Discovery
---------
nnU-Net v2 (2.8) finds a trainer by class name. This package lives *outside* the
``nnunetv2`` install on purpose (so the code stays version-controlled in the
repo). Point nnU-Net at it with the ``nnUNet_extTrainer`` environment variable::

    export nnUNet_extTrainer=/workspace/code/brats-student-project/nnunet_trainers

Then select a trainer with ``-tr``, e.g.::

    nnUNetv2_train 502 3d_fullres all -p nnUNetPlansMask -tr nnUNetTrainerScaleAwareDS

See ``nnunet_trainers/README.md`` for the full per-experiment workflow.
"""

from nnunet_trainers.scale_aware_ds import nnUNetTrainerScaleAwareDS
from nnunet_trainers.ed_boundary_head import nnUNetTrainerEDBoundary
from nnunet_trainers.hierarchical_logits import nnUNetTrainerHierarchical

__all__ = [
    "nnUNetTrainerScaleAwareDS",
    "nnUNetTrainerEDBoundary",
    "nnUNetTrainerHierarchical",
]
