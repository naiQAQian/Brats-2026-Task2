# Custom nnU-Net trainers — Week-6 CC/ED improvement plan

Three lightweight, version-controlled modifications to nnU-Net v2 `3d_fullres`,
implementing `docs/nnunet_3d_fullres_idea_plan.md` and the hands-on walkthrough in
`notebooks/week6/week6_cc_ed_improvement_plan.ipynb`. The targets are the two
regions the baseline fails on: the **cystic component (CC)** and **edema (ED)**.

| Trainer (`-tr`) | Idea | Targets | What it changes | Inference impact |
|---|---|---|---|---|
| `nnUNetTrainerScaleAwareDS` | 1 — scale-aware deep supervision | CC | loss only: per-class weights per DS scale | none |
| `nnUNetTrainerEDBoundary`   | 2 — ED boundary aux head | ED | adds a training-only boundary head + BCE | none |
| `nnUNetTrainerHierarchical` | 3 — hierarchical logits | CC + ED | swaps decoder seg layers for WT→{TC,ED}→{ET,NET,CC} heads | none |

All three keep the standard nnU-Net architecture for **prediction/export**, so
`scripts/predict_task2_submission.py` and `scripts/postprocess_small_components.py`
are reused unchanged — just pass the trainer name with `--trainer`.

## How nnU-Net finds these trainers

nnU-Net v2 (2.8) resolves a `-tr` name by class name. This package lives outside
the `nnunetv2` install (so it stays in the repo); point nnU-Net at it with the
`nnUNet_extTrainer` environment variable. `scripts/setup_runpod_env.sh` sets and
persists it. In an existing shell:

```bash
source /etc/profile.d/brats_env.sh          # exports nnUNet_extTrainer
# or manually:
export nnUNet_extTrainer=/workspace/code/brats-student-project/nnunet_trainers
```

> Set this in **both** the training shell and the prediction shell — inference
> reconstructs the same trainer by name.

Verify discovery + the maths without training (CPU, no data, no GPU):

```bash
python scripts/test_custom_trainers.py
```

## Per-experiment workflow

Change **one** thing, measure it against the `Dataset502` baseline, record it in
the ablation table (notebook §6). Recommended order: Idea 1 → 2 → 3 (cheapest and
most targeted first).

```bash
source /etc/profile.d/brats_env.sh

# --- Idea 1 (start here): scale-aware deep supervision, targets CC ---
nnUNetv2_train 502 3d_fullres all -p nnUNetPlansMask -tr nnUNetTrainerScaleAwareDS \
  > outputs/train_502_scaleaware.log 2>&1 &

# --- Idea 2: ED boundary aux head, targets ED surface accuracy ---
nnUNetv2_train 502 3d_fullres all -p nnUNetPlansMask -tr nnUNetTrainerEDBoundary

# --- Idea 3: hierarchical logits, targets CC + ED ---
nnUNetv2_train 502 3d_fullres all -p nnUNetPlansMask -tr nnUNetTrainerHierarchical
```

Resume an interrupted run by appending `--c`. Watch the per-class pseudo-Dice for
CC/ED in the log — that's the early signal. **One model per GPU.**

Predict the 91 validation cases with the matching trainer (writes to a separate
per-trainer output dir, so the baseline predictions are never overwritten):

```bash
python scripts/predict_task2_submission.py --dataset-id 502 --dataset-name BraTSPEDfull \
  --plan nnUNetPlansMask --trainer nnUNetTrainerScaleAwareDS --folds all \
  --checkpoint checkpoint_final.pth --postprocess --zip
```

Then upload the zip to the validation panel and record the per-region DSC/NSD.

## What each trainer does (and its knobs)

### Idea 1 — `nnUNetTrainerScaleAwareDS` (`scale_aware_ds.py`)
Loss-only. Keeps nnU-Net's per-scale scalar weights, but multiplies in a
per-class weight at each deep-supervision scale so a tiny CC blob isn't penalised
at coarse resolutions where it has downsampled away. Weight schedule (CC, ED):

| scale | full | 1/2 | 1/4 | 1/8 (+coarser) |
|---|---|---|---|---|
| CC | 1.0 | 0.7 | 0.2 | 0.0 |
| ED | 1.0 | 1.0 | 0.8 | 0.5 |

Tune in `_SCALE_TABLE`. CE uses per-class `weight`; Dice uses a normalized
weighted class-mean (`WeightedMemoryEfficientSoftDiceLoss`).

### Idea 2 — `nnUNetTrainerEDBoundary` (`ed_boundary_head.py`)
Wraps the network (`EDBoundaryNet`) with a 1×1-conv aux head on the full-res
decoder feature, trained against an ED **boundary band** (`dilate(ED) − erode(ED)`).
`train_step` adds `boundary_weight * BCE`. Knobs (class attrs): `boundary_weight`
(default `0.3`, plan range 0.1–0.5), `boundary_iterations` (band half-width, `2`).
Aux head is disabled in `eval()` → zero inference cost.

### Idea 3 — `nnUNetTrainerHierarchical` (`hierarchical_logits.py`)
Replaces each decoder seg layer with `HierarchicalSegHead`, which predicts
`p_wt`, `p_tc` and a 3-way core softmax and composes class probabilities along the
label tree, so **CC only competes inside the tumor core**. It returns `log P`;
since the 5 probs sum to 1, `softmax(log P) == P`, so training loss and inference
both work with no pipeline changes. Deep supervision is preserved (every scale
gets a head). The composition hard-codes the BraTS-PED label layout
(`bg,ET,NET,CC,ED` = `0,1,2,3,4`) and asserts 5 classes.

## Data safety
These trainers never read or write `/workspace/Data` outside nnU-Net's normal
pipeline, and add no raw images to the repo. Training/prediction obey the same
`CLAUDE.md` rules as the baseline.
