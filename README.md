# Custom nnU-Net Trainers for CC/ED Segmentation

This repository provides the source code for three lightweight modifications to the nnU-Net v2 `3d_fullres` trainer. The methods were developed to improve segmentation of small or challenging tumor subregions, particularly the **cystic component (CC)** and **edema (ED)**.

This is a **method-code release only**. It is intended to document the implementation of the proposed trainer modifications and support code inspection or adaptation.

It does not include the datasets, preprocessing pipeline, trained model weights, experiment configurations, prediction scripts, or complete reproduction environment used in the associated study.

## Included methods

| Trainer class               | Target    | Main modification                                    |
| --------------------------- | --------- | ---------------------------------------------------- |
| `nnUNetTrainerScaleAwareDS` | CC        | Scale-dependent class weighting for deep supervision |
| `nnUNetTrainerEDBoundary`   | ED        | Training-only auxiliary boundary prediction head     |
| `nnUNetTrainerHierarchical` | CC and ED | Hierarchical composition of segmentation logits      |

## Repository structure

```text
.
├── README.md
├── requirements.txt
└── nnunet_trainers/
    ├── __init__.py
    ├── scale_aware_ds.py
    ├── ed_boundary_head.py
    └── hierarchical_logits.py
```

## Method 1: Scale-aware deep supervision

`nnUNetTrainerScaleAwareDS` modifies the loss applied at each deep-supervision scale.

Small CC regions may disappear when the target segmentation is downsampled for coarse decoder outputs. Supervising CC equally at every scale can therefore introduce misleading negative supervision.

The trainer applies the following class-weight schedule:

| Output scale               | CC weight | ED weight |
| -------------------------- | --------: | --------: |
| Full resolution            |       1.0 |       1.0 |
| 1/2 resolution             |       0.7 |       1.0 |
| 1/4 resolution             |       0.2 |       0.8 |
| 1/8 resolution and coarser |       0.0 |       0.5 |

The modification affects the training loss only. The underlying nnU-Net network architecture is unchanged.

The weighting schedule can be adjusted in:

```python
_SCALE_TABLE
```

## Method 2: ED boundary auxiliary head

`nnUNetTrainerEDBoundary` adds a training-only auxiliary head to the full-resolution decoder feature map.

The auxiliary head predicts a binary ED boundary band defined approximately as:

```text
dilate(ED) - erode(ED)
```

A binary cross-entropy boundary loss is added to the standard nnU-Net segmentation loss:

```text
total loss = segmentation loss + boundary_weight × boundary loss
```

The main configurable attributes are:

```python
boundary_weight = 0.3
boundary_iterations = 2
```

The auxiliary output is used only during training and is not returned during evaluation.

## Method 3: Hierarchical logits

`nnUNetTrainerHierarchical` replaces each standard decoder segmentation layer with a hierarchical segmentation head.

The method models the BraTS-PED label structure as:

```text
Whole tumor
├── Edema
└── Tumor core
    ├── Enhancing tumor
    ├── Non-enhancing tumor
    └── Cystic component
```

The head predicts:

```text
p_wt = probability of whole tumor
p_tc = probability of tumor core within whole tumor
q    = conditional probabilities of ET, NET, and CC within tumor core
```

The final class probabilities are composed as:

```text
P(background) = 1 - p_wt
P(ED)         = p_wt × (1 - p_tc)
P(ET)         = p_wt × p_tc × q_ET
P(NET)        = p_wt × p_tc × q_NET
P(CC)         = p_wt × p_tc × q_CC
```

This reduces direct competition between rare tumor-core subclasses and the background.

The implementation assumes the following label layout:

```text
background = 0
ET         = 1
NET        = 2
CC         = 3
ED         = 4
```

The hierarchical trainer therefore requires five output classes in this order.

## Software requirements

The code was developed for:

```text
Python 3
PyTorch
nnU-Net v2
```

The exact non-PyTorch package versions used during development are listed in `requirements.txt`.

PyTorch installation depends on the operating system and CUDA environment and is therefore not pinned in this repository.

## Using the trainer classes

The trainer classes are exported from the `nnunet_trainers` package:

```python
from nnunet_trainers import (
    nnUNetTrainerScaleAwareDS,
    nnUNetTrainerEDBoundary,
    nnUNetTrainerHierarchical,
)
```

They are designed as extensions of the nnU-Net v2 trainer API.

Users who wish to integrate them into an nnU-Net project are responsible for configuring trainer discovery, dataset metadata, preprocessing, training plans, and execution commands for their own environment.

Compatibility may depend on the installed nnU-Net version.

## Scope of this release

Included:

* Source code for the three custom trainer implementations
* Comments describing the method logic
* Dependency information
* Import definitions for the trainer package

Not included:

* Medical imaging data
* Processed or derived datasets
* Patient information
* Model checkpoints
* Trained weights
* nnU-Net plans or dataset metadata
* Training or validation outputs
* Prediction and post-processing scripts
* Docker images
* Hardware-specific environment configuration
* Exact experiment reproduction instructions

The code is released as a reference implementation of the proposed methods rather than as a complete reproduction package.

## Data and privacy

No medical images, annotations, patient metadata, or controlled-access data are included in this repository.

The trainer implementations operate through the standard nnU-Net data pipeline and do not independently distribute or retrieve data.

Users are responsible for ensuring that any data used with this code are handled in accordance with applicable licenses, institutional policies, ethics approvals, and privacy requirements.

## Limitations

* The implementations depend on internal nnU-Net v2 trainer and decoder interfaces.
* Changes to nnU-Net may require corresponding updates to the code.
* The hierarchical trainer is specific to the five-class label ordering described above.
* The repository does not provide a guaranteed end-to-end executable pipeline.
* Performance may differ across datasets, preprocessing settings, and nnU-Net versions.

## Citation

Citation information will be added after the associated manuscript or technical report becomes publicly available.

## License

See the `LICENSE` file for the terms governing use and redistribution of this code.

