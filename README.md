![LesionTrace-Logo.png](Figures/LesionTrace-Logo.png)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.08900-red?logo=arXiv&logoColor=white)](https://)


LesionTrace is a weakly supervised change detection model for longitudinal retinal OCT that produces spatially grounded lesion and change maps as interpretable outputs, and a pair-level disease activity prediction.

---

## Overview

LesionTrace bridges the complementary limitations of segmentation-based and pair-level change detection approaches by jointly producing:

- **Per-visit lesion maps** $\mathbf{M}_t$, $\mathbf{M}_{t+1}$ — highlighting clinically relevant regions at each time point
- **Content change map** $\mathbf{E}_{\text{content}}$ — tissue variation within the persistent lesion footprint
- **Extent change map** $\mathbf{E}_{\text{extent}}$ — regions of new or resolved lesion activity
- **Pair-level change prediction** $\hat{y} \in \{\text{Reduced, Stable, Increased}\}$

![LesionTrace Overview](Figures/LesionTrace-LesionTrace.png)

---

## Method

LesionTrace comprises four learnable components:

| Component | Description |
|---|---|
| **Lesion scorer** | Lightweight MLP supervised by pseudo-lesion masks $\tilde{\mathbf{M}}_t$ via $\mathcal{L}_{\text{map}}$ |
| **Local lesion-aware pathway** | Lesion-masked cross-attention decomposing change into content $\cap$ and extent $\triangle$ |
| **Global progression-aware pathway** | CLS cross-attention producing change vector $\Delta\mathbf{g}$, supervised by $\mathcal{L}_{\text{dir}}$ |
| **Classification head** | Combines local and global representations to predict $\hat{y}$ |

### Training Strategy

LesionTrace uses a two-phase training strategy:

1. **Pseudo-lesion mask preparation** — PiPViT is trained on AMD-Mix and applied to the MARIO training set to generate per-visit pseudo-lesion masks $\tilde{\mathbf{M}}_t$
2. **Change detection training** — LesionTrace is trained end-to-end using pseudo-lesion masks and pair-level change labels

At inference, only LesionTrace is required.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python >= 3.8
- timm==0.4.12
- torch==1.13.0
- torchvision==0.14.0

---

## Installation

```bash
git clone https://github.com/marziehoghbaie/LesionTrace.git
cd LesionTrace
pip install -r requirements.txt
```

---

## Dataset Preparation

LesionTrace is trained and evaluated on the [MARIO challenge dataset](https://youvenz.github.io/MARIO_challenge.github.io/).
---


## Training LesionTrace

```bash
python [Smain.py](LesionTrace/main/Smain.py) --config_path [LesionChangeDetector.yaml](LesionTrace/config/Train/LesionChangeDetector.yaml)
```

---

## Evaluation

### Change Classification (MARIO)

```bash
python [Smain.py](LesionTrace/main/Smain.py) --config_path [LesionChangeDetector.yaml](LesionTrace/config/Test/LesionChangeDetector.yaml)

```
the code will come soon!

Reports macro F1, Kendall rank correlation (RK), specificity, and aggregate average score.

### Spatial Localisation (TREND / CSC)
Reports AUPRC and Dice coefficient for visit-level lesion maps and pair-level change maps.
---

## Results

### Change Classification on MARIO
Coming Soon!

[//]: # ()
[//]: # (| Model | F1 &#40;macro&#41; | RK corr. | Specificity | Average |)

[//]: # (|---|---|---|---|---|)

[//]: # (| Yiding Hao | **0.839** | **0.761** | **0.922** | **0.841** |)

[//]: # (| Beltrán-Carrero et al. | 0.832 | 0.758 | 0.906 | 0.832 |)

[//]: # (| **LesionTrace &#40;ours&#41;** | 0.833 | 0.754 | 0.912 | 0.833 |)

### Spatial Localisation on TREND and CSC
Coming Soon!

[//]: # ()
[//]: # (| Model | TREND AUPRC | CSC AUPRC |)

[//]: # (|---|---|---|)

[//]: # (| Best baseline | 0.045 | 0.118 |)

[//]: # (| **LesionTrace &#40;ours&#41;** | **0.144** | **0.266** |)


---


## Model weights
Coming Soon!
---

## Citation

If you find this work useful, please cite:

```bibtex
@article{LesionTrace2025,
  title     = {LesionTrace},
  author    = {Oghbaie, Marzieh and others},
  journal   = {},
  year      = {2025}
}
```

---

## Acknowledgements

This work uses [PiPViT](https://github.com/marziehoghbaie/PiPViT/tree/main) for pseudo-lesion mask generation and [RetFiner](https://github.com/ronnief1/RetFiner) for backbone initialisation. The MARIO challenge dataset is described in [Zeghlache et al. 2025](https://arxiv.org/abs/2501.xxxxx).

For the baselines, we used their repositories mentioned in [Baseline_readme.md](Baselines/Baseline_readme.md)


