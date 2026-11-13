## Configuration of the baselines and LesionTrace

## Baselines

| Method                 | Backbone | Pre-trained weights | Pair-level fusion | Params (M) | FLOPs (G) | BS | Epochs | Code                                                               |
|------------------------|---|---|---|---|---|---|---|--------------------------------------------------------------------|
| Bekkers et al.         | ResNet-18 | ImageNet | Concatenation of visit embeddings | 22.36 | 0.65 | 54 | 50 | [GitHub](https://github.com/ninamalou/TONIC-MICCAI-2024/tree/main) |
| Zhang et al.           | ResNet-50 | ImageNet | Concatenation of visit embeddings | 23.52 | 8.26 | 54 | 50 | [GitHub](https://github.com/pzhangwj/mario_challenge_code)         |
| Yi Ding                | ConvNeXt-Large | ImageNet | Concatenation of visit embeddings and their difference | 198.42 | 68.74 | 54 | 50 | [GitHub](https://github.com/YIDING4869/MARIO2024)                  |
| Yiding Hao             | ConvNeXt-Large | ImageNet | Multi-head self-attention per visit, followed by concatenation | 197.05 | 68.78 | 54 | 50 | [GitHub](https://github.com/lumine-1/MARIO_Project)                |
| CheXRelFormer          | Transformer | — | Multi-scale feature difference summary | 36.57 | 15.28 | 54 | 50 | [GitHub](https://github.com/PLAN-Lab/CheXRelFormer)                |
| Early fusion           | ViT-Large | RetFound | Channel-wise input concatenation | 303.10 | 59.63 | 54 | 50 | —                                                                  |
| Beltrán-Carrero et al. | ViT-Large | RetFound | Bidirectional cross-attention between [CLS] tokens, mean pooled | 307.30 | 119.38 | 54 | 50 | [GitHub](https://github.com/BIT-UPM/mario_miccai_24_step_amd)      |
| Budimir et al.         | ViT-Large | RetFound | Shared transformer with cross-visit patch attention; [CLS] token for classification | 312.56 | 122.15 | 44 | 50 | [GitHub](https://github.com/LovreAB17/FERLIV-MARIO)                |
| Huber et al.           | ViT-Base | DINOv2 | Concatenation of [CLS] tokens | 87.09 | 44.61 | 54 | 50 | [GitHub](https://github.com/marceljhuber/mario-miccai2024)         |
| SiamRETFound           | ViT-Large | RetFound | Concatenation of [CLS] tokens | 303.62 | 119.37 | 54 | 50 | [GitHub](https://github.com/EmreTaha/Siamese-EMD-for-AMD-Change)   |
| SWINBiomarker          | SWIN-Base | RetFound | Concatenation of [CLS] tokens | 86.69 | 30.34 | 54 | 50 | [GitHub](https://github.com/EmreTaha/Siamese-EMD-for-AMD-Change)   |
| **LesionTrace (ours)**   | ViT-Base | RetFiner | Cross-attention between visits + [CLS] difference | **85.87** | **33.75** | 32 | 50 | [GitHub](https://github.com/marziehoghbaie/LesionTrace)|iner | Cross-attention between visits + [CLS] difference | **85.87** | **33.75** | 32 | 50 |