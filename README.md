# Repository: master_inference_sen2sr_deep

This repository is part of the master’s thesis: [Evaluating_Sentinel-2_Super-Resolution_Algorithms_for_Automated_Building_Delineation](https://github.com/Zerhigh/Evaluating_Sentinel-2_Super-Resolution_Algorithms_for_Automated_Building_Delineation)

This project contains scripts and utilities for super-resolving Sentinel-2 imagery with the superIX framework (https://huggingface.co/isp-uv-es/superIX) and the, SEN2SR_RGBN model. This model requires mamba, which in return depends on CUDA>=12.

Access the corresponding weights from superIX and insert into the corresponding folders in `model/SEN2SR_RGBN/`. The code in this repository allows the SR and subsequent interpolation of all model outputs to images with a resolution of 2.5m and image shapes of `(4, 512, 512)`.

The repository includes:
- `inference.py` — script for loading image files and applying SR inference to them. geospatial information is retained.
---
