# GL-GS
This is the official repository for our paper "A Depth-Guided and Density-Adaptive Optimization Framework for Sparse-View 3D Gaussian Splatting"

<img width="755" height="430" alt="image" src="https://github.com/user-attachments/assets/ea01da32-33c6-42bd-82b7-e365c4ac360b" />



## Abstract
Sparse-view 3D Gaussian Splatting often suffers from unstable geometric structures and redundant point growth during training, partly due to insufficient geometric constraints and lack of precise regularization in the optimization process. To address these problems, this paper proposes a depth-driven, density-oriented optimization framework for 3D Gaussian Splatting (DDO-GS). First, we introduce a globally–locally stabilized depth-guided module, which employs single-view depth estimator predictions as supervisory signals and uses blurred depth maps to filter distant points. At the same time, we design a randomness-discard strategy guided by depth, allowing better preservation of near-field details while suppressing far-field redundancy. Second, to alleviate the loss of fine details in sparse-view input, we incorporate a depth-magnitude-driven selective dropout mechanism. Based on the magnitude of Gaussian density gradients, we regard the depth-direction gradient as an indicator of geometric confidence, and enhance regional refinement through gradient amplification, thereby improving the structural completeness of edge and high-frequency regions. Finally, to further stabilize point-density distribution during optimization, we propose a soft–hard combined opacity adjustment strategy, effectively suppressing redundant Gaussian growth and mitigating opacity overflow. Experiments on LLFF and Mip-NeRF360 datasets demonstrate that the proposed method yields stable geometric structures and delivers improvements in rendering quality, particularly in fine-detail reconstruction.

## Installation
Tested on Ubuntu 20.04, CUDA 11.8, PyTorch 1.12.1

``````
conda env create --file environment.yml
conda activate DDO-GS
``````

``````
pip install gaussian-splatting/submodules/diff-gaussian-rasterization-confidence
pip install gaussian-splatting/submodules/simple-knn
``````

## Required Data
```
├── /data
   ├── mipnerf360
        ├── bicycle
        ├── bonsai
        ├── ...
   ├── nerf_llff_data
        ├── fern
        ├── flower
        ├── ...
```

## Evaluation

### LLFF

1. Download LLFF from [the official download link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

2. run colmap to obtain initial point clouds with limited viewpoints:
    ```bash
   python tools/colmap_llff.py
   ```

3. Start training and testing:

   ```bash
   # for example
   bash scripts/run_llff.sh ${gpu_id} data/nerf_llff_data/fern output/llff/fern
   ```

### MipNeRF-360

1. Download MipNeRF-360 from [the official download link](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).

2. run colmap to obtain initial point clouds with limited viewpoints:
    ```bash
   python tools/colmap_360.py
   ```

# Acknowledgement

Special thanks to the following awesome projects!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [CoR-GS](https://github.com/jiaw-z/CoR-GS)
- [DropGaussian_release](https://github.com/DCVL-3D/DropGaussian_release)
