# MAFF-HRNet
## MAFF-HRNet: Multi-Attention Feature Fusion HRNet for Building Segmentation in Remote Sensing Images
This repository contains the code for the MAFF-HRNet project from this paper. At present, the program status is version 1.0, which will be further updated and improved in the future.
<img src="https://github.com/ZhihaoChe/MAFF-HRNet/img/structure.jpg">
## Environment Setup
### Create and activate a virtual environment to work in, e.g. using Conda:
It is worth noting that python 3.7 and torch 1.8 are recommended.
```
conda create -n MAFF python=3.7
conda activate MAFF
```
### Install the remaining requirements with pip:
```
pip install -r requirements.txt
```
## Training
- Training with Multi-GPU. （recommended）  

  set distributed = True
```
python -m torch.distributed.launch --nproc_per_node=num_gpu train.py
```
- Training with single GPU.
```
python train.py
```
## Inference
```
python predict.py
```
## Reference
https://github.com/bubbliiiing/unet-pytorch
