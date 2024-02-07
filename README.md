# Introduction

The implementation of paper [NormAUG: Normalization-guided Augmentation for Domain Generalization](https://arxiv.org/abs/2307.13492).

<img src="imgs/training stage.png" alt="training stage" style="zoom:50%;" />




# How to setup the environment

This code is built on top of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions provided in https://github.com/KaiyangZhou/Dassl.pytorch to install the `dassl` environment.


The data can be found in [Dassl.pytorch /DATASETS.md](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/README.md).

```
$DATA/
|–– pacs/
|   |–– images/
|   |–– splits/
|–– office_home_dg/
|   |–– art/
|   |–– clipart/
|   |–– product/
|   |–– real_world/
```



# How to run 

The script is provided in `NormAUG/scripts/NormAUG/run_ssdg.sh`. 

You need to update the `DATA` variable that points to the directory where you put the datasets.

```bash
conda activate dassl
cd scripts/NormAUG
bash run_dg.sh dg_pacs v1 0 
```

```
output/
|–– dg_pacs/
|   |–– NormAUG/
|   |   |–– resnet18/
|   |   |   |–– v1/ # contains results on four target domains
|   |   |   |   |–– art_painting/ 
|   |   |   |   |–– cartoon/
|   |   |   |   |–– photo/
|   |   |   |   |–– sketch/
```



# Citation

```tex
@article{qi2023normaug,
  title={NormAUG: Normalization-guided Augmentation for Domain Generalization},
  author={Qi, Lei and Yang, Hongpeng and Shi, Yinghuan and Geng, Xin},
  journal={IEEE Transactions on Image Processing},
  year={2024}
}
```



# Acknowledgements

Some code are adapted from [ssdg-benchmark](https://github.com/KaiyangZhou/ssdg-benchmark). We thank them for their projects.