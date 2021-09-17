![Logo](images/logo.png)

[![CI testing](https://github.com/Borda/MedicalNet/actions/workflows/ci-testing.yml/badge.svg?branch=master&event=push)](https://github.com/Borda/MedicalNet/actions/workflows/ci-testing.yml)
[![Install package](https://github.com/Borda/MedicalNet/actions/workflows/ci-install-pkg.yml/badge.svg?branch=master&event=push)](https://github.com/Borda/MedicalNet/actions/workflows/ci-install-pkg.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Borda/MedicalNet/master.svg)](https://results.pre-commit.ci/latest/github/Borda/MedicalNet/master)

# MedicalNet

This repository contains a Pytorch implementation
of [Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625). Many studies have shown
that the performance on deep learning is significantly affected by volume of training data. The MedicalNet project
aggregated the dataset with diverse modalities, target organs, and pathologies to to build relatively large datasets.
Based on this dataset, a series of 3D-ResNet pre-trained models and corresponding transfer-learning training code are
provided.

### License

MedicalNet is released under the MIT License (refer to the LICENSE file for detailso).

### Citing MedicalNet

If you use this code or pre-trained models, please cite the following:

```
@article{chen2019med3d,
    title={Med3D: Transfer Learning for 3D Medical Image Analysis},
    author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
    journal={arXiv preprint arXiv:1904.00625},
    year={2019}
}
```

### Update(2019/07/30)

We uploaded 4 pre-trained models based on more datasets (23 datasets).

```
Model name             : parameters settings
resnet_10_23dataset.pth: --model resnet --model_depth 10 --resnet_shortcut B
resnet_18_23dataset.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet_34_23dataset.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet_50_23dataset.pth: --model resnet --model_depth 50 --resnet_shortcut B
```

We transferred the above pre-trained models to the multi-class segmentation task (left lung, right lung and background)
on Visceral dataset. The results are as follows:

<table class="dataintable">
<tr>
   <th>Network</th>
   <th>Pretrain</th>
   <th>LungSeg(Dice)</th>
</tr>
<tr>
   <td rowspan="2">3D-ResNet10</td>
   <td>Train from scratch</td>
   <td>69.31%</td>
</tr>
<tr>
    <td>MedicalNet</td>
    <td>96.56%</td>
</tr>
<tr>
   <td rowspan="2">3D-ResNet18</td>
   <td>Train from scratch</td>
   <td>70.89%</td>
</tr>
<tr>
    <td>MedicalNet</td>
    <td>94.68%</td>
</tr>
<tr>
   <td rowspan="2">3D-ResNet34</td>
   <td>Train from scratch</td>
   <td>75.25%</td>
</tr>
<tr>
    <td>MedicalNet</td>
    <td>94.14%</td>
</tr>
<tr>
   <td rowspan="2">3D-ResNet50</td>
   <td>Train from scratch</td>
   <td>52.94%</td>
</tr>
<tr>
    <td>MedicalNet</td>
    <td>89.25%</td>
</tr>
</table>

### Contents

1. [Requirements](#Requirements)
1. [Installation](#Installation)
1. [Demo](#Demo)
1. [Experiments](#Experiments)
1. [TODO](#TODO)
1. [Acknowledgement](#Acknowledgement)

### Requirements

- Python 3.7.0
- PyTorch-0.4.1
- CUDA Version 9.0
- CUDNN 7.0.5

### Installation

- Install Python 3.7.0
- `pip install -r requirements.txt`

### Demo

- Structure of data directories

```
MedicalNet is used to transfer the pre-trained model to other datasets (here the MRBrainS18 dataset is used as an example).
MedicalNet/
    |--med3d
    |   |--datasets/：Data preprocessing module
    |   |   |--brains18.py：MRBrainS18 data preprocessing script
    |   |--models/：Model construction module
    |   |   |--resnet.py：3D-ResNet network build script
    |   |--utils/：tools
    |   |   |--logger.py：Logging script
    |   |--model.py: Network processing script
    |   |--setting.py: Parameter setting script
    |   |--train.py: MRBrainS18 training demo script
    |   |--test.py: MRBrainS18 testing demo script
    |--toy_data/：For CI test
    |--data/：Data storage module
    |   |--MRBrainS18/：MRBrainS18 dataset
    |	|   |--images/：source image named with patient ID
    |	|   |--labels/：mask named with patient ID
    |   |--train.txt: training data lists
    |   |--val.txt: validation data lists
    |--pretrain/：Pre-trained models storage module
    |--requirement.txt: Dependent library list
    |--README.md
```

- Network structure parameter settings

```
Model name   : parameters settings
resnet_10.pth: --model resnet --model_depth 10 --resnet_shortcut B
resnet_18.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet_34.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet_50.pth: --model resnet --model_depth 50 --resnet_shortcut B
resnet_101.pth: --model resnet --model_depth 101 --resnet_shortcut B
resnet_152.pth: --model resnet --model_depth 152 --resnet_shortcut B
resnet_200.pth: --model resnet --model_depth 200 --resnet_shortcut B
```

- After successfully completing basic installation, you'll be ready to run the demo.

1. Clone the MedicalNet repository
   ```bash
   git clone https://github.com/Tencent/MedicalNet
   ```
1. Download data & pre-trained
   models ([Google Drive](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view?usp=sharing)
   or [Tencent Weiyun](https://share.weiyun.com/55sZyIx))
   Unzip and move files
   ```bash
   mv MedicalNet_pytorch_files.zip MedicalNet/.
   cd MedicalNet
   unzip MedicalNet_pytorch_files.zip
   ```
1. Run the training code (e.g. 3D-ResNet-50)
   ```bash
   python train.py --gpu_id 0 1    # multi-gpu training on gpu 0,1
   # OR
   python train.py --gpu_id 0    # single-gpu training on gpu 0
   ```
1. Run the testing code (e.g. 3D-ResNet-50)
   ```bash
   python test.py --gpu_id 0 --resume_path trails/models/resnet_50_epoch_110_batch_0.pth.tar --img_list data/val.txt
   ```

### Experiments

- **Computational Cost**

  ```
  GPU：NVIDIA Tesla P40
  ```

  | Network      | Paramerers (M) | Running time (s) |
  | ------------ | -------------- | ---------------- |
  | 3D-ResNet10  | 14.36          | 0.18             |
  | 3D-ResNet18  | 32.99          | 0.19             |
  | 3D-ResNet34  | 63.31          | 0.22             |
  | 3D-ResNet50  | 46.21          | 0.21             |
  | 3D-ResNet101 | 85.31          | 0.29             |
  | 3D-ResNet152 | 117.51         | 0.34             |
  | 3D-ResNet200 | 126.74         | 0.45             |

- **Performance**

  Visualization of the segmentation results of our approach vs. the comparison ones after the same training epochs.
  It has demonstrated that the efficiency for training convergence and accuracy based on our MedicalNet pre-trained models.

  ![Compare results](images/efficiency.gif)

  Results of transfer MedicalNet pre-trained models to lung segmentation (LungSeg) and pulmonary nodule classification (NoduleCls) with Dice and accuracy evaluation metrics, respectively.

   <table class="dataintable">
   <tr>
      <th>Network</th>
      <th>Pretrain</th>
      <th>LungSeg(Dice)</th>
      <th>NoduleCls(accuracy)</th>
   </tr>
   <tr>
      <td rowspan="2">3D-ResNet10</td>
      <td>Train from scratch</td>
      <td>71.30%</td>
      <td>79.80%</td>
   </tr>
   <tr>
       <td>MedicalNet</td>
       <td>87.16%</td>
       <td>86.87%</td>
   </tr>
   <tr>
      <td rowspan="2">3D-ResNet18</td>
      <td>Train from scratch</td>
      <td>75.22%</td>
      <td>80.80%</td>
   </tr>
   <tr>
       <td>MedicalNet</td>
       <td>87.26%</td>
       <td>88.89%</td>
   </tr>
   <tr>
      <td rowspan="2">3D-ResNet34</td>
      <td>Train from scratch</td>
      <td>76.82%</td>
      <td>83.84%</td>
   </tr>
   <tr>
       <td>MedicalNet</td>
       <td>89.31%</td>
       <td>89.90%</td>
   </tr>
   <tr>
      <td rowspan="2">3D-ResNet50</td>
      <td>Train from scratch</td>
      <td>71.75%</td>
      <td>84.85%</td>
   </tr>
   <tr>
       <td>MedicalNet</td>
       <td>93.31%</td>
       <td>89.90%</td>
   </tr>
   <tr>
      <td rowspan="2">3D-ResNet101</td>
      <td>Train from scratch</td>
      <td>72.10%</td>
      <td>81.82%</td>
   </tr>
   <tr>
       <td>MedicalNet</td>
       <td>92.79%</td>
       <td>90.91%</td>
   </tr>
   <tr>
      <td rowspan="2">3D-ResNet152</td>
      <td>Train from scratch</td>
      <td>73.29%</td>
      <td>73.74%</td>
   </tr>
   <tr>
       <td>MedicalNet</td>
       <td>92.33%</td>
       <td>90.91%</td>
   </tr>
   <tr>
      <td rowspan="2">3D-ResNet200</td>
      <td>Train from scratch</td>
      <td>71.29%</td>
      <td>76.77%</td>
   </tr>
   <tr>
       <td>MedicalNet</td>
       <td>92.06%</td>
       <td>90.91%</td>
   </tr>
   </table>

- Please refer to [Med3D: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625) for more
  details：

### TODO

- [x] 3D-ResNet series pre-trained models
- [x] Transfer learning training code
- [x] Training with multi-gpu
- [ ] 3D efficient pre-trained models（e.g., 3D-MobileNet, 3D-ShuffleNet）
- [ ] 2D medical pre-trained models
- [x] Pre-trained MedicalNet models based on more medical dataset

### Acknowledgement

We thank [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
and [MRBrainS18](https://mrbrains18.isi.uu.nl/) which we build MedicalNet refer to this releasing code and the dataset.

### Contribution

If you want to contribute to MedicalNet, be sure to review
the [contribution guidelines](https://github.com/Tencent/MedicalNet/blob/master/CONTRIBUTING.md)
