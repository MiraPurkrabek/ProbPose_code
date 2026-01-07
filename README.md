</h1><div id="toc">
  <ul align="center" style="list-style: none; padding: 0; margin: 0;">
    <summary>
      <h1 style="margin-bottom: 0.0em;">
        ProbPose: A Probabilistic Approach to 2D Human Pose Estimation
      </h1>
    </summary>
  </ul>
</div>
</h1><div id="toc">
  <ul align="center" style="list-style: none; padding: 0; margin: 0;">
    <summary>
      <h2 style="margin-bottom: 0.2em;">
        CVPR 2025
      </h2>
    </summary>
  </ul>
</div>


<div align="center">
<img src="demo/resources/McLaughlin.gif" alt="ProbPose Showcase">


[![Paper](https://img.shields.io/badge/Paper-CVPR%202025-blue)](https://arxiv.org/abs/2412.02254) &nbsp;&nbsp;&nbsp;
[![Website](https://img.shields.io/badge/Website-ProbPose-green)](https://mirapurkrabek.github.io/ProbPose/) &nbsp;&nbsp;&nbsp;
[![License](https://img.shields.io/badge/License-GPL%203.0-orange.svg)](LICENSE)

</div>

## 📋 Overview

ProbPose introduces a probabilistic framework for human pose estimation, focusing on reducing false positives by predicting keypoint presence probabilities and handling out-of-image keypoints. It also introduces the new Ex-OKS metric to evaluate models on false positive predictions.

Key contributions:
- **Presence probability** concept that distinguishes keypoint presence from confidence
- **ProbPose**: top-down model for out-of-image keypoints estimation
- **OKSLoss adapted for dense predictions** in risk minimization formulation
- **Ex-OKS evaluation metric** penalizing false positive keypoints
- **CropCOCO dataset** for out-of-image and false positive keypoints evaluation

For more details, please visit our [project website](https://mirapurkrabek.github.io/ProbPose/).

## 📢 News

- **July 2025**: [exococotools PyPI package](https://github.com/MiraPurkrabek/exococotools) available
- **June 2025**: Live webcam demo branch available 
- **April 2025**: Code is released
- **March 2025**: Paper accepted to CVPR 2025! 🎉

## 🚀 Installation

This project is built on top of [MMPose](https://github.com/open-mmlab/mmpose). Please refer to the [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html) for detailed setup instructions.

Basic installation steps:
```bash
# Clone the repository
git clone https://github.com/mirapurkrabek/ProbPose_code.git ProbPose/
cd ProbPose

# Install your version of torch, torchvision, OpenCV and NumPy
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.25.1 opencv-python==4.9.0.80

# Install MMLibrary
pip install -U openmim
mim install mmengine "mmcv==2.1.0" "mmdet==3.3.0" "mmpretrain==1.2.0"

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## 🎮 Demo

### Single Image Demo

Run the following command to test ProbPose on a single image:

```bash
python demo/image_demo.py \
demo/resources/CropCOCO_single_example.jpg \
configs/body_2d_keypoint/topdown_probmap/coco/td-pm_ProbPose-small_8xb64-210e_coco-256x192.py \
path/to/pre-trained/weights.pth \
--out-file demo/results/CropCOCO_single_example.jpg \
--draw-heatmap
```

Expected result (click for full size):  
<a href="demo/resources/single_demo_result.jpg">
    <img src="demo/resources/single_demo_result.jpg" alt="Single Image Demo" width="100"/>
</a>

### Demo with MMDetection

For more complex scenarios with multiple people, use the MMDetection-based demo:

```bash
python demo/topdown_demo_with_mmdet.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/body_2d_keypoint/topdown_probmap/coco/td-pm_ProbPose-small_8xb64-210e_coco-256x192.py \
path/to/pre-trained/weights.pth \
--input demo/resources/CropCOCO_multi_example.jpg \
--draw-bbox \
--output-root demo/results/ \
--draw-heatmap
```

Expected result (click for full size):  
<a href="demo/resources/multi_demo_result.jpg">
    <img src="demo/resources/multi_demo_result.jpg" alt="Multi Person Demo" width="100"/>
</a>

For more detailed information on demos and visualization options, please refer to the [MMPose documentation](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html).

### Google Colab Demo

You can also run ProbPose directly in Google Colab without any local installation!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QKyFBlKqnVd0Q_TC19AKLFr2zrT-gqdZ?usp=sharing)

**Note:** The installation process in Colab takes approximately 5-10 minutes. Please be patient during setup - it is not stuck!

For more detailed information on demos and visualization options, please refer to the [MMPose documentation](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html).

## 📦 Pre-trained Models

Pre-trained models are available on [VRG Hugging Face 🤗](https://huggingface.co/vrg-prague/ProbPose-s/):
- [ProbPose-s weights](https://huggingface.co/vrg-prague/ProbPose-s/resolve/main/ProbPose-s.pth)

## ✂️ CropCOCO Dataset

The [CropCOCO dataset](https://huggingface.co/datasets/vrg-prague/CropCOCO) is available on VRG Hugging Face 🤗.

For Ex-OKS and Ex-mAP evaluation, you can use [cocoeval.py](mmpose/evaluation/metrics/_cocoeval.py) file which is a direct replacement for the original cocoeval.py file from xtcocotools.
We plan to release Ex-mAP evaluation tool as a standalone package similar to xtcocotools.

## 📏 Ex-OKS Evaluation

Our Ex-OKS metric can be computed via the standalone exococotools package, which is fully backward-compatible with xtcocotools/pycocotools. Install and run it as a drop-in replacement:

```bash
pip install exococotools
```

For more details and advanced options, see the package website: https://github.com/MiraPurkrabek/exococotools

## 🗺️ Roadmap

- [ ] Add config and weights for DoubleProbmap model
- [x] Add out-of-image pose visualization
- [x] Add new package with Ex-OKS implementation --> [exococotools](https://github.com/MiraPurkrabek/exococotools)
- [ ] Add ProbPose to MMPose library
- [x] Create HuggingFace demo

## 🙏 Acknowledgments

This project is built on top of [MMPose](https://github.com/open-mmlab/mmpose). We would like to thank the MMPose team for their excellent work and support.

## 📝 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@InProceedings{Purkrabek2025CVPR,
    author    = {Purkrabek, Miroslav and Matas, Jiri},
    title     = {ProbPose: A Probabilistic Approach to 2D Human Pose Estimation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27124-27133}
}
```
