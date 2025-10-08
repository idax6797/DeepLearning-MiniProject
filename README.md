# Deep Learning - Mini Project: MIP-PET Tumor Segmentation

## Project Overview
Automatic segmentation of tumors from Maximum Intensity Projected Positron Emission Tomography (MIP-PET) images using deep learning. This project aims to accurately identify tumor regions in whole-body medical scans. 

## Problem Statement
Given MIP-PET images of cancer patients, predict pixel-wise tumor segmentation masks. This is a binary segmentation task where each pixel must be classified as either tumor (white) or healthy tissue (black).

### Data Structure
```
data/
├── patients/
│   ├── imgs/        # 182 MIP-PET images with tumors
│   └── labels/      # Ground truth segmentation masks
└── controls/
    └── imgs/        # 426 healthy control images (no cancer)
```