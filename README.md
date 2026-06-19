# PointBiLSTM

Official PyTorch implementation of:

**Efficient and Lightweight Long-Range Modeling for 3D Point Cloud Classification and Segmentation**

## Introduction

PointBiLSTM is a lightweight long-range contextual modeling framework for 3D point cloud understanding. The method introduces a bidirectional sequence modeling strategy to efficiently capture long-range dependencies while maintaining low computational complexity.

The framework supports:

* Point cloud classification
* Point cloud part segmentation
* ModelNet40
* ScanObjectNN
* ShapeNet Part

## Requirements

* Python 3.8+
* PyTorch 1.10+
* CUDA 11.8+

## Installation

```bash
git clone https://github.com/wendaodao04/PointBiLSTM.git
cd PointBiLSTM
pip install -r requirements.txt
```

## Datasets

### ModelNet40

Download from:

https://modelnet.cs.princeton.edu/

### ScanObjectNN

Download from:

https://hkust-vgd.github.io/scanobjectnn/

### ShapeNet Part

Download from:

https://shapenet.org/

## Training

### Classification

```bash
python train_classification.py
```

### Part Segmentation

```bash
python train_partseg.py
```

## Evaluation

```bash
python test_classification.py
```

```bash
python test_partseg.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026pointbilstm,
  title={Efficient and Lightweight Long-Range Modeling for 3D Point Cloud Classification and Segmentation},
  author={Liu, Dongzhen， Deng, Yuzhong， Zou, Jianxiao and Fan, Shicai},
  journal={PLOS ONE},
  year={2026}
}
```


