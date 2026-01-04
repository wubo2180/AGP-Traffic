# AGP-Traffic: Adaptive Graph Pretraining for Traffic Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **AGP-Traffic**, a novel framework for spatiotemporal traffic forecasting using adaptive graph pretraining and masked self-supervised learning.

## 🌟 Highlights

- **Self-Supervised Pretraining**: Leverages masked autoencoding for learning robust spatiotemporal representations
- **Adaptive Graph Learning**: Dynamically captures spatial dependencies in traffic networks
- **Two-Stage Training**: Pretraining on large-scale data followed by task-specific fine-tuning
- **State-of-the-Art Performance**: Achieves competitive results on multiple traffic forecasting benchmarks

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## 🚀 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.13.1
- CUDA >= 11.0 (for GPU support)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/wubo2180/AGP-Traffic.git
cd AGP-Traffic

# Create conda environment (recommended)
conda env create -f environment.yaml
conda activate agp-traffic

# Or install via pip
pip install -r requirements.txt
```

### Dependencies

Key dependencies include:
- `torch >= 1.13.1`
- `easy-torch >= 1.2.12`
- `numpy`
- `scipy`
- `scikit-learn`
- `swanlab` (for experiment tracking)
- `timm >= 0.6.11`

## 📊 Dataset Preparation

### Supported Datasets

The framework supports the following traffic datasets:
- **PEMS03**: Highway traffic data from California
- **PEMS04**: Highway traffic data from California  
- **PEMS07**: Highway traffic data from California
- **PEMS08**: Highway traffic data from California
- **METR-LA**: Los Angeles County highway traffic data

### Generate Training Data

```bash
# Example: Generate data for PEMS04
cd scripts/data_preparation/PEMS04
python generate_training_data.py --history_seq_len 12 --future_seq_len 12

# Generate adjacency matrix
python generate_adj_mx.py
```

The processed data will be saved in `datasets/PEMS04/` directory.

### Data Format

- **Input sequence length**: 12 time steps (1 hour with 5-min intervals)
- **Output sequence length**: 12 time steps  
- **Features**: Traffic flow, speed, occupancy (depends on dataset)
- **Adjacency matrix**: Spatial connectivity of sensors

## 🎯 Quick Start

### Pretrain + Finetune (Recommended)

```bash
# Stage 1: Pretraining with masked autoencoding
python main.py \
    --lossType "mae" \
    --pretrain_epochs 100 \
    --preTrain "true" \
    --preTrainVal "false" \
    --preTrain_batch_size 64

# Stage 2: Fine-tuning for traffic forecasting
python main.py \
    --lossType "mae" \
    --finetune_epochs 100 \
    --preTrain "false" \
    --batch_size 8 \
    --load_pretrain_checkpoint "path/to/pretrain_model.pth"
```

### Train from Scratch

```bash
python main.py \
    --lossType "mae" \
    --pretrain_epochs 0 \
    --finetune_epochs 100 \
    --preTrain "false" \
    --batch_size 8
```

## 🔧 Training

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (PEMS03/PEMS04/PEMS07/PEMS08) | PEMS04 |
| `--lossType` | Loss function (mae/mse/huber) | mae |
| `--pretrain_epochs` | Number of pretraining epochs | 100 |
| `--finetune_epochs` | Number of fine-tuning epochs | 100 |
| `--preTrain` | Enable pretraining mode | true |
| `--preTrainVal` | Validate during pretraining | false |
| `--preTrain_batch_size` | Batch size for pretraining | 64 |
| `--batch_size` | Batch size for fine-tuning | 8 |
| `--learning_rate` | Learning rate | 0.001 |
| `--device` | GPU device ID | 0 |
| `--mask_ratio` | Masking ratio for pretraining | 0.75 |

### Example Commands

**Full training pipeline:**
```bash
# Pretraining
python main.py \
    --lossType "mae" \
    --pretrain_epochs 100 \
    --finetune_epochs 1 \
    --preTrainVal "false" \
    --preTrain "true" \
    --preTrain_batch_size 64 \
    --batch_size 8

# Fine-tuning  
python main.py \
    --lossType "mae" \
    --pretrain_epochs 100 \
    --finetune_epochs 100 \
    --preTrainVal "false" \
    --preTrain "false" \
    --preTrain_batch_size 64 \
    --batch_size 8
```

### Experiment Tracking

The framework integrates [SwanLab](https://swanlab.cn/) for experiment tracking. Training metrics, visualizations, and model checkpoints are automatically logged.

View experiments:
```bash
# Logs are saved in ./swanlog/
```

## 📈 Evaluation

Models are evaluated using standard traffic forecasting metrics:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)  
- **MAPE** (Mean Absolute Percentage Error)

Evaluation is performed on 3 horizons:
- **Short-term**: 15 minutes (3 steps)
- **Medium-term**: 30 minutes (6 steps)
- **Long-term**: 60 minutes (12 steps)

Results are saved in `./save_test/` directory.

## 📁 Project Structure

```
AGP-Traffic/
├── basicts/              # Base components
│   ├── data/            # Data processing utilities
│   ├── losses/          # Loss functions
│   ├── mask/            # Masked autoencoding models
│   ├── metrics/         # Evaluation metrics
│   ├── stgcn_arch/      # STGCN architecture
│   └── utils/           # Utility functions
├── data/                # Dataset classes
│   ├── pretraining_dataset.py
│   └── forecasting_dataset.py
├── datasets/            # Processed datasets (gitignored)
├── scripts/             # Data preparation scripts
│   └── data_preparation/
│       ├── PEMS03/
│       ├── PEMS04/
│       ├── PEMS07/
│       └── PEMS08/
├── checkpoints/         # Model checkpoints (gitignored)
├── figure/              # Visualization outputs (gitignored)
├── plot/                # Plotting scripts
├── main.py              # Main training script
├── requirements.txt     # Python dependencies
├── environment.yaml     # Conda environment
└── README.md           # This file
```

## 🎓 Citation

If you find this work useful, please cite:

```bibtex
@article{agp-traffic2024,
  title={AGP-Traffic: Adaptive Graph Pretraining for Traffic Forecasting},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🙏 Acknowledgements

This project is built upon:
- [BasicTS](https://github.com/zezhishao/BasicTS) - Baseline framework for time series forecasting
- [STD-MAE](https://github.com/Jimmy-7664/STD-MAE) - Spatiotemporal masked autoencoder
- [PyTorch](https://pytorch.org/) - Deep learning framework

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions and feedback:
- **Issues**: Please open an issue on GitHub
- **Email**: wubo2180@example.com

---

⭐ **Star this repo** if you find it helpful! 