# Federated Learning Privacy Attacks (FLPA)

This project demonstrates privacy vulnerabilities in federated learning (FL) systems through systematic membership inference attacks (MIAs). It includes a complete modular framework for federated training, logging, attack evaluation, and visualization.

## Overview

The project implements:
- Federated learning simulation using [Flower](https://flower.dev/) with the CIFAR-10 dataset
- Support for multiple model architectures:
  - Shallow CNN (baseline)
  - Deep ResNet-18 model (achieving ~80% accuracy)
- Support for multiple FL strategies:
  - FedAvg
  - FedAdam
- Comprehensive membership inference attacks:
  - Posterior-based (black-box)
  - Gradient-based (white-box)
  - Activation-based (white-box)
  - Fusion attack (posterior + gradient)
- End-to-end experiment reproducibility with deterministic partitioning and ground-truth membership tracking
- Metrics logging and attack result visualization

## Prerequisites

- [uv](https://github.com/astral-sh/uv) — Fast Python package installer and virtual environment manager

## Installation

### 1️⃣ Install `uv` (Linux)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2️⃣ Clone the repository

```bash
git clone git@github.com:hetparekh16/fl-privacy-attacks.git
cd fl-privacy-attacks
```

### 3️⃣ Install dependencies

```bash
uv sync
```

## Usage

### Step 1: Prepare CIFAR-10 dataset

```bash
uv run src/flpa/dataset.py
```

This downloads CIFAR-10 to the `data/` directory and prepares deterministic client partitions.

### Step 2: Run federated training

```bash
uv run flwr run . local-simulation-gpu
```

This will:
- Train either a shallow CNN or ResNet-18 (controlled via config)
- Select optimizer strategy (FedAvg or FedAdam)
- Log training metrics and save the final global model at `outputs/global_model/`

### Step 3: Run membership inference attacks

```bash
uv run src/flpa/attacks/run_attack_pipeline.py
```

This pipeline executes:
1. Build membership dataset from logged sample IDs
2. Extract posteriors, gradients, and activations from final model
3. Train and evaluate attack models (Logistic Regression, Random Forest, MLP)

Attack results (metrics + models) are stored under `outputs/attacks/`.

### Step 4: Visualize results (optional)

```bash
uv run streamlit run apps/streamlit_app.py
```

Features:
- Per-client training curves
- Global validation accuracy and loss
- Attack effectiveness (ROC curves, confusion matrices, metrics)

## Project Structure

```
├── src/flpa
│   ├── client_app.py         # Flower client
│   ├── server_app.py         # Flower server
│   ├── dataset.py            # CIFAR-10 dataset preparation
│   ├── task.py               # Model and training logic (CNN, ResNet-18)
│   ├── fed_strategies/       # LoggingFedAvg, LoggingFedAdam
│   ├── attacks/
│   │   ├── posterior/        # Posterior-based MIA
│   │   ├── gradient_based/   # Gradient-based MIA
│   │   ├── activation_based/ # Activation-based MIA
│   │   ├── fusion_posterior_and_gradient/ # Fusion attack
│   │   └── base/             # Membership dataset construction
│   └── utils.py              # Utilities (logging, helpers)
├── apps                      # Streamlit dashboard
├── data                      # CIFAR-10 dataset storage
├── outputs                   # All generated logs, models, metrics
├── pyproject.toml            # Dependency specification
├── uv.lock                   # Dependency lock file
└── README.md
```

## Key Features

📦 Modular design for attack extensibility

🔒 Insider and outsider threat simulation (black-box and white-box attacks)

📈 Robust metrics logging and visualization

🖥️ GPU acceleration ready

🔁 Fully reproducible experiments with deterministic partitioning and ground-truth membership tracking