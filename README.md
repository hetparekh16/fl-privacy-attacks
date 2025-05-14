# Federated Learning Privacy Attacks (FLPA)

This project demonstrates privacy vulnerabilities in federated learning systems through membership inference attacks. It includes a complete federated learning workflow with attack implementation and visualization tools.

## Overview

The project implements:
- Federated learning simulation with CIFAR-10 dataset (downloaded via torchvision) using [Flower](https://flower.dev/)
- Membership inference attacks on the global model
- Comprehensive attack evaluation metrics
- Interactive visualization dashboard

## Prerequisites

- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and virtual environment manager

## Installation

### 1. Install uv (Linux)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone git@github.com:hetparekh16/fl-privacy-attacks.git
cd fl-privacy-attacks
```

### 3. Install dependencies

```bash
uv sync
```

## Usage

### Step 1: Download the CIFAR-10 dataset

```bash
uv run src/flpa/dataset.py
```

This will download the CIFAR-10 dataset to the [`data`](data) directory and prepare it for the federated learning simulation.

### Step 2: Run the federated learning simulation

```bash
uv run flwr run
```

This command:
- Initializes a federated learning server
- Creates 10 simulated clients
- Trains the model for 6 rounds with 50% client participation per round
- Saves training metrics and the global model

### Step 3: Run the privacy attack pipeline

```bash
uv run src/flpa/attacks/run_attack_pipeline.py
```

This executes three sequential steps:
1. Building a membership dataset (determining which samples were used in training)
2. Extracting posterior probabilities from the global model
3. Training attack models (logistic regression, random forest, MLP) to infer membership

### Step 4: Visualize results with Streamlit

```bash
uv run streamlit run apps/streamlit_app.py
```

The visualization dashboard provides:
- Per-client training metrics visualization
- Global model performance tracking
- Attack effectiveness evaluation with ROC curves and confusion matrices

## Project Structure

- [`src/flpa`](src/flpa) - Core implementation
  - [`src/flpa/client_app.py`](src/flpa/client_app.py) - Flower client implementation
  - [`src/flpa/server_app.py`](src/flpa/server_app.py) - Flower server implementation
  - [`src/flpa/task.py`](src/flpa/task.py) - CNN model and training logic
  - [`src/flpa/dataset.py`](src/flpa/dataset.py) - CIFAR-10 dataset preparation
  - [`src/flpa/utils.py`](src/flpa/utils.py) - Utility functions for logging and metrics
  - `attacks/` - Attack implementation
    - [`src/flpa/attacks/build_membership_dataset.py`](src/flpa/attacks/build_membership_dataset.py) - Creates labeled member/non-member dataset
    - [`src/flpa/attacks/extract_posteriors.py`](src/flpa/attacks/extract_posteriors.py) - Extracts model predictions
    - [`src/flpa/attacks/train_attack_model.py`](src/flpa/attacks/train_attack_model.py) - Trains attack classifiers
    - [`src/flpa/attacks/run_attack_model.py`](src/flpa/attacks/run_attack_model.py) - Run the attack pipeline
- [`apps`](apps) - Visualization applications
- [`outputs`](outputs) - Generated outputs (models, logs, metrics)
- [`data`](data) - CIFAR-10 dataset
- [`pyproject.toml`](pyproject.toml) - Project configuration and dependencies
- [`README.md`](README.md) - Project documentation
- [`uv.lock`](uv.lock) - Dependency lock file

## Configuration

The federated learning system is configured in [`pyproject.toml`](pyproject.toml) with:
- 6 training rounds
- 50% client participation per round
- 2 local epochs per client
- 10 simulated clients
