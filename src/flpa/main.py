import torch
import flwr
from flpa.config import NUM_ROUNDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
print(f"Federated Learning rounds: {NUM_ROUNDS}")
