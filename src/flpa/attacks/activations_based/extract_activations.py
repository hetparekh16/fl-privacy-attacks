import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms
from torch import nn
from flpa.task import set_weights
from flpa.models import ResNet, ResNet18WithDropout, CNNWithDropout

DATA_ROOT = "./data"
MODEL_PATH = "outputs/global_model/global_model.pt"
MEMBERSHIP_DATASET = "outputs/attacks/membership_dataset.parquet"
OUTPUT_PATH = "outputs/attacks/activation_based/activation_attack_features.parquet"
BATCH_SIZE = 1
SEED = 42

# Define a hook to capture activations
activation_store = {}

def hook_fn(module, input, output):
    activation_store["activation"] = output.detach()

def extract_activations():
    df = pd.read_parquet(MEMBERSHIP_DATASET)
    print(f"Loaded membership dataset: {df.shape}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform)
    total_samples = len(dataset)
    indices = list(range(total_samples))
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNWithDropout().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Register hook on penultimate layer (CNN: after fc1, before dropout and fc2)
    penultimate_layer = model.fc1  # Hook after the first fully connected layer
    hook_handle = penultimate_layer.register_forward_hook(hook_fn)

    features_list = []

    with torch.no_grad():
        for i, row in df.iterrows():
            sample_id = row["sample_id"]
            member_label = row["member"]

            idx = sample_id  # Assuming shuffled dataset matches sample_id indices
            img, _ = dataset[idx]
            img = img.unsqueeze(0).to(device)

            # Forward pass
            _ = model(img)
            activation = activation_store["activation"].squeeze()  # Shape: [256] for CNN fc1

            feat_dict = {
                "sample_id": sample_id,
                "member": member_label,
            }
            for j, val in enumerate(activation.cpu().numpy()):
                feat_dict[f"act_{j}"] = val

            features_list.append(feat_dict)

            if i % 1000 == 0: # type: ignore
                print(f"Processed {i}/{len(df)} samples...")

    # Save dataset
    out_df = pd.DataFrame(features_list)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved activation features dataset: {OUTPUT_PATH}")

    hook_handle.remove()

if __name__ == "__main__":
    extract_activations()
