import os
import random
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from flpa.task import set_weights
from flpa.models import CNN, ResNet, ResNet18WithDropout

# Constants
DATA_ROOT = "./data"
MODEL_PATH = "outputs/global_model/global_model.pt"
MEMBERSHIP_DATASET = "outputs/attacks/membership_dataset.parquet"
OUTPUT_PATH = "outputs/attacks/posterior/attack_features.parquet"
SEED = 42
BATCH_SIZE = 128


def extract_posteriors():
    # Step 1: Load membership dataset
    df = pd.read_parquet(MEMBERSHIP_DATASET)
    print(f"Loaded membership dataset with shape: {df.shape}")

    # Step 2: Load CIFAR-10 train=True with same shuffling
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=transform
    )
    total_samples = len(dataset)
    indices = list(range(total_samples))
    random.seed(SEED)
    random.shuffle(indices)

    # Build sample_id → dataset index map
    sample_id_to_index = {
        sample_id: indices[sample_id] for sample_id in df["sample_id"]
    }

    # Step 3: Load global model
    model = ResNet18WithDropout()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print(f"Global model loaded from {MODEL_PATH}")

    # Step 4: Compute softmax predictions for each sample
    posteriors = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for i, row in df.iterrows():
            sample_id = row["sample_id"]
            label = row["member"]
            index = sample_id_to_index[sample_id]
            img, _ = dataset[index]
            img = img.unsqueeze(0).to(device)  # [1, 3, 32, 32]
            logits = model(img)
            softmax = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            posteriors.append(
                {
                    "sample_id": sample_id,
                    "member": label,
                    **{f"posterior_{j}": softmax[j] for j in range(10)},
                }
            )

            if i % 1000 == 0:  # type: ignore
                print(f"Processed {i}/{len(df)} samples...")

    # Step 5: Save to Parquet
    post_df = pd.DataFrame(posteriors)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    post_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved attack feature dataset to: {OUTPUT_PATH}")
    return post_df


if __name__ == "__main__":
    _ = extract_posteriors()
