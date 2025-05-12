import os
import random
import pandas as pd
from pathlib import Path
import torchvision

# Constants
DATA_ROOT = "./data"
METRICS_LOG_DIR = Path("outputs/sample_id_logs")
OUTPUT_PATH = Path("outputs/attacks/membership_dataset.parquet")
SEED = 42


def build_membership_dataset(save: bool = True) -> pd.DataFrame:
    # Load CIFAR-10 train=True deterministically
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=transform
    )

    train_samples = len(trainset)
    indices = list(range(train_samples))
    random.seed(SEED)
    random.shuffle(indices)

    # Read sample_id logs from training rounds
    all_member_ids = set()
    for file in sorted(METRICS_LOG_DIR.glob("round_*_client_train.parquet")):
        df = pd.read_parquet(file)
        for sample_list in df["sample_ids"]:
            all_member_ids.update(sample_list)

    member_ids = list(all_member_ids)
    print(f"✅ Found {len(member_ids)} unique member sample_ids from logs")

    # Create non-member list from remaining indices
    all_possible = set(range(train_samples))
    non_member_ids = list(all_possible - all_member_ids)

    sample_size = min(len(member_ids), len(non_member_ids))
    sampled_members = random.sample(member_ids, sample_size)
    sampled_non_members = random.sample(non_member_ids, sample_size)

    # Build labeled DataFrame
    member_df = pd.DataFrame({"sample_id": sampled_members, "member": 1})

    non_member_df = pd.DataFrame({"sample_id": sampled_non_members, "member": 0})

    membership_df = (
        pd.concat([member_df, non_member_df]).sample(frac=1).reset_index(drop=True)
    )

    if save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        membership_df.to_parquet(OUTPUT_PATH, index=False)
        print(f"📦 Saved membership dataset to: {OUTPUT_PATH.resolve()}")

    return membership_df


if __name__ == "__main__":
    _ = build_membership_dataset()
