import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from flpa.task import CNN
from flpa.attacks.fedmia_attack import compute_gradient, fedmia_score
from pathlib import Path
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "outputs/global_model/global_model.pt"
MEMBERSHIP_DATASET = "outputs/attacks/membership_dataset.parquet"
OUTPUT_PATH = "outputs/attacks/fedmia_scores.parquet"
DATA_ROOT = "./data"
SEED = 42

def main():
    # Load membership labels
    df = pd.read_parquet(MEMBERSHIP_DATASET)
    print(f"📥 Loaded membership dataset with shape: {df.shape}")

    # Load CIFAR-10 train=True with same shuffle
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform)
    indices = list(range(len(dataset)))
    random.seed(SEED)
    random.shuffle(indices)
    sample_id_to_index = {sample_id: indices[sample_id] for sample_id in df["sample_id"]}

    # Load global model
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"🧠 Loaded global model from: {MODEL_PATH}")

    # Choose 200 non-member samples to estimate Q_out
    non_member_df = df[df["member"] == 0].sample(n=200, random_state=SEED)
    nonmember_grads = []

    print("🔁 Collecting non-member gradients...")
    for _, row in tqdm(non_member_df.iterrows(), total=len(non_member_df)):
        idx = sample_id_to_index[row["sample_id"]]
        x, y = dataset[idx]
        x, y = x.unsqueeze(0).to(DEVICE), torch.tensor([y]).to(DEVICE)
        grad = compute_gradient(model, F.cross_entropy, x, y)
        nonmember_grads.append(grad)

    # df = df.sample(n=1000, random_state=SEED)  # Put this before the scoring loop

    # Compute FedMIA score for all samples
    print("🧪 Scoring samples...")
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sid, label = row["sample_id"], row["member"]
        idx = sample_id_to_index[sid]
        x, y = dataset[idx]
        x, y = x.unsqueeze(0).to(DEVICE), torch.tensor([y]).to(DEVICE)
        query_grad = compute_gradient(model, F.cross_entropy, x, y)
        score = fedmia_score(query_grad, nonmember_grads)
        results.append({"sample_id": sid, "member": label, "fedmia_score": score})

    # Save results
    score_df = pd.DataFrame(results)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    score_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"📦 Saved FedMIA results to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
