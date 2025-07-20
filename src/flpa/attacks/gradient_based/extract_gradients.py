import torch
import torch.nn.functional as F
from torch.autograd import grad
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms
from flpa.task import set_weights
from flpa.models import ResNet, ResNet18WithDropout

DATA_ROOT = "./data"
MODEL_PATH = "outputs/global_model/global_model.pt"
MEMBERSHIP_DATASET = "outputs/attacks/membership_dataset.parquet"
OUTPUT_PATH = "outputs/attacks/gradient_based/gradient_attack_features.parquet"
BATCH_SIZE = 1  # Process 1 at a time for gradient calc
SEED = 42

def extract_gradients():
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
    torch.random.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18WithDropout().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    gradient_features = []

    for i, row in df.iterrows():
        sample_id = row["sample_id"]
        label = row["member"]

        idx = sample_id  # Ensure `sample_id` matches shuffled index as before
        img, target = dataset[idx]
        img = img.unsqueeze(0).to(device)
        img.requires_grad = True

        output = model(img)
        loss = F.cross_entropy(output, torch.tensor([target]).to(device))

        grads = grad(loss, model.parameters(), retain_graph=False, create_graph=False) # type: ignore
        grad_norm = torch.norm(torch.cat([g.flatten() for g in grads])).item()

        gradient_features.append({
            "sample_id": sample_id,
            "member": label,
            "grad_norm": grad_norm,
        })

        if i % 1000 == 0: # type: ignore
            print(f"Processed {i}/{len(df)} samples")

    out_df = pd.DataFrame(gradient_features)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved gradient attack features: {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_gradients()
