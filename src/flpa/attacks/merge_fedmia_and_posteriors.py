import pandas as pd
from pathlib import Path

FEDMIA_PATH = "outputs/attacks/fedmia_scores.parquet"
POSTERIOR_PATH = "outputs/attacks/attack_features.parquet"
OUTPUT_PATH = "outputs/attacks/fused_attack_features.parquet"

def main():
    fedmia_df = pd.read_parquet(FEDMIA_PATH)
    posterior_df = pd.read_parquet(POSTERIOR_PATH)

    print(f"📥 FedMIA shape: {fedmia_df.shape}")
    print(f"📥 Posterior shape: {posterior_df.shape}")

    # Merge on sample_id and membership label
    merged = pd.merge(fedmia_df, posterior_df, on=["sample_id", "member"])

    print(f"✅ Merged shape: {merged.shape}")
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"📦 Saved fused features to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
