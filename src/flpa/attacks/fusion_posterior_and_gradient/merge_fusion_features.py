import pandas as pd
from pathlib import Path

# Paths
POSTERIOR_PATH = "outputs/attacks/posterior/attack_features.parquet"
GRADIENT_PATH = "outputs/attacks/gradient_based/gradient_attack_features.parquet"
OUTPUT_PATH = "outputs/attacks/fusion/fused_attack_features.parquet"

def merge_fusion_features():
    posterior_df = pd.read_parquet(POSTERIOR_PATH)
    gradient_df = pd.read_parquet(GRADIENT_PATH)

    print(f"Loaded posterior features: {posterior_df.shape}")
    print(f"Loaded gradient features: {gradient_df.shape}")

    # Merge on sample_id and ensure same label alignment
    merged_df = posterior_df.merge(
        gradient_df[["sample_id", "grad_norm"]],
        on="sample_id",
        how="inner"
    )

    print(f"Merged dataset shape: {merged_df.shape}")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved fused features to: {OUTPUT_PATH}")

if __name__ == "__main__":
    merge_fusion_features()
