import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df = pd.read_parquet("outputs/attacks/fedmia_scores.parquet")
y_true = df["member"]
y_scores = df["fedmia_score"]
roc_auc = roc_auc_score(y_true, y_scores)
print(f"📊 FedMIA ROC AUC: {roc_auc:.4f}")
