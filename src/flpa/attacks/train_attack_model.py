import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# Paths
FEATURES_PATH = "outputs/attacks/attack_features.parquet"
MODEL_DIR = Path("outputs/attacks/models")
METRICS_DIR = Path("outputs/attacks/metrics")

# Config
TEST_SIZE = 0.2
RANDOM_STATE = 42
POSTERIOR_FEATURES = [f"posterior_{i}" for i in range(10)]

MODELS = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE
    ),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=1000, random_state=RANDOM_STATE
    ),
}


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 {name.upper()} Evaluation:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "true_pos": cm[1][1],
        "false_pos": cm[0][1],
        "true_neg": cm[0][0],
        "false_neg": cm[1][0],
    }


def train_all_attack_models():
    df = pd.read_parquet(FEATURES_PATH)
    X = df[POSTERIOR_FEATURES].values
    y = df["member"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y  # type: ignore
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for name, model in MODELS.items():
        print(f"\n🚀 Training {name.upper()}...")
        model.fit(X_train, y_train)

        # Save model
        model_path = MODEL_DIR / f"{name}_attack_model.joblib"
        joblib.dump(model, model_path)
        print(f"💾 Saved {name} model to: {model_path}")

        # Evaluate and save metrics
        metrics = evaluate_model(name, model, X_test, y_test)
        all_metrics.append(metrics)

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_parquet(METRICS_DIR / f"{name}_metrics.parquet", index=False)
        print(f"📦 Saved metrics to: {METRICS_DIR}/{name}_metrics.parquet")


if __name__ == "__main__":
    train_all_attack_models()
