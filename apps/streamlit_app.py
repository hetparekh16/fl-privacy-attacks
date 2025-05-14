import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import ast

# Paths
CLIENT_LOGS_DIR = Path("outputs/client_logs")
AGG_LOGS_DIR = Path("outputs/aggregated_logs")
METRICS = ["accuracy", "precision", "recall", "f1", "loss"]

st.set_page_config(page_title="FL Metrics Dashboard", layout="wide")
st.title("📊 Federated Learning Metrics Dashboard")


# 📈 Custom chart function
def plot_line_with_axes(df, metric, title):
    df = df.copy()

    # Recover round_id from index if missing
    if "round_id" not in df.columns:
        df["round_id"] = df.index

    df["round_id"] = pd.to_numeric(df["round_id"], errors="coerce")
    df = df.dropna(subset=["round_id", metric])
    df["round_id"] = df["round_id"].astype(int)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if metric == "loss":
        y_scale = alt.Scale(domain=[0, 3])
        y_title = "Loss"
    else:
        df[metric] = df[metric] * 100  # Convert to %
        y_scale = alt.Scale(domain=[10, 100])
        y_title = f"{title} (%)"

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "round_id:Q",
                title="Round ID",
                scale=alt.Scale(domain=[1, df["round_id"].max()]),
            ),
            y=alt.Y(metric, title=y_title, scale=y_scale),
        )
        .properties(width=400, height=300, title=title)
    )

    return chart


# ========== CLIENT SECTION ==========
client_dfs = []
for file in sorted(CLIENT_LOGS_DIR.glob("round_*_client_eval.parquet")):
    df = pd.read_parquet(file)
    if "round_id" in df.columns and "client_id" in df.columns:
        client_dfs.append(df)
    else:
        st.warning(f"⚠️ Skipped {file.name} — missing required columns.")


if client_dfs:
    client_df = pd.concat(client_dfs).reset_index(drop=True)
    st.header("👤 Per-Client Metrics")

    client_ids = client_df["client_id"].unique().tolist()
    selected_client = st.selectbox("Select Client ID", client_ids)

    client_data = client_df[client_df["client_id"] == selected_client].sort_values(
        "round_id"
    )
    client_data = client_data.set_index("round_id")

    col1, col2 = st.columns(2)
    col1.altair_chart(
        plot_line_with_axes(client_data, "accuracy", "Accuracy Over Rounds")
    )
    col2.altair_chart(
        plot_line_with_axes(client_data, "precision", "Precision Over Rounds")
    )

    col3, col4 = st.columns(2)
    col3.altair_chart(plot_line_with_axes(client_data, "recall", "Recall Over Rounds"))
    col4.altair_chart(plot_line_with_axes(client_data, "f1", "F1 Score Over Rounds"))

    st.altair_chart(plot_line_with_axes(client_data, "loss", "Loss Over Rounds"))

else:
    st.warning("⚠️ No client evaluation logs found.")

# ========== AGGREGATED SECTION ==========
agg_dfs = []
for file in sorted(AGG_LOGS_DIR.glob("round_*_agg_eval.parquet")):
    df = pd.read_parquet(file)
    if "round_id" in df.columns:
        agg_dfs.append(df)
    else:
        st.warning(f"⚠️ Skipped {file.name} — missing 'round_id' column.")


if agg_dfs:
    agg_df = pd.concat(agg_dfs).reset_index(drop=True).sort_values("round_id")
    agg_df = agg_df.set_index("round_id")

    st.header("📊 Aggregated Metrics Over Rounds")

    col1, col2 = st.columns(2)
    col1.altair_chart(plot_line_with_axes(agg_df, "accuracy", "Accuracy Over Rounds"))
    col2.altair_chart(plot_line_with_axes(agg_df, "precision", "Precision Over Rounds"))

    col3, col4 = st.columns(2)
    col3.altair_chart(plot_line_with_axes(agg_df, "recall", "Recall Over Rounds"))
    col4.altair_chart(plot_line_with_axes(agg_df, "f1", "F1 Score Over Rounds"))

    st.altair_chart(plot_line_with_axes(agg_df, "loss", "Loss Over Rounds"))

else:
    st.warning("⚠️ No aggregated evaluation logs found.")


# ========== ATTACK METRICS SECTION ==========


st.header("🧠 Membership Inference Attack Evaluation")

attack_metrics_dir = Path("outputs/attacks/metrics")
available_models = [
    f.stem.replace("_metrics", "") for f in attack_metrics_dir.glob("*_metrics.parquet")
]
model_choice = st.selectbox("Select Attack Model", available_models)

metric_path = attack_metrics_dir / f"{model_choice}_metrics.parquet"
if metric_path.exists():
    df = pd.read_parquet(metric_path)

    # Metrics
    st.subheader("📋 Metrics Summary")
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    st.dataframe(df[metric_cols].round(4))

    # Confusion Matrix
    st.subheader("🧮 Confusion Matrix")

    cm_data = [
        [df["true_neg"][0], df["false_pos"][0]],
        [df["false_neg"][0], df["true_pos"][0]],
    ]
    cm_df = pd.DataFrame(
        cm_data,
        columns=["Predicted: 0", "Predicted: 1"],
        index=["Actual: 0", "Actual: 1"],
    )

    # Centered HTML table
    st.markdown(
        cm_df.style.set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]},
            ]
        )
        .set_properties(**{"text-align": "center"})  # type: ignore
        .to_html(),
        unsafe_allow_html=True,
    )

    # ROC Curve
    st.subheader("📈 ROC Curve")

    # Decode FPR/TPR lists
    try:
        fpr = (
            ast.literal_eval(df["fpr"][0])
            if isinstance(df["fpr"][0], str)
            else df["fpr"][0]
        )
        tpr = (
            ast.literal_eval(df["tpr"][0])
            if isinstance(df["tpr"][0], str)
            else df["tpr"][0]
        )

        roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        chart = (
            alt.Chart(roc_df)
            .mark_line()
            .encode(x="FPR", y="TPR")
            .properties(width=400, height=300, title="ROC Curve")
        )
        st.altair_chart(chart)
    except Exception as e:
        st.error(f"❌ Could not parse FPR/TPR: {e}")
else:
    st.warning("⚠️ Metrics file not found.")
