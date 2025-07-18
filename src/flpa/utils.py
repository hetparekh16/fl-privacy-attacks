import pandas as pd
from pathlib import Path
from typing import List, Dict, Union


def save_eval_round(
    round_id: int,
    client_rows: List[Dict[str, Union[str, int, float, list]]],
    agg_row: Dict[str, Union[str, int, float]],
    output_dir: Union[str, Path] = "outputs",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    client_logs_dir = output_dir / "client_logs"
    client_logs_dir.mkdir(exist_ok=True)

    agg_logs_dir = output_dir / "aggregated_logs"
    agg_logs_dir.mkdir(exist_ok=True)

    client_df = pd.DataFrame(client_rows)
    agg_df = pd.DataFrame([agg_row])

    # Fixed paths without leading slashes
    client_df.to_parquet(client_logs_dir / f"round_{round_id}_client_eval.parquet")
    agg_df.to_parquet(agg_logs_dir / f"round_{round_id}_agg_eval.parquet")

    print(f"Saved round {round_id} evaluation metrics to {output_dir}")


def save_train_round(
    round_id,
    client_id,
    sample_ids,
    output_dir="outputs",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create sample_id_logs directory if it doesn't exist
    sample_id_logs_dir = output_dir / "sample_id_logs"
    sample_id_logs_dir.mkdir(exist_ok=True)

    sample_ids_list = [int(x) for x in sample_ids.split(",")]

    print(
        f"Client {client_id} used the sample_ids: {len(sample_ids_list)} for training after converting it into a list in utils.py"
    )

    row = {
        "round_id": round_id,
        "client_id": client_id,
        "sample_ids": sample_ids_list,
    }
    df = pd.DataFrame([row])

    # Fixed path without leading slash
    parquet_path = sample_id_logs_dir / f"round_{round_id}_client_train.parquet"

    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_parquet(parquet_path)
    print(f"Saved round {round_id} training metadata to {output_dir}")


def clear_output_directory(output_dir: Union[str, Path] = "outputs"):
    """Clear all contents of the output directory for a fresh simulation run."""
    output_dir = Path(output_dir)
    import shutil

    if output_dir.exists():
        print(f"Cleaning output directory: {output_dir}")
        # Remove all contents recursively
        for item in output_dir.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # Create necessary subdirectories
    (output_dir / "client_logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregated_logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_id_logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "global_model").mkdir(parents=True, exist_ok=True)
    (output_dir / "attacks").mkdir(parents=True, exist_ok=True)
    (output_dir / "attacks/models").mkdir(parents=True, exist_ok=True)
    (output_dir / "attacks/metrics").mkdir(parents=True, exist_ok=True)

    print(f"Output directory is ready for fresh metrics")
