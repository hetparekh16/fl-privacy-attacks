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

    print(f"📦 Saved round {round_id} evaluation metrics to {output_dir}")


def save_train_round(
    round_id,
    client_id,
    sample_ids,
    output_dir="outputs",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # TODO: Think about some way so that we can save all the information of each round in a df before we dump it into a parquet file

    # Create sample_id_logs directory if it doesn't exist
    sample_id_logs_dir = output_dir / "sample_id_logs"
    sample_id_logs_dir.mkdir(exist_ok=True)

    print(
        f"Client {client_id} used the sample_ids: {len(sample_ids)} for training before converting it into a list in utils.py"
    )

    sample_ids_list = [int(x) for x in sample_ids.split(",")]

    print(
        f"Client {client_id} used the sample_ids: {len(sample_ids)} for training after converting it into a list in utils.py"
    )

    row = {
        "round_id": round_id,
        "client_id": client_id,
        "sample_ids": sample_ids_list,
    }
    df = pd.DataFrame([row])

    print(
        f"The datatype of sample_ids is of type {type(sample_ids_list)} and has length {len(sample_ids_list)}"
    )

    # Fixed path without leading slash
    parquet_path = sample_id_logs_dir / f"round_{round_id}_client_train.parquet"

    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_parquet(parquet_path)
    print(f"📦 Saved round {round_id} training metadata to {output_dir}")
