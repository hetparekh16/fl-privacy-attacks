import polars as pl
from pathlib import Path
from typing import List, Dict, Union


def save_eval_round(
    round_id: int,
    client_rows: List[Dict[str, Union[str, int, float]]],
    agg_row: Dict[str, Union[str, int, float]],
    output_dir: Union[str, Path] = "data",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    client_df = pl.DataFrame(client_rows)
    agg_df = pl.DataFrame([agg_row])

    client_df.write_parquet(output_dir / f"round_{round_id}_client_eval.parquet")
    agg_df.write_parquet(output_dir / f"round_{round_id}_agg_eval.parquet")

    print(f"📦 Saved round {round_id} evaluation metrics to {output_dir}")
