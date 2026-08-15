from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_synthetic_tables(
    data_dir: str | Path = "data/synthetic",
) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)

    return {
        "customers": pd.read_parquet(data_dir / "customers.parquet"),
        "providers": pd.read_parquet(data_dir / "providers.parquet"),
        "policies": pd.read_parquet(data_dir / "policies.parquet"),
        "claims": pd.read_parquet(data_dir / "claims.parquet"),
    }


def save_interim_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path = "data/interim",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, dataframe in tables.items():
        dataframe.to_parquet(
            output_dir / f"{name}.parquet",
            index=False,
        )