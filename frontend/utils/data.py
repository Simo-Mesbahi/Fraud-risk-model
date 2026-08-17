from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DEMO_CLAIMS_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "claims.parquet"
)


def serialize_value(
    value: Any,
):
    if pd.isna(value):
        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()

        except Exception:
            pass

    return value


def serialize_row(
    row: pd.Series,
) -> dict:
    return {
        column: serialize_value(value)
        for column, value in row.items()
    }


@st.cache_data(
    show_spinner=False
)
def load_demo_claims(
    limit: int = 500,
) -> pd.DataFrame:
    if not DEMO_CLAIMS_PATH.exists():
        raise FileNotFoundError(
            f"Demo claims not found: "
            f"{DEMO_CLAIMS_PATH}"
        )

    return (
        pd.read_parquet(
            DEMO_CLAIMS_PATH
        )
        .head(limit)
        .copy()
    )


def get_demo_claim(
    index: int,
) -> dict:
    claims = load_demo_claims()

    row = claims.iloc[
        int(index)
    ]

    return serialize_row(
        row
    )


def normalize_uploaded_claims(
    data,
) -> list[dict]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "claims" in data:
            claims = data["claims"]

            if not isinstance(
                claims,
                list,
            ):
                raise ValueError(
                    "'claims' must contain a list."
                )

            return claims

        return [data]

    raise ValueError(
        "Input must contain one claim "
        "or a list of claims."
    )


def read_uploaded_file(
    uploaded,
) -> list[dict]:
    filename = (
        uploaded.name.lower()
    )

    if filename.endswith(
        ".json"
    ):
        data = json.load(
            uploaded
        )

        return normalize_uploaded_claims(
            data
        )

    if filename.endswith(
        ".csv"
    ):
        dataframe = (
            pd.read_csv(
                uploaded
            )
        )

        return dataframe.to_dict(
            orient="records"
        )

    if filename.endswith(
        ".parquet"
    ):
        dataframe = (
            pd.read_parquet(
                uploaded
            )
        )

        return [
            serialize_row(row)
            for _, row in dataframe.iterrows()
        ]

    raise ValueError(
        "Supported formats: JSON, CSV, Parquet."
    )