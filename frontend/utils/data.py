from __future__ import annotations

import json

from datetime import (
    date,
    datetime,
)

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Paths
# =============================================================================


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


MAX_UPLOAD_ROWS = 100_000


# =============================================================================
# Serialization
# =============================================================================


def serialize_value(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    try:

        if pd.isna(
            value
        ):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

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
) -> dict[str, Any]:

    return {
        str(
            column
        ):
            serialize_value(
                value
            )

        for (
            column,
            value,
        ) in row.items()
    }


def dataframe_to_records(
    frame: pd.DataFrame,
) -> list[
    dict[str, Any]
]:

    if frame.empty:
        return []

    return [
        serialize_row(
            row
        )

        for _, row
        in frame.iterrows()
    ]


# =============================================================================
# Demo dataset
# =============================================================================


@st.cache_data(
    show_spinner=False,
)
def _read_demo_dataset() -> pd.DataFrame:

    if not DEMO_CLAIMS_PATH.exists():

        raise FileNotFoundError(
            (
                "Demo claims dataset not found: "
                f"{DEMO_CLAIMS_PATH}"
            )
        )

    frame = pd.read_parquet(
        DEMO_CLAIMS_PATH
    )

    if frame.empty:

        raise ValueError(
            "Demo claims dataset is empty."
        )

    return frame


def load_demo_claims(
    limit: int | None = 500,
) -> pd.DataFrame:

    frame = (
        _read_demo_dataset()
    )

    if limit is None:

        return (
            frame.copy()
        )

    limit = int(
        limit
    )

    if limit <= 0:

        return (
            frame.iloc[
                0:0
            ]
            .copy()
        )

    return (
        frame
        .head(
            limit
        )
        .copy()
    )


def get_demo_claim(
    index: int,
) -> dict[str, Any]:

    claims = (
        load_demo_claims(
            limit=None
        )
    )

    index = int(
        index
    )

    if not (
        0
        <= index
        < len(
            claims
        )
    ):

        raise IndexError(
            (
                f"Demo claim index {index} "
                "is outside the dataset."
            )
        )

    return (
        serialize_row(
            claims.iloc[
                index
            ]
        )
    )


# =============================================================================
# Upload normalization
# =============================================================================


def normalize_uploaded_claims(
    data: Any,
) -> list[
    dict[str, Any]
]:

    if isinstance(
        data,
        dict,
    ):

        if (
            "claims"
            in data
        ):

            claims = (
                data[
                    "claims"
                ]
            )

        else:

            claims = [
                data
            ]

    elif isinstance(
        data,
        list,
    ):

        claims = data

    else:

        raise ValueError(
            (
                "Input must contain one claim, "
                "a list of claims, or an object "
                "with a 'claims' field."
            )
        )

    if not claims:

        raise ValueError(
            "The uploaded portfolio is empty."
        )

    if (
        len(
            claims
        )
        > MAX_UPLOAD_ROWS
    ):

        raise ValueError(
            (
                f"Uploaded portfolio contains "
                f"{len(claims):,} rows; maximum supported "
                f"frontend upload is {MAX_UPLOAD_ROWS:,}."
            )
        )

    normalized: list[
        dict[str, Any]
    ] = []

    for (
        index,
        claim,
    ) in enumerate(
        claims
    ):

        if not isinstance(
            claim,
            dict,
        ):

            raise ValueError(
                (
                    f"Claim at position {index} "
                    "is not a JSON object."
                )
            )

        normalized.append(
            {
                str(
                    key
                ):
                    serialize_value(
                        value
                    )

                for (
                    key,
                    value,
                ) in claim.items()
            }
        )

    return normalized


# =============================================================================
# Upload reader
# =============================================================================


def read_uploaded_file(
    uploaded,
) -> list[
    dict[str, Any]
]:

    if uploaded is None:

        raise ValueError(
            "No file was uploaded."
        )

    filename = (
        str(
            uploaded.name
        )
        .lower()
        .strip()
    )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    if filename.endswith(
        ".json"
    ):

        try:

            data = json.load(
                uploaded
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                (
                    "Uploaded JSON is invalid: "
                    f"{exc}"
                )
            ) from exc

        return (
            normalize_uploaded_claims(
                data
            )
        )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    if filename.endswith(
        ".csv"
    ):

        try:

            frame = pd.read_csv(
                uploaded
            )

        except Exception as exc:

            raise ValueError(
                (
                    "Unable to read CSV file: "
                    f"{exc}"
                )
            ) from exc

        if frame.empty:

            raise ValueError(
                "Uploaded CSV contains no rows."
            )

        return (
            normalize_uploaded_claims(
                dataframe_to_records(
                    frame
                )
            )
        )

    # -------------------------------------------------------------------------
    # Parquet
    # -------------------------------------------------------------------------

    if filename.endswith(
        ".parquet"
    ):

        try:

            frame = pd.read_parquet(
                uploaded
            )

        except Exception as exc:

            raise ValueError(
                (
                    "Unable to read Parquet file: "
                    f"{exc}"
                )
            ) from exc

        if frame.empty:

            raise ValueError(
                "Uploaded Parquet contains no rows."
            )

        return (
            normalize_uploaded_claims(
                dataframe_to_records(
                    frame
                )
            )
        )

    raise ValueError(
        (
            "Unsupported file format. "
            "Use JSON, CSV or Parquet."
        )
    )