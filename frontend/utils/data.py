from __future__ import annotations

import json
import math

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


# =============================================================================
# Frontend limits
# =============================================================================


MAX_UPLOAD_ROWS = 100_000

SUPPORTED_UPLOAD_EXTENSIONS = (
    ".json",
    ".csv",
    ".parquet",
)


# =============================================================================
# Serialization
# =============================================================================


def serialize_value(
    value: Any,
) -> Any:
    """
    Convert a scalar value into a JSON-compatible representation.

    This function normalizes pandas, NumPy and datetime values while
    ensuring that missing or non-finite numerical values are represented
    as JSON null.
    """

    if value is None:
        return None

    # -------------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------------

    try:

        missing = pd.isna(
            value
        )

        if isinstance(
            missing,
            (bool, np.bool_),
        ) and bool(missing):

            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    # -------------------------------------------------------------------------
    # Datetime-like values
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
            date,
        ),
    ):

        return value.isoformat()

    # -------------------------------------------------------------------------
    # NumPy scalar values
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        np.generic,
    ):

        value = value.item()

    # -------------------------------------------------------------------------
    # Non-finite floating-point values
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        (float, np.floating),
    ):

        numeric_value = float(
            value
        )

        if not math.isfinite(
            numeric_value
        ):
            return None

        return numeric_value

    # -------------------------------------------------------------------------
    # Primitive JSON types
    # -------------------------------------------------------------------------

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    # -------------------------------------------------------------------------
    # Additional scalar-like objects
    # -------------------------------------------------------------------------

    if hasattr(
        value,
        "item",
    ):

        try:

            scalar = value.item()

            if scalar is not value:

                return serialize_value(
                    scalar
                )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            pass

    # -------------------------------------------------------------------------
    # Validate remaining object against JSON encoder
    # -------------------------------------------------------------------------

    try:

        json.dumps(
            value,
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:

        raise ValueError(
            (
                "Claim contains a value that cannot "
                "be serialized to JSON: "
                f"{type(value).__name__}."
            )
        ) from exc

    return value


def serialize_mapping(
    mapping: dict[Any, Any],
) -> dict[str, Any]:
    """
    Normalize one mapping into a JSON-safe claim record.
    """

    result: dict[
        str,
        Any,
    ] = {}

    for key, value in mapping.items():

        key_string = str(
            key
        )

        if key_string in result:

            raise ValueError(
                (
                    "Claim contains duplicate field names "
                    "after string normalization: "
                    f"{key_string!r}."
                )
            )

        result[
            key_string
        ] = serialize_value(
            value
        )

    return result


def serialize_row(
    row: pd.Series,
) -> dict[str, Any]:
    """
    Convert one pandas row into a JSON-safe claim dictionary.
    """

    return serialize_mapping(
        row.to_dict()
    )


def dataframe_to_records(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert a DataFrame into JSON-safe claim records.
    """

    if not isinstance(
        frame,
        pd.DataFrame,
    ):

        raise TypeError(
            "Expected a pandas DataFrame."
        )

    if frame.empty:
        return []

    # Duplicate DataFrame columns are dangerous because converting
    # them into dictionaries can silently discard information.
    duplicated_columns = (
        frame.columns[
            frame.columns.duplicated()
        ]
        .astype(str)
        .tolist()
    )

    if duplicated_columns:

        raise ValueError(
            (
                "Dataset contains duplicate columns: "
                + ", ".join(
                    sorted(
                        set(
                            duplicated_columns
                        )
                    )
                )
            )
        )

    if len(frame) > MAX_UPLOAD_ROWS:

        raise ValueError(
            (
                f"Dataset contains {len(frame):,} rows; "
                f"maximum supported frontend size is "
                f"{MAX_UPLOAD_ROWS:,}."
            )
        )

    return [
        serialize_mapping(
            record
        )
        for record
        in frame.to_dict(
            orient="records"
        )
    ]


# =============================================================================
# Demo dataset
# =============================================================================


@st.cache_data(
    show_spinner=False,
)
def _read_demo_dataset() -> pd.DataFrame:
    """
    Load and cache the bundled demonstration claim dataset.
    """

    if not DEMO_CLAIMS_PATH.exists():

        raise FileNotFoundError(
            (
                "Demo claims dataset not found: "
                f"{DEMO_CLAIMS_PATH}"
            )
        )

    if not DEMO_CLAIMS_PATH.is_file():

        raise FileNotFoundError(
            (
                "Demo claims path is not a file: "
                f"{DEMO_CLAIMS_PATH}"
            )
        )

    try:

        frame = pd.read_parquet(
            DEMO_CLAIMS_PATH
        )

    except Exception as exc:

        raise RuntimeError(
            (
                "Unable to load the demo claims dataset: "
                f"{exc}"
            )
        ) from exc

    if frame.empty:

        raise ValueError(
            "Demo claims dataset is empty."
        )

    if frame.columns.duplicated().any():

        duplicates = (
            frame.columns[
                frame.columns.duplicated()
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            (
                "Demo claims dataset contains "
                "duplicate columns: "
                + ", ".join(
                    sorted(
                        set(
                            duplicates
                        )
                    )
                )
            )
        )

    return frame


def load_demo_claims(
    limit: int | None = 500,
) -> pd.DataFrame:
    """
    Return a defensive copy of the demonstration dataset.

    Parameters
    ----------
    limit:
        Maximum number of rows to return.
        None returns the complete demo dataset.
    """

    frame = (
        _read_demo_dataset()
    )

    if limit is None:

        return (
            frame.copy(
                deep=True
            )
        )

    try:

        normalized_limit = int(
            limit
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "Demo dataset limit must be an integer."
        ) from exc

    if normalized_limit <= 0:

        return (
            frame.iloc[
                0:0
            ]
            .copy(
                deep=True
            )
        )

    return (
        frame
        .head(
            normalized_limit
        )
        .copy(
            deep=True
        )
    )


def get_demo_claim(
    index: int,
) -> dict[str, Any]:
    """
    Return one JSON-safe claim from the demo dataset.
    """

    claims = (
        _read_demo_dataset()
    )

    try:

        normalized_index = int(
            index
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "Demo claim index must be an integer."
        ) from exc

    if not (
        0
        <= normalized_index
        < len(claims)
    ):

        raise IndexError(
            (
                f"Demo claim index {normalized_index} "
                "is outside the dataset. "
                f"Valid range: 0–{len(claims) - 1}."
            )
        )

    return serialize_row(
        claims.iloc[
            normalized_index
        ]
    )


# =============================================================================
# Claim normalization
# =============================================================================


def normalize_uploaded_claims(
    data: Any,
) -> list[dict[str, Any]]:
    """
    Normalize supported uploaded JSON structures.

    Accepted structures
    -------------------
    1. One claim object:
       {"claim_id": "...", ...}

    2. List of claim objects:
       [{"claim_id": "..."}, ...]

    3. Portfolio envelope:
       {"claims": [{...}, {...}]}
    """

    if isinstance(
        data,
        dict,
    ):

        if "claims" in data:

            claims = data[
                "claims"
            ]

            if not isinstance(
                claims,
                list,
            ):

                raise ValueError(
                    (
                        "The 'claims' field must contain "
                        "a list of claim objects."
                    )
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
                "Input must contain one claim object, "
                "a list of claim objects, or an object "
                "with a 'claims' field."
            )
        )

    if not claims:

        raise ValueError(
            "The uploaded portfolio is empty."
        )

    claim_count = len(
        claims
    )

    if claim_count > MAX_UPLOAD_ROWS:

        raise ValueError(
            (
                f"Uploaded portfolio contains "
                f"{claim_count:,} rows; maximum supported "
                f"frontend upload is {MAX_UPLOAD_ROWS:,}."
            )
        )

    normalized: list[
        dict[str, Any]
    ] = []

    for index, claim in enumerate(
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

        if not claim:

            raise ValueError(
                (
                    f"Claim at position {index} "
                    "is empty."
                )
            )

        try:

            normalized_claim = (
                serialize_mapping(
                    claim
                )
            )

        except ValueError as exc:

            raise ValueError(
                (
                    f"Invalid claim at position "
                    f"{index}: {exc}"
                )
            ) from exc

        normalized.append(
            normalized_claim
        )

    return normalized


# =============================================================================
# Uploaded tabular dataset validation
# =============================================================================


def _validate_uploaded_frame(
    frame: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    """
    Validate an uploaded tabular dataset before serialization.
    """

    if not isinstance(
        frame,
        pd.DataFrame,
    ):

        raise ValueError(
            (
                f"Unable to construct a valid "
                f"{source} dataset."
            )
        )

    if frame.empty:

        raise ValueError(
            (
                f"Uploaded {source} "
                "contains no rows."
            )
        )

    if len(frame) > MAX_UPLOAD_ROWS:

        raise ValueError(
            (
                f"Uploaded {source} contains "
                f"{len(frame):,} rows; maximum supported "
                f"frontend upload is {MAX_UPLOAD_ROWS:,}."
            )
        )

    duplicated = (
        frame.columns[
            frame.columns.duplicated()
        ]
        .astype(str)
        .tolist()
    )

    if duplicated:

        raise ValueError(
            (
                f"Uploaded {source} contains duplicate "
                "columns: "
                + ", ".join(
                    sorted(
                        set(
                            duplicated
                        )
                    )
                )
            )
        )

    if len(
        frame.columns
    ) == 0:

        raise ValueError(
            (
                f"Uploaded {source} "
                "contains no columns."
            )
        )

    return frame


# =============================================================================
# Upload reader
# =============================================================================


def read_uploaded_file(
    uploaded: Any,
) -> list[dict[str, Any]]:
    """
    Read and normalize an uploaded claim or portfolio file.

    Supported formats:
        JSON
        CSV
        Parquet
    """

    if uploaded is None:

        raise ValueError(
            "No file was uploaded."
        )

    raw_name = getattr(
        uploaded,
        "name",
        None,
    )

    if not raw_name:

        raise ValueError(
            "Uploaded file has no filename."
        )

    filename = (
        str(
            raw_name
        )
        .lower()
        .strip()
    )

    suffix = (
        Path(
            filename
        )
        .suffix
        .lower()
    )

    if suffix not in (
        SUPPORTED_UPLOAD_EXTENSIONS
    ):

        raise ValueError(
            (
                "Unsupported file format. "
                "Use JSON, CSV or Parquet."
            )
        )

    # -------------------------------------------------------------------------
    # Ensure repeated Streamlit reads start from the beginning when possible.
    # -------------------------------------------------------------------------

    try:
        uploaded.seek(0)

    except (
        AttributeError,
        OSError,
    ):
        pass

    # =========================================================================
    # JSON
    # =========================================================================

    if suffix == ".json":

        try:

            data = json.load(
                uploaded
            )

        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:

            raise ValueError(
                (
                    "Uploaded JSON is invalid: "
                    f"{exc}"
                )
            ) from exc

        except Exception as exc:

            raise ValueError(
                (
                    "Unable to read JSON file: "
                    f"{exc}"
                )
            ) from exc

        return normalize_uploaded_claims(
            data
        )

    # =========================================================================
    # CSV
    # =========================================================================

    if suffix == ".csv":

        try:

            frame = pd.read_csv(
                uploaded
            )

        except pd.errors.EmptyDataError as exc:

            raise ValueError(
                "Uploaded CSV contains no data."
            ) from exc

        except pd.errors.ParserError as exc:

            raise ValueError(
                (
                    "Uploaded CSV could not be parsed: "
                    f"{exc}"
                )
            ) from exc

        except UnicodeDecodeError as exc:

            raise ValueError(
                (
                    "Uploaded CSV encoding could not "
                    "be decoded."
                )
            ) from exc

        except Exception as exc:

            raise ValueError(
                (
                    "Unable to read CSV file: "
                    f"{exc}"
                )
            ) from exc

        frame = _validate_uploaded_frame(
            frame,
            source="CSV",
        )

        return dataframe_to_records(
            frame
        )

    # =========================================================================
    # Parquet
    # =========================================================================

    if suffix == ".parquet":

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

        frame = _validate_uploaded_frame(
            frame,
            source="Parquet",
        )

        return dataframe_to_records(
            frame
        )

    # Defensive fallback. Normally unreachable because suffix was validated.
    raise ValueError(
        (
            "Unsupported file format. "
            "Use JSON, CSV or Parquet."
        )
    )