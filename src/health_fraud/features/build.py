from __future__ import annotations

import numpy as np
import pandas as pd


ENGINEERED_FEATURES = [
    "has_prescription_missing",
    "document_count_missing",
    "provider_missing",
    "submission_hour",
    "submission_dayofweek",
    "submission_month",
    "submission_is_weekend",
    "service_dayofweek",
    "service_month",
    "service_is_weekend",
    "requested_to_limit_ratio",
    "amount_above_service_typical",
    "recent_claim_share_30d_365d",
    "recent_amount_share_30d_365d",
    "provider_recent_activity_ratio",
    "customer_provider_intensity",
    "same_service_intensity",
    "days_since_policy_change_missing",
]


def build_model_features(
    claims: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build deterministic model features required by the frozen fraud model.

    This function must reproduce exactly the feature engineering used during
    model training and evaluation.

    It does not mutate the input dataframe.
    """

    df = claims.copy()

    required_columns = {
        "provider_id",
        "has_prescription",
        "document_count",
        "claim_submission_timestamp",
        "service_date",
        "requested_reimbursement",
        "coverage_limit",
        "claim_amount",
        "service_typical_amount",
        "customer_claims_30d",
        "customer_claims_365d",
        "customer_amount_30d",
        "customer_amount_365d",
        "provider_claims_30d",
        "provider_claims_90d",
        "customer_provider_claims_30d",
        "same_service_claims_30d",
        "days_since_policy_change",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Cannot build model features. "
            "Missing required source columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # ------------------------------------------------------------
    # Datetime validation / normalization
    # ------------------------------------------------------------

    datetime_columns = [
        "claim_submission_timestamp",
        "service_date",
    ]

    for column in datetime_columns:
        if not pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            df[column] = pd.to_datetime(
                df[column],
                errors="raise",
            )

    # ------------------------------------------------------------
    # Missingness indicators
    # ------------------------------------------------------------

    df["has_prescription_missing"] = (
        df["has_prescription"]
        .isna()
        .astype("int8")
    )

    df["document_count_missing"] = (
        df["document_count"]
        .isna()
        .astype("int8")
    )

    df["provider_missing"] = (
        df["provider_id"]
        .isna()
        .astype("int8")
    )

    # ------------------------------------------------------------
    # Submission calendar features
    # ------------------------------------------------------------

    df["submission_hour"] = (
        df[
            "claim_submission_timestamp"
        ]
        .dt.hour
    )

    df["submission_dayofweek"] = (
        df[
            "claim_submission_timestamp"
        ]
        .dt.dayofweek
    )

    df["submission_month"] = (
        df[
            "claim_submission_timestamp"
        ]
        .dt.month
    )

    df["submission_is_weekend"] = (
        df[
            "submission_dayofweek"
        ]
        >= 5
    ).astype("int8")

    # ------------------------------------------------------------
    # Service calendar features
    # ------------------------------------------------------------

    df["service_dayofweek"] = (
        df[
            "service_date"
        ]
        .dt.dayofweek
    )

    df["service_month"] = (
        df[
            "service_date"
        ]
        .dt.month
    )

    df["service_is_weekend"] = (
        df[
            "service_dayofweek"
        ]
        >= 5
    ).astype("int8")

    # ------------------------------------------------------------
    # Business ratios
    # ------------------------------------------------------------

    eps = 1e-6

    df[
        "requested_to_limit_ratio"
    ] = (
        df[
            "requested_reimbursement"
        ]
        / df[
            "coverage_limit"
        ].clip(
            lower=eps
        )
    )

    df[
        "amount_above_service_typical"
    ] = (
        df[
            "claim_amount"
        ]
        - df[
            "service_typical_amount"
        ]
    )

    # ------------------------------------------------------------
    # Customer activity ratios
    # ------------------------------------------------------------

    df[
        "recent_claim_share_30d_365d"
    ] = (
        df[
            "customer_claims_30d"
        ]
        / df[
            "customer_claims_365d"
        ].replace(
            0,
            np.nan,
        )
    )

    df[
        "recent_amount_share_30d_365d"
    ] = (
        df[
            "customer_amount_30d"
        ]
        / df[
            "customer_amount_365d"
        ].replace(
            0,
            np.nan,
        )
    )

    # ------------------------------------------------------------
    # Provider activity
    # ------------------------------------------------------------

    df[
        "provider_recent_activity_ratio"
    ] = (
        (
            df[
                "provider_claims_30d"
            ]
            + 1
        )
        /
        (
            df[
                "provider_claims_90d"
            ]
            / 3
            + 1
        )
    )

    # ------------------------------------------------------------
    # Relationship / repetition intensity
    # ------------------------------------------------------------

    df[
        "customer_provider_intensity"
    ] = (
        df[
            "customer_provider_claims_30d"
        ]
        / (
            df[
                "customer_claims_30d"
            ]
            + 1
        )
    )

    df[
        "same_service_intensity"
    ] = (
        df[
            "same_service_claims_30d"
        ]
        / (
            df[
                "customer_claims_30d"
            ]
            + 1
        )
    )

    # ------------------------------------------------------------
    # Policy-change missingness
    # ------------------------------------------------------------

    df[
        "days_since_policy_change_missing"
    ] = (
        df[
            "days_since_policy_change"
        ]
        .isna()
        .astype(
            "int8"
        )
    )

    # ------------------------------------------------------------
    # Final integrity check
    # ------------------------------------------------------------

    missing_engineered = [
        feature
        for feature
        in ENGINEERED_FEATURES
        if feature not in df.columns
    ]

    if missing_engineered:
        raise RuntimeError(
            "Feature engineering failed. "
            "Missing engineered features: "
            + ", ".join(
                missing_engineered
            )
        )

    return df