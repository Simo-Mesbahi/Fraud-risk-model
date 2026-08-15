from __future__ import annotations

import pandas as pd


def clean_claims(
    claims: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = claims.copy()

    invalid_mask = (
        (df["claim_amount"] <= 0)
        | (df["service_units"] < 1)
        | (
            df["service_date"]
            > df["claim_submission_date"]
        )
    )

    rejected = df.loc[invalid_mask].copy()

    clean = df.loc[~invalid_mask].copy()

    clean = clean.reset_index(drop=True)
    rejected = rejected.reset_index(drop=True)

    return clean, rejected