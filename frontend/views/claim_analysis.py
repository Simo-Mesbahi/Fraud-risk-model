from __future__ import annotations

import json
import uuid

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components import (
    human_review_notice,
    info_panel,
    metric_card,
    risk_badge,
    risk_gauge,
    section_header,
)

from utils.data import (
    get_demo_claim,
    load_demo_claims,
)

from utils.formatting import (
    risk_tier,
)


# =============================================================================
# Configuration
# =============================================================================


MAX_CONTEXT_ROWS = 100_000


LEAKAGE_COLUMNS = {
    "is_fraud",
    "latent_fraud_score",
    "synthetic_fraud_probability",
    "fraud_mechanism",
    "fraud_difficulty",
    "legitimate_anomaly",
    "legitimate_anomaly_type",
}


REQUIRED_CONTEXT_COLUMNS = {
    "claim_id",
    "customer_id",
    "provider_id",
    "service_category",
    "service_code",
    "claim_amount",
    "claim_submission_timestamp",
}


SMART_WIDGET_KEYS = {
    "smart_customer_id",
    "smart_provider_id",
    "smart_service_category",
    "smart_service_code",
    "smart_claim_amount",
    "smart_requested_reimbursement",
    "smart_service_date",
    "smart_submission_date",
    "smart_service_units",
    "smart_submission_channel",
    "smart_document_count",
    "smart_has_invoice",
    "smart_prescription",
    "smart_submission_hour",
    "demo_claim_selector",
    "advanced_json_payload",
}


# =============================================================================
# Generic helpers
# =============================================================================


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a finite float.
    """

    try:
        result = float(
            value
        )

        if np.isfinite(
            result
        ):
            return result

    except (
        TypeError,
        ValueError,
    ):
        pass

    return float(
        default
    )


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    Convert numeric-like values to integer safely.
    """

    try:

        if pd.isna(
            value
        ):
            return default

        return int(
            round(
                float(
                    value
                )
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_optional_float(
    value: Any,
) -> float | None:
    """
    Convert a value to float while preserving missing values.
    """

    try:

        if pd.isna(
            value
        ):
            return None

        result = float(
            value
        )

        if np.isfinite(
            result
        ):
            return result

    except (
        TypeError,
        ValueError,
    ):
        pass

    return None


def _safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
    """
    Convert heterogeneous boolean representations safely.
    """

    if value is None:
        return default

    try:

        if pd.isna(
            value
        ):
            return default

    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(
        value,
        str,
    ):

        return (
            value
            .strip()
            .lower()
            in {
                "true",
                "1",
                "yes",
                "y",
            }
        )

    return bool(
        value
    )


def _json_safe(
    value: Any,
) -> Any:
    """
    Convert pandas / numpy / datetime values into JSON-safe values.
    """

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

    if hasattr(
        value,
        "item",
    ):

        try:
            return value.item()

        except Exception:
            pass

    return value


def _new_claim_id() -> str:
    """
    Generate an identifier for a new live inference request.
    """

    return (
        "LIVE_"
        + uuid.uuid4()
        .hex[:12]
        .upper()
    )


def _strip_leakage(
    claim: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove target/generation variables that must never enter inference.
    """

    return {
        key:
            _json_safe(
                value
            )

        for (
            key,
            value,
        ) in claim.items()

        if key not in LEAKAGE_COLUMNS
    }


def _format_identifier(
    value: Any,
) -> str:
    """
    Normalize identifiers for display.
    """

    if value is None:
        return "—"

    value = str(
        value
    ).strip()

    return (
        value
        if value
        else "—"
    )


# =============================================================================
# Historical dataset
# =============================================================================


@st.cache_data(
    show_spinner=False,
)
def _load_context_data() -> pd.DataFrame:
    """
    Load and normalize the historical claim context.
    """

    frame = (
        load_demo_claims(
            limit=MAX_CONTEXT_ROWS
        )
        .copy()
    )

    if frame.empty:

        raise ValueError(
            (
                "No historical claim context "
                "is available."
            )
        )

    missing = (
        REQUIRED_CONTEXT_COLUMNS
        - set(
            frame.columns
        )
    )

    if missing:

        raise ValueError(
            (
                "Historical dataset is missing: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )
        )

    frame[
        "claim_submission_timestamp"
    ] = pd.to_datetime(
        frame[
            "claim_submission_timestamp"
        ],
        errors="coerce",
    )

    if (
        "service_date"
        in frame.columns
    ):

        frame[
            "service_date"
        ] = pd.to_datetime(
            frame[
                "service_date"
            ],
            errors="coerce",
        )

    frame = (
        frame
        .dropna(
            subset=[
                "claim_submission_timestamp"
            ]
        )
        .sort_values(
            "claim_submission_timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    if frame.empty:

        raise ValueError(
            (
                "Historical context contains no "
                "valid submission timestamps."
            )
        )

    return frame


# =============================================================================
# Historical feature helpers
# =============================================================================


def _history_before(
    claims: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """
    Restrict historical context strictly to information available
    before the current claim.
    """

    return (
        claims.loc[
            claims[
                "claim_submission_timestamp"
            ]
            < timestamp
        ]
        .copy()
    )


def _latest_row(
    frame: pd.DataFrame,
) -> pd.Series | None:
    """
    Return the most recent historical row.
    """

    if frame.empty:
        return None

    return (
        frame
        .sort_values(
            "claim_submission_timestamp"
        )
        .iloc[-1]
    )


def _window(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    days: int,
) -> pd.DataFrame:
    """
    Restrict data to a trailing historical window.
    """

    if frame.empty:
        return frame

    lower = (
        timestamp
        - pd.Timedelta(
            days=days
        )
    )

    mask = (
        (
            frame[
                "claim_submission_timestamp"
            ]
            >= lower
        )
        &
        (
            frame[
                "claim_submission_timestamp"
            ]
            < timestamp
        )
    )

    return (
        frame.loc[
            mask
        ]
    )


def _amount_values(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Return valid numerical claim amounts.
    """

    if (
        frame.empty
        or "claim_amount"
        not in frame.columns
    ):

        return pd.Series(
            dtype=float
        )

    return (
        pd.to_numeric(
            frame[
                "claim_amount"
            ],
            errors="coerce",
        )
        .dropna()
    )


def _amount_sum(
    frame: pd.DataFrame,
) -> float:
    """
    Sum claim amounts safely.
    """

    values = (
        _amount_values(
            frame
        )
    )

    if values.empty:
        return 0.0

    return float(
        values.sum()
    )


def _amount_mean(
    frame: pd.DataFrame,
    default: float,
) -> float:
    """
    Compute historical mean amount with safe fallback.
    """

    values = (
        _amount_values(
            frame
        )
    )

    if values.empty:
        return float(
            default
        )

    return float(
        values.mean()
    )


def _amount_median(
    frame: pd.DataFrame,
    default: float,
) -> float:
    """
    Compute historical median amount with safe fallback.
    """

    values = (
        _amount_values(
            frame
        )
    )

    if values.empty:
        return float(
            default
        )

    return float(
        values.median()
    )


def _days_since_last(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> float | None:
    """
    Calculate elapsed days since the latest previous claim.
    """

    if frame.empty:
        return None

    dates = (
        frame[
            "claim_submission_timestamp"
        ]
        .dropna()
    )

    dates = (
        dates[
            dates
            < timestamp
        ]
    )

    if dates.empty:
        return None

    delta = (
        timestamp
        - dates.max()
    )

    return max(
        0.0,
        float(
            delta.total_seconds()
            / 86_400
        ),
    )


# =============================================================================
# Smart input suggestions
# =============================================================================


def _service_reference(
    history: pd.DataFrame,
    service_code: str,
) -> tuple[
    float,
    float,
]:
    """
    Estimate historically typical amount and reimbursement for a service.
    """

    subset = (
        history.loc[
            history[
                "service_code"
            ]
            .astype(str)
            == str(
                service_code
            )
        ]
        .copy()
    )

    typical_amount = (
        _amount_median(
            subset,
            default=250.0,
        )
    )

    if (
        "requested_reimbursement"
        in subset.columns
    ):

        requested = (
            pd.to_numeric(
                subset[
                    "requested_reimbursement"
                ],
                errors="coerce",
            )
            .dropna()
        )

        if not requested.empty:

            typical_requested = (
                float(
                    requested.median()
                )
            )

        else:

            typical_requested = (
                typical_amount
                * 0.8
            )

    else:

        typical_requested = (
            typical_amount
            * 0.8
        )

    return (
        max(
            0.01,
            typical_amount,
        ),
        max(
            0.0,
            typical_requested,
        ),
    )


# =============================================================================
# Context enrichment
# =============================================================================


def _build_smart_claim(
    *,
    history: pd.DataFrame,
    customer_id: str,
    provider_id: str,
    service_category: str,
    service_code: str,
    service_units: int,
    service_date: date,
    submission_datetime: datetime,
    claim_amount: float,
    requested_reimbursement: float,
    submission_channel: str,
    document_count: int,
    has_invoice: bool,
    has_prescription: bool | None,
) -> dict[str, Any]:
    """
    Enrich a minimal operational claim with historical model features.

    Only information available before submission_datetime is used.
    """

    submission_ts = (
        pd.Timestamp(
            submission_datetime
        )
    )

    service_ts = (
        pd.Timestamp(
            service_date
        )
    )

    if (
        service_ts.normalize()
        > submission_ts.normalize()
    ):

        raise ValueError(
            (
                "Service date cannot occur after "
                "the claim submission date."
            )
        )

    if claim_amount <= 0:

        raise ValueError(
            "Claim amount must be positive."
        )

    if requested_reimbursement < 0:

        raise ValueError(
            (
                "Requested reimbursement "
                "cannot be negative."
            )
        )

    historical = (
        _history_before(
            history,
            submission_ts,
        )
    )

    customer_history = (
        historical.loc[
            historical[
                "customer_id"
            ]
            .astype(str)
            == str(
                customer_id
            )
        ]
        .copy()
    )

    provider_history = (
        historical.loc[
            historical[
                "provider_id"
            ]
            .astype(str)
            == str(
                provider_id
            )
        ]
        .copy()
    )

    relationship_history = (
        customer_history.loc[
            customer_history[
                "provider_id"
            ]
            .astype(str)
            == str(
                provider_id
            )
        ]
        .copy()
    )

    same_service_history = (
        customer_history.loc[
            customer_history[
                "service_code"
            ]
            .astype(str)
            == str(
                service_code
            )
        ]
        .copy()
    )

    service_history = (
        historical.loc[
            historical[
                "service_code"
            ]
            .astype(str)
            == str(
                service_code
            )
        ]
        .copy()
    )

    customer_latest = (
        _latest_row(
            customer_history
        )
    )

    provider_latest = (
        _latest_row(
            provider_history
        )
    )

    if customer_latest is None:

        raise ValueError(
            (
                "No historical context exists for "
                f"customer {customer_id} before "
                "the selected submission date."
            )
        )

    if provider_latest is None:

        raise ValueError(
            (
                "No historical context exists for "
                f"provider {provider_id} before "
                "the selected submission date."
            )
        )

    # -------------------------------------------------------------------------
    # Historical windows
    # -------------------------------------------------------------------------

    customer_7d = (
        _window(
            customer_history,
            submission_ts,
            7,
        )
    )

    customer_30d = (
        _window(
            customer_history,
            submission_ts,
            30,
        )
    )

    customer_90d = (
        _window(
            customer_history,
            submission_ts,
            90,
        )
    )

    customer_365d = (
        _window(
            customer_history,
            submission_ts,
            365,
        )
    )

    provider_30d = (
        _window(
            provider_history,
            submission_ts,
            30,
        )
    )

    provider_90d = (
        _window(
            provider_history,
            submission_ts,
            90,
        )
    )

    relationship_30d = (
        _window(
            relationship_history,
            submission_ts,
            30,
        )
    )

    same_service_30d = (
        _window(
            same_service_history,
            submission_ts,
            30,
        )
    )

    # -------------------------------------------------------------------------
    # Monetary references
    # -------------------------------------------------------------------------

    service_typical_amount = (
        _amount_median(
            service_history,
            default=claim_amount,
        )
    )

    customer_avg_amount = (
        _amount_mean(
            customer_365d,
            default=claim_amount,
        )
    )

    provider_avg_amount = (
        _amount_mean(
            provider_90d,
            default=claim_amount,
        )
    )

    eps = 1e-6

    coverage_limit = (
        _safe_float(
            customer_latest.get(
                "coverage_limit"
            ),
            default=max(
                requested_reimbursement,
                claim_amount,
            ),
        )
    )

    coverage_limit = (
        max(
            coverage_limit,
            eps,
        )
    )

    days_since_policy_change = (
        _safe_optional_float(
            customer_latest.get(
                "days_since_policy_change"
            )
        )
    )

    # -------------------------------------------------------------------------
    # Production-like inference payload
    # -------------------------------------------------------------------------

    claim = {
        "claim_id":
            _new_claim_id(),

        "customer_id":
            str(
                customer_id
            ),

        "policy_id":
            _json_safe(
                customer_latest.get(
                    "policy_id"
                )
            ),

        "provider_id":
            str(
                provider_id
            ),

        "service_category":
            str(
                service_category
            ),

        "service_code":
            str(
                service_code
            ),

        "service_units":
            int(
                service_units
            ),

        "service_date":
            service_date.isoformat(),

        "claim_submission_date":
            submission_datetime
            .date()
            .isoformat(),

        "claim_submission_timestamp":
            submission_datetime.isoformat(),

        "claim_amount":
            float(
                claim_amount
            ),

        "requested_reimbursement":
            float(
                requested_reimbursement
            ),

        "coverage_limit":
            float(
                coverage_limit
            ),

        "submission_channel":
            str(
                submission_channel
            ),

        "document_count":
            int(
                document_count
            ),

        "has_invoice":
            int(
                bool(
                    has_invoice
                )
            ),

        "has_prescription":
            has_prescription,

        "customer_age":
            _safe_int(
                customer_latest.get(
                    "customer_age"
                )
            ),

        "customer_tenure_months":
            _safe_int(
                customer_latest.get(
                    "customer_tenure_months"
                )
            ),

        "coverage_level":
            _json_safe(
                customer_latest.get(
                    "coverage_level"
                )
            ),

        "policy_tenure_months":
            _safe_int(
                customer_latest.get(
                    "policy_tenure_months"
                )
            ),

        "recent_policy_change":
            int(
                _safe_bool(
                    customer_latest.get(
                        "recent_policy_change"
                    )
                )
            ),

        "days_since_policy_change":
            days_since_policy_change,

        "provider_type":
            _json_safe(
                provider_latest.get(
                    "provider_type"
                )
            ),

        "provider_region":
            _json_safe(
                provider_latest.get(
                    "provider_region"
                )
            ),

        "provider_tenure_months":
            _safe_int(
                provider_latest.get(
                    "provider_tenure_months"
                )
            ),

        "days_service_to_submission":
            int(
                max(
                    0,
                    (
                        submission_ts.normalize()
                        - service_ts.normalize()
                    ).days,
                )
            ),

        "reimbursement_ratio":
            float(
                requested_reimbursement
                / max(
                    claim_amount,
                    eps,
                )
            ),

        # ---------------------------------------------------------------------
        # Customer history
        # ---------------------------------------------------------------------

        "customer_claims_7d":
            len(
                customer_7d
            ),

        "customer_claims_30d":
            len(
                customer_30d
            ),

        "customer_claims_90d":
            len(
                customer_90d
            ),

        "customer_claims_365d":
            len(
                customer_365d
            ),

        "customer_amount_30d":
            _amount_sum(
                customer_30d
            ),

        "customer_amount_365d":
            _amount_sum(
                customer_365d
            ),

        "customer_avg_claim_amount_365d":
            customer_avg_amount,

        "days_since_customer_previous_claim":
            _days_since_last(
                customer_history,
                submission_ts,
            ),

        # ---------------------------------------------------------------------
        # Customer-provider relationship
        # ---------------------------------------------------------------------

        "days_since_same_provider_claim":
            _days_since_last(
                relationship_history,
                submission_ts,
            ),

        "customer_provider_claims_30d":
            len(
                relationship_30d
            ),

        "same_service_claims_30d":
            len(
                same_service_30d
            ),

        # ---------------------------------------------------------------------
        # Provider history
        # ---------------------------------------------------------------------

        "provider_claims_30d":
            len(
                provider_30d
            ),

        "provider_claims_90d":
            len(
                provider_90d
            ),

        "provider_avg_claim_amount_90d":
            provider_avg_amount,

        # ---------------------------------------------------------------------
        # Contextual monetary ratios
        # ---------------------------------------------------------------------

        "service_typical_amount":
            service_typical_amount,

        "claim_to_service_median_ratio":
            float(
                claim_amount
                / max(
                    service_typical_amount,
                    eps,
                )
            ),

        "claim_to_customer_avg_ratio":
            float(
                claim_amount
                / max(
                    customer_avg_amount,
                    eps,
                )
            ),

        "claim_to_provider_avg_ratio":
            float(
                claim_amount
                / max(
                    provider_avg_amount,
                    eps,
                )
            ),
    }

    return (
        _strip_leakage(
            claim
        )
    )


# =============================================================================
# Prediction state
# =============================================================================


def save_prediction(
    claim: dict[str, Any],
    response: dict[str, Any],
    source: str,
) -> None:
    """
    Validate and persist one model prediction.
    """

    prediction = (
        response.get(
            "prediction"
        )
    )

    if not isinstance(
        prediction,
        dict,
    ):

        raise ValueError(
            (
                "Inference API response does not "
                "contain a valid prediction."
            )
        )

    if (
        "fraud_risk_score"
        not in prediction
    ):

        raise ValueError(
            (
                "Prediction does not contain "
                "fraud_risk_score."
            )
        )

    score = (
        float(
            prediction[
                "fraud_risk_score"
            ]
        )
    )

    if not np.isfinite(
        score
    ):

        raise ValueError(
            (
                "Inference API returned a "
                "non-finite risk score."
            )
        )

    st.session_state.single_prediction = (
        prediction
    )

    st.session_state.single_score = (
        min(
            max(
                score,
                0.0,
            ),
            1.0,
        )
    )

    st.session_state.single_claim = (
        claim
    )

    st.session_state.single_source = (
        source
    )


def _clear_prediction() -> None:
    """
    Clear the persisted prediction.
    """

    for key in (
        "single_prediction",
        "single_score",
        "single_claim",
        "single_source",
    ):

        st.session_state[
            key
        ] = None


def _reset_analysis() -> None:
    """
    Reset prediction and user-facing analysis controls.
    """

    _clear_prediction()

    for key in SMART_WIDGET_KEYS:

        if key in st.session_state:

            del st.session_state[
                key
            ]


# =============================================================================
# Recommendation
# =============================================================================


def _recommendation(
    score: float,
) -> tuple[
    str,
    str,
    str,
]:
    """
    Convert model risk into an operational recommendation.

    Returns
    -------
    tuple
        action, explanation, visual tone
    """

    if score >= 0.50:

        return (
            "Priority review",
            (
                "High individual model risk. "
                "Prioritized investigator review "
                "is recommended."
            ),
            "danger",
        )

    if score >= 0.20:

        return (
            "Review recommended",
            (
                "Elevated individual model risk. "
                "Investigator review should be considered."
            ),
            "warning",
        )

    if score >= 0.05:

        return (
            "Capacity dependent",
            (
                "Moderate individual model risk. "
                "Final priority depends on portfolio ranking "
                "and available investigation capacity."
            ),
            "info",
        )

    return (
        "Routine processing",
        (
            "Low individual model risk. "
            "The claim may still be selected if its "
            "relative portfolio rank warrants review."
        ),
        "success",
    )


# =============================================================================
# Claim snapshot
# =============================================================================


def _render_claim_snapshot(
    claim: dict[str, Any],
) -> None:
    """
    Render concise operational information about the scored claim.
    """

    section_header(
        "Claim Snapshot",
        (
            "Business context associated with "
            "the current model assessment."
        ),
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Claim Amount",
            (
                f"€"
                f"{_safe_float(claim.get('claim_amount')):,.2f}"
            ),
            "Submitted amount",
            tone="neutral",
        )

    with c2:

        metric_card(
            "Requested",
            (
                f"€"
                f"{_safe_float(claim.get('requested_reimbursement')):,.2f}"
            ),
            "Requested reimbursement",
            tone="info",
        )

    with c3:

        metric_card(
            "Customer 30d",
            str(
                _safe_int(
                    claim.get(
                        "customer_claims_30d"
                    )
                )
            ),
            "Previous customer claims",
            tone="neutral",
        )

    with c4:

        metric_card(
            "Provider 30d",
            str(
                _safe_int(
                    claim.get(
                        "provider_claims_30d"
                    )
                )
            ),
            "Provider activity",
            tone="neutral",
        )

    st.write("")

    with st.container(
        border=True
    ):

        c1, c2 = (
            st.columns(2)
        )

        with c1:

            st.caption(
                "CUSTOMER"
            )

            st.code(
                _format_identifier(
                    claim.get(
                        "customer_id"
                    )
                ),
                language=None,
            )

            st.caption(
                "PROVIDER"
            )

            st.code(
                _format_identifier(
                    claim.get(
                        "provider_id"
                    )
                ),
                language=None,
            )

        with c2:

            st.caption(
                "SERVICE CATEGORY"
            )

            st.write(
                str(
                    claim.get(
                        "service_category",
                        "—",
                    )
                )
            )

            st.caption(
                "SERVICE CODE"
            )

            st.code(
                _format_identifier(
                    claim.get(
                        "service_code"
                    )
                ),
                language=None,
            )


# =============================================================================
# Risk assessment
# =============================================================================


def _render_assessment() -> None:
    """
    Render the final individual fraud-risk assessment.
    """

    prediction = (
        st.session_state.get(
            "single_prediction"
        )
    )

    score = (
        st.session_state.get(
            "single_score"
        )
    )

    claim = (
        st.session_state.get(
            "single_claim"
        )
    )

    source = (
        st.session_state.get(
            "single_source"
        )
    )

    if (
        prediction is None
        or score is None
        or claim is None
    ):
        return

    score = float(
        score
    )

    st.write("")
    st.write("")

    section_header(
        "Risk Assessment",
        (
            "Individual prediction produced by "
            "the deployed fraud-risk pipeline."
        ),
    )

    left, right = (
        st.columns(
            [
                0.85,
                1.75,
            ],
            gap="large",
        )
    )

    # -------------------------------------------------------------------------
    # Risk gauge
    # -------------------------------------------------------------------------

    with left:

        risk_gauge(
            score
        )

        risk_badge(
            score
        )

    # -------------------------------------------------------------------------
    # Decision support
    # -------------------------------------------------------------------------

    with right:

        with st.container(
            border=True
        ):

            (
                recommendation,
                explanation,
                recommendation_tone,
            ) = (
                _recommendation(
                    score
                )
            )

            c1, c2, c3 = (
                st.columns(
                    [
                        1,
                        1,
                        1.35,
                    ]
                )
            )

            # IMPORTANT:
            # We deliberately use our own responsive metric_card()
            # rather than st.metric() here.
            #
            # Native Streamlit metrics can apply internal truncation
            # to long text such as "Routine processing".
            # metric_card() guarantees complete business text.
            with c1:

                metric_card(
                    "Fraud Risk",
                    f"{score:.2%}",
                    "Individual model score",
                    tone=recommendation_tone,
                )

            with c2:

                metric_card(
                    "Risk Tier",
                    risk_tier(
                        score
                    ),
                    "Individual risk level",
                    tone=recommendation_tone,
                )

            with c3:

                metric_card(
                    "Recommended Action",
                    recommendation,
                    "Decision-support recommendation",
                    tone=recommendation_tone,
                )

            st.write("")
            st.divider()

            # -----------------------------------------------------------------
            # Claim / model identity
            # -----------------------------------------------------------------

            c1, c2 = (
                st.columns(
                    [
                        1,
                        1,
                    ]
                )
            )

            with c1:

                st.caption(
                    "CLAIM"
                )

                st.code(
                    _format_identifier(
                        prediction.get(
                            "claim_id"
                        )
                    ),
                    language=None,
                )

                st.write(
                    (
                        "**Source:** "
                        f"{source or '—'}"
                    )
                )

            with c2:

                st.caption(
                    "MODEL"
                )

                model_name = (
                    _format_identifier(
                        prediction.get(
                            "model_name"
                        )
                    )
                )

                model_version = (
                    _format_identifier(
                        prediction.get(
                            "model_version"
                        )
                    )
                )

                st.write(
                    (
                        f"**{model_name}** "
                        f"v{model_version}"
                    )
                )

                st.write(
                    (
                        "**Risk tier:** "
                        f"{risk_tier(score)}"
                    )
                )

            st.divider()

            # -----------------------------------------------------------------
            # Operational recommendation
            # -----------------------------------------------------------------

            if score >= 0.50:

                st.error(
                    explanation
                )

            elif score >= 0.20:

                st.warning(
                    explanation
                )

            elif score >= 0.05:

                st.info(
                    explanation
                )

            else:

                st.success(
                    explanation
                )

            st.caption(
                (
                    "This recommendation is generated from "
                    "model risk and does not constitute a "
                    "fraud determination."
                )
            )

    st.write("")
    st.write("")

    _render_claim_snapshot(
        claim
    )

    st.write("")

    human_review_notice()

    st.write("")

    with st.expander(
        "Technical model input",
        expanded=False,
    ):

        st.json(
            _strip_leakage(
                claim
            )
        )


# =============================================================================
# Smart Analysis
# =============================================================================


def _render_smart_form(
    client,
) -> None:
    """
    Render low-friction operational claim intake.
    """

    section_header(
        "Smart Claim Intake",
        (
            "Enter only the core claim information. "
            "Customer history, provider activity and "
            "contextual model features are generated automatically."
        ),
    )

    try:

        history = (
            _load_context_data()
        )

    except Exception as exc:

        st.error(
            (
                "Historical context could not be loaded. "
                f"{exc}"
            )
        )

        return

    # -------------------------------------------------------------------------
    # Identity domains
    # -------------------------------------------------------------------------

    customers = (
        history[
            "customer_id"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    providers = (
        history[
            "provider_id"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    categories = (
        history[
            "service_category"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if (
        not customers
        or not providers
        or not categories
    ):

        st.error(
            (
                "Historical context does not contain "
                "enough entities for Smart Analysis."
            )
        )

        return

    # -------------------------------------------------------------------------
    # Context
    # -------------------------------------------------------------------------

    st.markdown(
        "#### Claim context"
    )

    st.caption(
        (
            f"{len(customers):,} historical customers • "
            f"{len(providers):,} providers available"
        )
    )

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        customer_id = (
            st.selectbox(
                "Customer",
                options=customers,
                key="smart_customer_id",
                help=(
                    "Customer history is retrieved "
                    "automatically from the historical context."
                ),
            )
        )

    with c2:

        provider_id = (
            st.selectbox(
                "Provider",
                options=providers,
                key="smart_provider_id",
                help=(
                    "Provider activity and monetary "
                    "benchmarks are generated automatically."
                ),
            )
        )

    st.write("")

    # -------------------------------------------------------------------------
    # Service
    # -------------------------------------------------------------------------

    st.markdown(
        "#### Service"
    )

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        service_category = (
            st.selectbox(
                "Service category",
                options=categories,
                key="smart_service_category",
            )
        )

    compatible_codes = (
        history.loc[
            history[
                "service_category"
            ]
            .astype(str)
            == str(
                service_category
            ),
            "service_code",
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not compatible_codes:

        st.error(
            (
                "No service codes are available "
                "for the selected category."
            )
        )

        return

    with c2:

        service_code = (
            st.selectbox(
                "Service",
                options=compatible_codes,
                key="smart_service_code",
            )
        )

    st.write("")

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    st.markdown(
        "#### Timing"
    )

    today = (
        datetime.now()
        .date()
    )

    default_service_date = (
        today
        - timedelta(
            days=3
        )
    )

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        service_date_value = (
            st.date_input(
                "Service date",
                value=default_service_date,
                max_value=today,
                key="smart_service_date",
            )
        )

    with c2:

        submission_date = (
            st.date_input(
                "Submission date",
                value=today,
                max_value=today,
                key="smart_submission_date",
            )
        )

    invalid_dates = (
        service_date_value
        > submission_date
    )

    if invalid_dates:

        st.error(
            (
                "Service date must be on or before "
                "the submission date."
            )
        )

    # -------------------------------------------------------------------------
    # Historical service reference
    #
    # Use only context preceding the selected submission date to avoid
    # introducing future information into the UI recommendation.
    # -------------------------------------------------------------------------

    reference_timestamp = (
        pd.Timestamp(
            datetime.combine(
                submission_date,
                time.max,
            )
        )
    )

    reference_history = (
        _history_before(
            history,
            reference_timestamp,
        )
    )

    (
        typical_amount,
        typical_requested,
    ) = (
        _service_reference(
            reference_history,
            service_code,
        )
    )

    st.write("")

    info_panel(
        "Historical Service Reference",
        (
            f"Typical historical claim ≈ €{typical_amount:,.2f} • "
            f"Typical reimbursement ≈ €{typical_requested:,.2f}. "
            "These values are suggestions only and are not imposed."
        ),
        tone="info",
    )

    st.write("")

    # -------------------------------------------------------------------------
    # Financials
    # -------------------------------------------------------------------------

    st.markdown(
        "#### Financial information"
    )

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        claim_amount = (
            st.number_input(
                "Claim amount (€)",
                min_value=0.01,
                value=float(
                    round(
                        typical_amount,
                        2,
                    )
                ),
                step=10.0,
                format="%.2f",
                key="smart_claim_amount",
            )
        )

    with c2:

        requested_reimbursement = (
            st.number_input(
                "Requested reimbursement (€)",
                min_value=0.0,
                value=float(
                    round(
                        min(
                            typical_requested,
                            max(
                                typical_amount,
                                0.01,
                            ),
                        ),
                        2,
                    )
                ),
                step=10.0,
                format="%.2f",
                key="smart_requested_reimbursement",
            )
        )

    reimbursement_ratio = (
        float(
            requested_reimbursement
        )
        / max(
            float(
                claim_amount
            ),
            1e-6,
        )
    )

    ratio_col1, ratio_col2 = (
        st.columns(2)
    )

    with ratio_col1:

        st.caption(
            (
                "Requested / claim amount: "
                f"{reimbursement_ratio:.1%}"
            )
        )

    with ratio_col2:

        if (
            requested_reimbursement
            > claim_amount
        ):

            st.warning(
                (
                    "Requested reimbursement exceeds "
                    "the submitted claim amount."
                )
            )

    st.write("")

    # -------------------------------------------------------------------------
    # Optional details
    # -------------------------------------------------------------------------

    with st.expander(
        "Optional claim details",
        expanded=False,
    ):

        c1, c2 = (
            st.columns(2)
        )

        with c1:

            service_units = (
                st.number_input(
                    "Service units",
                    min_value=1,
                    max_value=100,
                    value=1,
                    step=1,
                    key="smart_service_units",
                )
            )

            submission_channel = (
                st.selectbox(
                    "Submission channel",
                    options=[
                        "web",
                        "mobile_app",
                        "provider_direct",
                        "email",
                        "paper",
                    ],
                    key="smart_submission_channel",
                )
            )

            document_count = (
                st.number_input(
                    "Documents",
                    min_value=0,
                    max_value=100,
                    value=1,
                    step=1,
                    key="smart_document_count",
                )
            )

        with c2:

            has_invoice = (
                st.toggle(
                    "Invoice attached",
                    value=True,
                    key="smart_has_invoice",
                )
            )

            prescription_state = (
                st.selectbox(
                    "Prescription",
                    options=[
                        "Not required / unknown",
                        "Yes",
                        "No",
                    ],
                    key="smart_prescription",
                )
            )

            submission_hour = (
                st.slider(
                    "Submission hour",
                    min_value=0,
                    max_value=23,
                    value=12,
                    key="smart_submission_hour",
                )
            )

    # -------------------------------------------------------------------------
    # Analyze
    # -------------------------------------------------------------------------

    st.write("")

    analyze = (
        st.button(
            "Analyze Claim",
            type="primary",
            use_container_width=True,
            key="smart_analyze_claim",
            disabled=invalid_dates,
        )
    )

    if not analyze:
        return

    if prescription_state == "Yes":

        has_prescription = True

    elif prescription_state == "No":

        has_prescription = False

    else:

        has_prescription = None

    submission_datetime = (
        datetime.combine(
            submission_date,
            time(
                hour=int(
                    submission_hour
                )
            ),
        )
    )

    try:

        with st.spinner(
            (
                "Retrieving historical context, "
                "building model features and scoring claim..."
            )
        ):

            claim = (
                _build_smart_claim(
                    history=history,
                    customer_id=customer_id,
                    provider_id=provider_id,
                    service_category=service_category,
                    service_code=service_code,
                    service_units=int(
                        service_units
                    ),
                    service_date=service_date_value,
                    submission_datetime=submission_datetime,
                    claim_amount=float(
                        claim_amount
                    ),
                    requested_reimbursement=float(
                        requested_reimbursement
                    ),
                    submission_channel=submission_channel,
                    document_count=int(
                        document_count
                    ),
                    has_invoice=has_invoice,
                    has_prescription=has_prescription,
                )
            )

            response = (
                client.score_claim(
                    claim
                )
            )

        save_prediction(
            claim,
            response,
            "Smart Analysis",
        )

        st.success(
            (
                "Claim successfully enriched and scored "
                "with the deployed model."
            )
        )

    except Exception as exc:

        st.error(
            (
                "Unable to analyze this claim. "
                f"{exc}"
            )
        )


# =============================================================================
# Quick Demo
# =============================================================================


def _render_quick_demo(
    client,
) -> None:
    """
    Score one complete historical synthetic claim.
    """

    section_header(
        "Quick Demo",
        (
            "Run the deployed inference pipeline "
            "against a complete synthetic example."
        ),
    )

    try:

        demo_claims = (
            load_demo_claims()
        )

        if demo_claims.empty:

            st.info(
                (
                    "No demonstration claims "
                    "are available."
                )
            )

            return

        index = (
            st.selectbox(
                "Demo claim",
                options=range(
                    min(
                        len(
                            demo_claims
                        ),
                        100,
                    )
                ),
                format_func=lambda i:
                    str(
                        demo_claims
                        .iloc[i]
                        .get(
                            "claim_id",
                            i,
                        )
                    ),
                key="demo_claim_selector",
            )
        )

        raw_claim = (
            get_demo_claim(
                int(
                    index
                )
            )
        )

        claim = (
            _strip_leakage(
                raw_claim
            )
        )

        if st.button(
            "Analyze Demo Claim",
            type="primary",
            use_container_width=True,
            key="analyze_demo_claim",
        ):

            with st.spinner(
                (
                    "Building features and "
                    "scoring demonstration claim..."
                )
            ):

                response = (
                    client.score_claim(
                        claim
                    )
                )

            save_prediction(
                claim,
                response,
                "Quick Demo",
            )

            st.success(
                (
                    "Demo claim scored "
                    "successfully."
                )
            )

        with st.expander(
            "View inference payload",
            expanded=False,
        ):

            st.json(
                claim
            )

    except Exception as exc:

        st.error(
            str(
                exc
            )
        )


# =============================================================================
# Advanced JSON
# =============================================================================


def _render_advanced_json(
    client,
) -> None:
    """
    Render direct technical payload scoring.
    """

    section_header(
        "Advanced JSON",
        (
            "Developer mode for complete payloads "
            "and direct API-level testing."
        ),
    )

    info_panel(
        "Technical Mode",
        (
            "Advanced JSON is intended for API validation "
            "and controlled technical testing. "
            "For operational use, prefer Smart Analysis."
        ),
        tone="info",
    )

    st.write("")

    raw = (
        st.text_area(
            "Complete claim JSON",
            height=420,
            placeholder=(
                "{\n"
                '  "claim_id": "LIVE_...",\n'
                '  "...": "..."\n'
                "}"
            ),
            key="advanced_json_payload",
        )
    )

    analyze = (
        st.button(
            "Analyze JSON",
            type="primary",
            use_container_width=True,
            key="analyze_json",
        )
    )

    if not analyze:
        return

    if not raw.strip():

        st.warning(
            (
                "Paste a claim JSON "
                "payload first."
            )
        )

        return

    try:

        claim = (
            json.loads(
                raw
            )
        )

        if not isinstance(
            claim,
            dict,
        ):

            raise ValueError(
                (
                    "The JSON payload must contain "
                    "one claim object."
                )
            )

        clean_claim = (
            _strip_leakage(
                claim
            )
        )

        if not clean_claim:

            raise ValueError(
                (
                    "The payload contains no "
                    "usable inference fields."
                )
            )

        with st.spinner(
            (
                "Validating payload, building features "
                "and scoring claim..."
            )
        ):

            response = (
                client.score_claim(
                    clean_claim
                )
            )

        save_prediction(
            clean_claim,
            response,
            "Advanced JSON",
        )

        st.success(
            (
                "JSON claim scored "
                "successfully."
            )
        )

    except json.JSONDecodeError as exc:

        st.error(
            (
                "Invalid JSON syntax: "
                f"{exc}"
            )
        )

    except Exception as exc:

        st.error(
            str(
                exc
            )
        )


# =============================================================================
# Main page
# =============================================================================


def render(
    client,
) -> None:
    """
    Render the complete individual-claim investigation workflow.
    """

    section_header(
        "Claim Analysis",
        (
            "Analyze an individual health-insurance claim "
            "using automatic historical enrichment and "
            "the deployed fraud-risk model."
        ),
    )

    # -------------------------------------------------------------------------
    # Page controls
    # -------------------------------------------------------------------------

    left, right = (
        st.columns(
            [
                4,
                1,
            ]
        )
    )

    with left:

        st.caption(
            (
                "Smart Analysis minimizes manual input: "
                "customer history, provider activity, service "
                "benchmarks and contextual ratios are generated "
                "from the available historical context."
            )
        )

    with right:

        if st.button(
            "Reset Analysis",
            use_container_width=True,
            key="reset_claim_analysis",
        ):

            _reset_analysis()

            st.rerun()

    st.write("")

    # -------------------------------------------------------------------------
    # Analysis modes
    # -------------------------------------------------------------------------

    smart_tab, demo_tab, json_tab = (
        st.tabs(
            [
                "Smart Analysis",
                "Quick Demo",
                "Advanced JSON",
            ]
        )
    )

    with smart_tab:

        _render_smart_form(
            client
        )

    with demo_tab:

        _render_quick_demo(
            client
        )

    with json_tab:

        _render_advanced_json(
            client
        )

    # -------------------------------------------------------------------------
    # Persistent assessment
    # -------------------------------------------------------------------------

    _render_assessment()