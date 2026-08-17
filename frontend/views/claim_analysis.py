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
MAX_VISIBLE_SHAP_DRIVERS = 6
MAX_TECHNICAL_SHAP_ROWS = 107


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


PREDICTION_STATE_KEYS = {
    "single_prediction",
    "single_score",
    "single_claim",
    "single_source",
    "single_explanation",
}


FEATURE_LABELS = {
    "service_units": "Service units",
    "claim_amount": "Claim amount",
    "requested_reimbursement": "Requested reimbursement",
    "coverage_limit": "Coverage limit",
    "document_count": "Document count",
    "customer_age": "Customer age",
    "customer_tenure_months": "Customer tenure",
    "policy_tenure_months": "Policy tenure",
    "days_since_policy_change": "Days since policy change",
    "provider_tenure_months": "Provider tenure",
    "days_service_to_submission": "Service-to-submission delay",
    "reimbursement_ratio": "Reimbursement ratio",
    "customer_claims_7d": "Customer claims — 7 days",
    "customer_claims_30d": "Customer claims — 30 days",
    "customer_claims_90d": "Customer claims — 90 days",
    "customer_claims_365d": "Customer claims — 365 days",
    "customer_amount_30d": "Customer amount — 30 days",
    "customer_amount_365d": "Customer amount — 365 days",
    "customer_avg_claim_amount_365d": "Customer average claim amount",
    "days_since_customer_previous_claim": "Time since previous customer claim",
    "days_since_same_provider_claim": "Time since same-provider claim",
    "customer_provider_claims_30d": "Customer-provider claims — 30 days",
    "same_service_claims_30d": "Same-service claims — 30 days",
    "provider_claims_30d": "Provider claims — 30 days",
    "provider_claims_90d": "Provider claims — 90 days",
    "provider_avg_claim_amount_90d": "Provider average claim amount",
    "service_typical_amount": "Typical service amount",
    "claim_to_service_median_ratio": "Claim vs service benchmark",
    "claim_to_customer_avg_ratio": "Claim vs customer average",
    "claim_to_provider_avg_ratio": "Claim vs provider average",
    "submission_hour": "Submission hour",
    "submission_dayofweek": "Submission day of week",
    "submission_month": "Submission month",
    "service_dayofweek": "Service day of week",
    "service_month": "Service month",
    "requested_to_limit_ratio": "Requested vs coverage limit",
    "amount_above_service_typical": "Amount above service benchmark",
    "recent_claim_share_30d_365d": "Recent customer claim concentration",
    "recent_amount_share_30d_365d": "Recent customer amount concentration",
    "provider_recent_activity_ratio": "Provider recent activity",
    "customer_provider_intensity": "Customer-provider interaction intensity",
    "same_service_intensity": "Repeated-service intensity",
    "has_invoice": "Invoice attached",
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
        result = float(value)

        if np.isfinite(result):
            return result

    except (
        TypeError,
        ValueError,
    ):
        pass

    return float(default)


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    Convert numeric-like values to integer safely.
    """

    try:
        if pd.isna(value):
            return default

        return int(
            round(
                float(value)
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
        if pd.isna(value):
            return None

        result = float(value)

        if np.isfinite(result):
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
        if pd.isna(value):
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

    return bool(value)


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

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    try:
        if pd.isna(value):
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
            return _json_safe(
                value.item()
            )

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
        key: _json_safe(value)
        for key, value in claim.items()
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

    value = str(value).strip()

    return (
        value
        if value
        else "—"
    )


def _humanize_feature_name(
    feature_name: Any,
) -> str:
    """
    Convert transformed model feature names into analyst-friendly labels.
    """

    if feature_name is None:
        return "Unknown feature"

    name = str(feature_name).strip()

    if not name:
        return "Unknown feature"

    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    categorical_prefixes = {
        "service_category_": "Service category",
        "service_code_": "Service",
        "submission_channel_": "Submission channel",
        "coverage_level_": "Coverage level",
        "provider_type_": "Provider type",
        "provider_region_": "Provider region",
        "has_prescription_": "Prescription",
    }

    for prefix, label in categorical_prefixes.items():

        if name.startswith(prefix):

            value = (
                name[len(prefix):]
                .replace("_", " ")
                .strip()
            )

            value = (
                value.title()
                if value
                else "Unknown"
            )

            return (
                f"{label}: {value}"
            )

    return (
        name
        .replace("_", " ")
        .strip()
        .title()
    )


def _format_feature_value(
    value: Any,
) -> str:
    """
    Format model feature values safely for business display.
    """

    if value is None:
        return "—"

    if isinstance(
        value,
        bool,
    ):
        return (
            "Yes"
            if value
            else "No"
        )

    if isinstance(
        value,
        str,
    ):
        return value

    try:
        numeric = float(value)

        if not np.isfinite(numeric):
            return "—"

        if numeric.is_integer():
            return f"{int(numeric):,}"

        if abs(numeric) >= 1000:
            return f"{numeric:,.2f}"

        return f"{numeric:.4f}".rstrip("0").rstrip(".")

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


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
            "No historical claim context is available."
        )

    missing = (
        REQUIRED_CONTEXT_COLUMNS
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            (
                "Historical dataset is missing: "
                + ", ".join(
                    sorted(missing)
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

    if "service_date" in frame.columns:

        frame[
            "service_date"
        ] = pd.to_datetime(
            frame["service_date"],
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

    return frame.loc[mask]


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
            frame["claim_amount"],
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

    values = _amount_values(frame)

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

    values = _amount_values(frame)

    if values.empty:
        return float(default)

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

    values = _amount_values(frame)

    if values.empty:
        return float(default)

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
            dates < timestamp
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
            == str(service_code)
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
            typical_requested = float(
                requested.median()
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

    submission_ts = pd.Timestamp(
        submission_datetime
    )

    service_ts = pd.Timestamp(
        service_date
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

    historical = _history_before(
        history,
        submission_ts,
    )

    customer_history = (
        historical.loc[
            historical[
                "customer_id"
            ]
            .astype(str)
            == str(customer_id)
        ]
        .copy()
    )

    provider_history = (
        historical.loc[
            historical[
                "provider_id"
            ]
            .astype(str)
            == str(provider_id)
        ]
        .copy()
    )

    relationship_history = (
        customer_history.loc[
            customer_history[
                "provider_id"
            ]
            .astype(str)
            == str(provider_id)
        ]
        .copy()
    )

    same_service_history = (
        customer_history.loc[
            customer_history[
                "service_code"
            ]
            .astype(str)
            == str(service_code)
        ]
        .copy()
    )

    service_history = (
        historical.loc[
            historical[
                "service_code"
            ]
            .astype(str)
            == str(service_code)
        ]
        .copy()
    )

    customer_latest = _latest_row(
        customer_history
    )

    provider_latest = _latest_row(
        provider_history
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

    customer_7d = _window(
        customer_history,
        submission_ts,
        7,
    )

    customer_30d = _window(
        customer_history,
        submission_ts,
        30,
    )

    customer_90d = _window(
        customer_history,
        submission_ts,
        90,
    )

    customer_365d = _window(
        customer_history,
        submission_ts,
        365,
    )

    provider_30d = _window(
        provider_history,
        submission_ts,
        30,
    )

    provider_90d = _window(
        provider_history,
        submission_ts,
        90,
    )

    relationship_30d = _window(
        relationship_history,
        submission_ts,
        30,
    )

    same_service_30d = _window(
        same_service_history,
        submission_ts,
        30,
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

    coverage_limit = max(
        coverage_limit,
        eps,
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
            str(customer_id),

        "policy_id":
            _json_safe(
                customer_latest.get(
                    "policy_id"
                )
            ),

        "provider_id":
            str(provider_id),

        "service_category":
            str(service_category),

        "service_code":
            str(service_code),

        "service_units":
            int(service_units),

        "service_date":
            service_date.isoformat(),

        "claim_submission_date":
            submission_datetime
            .date()
            .isoformat(),

        "claim_submission_timestamp":
            submission_datetime.isoformat(),

        "claim_amount":
            float(claim_amount),

        "requested_reimbursement":
            float(requested_reimbursement),

        "coverage_limit":
            float(coverage_limit),

        "submission_channel":
            str(submission_channel),

        "document_count":
            int(document_count),

        "has_invoice":
            int(
                bool(has_invoice)
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
            len(customer_7d),

        "customer_claims_30d":
            len(customer_30d),

        "customer_claims_90d":
            len(customer_90d),

        "customer_claims_365d":
            len(customer_365d),

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
            len(provider_30d),

        "provider_claims_90d":
            len(provider_90d),

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

    return _strip_leakage(
        claim
    )


# =============================================================================
# Prediction / explanation state
# =============================================================================


def _extract_explanation_payload(
    response: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Normalize supported API explanation envelopes.

    The API client may return either the explanation object directly
    or an envelope containing it under ``explanation``.
    """

    if not isinstance(
        response,
        dict,
    ):
        return None

    explanation = response.get(
        "explanation"
    )

    if isinstance(
        explanation,
        dict,
    ):
        return explanation

    return response


def save_prediction(
    claim: dict[str, Any],
    response: dict[str, Any],
    source: str,
    explanation_response: dict[str, Any] | None = None,
) -> None:
    """
    Validate and persist one model prediction and optional explanation.
    """

    prediction = response.get(
        "prediction"
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

    score = float(
        prediction[
            "fraud_risk_score"
        ]
    )

    if not np.isfinite(score):
        raise ValueError(
            (
                "Inference API returned a "
                "non-finite risk score."
            )
        )

    st.session_state.single_prediction = (
        prediction
    )

    st.session_state.single_score = min(
        max(
            score,
            0.0,
        ),
        1.0,
    )

    st.session_state.single_claim = (
        claim
    )

    st.session_state.single_source = (
        source
    )

    st.session_state.single_explanation = (
        _extract_explanation_payload(
            explanation_response
        )
    )


def _clear_prediction() -> None:
    """
    Clear persisted prediction and explanation state.
    """

    for key in PREDICTION_STATE_KEYS:
        st.session_state[key] = None


def _reset_analysis() -> None:
    """
    Reset prediction and user-facing analysis controls.
    """

    _clear_prediction()

    for key in SMART_WIDGET_KEYS:

        if key in st.session_state:
            del st.session_state[key]


def _score_and_explain(
    client,
    claim: dict[str, Any],
    source: str,
) -> None:
    """
    Execute scoring and TreeSHAP explanation as one frontend workflow.

    Scoring remains authoritative. Explainability is additive:
    if explanation retrieval fails, the valid prediction is preserved.
    """

    clean_claim = _strip_leakage(
        claim
    )

    response = client.score_claim(
        clean_claim
    )

    explanation_response = None

    try:
        explanation_response = (
            client.explain_claim(
                clean_claim
            )
        )

    except Exception as exc:
        # Do not discard a successful model prediction because the
        # explanation endpoint is temporarily unavailable.
        explanation_response = {
            "_frontend_explanation_error": str(exc),
        }

    save_prediction(
        clean_claim,
        response,
        source,
        explanation_response,
    )


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
# SHAP helpers
# =============================================================================


def _get_contribution_name(
    item: dict[str, Any],
) -> str:
    """
    Extract a feature name from a SHAP contribution object.
    """

    for key in (
        "feature",
        "feature_name",
        "name",
    ):
        value = item.get(key)

        if value is not None:
            return str(value)

    return "unknown_feature"


def _get_contribution_value(
    item: dict[str, Any],
) -> float:
    """
    Extract a finite SHAP contribution.
    """

    for key in (
        "shap_value",
        "contribution",
        "value_contribution",
    ):

        if key not in item:
            continue

        try:
            value = float(
                item[key]
            )

            if np.isfinite(value):
                return value

        except (
            TypeError,
            ValueError,
        ):
            continue

    return 0.0


def _get_feature_value(
    item: dict[str, Any],
) -> Any:
    """
    Extract transformed feature value when supplied by the API.
    """

    for key in (
        "feature_value",
        "transformed_value",
        "input_value",
        "value",
    ):

        if key in item:
            return item[key]

    return None


def _normalize_contributions(
    explanation: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Normalize the API's full SHAP contribution collection.
    """

    candidates = (
        explanation.get(
            "all_contributions"
        )
        or explanation.get(
            "contributions"
        )
        or []
    )

    if not isinstance(
        candidates,
        list,
    ):
        return []

    normalized = []

    for item in candidates:

        if not isinstance(
            item,
            dict,
        ):
            continue

        shap_value = (
            _get_contribution_value(
                item
            )
        )

        normalized.append(
            {
                "feature":
                    _get_contribution_name(
                        item
                    ),

                "feature_value":
                    _get_feature_value(
                        item
                    ),

                "shap_value":
                    shap_value,

                "absolute_shap":
                    abs(
                        shap_value
                    ),
            }
        )

    normalized.sort(
        key=lambda item:
            item["absolute_shap"],
        reverse=True,
    )

    return normalized


def _normalize_driver_collection(
    explanation: dict[str, Any],
    key: str,
) -> list[dict[str, Any]]:
    """
    Normalize positive / negative driver lists returned by the API.
    """

    raw = explanation.get(
        key,
        []
    )

    if not isinstance(
        raw,
        list,
    ):
        return []

    result = []

    for item in raw:

        if not isinstance(
            item,
            dict,
        ):
            continue

        shap_value = (
            _get_contribution_value(
                item
            )
        )

        result.append(
            {
                "feature":
                    _get_contribution_name(
                        item
                    ),

                "feature_value":
                    _get_feature_value(
                        item
                    ),

                "shap_value":
                    shap_value,

                "absolute_shap":
                    abs(
                        shap_value
                    ),
            }
        )

    result.sort(
        key=lambda item:
            item["absolute_shap"],
        reverse=True,
    )

    return result


def _explanation_drivers(
    explanation: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Return risk-increasing and risk-reducing SHAP drivers.

    Prefer dedicated backend collections. Fall back to the sign of
    the complete SHAP contribution vector.
    """

    positive = (
        _normalize_driver_collection(
            explanation,
            "positive_drivers",
        )
    )

    negative = (
        _normalize_driver_collection(
            explanation,
            "negative_drivers",
        )
    )

    if positive or negative:
        return (
            positive,
            negative,
        )

    contributions = (
        _normalize_contributions(
            explanation
        )
    )

    positive = [
        item
        for item in contributions
        if item["shap_value"] > 0
    ]

    negative = [
        item
        for item in contributions
        if item["shap_value"] < 0
    ]

    return (
        positive,
        negative,
    )


def _render_driver(
    driver: dict[str, Any],
    *,
    direction: str,
) -> None:
    """
    Render one SHAP driver in an analyst-readable form.
    """

    feature_name = (
        _humanize_feature_name(
            driver.get(
                "feature"
            )
        )
    )

    feature_value = (
        _format_feature_value(
            driver.get(
                "feature_value"
            )
        )
    )

    contribution = (
        _safe_float(
            driver.get(
                "shap_value"
            )
        )
    )

    sign = (
        "+"
        if contribution >= 0
        else "−"
    )

    with st.container(
        border=True
    ):
        st.markdown(
            f"**{feature_name}**"
        )

        if (
            driver.get(
                "feature_value"
            )
            is not None
        ):
            st.caption(
                (
                    "Observed model value: "
                    f"{feature_value}"
                )
            )

        if direction == "increase":
            st.write(
                (
                    "Contribution toward higher "
                    "predicted fraud risk"
                )
            )

        else:
            st.write(
                (
                    "Contribution toward lower "
                    "predicted fraud risk"
                )
            )

        st.caption(
            (
                "SHAP contribution: "
                f"{sign}{abs(contribution):.4f} "
                "raw-margin units"
            )
        )


def _render_explainability() -> None:
    """
    Render local TreeSHAP explanation for the active claim.
    """

    explanation = (
        st.session_state.get(
            "single_explanation"
        )
    )

    if not isinstance(
        explanation,
        dict,
    ):
        return

    st.write("")
    st.write("")

    section_header(
        "Model Explanation",
        (
            "Local TreeSHAP attribution showing which model "
            "features moved this claim's prediction upward "
            "or downward relative to the model baseline."
        ),
    )

    explanation_error = (
        explanation.get(
            "_frontend_explanation_error"
        )
    )

    if explanation_error:
        st.warning(
            (
                "The claim was scored successfully, but the "
                "model explanation is currently unavailable. "
                f"{explanation_error}"
            )
        )
        return

    method = (
        explanation.get(
            "explanation_method"
        )
        or explanation.get(
            "method"
        )
        or "TreeSHAP"
    )

    output_space = (
        explanation.get(
            "explanation_space"
        )
        or explanation.get(
            "output_space"
        )
        or "raw_margin_log_odds"
    )

    transformed_feature_count = (
        explanation.get(
            "transformed_feature_count"
        )
    )

    contributions = (
        _normalize_contributions(
            explanation
        )
    )

    if transformed_feature_count is None:
        transformed_feature_count = len(
            contributions
        )

    consistency = explanation.get(
        "consistency",
        {},
    )

    if not isinstance(
        consistency,
        dict,
    ):
        consistency = {}

    shap_ok = consistency.get(
        "shap_additivity_ok"
    )

    probability_ok = consistency.get(
        "probability_consistency_ok"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Method",
            str(method),
            "Local explanation engine",
            tone="info",
        )

    with c2:
        metric_card(
            "Feature Space",
            str(
                transformed_feature_count
            ),
            "Transformed model features",
            tone="neutral",
        )

    with c3:
        metric_card(
            "SHAP Additivity",
            (
                "Verified"
                if shap_ok is True
                else (
                    "Failed"
                    if shap_ok is False
                    else "Unavailable"
                )
            ),
            "Raw-margin reconstruction",
            tone=(
                "success"
                if shap_ok is True
                else (
                    "danger"
                    if shap_ok is False
                    else "neutral"
                )
            ),
        )

    with c4:
        metric_card(
            "Probability Check",
            (
                "Verified"
                if probability_ok is True
                else (
                    "Failed"
                    if probability_ok is False
                    else "Unavailable"
                )
            ),
            "Probability reconstruction",
            tone=(
                "success"
                if probability_ok is True
                else (
                    "danger"
                    if probability_ok is False
                    else "neutral"
                )
            ),
        )

    st.write("")

    info_panel(
        "How to interpret this explanation",
        (
            "Positive SHAP values push the model prediction toward "
            "higher fraud risk; negative values push it toward lower "
            "fraud risk. These are predictive contributions, not "
            "causal findings and not evidence of fraud by themselves."
        ),
        tone="info",
    )

    st.write("")

    positive, negative = (
        _explanation_drivers(
            explanation
        )
    )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:
        st.markdown(
            "#### Factors increasing model risk"
        )

        st.caption(
            (
                "Largest positive local contributions "
                "for this claim."
            )
        )

        visible_positive = (
            positive[
                :MAX_VISIBLE_SHAP_DRIVERS
            ]
        )

        if not visible_positive:
            st.info(
                (
                    "No positive SHAP contribution "
                    "was returned for this claim."
                )
            )

        else:
            for driver in visible_positive:
                _render_driver(
                    driver,
                    direction="increase",
                )

    with right:
        st.markdown(
            "#### Factors reducing model risk"
        )

        st.caption(
            (
                "Largest negative local contributions "
                "for this claim."
            )
        )

        visible_negative = (
            negative[
                :MAX_VISIBLE_SHAP_DRIVERS
            ]
        )

        if not visible_negative:
            st.info(
                (
                    "No negative SHAP contribution "
                    "was returned for this claim."
                )
            )

        else:
            for driver in visible_negative:
                _render_driver(
                    driver,
                    direction="decrease",
                )

    # -------------------------------------------------------------------------
    # Technical explanation
    # -------------------------------------------------------------------------

    st.write("")

    with st.expander(
        "Technical SHAP diagnostics",
        expanded=False,
    ):
        st.markdown(
            "##### Explanation contract"
        )

        technical = {
            "method":
                method,

            "output_space":
                output_space,

            "transformed_feature_count":
                transformed_feature_count,

            "base_value":
                explanation.get(
                    "base_value"
                ),

            "model_raw_margin":
                explanation.get(
                    "model_raw_margin"
                ),

            "shap_reconstructed_margin":
                explanation.get(
                    "shap_reconstructed_margin"
                ),

            "fraud_probability":
                explanation.get(
                    "fraud_probability"
                ),

            "shap_reconstructed_probability":
                explanation.get(
                    "shap_reconstructed_probability"
                ),

            "consistency":
                consistency,
        }

        st.json(
            _json_safe(
                technical
            )
        )

        if contributions:

            st.markdown(
                "##### Full transformed-feature attribution"
            )

            technical_frame = pd.DataFrame(
                contributions[
                    :MAX_TECHNICAL_SHAP_ROWS
                ]
            )

            technical_frame[
                "feature_label"
            ] = (
                technical_frame[
                    "feature"
                ]
                .map(
                    _humanize_feature_name
                )
            )

            technical_frame = (
                technical_frame[
                    [
                        "feature",
                        "feature_label",
                        "feature_value",
                        "shap_value",
                        "absolute_shap",
                    ]
                ]
            )

            st.dataframe(
                technical_frame,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                (
                    f"{len(contributions):,} transformed "
                    "feature contributions returned."
                )
            )

        else:
            st.info(
                (
                    "The API did not expose the complete "
                    "transformed-feature contribution vector."
                )
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

    c1, c2, c3, c4 = st.columns(4)

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
        c1, c2 = st.columns(2)

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

    score = float(score)

    st.write("")
    st.write("")

    section_header(
        "Risk Assessment",
        (
            "Individual prediction produced by "
            "the deployed fraud-risk pipeline."
        ),
    )

    left, right = st.columns(
        [
            0.85,
            1.75,
        ],
        gap="large",
    )

    with left:
        risk_gauge(
            score
        )

        risk_badge(
            score
        )

    with right:
        with st.container(
            border=True
        ):
            (
                recommendation,
                explanation,
                recommendation_tone,
            ) = _recommendation(
                score
            )

            c1, c2, c3 = st.columns(
                [
                    1,
                    1,
                    1.35,
                ]
            )

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

            c1, c2 = st.columns(2)

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

    _render_explainability()

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
        history = _load_context_data()

    except Exception as exc:
        st.error(
            (
                "Historical context could not be loaded. "
                f"{exc}"
            )
        )
        return

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

    st.markdown(
        "#### Claim context"
    )

    st.caption(
        (
            f"{len(customers):,} historical customers • "
            f"{len(providers):,} providers available"
        )
    )

    c1, c2 = st.columns(2)

    with c1:
        customer_id = st.selectbox(
            "Customer",
            options=customers,
            key="smart_customer_id",
            help=(
                "Customer history is retrieved "
                "automatically from the historical context."
            ),
        )

    with c2:
        provider_id = st.selectbox(
            "Provider",
            options=providers,
            key="smart_provider_id",
            help=(
                "Provider activity and monetary "
                "benchmarks are generated automatically."
            ),
        )

    st.write("")

    st.markdown(
        "#### Service"
    )

    c1, c2 = st.columns(2)

    with c1:
        service_category = st.selectbox(
            "Service category",
            options=categories,
            key="smart_service_category",
        )

    compatible_codes = (
        history.loc[
            history[
                "service_category"
            ]
            .astype(str)
            == str(service_category),
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
        service_code = st.selectbox(
            "Service",
            options=compatible_codes,
            key="smart_service_code",
        )

    st.write("")

    st.markdown(
        "#### Timing"
    )

    today = datetime.now().date()

    default_service_date = (
        today
        - timedelta(
            days=3
        )
    )

    c1, c2 = st.columns(2)

    with c1:
        service_date_value = st.date_input(
            "Service date",
            value=default_service_date,
            max_value=today,
            key="smart_service_date",
        )

    with c2:
        submission_date = st.date_input(
            "Submission date",
            value=today,
            max_value=today,
            key="smart_submission_date",
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

    reference_timestamp = pd.Timestamp(
        datetime.combine(
            submission_date,
            time.max,
        )
    )

    reference_history = _history_before(
        history,
        reference_timestamp,
    )

    (
        typical_amount,
        typical_requested,
    ) = _service_reference(
        reference_history,
        service_code,
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

    st.markdown(
        "#### Financial information"
    )

    c1, c2 = st.columns(2)

    with c1:
        claim_amount = st.number_input(
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

    with c2:
        requested_reimbursement = st.number_input(
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

    ratio_col1, ratio_col2 = st.columns(2)

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

    with st.expander(
        "Optional claim details",
        expanded=False,
    ):
        c1, c2 = st.columns(2)

        with c1:
            service_units = st.number_input(
                "Service units",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key="smart_service_units",
            )

            submission_channel = st.selectbox(
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

            document_count = st.number_input(
                "Documents",
                min_value=0,
                max_value=100,
                value=1,
                step=1,
                key="smart_document_count",
            )

        with c2:
            has_invoice = st.toggle(
                "Invoice attached",
                value=True,
                key="smart_has_invoice",
            )

            prescription_state = st.selectbox(
                "Prescription",
                options=[
                    "Not required / unknown",
                    "Yes",
                    "No",
                ],
                key="smart_prescription",
            )

            submission_hour = st.slider(
                "Submission hour",
                min_value=0,
                max_value=23,
                value=12,
                key="smart_submission_hour",
            )

    st.write("")

    analyze = st.button(
        "Analyze Claim",
        type="primary",
        use_container_width=True,
        key="smart_analyze_claim",
        disabled=invalid_dates,
    )

    if not analyze:
        return

    if prescription_state == "Yes":
        has_prescription = True

    elif prescription_state == "No":
        has_prescription = False

    else:
        has_prescription = None

    submission_datetime = datetime.combine(
        submission_date,
        time(
            hour=int(
                submission_hour
            )
        ),
    )

    try:
        with st.spinner(
            (
                "Retrieving historical context, building "
                "model features, scoring and explaining claim..."
            )
        ):
            claim = _build_smart_claim(
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

            _score_and_explain(
                client,
                claim,
                "Smart Analysis",
            )

        st.success(
            (
                "Claim successfully enriched, scored "
                "and explained with the deployed model."
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
    Score and explain one complete historical synthetic claim.
    """

    section_header(
        "Quick Demo",
        (
            "Run the deployed inference and explainability "
            "pipeline against a complete synthetic example."
        ),
    )

    try:
        demo_claims = load_demo_claims()

        if demo_claims.empty:
            st.info(
                (
                    "No demonstration claims "
                    "are available."
                )
            )
            return

        index = st.selectbox(
            "Demo claim",
            options=range(
                min(
                    len(demo_claims),
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

        raw_claim = get_demo_claim(
            int(index)
        )

        claim = _strip_leakage(
            raw_claim
        )

        if st.button(
            "Analyze Demo Claim",
            type="primary",
            use_container_width=True,
            key="analyze_demo_claim",
        ):
            with st.spinner(
                (
                    "Building features, scoring and "
                    "explaining demonstration claim..."
                )
            ):
                _score_and_explain(
                    client,
                    claim,
                    "Quick Demo",
                )

            st.success(
                (
                    "Demo claim scored and "
                    "explained successfully."
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
            str(exc)
        )


# =============================================================================
# Advanced JSON
# =============================================================================


def _render_advanced_json(
    client,
) -> None:
    """
    Render direct technical payload scoring and explanation.
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

    raw = st.text_area(
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

    analyze = st.button(
        "Analyze JSON",
        type="primary",
        use_container_width=True,
        key="analyze_json",
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
        claim = json.loads(
            raw
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

        clean_claim = _strip_leakage(
            claim
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
                "Validating payload, building features, "
                "scoring and explaining claim..."
            )
        ):
            _score_and_explain(
                client,
                clean_claim,
                "Advanced JSON",
            )

        st.success(
            (
                "JSON claim scored and "
                "explained successfully."
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
            str(exc)
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
            "using automatic historical enrichment, deployed "
            "fraud-risk scoring and local TreeSHAP explainability."
        ),
    )

    left, right = st.columns(
        [
            4,
            1,
        ]
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

    smart_tab, demo_tab, json_tab = st.tabs(
        [
            "Smart Analysis",
            "Quick Demo",
            "Advanced JSON",
        ]
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

    # Persistent assessment survives Streamlit reruns.
    _render_assessment()