from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

from time import (
    perf_counter,
)

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components import (
    empty_state,
    human_review_notice,
    info_panel,
    key_value_row,
    metric_card,
    mini_metric,
    section_header,
)

from utils.data import (
    load_demo_claims,
    read_uploaded_file,
)

from utils.formatting import (
    risk_tier,
)


# =============================================================================
# Configuration
# =============================================================================


MAX_QUEUE_SOURCE_SIZE = 10_000

DEFAULT_REVIEW_FRACTION = 0.03


LEAKAGE_COLUMNS = {
    "is_fraud",
    "latent_fraud_score",
    "synthetic_fraud_probability",
    "fraud_mechanism",
    "fraud_difficulty",
    "legitimate_anomaly",
    "legitimate_anomaly_type",
}


PRIORITY_ORDER = [
    "P1 — Immediate",
    "P2 — High",
    "P3 — Standard",
    "P4 — Monitor",
]


DECISION_OPTIONS = [
    "Pending review",
    "Investigate",
    "Escalate",
    "Clear",
]


QUEUE_WIDGET_KEYS = {
    "queue_file",
    "queue_search",
    "queue_priority_filter",
    "queue_tier_filter",
    "queue_service_filter",
    "queue_decision_filter",
    "queue_claim_detail_selector",
    "queue_capacity",
    "queue_demo_size",
}


# =============================================================================
# Generic helpers
# =============================================================================


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert numeric-like values into finite floats.
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
    Convert numeric-like values safely to integers.
    """

    try:
        if pd.isna(value):
            return default

        number = float(value)

        if not np.isfinite(number):
            return default

        return int(
            round(number)
        )

    except (
        TypeError,
        ValueError,
    ):
        return default


def _probability(
    value: Any,
    *,
    field: str = "probability",
) -> float:
    """
    Require a finite probability in [0, 1].

    Invalid model outputs are rejected rather than silently clipped.
    """

    try:
        result = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{field} must be numeric."
        ) from exc

    if not np.isfinite(result):
        raise ValueError(
            f"{field} must be finite."
        )

    if not 0.0 <= result <= 1.0:
        raise ValueError(
            (
                f"{field} must lie in "
                "the interval [0, 1]."
            )
        )

    return result


def _format_currency(
    value: Any,
) -> str:
    """
    Format monetary values consistently.
    """

    return (
        f"€{_safe_float(value):,.2f}"
    )


def _format_identifier(
    value: Any,
) -> str:
    """
    Normalize business identifiers for display.
    """

    if value is None:
        return "—"

    text = str(value).strip()

    return (
        text
        if text
        else "—"
    )


def _utc_timestamp() -> str:
    """
    Return an audit-friendly UTC timestamp.
    """

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat(
            timespec="seconds"
        )
    )


def _safe_model_value(
    value: Any,
) -> Any:
    """
    Convert pandas missing values to None for model metadata.
    """

    try:
        if pd.isna(value):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    return value


# =============================================================================
# Leakage protection
# =============================================================================


def _strip_leakage(
    claims: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    Remove target and synthetic-generation variables before inference.
    """

    clean_claims: list[
        dict[str, Any]
    ] = []

    for claim in claims:

        if not isinstance(
            claim,
            dict,
        ):
            raise TypeError(
                (
                    "Every portfolio item must "
                    "be a claim dictionary."
                )
            )

        clean_claims.append(
            {
                key: value
                for key, value
                in claim.items()
                if key not in LEAKAGE_COLUMNS
            }
        )

    return clean_claims


# =============================================================================
# Model interpretation
# =============================================================================


def _risk_priority(
    score: float,
) -> str:
    """
    Translate individual model risk into operational queue priority.

    This is a model-driven triage label, not an investigator decision.
    """

    score = _probability(
        score,
        field="fraud_risk_score",
    )

    if score >= 0.50:
        return "P1 — Immediate"

    if score >= 0.20:
        return "P2 — High"

    if score >= 0.05:
        return "P3 — Standard"

    return "P4 — Monitor"


def _model_recommendation(
    score: float,
) -> str:
    """
    Generate a human-readable model recommendation.
    """

    score = _probability(
        score,
        field="fraud_risk_score",
    )

    if score >= 0.50:
        return (
            "Immediate investigator review"
        )

    if score >= 0.20:
        return (
            "Investigator review recommended"
        )

    if score >= 0.05:
        return (
            "Review if capacity allows"
        )

    return (
        "Routine monitoring"
    )


def _priority_tone(
    priority: str,
) -> str:
    """
    Return the design-system tone associated with a queue priority.
    """

    if priority == "P1 — Immediate":
        return "danger"

    if priority == "P2 — High":
        return "warning"

    if priority == "P3 — Standard":
        return "info"

    return "success"


# =============================================================================
# Runtime contract
# =============================================================================


def _runtime_model_info(
    client,
) -> dict[str, Any]:
    """
    Retrieve deployed model metadata without blocking queue generation.
    """

    try:
        payload = client.model_info()

        if isinstance(
            payload,
            dict,
        ):
            return payload

    except Exception:
        pass

    return {}


def _runtime_review_fraction(
    model_info: dict[str, Any],
) -> float:
    """
    Resolve the deployed default review policy.
    """

    policy = model_info.get(
        "review_policy",
        {},
    )

    if isinstance(
        policy,
        dict,
    ):
        value = _safe_float(
            policy.get("fraction"),
            default=-1.0,
        )

        if 0 < value <= 1:
            return value

    return DEFAULT_REVIEW_FRACTION


# =============================================================================
# Session state
# =============================================================================


def _initialize_queue_state() -> None:
    """
    Initialize queue workflow state.

    Compatible with the global application state contract.
    """

    defaults = {
        "queue_results":
            None,

        "queue_metadata":
            None,

        "queue_source_claims":
            None,

        "queue_source_name":
            None,

        "queue_human_decisions":
            {},

        "queue_human_notes":
            {},

        "queue_decision_timestamps":
            {},

        "queue_selected_claim_id":
            None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = (
                value.copy()
                if isinstance(
                    value,
                    dict,
                )
                else value
            )


def _reset_queue() -> None:
    """
    Reset queue data and queue-specific widgets.
    """

    st.session_state.queue_results = None
    st.session_state.queue_metadata = None
    st.session_state.queue_source_claims = None
    st.session_state.queue_source_name = None

    st.session_state.queue_human_decisions = {}
    st.session_state.queue_human_notes = {}
    st.session_state.queue_decision_timestamps = {}

    st.session_state.queue_selected_claim_id = None

    for key in QUEUE_WIDGET_KEYS:

        if key in st.session_state:
            del st.session_state[key]


# =============================================================================
# Portfolio validation
# =============================================================================


def _claims_to_frame(
    claims: list[
        dict[str, Any]
    ],
) -> pd.DataFrame:
    """
    Convert source claims into a validated portfolio DataFrame.
    """

    if not claims:
        raise ValueError(
            "Portfolio contains no claims."
        )

    if len(claims) > MAX_QUEUE_SOURCE_SIZE:
        raise ValueError(
            (
                f"Portfolio contains {len(claims):,} claims. "
                f"The queue API supports at most "
                f"{MAX_QUEUE_SOURCE_SIZE:,} claims."
            )
        )

    invalid = [
        index
        for index, claim
        in enumerate(claims)
        if not isinstance(
            claim,
            dict,
        )
    ]

    if invalid:
        raise TypeError(
            (
                f"{len(invalid):,} portfolio row(s) "
                "are not claim objects."
            )
        )

    frame = pd.DataFrame(
        claims
    )

    if frame.empty:
        raise ValueError(
            "Portfolio contains no usable rows."
        )

    if "claim_id" not in frame.columns:
        raise ValueError(
            (
                "The portfolio must contain "
                "a claim_id field."
            )
        )

    frame["claim_id"] = (
        frame["claim_id"]
        .astype(str)
        .str.strip()
    )

    invalid_ids = (
        frame["claim_id"]
        .isin(
            [
                "",
                "nan",
                "None",
                "<NA>",
            ]
        )
    )

    if invalid_ids.any():
        raise ValueError(
            (
                "Portfolio contains one or more "
                "missing claim identifiers."
            )
        )

    duplicated = (
        frame["claim_id"]
        .duplicated(
            keep=False
        )
    )

    if duplicated.any():

        duplicates = (
            frame.loc[
                duplicated,
                "claim_id",
            ]
            .unique()
            .tolist()
        )

        preview = ", ".join(
            duplicates[:5]
        )

        raise ValueError(
            (
                "claim_id must be unique within "
                "the portfolio. Duplicate examples: "
                f"{preview}"
            )
        )

    return frame


# =============================================================================
# /top-review response validation
# =============================================================================


def _validate_predictions(
    predictions: pd.DataFrame,
    *,
    expected_claim_ids: set[str],
    review_fraction: float,
) -> pd.DataFrame:
    """
    Validate the selected population returned by /top-review.
    """

    if predictions.empty:
        raise RuntimeError(
            (
                "The model returned an empty "
                "investigation queue."
            )
        )

    required = {
        "claim_id",
        "fraud_risk_score",
        "risk_rank",
        "risk_percentile",
        "review_fraction",
        "selected_for_review",
    }

    missing = (
        required
        - set(predictions.columns)
    )

    if missing:
        raise RuntimeError(
            (
                "Queue response is missing required "
                "prediction fields: "
                + ", ".join(
                    sorted(missing)
                )
            )
        )

    frame = predictions.copy()

    # -------------------------------------------------------------------------
    # Claim identity
    # -------------------------------------------------------------------------

    frame["claim_id"] = (
        frame["claim_id"]
        .astype(str)
        .str.strip()
    )

    invalid_ids = frame["claim_id"].isin(
        [
            "",
            "nan",
            "None",
            "<NA>",
        ]
    )

    if invalid_ids.any():
        raise RuntimeError(
            (
                "Queue response contains one or "
                "more invalid claim IDs."
            )
        )

    if (
        frame["claim_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            (
                "Queue response contains "
                "duplicate claim IDs."
            )
        )

    returned_ids = set(
        frame["claim_id"]
        .tolist()
    )

    unexpected_ids = (
        returned_ids
        - expected_claim_ids
    )

    if unexpected_ids:
        examples = ", ".join(
            sorted(
                unexpected_ids
            )[:5]
        )

        raise RuntimeError(
            (
                "Queue API returned claim IDs that "
                "were not present in the source portfolio. "
                f"Examples: {examples}"
            )
        )

    # -------------------------------------------------------------------------
    # Scores
    # -------------------------------------------------------------------------

    frame[
        "fraud_risk_score"
    ] = pd.to_numeric(
        frame["fraud_risk_score"],
        errors="coerce",
    )

    if (
        frame["fraud_risk_score"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            (
                "Queue response contains invalid "
                "fraud-risk scores."
            )
        )

    scores = frame[
        "fraud_risk_score"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(scores).all():
        raise RuntimeError(
            (
                "Queue response contains "
                "non-finite fraud-risk scores."
            )
        )

    if (
        (scores < 0)
        | (scores > 1)
    ).any():
        raise RuntimeError(
            (
                "Queue response contains fraud-risk "
                "scores outside [0, 1]."
            )
        )

    # -------------------------------------------------------------------------
    # Ranks
    # -------------------------------------------------------------------------

    frame[
        "risk_rank"
    ] = pd.to_numeric(
        frame["risk_rank"],
        errors="coerce",
    )

    if (
        frame["risk_rank"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            (
                "Queue response contains "
                "invalid risk ranks."
            )
        )

    ranks = frame[
        "risk_rank"
    ].to_numpy(
        dtype=float
    )

    if not np.all(
        ranks
        == np.floor(ranks)
    ):
        raise RuntimeError(
            (
                "Queue risk_rank values "
                "must be integers."
            )
        )

    frame["risk_rank"] = (
        frame["risk_rank"]
        .astype(int)
    )

    expected_ranks = list(
        range(
            1,
            len(frame) + 1,
        )
    )

    sorted_ranks = sorted(
        frame["risk_rank"]
        .tolist()
    )

    if sorted_ranks != expected_ranks:
        raise RuntimeError(
            (
                "Queue risk ranks must form a "
                "continuous sequence starting at 1."
            )
        )

    # -------------------------------------------------------------------------
    # Percentiles
    # -------------------------------------------------------------------------

    frame[
        "risk_percentile"
    ] = pd.to_numeric(
        frame["risk_percentile"],
        errors="coerce",
    )

    if (
        frame["risk_percentile"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            (
                "Queue response contains invalid "
                "risk percentiles."
            )
        )

    percentiles = frame[
        "risk_percentile"
    ].to_numpy(
        dtype=float
    )

    if (
        (~np.isfinite(percentiles))
        |
        (percentiles <= 0)
        |
        (percentiles > 1)
    ).any():
        raise RuntimeError(
            (
                "risk_percentile must lie "
                "in the interval (0, 1]."
            )
        )

    # -------------------------------------------------------------------------
    # Review-policy consistency
    # -------------------------------------------------------------------------

    returned_fraction = pd.to_numeric(
        frame["review_fraction"],
        errors="coerce",
    )

    if returned_fraction.isna().any():
        raise RuntimeError(
            (
                "Queue response contains invalid "
                "review_fraction values."
            )
        )

    if not np.allclose(
        returned_fraction.to_numpy(
            dtype=float
        ),
        review_fraction,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            (
                "Queue prediction review_fraction "
                "does not match the requested capacity."
            )
        )

    if not (
        frame["selected_for_review"]
        .map(
            lambda value:
                value is True
        )
        .all()
    ):
        raise RuntimeError(
            (
                "Every claim returned by /top-review "
                "must be selected_for_review=True."
            )
        )

    # -------------------------------------------------------------------------
    # Ordering
    # -------------------------------------------------------------------------

    frame = (
        frame
        .sort_values(
            [
                "risk_rank",
                "fraud_risk_score",
            ],
            ascending=[
                True,
                False,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    scores_by_rank = (
        frame["fraud_risk_score"]
        .to_numpy()
    )

    if len(
        scores_by_rank
    ) > 1:

        if np.any(
            np.diff(
                scores_by_rank
            )
            > 1e-12
        ):
            raise RuntimeError(
                (
                    "Queue ranking is inconsistent "
                    "with descending fraud-risk scores."
                )
            )

    return frame


# =============================================================================
# Queue enrichment
# =============================================================================


def _enrich_queue(
    predictions: pd.DataFrame,
    claims: list[
        dict[str, Any]
    ],
) -> pd.DataFrame:
    """
    Merge selected predictions with business attributes
    and persisted investigator state.
    """

    source = _claims_to_frame(
        claims
    )

    expected_claim_ids = set(
        source["claim_id"]
        .astype(str)
        .tolist()
    )

    review_fraction = _safe_float(
        predictions[
            "review_fraction"
        ]
        .iloc[0],
        default=-1.0,
    )

    queue = _validate_predictions(
        predictions,
        expected_claim_ids=(
            expected_claim_ids
        ),
        review_fraction=(
            review_fraction
        ),
    )

    business_columns = [
        "claim_id",
        "customer_id",
        "policy_id",
        "provider_id",
        "service_category",
        "service_code",
        "claim_amount",
        "requested_reimbursement",
        "coverage_limit",
        "submission_channel",
        "claim_submission_timestamp",
        "service_date",
        "customer_age",
        "provider_type",
        "provider_region",
    ]

    available_columns = [
        column
        for column in business_columns
        if column in source.columns
    ]

    source_context = (
        source[
            available_columns
        ]
        .copy()
    )

    duplicate_business_columns = [
        column
        for column in available_columns
        if (
            column != "claim_id"
            and column in queue.columns
        )
    ]

    if duplicate_business_columns:
        queue = queue.drop(
            columns=(
                duplicate_business_columns
            )
        )

    queue = queue.merge(
        source_context,
        on="claim_id",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # Derived operational interpretation
    # -------------------------------------------------------------------------

    queue["risk_tier"] = (
        queue["fraud_risk_score"]
        .apply(
            risk_tier
        )
    )

    queue["priority"] = (
        queue["fraud_risk_score"]
        .apply(
            _risk_priority
        )
    )

    queue[
        "model_recommendation"
    ] = (
        queue["fraud_risk_score"]
        .apply(
            _model_recommendation
        )
    )

    # -------------------------------------------------------------------------
    # Persistent human state
    # -------------------------------------------------------------------------

    decisions = (
        st.session_state
        .queue_human_decisions
    )

    notes = (
        st.session_state
        .queue_human_notes
    )

    timestamps = (
        st.session_state
        .queue_decision_timestamps
    )

    queue["human_decision"] = (
        queue["claim_id"]
        .map(decisions)
        .fillna(
            "Pending review"
        )
    )

    queue["investigator_note"] = (
        queue["claim_id"]
        .map(notes)
        .fillna("")
    )

    queue[
        "decision_updated_at"
    ] = (
        queue["claim_id"]
        .map(timestamps)
        .fillna("")
    )

    return (
        queue
        .sort_values(
            "risk_rank",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# Queue generation
# =============================================================================


def _generate_queue(
    client,
    claims: list[
        dict[str, Any]
    ],
    capacity: float,
    source_name: str,
    source_type: str,
) -> None:
    """
    Build the authoritative investigation queue through /top-review.
    """

    capacity = _probability(
        capacity,
        field="review_fraction",
    )

    if capacity <= 0:
        raise ValueError(
            (
                "Investigation capacity must "
                "be greater than zero."
            )
        )

    clean_claims = _strip_leakage(
        claims
    )

    source_frame = _claims_to_frame(
        clean_claims
    )

    expected_claim_ids = set(
        source_frame["claim_id"]
        .astype(str)
        .tolist()
    )

    model_info = (
        _runtime_model_info(
            client
        )
    )

    started = (
        perf_counter()
    )

    with st.spinner(
        (
            "Scoring portfolio, ranking claims "
            "and building the investigation queue..."
        )
    ):
        response = client.top_review(
            clean_claims,
            capacity,
        )

    elapsed_seconds = max(
        perf_counter()
        - started,
        0.0,
    )

    if not isinstance(
        response,
        dict,
    ):
        raise RuntimeError(
            (
                "The API returned an invalid "
                "queue response."
            )
        )

    # -------------------------------------------------------------------------
    # Response envelope
    # -------------------------------------------------------------------------

    required_envelope = {
        "total_claims",
        "selected_claims",
        "review_fraction",
        "predictions",
    }

    missing = (
        required_envelope
        - set(response)
    )

    if missing:
        raise RuntimeError(
            (
                "Queue API response is missing: "
                + ", ".join(
                    sorted(missing)
                )
            )
        )

    total_claims = _safe_int(
        response.get(
            "total_claims"
        ),
        default=-1,
    )

    selected_claims = _safe_int(
        response.get(
            "selected_claims"
        ),
        default=-1,
    )

    returned_capacity = _safe_float(
        response.get(
            "review_fraction"
        ),
        default=-1.0,
    )

    if (
        total_claims
        != len(clean_claims)
    ):
        raise RuntimeError(
            (
                "Queue API total_claims does not "
                "match the submitted portfolio size."
            )
        )

    if not np.isclose(
        returned_capacity,
        capacity,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            (
                "Queue API review_fraction differs "
                "from the requested capacity."
            )
        )

    raw_predictions = (
        response.get(
            "predictions"
        )
    )

    if not isinstance(
        raw_predictions,
        list,
    ):
        raise RuntimeError(
            (
                "Queue API response does not contain "
                "a valid predictions list."
            )
        )

    if (
        selected_claims
        != len(raw_predictions)
    ):
        raise RuntimeError(
            (
                "Queue API selected_claims does not "
                "match the returned prediction count."
            )
        )

    if selected_claims <= 0:
        raise RuntimeError(
            (
                "The model returned an empty "
                "investigation queue."
            )
        )

    predictions = pd.DataFrame(
        raw_predictions
    )

    predictions = _validate_predictions(
        predictions,
        expected_claim_ids=(
            expected_claim_ids
        ),
        review_fraction=(
            capacity
        ),
    )

    enriched = _enrich_queue(
        predictions,
        clean_claims,
    )

    if len(enriched) != selected_claims:
        raise RuntimeError(
            (
                "Queue enrichment changed the number "
                "of selected investigation claims."
            )
        )

    throughput = (
        len(clean_claims)
        / elapsed_seconds
        if elapsed_seconds > 0
        else None
    )

    st.session_state.queue_results = (
        enriched
    )

    st.session_state.queue_source_claims = (
        clean_claims
    )

    st.session_state.queue_source_name = (
        str(source_name)
    )

    st.session_state.queue_metadata = {
        "total_claims":
            total_claims,

        "selected_claims":
            selected_claims,

        "capacity":
            capacity,

        "generated_at":
            _utc_timestamp(),

        "source_name":
            str(source_name),

        "source_type":
            str(source_type),

        "generation_seconds":
            float(
                elapsed_seconds
            ),

        "throughput_claims_per_second":
            (
                float(throughput)
                if throughput is not None
                else None
            ),

        "mean_selected_risk":
            float(
                enriched[
                    "fraud_risk_score"
                ]
                .mean()
            ),

        "max_selected_risk":
            float(
                enriched[
                    "fraud_risk_score"
                ]
                .max()
            ),

        "model_name":
            model_info.get(
                "model_name"
            ),

        "model_version":
            model_info.get(
                "model_version"
            ),

        "target":
            model_info.get(
                "target"
            ),

        "feature_count":
            model_info.get(
                "feature_count"
            ),

        "transformed_feature_count":
            model_info.get(
                "transformed_feature_count"
            ),

        "probability_method":
            model_info.get(
                "probability_method"
            ),

        "review_policy":
            model_info.get(
                "review_policy"
            ),

        "explainability":
            model_info.get(
                "explainability"
            ),
    }

    st.session_state.queue_selected_claim_id = (
        None
    )


# =============================================================================
# Queue builder
# =============================================================================


def _render_queue_builder(
    client,
) -> None:
    """
    Render portfolio source and investigation-capacity controls.
    """

    section_header(
        "Build Investigation Queue",
        (
            "Score a portfolio and select the "
            "highest-risk claims according to "
            "available investigation capacity."
        ),
        eyebrow="QUEUE GENERATION",
    )

    model_info = (
        _runtime_model_info(
            client
        )
    )

    default_capacity = (
        _runtime_review_fraction(
            model_info
        )
    )

    default_capacity_percent = int(
        round(
            default_capacity
            * 100
        )
    )

    default_capacity_percent = min(
        max(
            default_capacity_percent,
            1,
        ),
        25,
    )

    control_left, control_right = (
        st.columns(
            [
                3,
                1,
            ]
        )
    )

    with control_left:

        capacity_percent = (
            st.slider(
                "Investigation capacity",
                min_value=1,
                max_value=25,
                value=(
                    default_capacity_percent
                ),
                step=1,
                key="queue_capacity",
                help=(
                    "Percentage of the scored portfolio "
                    "that investigators can review."
                ),
            )
        )

    capacity = (
        capacity_percent
        / 100
    )

    with control_right:

        metric_card(
            "Review Capacity",
            f"{capacity:.0%}",
            "Portfolio selection rate",
            tone="info",
        )

    st.write("")

    info_panel(
        "Operational Policy",
        (
            "The backend ranks the complete portfolio and returns "
            f"the highest-risk claims at {capacity:.0%} review "
            "capacity. Queue inclusion supports prioritization only "
            "and does not establish that fraud occurred."
        ),
        tone="info",
    )

    if (
        abs(
            capacity
            - default_capacity
        )
        > 1e-12
    ):
        st.caption(
            (
                "Current queue capacity differs from the "
                f"deployed default policy ({default_capacity:.1%})."
            )
        )

    st.write("")

    upload_tab, demo_tab = (
        st.tabs(
            [
                "Upload Portfolio",
                "Demo Portfolio",
            ]
        )
    )

    # -------------------------------------------------------------------------
    # Uploaded portfolio
    # -------------------------------------------------------------------------

    with upload_tab:

        uploaded = (
            st.file_uploader(
                "Upload claims",
                type=[
                    "json",
                    "csv",
                    "parquet",
                ],
                key="queue_file",
                help=(
                    "Supported formats: JSON, CSV and Parquet. "
                    f"Maximum {MAX_QUEUE_SOURCE_SIZE:,} claims."
                ),
            )
        )

        if uploaded is None:

            st.caption(
                (
                    "Upload a complete portfolio "
                    "to create a prioritized queue."
                )
            )

        else:

            try:

                claims = read_uploaded_file(
                    uploaded
                )

                preview = _claims_to_frame(
                    claims
                )

                selected_estimate = min(
                    len(preview),
                    max(
                        1,
                        int(
                            np.ceil(
                                len(preview)
                                * capacity
                            )
                        ),
                    ),
                )

                leakage_present = [
                    column
                    for column
                    in LEAKAGE_COLUMNS
                    if column
                    in preview.columns
                ]

                p1, p2, p3, p4 = (
                    st.columns(4)
                )

                with p1:

                    metric_card(
                        "Claims Detected",
                        f"{len(preview):,}",
                        "Valid portfolio records",
                    )

                with p2:

                    metric_card(
                        "Expected Queue",
                        f"{selected_estimate:,}",
                        (
                            f"At {capacity:.0%} "
                            "review capacity"
                        ),
                        tone="info",
                    )

                with p3:

                    missing_values = int(
                        preview
                        .isna()
                        .sum()
                        .sum()
                    )

                    metric_card(
                        "Missing Values",
                        f"{missing_values:,}",
                        "Across source portfolio",
                        tone=(
                            "warning"
                            if missing_values
                            else "success"
                        ),
                    )

                with p4:

                    metric_card(
                        "Input Columns",
                        f"{len(preview.columns):,}",
                        "Available source fields",
                    )

                if leakage_present:

                    st.write("")

                    info_panel(
                        "Protected Evaluation Fields Detected",
                        (
                            f"{len(leakage_present)} synthetic/evaluation "
                            "field(s) were found and will be removed "
                            "automatically before inference."
                        ),
                        tone="info",
                    )

                with st.expander(
                    "Portfolio preview",
                    expanded=False,
                ):

                    st.dataframe(
                        preview.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

                if st.button(
                    "Build Investigation Queue",
                    type="primary",
                    use_container_width=True,
                    key="build_uploaded_queue",
                ):

                    _generate_queue(
                        client=client,
                        claims=claims,
                        capacity=capacity,
                        source_name=(
                            uploaded.name
                        ),
                        source_type="uploaded",
                    )

                    st.success(
                        (
                            "Investigation queue "
                            "created successfully."
                        )
                    )

                    st.rerun()

            except Exception as exc:

                st.error(
                    (
                        "Unable to process portfolio. "
                        f"{exc}"
                    )
                )

    # -------------------------------------------------------------------------
    # Demo portfolio
    # -------------------------------------------------------------------------

    with demo_tab:

        st.caption(
            (
                "Run the complete prioritization workflow "
                "using the synthetic portfolio bundled with "
                "the project."
            )
        )

        demo_size = (
            st.select_slider(
                "Demo portfolio size",
                options=[
                    100,
                    250,
                    500,
                    1_000,
                    2_500,
                    5_000,
                    10_000,
                ],
                value=500,
                key="queue_demo_size",
            )
        )

        estimated_selection = min(
            int(demo_size),
            max(
                1,
                int(
                    np.ceil(
                        int(demo_size)
                        * capacity
                    )
                ),
            ),
        )

        st.caption(
            (
                f"Approximately {estimated_selection:,} "
                f"of {int(demo_size):,} claims will be "
                f"selected at {capacity:.0%} capacity."
            )
        )

        if st.button(
            "Build Demo Queue",
            type="primary",
            use_container_width=True,
            key="build_demo_queue",
        ):

            try:

                demo = load_demo_claims(
                    limit=int(
                        demo_size
                    )
                )

                claims = demo.to_dict(
                    orient="records"
                )

                _generate_queue(
                    client=client,
                    claims=claims,
                    capacity=capacity,
                    source_name=(
                        "Synthetic demo portfolio"
                    ),
                    source_type="demo",
                )

                st.success(
                    (
                        "Demo investigation queue "
                        "created successfully."
                    )
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    str(exc)
                )


# =============================================================================
# Queue integrity
# =============================================================================


def _validate_stored_queue(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[str]:
    """
    Validate the queue snapshot persisted in Streamlit session state.
    """

    errors: list[str] = []

    required = {
        "claim_id",
        "fraud_risk_score",
        "risk_rank",
        "risk_percentile",
        "review_fraction",
        "selected_for_review",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:

        errors.append(
            (
                "Stored queue is missing: "
                + ", ".join(
                    sorted(missing)
                )
            )
        )

        return errors

    if frame.empty:

        errors.append(
            "Stored queue is empty."
        )

        return errors

    if (
        frame["claim_id"]
        .astype(str)
        .duplicated()
        .any()
    ):
        errors.append(
            (
                "Stored queue contains "
                "duplicate claim identifiers."
            )
        )

    expected_selected = _safe_int(
        metadata.get(
            "selected_claims"
        ),
        default=-1,
    )

    if (
        expected_selected >= 0
        and expected_selected
        != len(frame)
    ):
        errors.append(
            (
                "Stored queue row count differs "
                "from queue metadata."
            )
        )

    return errors


# =============================================================================
# Executive summary
# =============================================================================


def _render_summary(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """
    Render executive operational KPIs.
    """

    section_header(
        "Queue Overview",
        (
            "Operational summary of claims "
            "currently selected for human review."
        ),
        eyebrow="SELECTED WORKLIST",
    )

    total = _safe_int(
        metadata.get(
            "total_claims"
        )
    )

    selected = _safe_int(
        metadata.get(
            "selected_claims"
        )
    )

    capacity = _safe_float(
        metadata.get(
            "capacity"
        )
    )

    mean_risk = _safe_float(
        frame["fraud_risk_score"]
        .mean()
    )

    max_risk = _safe_float(
        frame["fraud_risk_score"]
        .max()
    )

    immediate_count = int(
        (
            frame["priority"]
            == "P1 — Immediate"
        )
        .sum()
    )

    pending_count = int(
        (
            frame["human_decision"]
            == "Pending review"
        )
        .sum()
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Portfolio",
            f"{total:,}",
            "Total claims ranked",
        )

    with c2:

        metric_card(
            "Selected",
            f"{selected:,}",
            (
                f"Top {capacity:.0%} "
                "for review"
            ),
            tone="info",
        )

    with c3:

        metric_card(
            "Mean Selected Risk",
            f"{mean_risk:.2%}",
            "Current worklist",
            tone="warning",
        )

    with c4:

        metric_card(
            "Immediate Priority",
            f"{immediate_count:,}",
            (
                f"Max risk {max_risk:.2%}"
            ),
            tone=(
                "danger"
                if immediate_count
                else "success"
            ),
        )

    st.write("")

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:

        metric_card(
            "Pending Reviews",
            f"{pending_count:,}",
            "Awaiting investigator action",
            tone=(
                "warning"
                if pending_count
                else "success"
            ),
        )

    generation_seconds = _safe_float(
        metadata.get(
            "generation_seconds"
        ),
        default=-1.0,
    )

    with c2:

        metric_card(
            "Generation Runtime",
            (
                f"{generation_seconds:.2f}s"
                if generation_seconds >= 0
                else "—"
            ),
            "End-to-end queue creation",
            tone="info",
        )

    throughput = _safe_float(
        metadata.get(
            "throughput_claims_per_second"
        ),
        default=-1.0,
    )

    with c3:

        metric_card(
            "Throughput",
            (
                f"{throughput:,.0f}/s"
                if throughput >= 0
                else "—"
            ),
            "Portfolio claims processed",
            tone="neutral",
        )

    source_name = (
        metadata.get(
            "source_name"
        )
        or st.session_state.get(
            "queue_source_name"
        )
        or "—"
    )

    generated_at = metadata.get(
        "generated_at",
        "—",
    )

    st.caption(
        (
            f"Source: {source_name} • "
            f"Queue generated: {generated_at} • "
            f"Model: {metadata.get('model_name') or '—'} "
            f"v{metadata.get('model_version') or '—'}"
        )
    )


# =============================================================================
# Runtime contract
# =============================================================================


def _render_queue_contract(
    metadata: dict[str, Any],
) -> None:
    """
    Render the model/policy contract used to generate the queue.
    """

    st.write("")
    st.write("")

    section_header(
        "Queue Contract",
        (
            "Runtime model and review policy used when "
            "the current investigation queue was generated."
        ),
        eyebrow="TRACEABILITY",
    )

    explainability = (
        metadata.get(
            "explainability"
        )
    )

    if not isinstance(
        explainability,
        dict,
    ):
        explainability = {}

    c1, c2, c3 = st.columns(3)

    with c1:

        mini_metric(
            "Model",
            str(
                metadata.get(
                    "model_name"
                )
                or "—"
            ),
            helper=(
                str(
                    metadata.get(
                        "model_version"
                    )
                    or "Version unavailable"
                )
            ),
            tone="info",
        )

    with c2:

        mini_metric(
            "Source Features",
            str(
                metadata.get(
                    "feature_count"
                )
                or "—"
            ),
            helper="Frozen model contract",
            tone="neutral",
        )

    with c3:

        mini_metric(
            "Transformed Features",
            str(
                metadata.get(
                    "transformed_feature_count"
                )
                or "—"
            ),
            helper="Inference feature space",
            tone="neutral",
        )

    st.write("")

    with st.expander(
        "Runtime contract details",
        expanded=False,
    ):

        key_value_row(
            "Prediction target",
            str(
                metadata.get(
                    "target"
                )
                or "—"
            ),
            monospace=True,
        )

        key_value_row(
            "Probability method",
            str(
                metadata.get(
                    "probability_method"
                )
                or "—"
            ),
            monospace=True,
        )

        key_value_row(
            "Queue capacity",
            (
                f"{_safe_float(metadata.get('capacity')):.1%}"
            ),
        )

        if explainability:

            key_value_row(
                "Explainability",
                str(
                    explainability.get(
                        "method"
                    )
                    or "Unavailable"
                ),
            )

            key_value_row(
                "Explanation space",
                str(
                    explainability.get(
                        "output_space"
                    )
                    or "—"
                ),
                monospace=True,
            )


# =============================================================================
# Priority distribution
# =============================================================================


def _render_priority_distribution(
    frame: pd.DataFrame,
) -> None:
    """
    Render model-driven queue-priority composition.
    """

    st.write("")
    st.write("")

    section_header(
        "Priority Distribution",
        (
            "Model-driven operational segmentation "
            "within the selected review population."
        ),
        eyebrow="TRIAGE",
    )

    counts = (
        frame["priority"]
        .value_counts()
        .reindex(
            PRIORITY_ORDER,
            fill_value=0,
        )
    )

    helpers = {
        "P1 — Immediate":
            "≥ 50% model risk",

        "P2 — High":
            "20–50% model risk",

        "P3 — Standard":
            "5–20% model risk",

        "P4 — Monitor":
            "< 5% model risk",
    }

    columns = st.columns(4)

    for column, priority in zip(
        columns,
        PRIORITY_ORDER,
    ):

        with column:

            metric_card(
                priority,
                f"{int(counts[priority]):,}",
                helpers[priority],
                tone=(
                    _priority_tone(
                        priority
                    )
                ),
            )


# =============================================================================
# Decision progress
# =============================================================================


def _render_decision_progress(
    frame: pd.DataFrame,
) -> None:
    """
    Render human-review workflow progress independently of model priority.
    """

    st.write("")
    st.write("")

    section_header(
        "Human Review Progress",
        (
            "Investigator decisions are tracked separately "
            "from model scores and model recommendations."
        ),
        eyebrow="HUMAN WORKFLOW",
    )

    counts = (
        frame["human_decision"]
        .value_counts()
        .reindex(
            DECISION_OPTIONS,
            fill_value=0,
        )
    )

    total = max(
        len(frame),
        1,
    )

    columns = st.columns(
        len(
            DECISION_OPTIONS
        )
    )

    tones = {
        "Pending review": "warning",
        "Investigate": "info",
        "Escalate": "danger",
        "Clear": "success",
    }

    for column, decision in zip(
        columns,
        DECISION_OPTIONS,
    ):

        count = int(
            counts[
                decision
            ]
        )

        with column:

            metric_card(
                decision,
                f"{count:,}",
                (
                    f"{count / total:.1%} "
                    "of selected queue"
                ),
                tone=tones[
                    decision
                ],
            )


# =============================================================================
# Filters
# =============================================================================


def _render_filters(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter the queue without mutating model or human-review state.
    """

    st.write("")
    st.write("")

    section_header(
        "Investigation Worklist",
        (
            "Search and filter selected claims "
            "before investigator review."
        ),
        eyebrow="OPERATIONS",
    )

    row1_col1, row1_col2 = (
        st.columns(2)
    )

    with row1_col1:

        search = st.text_input(
            "Search",
            placeholder=(
                "Claim, customer or provider ID"
            ),
            key="queue_search",
        )

    with row1_col2:

        priority_options = [
            priority
            for priority
            in PRIORITY_ORDER
            if priority
            in frame["priority"]
            .unique()
        ]

        selected_priorities = (
            st.multiselect(
                "Priority",
                options=(
                    priority_options
                ),
                default=(
                    priority_options
                ),
                key=(
                    "queue_priority_filter"
                ),
            )
        )

    row2_col1, row2_col2, row2_col3 = (
        st.columns(3)
    )

    tiers = [
        value
        for value in [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        ]
        if value
        in frame["risk_tier"]
        .unique()
    ]

    with row2_col1:

        selected_tiers = (
            st.multiselect(
                "Risk tier",
                options=tiers,
                default=tiers,
                key="queue_tier_filter",
            )
        )

    with row2_col2:

        if (
            "service_category"
            in frame.columns
        ):

            service_options = (
                frame[
                    "service_category"
                ]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )

            selected_services = (
                st.multiselect(
                    "Service category",
                    options=(
                        service_options
                    ),
                    key=(
                        "queue_service_filter"
                    ),
                )
            )

        else:

            selected_services = []

    with row2_col3:

        selected_decisions = (
            st.multiselect(
                "Human decision",
                options=(
                    DECISION_OPTIONS
                ),
                key=(
                    "queue_decision_filter"
                ),
            )
        )

    display = frame.copy()

    if selected_priorities:

        display = display.loc[
            display["priority"]
            .isin(
                selected_priorities
            )
        ]

    else:

        display = display.iloc[0:0]

    if selected_tiers:

        display = display.loc[
            display["risk_tier"]
            .isin(
                selected_tiers
            )
        ]

    else:

        display = display.iloc[0:0]

    if (
        selected_services
        and "service_category"
        in display.columns
    ):

        display = display.loc[
            display["service_category"]
            .astype(str)
            .isin(
                selected_services
            )
        ]

    if selected_decisions:

        display = display.loc[
            display["human_decision"]
            .isin(
                selected_decisions
            )
        ]

    if search:

        normalized = (
            search
            .strip()
            .lower()
        )

        searchable_columns = [
            column
            for column in [
                "claim_id",
                "customer_id",
                "provider_id",
            ]
            if column
            in display.columns
        ]

        mask = pd.Series(
            False,
            index=display.index,
        )

        for column in searchable_columns:

            mask = (
                mask
                |
                display[column]
                .astype(str)
                .str.lower()
                .str.contains(
                    normalized,
                    regex=False,
                    na=False,
                )
            )

        display = display.loc[
            mask
        ]

    st.caption(
        (
            f"{len(display):,} of "
            f"{len(frame):,} selected claims shown"
        )
    )

    return (
        display
        .sort_values(
            "risk_rank"
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# Worklist table
# =============================================================================


def _render_worklist_table(
    display: pd.DataFrame,
) -> None:
    """
    Render the selected investigation worklist.
    """

    if display.empty:

        empty_state(
            "No Matching Claims",
            (
                "No queue item matches "
                "the current filters."
            ),
            hint=(
                "Adjust the search or "
                "filter criteria."
            ),
        )

        return

    preferred_columns = [
        "risk_rank",
        "claim_id",
        "fraud_risk_score",
        "risk_percentile",
        "priority",
        "risk_tier",
        "model_recommendation",
        "human_decision",
        "claim_amount",
        "requested_reimbursement",
        "service_category",
        "provider_id",
        "customer_id",
    ]

    columns = [
        column
        for column
        in preferred_columns
        if column
        in display.columns
    ]

    st.dataframe(
        display[
            columns
        ],
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "risk_rank":
                st.column_config.NumberColumn(
                    "Rank",
                    format="%d",
                    width="small",
                ),

            "claim_id":
                st.column_config.TextColumn(
                    "Claim",
                    width="medium",
                ),

            "fraud_risk_score":
                st.column_config.ProgressColumn(
                    "Fraud Risk",
                    min_value=0,
                    max_value=1,
                    format="%.3f",
                ),

            "risk_percentile":
                st.column_config.NumberColumn(
                    "Risk Percentile",
                    format="%.1%%",
                ),

            "priority":
                st.column_config.TextColumn(
                    "Priority",
                    width="medium",
                ),

            "risk_tier":
                st.column_config.TextColumn(
                    "Risk Tier",
                    width="small",
                ),

            "model_recommendation":
                st.column_config.TextColumn(
                    "Model Recommendation",
                    width="large",
                ),

            "human_decision":
                st.column_config.TextColumn(
                    "Human Decision",
                    width="medium",
                ),

            "claim_amount":
                st.column_config.NumberColumn(
                    "Claim Amount",
                    format="€ %.2f",
                ),

            "requested_reimbursement":
                st.column_config.NumberColumn(
                    "Requested",
                    format="€ %.2f",
                ),
        },
    )


# =============================================================================
# Claim lookup
# =============================================================================


def _selected_source_claim(
    claim_id: str,
) -> dict[str, Any] | None:
    """
    Retrieve the complete source claim used during queue generation.
    """

    claims = (
        st.session_state.get(
            "queue_source_claims"
        )
        or []
    )

    for claim in claims:

        if (
            str(
                claim.get(
                    "claim_id"
                )
            )
            == str(
                claim_id
            )
        ):
            return claim

    return None


# =============================================================================
# Claim Analysis transfer
# =============================================================================


def _open_claim_analysis(
    row: pd.Series,
) -> None:
    """
    Transfer a queue claim into Claim Analysis.

    The full source claim is preserved so Claim Analysis can request
    a fresh local TreeSHAP explanation through /explain.
    """

    claim_id = str(
        row["claim_id"]
    )

    source_claim = (
        _selected_source_claim(
            claim_id
        )
    )

    if source_claim is None:
        raise ValueError(
            (
                "The source claim payload "
                "could not be recovered."
            )
        )

    score = _probability(
        row["fraud_risk_score"],
        field="fraud_risk_score",
    )

    metadata = (
        st.session_state.get(
            "queue_metadata"
        )
        or {}
    )

    model_name = _safe_model_value(
        row.get(
            "model_name"
        )
    )

    model_version = _safe_model_value(
        row.get(
            "model_version"
        )
    )

    prediction = {
        "claim_id":
            claim_id,

        "fraud_risk_score":
            score,

        "model_name":
            (
                model_name
                or metadata.get(
                    "model_name"
                )
                or "—"
            ),

        "model_version":
            (
                model_version
                or metadata.get(
                    "model_version"
                )
                or "—"
            ),
    }

    st.session_state.single_prediction = (
        prediction
    )

    st.session_state.single_score = (
        score
    )

    st.session_state.single_claim = (
        source_claim
    )

    st.session_state.single_source = (
        "Investigation Queue"
    )

    # Do not preserve a stale explanation from another claim.
    st.session_state.single_explanation = (
        None
    )

    st.session_state.main_navigation = (
        "Claim Analysis"
    )


# =============================================================================
# Claim review
# =============================================================================


def _render_claim_detail(
    frame: pd.DataFrame,
) -> None:
    """
    Inspect one queue case and persist the investigator decision.
    """

    if frame.empty:
        return

    st.write("")
    st.write("")

    section_header(
        "Claim Review",
        (
            "Inspect an individual queue item, compare "
            "the model recommendation and record the "
            "investigator's independent decision."
        ),
        eyebrow="CASE REVIEW",
    )

    lookup = (
        frame
        .assign(
            _claim_id_text=(
                frame["claim_id"]
                .astype(str)
            )
        )
        .set_index(
            "_claim_id_text",
            drop=False,
        )
    )

    claim_options = (
        frame["claim_id"]
        .astype(str)
        .tolist()
    )

    def _claim_label(
        claim_id: str,
    ) -> str:

        item = lookup.loc[
            str(
                claim_id
            )
        ]

        if isinstance(
            item,
            pd.DataFrame,
        ):
            item = item.iloc[0]

        rank = _safe_int(
            item.get(
                "risk_rank"
            )
        )

        score = _probability(
            item.get(
                "fraud_risk_score"
            ),
            field="fraud_risk_score",
        )

        return (
            f"#{rank} • "
            f"{claim_id} • "
            f"{score:.2%}"
        )

    selected_claim_id = st.selectbox(
        "Open claim",
        options=claim_options,
        key=(
            "queue_claim_detail_selector"
        ),
        format_func=_claim_label,
    )

    st.session_state.queue_selected_claim_id = (
        selected_claim_id
    )

    row = lookup.loc[
        str(
            selected_claim_id
        )
    ]

    if isinstance(
        row,
        pd.DataFrame,
    ):
        row = row.iloc[0]

    score = _probability(
        row.get(
            "fraud_risk_score"
        ),
        field="fraud_risk_score",
    )

    priority = str(
        row.get(
            "priority",
            "P4 — Monitor",
        )
    )

    tone = _priority_tone(
        priority
    )

    left, right = st.columns(
        [
            1.25,
            1,
        ],
        gap="large",
    )

    # -------------------------------------------------------------------------
    # Model assessment
    # -------------------------------------------------------------------------

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Model Assessment"
            )

            a1, a2, a3 = st.columns(
                [
                    .8,
                    1,
                    1.2,
                ]
            )

            with a1:

                metric_card(
                    "Queue Rank",
                    (
                        f"#{_safe_int(row.get('risk_rank'))}"
                    ),
                    "Model ordering",
                    tone=tone,
                )

            with a2:

                metric_card(
                    "Fraud Risk",
                    f"{score:.2%}",
                    "Individual model score",
                    tone=tone,
                )

            with a3:

                metric_card(
                    "Priority",
                    priority,
                    "Model-driven triage",
                    tone=tone,
                )

            st.write("")
            st.divider()

            st.caption(
                "MODEL RECOMMENDATION"
            )

            st.write(
                str(
                    row.get(
                        "model_recommendation",
                        "—",
                    )
                )
            )

            st.caption(
                (
                    "Predictive recommendation only; "
                    "it is not an adjudication decision."
                )
            )

            st.divider()

            detail_left, detail_right = (
                st.columns(2)
            )

            with detail_left:

                st.caption(
                    "CLAIM"
                )

                st.code(
                    _format_identifier(
                        row.get(
                            "claim_id"
                        )
                    ),
                    language=None,
                )

                if (
                    "claim_amount"
                    in row.index
                ):

                    st.write(
                        (
                            "**Claim amount:** "
                            f"{_format_currency(row.get('claim_amount'))}"
                        )
                    )

                if (
                    "requested_reimbursement"
                    in row.index
                ):

                    st.write(
                        (
                            "**Requested:** "
                            f"{_format_currency(row.get('requested_reimbursement'))}"
                        )
                    )

            with detail_right:

                if (
                    "service_category"
                    in row.index
                ):

                    st.write(
                        (
                            "**Service:** "
                            f"{_format_identifier(row.get('service_category'))}"
                        )
                    )

                if (
                    "provider_id"
                    in row.index
                ):

                    st.write(
                        (
                            "**Provider:** "
                            f"`{_format_identifier(row.get('provider_id'))}`"
                        )
                    )

                if (
                    "customer_id"
                    in row.index
                ):

                    st.write(
                        (
                            "**Customer:** "
                            f"`{_format_identifier(row.get('customer_id'))}`"
                        )
                    )

            st.write("")

            if st.button(
                "Open Full Claim Analysis",
                type="primary",
                use_container_width=True,
                key=(
                    "queue_open_claim_analysis_"
                    + str(
                        selected_claim_id
                    )
                ),
            ):

                try:

                    _open_claim_analysis(
                        row
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        (
                            "Unable to open Claim Analysis. "
                            f"{exc}"
                        )
                    )

    # -------------------------------------------------------------------------
    # Human decision
    # -------------------------------------------------------------------------

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Investigator Decision"
            )

            claim_key = str(
                selected_claim_id
            )

            current_decision = (
                st.session_state
                .queue_human_decisions
                .get(
                    claim_key,
                    "Pending review",
                )
            )

            current_index = (
                DECISION_OPTIONS.index(
                    current_decision
                )
                if current_decision
                in DECISION_OPTIONS
                else 0
            )

            decision = st.selectbox(
                "Human decision",
                options=(
                    DECISION_OPTIONS
                ),
                index=current_index,
                key=(
                    f"decision_"
                    f"{claim_key}"
                ),
            )

            existing_note = (
                st.session_state
                .queue_human_notes
                .get(
                    claim_key,
                    "",
                )
            )

            investigation_note = (
                st.text_area(
                    "Investigation note",
                    value=existing_note,
                    placeholder=(
                        "Document relevant review observations..."
                    ),
                    height=140,
                    key=(
                        f"note_"
                        f"{claim_key}"
                    ),
                )
            )

            previous_timestamp = (
                st.session_state
                .queue_decision_timestamps
                .get(
                    claim_key
                )
            )

            if previous_timestamp:

                st.caption(
                    (
                        "Last saved decision: "
                        f"{previous_timestamp}"
                    )
                )

            if st.button(
                "Save Human Decision",
                type="primary",
                use_container_width=True,
                key=(
                    f"save_decision_"
                    f"{claim_key}"
                ),
            ):

                timestamp = _utc_timestamp()

                st.session_state[
                    "queue_human_decisions"
                ][claim_key] = (
                    decision
                )

                st.session_state[
                    "queue_human_notes"
                ][claim_key] = (
                    investigation_note.strip()
                )

                st.session_state[
                    "queue_decision_timestamps"
                ][claim_key] = (
                    timestamp
                )

                mask = (
                    st.session_state
                    .queue_results[
                        "claim_id"
                    ]
                    .astype(str)
                    == claim_key
                )

                st.session_state.queue_results.loc[
                    mask,
                    "human_decision",
                ] = decision

                st.session_state.queue_results.loc[
                    mask,
                    "investigator_note",
                ] = (
                    investigation_note.strip()
                )

                st.session_state.queue_results.loc[
                    mask,
                    "decision_updated_at",
                ] = timestamp

                st.success(
                    (
                        "Human decision saved "
                        "for this session."
                    )
                )

                st.rerun()

            st.write("")

            info_panel(
                "Human-in-the-Loop",
                (
                    "The investigator decision is stored separately "
                    "from the model score and recommendation. "
                    "Changing the human decision never changes "
                    "the model output."
                ),
                tone="info",
            )


# =============================================================================
# Export
# =============================================================================


def _render_export(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """
    Export the full prioritized worklist including human-review fields.
    """

    st.write("")
    st.write("")

    section_header(
        "Export",
        (
            "Export the prioritized queue with model, "
            "policy and human-review fields for audit "
            "or operational follow-up."
        ),
        eyebrow="AUDIT OUTPUT",
    )

    export = frame.copy()

    export[
        "review_capacity"
    ] = _safe_float(
        metadata.get(
            "capacity"
        )
    )

    export[
        "queue_source"
    ] = str(
        metadata.get(
            "source_name",
            st.session_state.get(
                "queue_source_name",
                "—",
            ),
        )
    )

    export[
        "queue_source_type"
    ] = str(
        metadata.get(
            "source_type",
            "—",
        )
    )

    export[
        "queue_generated_at"
    ] = metadata.get(
        "generated_at",
        "",
    )

    export[
        "queue_model_name"
    ] = metadata.get(
        "model_name",
        "",
    )

    export[
        "queue_model_version"
    ] = metadata.get(
        "model_version",
        "",
    )

    csv = (
        export
        .to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )

    st.download_button(
        "Download Investigation Queue",
        data=csv,
        file_name=(
            "investigation_queue.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        (
            "Export contains model score, rank, percentile, "
            "priority, model recommendation, source context, "
            "human decision, investigator note and audit timestamp."
        )
    )


# =============================================================================
# Main page
# =============================================================================


def render(
    client,
) -> None:
    """
    Render the operational investigation worklist.
    """

    _initialize_queue_state()

    section_header(
        "Investigation Queue",
        (
            "Prioritize model-selected claims, support "
            "investigator triage and keep human decisions "
            "strictly separate from model recommendations."
        ),
    )

    # -------------------------------------------------------------------------
    # Header controls
    # -------------------------------------------------------------------------

    header_left, header_right = (
        st.columns(
            [
                4,
                1,
            ]
        )
    )

    with header_left:

        st.caption(
            (
                "The deployed model ranks the portfolio and "
                "the backend applies the chosen review capacity. "
                "Investigators retain final decision authority."
            )
        )

    with header_right:

        if st.button(
            "Reset Queue",
            use_container_width=True,
            key="reset_investigation_queue",
        ):

            _reset_queue()

            st.rerun()

    st.write("")

    # -------------------------------------------------------------------------
    # Queue generation
    # -------------------------------------------------------------------------

    _render_queue_builder(
        client
    )

    frame = (
        st.session_state.get(
            "queue_results"
        )
    )

    metadata = (
        st.session_state.get(
            "queue_metadata"
        )
        or {}
    )

    source_claims = (
        st.session_state.get(
            "queue_source_claims"
        )
    )

    if (
        frame is None
        or not isinstance(
            frame,
            pd.DataFrame,
        )
        or frame.empty
        or not source_claims
    ):

        st.write("")
        st.write("")

        empty_state(
            "No Investigation Queue",
            (
                "Upload a portfolio or use the demo "
                "portfolio to generate a prioritized "
                "human-review worklist."
            ),
            hint=(
                "Queue selection is produced through "
                "the deployed /top-review endpoint."
            ),
        )

        return

    # -------------------------------------------------------------------------
    # Integrity validation
    # -------------------------------------------------------------------------

    integrity_errors = (
        _validate_stored_queue(
            frame,
            metadata,
        )
    )

    if integrity_errors:

        for message in integrity_errors:
            st.error(message)

        info_panel(
            "Queue Snapshot Invalid",
            (
                "Reset the queue and regenerate it before "
                "continuing with investigation."
            ),
            tone="danger",
        )

        return

    # -------------------------------------------------------------------------
    # Synchronize human-state columns
    # -------------------------------------------------------------------------

    prediction_columns = [
        column
        for column in [
            "claim_id",
            "fraud_risk_score",
            "model_name",
            "model_version",
            "risk_rank",
            "risk_percentile",
            "review_fraction",
            "selected_for_review",
        ]
        if column
        in frame.columns
    ]

    prediction_frame = (
        frame[
            prediction_columns
        ]
        .copy()
    )

    frame = _enrich_queue(
        prediction_frame,
        source_claims,
    )

    st.session_state.queue_results = (
        frame
    )

    # -------------------------------------------------------------------------
    # Queue analytics
    # -------------------------------------------------------------------------

    st.write("")
    st.write("")

    _render_summary(
        frame,
        metadata,
    )

    _render_queue_contract(
        metadata
    )

    _render_priority_distribution(
        frame
    )

    _render_decision_progress(
        frame
    )

    filtered = _render_filters(
        frame
    )

    _render_worklist_table(
        filtered
    )

    _render_claim_detail(
        (
            filtered
            if not filtered.empty
            else frame
        )
    )

    _render_export(
        st.session_state.queue_results,
        metadata,
    )

    st.write("")

    human_review_notice()