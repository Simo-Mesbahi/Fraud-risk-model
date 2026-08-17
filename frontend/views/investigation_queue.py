from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components import (
    empty_state,
    human_review_notice,
    info_panel,
    metric_card,
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
    Convert numeric-like values safely to integer.
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


def _bounded_score(
    value: Any,
) -> float:
    """
    Normalize a model probability to [0, 1].
    """

    return min(
        max(
            _safe_float(
                value
            ),
            0.0,
        ),
        1.0,
    )


def _format_currency(
    value: Any,
) -> str:
    """
    Format a monetary value consistently.
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

    value = str(
        value
    ).strip()

    return (
        value
        if value
        else "—"
    )


def _utc_timestamp() -> str:
    """
    Return an auditable UTC timestamp.
    """

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat(
            timespec="seconds"
        )
    )


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
    Remove target / synthetic generation variables before inference.
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
                key:
                    value

                for (
                    key,
                    value,
                ) in claim.items()

                if key
                not in LEAKAGE_COLUMNS
            }
        )

    return clean_claims


# =============================================================================
# Operational interpretation
# =============================================================================


def _risk_priority(
    score: float,
) -> str:
    """
    Convert model risk into an operational queue priority.

    Priority remains a model-driven recommendation and does not
    represent the investigator's final decision.
    """

    score = (
        _bounded_score(
            score
        )
    )

    if score >= 0.50:

        return (
            "P1 — Immediate"
        )

    if score >= 0.20:

        return (
            "P2 — High"
        )

    if score >= 0.05:

        return (
            "P3 — Standard"
        )

    return (
        "P4 — Monitor"
    )


def _model_recommendation(
    score: float,
) -> str:
    """
    Human-readable recommendation generated from model risk.
    """

    score = (
        _bounded_score(
            score
        )
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
    Return design tone for reusable metric cards.
    """

    if priority == "P1 — Immediate":

        return "danger"

    if priority == "P2 — High":

        return "warning"

    if priority == "P3 — Standard":

        return "info"

    return "success"


# =============================================================================
# Session state
# =============================================================================


def _initialize_queue_state() -> None:
    """
    Initialize all queue-specific state.
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

    for (
        key,
        value,
    ) in defaults.items():

        if key not in (
            st.session_state
        ):

            st.session_state[
                key
            ] = value.copy() if isinstance(
                value,
                dict,
            ) else value


def _reset_queue() -> None:
    """
    Reset queue state and queue-specific widgets.
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

            del st.session_state[
                key
            ]


# =============================================================================
# Portfolio preparation
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

        return pd.DataFrame()

    frame = (
        pd.DataFrame(
            claims
        )
    )

    if (
        "claim_id"
        not in frame.columns
    ):

        raise ValueError(
            (
                "The portfolio must contain "
                "a claim_id field."
            )
        )

    frame[
        "claim_id"
    ] = (
        frame[
            "claim_id"
        ]
        .astype(str)
        .str.strip()
    )

    if (
        frame[
            "claim_id"
        ]
        .eq("")
        .any()
    ):

        raise ValueError(
            (
                "Portfolio contains one or more "
                "empty claim identifiers."
            )
        )

    duplicated = (
        frame[
            "claim_id"
        ]
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
            .astype(str)
            .unique()
            .tolist()
        )

        preview = (
            ", ".join(
                duplicates[:5]
            )
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
# Prediction validation
# =============================================================================


def _validate_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalize /top-review model output.
    """

    required = {
        "claim_id",
        "fraud_risk_score",
    }

    missing = (
        required
        - set(
            predictions.columns
        )
    )

    if missing:

        raise RuntimeError(
            (
                "Queue response is missing required "
                "prediction fields: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )
        )

    frame = (
        predictions.copy()
    )

    frame[
        "claim_id"
    ] = (
        frame[
            "claim_id"
        ]
        .astype(str)
        .str.strip()
    )

    frame[
        "fraud_risk_score"
    ] = (
        pd.to_numeric(
            frame[
                "fraud_risk_score"
            ],
            errors="coerce",
        )
    )

    if (
        frame[
            "fraud_risk_score"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            (
                "Queue response contains "
                "invalid fraud-risk scores."
            )
        )

    frame[
        "fraud_risk_score"
    ] = (
        frame[
            "fraud_risk_score"
        ]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    if (
        "risk_rank"
        not in frame.columns
    ):

        frame = (
            frame
            .sort_values(
                "fraud_risk_score",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        frame[
            "risk_rank"
        ] = (
            np.arange(
                1,
                len(
                    frame
                )
                + 1,
            )
        )

    else:

        frame[
            "risk_rank"
        ] = (
            pd.to_numeric(
                frame[
                    "risk_rank"
                ],
                errors="coerce",
            )
        )

        if (
            frame[
                "risk_rank"
            ]
            .isna()
            .any()
        ):

            raise RuntimeError(
                (
                    "Queue response contains "
                    "invalid risk ranks."
                )
            )

        frame[
            "risk_rank"
        ] = (
            frame[
                "risk_rank"
            ]
            .astype(int)
        )

    return (
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
        )
        .reset_index(
            drop=True
        )
    )


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
    Merge model output with business attributes and human-review state.
    """

    queue = (
        _validate_predictions(
            predictions
        )
    )

    source = (
        _claims_to_frame(
            claims
        )
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
    ]

    available_columns = [
        column

        for column
        in business_columns

        if column
        in source.columns
    ]

    source = (
        source[
            available_columns
        ]
        .copy()
    )

    # Remove business columns already returned directly by API,
    # except claim_id which remains the join key.
    duplicate_business_columns = [
        column

        for column
        in available_columns

        if (
            column != "claim_id"
            and column
            in queue.columns
        )
    ]

    if duplicate_business_columns:

        queue = (
            queue.drop(
                columns=duplicate_business_columns
            )
        )

    queue = (
        queue.merge(
            source,
            on="claim_id",
            how="left",
            validate="one_to_one",
        )
    )

    queue[
        "risk_tier"
    ] = (
        queue[
            "fraud_risk_score"
        ]
        .apply(
            risk_tier
        )
    )

    queue[
        "priority"
    ] = (
        queue[
            "fraud_risk_score"
        ]
        .apply(
            _risk_priority
        )
    )

    queue[
        "model_recommendation"
    ] = (
        queue[
            "fraud_risk_score"
        ]
        .apply(
            _model_recommendation
        )
    )

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

    queue[
        "human_decision"
    ] = (
        queue[
            "claim_id"
        ]
        .map(
            decisions
        )
        .fillna(
            "Pending review"
        )
    )

    queue[
        "investigator_note"
    ] = (
        queue[
            "claim_id"
        ]
        .map(
            notes
        )
        .fillna("")
    )

    queue[
        "decision_updated_at"
    ] = (
        queue[
            "claim_id"
        ]
        .map(
            timestamps
        )
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
) -> None:
    """
    Score portfolio and persist selected investigation worklist.
    """

    if not claims:

        raise ValueError(
            (
                "Portfolio contains "
                "no claims."
            )
        )

    capacity = (
        float(
            capacity
        )
    )

    if not (
        0
        < capacity
        <= 1
    ):

        raise ValueError(
            (
                "Investigation capacity must "
                "be greater than 0 and at most 100%."
            )
        )

    clean_claims = (
        _strip_leakage(
            claims
        )
    )

    # Validate claim IDs before API request.
    _claims_to_frame(
        clean_claims
    )

    with st.spinner(
        (
            "Scoring portfolio, ranking claims "
            "and building the investigation queue..."
        )
    ):

        response = (
            client.top_review(
                clean_claims,
                capacity,
            )
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

    predictions = (
        pd.DataFrame(
            raw_predictions
        )
    )

    if predictions.empty:

        raise RuntimeError(
            (
                "The model returned an empty "
                "investigation queue."
            )
        )

    enriched = (
        _enrich_queue(
            predictions,
            clean_claims,
        )
    )

    total_claims = (
        _safe_int(
            response.get(
                "total_claims"
            ),
            default=len(
                clean_claims
            ),
        )
    )

    selected_claims = (
        _safe_int(
            response.get(
                "selected_claims"
            ),
            default=len(
                enriched
            ),
        )
    )

    st.session_state.queue_results = (
        enriched
    )

    st.session_state.queue_source_claims = (
        clean_claims
    )

    st.session_state.queue_source_name = (
        str(
            source_name
        )
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
            str(
                source_name
            ),
    }


# =============================================================================
# Queue creation controls
# =============================================================================


def _render_queue_builder(
    client,
) -> None:
    """
    Render portfolio source and operational capacity controls.
    """

    section_header(
        "Build Investigation Queue",
        (
            "Score a portfolio and select the "
            "highest-risk claims according to "
            "available investigation capacity."
        ),
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
                value=3,
                step=1,
                key="queue_capacity",
                help=(
                    "Percentage of the portfolio "
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
            "The model ranks the entire portfolio. "
            f"At {capacity:.0%} capacity, only the highest-ranked "
            "claims are placed in the investigation queue. "
            "Selection does not establish fraud."
        ),
        tone="info",
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
                    "Supported formats: "
                    "JSON, CSV and Parquet."
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

                claims = (
                    read_uploaded_file(
                        uploaded
                    )
                )

                preview = (
                    _claims_to_frame(
                        claims
                    )
                )

                selected_estimate = (
                    max(
                        1,
                        int(
                            np.ceil(
                                len(
                                    preview
                                )
                                * capacity
                            )
                        ),
                    )
                )

                p1, p2, p3 = (
                    st.columns(3)
                )

                with p1:

                    metric_card(
                        "Claims Detected",
                        f"{len(preview):,}",
                        "Valid portfolio records",
                    )

                with p2:

                    metric_card(
                        "Estimated Selected",
                        f"{selected_estimate:,}",
                        (
                            f"At {capacity:.0%} "
                            "review capacity"
                        ),
                        tone="info",
                    )

                with p3:

                    metric_card(
                        "Input Columns",
                        f"{len(preview.columns):,}",
                        "Available source fields",
                    )

                with st.expander(
                    "Portfolio preview",
                    expanded=False,
                ):

                    st.dataframe(
                        preview.head(
                            20
                        ),
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
                    )

                    st.success(
                        (
                            "Investigation queue "
                            "created successfully."
                        )
                    )

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
                "Run the complete queue workflow "
                "using the synthetic portfolio bundled "
                "with this project."
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
                ],
                value=500,
                key="queue_demo_size",
            )
        )

        estimated_selection = (
            max(
                1,
                int(
                    np.ceil(
                        int(
                            demo_size
                        )
                        * capacity
                    )
                ),
            )
        )

        st.caption(
            (
                f"Approximately {estimated_selection:,} "
                "claims will be selected at "
                f"{capacity:.0%} capacity."
            )
        )

        if st.button(
            "Build Demo Queue",
            type="primary",
            use_container_width=True,
            key="build_demo_queue",
        ):

            try:

                demo = (
                    load_demo_claims(
                        limit=int(
                            demo_size
                        )
                    )
                )

                claims = (
                    demo.to_dict(
                        orient="records"
                    )
                )

                _generate_queue(
                    client=client,
                    claims=claims,
                    capacity=capacity,
                    source_name=(
                        "Synthetic demo portfolio"
                    ),
                )

                st.success(
                    (
                        "Demo investigation queue "
                        "created successfully."
                    )
                )

            except Exception as exc:

                st.error(
                    str(
                        exc
                    )
                )


# =============================================================================
# Executive summary
# =============================================================================


def _render_summary(
    frame: pd.DataFrame,
    metadata: dict[
        str,
        Any,
    ],
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
    )

    total = (
        _safe_int(
            metadata.get(
                "total_claims"
            )
        )
    )

    selected = (
        _safe_int(
            metadata.get(
                "selected_claims"
            )
        )
    )

    capacity = (
        _safe_float(
            metadata.get(
                "capacity"
            )
        )
    )

    mean_risk = (
        _safe_float(
            frame[
                "fraud_risk_score"
            ]
            .mean()
        )
    )

    max_risk = (
        _safe_float(
            frame[
                "fraud_risk_score"
            ]
            .max()
        )
    )

    immediate_count = (
        int(
            (
                frame[
                    "priority"
                ]
                == "P1 — Immediate"
            )
            .sum()
        )
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Portfolio",
            f"{total:,}",
            "Total claims scored",
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
            "Selected worklist",
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

    source_name = (
        metadata.get(
            "source_name"
        )
        or st.session_state.get(
            "queue_source_name"
        )
        or "—"
    )

    generated_at = (
        metadata.get(
            "generated_at",
            "—",
        )
    )

    st.caption(
        (
            f"Source: {source_name} • "
            f"Queue generated: {generated_at}"
        )
    )


# =============================================================================
# Priority distribution
# =============================================================================


def _render_priority_distribution(
    frame: pd.DataFrame,
) -> None:
    """
    Render operational priority composition.
    """

    st.write("")
    st.write("")

    section_header(
        "Priority Distribution",
        (
            "Model-based operational segmentation "
            "within the selected queue."
        ),
    )

    counts = (
        frame[
            "priority"
        ]
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

    columns = (
        st.columns(
            4
        )
    )

    for (
        column,
        priority,
    ) in zip(
        columns,
        PRIORITY_ORDER,
    ):

        with column:

            metric_card(
                priority,
                f"{int(counts[priority]):,}",
                helpers[
                    priority
                ],
                tone=_priority_tone(
                    priority
                ),
            )


# =============================================================================
# Filters
# =============================================================================


def _render_filters(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply operational worklist filters without mutating source state.
    """

    st.write("")
    st.write("")

    section_header(
        "Investigation Worklist",
        (
            "Search and filter model-selected "
            "claims before investigator review."
        ),
    )

    row1_col1, row1_col2 = (
        st.columns(2)
    )

    with row1_col1:

        search = (
            st.text_input(
                "Search",
                placeholder=(
                    "Claim, customer or provider ID"
                ),
                key="queue_search",
            )
        )

    with row1_col2:

        priority_options = [
            priority

            for priority
            in PRIORITY_ORDER

            if priority
            in frame[
                "priority"
            ]
            .unique()
        ]

        selected_priorities = (
            st.multiselect(
                "Priority",
                options=priority_options,
                default=priority_options,
                key="queue_priority_filter",
            )
        )

    row2_col1, row2_col2, row2_col3 = (
        st.columns(3)
    )

    tiers = [
        value

        for value
        in [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        ]

        if value
        in frame[
            "risk_tier"
        ]
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
                    options=service_options,
                    key="queue_service_filter",
                )
            )

        else:

            selected_services = []

    with row2_col3:

        selected_decisions = (
            st.multiselect(
                "Human decision",
                options=DECISION_OPTIONS,
                key="queue_decision_filter",
            )
        )

    display = (
        frame.copy()
    )

    if selected_priorities:

        display = (
            display.loc[
                display[
                    "priority"
                ]
                .isin(
                    selected_priorities
                )
            ]
        )

    if selected_tiers:

        display = (
            display.loc[
                display[
                    "risk_tier"
                ]
                .isin(
                    selected_tiers
                )
            ]
        )

    if (
        selected_services
        and "service_category"
        in display.columns
    ):

        display = (
            display.loc[
                display[
                    "service_category"
                ]
                .astype(str)
                .isin(
                    selected_services
                )
            ]
        )

    if selected_decisions:

        display = (
            display.loc[
                display[
                    "human_decision"
                ]
                .isin(
                    selected_decisions
                )
            ]
        )

    if search:

        normalized = (
            search
            .strip()
            .lower()
        )

        searchable_columns = [
            column

            for column
            in [
                "claim_id",
                "customer_id",
                "provider_id",
            ]

            if column
            in display.columns
        ]

        mask = (
            pd.Series(
                False,
                index=display.index,
            )
        )

        for column in searchable_columns:

            mask = (
                mask
                |
                display[
                    column
                ]
                .astype(str)
                .str.lower()
                .str.contains(
                    normalized,
                    regex=False,
                    na=False,
                )
            )

        display = (
            display.loc[
                mask
            ]
        )

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
    Render investigation worklist.
    """

    if display.empty:

        empty_state(
            "No matching claims",
            (
                "No queue item matches the "
                "current filters."
            ),
            hint=(
                "Adjust search or filter criteria."
            ),
        )

        return

    preferred_columns = [
        "risk_rank",
        "claim_id",
        "fraud_risk_score",
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

    table = (
        display[
            columns
        ]
        .copy()
    )

    st.dataframe(
        table,
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
                    format="%.2f",
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
) -> dict[
    str,
    Any,
] | None:
    """
    Retrieve complete source payload for one queue claim.
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
# Claim Analysis drill-down
# =============================================================================


def _open_claim_analysis(
    row: pd.Series,
) -> None:
    """
    Transfer the selected queue claim into Claim Analysis.
    """

    claim_id = (
        str(
            row[
                "claim_id"
            ]
        )
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

    score = (
        _bounded_score(
            row[
                "fraud_risk_score"
            ]
        )
    )

    prediction = {
        "claim_id":
            claim_id,

        "fraud_risk_score":
            score,

        "model_name":
            row.get(
                "model_name",
                "XGBoost",
            ),

        "model_version":
            row.get(
                "model_version",
                "1.0.0",
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

    # This matches the radio key used by frontend/app.py.
    st.session_state.main_navigation = (
        "Claim Analysis"
    )


# =============================================================================
# Claim detail / human decision
# =============================================================================


def _render_claim_detail(
    frame: pd.DataFrame,
) -> None:
    """
    Render one queue case and record investigator decision.
    """

    if frame.empty:
        return

    st.write("")
    st.write("")

    section_header(
        "Claim Review",
        (
            "Inspect an individual queue item, "
            "compare model recommendation and "
            "record the investigator decision."
        ),
    )

    claim_options = (
        frame[
            "claim_id"
        ]
        .astype(str)
        .tolist()
    )

    selected_claim_id = (
        st.selectbox(
            "Open claim",
            options=claim_options,
            key="queue_claim_detail_selector",
            format_func=lambda claim_id:
                (
                    f"#{_safe_int(frame.loc[frame['claim_id'].astype(str) == str(claim_id), 'risk_rank'].iloc[0])}"
                    f" • {claim_id}"
                    f" • "
                    f"{_bounded_score(frame.loc[frame['claim_id'].astype(str) == str(claim_id), 'fraud_risk_score'].iloc[0]):.2%}"
                ),
        )
    )

    st.session_state.queue_selected_claim_id = (
        selected_claim_id
    )

    row = (
        frame.loc[
            frame[
                "claim_id"
            ]
            .astype(str)
            == str(
                selected_claim_id
            )
        ]
        .iloc[
            0
        ]
    )

    left, right = (
        st.columns(
            [
                1.25,
                1,
            ],
            gap="large",
        )
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

            priority = (
                str(
                    row.get(
                        "priority",
                        "P4 — Monitor",
                    )
                )
            )

            tone = (
                _priority_tone(
                    priority
                )
            )

            a1, a2, a3 = (
                st.columns(
                    [
                        .8,
                        1,
                        1.2,
                    ]
                )
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
                    (
                        f"{_bounded_score(row.get('fraud_risk_score')):.2%}"
                    ),
                    "Individual score",
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
                    "This recommendation is generated "
                    "from model risk and portfolio ranking."
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
                            f"{row.get('service_category', '—')}"
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

            current_decision = (
                st.session_state
                .queue_human_decisions
                .get(
                    str(
                        selected_claim_id
                    ),
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

            decision = (
                st.selectbox(
                    "Human decision",
                    options=DECISION_OPTIONS,
                    index=current_index,
                    key=(
                        f"decision_"
                        f"{selected_claim_id}"
                    ),
                )
            )

            existing_note = (
                st.session_state
                .queue_human_notes
                .get(
                    str(
                        selected_claim_id
                    ),
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
                        f"{selected_claim_id}"
                    ),
                )
            )

            if st.button(
                "Save Human Decision",
                type="primary",
                use_container_width=True,
                key=(
                    f"save_decision_"
                    f"{selected_claim_id}"
                ),
            ):

                claim_key = (
                    str(
                        selected_claim_id
                    )
                )

                timestamp = (
                    _utc_timestamp()
                )

                st.session_state[
                    "queue_human_decisions"
                ][
                    claim_key
                ] = (
                    decision
                )

                st.session_state[
                    "queue_human_notes"
                ][
                    claim_key
                ] = (
                    investigation_note.strip()
                )

                st.session_state[
                    "queue_decision_timestamps"
                ][
                    claim_key
                ] = (
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

            st.write("")

            info_panel(
                "Human-in-the-loop",
                (
                    "The investigator decision is stored "
                    "separately from the model recommendation. "
                    "Changing the human decision does not alter "
                    "the model score."
                ),
                tone="info",
            )


# =============================================================================
# Export
# =============================================================================


def _render_export(
    frame: pd.DataFrame,
    metadata: dict[
        str,
        Any,
    ],
) -> None:
    """
    Export full review queue including human-review fields.
    """

    st.write("")
    st.write("")

    section_header(
        "Export",
        (
            "Export the prioritized queue with "
            "model and human-review fields for "
            "operational follow-up or audit."
        ),
    )

    export = (
        frame.copy()
    )

    export[
        "review_capacity"
    ] = (
        _safe_float(
            metadata.get(
                "capacity"
            )
        )
    )

    export[
        "queue_source"
    ] = (
        str(
            metadata.get(
                "source_name",
                st.session_state.get(
                    "queue_source_name",
                    "—",
                ),
            )
        )
    )

    export[
        "queue_generated_at"
    ] = (
        metadata.get(
            "generated_at",
            "",
        )
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
        mime=(
            "text/csv"
        ),
        use_container_width=True,
    )

    st.caption(
        (
            "Export contains model score, rank, "
            "priority, model recommendation, human decision, "
            "investigator note and decision timestamp."
        )
    )


# =============================================================================
# Main page
# =============================================================================


def render(
    client,
) -> None:
    """
    Render the operational human-review queue.
    """

    _initialize_queue_state()

    section_header(
        "Investigation Queue",
        (
            "Prioritize model-selected claims, "
            "support investigator triage and keep "
            "human decisions distinct from model recommendations."
        ),
    )

    # -------------------------------------------------------------------------
    # Page controls
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
                "The deployed model ranks the portfolio. "
                "The selected queue reflects investigation "
                "capacity; investigators retain final decision authority."
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
    # Queue builder
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
    )

    source_claims = (
        st.session_state.get(
            "queue_source_claims"
        )
    )

    if (
        frame is None
        or metadata is None
        or source_claims is None
        or frame.empty
    ):

        st.write("")
        st.write("")

        empty_state(
            "No investigation queue",
            (
                "Upload a portfolio or use the demo "
                "portfolio to generate a prioritized "
                "human-review queue."
            ),
            hint=(
                "Model ranking is applied only after "
                "a portfolio has been scored."
            ),
        )

        return

    # -------------------------------------------------------------------------
    # Synchronize persistent human state
    #
    # Re-enrichment is performed from the model-result columns plus original
    # source claims. This avoids stale human-review values while preserving
    # the immutable model score and rank.
    # -------------------------------------------------------------------------

    prediction_columns = [
        column

        for column
        in [
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

    frame = (
        _enrich_queue(
            prediction_frame,
            source_claims,
        )
    )

    st.session_state.queue_results = (
        frame
    )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    st.write("")
    st.write("")

    _render_summary(
        frame,
        metadata,
    )

    _render_priority_distribution(
        frame
    )

    filtered = (
        _render_filters(
            frame
        )
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