from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
from time import perf_counter
from typing import Any

import altair as alt
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


MAX_BATCH_SIZE = 10_000

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


RISK_ORDER = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]


PORTFOLIO_WIDGET_KEYS = {
    "portfolio_upload",
    "portfolio_demo_size",
    "portfolio_top_n",
    "portfolio_search",
    "portfolio_tier_filter",
    "portfolio_service_filter",
    "portfolio_min_risk",
    "portfolio_review_only",
    "portfolio_claim_detail",
}


# =============================================================================
# Generic helpers
# =============================================================================


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a numeric-like value to a finite float.
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
    Convert a numeric-like value safely to integer.
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


def _bounded_score(
    value: Any,
) -> float:
    """
    Normalize a probability-like value to [0, 1].
    """

    return min(
        max(
            _safe_float(value),
            0.0,
        ),
        1.0,
    )


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
    Normalize identifiers for display.
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


def _safe_model_label(
    metadata: dict[str, Any],
) -> str:
    """
    Build a compact model label from portfolio metadata.
    """

    model_name = (
        metadata.get("model_name")
        or "—"
    )

    model_version = (
        metadata.get("model_version")
        or "—"
    )

    return (
        f"{model_name} v{model_version}"
    )


# =============================================================================
# Leakage protection
# =============================================================================


def _strip_leakage(
    claims: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove target and synthetic-generation-only variables before inference.

    These fields may exist in the bundled synthetic dataset for evaluation
    purposes but must never be exposed to the deployed inference model.
    """

    cleaned: list[
        dict[str, Any]
    ] = []

    for claim in claims:

        if not isinstance(
            claim,
            dict,
        ):
            raise TypeError(
                (
                    "Each portfolio item must be "
                    "a claim dictionary."
                )
            )

        cleaned.append(
            {
                key: value
                for key, value
                in claim.items()
                if key not in LEAKAGE_COLUMNS
            }
        )

    return cleaned


# =============================================================================
# Runtime model policy
# =============================================================================


def _runtime_model_info(
    client,
) -> dict[str, Any]:
    """
    Retrieve the deployed model contract.

    Failure is intentionally non-blocking here because the scoring endpoint
    remains the authoritative test of inference availability.
    """

    try:
        payload = (
            client.model_info()
        )

        if isinstance(
            payload,
            dict,
        ):
            return payload

    except Exception:
        pass

    return {}


def _review_fraction(
    model_info: dict[str, Any],
) -> float:
    """
    Resolve the operational investigation fraction from model metadata.
    """

    policy = (
        model_info.get(
            "review_policy",
            {},
        )
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


def _initialize_state() -> None:
    """
    Ensure the portfolio state contract exists.

    This is intentionally idempotent and remains compatible with the
    application's central state initialization.
    """

    defaults = {
        "batch_results": None,
        "batch_input": None,
        "batch_source": None,
        "batch_metadata": None,
        "batch_selected_claim_id": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


def _reset_portfolio() -> None:
    """
    Reset portfolio workflow state and portfolio-specific widgets.
    """

    st.session_state.batch_results = None
    st.session_state.batch_input = None
    st.session_state.batch_source = None
    st.session_state.batch_metadata = None
    st.session_state.batch_selected_claim_id = None

    for key in PORTFOLIO_WIDGET_KEYS:

        if key in st.session_state:
            del st.session_state[key]


# =============================================================================
# Source portfolio validation
# =============================================================================


def _claims_to_frame(
    claims: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert claim dictionaries into a structurally validated DataFrame.
    """

    if not claims:
        raise ValueError(
            "The portfolio contains no claims."
        )

    frame = pd.DataFrame(claims)

    if frame.empty:
        raise ValueError(
            "The portfolio contains no usable rows."
        )

    if "claim_id" not in frame.columns:
        raise ValueError(
            (
                "The portfolio must contain "
                "a 'claim_id' field."
            )
        )

    frame["claim_id"] = (
        frame["claim_id"]
        .astype(str)
        .str.strip()
    )

    invalid_claim_ids = (
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

    if invalid_claim_ids.any():
        raise ValueError(
            (
                "Portfolio contains one or more "
                "missing or empty claim identifiers."
            )
        )

    return frame


def _validate_portfolio(
    claims: list[dict[str, Any]],
) -> tuple[
    bool,
    list[str],
]:
    """
    Validate portfolio structure before API submission.
    """

    errors: list[str] = []

    if not claims:
        return (
            False,
            [
                "No claims detected."
            ],
        )

    if len(claims) > MAX_BATCH_SIZE:
        errors.append(
            (
                f"Portfolio contains {len(claims):,} claims. "
                f"The current batch limit is "
                f"{MAX_BATCH_SIZE:,}."
            )
        )

    invalid_rows = [
        index
        for index, claim
        in enumerate(claims)
        if not isinstance(
            claim,
            dict,
        )
    ]

    if invalid_rows:
        errors.append(
            (
                f"{len(invalid_rows):,} row(s) "
                "are not valid claim objects."
            )
        )

        return (
            False,
            errors,
        )

    try:
        frame = _claims_to_frame(
            claims
        )

        duplicate_mask = (
            frame["claim_id"]
            .duplicated(
                keep=False
            )
        )

        if duplicate_mask.any():

            duplicates = (
                frame.loc[
                    duplicate_mask,
                    "claim_id",
                ]
                .unique()
                .tolist()
            )

            examples = ", ".join(
                duplicates[:5]
            )

            errors.append(
                (
                    f"{len(duplicates):,} duplicated "
                    "claim ID(s) detected. "
                    f"Examples: {examples}"
                )
            )

    except Exception as exc:
        errors.append(
            str(exc)
        )

    return (
        len(errors) == 0,
        errors,
    )


# =============================================================================
# Prediction contract validation
# =============================================================================


def _validate_predictions(
    predictions: pd.DataFrame,
    expected_claim_ids: set[str] | None = None,
) -> pd.DataFrame:
    """
    Validate and normalize the /score-batch API result.
    """

    if predictions.empty:
        raise RuntimeError(
            (
                "The inference API returned "
                "no predictions."
            )
        )

    required = {
        "claim_id",
        "fraud_risk_score",
    }

    missing = (
        required
        - set(predictions.columns)
    )

    if missing:
        raise RuntimeError(
            (
                "Batch response is missing required fields: "
                + ", ".join(
                    sorted(missing)
                )
            )
        )

    frame = predictions.copy()

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
        raise RuntimeError(
            (
                "The inference API returned one or more "
                "invalid claim identifiers."
            )
        )

    frame["fraud_risk_score"] = (
        pd.to_numeric(
            frame["fraud_risk_score"],
            errors="coerce",
        )
    )

    if (
        frame["fraud_risk_score"]
        .isna()
        .any()
    ):
        raise RuntimeError(
            (
                "The model returned one or more "
                "non-numeric fraud-risk scores."
            )
        )

    non_finite = ~np.isfinite(
        frame["fraud_risk_score"]
        .to_numpy(
            dtype=float
        )
    )

    if non_finite.any():
        raise RuntimeError(
            (
                "The model returned one or more "
                "non-finite fraud-risk scores."
            )
        )

    outside_probability_range = (
        (
            frame["fraud_risk_score"]
            < 0.0
        )
        |
        (
            frame["fraud_risk_score"]
            > 1.0
        )
    )

    if outside_probability_range.any():
        raise RuntimeError(
            (
                "The inference API returned fraud-risk "
                "scores outside the expected [0, 1] range."
            )
        )

    duplicated = (
        frame["claim_id"]
        .duplicated(
            keep=False
        )
    )

    if duplicated.any():
        raise RuntimeError(
            (
                "The inference API returned duplicate "
                "claim IDs."
            )
        )

    if expected_claim_ids is not None:

        returned_ids = set(
            frame["claim_id"]
            .tolist()
        )

        missing_predictions = (
            expected_claim_ids
            - returned_ids
        )

        unexpected_predictions = (
            returned_ids
            - expected_claim_ids
        )

        if missing_predictions:
            examples = ", ".join(
                sorted(
                    missing_predictions
                )[:5]
            )

            raise RuntimeError(
                (
                    f"The inference API omitted "
                    f"{len(missing_predictions):,} submitted "
                    f"claim(s). Examples: {examples}"
                )
            )

        if unexpected_predictions:
            examples = ", ".join(
                sorted(
                    unexpected_predictions
                )[:5]
            )

            raise RuntimeError(
                (
                    f"The inference API returned "
                    f"{len(unexpected_predictions):,} unexpected "
                    f"claim(s). Examples: {examples}"
                )
            )

    return frame


# =============================================================================
# Result enrichment
# =============================================================================


def _enrich_predictions(
    predictions: pd.DataFrame,
    claims: list[dict[str, Any]],
    review_fraction: float,
) -> pd.DataFrame:
    """
    Join predictions with business context and derive portfolio ranking.

    The model score remains untouched. Portfolio rank, percentile and review
    selection are operational attributes derived from the scored population.
    """

    source = _claims_to_frame(
        claims
    )

    expected_claim_ids = set(
        source["claim_id"]
        .astype(str)
        .tolist()
    )

    predictions = _validate_predictions(
        predictions,
        expected_claim_ids=expected_claim_ids,
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
        "service_date",
        "claim_submission_timestamp",
        "customer_age",
        "provider_type",
        "provider_region",
    ]

    available = [
        column
        for column in business_columns
        if column in source.columns
    ]

    source_context = (
        source[available]
        .drop_duplicates(
            subset=["claim_id"],
            keep="first",
        )
    )

    duplicate_business_columns = [
        column
        for column in available
        if (
            column != "claim_id"
            and column in predictions.columns
        )
    ]

    if duplicate_business_columns:
        predictions = predictions.drop(
            columns=duplicate_business_columns
        )

    frame = predictions.merge(
        source_context,
        on="claim_id",
        how="left",
        validate="one_to_one",
    )

    frame = (
        frame
        .sort_values(
            [
                "fraud_risk_score",
                "claim_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    frame["risk_tier"] = (
        frame["fraud_risk_score"]
        .apply(risk_tier)
    )

    frame["portfolio_rank"] = np.arange(
        1,
        len(frame) + 1,
        dtype=int,
    )

    count = len(frame)

    if count == 1:
        frame["risk_percentile"] = 1.0

    else:
        frame["risk_percentile"] = (
            1.0
            - (
                (
                    frame["portfolio_rank"]
                    - 1
                )
                / (
                    count
                    - 1
                )
            )
        )

    review_count = min(
        count,
        max(
            1,
            int(
                np.ceil(
                    count
                    * review_fraction
                )
            ),
        ),
    )

    frame["selected_for_review"] = (
        frame["portfolio_rank"]
        <= review_count
    )

    return frame


# =============================================================================
# Portfolio scoring
# =============================================================================


def _score_portfolio(
    client,
    claims: list[dict[str, Any]],
    source_name: str,
    source_type: str,
) -> None:
    """
    Score a complete portfolio through the deployed batch endpoint.

    The function validates:
    - source structure,
    - anti-leakage policy,
    - API response contract,
    - response count,
    - claim identity preservation,
    - score validity.

    It then stores a normalized portfolio snapshot in session state.
    """

    valid, errors = _validate_portfolio(
        claims
    )

    if not valid:
        raise ValueError(
            " | ".join(errors)
        )

    clean_claims = _strip_leakage(
        claims
    )

    model_info = _runtime_model_info(
        client
    )

    review_fraction = _review_fraction(
        model_info
    )

    started = perf_counter()

    with st.spinner(
        (
            "Validating portfolio, building features "
            "and scoring claims with the deployed model..."
        )
    ):
        response = client.score_batch(
            clean_claims
        )

    elapsed_seconds = max(
        perf_counter() - started,
        0.0,
    )

    if not isinstance(
        response,
        dict,
    ):
        raise RuntimeError(
            (
                "The inference API returned "
                "an invalid batch response."
            )
        )

    raw_predictions = response.get(
        "predictions"
    )

    if not isinstance(
        raw_predictions,
        list,
    ):
        raise RuntimeError(
            (
                "Batch response does not contain "
                "a valid predictions list."
            )
        )

    declared_count = response.get(
        "count"
    )

    if declared_count is not None:

        declared_count = _safe_int(
            declared_count,
            default=-1,
        )

        if declared_count != len(
            raw_predictions
        ):
            raise RuntimeError(
                (
                    "Batch response count is inconsistent "
                    "with the predictions payload."
                )
            )

    if len(raw_predictions) != len(
        clean_claims
    ):
        raise RuntimeError(
            (
                "The number of model predictions does not "
                "match the number of submitted claims."
            )
        )

    predictions = pd.DataFrame(
        raw_predictions
    )

    frame = _enrich_predictions(
        predictions=predictions,
        claims=clean_claims,
        review_fraction=review_fraction,
    )

    if len(frame) != len(
        clean_claims
    ):
        raise RuntimeError(
            (
                "Portfolio enrichment changed the "
                "number of scored claims."
            )
        )

    review_count = int(
        frame["selected_for_review"]
        .sum()
    )

    selected_scores = (
        frame.loc[
            frame["selected_for_review"],
            "fraud_risk_score",
        ]
    )

    throughput = (
        len(frame)
        / elapsed_seconds
        if elapsed_seconds > 0
        else None
    )

    st.session_state.batch_results = (
        frame
    )

    st.session_state.batch_input = (
        clean_claims
    )

    st.session_state.batch_source = (
        str(source_name)
    )

    st.session_state.batch_metadata = {
        "claim_count":
            len(frame),

        "source":
            str(source_name),

        "source_type":
            str(source_type),

        "generated_at":
            _utc_timestamp(),

        "review_fraction":
            review_fraction,

        "review_count":
            review_count,

        "mean_risk":
            float(
                frame["fraud_risk_score"]
                .mean()
            ),

        "median_risk":
            float(
                frame["fraud_risk_score"]
                .median()
            ),

        "max_risk":
            float(
                frame["fraud_risk_score"]
                .max()
            ),

        "selected_mean_risk":
            (
                float(
                    selected_scores.mean()
                )
                if not selected_scores.empty
                else None
            ),

        "scoring_seconds":
            float(elapsed_seconds),

        "throughput_claims_per_second":
            (
                float(throughput)
                if throughput is not None
                else None
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

    st.session_state.batch_selected_claim_id = (
        None
    )


# =============================================================================
# Portfolio input
# =============================================================================


def _render_portfolio_input(
    client,
) -> None:
    """
    Render uploaded and bundled-demo portfolio entry points.
    """

    section_header(
        "Portfolio Input",
        (
            "Upload a claims portfolio or use the bundled "
            "synthetic dataset to run true batch inference."
        ),
    )

    upload_tab, demo_tab = st.tabs(
        [
            "Upload Portfolio",
            "Demo Portfolio",
        ]
    )

    # -------------------------------------------------------------------------
    # Uploaded portfolio
    # -------------------------------------------------------------------------

    with upload_tab:

        uploaded = st.file_uploader(
            "Portfolio file",
            type=[
                "json",
                "csv",
                "parquet",
            ],
            key="portfolio_upload",
            help=(
                "Supported formats: JSON, CSV and Parquet. "
                f"Maximum {MAX_BATCH_SIZE:,} claims."
            ),
        )

        if uploaded is None:

            st.caption(
                (
                    "Upload a portfolio to validate "
                    "and score it."
                )
            )

        else:

            try:
                claims = read_uploaded_file(
                    uploaded
                )

                raw_frame = _claims_to_frame(
                    claims
                )

                valid, validation_errors = (
                    _validate_portfolio(
                        claims
                    )
                )

                missing_values = int(
                    raw_frame
                    .isna()
                    .sum()
                    .sum()
                )

                duplicate_ids = int(
                    raw_frame["claim_id"]
                    .duplicated()
                    .sum()
                )

                leakage_present = [
                    column
                    for column in LEAKAGE_COLUMNS
                    if column in raw_frame.columns
                ]

                c1, c2, c3, c4 = st.columns(
                    4
                )

                with c1:
                    metric_card(
                        "Claims",
                        f"{len(raw_frame):,}",
                        "Portfolio records",
                    )

                with c2:
                    metric_card(
                        "Columns",
                        f"{len(raw_frame.columns):,}",
                        "Input fields",
                    )

                with c3:
                    metric_card(
                        "Missing Values",
                        f"{missing_values:,}",
                        "Across source dataset",
                        tone=(
                            "warning"
                            if missing_values
                            else "success"
                        ),
                    )

                with c4:
                    metric_card(
                        "Duplicate IDs",
                        f"{duplicate_ids:,}",
                        "claim_id uniqueness",
                        tone=(
                            "danger"
                            if duplicate_ids
                            else "success"
                        ),
                    )

                st.write("")

                if valid:
                    info_panel(
                        "Portfolio Validation Passed",
                        (
                            "The portfolio passed frontend "
                            "structural validation and is ready "
                            "for model scoring."
                        ),
                        tone="success",
                    )

                else:
                    for message in validation_errors:
                        st.error(message)

                if leakage_present:
                    info_panel(
                        "Protected Evaluation Fields Detected",
                        (
                            f"{len(leakage_present)} evaluation or "
                            "synthetic-generation field(s) were found. "
                            "They will be removed automatically before "
                            "the portfolio is sent to the model."
                        ),
                        tone="info",
                    )

                with st.expander(
                    "Preview portfolio",
                    expanded=False,
                ):
                    st.dataframe(
                        raw_frame.head(25),
                        use_container_width=True,
                        hide_index=True,
                    )

                if st.button(
                    "Score Portfolio",
                    type="primary",
                    use_container_width=True,
                    disabled=not valid,
                    key="score_uploaded_portfolio",
                ):
                    _score_portfolio(
                        client=client,
                        claims=claims,
                        source_name=uploaded.name,
                        source_type="uploaded",
                    )

                    st.success(
                        (
                            f"{len(claims):,} claims "
                            "scored successfully."
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
                "Run the complete batch-scoring workflow "
                "with synthetic claims bundled with the project."
            )
        )

        demo_size = st.select_slider(
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
            key="portfolio_demo_size",
        )

        if st.button(
            "Score Demo Portfolio",
            type="primary",
            use_container_width=True,
            key="score_demo_portfolio",
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

                _score_portfolio(
                    client=client,
                    claims=claims,
                    source_name=(
                        "Synthetic demo portfolio"
                    ),
                    source_type="demo",
                )

                st.success(
                    (
                        f"{len(claims):,} demo claims "
                        "scored successfully."
                    )
                )

                st.rerun()

            except Exception as exc:
                st.error(
                    str(exc)
                )


# =============================================================================
# Executive KPIs
# =============================================================================


def _render_kpis(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """
    Render portfolio-level operational risk summary.
    """

    section_header(
        "Portfolio Risk Overview",
        (
            "Executive summary of model risk "
            "across the scored population."
        ),
        eyebrow="BATCH INFERENCE",
    )

    count = len(frame)

    mean_risk = float(
        frame["fraud_risk_score"]
        .mean()
    )

    median_risk = float(
        frame["fraud_risk_score"]
        .median()
    )

    max_risk = float(
        frame["fraud_risk_score"]
        .max()
    )

    high_critical = int(
        (
            frame["fraud_risk_score"]
            >= 0.20
        )
        .sum()
    )

    critical = int(
        (
            frame["risk_tier"]
            == "CRITICAL"
        )
        .sum()
    )

    review_count = int(
        frame["selected_for_review"]
        .sum()
    )

    review_fraction = _safe_float(
        metadata.get(
            "review_fraction"
        ),
        DEFAULT_REVIEW_FRACTION,
    )

    selected_mean = (
        frame.loc[
            frame["selected_for_review"],
            "fraud_risk_score",
        ]
        .mean()
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:
        metric_card(
            "Claims Scored",
            f"{count:,}",
            str(
                metadata.get(
                    "source",
                    "Portfolio",
                )
            ),
        )

    with c2:
        metric_card(
            "Mean Risk",
            f"{mean_risk:.2%}",
            f"Median {median_risk:.2%}",
            tone="info",
        )

    with c3:
        metric_card(
            "Maximum Risk",
            f"{max_risk:.2%}",
            "Highest individual score",
            tone=(
                "danger"
                if max_risk >= 0.50
                else "warning"
                if max_risk >= 0.20
                else "info"
            ),
        )

    with c4:
        metric_card(
            "High / Critical",
            f"{high_critical:,}",
            (
                f"{high_critical / count:.1%} "
                "of portfolio"
                if count
                else "—"
            ),
            tone=(
                "warning"
                if high_critical
                else "success"
            ),
        )

    st.write("")

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:
        metric_card(
            "Critical Claims",
            f"{critical:,}",
            "≥ 50% individual model risk",
            tone=(
                "danger"
                if critical
                else "success"
            ),
        )

    with c2:
        metric_card(
            (
                f"Review Population "
                f"@ {review_fraction:.0%}"
            ),
            f"{review_count:,}",
            "Operational capacity policy",
            tone="info",
        )

    with c3:
        metric_card(
            "Selected Mean Risk",
            (
                f"{selected_mean:.2%}"
                if pd.notna(selected_mean)
                else "—"
            ),
            "Highest-ranked review population",
            tone="warning",
        )

    scoring_seconds = _safe_float(
        metadata.get(
            "scoring_seconds"
        ),
        default=-1.0,
    )

    throughput = _safe_float(
        metadata.get(
            "throughput_claims_per_second"
        ),
        default=-1.0,
    )

    with c4:
        metric_card(
            "Scoring Runtime",
            (
                f"{scoring_seconds:.2f}s"
                if scoring_seconds >= 0
                else "—"
            ),
            (
                f"{throughput:,.0f} claims/sec"
                if throughput >= 0
                else "Runtime telemetry"
            ),
            tone="info",
        )

    st.caption(
        (
            f"Scored: {metadata.get('generated_at', '—')} • "
            f"Model: {_safe_model_label(metadata)} • "
            f"Source type: "
            f"{metadata.get('source_type', '—')}"
        )
    )


# =============================================================================
# Distribution analytics
# =============================================================================


def _render_distribution(
    frame: pd.DataFrame,
) -> None:
    """
    Render continuous and categorical portfolio-risk distributions.
    """

    st.write("")
    st.write("")

    section_header(
        "Risk Distribution",
        (
            "Distribution and concentration of model "
            "risk across the scored portfolio."
        ),
        eyebrow="PORTFOLIO ANALYTICS",
    )

    left, right = st.columns(
        [
            1.55,
            1,
        ],
        gap="large",
    )

    with left:

        histogram = (
            alt.Chart(frame)
            .mark_bar()
            .encode(
                x=alt.X(
                    "fraud_risk_score:Q",
                    bin=alt.Bin(
                        maxbins=40
                    ),
                    title="Fraud Risk Score",
                    axis=alt.Axis(
                        format=".0%",
                    ),
                ),
                y=alt.Y(
                    "count():Q",
                    title="Claims",
                ),
                tooltip=[
                    alt.Tooltip(
                        "count():Q",
                        title="Claims",
                    ),
                ],
            )
            .properties(
                height=340
            )
        )

        st.altair_chart(
            histogram,
            use_container_width=True,
        )

    with right:

        distribution = (
            frame["risk_tier"]
            .value_counts()
            .reindex(
                RISK_ORDER,
                fill_value=0,
            )
            .rename_axis(
                "Risk Tier"
            )
            .reset_index(
                name="Claims"
            )
        )

        distribution["Share"] = (
            distribution["Claims"]
            / max(
                len(frame),
                1,
            )
        )

        tier_chart = (
            alt.Chart(distribution)
            .mark_bar(
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6,
            )
            .encode(
                x=alt.X(
                    "Risk Tier:N",
                    sort=RISK_ORDER,
                    title=None,
                ),
                y=alt.Y(
                    "Claims:Q",
                    title="Claims",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Risk Tier:N",
                        title="Tier",
                    ),
                    alt.Tooltip(
                        "Claims:Q",
                        title="Claims",
                    ),
                    alt.Tooltip(
                        "Share:Q",
                        title="Share",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=340
            )
        )

        st.altair_chart(
            tier_chart,
            use_container_width=True,
        )


# =============================================================================
# Review-set concentration
# =============================================================================


def _render_review_concentration(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """
    Compare the operational review population with the remaining portfolio.
    """

    st.write("")
    st.write("")

    section_header(
        "Review-Set Concentration",
        (
            "Measure how strongly model risk is concentrated "
            "inside the current investigation capacity."
        ),
        eyebrow="OPERATIONAL PRIORITIZATION",
    )

    selected = frame.loc[
        frame["selected_for_review"]
    ]

    remaining = frame.loc[
        ~frame["selected_for_review"]
    ]

    review_fraction = _safe_float(
        metadata.get(
            "review_fraction"
        ),
        DEFAULT_REVIEW_FRACTION,
    )

    selected_mean = (
        float(
            selected["fraud_risk_score"]
            .mean()
        )
        if not selected.empty
        else 0.0
    )

    remaining_mean = (
        float(
            remaining["fraud_risk_score"]
            .mean()
        )
        if not remaining.empty
        else 0.0
    )

    concentration_ratio = (
        selected_mean
        / remaining_mean
        if remaining_mean > 0
        else None
    )

    cutoff = (
        float(
            selected["fraud_risk_score"]
            .min()
        )
        if not selected.empty
        else None
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:
        metric_card(
            "Review Capacity",
            f"{review_fraction:.1%}",
            f"{len(selected):,} claims selected",
            tone="info",
        )

    with c2:
        metric_card(
            "Review Mean Risk",
            f"{selected_mean:.2%}",
            "Selected population",
            tone="warning",
        )

    with c3:
        metric_card(
            "Review Cutoff",
            (
                f"{cutoff:.2%}"
                if cutoff is not None
                else "—"
            ),
            "Lowest selected model score",
            tone="info",
        )

    with c4:
        metric_card(
            "Risk Concentration",
            (
                f"{concentration_ratio:.1f}×"
                if concentration_ratio is not None
                else "—"
            ),
            "Selected vs remaining mean",
            tone="warning",
        )


# =============================================================================
# Quantiles
# =============================================================================


def _render_quantiles(
    frame: pd.DataFrame,
) -> None:
    """
    Display score concentration thresholds.
    """

    st.write("")
    st.write("")

    section_header(
        "Risk Concentration",
        (
            "Model-score thresholds across the "
            "portfolio distribution."
        ),
        eyebrow="QUANTILES",
    )

    quantiles = {
        "Median": 0.50,
        "75th Percentile": 0.75,
        "90th Percentile": 0.90,
        "95th Percentile": 0.95,
        "99th Percentile": 0.99,
    }

    columns = st.columns(
        len(quantiles)
    )

    for column, item in zip(
        columns,
        quantiles.items(),
    ):
        label, quantile = item

        value = float(
            frame["fraud_risk_score"]
            .quantile(
                quantile
            )
        )

        with column:
            metric_card(
                label,
                f"{value:.2%}",
                "Portfolio risk threshold",
            )


# =============================================================================
# Highest-risk claims
# =============================================================================


def _render_top_risk(
    frame: pd.DataFrame,
) -> None:
    """
    Render the highest-scoring portfolio claims.
    """

    st.write("")
    st.write("")

    section_header(
        "Highest-Risk Claims",
        (
            "Claims with the highest individual "
            "model scores in the portfolio."
        ),
        eyebrow="PRIORITIZATION",
    )

    maximum = min(
        100,
        len(frame),
    )

    if maximum <= 5:

        top_n = maximum

        st.caption(
            (
                f"Showing all {maximum:,} "
                "portfolio claims."
            )
        )

    else:

        top_n = st.slider(
            "Number of top-risk claims",
            min_value=5,
            max_value=maximum,
            value=min(
                20,
                maximum,
            ),
            step=5,
            key="portfolio_top_n",
        )

    top = (
        frame
        .head(
            int(top_n)
        )
        .copy()
    )

    preferred = [
        "portfolio_rank",
        "claim_id",
        "fraud_risk_score",
        "risk_percentile",
        "risk_tier",
        "selected_for_review",
        "claim_amount",
        "requested_reimbursement",
        "service_category",
        "provider_id",
        "customer_id",
    ]

    columns = [
        column
        for column in preferred
        if column in top.columns
    ]

    st.dataframe(
        top[columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "portfolio_rank":
                st.column_config.NumberColumn(
                    "Rank",
                    format="%d",
                    width="small",
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

            "selected_for_review":
                st.column_config.CheckboxColumn(
                    "Review",
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
# Filtering
# =============================================================================


def _render_filters(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Search and filter scored claims without modifying model predictions.
    """

    st.write("")
    st.write("")

    section_header(
        "Portfolio Explorer",
        (
            "Search and filter scored claims without "
            "modifying model predictions."
        ),
        eyebrow="EXPLORATION",
    )

    c1, c2 = st.columns(2)

    with c1:
        search = st.text_input(
            "Search",
            placeholder=(
                "Claim, customer or provider ID"
            ),
            key="portfolio_search",
        )

    with c2:

        available_tiers = [
            tier
            for tier in RISK_ORDER
            if tier
            in frame["risk_tier"]
            .unique()
        ]

        selected_tiers = st.multiselect(
            "Risk tier",
            options=available_tiers,
            default=available_tiers,
            key="portfolio_tier_filter",
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        if "service_category" in frame.columns:

            services = (
                frame["service_category"]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )

            selected_services = st.multiselect(
                "Service category",
                options=services,
                key="portfolio_service_filter",
            )

        else:
            selected_services = []

    with c2:

        minimum_risk = st.slider(
            "Minimum risk",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            format="%.0f%%",
            key="portfolio_min_risk",
        )

    with c3:

        review_only = st.toggle(
            "Selected for review only",
            value=False,
            key="portfolio_review_only",
        )

    display = frame.copy()

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

    display = display.loc[
        display["fraud_risk_score"]
        >= minimum_risk
    ]

    if review_only:
        display = display.loc[
            display["selected_for_review"]
        ]

    if search:

        term = (
            search
            .strip()
            .lower()
        )

        searchable = [
            column
            for column in [
                "claim_id",
                "customer_id",
                "provider_id",
            ]
            if column in display.columns
        ]

        mask = pd.Series(
            False,
            index=display.index,
        )

        for column in searchable:

            mask = (
                mask
                |
                display[column]
                .astype(str)
                .str.lower()
                .str.contains(
                    term,
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
            f"{len(frame):,} claims displayed"
        )
    )

    return (
        display
        .sort_values(
            "portfolio_rank",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# Portfolio table
# =============================================================================


def _render_table(
    display: pd.DataFrame,
) -> None:
    """
    Render the filtered scored portfolio.
    """

    if display.empty:

        empty_state(
            "No Matching Claims",
            (
                "No scored claim matches the "
                "current portfolio filters."
            ),
            hint=(
                "Adjust the search term or risk filters."
            ),
        )

        return

    preferred_columns = [
        "portfolio_rank",
        "claim_id",
        "fraud_risk_score",
        "risk_percentile",
        "risk_tier",
        "selected_for_review",
        "claim_amount",
        "requested_reimbursement",
        "service_category",
        "provider_id",
        "customer_id",
        "model_name",
        "model_version",
    ]

    columns = [
        column
        for column in preferred_columns
        if column in display.columns
    ]

    st.dataframe(
        display[columns],
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={
            "portfolio_rank":
                st.column_config.NumberColumn(
                    "Rank",
                    format="%d",
                    width="small",
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

            "risk_tier":
                st.column_config.TextColumn(
                    "Risk Tier",
                ),

            "selected_for_review":
                st.column_config.CheckboxColumn(
                    "Review",
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
# Source claim lookup
# =============================================================================


def _source_claim(
    claim_id: str,
) -> dict[str, Any] | None:
    """
    Retrieve the complete inference payload for one scored claim.
    """

    claims = (
        st.session_state.get(
            "batch_input"
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
            == str(claim_id)
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
    Transfer one scored portfolio claim into Claim Analysis.

    Claim Analysis can subsequently request its full TreeSHAP explanation
    from the deployed API using the preserved source claim payload.
    """

    claim_id = str(
        row["claim_id"]
    )

    claim = _source_claim(
        claim_id
    )

    if claim is None:
        raise ValueError(
            (
                "The source claim payload "
                "could not be recovered."
            )
        )

    score = _bounded_score(
        row["fraud_risk_score"]
    )

    metadata = (
        st.session_state.get(
            "batch_metadata"
        )
        or {}
    )

    model_name = row.get(
        "model_name"
    )

    if pd.isna(model_name):
        model_name = None

    model_version = row.get(
        "model_version"
    )

    if pd.isna(model_version):
        model_version = None

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
        claim
    )

    st.session_state.single_source = (
        "Portfolio Scoring"
    )

    st.session_state.main_navigation = (
        "Claim Analysis"
    )


# =============================================================================
# Claim drill-down
# =============================================================================


def _render_claim_detail(
    frame: pd.DataFrame,
) -> None:
    """
    Inspect one scored claim in portfolio context.
    """

    if frame.empty:
        return

    st.write("")
    st.write("")

    section_header(
        "Claim Drill-Down",
        (
            "Inspect one scored claim in its portfolio "
            "context and continue to full individual analysis."
        ),
        eyebrow="CASE ANALYSIS",
    )

    options = (
        frame["claim_id"]
        .astype(str)
        .tolist()
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

    def _claim_label(
        claim_id: str,
    ) -> str:

        row = lookup.loc[
            str(claim_id)
        ]

        if isinstance(
            row,
            pd.DataFrame,
        ):
            row = row.iloc[0]

        rank = _safe_int(
            row.get(
                "portfolio_rank"
            )
        )

        score = _bounded_score(
            row.get(
                "fraud_risk_score"
            )
        )

        return (
            f"#{rank} • "
            f"{claim_id} • "
            f"{score:.2%}"
        )

    selected = st.selectbox(
        "Claim",
        options=options,
        key="portfolio_claim_detail",
        format_func=_claim_label,
    )

    st.session_state.batch_selected_claim_id = (
        selected
    )

    row = lookup.loc[
        str(selected)
    ]

    if isinstance(
        row,
        pd.DataFrame,
    ):
        row = row.iloc[0]

    score = _bounded_score(
        row.get(
            "fraud_risk_score"
        )
    )

    left, right = st.columns(
        [
            1.05,
            1.25,
        ],
        gap="large",
    )

    # -------------------------------------------------------------------------
    # Risk position
    # -------------------------------------------------------------------------

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Risk Position"
            )

            c1, c2 = st.columns(2)

            with c1:
                metric_card(
                    "Fraud Risk",
                    f"{score:.2%}",
                    "Individual model score",
                    tone=(
                        "danger"
                        if score >= 0.50
                        else "warning"
                        if score >= 0.20
                        else "info"
                    ),
                )

            with c2:
                metric_card(
                    "Portfolio Rank",
                    (
                        f"#{_safe_int(row.get('portfolio_rank'))}"
                    ),
                    "Relative model ordering",
                    tone="info",
                )

            st.write("")

            c1, c2 = st.columns(2)

            with c1:
                metric_card(
                    "Risk Tier",
                    str(
                        row.get(
                            "risk_tier",
                            "—",
                        )
                    ),
                    "Individual category",
                )

            with c2:
                metric_card(
                    "Risk Percentile",
                    (
                        f"{_bounded_score(row.get('risk_percentile')):.1%}"
                    ),
                    "Relative portfolio position",
                )

            st.write("")

            selected_for_review = bool(
                row.get(
                    "selected_for_review",
                    False,
                )
            )

            info_panel(
                (
                    "Selected for Review"
                    if selected_for_review
                    else "Outside Current Review Set"
                ),
                (
                    "This claim is currently inside the "
                    "highest-ranked operational review population."
                    if selected_for_review
                    else (
                        "This claim is scored but falls outside "
                        "the current operational review capacity."
                    )
                ),
                tone=(
                    "warning"
                    if selected_for_review
                    else "info"
                ),
            )

    # -------------------------------------------------------------------------
    # Business context
    # -------------------------------------------------------------------------

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Claim Context"
            )

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

            c1, c2 = st.columns(2)

            with c1:

                if "claim_amount" in row.index:
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
                            "**Requested reimbursement:** "
                            f"{_format_currency(row.get('requested_reimbursement'))}"
                        )
                    )

                if "service_category" in row.index:
                    st.write(
                        (
                            "**Service:** "
                            f"{_format_identifier(row.get('service_category'))}"
                        )
                    )

            with c2:

                if "provider_id" in row.index:
                    st.write(
                        (
                            "**Provider:** "
                            f"`{_format_identifier(row.get('provider_id'))}`"
                        )
                    )

                if "customer_id" in row.index:
                    st.write(
                        (
                            "**Customer:** "
                            f"`{_format_identifier(row.get('customer_id'))}`"
                        )
                    )

                if "service_code" in row.index:
                    st.write(
                        (
                            "**Service code:** "
                            f"`{_format_identifier(row.get('service_code'))}`"
                        )
                    )

            st.write("")

            if st.button(
                "Open Full Claim Analysis",
                type="primary",
                use_container_width=True,
                key=(
                    "portfolio_open_claim_analysis_"
                    + str(selected)
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


# =============================================================================
# Export
# =============================================================================


def _render_exports(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """
    Export the complete portfolio and operational review population.
    """

    st.write("")
    st.write("")

    section_header(
        "Export Results",
        (
            "Download complete scoring results or "
            "the current operational review population."
        ),
        eyebrow="OUTPUT",
    )

    export = frame.copy()

    export["portfolio_source"] = (
        metadata.get(
            "source",
            "",
        )
    )

    export["portfolio_source_type"] = (
        metadata.get(
            "source_type",
            "",
        )
    )

    export["scored_at"] = (
        metadata.get(
            "generated_at",
            "",
        )
    )

    export["review_fraction"] = _safe_float(
        metadata.get(
            "review_fraction"
        ),
        DEFAULT_REVIEW_FRACTION,
    )

    review = (
        export.loc[
            export["selected_for_review"]
        ]
        .copy()
    )

    fraction = _safe_float(
        metadata.get(
            "review_fraction"
        ),
        DEFAULT_REVIEW_FRACTION,
    )

    percentage_label = (
        f"{fraction * 100:g}"
        .replace(
            ".",
            "_",
        )
    )

    left, right = st.columns(2)

    with left:
        st.download_button(
            "Download All Scores",
            data=(
                export
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8-sig"
                )
            ),
            file_name=(
                "fraud_portfolio_scores.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with right:
        st.download_button(
            (
                f"Download Top "
                f"{fraction:.0%} Review Set"
            ),
            data=(
                review
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8-sig"
                )
            ),
            file_name=(
                f"fraud_top_{percentage_label}"
                "_review.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        (
            "Exports include model score, portfolio rank, "
            "risk percentile, operational review selection, "
            "model identity and available source business attributes."
        )
    )


# =============================================================================
# Portfolio integrity
# =============================================================================


def _validate_stored_results(
    frame: pd.DataFrame,
) -> list[str]:
    """
    Validate the portfolio snapshot stored in Streamlit session state.
    """

    required_result_columns = {
        "claim_id",
        "fraud_risk_score",
        "portfolio_rank",
        "risk_percentile",
        "risk_tier",
        "selected_for_review",
    }

    missing = (
        required_result_columns
        - set(frame.columns)
    )

    errors: list[str] = []

    if missing:
        errors.append(
            (
                "Stored portfolio results are incomplete: "
                + ", ".join(
                    sorted(missing)
                )
            )
        )

        return errors

    if frame.empty:
        errors.append(
            "Stored portfolio results are empty."
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
                "Stored portfolio results contain "
                "duplicate claim identifiers."
            )
        )

    scores = pd.to_numeric(
        frame["fraud_risk_score"],
        errors="coerce",
    )

    if scores.isna().any():
        errors.append(
            (
                "Stored portfolio results contain "
                "invalid fraud-risk scores."
            )
        )

    elif (
        (scores < 0)
        | (scores > 1)
    ).any():
        errors.append(
            (
                "Stored portfolio results contain "
                "scores outside [0, 1]."
            )
        )

    return errors


# =============================================================================
# Main page
# =============================================================================


def render(
    client,
) -> None:
    """
    Render the complete portfolio-scoring workspace.
    """

    _initialize_state()

    section_header(
        "Portfolio Scoring",
        (
            "Score an entire claims portfolio, analyze "
            "risk concentration and identify the highest-risk "
            "cases using the deployed fraud-risk model."
        ),
    )

    # -------------------------------------------------------------------------
    # Header controls
    # -------------------------------------------------------------------------

    left, right = st.columns(
        [
            4,
            1,
        ]
    )

    with left:
        st.caption(
            (
                "Portfolio Scoring evaluates every claim independently "
                "and ranks the resulting scores across the current "
                "population. Investigation Queue converts those scores "
                "into an operational human-review workflow."
            )
        )

    with right:

        if st.button(
            "Reset Portfolio",
            use_container_width=True,
            key="reset_portfolio_scoring",
        ):
            _reset_portfolio()

            st.rerun()

    st.write("")

    # -------------------------------------------------------------------------
    # Portfolio input
    # -------------------------------------------------------------------------

    _render_portfolio_input(
        client
    )

    frame = st.session_state.get(
        "batch_results"
    )

    metadata = (
        st.session_state.get(
            "batch_metadata"
        )
        or {}
    )

    if (
        frame is None
        or not isinstance(
            frame,
            pd.DataFrame,
        )
        or frame.empty
    ):

        st.write("")
        st.write("")

        empty_state(
            "No Portfolio Scored",
            (
                "Upload a JSON, CSV or Parquet portfolio, "
                "or use the synthetic demo portfolio to "
                "generate fraud-risk scores."
            ),
            hint=(
                "No portfolio analytics are generated "
                "until model inference has completed."
            ),
        )

        return

    # -------------------------------------------------------------------------
    # Stored-result integrity
    # -------------------------------------------------------------------------

    integrity_errors = (
        _validate_stored_results(
            frame
        )
    )

    if integrity_errors:

        for message in integrity_errors:
            st.error(message)

        info_panel(
            "Portfolio Snapshot Invalid",
            (
                "Reset the current portfolio and run scoring again "
                "before continuing with analytics or investigation."
            ),
            tone="danger",
        )

        return

    # -------------------------------------------------------------------------
    # Analytics
    # -------------------------------------------------------------------------

    st.write("")
    st.write("")

    _render_kpis(
        frame,
        metadata,
    )

    _render_distribution(
        frame
    )

    _render_review_concentration(
        frame,
        metadata,
    )

    _render_quantiles(
        frame
    )

    _render_top_risk(
        frame
    )

    filtered = _render_filters(
        frame
    )

    _render_table(
        filtered
    )

    _render_claim_detail(
        (
            filtered
            if not filtered.empty
            else frame
        )
    )

    _render_exports(
        frame,
        metadata,
    )

    st.write("")

    human_review_notice()