from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from components import (
    human_review_notice,
    info_panel,
    metric_card,
    section_header,
)


# =============================================================================
# Paths
# =============================================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "artifacts"
)


EXPLAINABILITY_DIR = (
    ARTIFACTS_DIR
    / "explainability"
)


FIGURES_DIR = (
    EXPLAINABILITY_DIR
    / "figures"
)


METADATA_PATH = (
    ARTIFACTS_DIR
    / "metadata"
    / "health_fraud_model_metadata.json"
)


FINAL_EVALUATION_DIR = (
    ARTIFACTS_DIR
    / "metadata"
    / "final_evaluation"
)


# =============================================================================
# Constants
# =============================================================================


TOP_SHAP_FEATURES = 15


MECHANISM_SCORE_FILE = (
    "mechanism_score_summary.csv"
)


FALSE_NEGATIVE_FILE = (
    "false_negative_by_mechanism.csv"
)


DIFFICULTY_SCORE_FILE = (
    "difficulty_score_summary.csv"
)


BUSINESS_IMPORTANCE_FILE = (
    "business_feature_importance.csv"
)


EVALUATION_FIGURES = [
    (
        "01_confusion_matrix_top3.png",
        "Confusion Matrix — Top 3%",
        (
            "Classification outcomes at the "
            "3% operational review policy."
        ),
    ),
    (
        "02_precision_recall_test.png",
        "Precision–Recall Curve",
        (
            "Precision/recall trade-off on the "
            "out-of-time test population."
        ),
    ),
    (
        "03_roc_test.png",
        "ROC Curve",
        (
            "Global ranking discrimination "
            "across decision thresholds."
        ),
    ),
    (
        "04_calibration_test.png",
        "Calibration Curve",
        (
            "Agreement between predicted risk "
            "and observed synthetic fraud frequency."
        ),
    ),
    (
        "05_capacity_curve.png",
        "Investigation Capacity",
        (
            "Fraud capture as investigation "
            "capacity increases."
        ),
    ),
]


ERROR_FIGURES = [
    (
        "case_false_negative_extreme.png",
        "Extreme False Negative",
    ),
    (
        "case_false_positive_high_risk.png",
        "High-Risk False Positive",
    ),
    (
        "case_amount_inflation_false_negative.png",
        "Amount Inflation False Negative",
    ),
    (
        "case_legitimate_anomaly_not_reviewed.png",
        "Legitimate Anomaly",
    ),
]


# =============================================================================
# Generic helpers
# =============================================================================


def _safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
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

    return default


def _safe_int(
    value: Any,
    default: int | None = None,
) -> int | None:
    """
    Convert a numeric-like value safely to integer.
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


def _pretty_name(
    value: Any,
) -> str:
    """
    Convert a machine feature / category name into readable text.
    """

    if value is None:
        return "—"

    return (
        str(
            value
        )
        .replace(
            "_",
            " ",
        )
        .strip()
        .title()
    )


def _format_metric(
    value: Any,
    digits: int = 3,
) -> str:
    """
    Format decimal metric safely.
    """

    number = (
        _safe_float(
            value
        )
    )

    if number is None:
        return "—"

    return (
        f"{number:.{digits}f}"
    )


def _format_percent(
    value: Any,
    digits: int = 1,
) -> str:
    """
    Format probability / ratio safely.
    """

    number = (
        _safe_float(
            value
        )
    )

    if number is None:
        return "—"

    return (
        f"{number:.{digits}%}"
    )


def _format_lift(
    value: Any,
) -> str:
    """
    Format lift metric.
    """

    number = (
        _safe_float(
            value
        )
    )

    if number is None:
        return "—"

    return (
        f"{number:.2f}×"
    )


# =============================================================================
# File loading
# =============================================================================


@st.cache_data(
    show_spinner=False,
)
def _read_csv(
    path_string: str,
) -> pd.DataFrame:
    """
    Read analytical CSV defensively.
    """

    path = Path(
        path_string
    )

    if not path.exists():
        return pd.DataFrame()

    try:

        frame = (
            pd.read_csv(
                path
            )
        )

    except Exception:
        return pd.DataFrame()

    return frame


@st.cache_data(
    show_spinner=False,
)
def _read_metadata(
    path_string: str,
) -> dict[str, Any]:
    """
    Read frozen-model metadata defensively.
    """

    path = Path(
        path_string
    )

    if not path.exists():
        return {}

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            payload = (
                json.load(
                    file
                )
            )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    if not isinstance(
        payload,
        dict,
    ):
        return {}

    return payload


def _load_csv(
    filename: str,
) -> pd.DataFrame:
    """
    Resolve one explainability CSV.
    """

    return (
        _read_csv(
            str(
                EXPLAINABILITY_DIR
                / filename
            )
        )
    )


def _load_metadata() -> dict[str, Any]:
    """
    Resolve model metadata.
    """

    return (
        _read_metadata(
            str(
                METADATA_PATH
            )
        )
    )


def _existing_image(
    directory: Path,
    name: str,
) -> Path | None:
    """
    Return image path only when the file exists.
    """

    path = (
        directory
        / name
    )

    if (
        path.exists()
        and path.is_file()
    ):
        return path

    return None


# =============================================================================
# Runtime model information
# =============================================================================


def _read_runtime_model(
    client,
) -> tuple[
    dict[str, Any],
    str | None,
]:
    """
    Retrieve the model contract exposed by the live API.

    Local artifact metadata remains useful for evaluation statistics,
    while the API is the source of truth for the currently served model.
    """

    try:

        payload = (
            client.model_info()
        )

        if not isinstance(
            payload,
            dict,
        ):
            return (
                {},
                (
                    "Inference API returned an invalid "
                    "model contract."
                ),
            )

        return (
            payload,
            None,
        )

    except Exception as exc:

        return (
            {},
            str(
                exc
            ),
        )


# =============================================================================
# Metadata helpers
# =============================================================================


def _final_metrics(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Return final out-of-time metrics.
    """

    metrics = (
        metadata.get(
            "final_test_metrics",
            {},
        )
    )

    if not isinstance(
        metrics,
        dict,
    ):
        return {}

    return metrics


def _review_policy(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
) -> dict[str, Any]:
    """
    Resolve deployed review policy, preferring live API contract.
    """

    runtime_policy = (
        runtime_model.get(
            "review_policy"
        )
    )

    if isinstance(
        runtime_policy,
        dict,
    ):
        return runtime_policy

    metadata_policy = (
        metadata.get(
            "review_policy",
            {},
        )
    )

    if isinstance(
        metadata_policy,
        dict,
    ):
        return metadata_policy

    return {}


# =============================================================================
# Contract consistency
# =============================================================================


def _contract_consistency(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
) -> tuple[
    bool | None,
    list[str],
]:
    """
    Compare local frozen metadata with the currently served API model.
    """

    if (
        not metadata
        or not runtime_model
    ):
        return (
            None,
            [],
        )

    comparisons = [
        (
            "model_name",
            "Model name",
        ),
        (
            "model_version",
            "Model version",
        ),
        (
            "target",
            "Target",
        ),
        (
            "feature_count",
            "Feature count",
        ),
    ]

    mismatches: list[
        str
    ] = []

    compared = 0

    for (
        key,
        label,
    ) in comparisons:

        local_value = (
            metadata.get(
                key
            )
        )

        runtime_value = (
            runtime_model.get(
                key
            )
        )

        if (
            local_value is None
            or runtime_value is None
        ):
            continue

        compared += 1

        if (
            str(
                local_value
            )
            != str(
                runtime_value
            )
        ):

            mismatches.append(
                (
                    f"{label}: artifact={local_value} "
                    f"• runtime={runtime_value}"
                )
            )

    if compared == 0:

        return (
            None,
            [],
        )

    return (
        len(
            mismatches
        )
        == 0,
        mismatches,
    )


# =============================================================================
# Performance summary
# =============================================================================


def _render_model_summary(
    metadata: dict[str, Any],
) -> None:
    """
    Render primary out-of-time performance metrics.
    """

    section_header(
        "Model Performance",
        (
            "Out-of-time evaluation of the frozen "
            "fraud-risk ranking model."
        ),
        eyebrow="OUT-OF-TIME EVALUATION",
    )

    metrics = (
        _final_metrics(
            metadata
        )
    )

    if not metrics:

        info_panel(
            "Evaluation Metrics Unavailable",
            (
                "The final_test_metrics section is not "
                "available in the local model metadata."
            ),
            tone="warning",
        )

        return

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Average Precision",
            _format_metric(
                metrics.get(
                    "average_precision"
                ),
                4,
            ),
            "Primary ranking metric",
            tone="info",
        )

    with c2:

        metric_card(
            "ROC-AUC",
            _format_metric(
                metrics.get(
                    "roc_auc"
                ),
                4,
            ),
            "Global discrimination",
            tone="info",
        )

    with c3:

        metric_card(
            "Recall @ 3%",
            _format_percent(
                metrics.get(
                    "recall_at_3pct"
                ),
                2,
            ),
            "Fraud cases captured",
            tone="success",
        )

    with c4:

        metric_card(
            "Lift @ 3%",
            _format_lift(
                metrics.get(
                    "lift_at_3pct"
                )
            ),
            "Versus random review",
            tone="success",
        )

    st.write("")

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Precision @ 3%",
            _format_percent(
                metrics.get(
                    "precision_at_3pct"
                ),
                2,
            ),
            "Investigation yield",
        )

    with c2:

        metric_card(
            "Fraud Amount Capture",
            _format_percent(
                metrics.get(
                    "fraud_amount_capture_at_3pct"
                ),
                2,
            ),
            "At 3% review capacity",
            tone="success",
        )

    with c3:

        metric_card(
            "Brier Score",
            _format_metric(
                metrics.get(
                    "brier_score"
                ),
                4,
            ),
            "Lower is better",
        )

    with c4:

        metric_card(
            "Log Loss",
            _format_metric(
                metrics.get(
                    "log_loss"
                ),
                4,
            ),
            "Probability quality",
        )

    st.write("")

    info_panel(
        "Operational Reading",
        (
            "For this use case, ranking quality under constrained "
            "review capacity is more operationally important than "
            "a single classification threshold. Recall, precision, "
            "lift and amount capture should therefore be interpreted "
            "together."
        ),
        tone="info",
    )


# =============================================================================
# Model contract
# =============================================================================


def _render_model_contract(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
    runtime_error: str | None,
) -> None:
    """
    Render frozen artifact information and live inference contract.
    """

    st.write("")
    st.write("")

    section_header(
        "Frozen Model Contract",
        (
            "Identity, deployment contract and operational "
            "review policy of the current model."
        ),
        eyebrow="MODEL GOVERNANCE",
    )

    # API contract takes precedence for deployment identity.
    model_name = (
        runtime_model.get(
            "model_name"
        )
        or metadata.get(
            "model_name"
        )
        or "—"
    )

    model_version = (
        runtime_model.get(
            "model_version"
        )
        or metadata.get(
            "model_version"
        )
        or "—"
    )

    target = (
        runtime_model.get(
            "target"
        )
        or metadata.get(
            "target"
        )
        or "—"
    )

    feature_count = (
        runtime_model.get(
            "feature_count"
        )
        or metadata.get(
            "feature_count"
        )
        or "—"
    )

    probability_method = (
        runtime_model.get(
            "probability_method"
        )
        or metadata.get(
            "probability_method"
        )
        or "—"
    )

    policy = (
        _review_policy(
            metadata,
            runtime_model,
        )
    )

    fraction = (
        _safe_float(
            policy.get(
                "fraction"
            )
        )
    )

    left, right = (
        st.columns(
            [
                1.2,
                1,
            ],
            gap="large",
        )
    )

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### Deployed Model"
            )

            m1, m2 = (
                st.columns(2)
            )

            with m1:

                metric_card(
                    "Algorithm",
                    str(
                        model_name
                    ),
                    "Runtime estimator",
                    tone="info",
                )

            with m2:

                metric_card(
                    "Version",
                    str(
                        model_version
                    ),
                    "Frozen deployment version",
                )

            st.write("")

            details_left, details_right = (
                st.columns(2)
            )

            with details_left:

                st.caption(
                    "TARGET"
                )

                st.code(
                    str(
                        target
                    ),
                    language=None,
                )

                st.caption(
                    "FEATURE COUNT"
                )

                st.write(
                    str(
                        feature_count
                    )
                )

            with details_right:

                st.caption(
                    "PROBABILITY METHOD"
                )

                st.write(
                    str(
                        probability_method
                    )
                )

                st.caption(
                    "TRAINING PERIOD END"
                )

                st.write(
                    str(
                        metadata.get(
                            "training_period_end",
                            "—",
                        )
                    )
                )

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### Review Policy"
            )

            metric_card(
                "Investigation Capacity",
                (
                    f"{fraction:.0%}"
                    if fraction
                    is not None
                    else "—"
                ),
                "Default operational policy",
                tone="info",
            )

            st.write("")

            st.write(
                (
                    "**Policy type:** "
                    f"{policy.get('type', '—')}"
                )
            )

            test_period = (
                metadata.get(
                    "test_period",
                    {},
                )
            )

            if not isinstance(
                test_period,
                dict,
            ):
                test_period = {}

            st.write(
                (
                    "**Test start:** "
                    f"{test_period.get('start', '—')}"
                )
            )

            st.write(
                (
                    "**Test end:** "
                    f"{test_period.get('end', '—')}"
                )
            )

            st.caption(
                (
                    "The review fraction controls portfolio "
                    "selection. It is not an individual fraud "
                    "classification threshold."
                )
            )

    st.write("")

    consistent, mismatches = (
        _contract_consistency(
            metadata,
            runtime_model,
        )
    )

    if consistent is True:

        info_panel(
            "Contract Consistency",
            (
                "The local frozen-model metadata is consistent "
                "with the model contract currently exposed by "
                "the inference API."
            ),
            tone="success",
        )

    elif consistent is False:

        st.warning(
            (
                "Local model metadata differs from "
                "the model currently exposed by the API."
            )
        )

        with st.expander(
            "Contract differences",
            expanded=False,
        ):

            for mismatch in mismatches:

                st.write(
                    f"- {mismatch}"
                )

    elif runtime_error:

        info_panel(
            "Runtime Contract Unavailable",
            (
                "Local evaluation metadata remains available, "
                "but the live inference contract could not be "
                "verified against the API."
            ),
            tone="warning",
        )


# =============================================================================
# Global SHAP
# =============================================================================


def _render_global_explainability() -> None:
    """
    Render business-level global SHAP diagnostics.
    """

    st.write("")
    st.write("")

    section_header(
        "Global Explainability",
        (
            "Business features with the strongest average "
            "influence on model output."
        ),
        eyebrow="SHAP",
    )

    importance = (
        _load_csv(
            BUSINESS_IMPORTANCE_FILE
        )
    )

    required_columns = {
        "business_feature",
        "mean_abs_shap",
    }

    if (
        importance.empty
        or not required_columns.issubset(
            importance.columns
        )
    ):

        info_panel(
            "Global SHAP Unavailable",
            (
                "The business-level SHAP importance artifact "
                "is missing or does not contain the expected fields."
            ),
            tone="warning",
        )

    else:

        importance = (
            importance.copy()
        )

        importance[
            "mean_abs_shap"
        ] = pd.to_numeric(
            importance[
                "mean_abs_shap"
            ],
            errors="coerce",
        )

        importance = (
            importance
            .dropna(
                subset=[
                    "mean_abs_shap"
                ]
            )
        )

        top = (
            importance
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .head(
                TOP_SHAP_FEATURES
            )
            .copy()
        )

        top[
            "display_feature"
        ] = (
            top[
                "business_feature"
            ]
            .apply(
                _pretty_name
            )
        )

        chart = (
            alt.Chart(
                top
            )
            .mark_bar(
                cornerRadiusEnd=5,
            )
            .encode(
                x=alt.X(
                    "mean_abs_shap:Q",
                    title=(
                        "Mean absolute SHAP contribution"
                    ),
                ),

                y=alt.Y(
                    "display_feature:N",
                    sort="-x",
                    title=None,
                ),

                tooltip=[
                    alt.Tooltip(
                        "display_feature:N",
                        title="Feature",
                    ),

                    alt.Tooltip(
                        "mean_abs_shap:Q",
                        title="Mean |SHAP|",
                        format=".4f",
                    ),
                ],
            )
            .properties(
                height=500
            )
        )

        if (
            "signed_mean_shap"
            in top.columns
        ):

            chart = (
                alt.Chart(
                    top
                )
                .mark_bar(
                    cornerRadiusEnd=5,
                )
                .encode(
                    x=alt.X(
                        "mean_abs_shap:Q",
                        title=(
                            "Mean absolute SHAP contribution"
                        ),
                    ),

                    y=alt.Y(
                        "display_feature:N",
                        sort="-x",
                        title=None,
                    ),

                    tooltip=[
                        alt.Tooltip(
                            "display_feature:N",
                            title="Feature",
                        ),

                        alt.Tooltip(
                            "mean_abs_shap:Q",
                            title="Mean |SHAP|",
                            format=".4f",
                        ),

                        alt.Tooltip(
                            "signed_mean_shap:Q",
                            title="Signed mean SHAP",
                            format=".4f",
                        ),
                    ],
                )
                .properties(
                    height=500
                )
            )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

        st.write("")

        table_columns = [
            column

            for column
            in [
                "display_feature",
                "mean_abs_shap",
                "signed_mean_shap",
            ]

            if column
            in top.columns
        ]

        with st.expander(
            "Top SHAP features",
            expanded=False,
        ):

            st.dataframe(
                top[
                    table_columns
                ],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "display_feature":
                        st.column_config.TextColumn(
                            "Feature",
                            width="large",
                        ),

                    "mean_abs_shap":
                        st.column_config.NumberColumn(
                            "Mean |SHAP|",
                            format="%.5f",
                        ),

                    "signed_mean_shap":
                        st.column_config.NumberColumn(
                            "Signed Mean SHAP",
                            format="%.5f",
                        ),
                },
            )

    st.write("")

    left, right = (
        st.columns(2)
    )

    shap_global = (
        _existing_image(
            FIGURES_DIR,
            "01_shap_global_bar.png",
        )
    )

    shap_beeswarm = (
        _existing_image(
            FIGURES_DIR,
            "02_shap_beeswarm.png",
        )
    )

    with left:

        st.markdown(
            "#### SHAP Global Importance"
        )

        if shap_global is not None:

            st.image(
                str(
                    shap_global
                ),
                use_container_width=True,
            )

        else:

            empty_state_message = (
                "Global SHAP figure is not available "
                "in the current explainability artifacts."
            )

            st.info(
                empty_state_message
            )

    with right:

        st.markdown(
            "#### SHAP Distribution"
        )

        if shap_beeswarm is not None:

            st.image(
                str(
                    shap_beeswarm
                ),
                use_container_width=True,
            )

        else:

            st.info(
                (
                    "SHAP beeswarm figure is not "
                    "available in the current artifacts."
                )
            )

    st.write("")

    info_panel(
        "Interpretation",
        (
            "Mean absolute SHAP quantifies influence magnitude, "
            "not causal importance. A globally important feature "
            "can increase risk for some claims and decrease it for others."
        ),
        tone="info",
    )


# =============================================================================
# Fraud mechanism analysis
# =============================================================================


def _render_mechanism_analysis() -> dict[str, Any]:
    """
    Analyze performance across synthetic fraud mechanisms.

    Returns a compact summary used by the final interpretation section.
    """

    st.write("")
    st.write("")

    section_header(
        "Fraud Mechanism Analysis",
        (
            "How risk scores vary across different "
            "synthetic fraud-generation mechanisms."
        ),
        eyebrow="SEGMENT PERFORMANCE",
    )

    mechanism = (
        _load_csv(
            MECHANISM_SCORE_FILE
        )
    )

    missed = (
        _load_csv(
            FALSE_NEGATIVE_FILE
        )
    )

    required = {
        "fraud_mechanism",
        "mean_score",
    }

    if (
        mechanism.empty
        or not required.issubset(
            mechanism.columns
        )
    ):

        info_panel(
            "Mechanism Analysis Unavailable",
            (
                "The mechanism score artifact is missing "
                "or does not contain the required fields."
            ),
            tone="warning",
        )

        return {}

    data = (
        mechanism.copy()
    )

    for column in [
        "mean_score",
        "median_score",
        "fraud_claims",
    ]:

        if column in data.columns:

            data[
                column
            ] = pd.to_numeric(
                data[
                    column
                ],
                errors="coerce",
            )

    data[
        "Mechanism"
    ] = (
        data[
            "fraud_mechanism"
        ]
        .apply(
            _pretty_name
        )
    )

    if (
        not missed.empty
        and {
            "fraud_mechanism",
            "missed_fraud_claims",
        }.issubset(
            missed.columns
        )
    ):

        missed = (
            missed.copy()
        )

        missed[
            "Mechanism"
        ] = (
            missed[
                "fraud_mechanism"
            ]
            .apply(
                _pretty_name
            )
        )

        data = (
            data.merge(
                missed[
                    [
                        "Mechanism",
                        "missed_fraud_claims",
                    ]
                ],
                on="Mechanism",
                how="left",
            )
        )

    left, right = (
        st.columns(
            [
                1.3,
                1,
            ],
            gap="large",
        )
    )

    with left:

        score_chart = (
            alt.Chart(
                data
            )
            .mark_bar(
                cornerRadiusEnd=5,
            )
            .encode(
                x=alt.X(
                    "mean_score:Q",
                    title="Mean fraud-risk score",
                    axis=alt.Axis(
                        format=".0%",
                    ),
                ),

                y=alt.Y(
                    "Mechanism:N",
                    sort="-x",
                    title=None,
                ),

                tooltip=[
                    alt.Tooltip(
                        "Mechanism:N",
                        title="Mechanism",
                    ),

                    alt.Tooltip(
                        "fraud_claims:Q",
                        title="Fraud claims",
                    ),

                    alt.Tooltip(
                        "mean_score:Q",
                        title="Mean score",
                        format=".2%",
                    ),

                    alt.Tooltip(
                        "median_score:Q",
                        title="Median score",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=360
            )
        )

        st.altair_chart(
            score_chart,
            use_container_width=True,
        )

    with right:

        visible_columns = [
            column

            for column
            in [
                "Mechanism",
                "fraud_claims",
                "mean_score",
                "median_score",
                "missed_fraud_claims",
            ]

            if column
            in data.columns
        ]

        st.dataframe(
            data[
                visible_columns
            ],
            use_container_width=True,
            hide_index=True,
            height=360,
            column_config={
                "Mechanism":
                    st.column_config.TextColumn(
                        "Mechanism",
                        width="large",
                    ),

                "fraud_claims":
                    st.column_config.NumberColumn(
                        "Fraud Claims",
                        format="%d",
                    ),

                "mean_score":
                    st.column_config.ProgressColumn(
                        "Mean Risk",
                        min_value=0,
                        max_value=1,
                        format="%.2f",
                    ),

                "median_score":
                    st.column_config.NumberColumn(
                        "Median Risk",
                        format="%.3f",
                    ),

                "missed_fraud_claims":
                    st.column_config.NumberColumn(
                        "Missed",
                        format="%d",
                    ),
            },
        )

    valid_scores = (
        data.dropna(
            subset=[
                "mean_score"
            ]
        )
    )

    if valid_scores.empty:
        return {}

    weakest = (
        valid_scores
        .sort_values(
            "mean_score",
            ascending=True,
        )
        .iloc[
            0
        ]
    )

    strongest = (
        valid_scores
        .sort_values(
            "mean_score",
            ascending=False,
        )
        .iloc[
            0
        ]
    )

    st.write("")

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        metric_card(
            "Strongest Mechanism",
            str(
                strongest[
                    "Mechanism"
                ]
            ),
            (
                "Mean risk "
                f"{_format_percent(strongest['mean_score'], 1)}"
            ),
            tone="success",
        )

    with c2:

        metric_card(
            "Weakest Mechanism",
            str(
                weakest[
                    "Mechanism"
                ]
            ),
            (
                "Mean risk "
                f"{_format_percent(weakest['mean_score'], 1)}"
            ),
            tone="warning",
        )

    st.write("")

    info_panel(
        "Observed Weakness",
        (
            f"{weakest['Mechanism']} currently has the "
            "lowest mean model risk among the analyzed "
            "synthetic fraud mechanisms "
            f"({_format_percent(weakest['mean_score'], 1)})."
        ),
        tone="warning",
    )

    return {
        "weakest_mechanism":
            str(
                weakest[
                    "Mechanism"
                ]
            ),

        "weakest_mechanism_score":
            _safe_float(
                weakest[
                    "mean_score"
                ]
            ),

        "strongest_mechanism":
            str(
                strongest[
                    "Mechanism"
                ]
            ),

        "strongest_mechanism_score":
            _safe_float(
                strongest[
                    "mean_score"
                ]
            ),
    }


# =============================================================================
# Difficulty analysis
# =============================================================================


def _render_difficulty_analysis() -> dict[str, Any]:
    """
    Analyze scores across synthetic fraud difficulty levels.
    """

    st.write("")
    st.write("")

    section_header(
        "Fraud Difficulty",
        (
            "Model confidence across easy, medium "
            "and hard synthetic fraud cases."
        ),
        eyebrow="ROBUSTNESS",
    )

    difficulty = (
        _load_csv(
            DIFFICULTY_SCORE_FILE
        )
    )

    required = {
        "fraud_difficulty",
        "mean_score",
    }

    if (
        difficulty.empty
        or not required.issubset(
            difficulty.columns
        )
    ):

        info_panel(
            "Difficulty Analysis Unavailable",
            (
                "The difficulty analysis artifact "
                "is missing or incomplete."
            ),
            tone="warning",
        )

        return {}

    difficulty = (
        difficulty.copy()
    )

    for column in [
        "mean_score",
        "median_score",
        "fraud_claims",
    ]:

        if column in difficulty.columns:

            difficulty[
                column
            ] = pd.to_numeric(
                difficulty[
                    column
                ],
                errors="coerce",
            )

    difficulty[
        "Difficulty"
    ] = (
        difficulty[
            "fraud_difficulty"
        ]
        .apply(
            _pretty_name
        )
    )

    chart = (
        alt.Chart(
            difficulty
        )
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X(
                "Difficulty:N",
                sort=[
                    "Easy",
                    "Medium",
                    "Hard",
                ],
                title=None,
            ),

            y=alt.Y(
                "mean_score:Q",
                title="Mean Fraud Risk",
                axis=alt.Axis(
                    format=".0%",
                ),
            ),

            tooltip=[
                alt.Tooltip(
                    "Difficulty:N",
                    title="Difficulty",
                ),

                alt.Tooltip(
                    "fraud_claims:Q",
                    title="Fraud claims",
                ),

                alt.Tooltip(
                    "mean_score:Q",
                    title="Mean risk",
                    format=".2%",
                ),

                alt.Tooltip(
                    "median_score:Q",
                    title="Median risk",
                    format=".2%",
                ),
            ],
        )
        .properties(
            height=320
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    columns = (
        st.columns(
            3
        )
    )

    summary: dict[
        str,
        Any,
    ] = {}

    tone_map = {
        "easy":
            "success",

        "medium":
            "info",

        "hard":
            "warning",
    }

    for (
        column,
        level,
    ) in zip(
        columns,
        [
            "easy",
            "medium",
            "hard",
        ],
    ):

        subset = (
            difficulty.loc[
                difficulty[
                    "fraud_difficulty"
                ]
                .astype(str)
                .str.lower()
                == level
            ]
        )

        if subset.empty:

            with column:

                metric_card(
                    f"{level.title()} Fraud",
                    "—",
                    "No available observations",
                )

            continue

        row = (
            subset.iloc[
                0
            ]
        )

        mean_score = (
            _safe_float(
                row.get(
                    "mean_score"
                )
            )
        )

        fraud_claims = (
            _safe_int(
                row.get(
                    "fraud_claims"
                )
            )
        )

        summary[
            level
        ] = mean_score

        with column:

            metric_card(
                f"{level.title()} Fraud",
                _format_percent(
                    mean_score,
                    1,
                ),
                (
                    f"{fraud_claims:,} fraud claims"
                    if fraud_claims
                    is not None
                    else "Fraud cases"
                ),
                tone=tone_map[
                    level
                ],
            )

    easy_score = (
        summary.get(
            "easy"
        )
    )

    hard_score = (
        summary.get(
            "hard"
        )
    )

    if (
        easy_score is not None
        and hard_score is not None
    ):

        gap = (
            easy_score
            - hard_score
        )

        st.write("")

        info_panel(
            "Difficulty Gap",
            (
                "Mean risk decreases by "
                f"{gap:.1%} from easy to hard synthetic fraud."
            ),
            tone=(
                "warning"
                if gap > 0
                else "info"
            ),
        )

        summary[
            "easy_hard_gap"
        ] = gap

    return summary


# =============================================================================
# Evaluation diagnostics
# =============================================================================


def _render_evaluation_diagnostics() -> None:
    """
    Render frozen out-of-time diagnostic plots.
    """

    st.write("")
    st.write("")

    section_header(
        "Evaluation Diagnostics",
        (
            "Out-of-time figures used to inspect ranking, "
            "classification behavior and probability calibration."
        ),
        eyebrow="MODEL VALIDATION",
    )

    tabs = (
        st.tabs(
            [
                title

                for (
                    _,
                    title,
                    _,
                ) in EVALUATION_FIGURES
            ]
        )
    )

    for (
        tab,
        specification,
    ) in zip(
        tabs,
        EVALUATION_FIGURES,
    ):

        (
            filename,
            title,
            description,
        ) = specification

        with tab:

            st.caption(
                description
            )

            path = (
                _existing_image(
                    FINAL_EVALUATION_DIR,
                    filename,
                )
            )

            if path is not None:

                st.image(
                    str(
                        path
                    ),
                    use_container_width=True,
                )

            else:

                info_panel(
                    "Figure Unavailable",
                    (
                        f"{title} is not present in the "
                        "current final-evaluation artifacts."
                    ),
                    tone="warning",
                )


# =============================================================================
# Error analysis
# =============================================================================


def _render_error_analysis() -> None:
    """
    Render case-level qualitative failure analysis.
    """

    st.write("")
    st.write("")

    section_header(
        "Observed Failure Modes",
        (
            "Representative false-positive, false-negative "
            "and legitimate-anomaly cases."
        ),
        eyebrow="ERROR ANALYSIS",
    )

    available: list[
        tuple[
            Path,
            str,
        ]
    ] = []

    for (
        filename,
        label,
    ) in ERROR_FIGURES:

        path = (
            _existing_image(
                FIGURES_DIR,
                filename,
            )
        )

        if path is not None:

            available.append(
                (
                    path,
                    label,
                )
            )

    if not available:

        info_panel(
            "Error Analysis Unavailable",
            (
                "No case-level error-analysis figures "
                "are present in the current artifacts."
            ),
            tone="warning",
        )

        return

    columns = (
        st.columns(
            2
        )
    )

    for (
        index,
        item,
    ) in enumerate(
        available
    ):

        (
            path,
            label,
        ) = item

        with columns[
            index % 2
        ]:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"#### {label}"
                )

                st.image(
                    str(
                        path
                    ),
                    use_container_width=True,
                )


# =============================================================================
# Data / artifact coverage
# =============================================================================


def _render_artifact_coverage() -> None:
    """
    Surface analytical-artifact availability explicitly.
    """

    st.write("")
    st.write("")

    section_header(
        "Analytical Artifact Coverage",
        (
            "Availability of the evidence supporting "
            "this model-insights page."
        ),
        eyebrow="TRACEABILITY",
    )

    checks = [
        (
            "Model metadata",
            METADATA_PATH.exists(),
        ),
        (
            "Business SHAP importance",
            (
                EXPLAINABILITY_DIR
                / BUSINESS_IMPORTANCE_FILE
            ).exists(),
        ),
        (
            "Mechanism analysis",
            (
                EXPLAINABILITY_DIR
                / MECHANISM_SCORE_FILE
            ).exists(),
        ),
        (
            "Difficulty analysis",
            (
                EXPLAINABILITY_DIR
                / DIFFICULTY_SCORE_FILE
            ).exists(),
        ),
        (
            "Global SHAP figure",
            (
                FIGURES_DIR
                / "01_shap_global_bar.png"
            ).exists(),
        ),
        (
            "SHAP beeswarm",
            (
                FIGURES_DIR
                / "02_shap_beeswarm.png"
            ).exists(),
        ),
    ]

    available_count = (
        sum(
            int(
                available
            )

            for (
                _,
                available,
            ) in checks
        )
    )

    total_count = (
        len(
            checks
        )
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:

        metric_card(
            "Artifacts Available",
            f"{available_count}/{total_count}",
            "Core analytical artifacts",
            tone=(
                "success"
                if available_count
                == total_count
                else "warning"
            ),
        )

    with c2:

        metric_card(
            "Explainability Directory",
            (
                "READY"
                if EXPLAINABILITY_DIR.exists()
                else "MISSING"
            ),
            "Frontend analytical assets",
            tone=(
                "success"
                if EXPLAINABILITY_DIR.exists()
                else "danger"
            ),
        )

    with c3:

        metric_card(
            "Evaluation Figures",
            str(
                sum(
                    1

                    for (
                        filename,
                        _,
                        _,
                    ) in EVALUATION_FIGURES

                    if (
                        FINAL_EVALUATION_DIR
                        / filename
                    ).exists()
                )
            ),
            (
                f"of {len(EVALUATION_FIGURES)} expected"
            ),
        )

    st.write("")

    coverage = (
        pd.DataFrame(
            [
                {
                    "Artifact":
                        label,

                    "Status":
                        (
                            "AVAILABLE"
                            if available
                            else "MISSING"
                        ),
                }

                for (
                    label,
                    available,
                ) in checks
            ]
        )
    )

    with st.expander(
        "Artifact inventory",
        expanded=False,
    ):

        st.dataframe(
            coverage,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# Interpretation summary
# =============================================================================


def _render_conclusions(
    metadata: dict[str, Any],
    mechanism_summary: dict[str, Any],
    difficulty_summary: dict[str, Any],
) -> None:
    """
    Generate conclusions from available analytical evidence.
    """

    st.write("")
    st.write("")

    section_header(
        "Model Interpretation Summary",
        (
            "Operational conclusions supported by the "
            "current evaluation and explainability artifacts."
        ),
        eyebrow="MODEL GOVERNANCE",
    )

    metrics = (
        _final_metrics(
            metadata
        )
    )

    lift = (
        _safe_float(
            metrics.get(
                "lift_at_3pct"
            )
        )
    )

    recall = (
        _safe_float(
            metrics.get(
                "recall_at_3pct"
            )
        )
    )

    precision = (
        _safe_float(
            metrics.get(
                "precision_at_3pct"
            )
        )
    )

    amount_capture = (
        _safe_float(
            metrics.get(
                "fraud_amount_capture_at_3pct"
            )
        )
    )

    strongest = (
        mechanism_summary.get(
            "strongest_mechanism"
        )
    )

    weakest = (
        mechanism_summary.get(
            "weakest_mechanism"
        )
    )

    difficulty_gap = (
        difficulty_summary.get(
            "easy_hard_gap"
        )
    )

    left, right = (
        st.columns(2)
    )

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Observed Strengths"
            )

            if lift is not None:

                st.write(
                    (
                        "• At the 3% review policy, "
                        f"model targeting delivers approximately "
                        f"**{lift:.2f}× lift** versus untargeted review."
                    )
                )

            if recall is not None:

                st.write(
                    (
                        "• The selected 3% of claims captures "
                        f"**{recall:.1%}** of synthetic fraud cases."
                    )
                )

            if amount_capture is not None:

                st.write(
                    (
                        "• The same policy captures "
                        f"**{amount_capture:.1%}** of synthetic "
                        "fraud amount."
                    )
                )

            if precision is not None:

                st.write(
                    (
                        "• Investigation yield at the policy "
                        f"point is **{precision:.1%}**."
                    )
                )

            if strongest:

                st.write(
                    (
                        "• Highest observed mean risk by "
                        f"fraud mechanism: **{strongest}**."
                    )
                )

            st.write(
                (
                    "• Global explainability is expressed "
                    "through business-level features rather than "
                    "target or synthetic-generation variables."
                )
            )

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Known Limitations"
            )

            if weakest:

                st.write(
                    (
                        "• Lowest observed mean fraud risk by "
                        f"mechanism: **{weakest}**."
                    )
                )

            if difficulty_gap is not None:

                st.write(
                    (
                        "• Hard synthetic fraud receives "
                        f"approximately **{difficulty_gap:.1%} lower "
                        "mean risk** than easy fraud."
                    )
                )

            st.write(
                (
                    "• SHAP contributions explain model behavior; "
                    "they do not establish causal relationships."
                )
            )

            st.write(
                (
                    "• Individual probability estimates must not "
                    "be interpreted as proof that a claim is fraudulent."
                )
            )

            st.write(
                (
                    "• Current validation is based on a synthetic "
                    "health-insurance environment and therefore does "
                    "not constitute evidence of real-world generalization."
                )
            )

    st.write("")

    human_review_notice()


# =============================================================================
# Main page
# =============================================================================


def render(
    client,
) -> None:
    """
    Render complete model-performance and explainability workspace.
    """

    section_header(
        "Model Insights",
        (
            "Performance, explainability, robustness, "
            "error analysis and governance of the frozen "
            "fraud-risk model."
        ),
    )

    metadata = (
        _load_metadata()
    )

    (
        runtime_model,
        runtime_error,
    ) = (
        _read_runtime_model(
            client
        )
    )

    if not metadata:

        info_panel(
            "Model Metadata Unavailable",
            (
                "The local frozen-model metadata artifact could "
                "not be loaded. Runtime model information may still "
                "be available from the inference API."
            ),
            tone="warning",
        )

        st.write("")

    _render_model_summary(
        metadata
    )

    _render_model_contract(
        metadata,
        runtime_model,
        runtime_error,
    )

    _render_global_explainability()

    mechanism_summary = (
        _render_mechanism_analysis()
    )

    difficulty_summary = (
        _render_difficulty_analysis()
    )

    _render_evaluation_diagnostics()

    _render_error_analysis()

    _render_artifact_coverage()

    _render_conclusions(
        metadata,
        mechanism_summary,
        difficulty_summary,
    )