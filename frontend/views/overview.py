from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from frontend.components import (
    human_review_notice,
    info_panel,
    key_value_row,
    metric_card,
    mini_metric,
    section_header,
    status_badge,
)

from frontend.utils.formatting import (
    format_review_policy,
)


# =============================================================================
# Configuration
# =============================================================================


DEFAULT_REVIEW_FRACTION = 0.03

MAX_RISK_DRIVERS = 8


# =============================================================================
# Project / artifact resolution
# =============================================================================


MODULE_PATH = (
    Path(
        __file__
    )
    .resolve()
)


PROJECT_ROOT = (
    MODULE_PATH.parents[2]
)


def _candidate_artifact_roots() -> list[Path]:
    """
    Resolve artifact roots consistently across:

    - local development;
    - GitHub Codespaces;
    - Docker frontend runtime;
    - explicit ARTIFACTS_ROOT configuration.
    """

    candidates: list[
        Path
    ] = []

    configured = (
        os.getenv(
            "ARTIFACTS_ROOT"
        )
    )

    if configured:

        candidates.append(
            Path(
                configured
            )
            .expanduser()
            .resolve()
        )

    candidates.extend(
        [
            PROJECT_ROOT
            / "artifacts",

            Path(
                "/app/artifacts"
            ),

            Path.cwd()
            / "artifacts",
        ]
    )

    unique: list[
        Path
    ] = []

    seen: set[
        str
    ] = set()

    for path in candidates:

        key = str(
            path
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            path
        )

    return unique


def _find_artifact_root() -> Path | None:
    """
    Return the first existing artifact root.
    """

    for path in (
        _candidate_artifact_roots()
    ):

        if (
            path.exists()
            and path.is_dir()
        ):
            return path

    return None


ARTIFACTS_DIR = (
    _find_artifact_root()
)


def _artifact_path(
    *parts: str,
) -> Path | None:
    """
    Resolve one artifact path relative to the detected root.
    """

    if ARTIFACTS_DIR is None:
        return None

    return (
        ARTIFACTS_DIR.joinpath(
            *parts
        )
    )


def _existing_artifact_path(
    *parts: str,
) -> Path | None:
    """
    Return an artifact path only when it exists and is a regular file.
    """

    path = (
        _artifact_path(
            *parts
        )
    )

    if (
        path is None
        or not path.exists()
        or not path.is_file()
    ):
        return None

    return path


def _artifact_relative_label(
    path: Path,
) -> str:
    """
    Return a human-readable artifact path relative to the artifact root.
    """

    if ARTIFACTS_DIR is None:
        return str(
            path
        )

    try:

        return str(
            path.relative_to(
                ARTIFACTS_DIR
            )
        )

    except ValueError:

        return str(
            path
        )


# =============================================================================
# Cached artifact loading
# =============================================================================


@st.cache_data(
    show_spinner=False,
)
def _read_json(
    path_string: str,
) -> dict[str, Any]:
    """
    Read one dictionary JSON artifact safely.
    """

    path = (
        Path(
            path_string
        )
    )

    if (
        not path.exists()
        or not path.is_file()
    ):
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


@st.cache_data(
    show_spinner=False,
)
def _read_csv(
    path_string: str,
) -> pd.DataFrame:
    """
    Read one analytical CSV artifact safely.
    """

    path = (
        Path(
            path_string
        )
    )

    if (
        not path.exists()
        or not path.is_file()
    ):
        return pd.DataFrame()

    try:

        frame = pd.read_csv(
            path
        )

    except (
        OSError,
        ValueError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ):

        return pd.DataFrame()

    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        return pd.DataFrame()

    return frame


def _load_metadata() -> dict[str, Any]:
    """
    Load frozen model metadata.
    """

    path = (
        _existing_artifact_path(
            "metadata",
            "health_fraud_model_metadata.json",
        )
    )

    if path is None:
        return {}

    return (
        _read_json(
            str(
                path
            )
        )
    )


def _load_artifact_csv(
    *parts: str,
) -> pd.DataFrame:
    """
    Load one CSV artifact relative to the detected artifact root.
    """

    path = (
        _existing_artifact_path(
            *parts
        )
    )

    if path is None:
        return pd.DataFrame()

    return (
        _read_csv(
            str(
                path
            )
        )
    )


# =============================================================================
# Generic numerical helpers
# =============================================================================


def _safe_float(
    value: Any,
) -> float | None:
    """
    Convert one value into a finite float.
    """

    if isinstance(
        value,
        bool,
    ):
        return None

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if not np.isfinite(
        result
    ):
        return None

    return result


def _safe_int(
    value: Any,
) -> int | None:
    """
    Convert an integer-like value to int.
    """

    result = (
        _safe_float(
            value
        )
    )

    if result is None:
        return None

    rounded = int(
        round(
            result
        )
    )

    return rounded


def _metric_number(
    value: Any,
    digits: int = 4,
) -> str:
    """
    Format a numerical KPI.
    """

    result = (
        _safe_float(
            value
        )
    )

    if result is None:
        return "—"

    return (
        f"{result:.{digits}f}"
    )


def _metric_percent(
    value: Any,
    digits: int = 2,
) -> str:
    """
    Format a fraction as a percentage.
    """

    result = (
        _safe_float(
            value
        )
    )

    if result is None:
        return "—"

    return (
        f"{result:.{digits}%}"
    )


def _metric_multiplier(
    value: Any,
    digits: int = 2,
) -> str:
    """
    Format a lift-like multiplier.
    """

    result = (
        _safe_float(
            value
        )
    )

    if result is None:
        return "—"

    return (
        f"{result:.{digits}f}×"
    )


def _pretty_feature(
    value: Any,
) -> str:
    """
    Convert a technical identifier into a readable label.
    """

    if value is None:
        return "—"

    text = (
        str(
            value
        )
        .strip()
    )

    if not text:
        return "—"

    return (
        text
        .replace(
            "_",
            " ",
        )
        .title()
    )


def _extract_metrics(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Return frozen final out-of-time metrics.
    """

    metrics = (
        metadata.get(
            "final_test_metrics"
        )
    )

    if isinstance(
        metrics,
        dict,
    ):
        return metrics

    return {}


# =============================================================================
# Review policy
# =============================================================================


def _review_fraction(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any] | None = None,
) -> float:
    """
    Resolve operational review capacity.

    Priority
    --------
    1. deployed runtime contract;
    2. frozen metadata;
    3. application default.
    """

    if isinstance(
        runtime_model,
        dict,
    ):

        policy = (
            runtime_model.get(
                "review_policy"
            )
        )

        if isinstance(
            policy,
            dict,
        ):

            value = (
                _safe_float(
                    policy.get(
                        "fraction"
                    )
                )
            )

            if (
                value is not None
                and 0 < value <= 1
            ):
                return value

    policy = (
        metadata.get(
            "review_policy"
        )
    )

    if isinstance(
        policy,
        dict,
    ):

        value = (
            _safe_float(
                policy.get(
                    "fraction"
                )
            )
        )

        if (
            value is not None
            and 0 < value <= 1
        ):
            return value

    return (
        DEFAULT_REVIEW_FRACTION
    )


def _review_fraction_label(
    fraction: float,
) -> str:
    """
    Format investigation capacity consistently.
    """

    percent_value = (
        fraction
        * 100
    )

    if float(
        percent_value
    ).is_integer():

        return (
            f"{fraction:.0%}"
        )

    return (
        f"{fraction:.1%}"
    )


def _capacity_metric_suffix(
    fraction: float,
) -> str:
    """
    Convert capacity fraction into persisted metric suffix.

    Examples
    --------
    0.03  -> 3pct
    0.05  -> 5pct
    0.025 -> 2_5pct
    """

    percent_value = (
        fraction
        * 100
    )

    if float(
        percent_value
    ).is_integer():

        return (
            f"{int(percent_value)}pct"
        )

    text = (
        f"{percent_value:.6f}"
        .rstrip(
            "0"
        )
        .rstrip(
            "."
        )
        .replace(
            ".",
            "_",
        )
    )

    return (
        f"{text}pct"
    )


def _capacity_metric(
    metrics: dict[str, Any],
    metric_name: str,
    fraction: float,
) -> Any:
    """
    Read a metric corresponding exactly to the active review capacity.

    Metrics from another capacity are never silently substituted.
    """

    suffix = (
        _capacity_metric_suffix(
            fraction
        )
    )

    key = (
        f"{metric_name}_at_{suffix}"
    )

    return (
        metrics.get(
            key
        )
    )


# =============================================================================
# Runtime API
# =============================================================================


def _read_runtime_system(
    client: Any,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str | None,
]:
    """
    Read health and deployed model contract once per page render.
    """

    try:

        health = (
            client.health()
        )

        model = (
            client.model_info()
        )

    except Exception as exc:

        return (
            {},
            {},
            str(
                exc
            ),
        )

    if not isinstance(
        health,
        dict,
    ):
        health = {}

    if not isinstance(
        model,
        dict,
    ):
        model = {}

    return (
        health,
        model,
        None,
    )


# =============================================================================
# Metadata notice
# =============================================================================


def _render_metadata_notice(
    metadata: dict[str, Any],
) -> None:
    """
    Surface analytical artifact availability.
    """

    if metadata:
        return

    if ARTIFACTS_DIR is None:

        info_panel(
            "Evaluation Artifacts Unavailable",
            (
                "Performance and explainability artifacts are not "
                "mounted in the frontend runtime. Live inference may "
                "still remain available through the API."
            ),
            tone="warning",
        )

        return

    info_panel(
        "Model Metadata Unavailable",
        (
            "The artifact directory is mounted, but the frozen "
            "model metadata could not be loaded."
        ),
        tone="warning",
    )


# =============================================================================
# Executive KPIs
# =============================================================================


def _render_executive_kpis(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
) -> None:
    """
    Render frozen out-of-time model performance.
    """

    metrics = (
        _extract_metrics(
            metadata
        )
    )

    review_fraction = (
        _review_fraction(
            metadata,
            runtime_model,
        )
    )

    review_label = (
        _review_fraction_label(
            review_fraction
        )
    )

    section_header(
        "Executive Performance",
        (
            "Out-of-time evaluation of the frozen "
            "fraud-risk ranking model."
        ),
        eyebrow="MODEL PERFORMANCE",
    )

    if not metrics:

        info_panel(
            "Evaluation Metrics Unavailable",
            (
                "Final out-of-time evaluation metrics are not "
                "available in the current model metadata."
            ),
            tone="warning",
        )

        return

    recall = (
        _capacity_metric(
            metrics,
            "recall",
            review_fraction,
        )
    )

    lift = (
        _capacity_metric(
            metrics,
            "lift",
            review_fraction,
        )
    )

    precision = (
        _capacity_metric(
            metrics,
            "precision",
            review_fraction,
        )
    )

    fraud_amount_capture = (
        _capacity_metric(
            metrics,
            "fraud_amount_capture",
            review_fraction,
        )
    )

    c1, c2, c3, c4 = (
        st.columns(
            4
        )
    )

    with c1:

        metric_card(
            "Average Precision",
            _metric_number(
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
            _metric_number(
                metrics.get(
                    "roc_auc"
                ),
                4,
            ),
            "Global discrimination",
            tone="neutral",
        )

    with c3:

        metric_card(
            f"Recall @ {review_label}",
            _metric_percent(
                recall,
                2,
            ),
            (
                "Fraud cases captured"
                if recall is not None
                else "Not persisted at this capacity"
            ),
            tone=(
                "success"
                if recall is not None
                else "neutral"
            ),
        )

    with c4:

        metric_card(
            f"Lift @ {review_label}",
            _metric_multiplier(
                lift,
                2,
            ),
            (
                "Versus untargeted review"
                if lift is not None
                else "Not persisted at this capacity"
            ),
            tone=(
                "success"
                if lift is not None
                else "neutral"
            ),
        )

    st.write("")

    c1, c2, c3 = (
        st.columns(
            3
        )
    )

    with c1:

        metric_card(
            f"Precision @ {review_label}",
            _metric_percent(
                precision,
                2,
            ),
            (
                "Investigation yield"
                if precision is not None
                else "Not persisted at this capacity"
            ),
            tone=(
                "info"
                if precision is not None
                else "neutral"
            ),
        )

    with c2:

        metric_card(
            "Fraud Amount Captured",
            _metric_percent(
                fraud_amount_capture,
                2,
            ),
            (
                f"At {review_label} review capacity"
                if fraud_amount_capture is not None
                else "Not persisted at this capacity"
            ),
            tone=(
                "success"
                if fraud_amount_capture is not None
                else "neutral"
            ),
        )

    with c3:

        prevalence = (
            metrics.get(
                "test_fraud_prevalence"
            )
        )

        if prevalence is None:

            prevalence = (
                metadata.get(
                    "test_fraud_prevalence"
                )
            )

        if prevalence is None:

            metric_card(
                "Test Fraud Prevalence",
                "Not recorded",
                (
                    "Not persisted in frozen evaluation metadata"
                ),
                tone="neutral",
            )

        else:

            metric_card(
                "Test Fraud Prevalence",
                _metric_percent(
                    prevalence,
                    3,
                ),
                "Out-of-time population",
                tone="neutral",
            )

    st.write("")

    if any(
        value is None
        for value in (
            recall,
            lift,
            precision,
            fraud_amount_capture,
        )
    ):

        info_panel(
            "Capacity-Specific Metrics",
            (
                f"The active review policy is {review_label}. "
                "Metrics evaluated at another capacity are not "
                "substituted or relabelled. Missing values therefore "
                "remain explicitly unavailable."
            ),
            tone="warning",
        )

        st.write("")

    info_panel(
        "Operational Interpretation",
        (
            "The deployed model is used primarily as a ranking system. "
            "Under constrained investigation capacity, recall, precision, "
            "lift and fraud-amount capture are more operationally useful "
            "than relying on a standalone binary classification threshold."
        ),
        tone="info",
    )


# =============================================================================
# Evaluation context
# =============================================================================


def _render_evaluation_context(
    metadata: dict[str, Any],
) -> None:
    """
    Render frozen evaluation period and review policy metadata.
    """

    if not metadata:
        return

    training_end = (
        metadata.get(
            "training_period_end"
        )
    )

    test_period = (
        metadata.get(
            "test_period"
        )
    )

    review_policy = (
        metadata.get(
            "review_policy"
        )
    )

    if not isinstance(
        test_period,
        dict,
    ):
        test_period = {}

    with st.expander(
        "Evaluation context",
        expanded=False,
    ):

        key_value_row(
            "Training period end",
            (
                str(
                    training_end
                )
                if training_end
                else "Not recorded"
            ),
        )

        key_value_row(
            "Out-of-time test start",
            (
                str(
                    test_period.get(
                        "start"
                    )
                )
                if test_period.get(
                    "start"
                )
                else "Not recorded"
            ),
        )

        key_value_row(
            "Out-of-time test end",
            (
                str(
                    test_period.get(
                        "end"
                    )
                )
                if test_period.get(
                    "end"
                )
                else "Not recorded"
            ),
        )

        key_value_row(
            "Frozen review policy",
            format_review_policy(
                review_policy
            ),
        )


# =============================================================================
# Capacity analysis
# =============================================================================


def _capacity_candidates() -> list[
    tuple[str, ...]
]:
    """
    Candidate locations for a machine-readable frozen capacity curve.
    """

    return [
        (
            "evaluation",
            "capacity_curve.csv",
        ),
        (
            "metadata",
            "capacity_curve.csv",
        ),
        (
            "metadata",
            "final_evaluation",
            "capacity_curve.csv",
        ),
        (
            "explainability",
            "capacity_curve.csv",
        ),
    ]


def _load_capacity_data() -> tuple[
    pd.DataFrame,
    str | None,
]:
    """
    Load the first available machine-readable capacity artifact.
    """

    for parts in (
        _capacity_candidates()
    ):

        frame = (
            _load_artifact_csv(
                *parts
            )
        )

        if not frame.empty:

            return (
                frame,
                "/".join(
                    parts
                ),
            )

    return (
        pd.DataFrame(),
        None,
    )


def _normalize_capacity_data(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize supported capacity-curve schemas.
    """

    if frame.empty:
        return pd.DataFrame()

    data = (
        frame.copy()
    )

    rename_map = {
        "review_fraction":
            "capacity",

        "review_capacity":
            "capacity",

        "capacity_fraction":
            "capacity",

        "recall":
            "Recall",

        "recall_at_capacity":
            "Recall",

        "fraud_amount_capture":
            "Fraud Amount Capture",

        "fraud_amount_recall":
            "Fraud Amount Capture",

        "amount_capture":
            "Fraud Amount Capture",
    }

    data = (
        data.rename(
            columns=rename_map
        )
    )

    required = {
        "capacity",
        "Recall",
        "Fraud Amount Capture",
    }

    if not required.issubset(
        data.columns
    ):

        return pd.DataFrame()

    for column in required:

        data[
            column
        ] = pd.to_numeric(
            data[
                column
            ],
            errors="coerce",
        )

    data = (
        data
        .dropna(
            subset=list(
                required
            )
        )
    )

    if data.empty:
        return pd.DataFrame()

    valid_mask = (
        data[
            "capacity"
        ]
        .between(
            0,
            1,
        )
        &
        data[
            "Recall"
        ]
        .between(
            0,
            1,
        )
        &
        data[
            "Fraud Amount Capture"
        ]
        .between(
            0,
            1,
        )
    )

    data = (
        data
        .loc[
            valid_mask
        ]
        .sort_values(
            "capacity"
        )
        .drop_duplicates(
            subset=[
                "capacity"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return data


def _capacity_figure_path() -> Path | None:
    """
    Resolve the frozen capacity-analysis figure.

    A real frozen evaluation figure is preferred over generating any
    synthetic or interpolated curve when machine-readable data is absent.
    """

    candidates = [
        (
            "metadata",
            "final_evaluation",
            "05_capacity_curve.png",
        ),
        (
            "metadata",
            "model_experiments",
            "04_recall_by_review_capacity.png",
        ),
    ]

    for parts in candidates:

        path = (
            _existing_artifact_path(
                *parts
            )
        )

        if path is not None:
            return path

    return None


def _render_capacity_chart(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
) -> None:
    """
    Render real investigation-capacity behavior.

    Priority
    --------
    1. machine-readable capacity curve;
    2. frozen evaluation figure;
    3. explicit unavailable state.

    Synthetic fallback values are never generated.
    """

    review_fraction = (
        _review_fraction(
            metadata,
            runtime_model,
        )
    )

    review_label = (
        _review_fraction_label(
            review_fraction
        )
    )

    section_header(
        "Investigation Capacity",
        (
            "How fraud capture evolves as "
            "human-review capacity increases."
        ),
        eyebrow="OPERATIONAL POLICY",
    )

    raw_data, source = (
        _load_capacity_data()
    )

    data = (
        _normalize_capacity_data(
            raw_data
        )
    )

    if data.empty:

        figure_path = (
            _capacity_figure_path()
        )

        if figure_path is not None:

            st.image(
                str(
                    figure_path
                ),
                use_container_width=True,
            )

            st.caption(
                (
                    "Frozen out-of-time capacity analysis. "
                    "The original evaluation figure is displayed because "
                    "a machine-readable capacity curve was not persisted. "
                    "No synthetic values or interpolation are used."
                )
            )

            st.caption(
                (
                    "Artifact source: "
                    + _artifact_relative_label(
                        figure_path
                    )
                )
            )

            info_panel(
                "Current Review Policy",
                (
                    f"The live investigation policy reviews the top "
                    f"{review_label} highest-risk claims."
                ),
                tone="info",
            )

            return

        info_panel(
            "Capacity Analysis Unavailable",
            (
                "Neither a machine-readable capacity curve nor "
                "a frozen capacity-analysis figure is available. "
                "No synthetic fallback values are generated."
            ),
            tone="warning",
        )

        return

    chart_data = (
        data[
            [
                "capacity",
                "Recall",
                "Fraud Amount Capture",
            ]
        ]
        .melt(
            id_vars=[
                "capacity"
            ],
            var_name="Metric",
            value_name="Value",
        )
    )

    maximum_capacity = max(
        float(
            data[
                "capacity"
            ]
            .max()
        ),
        review_fraction,
    )

    lines = (
        alt.Chart(
            chart_data
        )
        .mark_line(
            point=True,
            strokeWidth=3,
        )
        .encode(
            x=alt.X(
                "capacity:Q",
                title="Investigation Capacity",
                axis=alt.Axis(
                    format=".1%",
                ),
                scale=alt.Scale(
                    domain=[
                        0,
                        maximum_capacity,
                    ],
                    nice=True,
                ),
            ),

            y=alt.Y(
                "Value:Q",
                title="Fraud Capture",
                axis=alt.Axis(
                    format=".0%",
                ),
                scale=alt.Scale(
                    domain=[
                        0,
                        1,
                    ]
                ),
            ),

            color=alt.Color(
                "Metric:N",
                title=None,
            ),

            tooltip=[
                alt.Tooltip(
                    "capacity:Q",
                    title="Capacity",
                    format=".1%",
                ),

                alt.Tooltip(
                    "Metric:N",
                    title="Metric",
                ),

                alt.Tooltip(
                    "Value:Q",
                    title="Value",
                    format=".2%",
                ),
            ],
        )
    )

    policy_data = (
        pd.DataFrame(
            {
                "capacity": [
                    review_fraction
                ],
            }
        )
    )

    policy_rule = (
        alt.Chart(
            policy_data
        )
        .mark_rule(
            strokeDash=[
                6,
                5,
            ],
            strokeWidth=2,
        )
        .encode(
            x=alt.X(
                "capacity:Q"
            )
        )
    )

    chart = (
        (
            lines
            + policy_rule
        )
        .properties(
            height=390
        )
        .interactive()
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    st.caption(
        (
            "Dashed marker: current operational review "
            f"capacity ({review_label})."
        )
    )

    if source:

        st.caption(
            (
                "Artifact source: "
                f"{source}"
            )
        )


# =============================================================================
# Live system
# =============================================================================


def _render_live_system(
    health: dict[str, Any],
    model: dict[str, Any],
    runtime_error: str | None,
) -> None:
    """
    Render deployed inference-service readiness.
    """

    section_header(
        "Live System",
        (
            "Current readiness of the deployed "
            "fraud inference service."
        ),
        eyebrow="RUNTIME",
    )

    if runtime_error:

        info_panel(
            "Inference Service Offline",
            (
                "The analytical dashboard remains available, "
                "but live scoring cannot currently reach the "
                "inference API."
            ),
            tone="danger",
        )

        with st.expander(
            "Technical details",
            expanded=False,
        ):

            st.code(
                runtime_error,
                language=None,
            )

        return

    status = (
        str(
            health.get(
                "status",
                "unknown",
            )
        )
        .strip()
    )

    model_loaded = (
        health.get(
            "model_loaded"
        )
    )

    healthy = (
        status.lower()
        in {
            "ok",
            "healthy",
            "ready",
        }
    )

    if model_loaded is False:
        healthy = False

    explainability = (
        model.get(
            "explainability"
        )
    )

    if not isinstance(
        explainability,
        dict,
    ):
        explainability = {}

    explanation_available = (
        explainability.get(
            "available"
        )
        is True
    )

    with st.container(
        border=True
    ):

        status_badge(
            healthy,
            label=(
                "INFERENCE READY"
                if healthy
                else "SERVICE REACHABLE"
            ),
        )

        st.caption(
            (
                "Runtime state reported directly "
                "by the inference API."
            )
        )

        st.write("")

        c1, c2 = (
            st.columns(
                2
            )
        )

        with c1:

            metric_card(
                "Model",
                str(
                    model.get(
                        "model_name",
                        "—",
                    )
                ),
                "Deployed estimator",
                tone="info",
            )

        with c2:

            metric_card(
                "Version",
                str(
                    model.get(
                        "model_version",
                        "—",
                    )
                ),
                "Runtime contract",
                tone="neutral",
            )

        st.write("")

        c1, c2 = (
            st.columns(
                2
            )
        )

        with c1:

            mini_metric(
                "API Health",
                status.upper(),
                helper="Inference endpoint",
                tone=(
                    "success"
                    if healthy
                    else "warning"
                ),
            )

        with c2:

            loaded_label = (
                "UNKNOWN"
                if model_loaded is None
                else (
                    "YES"
                    if model_loaded is True
                    else "NO"
                )
            )

            mini_metric(
                "Model Loaded",
                loaded_label,
                helper="Runtime readiness",
                tone=(
                    "success"
                    if model_loaded is True
                    else "warning"
                ),
            )

        st.write("")

        st.markdown(
            "##### Inference Contract"
        )

        key_value_row(
            "Prediction target",
            str(
                model.get(
                    "target",
                    "—",
                )
            ),
            monospace=True,
        )

        key_value_row(
            "Source features",
            str(
                model.get(
                    "feature_count",
                    "—",
                )
            ),
        )

        key_value_row(
            "Transformed features",
            str(
                model.get(
                    "transformed_feature_count",
                    "—",
                )
            ),
        )

        key_value_row(
            "Probability interface",
            str(
                model.get(
                    "probability_method",
                    "—",
                )
            ),
            monospace=True,
        )

        key_value_row(
            "Review policy",
            format_review_policy(
                model.get(
                    "review_policy"
                )
            ),
        )

        key_value_row(
            "Explainability",
            (
                str(
                    explainability.get(
                        "method",
                        "TreeSHAP",
                    )
                )
                if explanation_available
                else "Unavailable"
            ),
        )

        if explanation_available:

            key_value_row(
                "Explanation space",
                str(
                    explainability.get(
                        "output_space",
                        "—",
                    )
                ),
                monospace=True,
            )

        st.caption(
            (
                "Claim Analysis and Portfolio Scoring consume "
                "this frozen live inference contract."
            )
        )


# =============================================================================
# Contract consistency
# =============================================================================


def _normalize_contract_value(
    value: Any,
    value_type: str,
) -> Any:
    """
    Normalize contract values before comparison.
    """

    if value is None:
        return None

    if value_type == "integer":

        return (
            _safe_int(
                value
            )
        )

    return (
        str(
            value
        )
        .strip()
    )


def _render_contract_consistency(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
    runtime_error: str | None,
) -> None:
    """
    Compare frozen analytical metadata against deployed runtime contract.

    Status semantics
    ----------------
    MATCH
        Both values are present and equivalent.

    MISMATCH
        Both values are present but differ.

    NOT COMPARED
        One side does not expose the field.

    Missing metadata therefore never becomes a false deployment mismatch.
    """

    st.write("")
    st.write("")

    section_header(
        "Deployment Consistency",
        (
            "Consistency between frozen analytical metadata "
            "and the model currently served by the API."
        ),
        eyebrow="MODEL GOVERNANCE",
    )

    if runtime_error:

        info_panel(
            "Runtime Verification Unavailable",
            (
                "The API could not be queried, so the deployed model "
                "cannot currently be compared with frozen analytical metadata."
            ),
            tone="warning",
        )

        return

    if not runtime_model:

        info_panel(
            "Runtime Contract Unavailable",
            (
                "The inference API is reachable but did not expose "
                "a usable model contract."
            ),
            tone="warning",
        )

        return

    if not metadata:

        info_panel(
            "Metadata Comparison Unavailable",
            (
                "The runtime model is reachable, but frozen "
                "analytical metadata is unavailable."
            ),
            tone="warning",
        )

        return

    # Deliberately exclude probability_method:
    #
    # frozen metadata currently records "raw", while the runtime exposes
    # "predict_proba". These describe different interfaces and should not be
    # interpreted as deployment inconsistency without a harmonized semantic
    # contract.
    fields = [
        (
            "model_name",
            "Model",
            "text",
        ),
        (
            "model_version",
            "Version",
            "text",
        ),
        (
            "target",
            "Prediction Target",
            "text",
        ),
        (
            "feature_count",
            "Source Feature Count",
            "integer",
        ),
        (
            "transformed_feature_count",
            "Transformed Feature Count",
            "integer",
        ),
    ]

    records: list[
        dict[str, Any]
    ] = []

    match_count = 0
    mismatch_count = 0
    unavailable_count = 0

    for (
        key,
        label,
        value_type,
    ) in fields:

        artifact_value = (
            metadata.get(
                key
            )
        )

        runtime_value = (
            runtime_model.get(
                key
            )
        )

        normalized_artifact = (
            _normalize_contract_value(
                artifact_value,
                value_type,
            )
        )

        normalized_runtime = (
            _normalize_contract_value(
                runtime_value,
                value_type,
            )
        )

        if (
            normalized_artifact is None
            or normalized_runtime is None
        ):

            status = (
                "NOT COMPARED"
            )

            unavailable_count += 1

        elif (
            normalized_artifact
            == normalized_runtime
        ):

            status = "MATCH"

            match_count += 1

        else:

            status = "MISMATCH"

            mismatch_count += 1

        records.append(
            {
                "Field":
                    label,

                "Frozen Artifact":
                    (
                        "Not recorded"
                        if artifact_value is None
                        else str(
                            artifact_value
                        )
                    ),

                "Live Runtime":
                    (
                        "Not exposed"
                        if runtime_value is None
                        else str(
                            runtime_value
                        )
                    ),

                "Status":
                    status,
            }
        )

    comparable_count = (
        match_count
        + mismatch_count
    )

    if (
        comparable_count > 0
        and mismatch_count == 0
    ):

        message = (
            f"All {comparable_count} comparable model-contract "
            "field(s) match the deployed runtime."
        )

        if unavailable_count:

            message += (
                f" {unavailable_count} additional field(s) are "
                "not exposed by both contracts and are therefore "
                "reported as not compared."
            )

        info_panel(
            "Deployment Contract Consistent",
            message,
            tone="success",
        )

    elif mismatch_count > 0:

        info_panel(
            "Deployment Contract Mismatch",
            (
                f"{mismatch_count} comparable model-contract "
                "field(s) differ between the frozen artifact "
                "and the deployed runtime."
            ),
            tone="danger",
        )

    else:

        info_panel(
            "No Comparable Contract Fields",
            (
                "The available contracts do not expose enough shared "
                "fields for deployment verification."
            ),
            tone="warning",
        )

    with st.expander(
        "Contract comparison",
        expanded=False,
    ):

        st.dataframe(
            pd.DataFrame(
                records
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Field":
                    st.column_config.TextColumn(
                        "Field",
                        width="medium",
                    ),

                "Frozen Artifact":
                    st.column_config.TextColumn(
                        "Frozen Artifact",
                        width="large",
                    ),

                "Live Runtime":
                    st.column_config.TextColumn(
                        "Live Runtime",
                        width="large",
                    ),

                "Status":
                    st.column_config.TextColumn(
                        "Status",
                        width="small",
                    ),
            },
        )

    st.caption(
        (
            "Only fields represented by both contracts can produce "
            "a deployment mismatch. Missing fields are explicitly "
            "classified as not compared."
        )
    )


# =============================================================================
# Risk drivers
# =============================================================================


def _render_risk_drivers() -> None:
    """
    Render global business-level SHAP importance.
    """

    st.write("")
    st.write("")

    section_header(
        "Main Risk Drivers",
        (
            "Highest-impact business variables identified "
            "by global SHAP analysis."
        ),
        eyebrow="GLOBAL EXPLAINABILITY",
    )

    importance = (
        _load_artifact_csv(
            "explainability",
            "business_feature_importance.csv",
        )
    )

    if importance.empty:

        info_panel(
            "SHAP Importance Unavailable",
            (
                "Global business-level SHAP importance is not "
                "available in the frontend runtime."
            ),
            tone="warning",
        )

        return

    required = {
        "business_feature",
        "mean_abs_shap",
    }

    if not required.issubset(
        importance.columns
    ):

        info_panel(
            "Invalid SHAP Artifact",
            (
                "The SHAP importance artifact exists, but its schema "
                "does not contain the required fields."
            ),
            tone="warning",
        )

        return

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

    if (
        "signed_mean_shap"
        in importance.columns
    ):

        importance[
            "signed_mean_shap"
        ] = pd.to_numeric(
            importance[
                "signed_mean_shap"
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
        .loc[
            lambda current:
                current[
                    "mean_abs_shap"
                ]
                >= 0
        ]
    )

    if importance.empty:

        info_panel(
            "No Usable SHAP Values",
            (
                "The SHAP artifact was loaded but contains "
                "no valid numerical importance values."
            ),
            tone="warning",
        )

        return

    top = (
        importance
        .nlargest(
            MAX_RISK_DRIVERS,
            "mean_abs_shap",
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    top[
        "Rank"
    ] = (
        np.arange(
            1,
            len(
                top
            )
            + 1,
        )
    )

    top[
        "Feature"
    ] = (
        top[
            "business_feature"
        ]
        .apply(
            _pretty_feature
        )
    )

    left, right = (
        st.columns(
            [
                1.4,
                1,
            ],
            gap="large",
        )
    )

    with left:

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
                    title="Mean absolute SHAP impact",
                ),

                y=alt.Y(
                    "Feature:N",
                    sort="-x",
                    title=None,
                ),

                tooltip=[
                    alt.Tooltip(
                        "Rank:Q",
                        title="Rank",
                    ),

                    alt.Tooltip(
                        "Feature:N",
                        title="Feature",
                    ),

                    alt.Tooltip(
                        "mean_abs_shap:Q",
                        title="Mean |SHAP|",
                        format=".5f",
                    ),
                ],
            )
            .properties(
                height=350
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

    with right:

        display_columns = [
            "Rank",
            "Feature",
            "mean_abs_shap",
        ]

        if (
            "signed_mean_shap"
            in top.columns
        ):

            display_columns.append(
                "signed_mean_shap"
            )

        st.dataframe(
            top[
                display_columns
            ],
            use_container_width=True,
            hide_index=True,
            height=350,
            column_config={
                "Rank":
                    st.column_config.NumberColumn(
                        "Rank",
                        format="%d",
                        width="small",
                    ),

                "Feature":
                    st.column_config.TextColumn(
                        "Risk Driver",
                        width="large",
                    ),

                "mean_abs_shap":
                    st.column_config.NumberColumn(
                        "Mean |SHAP|",
                        format="%.5f",
                    ),

                "signed_mean_shap":
                    st.column_config.NumberColumn(
                        "Mean Direction",
                        format="%.5f",
                    ),
            },
        )

    st.write("")

    info_panel(
        "SHAP Interpretation",
        (
            "Mean absolute SHAP measures average predictive influence "
            "on model output. It does not imply causality. The direction "
            "of an individual claim's contribution can differ from the "
            "global average. Claim Analysis provides local TreeSHAP "
            "attribution for a specific claim."
        ),
        tone="info",
    )


# =============================================================================
# Risk intelligence
# =============================================================================


def _render_model_observations() -> None:
    """
    Render high-level model behavior from frozen analytical artifacts.
    """

    st.write("")
    st.write("")

    section_header(
        "Risk Intelligence",
        (
            "High-level observations derived from current "
            "explainability and error-analysis artifacts."
        ),
        eyebrow="MODEL BEHAVIOR",
    )

    mechanism = (
        _load_artifact_csv(
            "explainability",
            "mechanism_score_summary.csv",
        )
    )

    difficulty = (
        _load_artifact_csv(
            "explainability",
            "difficulty_score_summary.csv",
        )
    )

    missed = (
        _load_artifact_csv(
            "explainability",
            "false_negative_by_mechanism.csv",
        )
    )

    valid_mechanism = (
        pd.DataFrame()
    )

    if (
        not mechanism.empty
        and {
            "fraud_mechanism",
            "mean_score",
        }.issubset(
            mechanism.columns
        )
    ):

        valid_mechanism = (
            mechanism.copy()
        )

        valid_mechanism[
            "mean_score"
        ] = pd.to_numeric(
            valid_mechanism[
                "mean_score"
            ],
            errors="coerce",
        )

        valid_mechanism = (
            valid_mechanism
            .dropna(
                subset=[
                    "mean_score"
                ]
            )
        )

    strongest = None
    weakest = None

    if not valid_mechanism.empty:

        strongest = (
            valid_mechanism
            .nlargest(
                1,
                "mean_score",
            )
            .iloc[0]
        )

        weakest = (
            valid_mechanism
            .nsmallest(
                1,
                "mean_score",
            )
            .iloc[0]
        )

    hard_row = None

    if (
        not difficulty.empty
        and {
            "fraud_difficulty",
            "mean_score",
        }.issubset(
            difficulty.columns
        )
    ):

        difficulty = (
            difficulty.copy()
        )

        difficulty[
            "mean_score"
        ] = pd.to_numeric(
            difficulty[
                "mean_score"
            ],
            errors="coerce",
        )

        hard = (
            difficulty.loc[
                difficulty[
                    "fraud_difficulty"
                ]
                .astype(
                    str
                )
                .str.lower()
                .eq(
                    "hard"
                )
            ]
        )

        if not hard.empty:

            hard_row = (
                hard.iloc[0]
            )

    c1, c2, c3 = (
        st.columns(
            3
        )
    )

    with c1:

        if strongest is not None:

            metric_card(
                "Strongest Pattern",
                _pretty_feature(
                    strongest[
                        "fraud_mechanism"
                    ]
                ),
                (
                    "Mean risk "
                    + _metric_percent(
                        strongest[
                            "mean_score"
                        ],
                        1,
                    )
                ),
                tone="success",
            )

        else:

            metric_card(
                "Strongest Pattern",
                "—",
                "Artifact unavailable",
                tone="neutral",
            )

    with c2:

        if weakest is not None:

            mechanism_name = (
                weakest[
                    "fraud_mechanism"
                ]
            )

            caption = (
                "Mean risk "
                + _metric_percent(
                    weakest[
                        "mean_score"
                    ],
                    1,
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

                matching = (
                    missed.loc[
                        missed[
                            "fraud_mechanism"
                        ]
                        .astype(
                            str
                        )
                        .eq(
                            str(
                                mechanism_name
                            )
                        )
                    ]
                )

                if not matching.empty:

                    missed_count = (
                        _safe_int(
                            matching
                            .iloc[0]
                            .get(
                                "missed_fraud_claims"
                            )
                        )
                    )

                    if (
                        missed_count
                        is not None
                    ):

                        caption += (
                            f" • {missed_count:,} missed"
                        )

            metric_card(
                "Main Weakness",
                _pretty_feature(
                    mechanism_name
                ),
                caption,
                tone="warning",
            )

        else:

            metric_card(
                "Main Weakness",
                "—",
                "Artifact unavailable",
                tone="neutral",
            )

    with c3:

        if hard_row is not None:

            fraud_claims = (
                _safe_int(
                    hard_row.get(
                        "fraud_claims"
                    )
                )
            )

            metric_card(
                "Hard Fraud Mean Risk",
                _metric_percent(
                    hard_row.get(
                        "mean_score"
                    ),
                    1,
                ),
                (
                    f"{fraud_claims:,} cases evaluated"
                    if fraud_claims is not None
                    else "Hard-fraud evaluation subset"
                ),
                tone="warning",
            )

        else:

            metric_card(
                "Hard Fraud Mean Risk",
                "—",
                "Artifact unavailable",
                tone="neutral",
            )


# =============================================================================
# Artifact coverage
# =============================================================================


def _render_artifact_coverage() -> None:
    """
    Render traceability of analytical evidence used by the overview.
    """

    st.write("")
    st.write("")

    section_header(
        "Evidence Coverage",
        (
            "Availability of analytical artifacts supporting "
            "the executive conclusions shown above."
        ),
        eyebrow="TRACEABILITY",
    )

    artifact_specs = [
        (
            "Model metadata",
            (
                "metadata",
                "health_fraud_model_metadata.json",
            ),
        ),
        (
            "Capacity analysis",
            (
                "metadata",
                "final_evaluation",
                "05_capacity_curve.png",
            ),
        ),
        (
            "Global SHAP importance",
            (
                "explainability",
                "business_feature_importance.csv",
            ),
        ),
        (
            "Mechanism analysis",
            (
                "explainability",
                "mechanism_score_summary.csv",
            ),
        ),
        (
            "Difficulty analysis",
            (
                "explainability",
                "difficulty_score_summary.csv",
            ),
        ),
        (
            "False-negative analysis",
            (
                "explainability",
                "false_negative_by_mechanism.csv",
            ),
        ),
    ]

    records: list[
        dict[str, str]
    ] = []

    for (
        label,
        parts,
    ) in artifact_specs:

        path = (
            _existing_artifact_path(
                *parts
            )
        )

        available = (
            path is not None
        )

        records.append(
            {
                "Artifact":
                    label,

                "Status":
                    (
                        "AVAILABLE"
                        if available
                        else "MISSING"
                    ),

                "Source":
                    (
                        "/".join(
                            parts
                        )
                        if available
                        else "—"
                    ),
            }
        )

    available_count = (
        sum(
            record[
                "Status"
            ]
            == "AVAILABLE"

            for record
            in records
        )
    )

    total_count = (
        len(
            records
        )
    )

    coverage = (
        available_count
        / total_count
        if total_count
        else 0.0
    )

    c1, c2, c3 = (
        st.columns(
            3
        )
    )

    with c1:

        metric_card(
            "Analytical Artifacts",
            (
                f"{available_count}/"
                f"{total_count}"
            ),
            "Core evidence available",
            tone=(
                "success"
                if available_count
                == total_count
                else "warning"
            ),
        )

    with c2:

        metric_card(
            "Evidence Coverage",
            f"{coverage:.0%}",
            "Analytical traceability",
            tone=(
                "success"
                if coverage == 1
                else "warning"
            ),
        )

    with c3:

        metric_card(
            "Artifact Runtime",
            (
                "READY"
                if ARTIFACTS_DIR
                is not None
                else "UNAVAILABLE"
            ),
            (
                str(
                    ARTIFACTS_DIR
                )
                if ARTIFACTS_DIR
                is not None
                else "No artifact root detected"
            ),
            tone=(
                "success"
                if ARTIFACTS_DIR
                is not None
                else "danger"
            ),
        )

    with st.expander(
        "Artifact inventory",
        expanded=False,
    ):

        st.dataframe(
            pd.DataFrame(
                records
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Artifact":
                    st.column_config.TextColumn(
                        "Artifact",
                        width="medium",
                    ),

                "Status":
                    st.column_config.TextColumn(
                        "Status",
                        width="small",
                    ),

                "Source":
                    st.column_config.TextColumn(
                        "Source",
                        width="large",
                    ),
            },
        )


# =============================================================================
# Governance
# =============================================================================


def _render_governance() -> None:
    """
    Render human-in-the-loop operating boundary.
    """

    st.write("")
    st.write("")

    section_header(
        "Decision Governance",
        (
            "Operational boundary between model-assisted "
            "prioritization and human fraud investigation."
        ),
        eyebrow="HUMAN OVERSIGHT",
    )

    human_review_notice()


# =============================================================================
# Main page
# =============================================================================


def render(
    client: Any,
) -> None:
    """
    Render the executive fraud-intelligence overview.
    """

    section_header(
        "Executive Overview",
        (
            "Fraud-risk performance, investigation capacity, "
            "model intelligence, explainability and live "
            "inference readiness."
        ),
    )

    metadata = (
        _load_metadata()
    )

    (
        health,
        runtime_model,
        runtime_error,
    ) = (
        _read_runtime_system(
            client
        )
    )

    # =========================================================================
    # Artifact status
    # =========================================================================

    _render_metadata_notice(
        metadata
    )

    # =========================================================================
    # Executive performance
    # =========================================================================

    _render_executive_kpis(
        metadata,
        runtime_model,
    )

    _render_evaluation_context(
        metadata
    )

    st.write("")
    st.write("")

    # =========================================================================
    # Capacity + live runtime
    # =========================================================================

    left, right = (
        st.columns(
            [
                1.55,
                1,
            ],
            gap="large",
        )
    )

    with left:

        _render_capacity_chart(
            metadata,
            runtime_model,
        )

    with right:

        _render_live_system(
            health,
            runtime_model,
            runtime_error,
        )

    # =========================================================================
    # Deployment governance
    # =========================================================================

    _render_contract_consistency(
        metadata,
        runtime_model,
        runtime_error,
    )

    # =========================================================================
    # Global explainability
    # =========================================================================

    _render_risk_drivers()

    # =========================================================================
    # Model behavior
    # =========================================================================

    _render_model_observations()

    # =========================================================================
    # Analytical traceability
    # =========================================================================

    _render_artifact_coverage()

    # =========================================================================
    # Human governance
    # =========================================================================

    _render_governance()
    