from __future__ import annotations

import json
import os

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

from utils.formatting import (
    format_review_policy,
)


# =============================================================================
# Configuration
# =============================================================================


DEFAULT_REVIEW_FRACTION = 0.03


# =============================================================================
# Project / artifact resolution
# =============================================================================


MODULE_PATH = Path(__file__).resolve()

PROJECT_ROOT = MODULE_PATH.parents[2]


def _candidate_artifact_roots() -> list[Path]:
    """
    Resolve artifacts consistently across:
    - local development
    - GitHub Codespaces
    - Docker frontend container
    - explicit ARTIFACTS_ROOT configuration
    """

    candidates: list[Path] = []

    configured = os.getenv(
        "ARTIFACTS_ROOT"
    )

    if configured:
        candidates.append(
            Path(configured)
            .expanduser()
            .resolve()
        )

    candidates.extend(
        [
            PROJECT_ROOT / "artifacts",
            Path("/app/artifacts"),
            Path.cwd() / "artifacts",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()

    for path in candidates:

        key = str(path)

        if key in seen:
            continue

        seen.add(key)
        unique.append(path)

    return unique


def _find_artifact_root() -> Path | None:

    for path in _candidate_artifact_roots():

        if (
            path.exists()
            and path.is_dir()
        ):
            return path

    return None


ARTIFACTS_DIR = _find_artifact_root()


def _artifact_path(
    *parts: str,
) -> Path | None:

    if ARTIFACTS_DIR is None:
        return None

    return ARTIFACTS_DIR.joinpath(
        *parts
    )


# =============================================================================
# Cached loading
# =============================================================================


@st.cache_data(
    show_spinner=False,
)
def _read_json(
    path_string: str,
) -> dict[str, Any]:

    path = Path(path_string)

    if not path.exists():
        return {}

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(file)

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

    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:

        frame = pd.read_csv(path)

    except Exception:
        return pd.DataFrame()

    return frame


def _load_metadata() -> dict[str, Any]:

    path = _artifact_path(
        "metadata",
        "health_fraud_model_metadata.json",
    )

    if path is None:
        return {}

    return _read_json(
        str(path)
    )


def _load_artifact_csv(
    *parts: str,
) -> pd.DataFrame:

    path = _artifact_path(
        *parts
    )

    if path is None:
        return pd.DataFrame()

    return _read_csv(
        str(path)
    )


# =============================================================================
# Generic helpers
# =============================================================================


def _safe_float(
    value: Any,
) -> float | None:

    try:

        result = float(value)

        if np.isfinite(result):
            return result

    except (
        TypeError,
        ValueError,
    ):
        pass

    return None


def _safe_int(
    value: Any,
) -> int | None:

    result = _safe_float(value)

    if result is None:
        return None

    return int(
        round(result)
    )


def _metric_number(
    value: Any,
    digits: int = 4,
) -> str:

    result = _safe_float(value)

    if result is None:
        return "—"

    return f"{result:.{digits}f}"


def _metric_percent(
    value: Any,
    digits: int = 2,
) -> str:

    result = _safe_float(value)

    if result is None:
        return "—"

    return f"{result:.{digits}%}"


def _metric_multiplier(
    value: Any,
    digits: int = 2,
) -> str:

    result = _safe_float(value)

    if result is None:
        return "—"

    return f"{result:.{digits}f}×"


def _pretty_feature(
    value: Any,
) -> str:

    if value is None:
        return "—"

    return (
        str(value)
        .replace("_", " ")
        .strip()
        .title()
    )


def _extract_metrics(
    metadata: dict[str, Any],
) -> dict[str, Any]:

    metrics = metadata.get(
        "final_test_metrics"
    )

    if isinstance(
        metrics,
        dict,
    ):
        return metrics

    return {}


def _review_fraction(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any] | None = None,
) -> float:

    # Runtime contract has priority.
    if isinstance(
        runtime_model,
        dict,
    ):

        policy = runtime_model.get(
            "review_policy"
        )

        if isinstance(
            policy,
            dict,
        ):

            value = _safe_float(
                policy.get("fraction")
            )

            if (
                value is not None
                and 0 < value <= 1
            ):
                return value

    policy = metadata.get(
        "review_policy",
        {},
    )

    if isinstance(
        policy,
        dict,
    ):

        value = _safe_float(
            policy.get("fraction")
        )

        if (
            value is not None
            and 0 < value <= 1
        ):
            return value

    return DEFAULT_REVIEW_FRACTION


# =============================================================================
# Runtime API
# =============================================================================


def _read_runtime_system(
    client,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str | None,
]:
    """
    Read health + deployed model contract once.

    This prevents the Overview page from making several
    redundant API calls during one render.
    """

    try:

        health = client.health()
        model = client.model_info()

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

    except Exception as exc:

        return (
            {},
            {},
            str(exc),
        )


# =============================================================================
# Metadata / artifact status
# =============================================================================


def _render_metadata_notice(
    metadata: dict[str, Any],
) -> None:

    if metadata:
        return

    if ARTIFACTS_DIR is None:

        info_panel(
            "Evaluation Artifacts Unavailable",
            (
                "Performance and explainability artifacts are not "
                "mounted in the frontend runtime. Live inference "
                "may still remain available through the API."
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

    metrics = _extract_metrics(
        metadata
    )

    review_fraction = _review_fraction(
        metadata,
        runtime_model,
    )

    review_label = (
        f"{review_fraction:.0%}"
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
                "The final out-of-time metrics are not available "
                "in the current model metadata."
            ),
            tone="warning",
        )

        return

    c1, c2, c3, c4 = st.columns(4)

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
            f"Lift @ {review_label}",
            _metric_multiplier(
                metrics.get(
                    "lift_at_3pct"
                ),
                2,
            ),
            "Versus untargeted review",
            tone="success",
        )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(
            f"Precision @ {review_label}",
            _metric_percent(
                metrics.get(
                    "precision_at_3pct"
                ),
                2,
            ),
            "Investigation yield",
            tone="info",
        )

    with c2:

        metric_card(
            "Fraud Amount Captured",
            _metric_percent(
                metrics.get(
                    "fraud_amount_capture_at_3pct"
                ),
                2,
            ),
            (
                f"At {review_label} review capacity"
            ),
            tone="success",
        )

    with c3:

        prevalence = metrics.get(
            "test_fraud_prevalence"
        )

        if prevalence is None:

            prevalence = metadata.get(
                "test_fraud_prevalence"
            )

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

    info_panel(
        "Operational Interpretation",
        (
            "The model is used primarily as a ranking system. "
            "At constrained investigation capacity, recall, "
            "precision, lift and fraud-amount capture are more "
            "operationally informative than a standalone "
            "classification threshold."
        ),
        tone="info",
    )


# =============================================================================
# Capacity analysis
# =============================================================================


def _capacity_candidates() -> list[
    tuple[str, ...]
]:

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

    for parts in _capacity_candidates():

        frame = _load_artifact_csv(
            *parts
        )

        if not frame.empty:

            return (
                frame,
                "/".join(parts),
            )

    return (
        pd.DataFrame(),
        None,
    )


def _normalize_capacity_data(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    if frame.empty:
        return pd.DataFrame()

    data = frame.copy()

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

    data = data.rename(
        columns=rename_map
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

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = (
        data
        .dropna(
            subset=list(required)
        )
        .loc[
            lambda frame:
                (
                    frame["capacity"]
                    .between(0, 1)
                )
                &
                (
                    frame["Recall"]
                    .between(0, 1)
                )
                &
                (
                    frame[
                        "Fraud Amount Capture"
                    ]
                    .between(0, 1)
                )
        ]
        .sort_values(
            "capacity"
        )
        .drop_duplicates(
            subset=["capacity"]
        )
        .reset_index(
            drop=True
        )
    )

    return data


def _render_capacity_chart(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
) -> None:

    review_fraction = _review_fraction(
        metadata,
        runtime_model,
    )

    section_header(
        "Investigation Capacity",
        (
            "How fraud capture evolves as "
            "human-review capacity increases."
        ),
        eyebrow="OPERATIONAL POLICY",
    )

    raw_data, source = _load_capacity_data()

    data = _normalize_capacity_data(
        raw_data
    )

    if data.empty:

        info_panel(
            "Capacity Curve Unavailable",
            (
                "No valid capacity-curve artifact is available. "
                "No synthetic fallback values are displayed."
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
                        max(
                            float(
                                data[
                                    "capacity"
                                ]
                                .max()
                            ),
                            review_fraction,
                        ),
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

    policy_data = pd.DataFrame(
        {
            "capacity": [
                review_fraction
            ],
        }
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
            f"capacity ({review_fraction:.0%})."
        )
    )

    if source:

        st.caption(
            f"Artifact source: {source}"
        )


# =============================================================================
# Live system
# =============================================================================


def _render_live_system(
    health: dict[str, Any],
    model: dict[str, Any],
    runtime_error: str | None,
) -> None:

    section_header(
        "Live System",
        (
            "Current readiness of the deployed "
            "fraud inference service."
        ),
        eyebrow="RUNTIME",
    )

    if runtime_error:

        with st.container(
            border=True
        ):

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

    status = str(
        health.get(
            "status",
            "unknown",
        )
    )

    model_loaded = health.get(
        "model_loaded"
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

    with st.container(
        border=True
    ):

        if healthy:

            st.success(
                "● INFERENCE SERVICE ONLINE"
            )

        else:

            st.warning(
                "● INFERENCE SERVICE REACHABLE"
            )

        st.caption(
            (
                "Runtime state reported directly "
                "by the inference API."
            )
        )

        st.write("")

        c1, c2 = st.columns(2)

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

        c1, c2 = st.columns(2)

        with c1:

            metric_card(
                "API Health",
                status.upper(),
                "Inference endpoint",
                tone=(
                    "success"
                    if healthy
                    else "warning"
                ),
            )

        with c2:

            if model_loaded is None:

                loaded_label = "UNKNOWN"

            else:

                loaded_label = (
                    "YES"
                    if bool(model_loaded)
                    else "NO"
                )

            metric_card(
                "Model Loaded",
                loaded_label,
                "Runtime readiness",
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

        st.write(
            (
                "**Prediction target:** "
                f"`{model.get('target', '—')}`"
            )
        )

        st.write(
            (
                "**Business features:** "
                f"{model.get('feature_count', '—')}"
            )
        )

        st.write(
            (
                "**Review policy:** "
                + format_review_policy(
                    model.get(
                        "review_policy"
                    )
                )
            )
        )

        st.caption(
            (
                "Claim Analysis and Portfolio Scoring "
                "use this live frozen inference service."
            )
        )


# =============================================================================
# Contract consistency
# =============================================================================


def _render_contract_consistency(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
    runtime_error: str | None,
) -> None:

    st.write("")
    st.write("")

    section_header(
        "Deployment Consistency",
        (
            "Consistency between frozen analytical "
            "metadata and the model currently served by the API."
        ),
        eyebrow="MODEL GOVERNANCE",
    )

    if runtime_error:

        info_panel(
            "Runtime Verification Unavailable",
            (
                "The API could not be queried, so the deployed "
                "model cannot currently be compared with the "
                "local analytical metadata."
            ),
            tone="warning",
        )

        return

    if not metadata:

        info_panel(
            "Metadata Comparison Unavailable",
            (
                "The runtime model is reachable, but local frozen "
                "metadata is unavailable."
            ),
            tone="warning",
        )

        return

    fields = [
        (
            "model_name",
            "Model",
        ),
        (
            "model_version",
            "Version",
        ),
        (
            "target",
            "Target",
        ),
        (
            "feature_count",
            "Feature Count",
        ),
    ]

    records: list[
        dict[str, Any]
    ] = []

    mismatch_count = 0
    comparison_count = 0

    for key, label in fields:

        local_value = metadata.get(
            key
        )

        runtime_value = runtime_model.get(
            key
        )

        if (
            local_value is None
            and runtime_value is None
        ):
            continue

        if (
            local_value is None
            or runtime_value is None
        ):

            match = False

        else:

            match = (
                str(local_value)
                == str(runtime_value)
            )

            comparison_count += 1

        if not match:
            mismatch_count += 1

        records.append(
            {
                "Field":
                    label,

                "Artifact":
                    (
                        "—"
                        if local_value is None
                        else str(local_value)
                    ),

                "Runtime":
                    (
                        "—"
                        if runtime_value is None
                        else str(runtime_value)
                    ),

                "Status":
                    (
                        "MATCH"
                        if match
                        else "MISMATCH"
                    ),
            }
        )

    if (
        records
        and mismatch_count == 0
        and comparison_count > 0
    ):

        info_panel(
            "Deployment Contract Consistent",
            (
                "The core local model metadata matches the "
                "model currently exposed by the inference API."
            ),
            tone="success",
        )

    elif records:

        info_panel(
            "Deployment Contract Mismatch",
            (
                f"{mismatch_count} model-contract field(s) differ "
                "between local artifacts and the live API."
            ),
            tone="warning",
        )

    else:

        info_panel(
            "No Comparable Contract Fields",
            (
                "The available metadata does not expose enough "
                "shared fields for a runtime consistency check."
            ),
            tone="warning",
        )

    if records:

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

                    "Artifact":
                        st.column_config.TextColumn(
                            "Artifact",
                            width="large",
                        ),

                    "Runtime":
                        st.column_config.TextColumn(
                            "Runtime",
                            width="large",
                        ),

                    "Status":
                        st.column_config.TextColumn(
                            "Status",
                            width="small",
                        ),
                },
            )


# =============================================================================
# Risk drivers
# =============================================================================


def _render_risk_drivers() -> None:

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

    importance = _load_artifact_csv(
        "explainability",
        "business_feature_importance.csv",
    )

    if importance.empty:

        info_panel(
            "SHAP Importance Unavailable",
            (
                "Global business-level SHAP importance is "
                "not available in the frontend runtime."
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
                "The SHAP importance artifact exists, but "
                "its schema does not contain the required fields."
            ),
            tone="warning",
        )

        return

    importance = importance.copy()

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

    importance = importance.dropna(
        subset=[
            "mean_abs_shap"
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
            8,
            "mean_abs_shap",
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    top[
        "Rank"
    ] = np.arange(
        1,
        len(top) + 1,
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

    left, right = st.columns(
        [
            1.4,
            1,
        ],
        gap="large",
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
                    title=(
                        "Mean absolute SHAP impact"
                    ),
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
            "Mean absolute SHAP measures average influence on model "
            "output. It does not imply causality, and the direction "
            "of an individual claim's contribution can differ from "
            "the global average."
        ),
        tone="info",
    )


# =============================================================================
# Risk intelligence
# =============================================================================


def _render_model_observations() -> None:

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

    mechanism = _load_artifact_csv(
        "explainability",
        "mechanism_score_summary.csv",
    )

    difficulty = _load_artifact_csv(
        "explainability",
        "difficulty_score_summary.csv",
    )

    missed = _load_artifact_csv(
        "explainability",
        "false_negative_by_mechanism.csv",
    )

    valid_mechanism = pd.DataFrame()

    if (
        not mechanism.empty
        and {
            "fraud_mechanism",
            "mean_score",
        }.issubset(
            mechanism.columns
        )
    ):

        valid_mechanism = mechanism.copy()

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

        difficulty = difficulty.copy()

        difficulty[
            "mean_score"
        ] = pd.to_numeric(
            difficulty[
                "mean_score"
            ],
            errors="coerce",
        )

        hard = difficulty.loc[
            difficulty[
                "fraud_difficulty"
            ]
            .astype(str)
            .str.lower()
            .eq("hard")
        ]

        if not hard.empty:
            hard_row = hard.iloc[0]

    c1, c2, c3 = st.columns(3)

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

            mechanism_name = weakest[
                "fraud_mechanism"
            ]

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

                matching = missed.loc[
                    missed[
                        "fraud_mechanism"
                    ]
                    .astype(str)
                    .eq(
                        str(
                            mechanism_name
                        )
                    )
                ]

                if not matching.empty:

                    missed_count = _safe_int(
                        matching
                        .iloc[0]
                        .get(
                            "missed_fraud_claims"
                        )
                    )

                    if missed_count is not None:

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

            fraud_claims = _safe_int(
                hard_row.get(
                    "fraud_claims"
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
                    else "Hard synthetic fraud"
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

    records = []

    for label, parts in artifact_specs:

        path = _artifact_path(
            *parts
        )

        available = (
            path is not None
            and path.exists()
            and path.is_file()
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
            }
        )

    available_count = sum(
        record[
            "Status"
        ]
        == "AVAILABLE"

        for record in records
    )

    c1, c2 = st.columns(2)

    with c1:

        metric_card(
            "Analytical Artifacts",
            (
                f"{available_count}/"
                f"{len(records)}"
            ),
            "Core evidence available",
            tone=(
                "success"
                if available_count
                == len(records)
                else "warning"
            ),
        )

    with c2:

        metric_card(
            "Artifact Runtime",
            (
                "READY"
                if ARTIFACTS_DIR
                is not None
                else "UNAVAILABLE"
            ),
            (
                str(ARTIFACTS_DIR)
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
        )


# =============================================================================
# Governance
# =============================================================================


def _render_governance() -> None:

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
    client,
) -> None:

    section_header(
        "Executive Overview",
        (
            "Fraud-risk performance, investigation capacity, "
            "model intelligence and live inference readiness."
        ),
    )

    metadata = _load_metadata()

    (
        health,
        runtime_model,
        runtime_error,
    ) = _read_runtime_system(
        client
    )

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

    st.write("")
    st.write("")

    # =========================================================================
    # Capacity + runtime
    # =========================================================================

    left, right = st.columns(
        [
            1.55,
            1,
        ],
        gap="large",
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
    # Governance consistency
    # =========================================================================

    _render_contract_consistency(
        metadata,
        runtime_model,
        runtime_error,
    )

    # =========================================================================
    # Explainability
    # =========================================================================

    _render_risk_drivers()

    # =========================================================================
    # Risk intelligence
    # =========================================================================

    _render_model_observations()

    # =========================================================================
    # Evidence traceability
    # =========================================================================

    _render_artifact_coverage()

    # =========================================================================
    # Human governance
    # =========================================================================

    _render_governance()