from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from components import (
    human_review_notice,
    info_panel,
    key_value_row,
    metric_card,
    mini_metric,
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


METADATA_DIR = (
    ARTIFACTS_DIR
    / "metadata"
)


EXPLAINABILITY_DIR = (
    ARTIFACTS_DIR
    / "explainability"
)


FIGURES_DIR = (
    EXPLAINABILITY_DIR
    / "figures"
)


FINAL_EVALUATION_DIR = (
    METADATA_DIR
    / "final_evaluation"
)


METADATA_PATH = (
    METADATA_DIR
    / "health_fraud_model_metadata.json"
)


# =============================================================================
# Analytical artifacts
# =============================================================================


BUSINESS_IMPORTANCE_FILE = (
    "business_feature_importance.csv"
)


MECHANISM_SCORE_FILE = (
    "mechanism_score_summary.csv"
)


FALSE_NEGATIVE_FILE = (
    "false_negative_by_mechanism.csv"
)


DIFFICULTY_SCORE_FILE = (
    "difficulty_score_summary.csv"
)


TOP_SHAP_FEATURES = 15


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
            "Precision and recall trade-off on "
            "the out-of-time test population."
        ),
    ),
    (
        "03_roc_test.png",
        "ROC Curve",
        (
            "Global ranking discrimination "
            "across possible decision thresholds."
        ),
    ),
    (
        "04_calibration_test.png",
        "Calibration Curve",
        (
            "Agreement between predicted risk and "
            "observed synthetic fraud frequency."
        ),
    ),
    (
        "05_capacity_curve.png",
        "Investigation Capacity",
        (
            "Fraud capture as operational "
            "investigation capacity increases."
        ),
    ),
]


ERROR_FIGURES = [
    (
        "case_false_negative_extreme.png",
        "Extreme False Negative",
        (
            "High-severity fraudulent claim that "
            "received insufficient model risk."
        ),
    ),
    (
        "case_false_positive_high_risk.png",
        "High-Risk False Positive",
        (
            "Legitimate claim receiving a high "
            "fraud-risk score."
        ),
    ),
    (
        "case_amount_inflation_false_negative.png",
        "Amount Inflation False Negative",
        (
            "Amount-inflation fraud pattern not "
            "sufficiently prioritized by the model."
        ),
    ),
    (
        "case_legitimate_anomaly_not_reviewed.png",
        "Legitimate Anomaly",
        (
            "Unusual but legitimate behavior illustrating "
            "the distinction between anomaly and fraud."
        ),
    ),
]


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class ArtifactCheck:
    """
    Analytical artifact availability record.
    """

    label: str
    path: Path
    category: str

    @property
    def available(self) -> bool:
        return (
            self.path.exists()
            and self.path.is_file()
        )


# =============================================================================
# Generic helpers
# =============================================================================


def _safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """
    Convert a value into a finite float.
    """

    try:
        result = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default

    if not np.isfinite(result):
        return default

    return result


def _safe_int(
    value: Any,
    default: int | None = None,
) -> int | None:
    """
    Convert a numeric-like value safely into an integer.
    """

    try:
        if pd.isna(value):
            return default

        result = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default

    if not np.isfinite(result):
        return default

    return int(
        round(result)
    )


def _pretty_name(
    value: Any,
) -> str:
    """
    Convert machine-oriented names into readable labels.
    """

    if value is None:
        return "—"

    text = str(value).strip()

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


def _format_metric(
    value: Any,
    digits: int = 3,
) -> str:
    """
    Format a decimal metric.
    """

    number = _safe_float(
        value
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
    Format a ratio as percentage.
    """

    number = _safe_float(
        value
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
    Format lift relative to random review.
    """

    number = _safe_float(
        value
    )

    if number is None:
        return "—"

    return f"{number:.2f}×"


def _nonempty_dict(
    value: Any,
) -> dict[str, Any]:
    """
    Normalize arbitrary values into dictionaries.
    """

    return (
        value
        if isinstance(
            value,
            dict,
        )
        else {}
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
    Read an analytical CSV defensively.
    """

    path = Path(
        path_string
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

    return frame


@st.cache_data(
    show_spinner=False,
)
def _read_metadata(
    path_string: str,
) -> dict[str, Any]:
    """
    Read frozen-model metadata.
    """

    path = Path(
        path_string
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

            payload = json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    return (
        payload
        if isinstance(
            payload,
            dict,
        )
        else {}
    )


def _load_csv(
    filename: str,
) -> pd.DataFrame:
    """
    Load one explainability artifact.
    """

    return _read_csv(
        str(
            EXPLAINABILITY_DIR
            / filename
        )
    )


def _load_metadata() -> dict[str, Any]:
    """
    Load frozen-model metadata.
    """

    return _read_metadata(
        str(
            METADATA_PATH
        )
    )


def _existing_image(
    directory: Path,
    filename: str,
) -> Path | None:
    """
    Resolve an existing image artifact.
    """

    path = (
        directory
        / filename
    )

    if (
        path.exists()
        and path.is_file()
    ):
        return path

    return None


# =============================================================================
# Runtime model contract
# =============================================================================


def _read_runtime_model(
    client,
) -> tuple[
    dict[str, Any],
    str | None,
]:
    """
    Retrieve the contract of the model currently served by the API.

    Runtime API information is authoritative for deployment state.
    Local metadata remains authoritative for frozen evaluation results.
    """

    try:
        payload = client.model_info()

    except Exception as exc:

        return (
            {},
            str(exc),
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


# =============================================================================
# Metadata access
# =============================================================================


def _final_metrics(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Return frozen out-of-time evaluation metrics.
    """

    return _nonempty_dict(
        metadata.get(
            "final_test_metrics"
        )
    )


def _review_policy(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
) -> dict[str, Any]:
    """
    Resolve review policy.

    Runtime contract takes precedence because it represents
    the model currently served by the API.
    """

    runtime_policy = _nonempty_dict(
        runtime_model.get(
            "review_policy"
        )
    )

    if runtime_policy:
        return runtime_policy

    return _nonempty_dict(
        metadata.get(
            "review_policy"
        )
    )


def _runtime_explainability(
    runtime_model: dict[str, Any],
) -> dict[str, Any]:
    """
    Resolve live explainability contract.
    """

    return _nonempty_dict(
        runtime_model.get(
            "explainability"
        )
    )


# =============================================================================
# Contract consistency
# =============================================================================


def _normalized_contract_value(
    value: Any,
) -> Any:
    """
    Normalize values before contract comparison.
    """

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        (int, np.integer),
    ):
        return int(value)

    if isinstance(
        value,
        (float, np.floating),
    ):
        result = float(value)

        return (
            result
            if np.isfinite(result)
            else None
        )

    return str(value).strip()


def _contract_consistency(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
) -> tuple[
    bool | None,
    list[str],
    int,
]:
    """
    Compare frozen metadata against the currently served model.

    Only fields available in both contracts are compared.
    """

    if (
        not metadata
        or not runtime_model
    ):
        return (
            None,
            [],
            0,
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
            "Prediction target",
        ),
        (
            "feature_count",
            "Source feature count",
        ),
        (
            "transformed_feature_count",
            "Transformed feature count",
        ),
        (
            "probability_method",
            "Probability method",
        ),
    ]

    mismatches: list[str] = []
    compared = 0

    for key, label in comparisons:

        local_value = (
            metadata.get(key)
        )

        runtime_value = (
            runtime_model.get(key)
        )

        if (
            local_value is None
            or runtime_value is None
        ):
            continue

        compared += 1

        if (
            _normalized_contract_value(
                local_value
            )
            !=
            _normalized_contract_value(
                runtime_value
            )
        ):

            mismatches.append(
                (
                    f"{label}: "
                    f"artifact={local_value} • "
                    f"runtime={runtime_value}"
                )
            )

    # -------------------------------------------------------------------------
    # Review policy
    # -------------------------------------------------------------------------

    local_policy = _nonempty_dict(
        metadata.get(
            "review_policy"
        )
    )

    runtime_policy = _nonempty_dict(
        runtime_model.get(
            "review_policy"
        )
    )

    for key, label in [
        (
            "type",
            "Review policy type",
        ),
        (
            "fraction",
            "Review fraction",
        ),
    ]:

        local_value = (
            local_policy.get(key)
        )

        runtime_value = (
            runtime_policy.get(key)
        )

        if (
            local_value is None
            or runtime_value is None
        ):
            continue

        compared += 1

        local_number = _safe_float(
            local_value
        )

        runtime_number = _safe_float(
            runtime_value
        )

        if (
            key == "fraction"
            and local_number is not None
            and runtime_number is not None
        ):

            if not np.isclose(
                local_number,
                runtime_number,
                rtol=0.0,
                atol=1e-12,
            ):
                mismatches.append(
                    (
                        f"{label}: "
                        f"artifact={local_value} • "
                        f"runtime={runtime_value}"
                    )
                )

        elif (
            str(local_value)
            != str(runtime_value)
        ):

            mismatches.append(
                (
                    f"{label}: "
                    f"artifact={local_value} • "
                    f"runtime={runtime_value}"
                )
            )

    if compared == 0:
        return (
            None,
            [],
            0,
        )

    return (
        not mismatches,
        mismatches,
        compared,
    )


# =============================================================================
# Model status
# =============================================================================


def _render_runtime_status(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
    runtime_error: str | None,
) -> None:
    """
    Render deployment and artifact synchronization status.
    """

    section_header(
        "Model Status",
        (
            "Live inference availability and consistency "
            "between deployment and frozen analytical artifacts."
        ),
        eyebrow="RUNTIME",
    )

    (
        consistent,
        mismatches,
        compared,
    ) = _contract_consistency(
        metadata,
        runtime_model,
    )

    explainability = (
        _runtime_explainability(
            runtime_model
        )
    )

    explainability_available = (
        explainability.get(
            "available"
        )
        is True
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Inference API",
            (
                "ONLINE"
                if runtime_model
                else "UNAVAILABLE"
            ),
            (
                "Runtime contract loaded"
                if runtime_model
                else "Runtime contract unavailable"
            ),
            tone=(
                "success"
                if runtime_model
                else "danger"
            ),
        )

    with c2:

        metric_card(
            "Model Version",
            str(
                runtime_model.get(
                    "model_version"
                )
                or metadata.get(
                    "model_version"
                )
                or "—"
            ),
            "Currently evaluated deployment",
            tone="info",
        )

    with c3:

        metric_card(
            "Contract",
            (
                "CONSISTENT"
                if consistent is True
                else (
                    "MISMATCH"
                    if consistent is False
                    else "UNVERIFIED"
                )
            ),
            (
                f"{compared} fields compared"
                if compared
                else "No comparable contract"
            ),
            tone=(
                "success"
                if consistent is True
                else (
                    "danger"
                    if consistent is False
                    else "warning"
                )
            ),
        )

    with c4:

        metric_card(
            "TreeSHAP",
            (
                "READY"
                if explainability_available
                else "UNAVAILABLE"
            ),
            (
                str(
                    explainability.get(
                        "method"
                    )
                    or "Runtime explainability"
                )
            ),
            tone=(
                "success"
                if explainability_available
                else "warning"
            ),
        )

    if runtime_error:

        st.write("")

        info_panel(
            "Runtime Verification Failed",
            (
                "The frozen analytical artifacts remain available, "
                "but the currently served model could not be verified "
                f"through the inference API: {runtime_error}"
            ),
            tone="warning",
        )

    if mismatches:

        st.write("")

        info_panel(
            "Model Contract Mismatch",
            (
                "The deployment contract differs from the frozen "
                "metadata used by this analytical workspace. "
                "Evaluation results should not be attributed to "
                "the runtime model until the discrepancy is resolved."
            ),
            tone="danger",
        )

        with st.expander(
            "Contract differences",
            expanded=True,
        ):

            for mismatch in mismatches:
                st.write(
                    f"- {mismatch}"
                )


# =============================================================================
# Performance summary
# =============================================================================


def _render_model_summary(
    metadata: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    """
    Render frozen out-of-time model performance.
    """

    st.write("")
    st.write("")

    section_header(
        "Model Performance",
        (
            "Out-of-time evaluation of the frozen "
            "fraud-risk ranking model."
        ),
        eyebrow="OUT-OF-TIME EVALUATION",
    )

    metrics = _final_metrics(
        metadata
    )

    if not metrics:

        info_panel(
            "Evaluation Metrics Unavailable",
            (
                "The final_test_metrics section is not "
                "available in the frozen model metadata."
            ),
            tone="warning",
        )

        return

    review_fraction = _safe_float(
        policy.get(
            "fraction"
        )
    )

    policy_label = (
        f"{review_fraction:.0%}"
        if review_fraction is not None
        else "operational"
    )

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
            "Primary imbalanced ranking metric",
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
            "Global ranking discrimination",
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
            "Probability error • lower is better",
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
            "Probability quality • lower is better",
        )

    st.write("")

    info_panel(
        "Operational Interpretation",
        (
            "This model is deployed as a ranking system rather than "
            "a binary fraud adjudicator. Under constrained investigation "
            f"capacity ({policy_label} by default), recall, precision, "
            "lift and fraud-amount capture are therefore the primary "
            "operational measures. ROC-AUC and Average Precision describe "
            "broader ranking quality, while Brier score and log loss "
            "describe probability quality."
        ),
        tone="info",
    )


# =============================================================================
# Model architecture / contract
# =============================================================================


def _render_model_contract(
    metadata: dict[str, Any],
    runtime_model: dict[str, Any],
) -> None:
    """
    Render model identity, feature-space contract and review policy.
    """

    st.write("")
    st.write("")

    section_header(
        "Frozen Model Contract",
        (
            "Deployment identity, inference feature space, "
            "explainability contract and operational review policy."
        ),
        eyebrow="MODEL GOVERNANCE",
    )

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
    )

    transformed_count = (
        runtime_model.get(
            "transformed_feature_count"
        )
        or metadata.get(
            "transformed_feature_count"
        )
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

    policy = _review_policy(
        metadata,
        runtime_model,
    )

    fraction = _safe_float(
        policy.get(
            "fraction"
        )
    )

    explainability = (
        _runtime_explainability(
            runtime_model
        )
    )

    left, right = st.columns(
        [
            1.25,
            1,
        ],
        gap="large",
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Deployed Model"
            )

            m1, m2 = st.columns(2)

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

            m1, m2, m3 = st.columns(3)

            with m1:

                mini_metric(
                    "Source Features",
                    (
                        str(feature_count)
                        if feature_count is not None
                        else "—"
                    ),
                    helper="API input space",
                )

            with m2:

                mini_metric(
                    "Transformed Features",
                    (
                        str(transformed_count)
                        if transformed_count is not None
                        else "—"
                    ),
                    helper="Model input space",
                    tone="info",
                )

            with m3:

                mini_metric(
                    "Target",
                    str(target),
                    helper="Prediction target",
                )

            st.write("")
            st.divider()

            key_value_row(
                "Probability method",
                str(
                    probability_method
                ),
                monospace=True,
            )

            key_value_row(
                "Training period end",
                str(
                    metadata.get(
                        "training_period_end",
                        "—",
                    )
                ),
            )

            test_period = _nonempty_dict(
                metadata.get(
                    "test_period"
                )
            )

            key_value_row(
                "Out-of-time test start",
                str(
                    test_period.get(
                        "start",
                        "—",
                    )
                ),
            )

            key_value_row(
                "Out-of-time test end",
                str(
                    test_period.get(
                        "end",
                        "—",
                    )
                ),
            )

    # -------------------------------------------------------------------------
    # Policy / explainability
    # -------------------------------------------------------------------------

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Operational Contract"
            )

            metric_card(
                "Investigation Capacity",
                (
                    f"{fraction:.0%}"
                    if fraction is not None
                    else "—"
                ),
                "Default review policy",
                tone="info",
            )

            st.write("")

            key_value_row(
                "Review policy",
                str(
                    policy.get(
                        "type",
                        "—",
                    )
                ),
                monospace=True,
            )

            key_value_row(
                "Explainability",
                (
                    "Available"
                    if explainability.get(
                        "available"
                    )
                    is True
                    else "Unavailable"
                ),
            )

            key_value_row(
                "Method",
                str(
                    explainability.get(
                        "method",
                        "—",
                    )
                ),
                monospace=True,
            )

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

            key_value_row(
                "SHAP feature count",
                str(
                    explainability.get(
                        "transformed_feature_count",
                        "—",
                    )
                ),
            )

            st.write("")

            st.caption(
                (
                    "Review capacity determines which claims enter "
                    "the investigation queue. It is not an individual "
                    "fraud classification threshold."
                )
            )


# =============================================================================
# Inference pipeline
# =============================================================================


def _render_inference_pipeline(
    runtime_model: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """
    Explain the production inference and explanation spaces.
    """

    st.write("")
    st.write("")

    section_header(
        "Inference Architecture",
        (
            "Relationship between business inputs, preprocessing, "
            "XGBoost scoring and TreeSHAP explanations."
        ),
        eyebrow="ML PIPELINE",
    )

    source_features = (
        runtime_model.get(
            "feature_count"
        )
        or metadata.get(
            "feature_count"
        )
        or "—"
    )

    transformed_features = (
        runtime_model.get(
            "transformed_feature_count"
        )
        or metadata.get(
            "transformed_feature_count"
        )
        or "—"
    )

    explainability = (
        _runtime_explainability(
            runtime_model
        )
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Business Input",
            str(
                source_features
            ),
            "Source claim features",
        )

    with c2:

        metric_card(
            "Preprocessing",
            str(
                transformed_features
            ),
            "Transformed model features",
            tone="info",
        )

    with c3:

        metric_card(
            "Risk Model",
            str(
                runtime_model.get(
                    "model_name"
                )
                or metadata.get(
                    "model_name"
                )
                or "XGBoost"
            ),
            "Fraud-risk ranking",
            tone="info",
        )

    with c4:

        metric_card(
            "Explanation",
            str(
                explainability.get(
                    "method"
                )
                or "TreeSHAP"
            ),
            (
                str(
                    explainability.get(
                        "output_space"
                    )
                    or "Model output space"
                )
            ),
            tone=(
                "success"
                if explainability.get(
                    "available"
                )
                is True
                else "warning"
            ),
        )

    st.write("")

    info_panel(
        "Feature-Space Interpretation",
        (
            f"A claim enters the API using {source_features} source "
            f"features. The frozen preprocessor maps those variables "
            f"into {transformed_features} model features consumed by "
            "XGBoost. TreeSHAP operates in that transformed feature "
            "space. Business-level global importance artifacts can "
            "subsequently aggregate transformed contributions back "
            "into interpretable business concepts."
        ),
        tone="info",
    )


# =============================================================================
# SHAP validation
# =============================================================================


def _prepare_shap_importance(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    str | None,
]:
    """
    Validate and normalize global SHAP importance.
    """

    required = {
        "business_feature",
        "mean_abs_shap",
    }

    if frame.empty:
        return (
            pd.DataFrame(),
            "Global SHAP artifact is empty.",
        )

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        return (
            pd.DataFrame(),
            (
                "Global SHAP artifact is missing: "
                + ", ".join(
                    sorted(missing)
                )
            ),
        )

    data = frame.copy()

    data["business_feature"] = (
        data["business_feature"]
        .astype(str)
        .str.strip()
    )

    data["mean_abs_shap"] = (
        pd.to_numeric(
            data["mean_abs_shap"],
            errors="coerce",
        )
    )

    if "signed_mean_shap" in data.columns:

        data[
            "signed_mean_shap"
        ] = pd.to_numeric(
            data["signed_mean_shap"],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "mean_abs_shap"
        ]
    )

    if data.empty:

        return (
            pd.DataFrame(),
            (
                "Global SHAP artifact contains "
                "no valid importance values."
            ),
        )

    values = data[
        "mean_abs_shap"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():

        return (
            pd.DataFrame(),
            (
                "Global SHAP artifact contains "
                "non-finite importance values."
            ),
        )

    if (
        values < 0
    ).any():

        return (
            pd.DataFrame(),
            (
                "mean_abs_shap contains negative values, "
                "which violates the expected artifact contract."
            ),
        )

    data = data.loc[
        data["business_feature"]
        .ne("")
    ]

    data[
        "display_feature"
    ] = (
        data["business_feature"]
        .apply(
            _pretty_name
        )
    )

    return (
        data
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        ),
        None,
    )


# =============================================================================
# Global explainability
# =============================================================================


def _render_global_explainability(
    runtime_model: dict[str, Any],
) -> dict[str, Any]:
    """
    Render global business-level SHAP diagnostics.
    """

    st.write("")
    st.write("")

    section_header(
        "Global Explainability",
        (
            "Business features with the strongest average "
            "influence on model output across the analyzed population."
        ),
        eyebrow="GLOBAL SHAP",
    )

    runtime_explainability = (
        _runtime_explainability(
            runtime_model
        )
    )

    importance = _load_csv(
        BUSINESS_IMPORTANCE_FILE
    )

    importance, error = (
        _prepare_shap_importance(
            importance
        )
    )

    if error:

        info_panel(
            "Global SHAP Unavailable",
            error,
            tone="warning",
        )

        return {}

    top = (
        importance
        .head(
            TOP_SHAP_FEATURES
        )
        .copy()
    )

    top_feature = (
        top.iloc[0]
        if not top.empty
        else None
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:

        metric_card(
            "Business Features",
            f"{len(importance):,}",
            "Global SHAP aggregation",
        )

    with c2:

        metric_card(
            "Top Driver",
            (
                str(
                    top_feature[
                        "display_feature"
                    ]
                )
                if top_feature is not None
                else "—"
            ),
            (
                (
                    f"Mean |SHAP| "
                    f"{top_feature['mean_abs_shap']:.4f}"
                )
                if top_feature is not None
                else "Unavailable"
            ),
            tone="info",
        )

    with c3:

        metric_card(
            "SHAP Method",
            str(
                runtime_explainability.get(
                    "method"
                )
                or "TreeSHAP"
            ),
            str(
                runtime_explainability.get(
                    "output_space"
                )
                or "Global artifact"
            ),
            tone="success",
        )

    st.write("")

    tooltip = [
        alt.Tooltip(
            "display_feature:N",
            title="Feature",
        ),
        alt.Tooltip(
            "mean_abs_shap:Q",
            title="Mean |SHAP|",
            format=".5f",
        ),
    ]

    if (
        "signed_mean_shap"
        in top.columns
    ):

        tooltip.append(
            alt.Tooltip(
                "signed_mean_shap:Q",
                title="Signed Mean SHAP",
                format=".5f",
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
            tooltip=tooltip,
        )
        .properties(
            height=500
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    table_columns = [
        column
        for column in [
            "display_feature",
            "mean_abs_shap",
            "signed_mean_shap",
        ]
        if column in top.columns
    ]

    with st.expander(
        "Global SHAP ranking",
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
                        "Business Feature",
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

    global_figure = _existing_image(
        FIGURES_DIR,
        "01_shap_global_bar.png",
    )

    beeswarm = _existing_image(
        FIGURES_DIR,
        "02_shap_beeswarm.png",
    )

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### Global Importance"
            )

            if global_figure:

                st.image(
                    str(
                        global_figure
                    ),
                    use_container_width=True,
                )

            else:

                st.info(
                    (
                        "Global SHAP figure is not "
                        "available in current artifacts."
                    )
                )

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### SHAP Distribution"
            )

            if beeswarm:

                st.image(
                    str(
                        beeswarm
                    ),
                    use_container_width=True,
                )

            else:

                st.info(
                    (
                        "SHAP beeswarm figure is not "
                        "available in current artifacts."
                    )
                )

    st.write("")

    info_panel(
        "How to Read SHAP",
        (
            "Mean absolute SHAP measures average influence magnitude, "
            "not causal importance. Signed mean SHAP summarizes average "
            "direction in the model-output space, but opposite effects "
            "across observations may cancel. Claim-level explanations "
            "should therefore be inspected separately through the "
            "Claim Analysis workflow."
        ),
        tone="info",
    )

    return {
        "feature_count":
            len(importance),

        "top_feature":
            (
                str(
                    top_feature[
                        "display_feature"
                    ]
                )
                if top_feature is not None
                else None
            ),

        "top_mean_abs_shap":
            (
                _safe_float(
                    top_feature[
                        "mean_abs_shap"
                    ]
                )
                if top_feature is not None
                else None
            ),
    }


# =============================================================================
# Fraud mechanism analysis
# =============================================================================


def _render_mechanism_analysis() -> dict[str, Any]:
    """
    Analyze model behavior across synthetic fraud mechanisms.
    """

    st.write("")
    st.write("")

    section_header(
        "Fraud Mechanism Analysis",
        (
            "Risk-score behavior across the synthetic "
            "fraud mechanisms represented in evaluation data."
        ),
        eyebrow="SEGMENT PERFORMANCE",
    )

    mechanism = _load_csv(
        MECHANISM_SCORE_FILE
    )

    missed = _load_csv(
        FALSE_NEGATIVE_FILE
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

    data = mechanism.copy()

    numeric_columns = [
        "mean_score",
        "median_score",
        "fraud_claims",
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = (
                pd.to_numeric(
                    data[column],
                    errors="coerce",
                )
            )

    data = data.dropna(
        subset=[
            "mean_score"
        ]
    )

    data = data.loc[
        np.isfinite(
            data["mean_score"]
        )
    ]

    if data.empty:

        info_panel(
            "Mechanism Analysis Unavailable",
            (
                "No valid mechanism-level risk "
                "statistics are available."
            ),
            tone="warning",
        )

        return {}

    data["Mechanism"] = (
        data["fraud_mechanism"]
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

        missed = missed.copy()

        missed[
            "missed_fraud_claims"
        ] = pd.to_numeric(
            missed[
                "missed_fraud_claims"
            ],
            errors="coerce",
        )

        missed["Mechanism"] = (
            missed["fraud_mechanism"]
            .apply(
                _pretty_name
            )
        )

        data = data.merge(
            missed[
                [
                    "Mechanism",
                    "missed_fraud_claims",
                ]
            ],
            on="Mechanism",
            how="left",
            validate="one_to_one",
        )

    left, right = st.columns(
        [
            1.3,
            1,
        ],
        gap="large",
    )

    with left:

        tooltip = [
            alt.Tooltip(
                "Mechanism:N",
                title="Mechanism",
            ),
            alt.Tooltip(
                "mean_score:Q",
                title="Mean risk",
                format=".2%",
            ),
        ]

        if "fraud_claims" in data.columns:

            tooltip.append(
                alt.Tooltip(
                    "fraud_claims:Q",
                    title="Fraud claims",
                    format=",",
                )
            )

        if "median_score" in data.columns:

            tooltip.append(
                alt.Tooltip(
                    "median_score:Q",
                    title="Median risk",
                    format=".2%",
                )
            )

        chart = (
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
                tooltip=tooltip,
            )
            .properties(
                height=360
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

    with right:

        visible_columns = [
            column
            for column in [
                "Mechanism",
                "fraud_claims",
                "mean_score",
                "median_score",
                "missed_fraud_claims",
            ]
            if column in data.columns
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
                        format="%.3f",
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

    ordered = data.sort_values(
        "mean_score",
        ascending=False,
    )

    strongest = ordered.iloc[0]
    weakest = ordered.iloc[-1]

    st.write("")

    c1, c2 = st.columns(2)

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

    return {
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
    }


# =============================================================================
# Difficulty analysis
# =============================================================================


def _render_difficulty_analysis() -> dict[str, Any]:
    """
    Analyze score behavior across synthetic fraud difficulty.
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

    difficulty = _load_csv(
        DIFFICULTY_SCORE_FILE
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

    difficulty = difficulty.copy()

    for column in [
        "mean_score",
        "median_score",
        "fraud_claims",
    ]:

        if column in difficulty.columns:

            difficulty[column] = (
                pd.to_numeric(
                    difficulty[column],
                    errors="coerce",
                )
            )

    difficulty = difficulty.dropna(
        subset=[
            "mean_score"
        ]
    )

    difficulty = difficulty.loc[
        np.isfinite(
            difficulty[
                "mean_score"
            ]
        )
    ]

    if difficulty.empty:

        info_panel(
            "Difficulty Analysis Unavailable",
            (
                "No valid difficulty-level "
                "risk statistics are available."
            ),
            tone="warning",
        )

        return {}

    difficulty["Difficulty"] = (
        difficulty["fraud_difficulty"]
        .apply(
            _pretty_name
        )
    )

    tooltip = [
        alt.Tooltip(
            "Difficulty:N",
            title="Difficulty",
        ),
        alt.Tooltip(
            "mean_score:Q",
            title="Mean risk",
            format=".2%",
        ),
    ]

    if (
        "fraud_claims"
        in difficulty.columns
    ):

        tooltip.append(
            alt.Tooltip(
                "fraud_claims:Q",
                title="Fraud claims",
                format=",",
            )
        )

    if (
        "median_score"
        in difficulty.columns
    ):

        tooltip.append(
            alt.Tooltip(
                "median_score:Q",
                title="Median risk",
                format=".2%",
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
            tooltip=tooltip,
        )
        .properties(
            height=320
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    summary: dict[str, Any] = {}

    columns = st.columns(3)

    tone_map = {
        "easy": "success",
        "medium": "info",
        "hard": "warning",
    }

    for column, level in zip(
        columns,
        [
            "easy",
            "medium",
            "hard",
        ],
    ):

        subset = difficulty.loc[
            difficulty[
                "fraud_difficulty"
            ]
            .astype(str)
            .str.lower()
            .eq(level)
        ]

        if subset.empty:

            with column:

                metric_card(
                    f"{level.title()} Fraud",
                    "—",
                    "No observations",
                )

            continue

        row = subset.iloc[0]

        mean_score = _safe_float(
            row.get(
                "mean_score"
            )
        )

        fraud_claims = _safe_int(
            row.get(
                "fraud_claims"
            )
        )

        summary[level] = mean_score

        with column:

            metric_card(
                f"{level.title()} Fraud",
                _format_percent(
                    mean_score,
                    1,
                ),
                (
                    f"{fraud_claims:,} fraud claims"
                    if fraud_claims is not None
                    else "Evaluation segment"
                ),
                tone=tone_map[
                    level
                ],
            )

    easy_score = summary.get(
        "easy"
    )

    hard_score = summary.get(
        "hard"
    )

    if (
        easy_score is not None
        and hard_score is not None
    ):

        gap = (
            easy_score
            - hard_score
        )

        summary[
            "easy_hard_gap"
        ] = gap

        st.write("")

        info_panel(
            "Difficulty Sensitivity",
            (
                "Mean model risk changes by "
                f"{gap:+.1%} when comparing easy "
                "with hard synthetic fraud."
            ),
            tone=(
                "warning"
                if gap > 0
                else "info"
            ),
        )

    return summary


# =============================================================================
# Evaluation diagnostics
# =============================================================================


def _render_evaluation_diagnostics() -> None:
    """
    Render frozen out-of-time validation figures.
    """

    st.write("")
    st.write("")

    section_header(
        "Evaluation Diagnostics",
        (
            "Frozen out-of-time figures used to inspect "
            "ranking, policy performance and probability quality."
        ),
        eyebrow="MODEL VALIDATION",
    )

    tabs = st.tabs(
        [
            title
            for _, title, _
            in EVALUATION_FIGURES
        ]
    )

    for tab, specification in zip(
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

            path = _existing_image(
                FINAL_EVALUATION_DIR,
                filename,
            )

            if path:

                st.image(
                    str(path),
                    use_container_width=True,
                )

            else:

                info_panel(
                    "Figure Unavailable",
                    (
                        f"{title} is not present in "
                        "the frozen evaluation artifacts."
                    ),
                    tone="warning",
                )


# =============================================================================
# Error analysis
# =============================================================================


def _render_error_analysis() -> None:
    """
    Render representative case-level model failures.
    """

    st.write("")
    st.write("")

    section_header(
        "Observed Failure Modes",
        (
            "Representative false-positive, false-negative "
            "and legitimate-anomaly cases identified during evaluation."
        ),
        eyebrow="ERROR ANALYSIS",
    )

    available: list[
        tuple[
            Path,
            str,
            str,
        ]
    ] = []

    for (
        filename,
        label,
        description,
    ) in ERROR_FIGURES:

        path = _existing_image(
            FIGURES_DIR,
            filename,
        )

        if path:

            available.append(
                (
                    path,
                    label,
                    description,
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

    columns = st.columns(2)

    for index, (
        path,
        label,
        description,
    ) in enumerate(
        available
    ):

        with columns[
            index % 2
        ]:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"#### {label}"
                )

                st.caption(
                    description
                )

                st.image(
                    str(path),
                    use_container_width=True,
                )


# =============================================================================
# Artifact coverage
# =============================================================================


def _artifact_inventory() -> list[
    ArtifactCheck
]:
    """
    Build the expected analytical artifact inventory.
    """

    checks = [
        ArtifactCheck(
            label="Model metadata",
            path=METADATA_PATH,
            category="Governance",
        ),
        ArtifactCheck(
            label="Business SHAP importance",
            path=(
                EXPLAINABILITY_DIR
                / BUSINESS_IMPORTANCE_FILE
            ),
            category="Explainability",
        ),
        ArtifactCheck(
            label="Mechanism analysis",
            path=(
                EXPLAINABILITY_DIR
                / MECHANISM_SCORE_FILE
            ),
            category="Robustness",
        ),
        ArtifactCheck(
            label="False-negative analysis",
            path=(
                EXPLAINABILITY_DIR
                / FALSE_NEGATIVE_FILE
            ),
            category="Error Analysis",
        ),
        ArtifactCheck(
            label="Difficulty analysis",
            path=(
                EXPLAINABILITY_DIR
                / DIFFICULTY_SCORE_FILE
            ),
            category="Robustness",
        ),
        ArtifactCheck(
            label="Global SHAP figure",
            path=(
                FIGURES_DIR
                / "01_shap_global_bar.png"
            ),
            category="Explainability",
        ),
        ArtifactCheck(
            label="SHAP beeswarm",
            path=(
                FIGURES_DIR
                / "02_shap_beeswarm.png"
            ),
            category="Explainability",
        ),
    ]

    for filename, title, _ in (
        EVALUATION_FIGURES
    ):

        checks.append(
            ArtifactCheck(
                label=title,
                path=(
                    FINAL_EVALUATION_DIR
                    / filename
                ),
                category="Evaluation",
            )
        )

    return checks


def _render_artifact_coverage() -> None:
    """
    Surface analytical evidence availability.
    """

    st.write("")
    st.write("")

    section_header(
        "Analytical Artifact Coverage",
        (
            "Traceability of the frozen evidence supporting "
            "performance, explainability and robustness analysis."
        ),
        eyebrow="TRACEABILITY",
    )

    checks = _artifact_inventory()

    available_count = sum(
        check.available
        for check in checks
    )

    total_count = len(
        checks
    )

    coverage_ratio = (
        available_count
        / total_count
        if total_count
        else 0.0
    )

    evaluation_checks = [
        check
        for check in checks
        if check.category
        == "Evaluation"
    ]

    explainability_checks = [
        check
        for check in checks
        if check.category
        == "Explainability"
    ]

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Artifacts Available",
            (
                f"{available_count}/"
                f"{total_count}"
            ),
            (
                f"{coverage_ratio:.0%} coverage"
            ),
            tone=(
                "success"
                if coverage_ratio == 1
                else "warning"
            ),
        )

    with c2:

        metric_card(
            "Evaluation",
            (
                f"{sum(c.available for c in evaluation_checks)}"
                f"/{len(evaluation_checks)}"
            ),
            "Frozen validation figures",
        )

    with c3:

        metric_card(
            "Explainability",
            (
                f"{sum(c.available for c in explainability_checks)}"
                f"/{len(explainability_checks)}"
            ),
            "Global SHAP artifacts",
        )

    with c4:

        metric_card(
            "Metadata",
            (
                "READY"
                if METADATA_PATH.exists()
                else "MISSING"
            ),
            "Frozen model contract",
            tone=(
                "success"
                if METADATA_PATH.exists()
                else "danger"
            ),
        )

    inventory = pd.DataFrame(
        [
            {
                "Category":
                    check.category,

                "Artifact":
                    check.label,

                "Status":
                    (
                        "AVAILABLE"
                        if check.available
                        else "MISSING"
                    ),

                "Path":
                    str(
                        check.path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
            }
            for check in checks
        ]
    )

    st.write("")

    with st.expander(
        "Artifact inventory",
        expanded=False,
    ):

        st.dataframe(
            inventory,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# Conclusions
# =============================================================================


def _render_conclusions(
    metadata: dict[str, Any],
    shap_summary: dict[str, Any],
    mechanism_summary: dict[str, Any],
    difficulty_summary: dict[str, Any],
) -> None:
    """
    Generate evidence-grounded model-governance conclusions.
    """

    st.write("")
    st.write("")

    section_header(
        "Model Interpretation Summary",
        (
            "Operational conclusions supported by the "
            "available frozen evaluation and explainability evidence."
        ),
        eyebrow="MODEL GOVERNANCE",
    )

    metrics = _final_metrics(
        metadata
    )

    lift = _safe_float(
        metrics.get(
            "lift_at_3pct"
        )
    )

    recall = _safe_float(
        metrics.get(
            "recall_at_3pct"
        )
    )

    precision = _safe_float(
        metrics.get(
            "precision_at_3pct"
        )
    )

    amount_capture = _safe_float(
        metrics.get(
            "fraud_amount_capture_at_3pct"
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

    top_feature = (
        shap_summary.get(
            "top_feature"
        )
    )

    left, right = st.columns(2)

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Observed Strengths"
            )

            observations = []

            if lift is not None:

                observations.append(
                    (
                        "At the 3% review policy, model targeting "
                        f"delivers approximately **{lift:.2f}× lift** "
                        "versus random review."
                    )
                )

            if recall is not None:

                observations.append(
                    (
                        "The selected 3% of claims captures "
                        f"**{recall:.1%}** of synthetic fraud cases."
                    )
                )

            if amount_capture is not None:

                observations.append(
                    (
                        "The same policy captures "
                        f"**{amount_capture:.1%}** of synthetic "
                        "fraud amount."
                    )
                )

            if precision is not None:

                observations.append(
                    (
                        "Investigation yield at the policy "
                        f"point is **{precision:.1%}**."
                    )
                )

            if strongest:

                observations.append(
                    (
                        "Highest observed mean risk among "
                        f"fraud mechanisms: **{strongest}**."
                    )
                )

            if top_feature:

                observations.append(
                    (
                        "Highest business-level global SHAP "
                        f"importance: **{top_feature}**."
                    )
                )

            if observations:

                for observation in observations:
                    st.write(
                        f"• {observation}"
                    )

            else:

                st.caption(
                    (
                        "No quantitative strength statement "
                        "can be generated from current artifacts."
                    )
                )

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Known Limitations"
            )

            limitations = []

            if weakest:

                limitations.append(
                    (
                        "Lowest observed mean fraud risk among "
                        f"mechanisms: **{weakest}**."
                    )
                )

            if (
                difficulty_gap is not None
                and difficulty_gap > 0
            ):

                limitations.append(
                    (
                        "Hard synthetic fraud receives approximately "
                        f"**{difficulty_gap:.1%} lower mean risk** "
                        "than easy fraud."
                    )
                )

            limitations.extend(
                [
                    (
                        "SHAP explains model behavior and "
                        "does not establish causal relationships."
                    ),
                    (
                        "A fraud-risk probability is a model output, "
                        "not proof that a claim is fraudulent."
                    ),
                    (
                        "Queue selection is an operational "
                        "prioritization mechanism, not adjudication."
                    ),
                    (
                        "Current validation is based on a synthetic "
                        "health-insurance environment and does not "
                        "establish real-world generalization."
                    ),
                ]
            )

            for limitation in limitations:
                st.write(
                    f"• {limitation}"
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
    Render the complete model intelligence and governance workspace.
    """

    section_header(
        "Model Insights",
        (
            "Performance, explainability, robustness, "
            "failure analysis and governance of the "
            "deployed fraud-risk model."
        ),
    )

    # -------------------------------------------------------------------------
    # Evidence sources
    # -------------------------------------------------------------------------

    metadata = _load_metadata()

    (
        runtime_model,
        runtime_error,
    ) = _read_runtime_model(
        client
    )

    policy = _review_policy(
        metadata,
        runtime_model,
    )

    # -------------------------------------------------------------------------
    # Availability
    # -------------------------------------------------------------------------

    if not metadata:

        info_panel(
            "Frozen Metadata Unavailable",
            (
                "The local model metadata artifact could not "
                "be loaded. Runtime deployment information may "
                "still be available, but historical evaluation "
                "results cannot be fully verified."
            ),
            tone="warning",
        )

        st.write("")

    # -------------------------------------------------------------------------
    # Runtime / governance
    # -------------------------------------------------------------------------

    _render_runtime_status(
        metadata,
        runtime_model,
        runtime_error,
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    _render_model_summary(
        metadata,
        policy,
    )

    # -------------------------------------------------------------------------
    # Contract
    # -------------------------------------------------------------------------

    _render_model_contract(
        metadata,
        runtime_model,
    )

    # -------------------------------------------------------------------------
    # Architecture
    # -------------------------------------------------------------------------

    _render_inference_pipeline(
        runtime_model,
        metadata,
    )

    # -------------------------------------------------------------------------
    # Explainability
    # -------------------------------------------------------------------------

    shap_summary = (
        _render_global_explainability(
            runtime_model
        )
    )

    # -------------------------------------------------------------------------
    # Robustness
    # -------------------------------------------------------------------------

    mechanism_summary = (
        _render_mechanism_analysis()
    )

    difficulty_summary = (
        _render_difficulty_analysis()
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    _render_evaluation_diagnostics()

    _render_error_analysis()

    # -------------------------------------------------------------------------
    # Governance
    # -------------------------------------------------------------------------

    _render_artifact_coverage()

    _render_conclusions(
        metadata,
        shap_summary,
        mechanism_summary,
        difficulty_summary,
    )