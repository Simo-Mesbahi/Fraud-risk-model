from __future__ import annotations

import json
import os
import platform
import sys
import time

from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from components import (
    info_panel,
    key_value_row,
    metric_card,
    mini_metric,
    section_header,
)


# =============================================================================
# Runtime path resolution
# =============================================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def _candidate_artifact_roots() -> list[Path]:
    """
    Resolve possible artifact roots across local development,
    Codespaces and containerized runtime environments.
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
            PROJECT_ROOT
            / "artifacts",

            Path(
                "/app/artifacts"
            ),

            Path.cwd()
            / "artifacts",
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


def _find_artifact_root() -> Path:
    """
    Return the first existing artifact root.

    If no candidate currently exists, return the configured or
    project-default location so diagnostics can still expose
    the expected runtime path.
    """

    for path in _candidate_artifact_roots():

        if (
            path.exists()
            and path.is_dir()
        ):
            return path

    configured = os.getenv(
        "ARTIFACTS_ROOT"
    )

    if configured:
        return (
            Path(configured)
            .expanduser()
        )

    return (
        PROJECT_ROOT
        / "artifacts"
    )


ARTIFACTS_ROOT = (
    _find_artifact_root()
)


MODELS_DIR = (
    ARTIFACTS_ROOT
    / "models"
)


PREPROCESSORS_DIR = (
    ARTIFACTS_ROOT
    / "preprocessors"
)


METADATA_DIR = (
    ARTIFACTS_ROOT
    / "metadata"
)


EXPLAINABILITY_DIR = (
    ARTIFACTS_ROOT
    / "explainability"
)


FINAL_EVALUATION_DIR = (
    METADATA_DIR
    / "final_evaluation"
)


MODEL_PATH = (
    MODELS_DIR
    / "health_fraud_xgboost.joblib"
)


PREPROCESSOR_PATH = (
    PREPROCESSORS_DIR
    / "health_fraud_preprocessor.joblib"
)


METADATA_PATH = (
    METADATA_DIR
    / "health_fraud_model_metadata.json"
)


# =============================================================================
# Expected analytical artifacts
# =============================================================================


EXPECTED_EXPLAINABILITY_ARTIFACTS = [
    (
        "Business SHAP importance",
        EXPLAINABILITY_DIR
        / "business_feature_importance.csv",
    ),
    (
        "Mechanism score summary",
        EXPLAINABILITY_DIR
        / "mechanism_score_summary.csv",
    ),
    (
        "False-negative mechanisms",
        EXPLAINABILITY_DIR
        / "false_negative_by_mechanism.csv",
    ),
    (
        "Difficulty score summary",
        EXPLAINABILITY_DIR
        / "difficulty_score_summary.csv",
    ),
    (
        "Global SHAP figure",
        EXPLAINABILITY_DIR
        / "figures"
        / "01_shap_global_bar.png",
    ),
    (
        "SHAP beeswarm",
        EXPLAINABILITY_DIR
        / "figures"
        / "02_shap_beeswarm.png",
    ),
]


EXPECTED_EVALUATION_ARTIFACTS = [
    (
        "Confusion Matrix",
        FINAL_EVALUATION_DIR
        / "01_confusion_matrix_top3.png",
    ),
    (
        "Precision–Recall",
        FINAL_EVALUATION_DIR
        / "02_precision_recall_test.png",
    ),
    (
        "ROC",
        FINAL_EVALUATION_DIR
        / "03_roc_test.png",
    ),
    (
        "Calibration",
        FINAL_EVALUATION_DIR
        / "04_calibration_test.png",
    ),
    (
        "Capacity Curve",
        FINAL_EVALUATION_DIR
        / "05_capacity_curve.png",
    ),
]


# =============================================================================
# Diagnostic structures
# =============================================================================


@dataclass(frozen=True)
class ArtifactStatus:
    """
    Status of one expected model artifact.
    """

    name: str
    category: str
    path: Path
    required_for_inference: bool

    @property
    def available(self) -> bool:
        return (
            self.path.exists()
            and self.path.is_file()
        )

    @property
    def size_bytes(self) -> int:
        if not self.available:
            return 0

        try:
            return (
                self.path.stat()
                .st_size
            )

        except OSError:
            return 0


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
    default: int = 0,
) -> int:
    """
    Convert numeric-like values safely into integers.
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


def _format_bytes(
    size: int,
) -> str:
    """
    Format byte counts for diagnostics.
    """

    value = float(
        max(
            int(size),
            0,
        )
    )

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    for unit in units:

        if (
            value < 1024
            or unit == "TB"
        ):
            return (
                f"{value:.1f} {unit}"
            )

        value /= 1024

    return "—"


def _format_latency(
    value: Any,
) -> str:
    """
    Format latency consistently.
    """

    latency = _safe_float(
        value
    )

    if latency is None:
        return "—"

    return (
        f"{latency:.1f} ms"
    )


def _utc_now() -> str:
    """
    Return a human-readable UTC timestamp.
    """

    return (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )


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


def _normalized_contract_value(
    value: Any,
) -> Any:
    """
    Normalize contract values before comparison.
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

        number = float(value)

        return (
            number
            if np.isfinite(number)
            else None
        )

    return str(value).strip()


def _path_status(
    path: Path,
) -> str:
    """
    Return compact filesystem status.
    """

    if not path.exists():
        return "MISSING"

    if path.is_file():
        return "FILE"

    if path.is_dir():
        return "DIRECTORY"

    return "UNKNOWN"


# =============================================================================
# Metadata
# =============================================================================


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def _read_metadata(
    path_string: str,
) -> dict[str, Any]:
    """
    Read frozen model metadata defensively.
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


# =============================================================================
# Artifact inventory
# =============================================================================


def _artifact_inventory() -> list[
    ArtifactStatus
]:
    """
    Build the expected model-system artifact inventory.
    """

    inventory = [
        ArtifactStatus(
            name="XGBoost Model",
            category="Inference",
            path=MODEL_PATH,
            required_for_inference=True,
        ),
        ArtifactStatus(
            name="Frozen Preprocessor",
            category="Inference",
            path=PREPROCESSOR_PATH,
            required_for_inference=True,
        ),
        ArtifactStatus(
            name="Model Metadata",
            category="Governance",
            path=METADATA_PATH,
            required_for_inference=True,
        ),
    ]

    inventory.extend(
        ArtifactStatus(
            name=name,
            category="Explainability",
            path=path,
            required_for_inference=False,
        )
        for name, path
        in EXPECTED_EXPLAINABILITY_ARTIFACTS
    )

    inventory.extend(
        ArtifactStatus(
            name=name,
            category="Evaluation",
            path=path,
            required_for_inference=False,
        )
        for name, path
        in EXPECTED_EVALUATION_ARTIFACTS
    )

    return inventory


def _inference_artifacts_ready() -> bool:
    """
    Check whether all local inference artifacts exist.
    """

    inference_artifacts = [
        artifact
        for artifact
        in _artifact_inventory()
        if artifact.required_for_inference
    ]

    return bool(
        inference_artifacts
        and all(
            artifact.available
            for artifact
            in inference_artifacts
        )
    )


def _analytics_coverage() -> tuple[
    int,
    int,
]:
    """
    Return analytical artifact availability.
    """

    analytical = [
        artifact
        for artifact
        in _artifact_inventory()
        if not artifact.required_for_inference
    ]

    available = sum(
        artifact.available
        for artifact
        in analytical
    )

    return (
        available,
        len(analytical),
    )


# =============================================================================
# API diagnostics
# =============================================================================


def _check_api(
    client,
) -> dict[str, Any]:
    """
    Execute health and model-contract diagnostics.

    Connectivity, endpoint validity and model readiness are kept
    separate so failures remain operationally interpretable.
    """

    result: dict[str, Any] = {
        "online": False,
        "health_ok": False,
        "model_info_ok": False,
        "health": {},
        "model": {},
        "health_latency_ms": None,
        "model_info_latency_ms": None,
        "total_latency_ms": None,
        "health_error": None,
        "model_info_error": None,
    }

    total_start = (
        time.perf_counter()
    )

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    try:

        start = (
            time.perf_counter()
        )

        health = client.health()

        result[
            "health_latency_ms"
        ] = (
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        result[
            "online"
        ] = True

        if isinstance(
            health,
            dict,
        ):

            result[
                "health"
            ] = health

            result[
                "health_ok"
            ] = True

        else:

            result[
                "health_error"
            ] = (
                "Health endpoint returned "
                "an invalid payload."
            )

    except Exception as exc:

        result[
            "health_error"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

        result[
            "total_latency_ms"
        ] = (
            (
                time.perf_counter()
                - total_start
            )
            * 1000
        )

        return result

    # -------------------------------------------------------------------------
    # Model info
    # -------------------------------------------------------------------------

    try:

        start = (
            time.perf_counter()
        )

        model = client.model_info()

        result[
            "model_info_latency_ms"
        ] = (
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        if isinstance(
            model,
            dict,
        ):

            result[
                "model"
            ] = model

            result[
                "model_info_ok"
            ] = True

        else:

            result[
                "model_info_error"
            ] = (
                "Model-info endpoint returned "
                "an invalid payload."
            )

    except Exception as exc:

        result[
            "model_info_error"
        ] = (
            f"{type(exc).__name__}: {exc}"
        )

    result[
        "total_latency_ms"
    ] = (
        (
            time.perf_counter()
            - total_start
        )
        * 1000
    )

    return result


# =============================================================================
# Runtime readiness
# =============================================================================


def _health_reports_ready(
    diagnostics: dict[str, Any],
) -> bool:
    """
    Interpret the semantic health endpoint contract.
    """

    health = _nonempty_dict(
        diagnostics.get(
            "health"
        )
    )

    if not health:
        return False

    status = (
        str(
            health.get(
                "status",
                "",
            )
        )
        .strip()
        .lower()
    )

    return (
        status == "ok"
        and health.get(
            "model_loaded"
        )
        is True
    )


def _api_model_ready(
    diagnostics: dict[str, Any],
) -> bool:
    """
    Confirm that the API exposes a usable inference model.
    """

    if not diagnostics.get(
        "online"
    ):
        return False

    if not diagnostics.get(
        "health_ok"
    ):
        return False

    if not diagnostics.get(
        "model_info_ok"
    ):
        return False

    if not _health_reports_ready(
        diagnostics
    ):
        return False

    model = _nonempty_dict(
        diagnostics.get(
            "model"
        )
    )

    required = [
        "model_name",
        "model_version",
        "target",
        "feature_count",
    ]

    return all(
        model.get(key)
        is not None
        for key in required
    )


def _runtime_explainability_ready(
    diagnostics: dict[str, Any],
) -> bool:
    """
    Determine whether the served model reports explainability support.
    """

    model = _nonempty_dict(
        diagnostics.get(
            "model"
        )
    )

    explainability = _nonempty_dict(
        model.get(
            "explainability"
        )
    )

    return (
        explainability.get(
            "available"
        )
        is True
    )


def _analytics_ready(
    metadata: dict[str, Any],
) -> bool:
    """
    Determine whether model-insight assets are sufficiently available.
    """

    available, total = (
        _analytics_coverage()
    )

    return bool(
        metadata
        and total > 0
        and available == total
    )


# =============================================================================
# Contract consistency
# =============================================================================


def _contract_comparison(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[
    list[dict[str, str]],
    int,
    int,
]:
    """
    Compare frozen analytical metadata with live deployment contract.
    """

    runtime = _nonempty_dict(
        diagnostics.get(
            "model"
        )
    )

    if (
        not runtime
        or not metadata
    ):
        return (
            [],
            0,
            0,
        )

    rows: list[
        dict[str, str]
    ] = []

    mismatches = 0
    comparisons = 0

    fields = [
        (
            "model_name",
            "Model Name",
        ),
        (
            "model_version",
            "Model Version",
        ),
        (
            "target",
            "Prediction Target",
        ),
        (
            "feature_count",
            "Source Feature Count",
        ),
        (
            "transformed_feature_count",
            "Transformed Feature Count",
        ),
        (
            "probability_method",
            "Probability Method",
        ),
    ]

    for key, label in fields:

        local = metadata.get(
            key
        )

        live = runtime.get(
            key
        )

        if (
            local is None
            and live is None
        ):
            continue

        comparisons += 1

        match = (
            local is not None
            and live is not None
            and _normalized_contract_value(
                local
            )
            ==
            _normalized_contract_value(
                live
            )
        )

        if not match:
            mismatches += 1

        rows.append(
            {
                "Field": label,
                "Local Artifact": (
                    "—"
                    if local is None
                    else str(local)
                ),
                "Runtime API": (
                    "—"
                    if live is None
                    else str(live)
                ),
                "Status": (
                    "MATCH"
                    if match
                    else "MISMATCH"
                ),
            }
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
        runtime.get(
            "review_policy"
        )
    )

    policy_fields = [
        (
            "type",
            "Review Policy",
        ),
        (
            "fraction",
            "Review Fraction",
        ),
    ]

    for key, label in policy_fields:

        local = local_policy.get(
            key
        )

        live = runtime_policy.get(
            key
        )

        if (
            local is None
            and live is None
        ):
            continue

        comparisons += 1

        if key == "fraction":

            local_float = _safe_float(
                local
            )

            live_float = _safe_float(
                live
            )

            match = (
                local_float is not None
                and live_float is not None
                and np.isclose(
                    local_float,
                    live_float,
                    rtol=0.0,
                    atol=1e-12,
                )
            )

        else:

            match = (
                local is not None
                and live is not None
                and str(local)
                == str(live)
            )

        if not match:
            mismatches += 1

        rows.append(
            {
                "Field": label,
                "Local Artifact": (
                    "—"
                    if local is None
                    else str(local)
                ),
                "Runtime API": (
                    "—"
                    if live is None
                    else str(live)
                ),
                "Status": (
                    "MATCH"
                    if match
                    else "MISMATCH"
                ),
            }
        )

    return (
        rows,
        comparisons,
        mismatches,
    )


def _contracts_consistent(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> bool | None:
    """
    Return deployment consistency state.
    """

    (
        _,
        comparisons,
        mismatches,
    ) = _contract_comparison(
        diagnostics,
        metadata,
    )

    if comparisons == 0:
        return None

    return (
        mismatches == 0
    )


# =============================================================================
# Global system status
# =============================================================================


def _system_status(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[
    str,
    str,
    str,
]:
    """
    Derive global application readiness.

    READY:
        inference operational, model contract consistent and
        analytical evidence complete.

    DEGRADED:
        inference remains operational but one or more non-critical
        governance / analytical checks are incomplete.

    NOT READY:
        inference itself cannot be guaranteed.
    """

    inference_ready = (
        _api_model_ready(
            diagnostics
        )
    )

    if not inference_ready:

        return (
            "NOT READY",
            "danger",
            (
                "End-to-end inference "
                "cannot be guaranteed."
            ),
        )

    consistent = (
        _contracts_consistent(
            diagnostics,
            metadata,
        )
    )

    analytics_ready = (
        _analytics_ready(
            metadata
        )
    )

    local_inference_ready = (
        _inference_artifacts_ready()
    )

    explainability_ready = (
        _runtime_explainability_ready(
            diagnostics
        )
    )

    if (
        consistent is True
        and analytics_ready
        and local_inference_ready
        and explainability_ready
    ):

        return (
            "READY",
            "success",
            (
                "Inference, governance and "
                "analytical evidence verified."
            ),
        )

    return (
        "DEGRADED",
        "warning",
        (
            "Inference is operational, but one or more "
            "governance or analytical checks are incomplete."
        ),
    )


# =============================================================================
# Service health
# =============================================================================


def _render_service_health(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """
    Render primary operational health.
    """

    section_header(
        "Inference Service",
        (
            "Live connectivity, semantic health, latency "
            "and readiness of the deployed scoring backend."
        ),
        eyebrow="RUNTIME HEALTH",
    )

    (
        overall_status,
        overall_tone,
        overall_description,
    ) = _system_status(
        diagnostics,
        metadata,
    )

    online = bool(
        diagnostics.get(
            "online"
        )
    )

    health_ok = bool(
        diagnostics.get(
            "health_ok"
        )
    )

    model_info_ok = bool(
        diagnostics.get(
            "model_info_ok"
        )
    )

    model_ready = (
        _api_model_ready(
            diagnostics
        )
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "System Status",
            overall_status,
            overall_description,
            tone=overall_tone,
        )

    with c2:

        metric_card(
            "API",
            (
                "ONLINE"
                if online
                else "OFFLINE"
            ),
            (
                "Backend reachable"
                if online
                else "Connection unavailable"
            ),
            tone=(
                "success"
                if online
                else "danger"
            ),
        )

    with c3:

        metric_card(
            "Health Endpoint",
            (
                "HEALTHY"
                if (
                    health_ok
                    and _health_reports_ready(
                        diagnostics
                    )
                )
                else "FAILED"
            ),
            _format_latency(
                diagnostics.get(
                    "health_latency_ms"
                )
            ),
            tone=(
                "success"
                if (
                    health_ok
                    and _health_reports_ready(
                        diagnostics
                    )
                )
                else "danger"
            ),
        )

    with c4:

        metric_card(
            "Model Contract",
            (
                "AVAILABLE"
                if model_info_ok
                else "FAILED"
            ),
            _format_latency(
                diagnostics.get(
                    "model_info_latency_ms"
                )
            ),
            tone=(
                "success"
                if model_info_ok
                else "danger"
            ),
        )

    st.write("")

    model = _nonempty_dict(
        diagnostics.get(
            "model"
        )
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        mini_metric(
            "Model",
            str(
                model.get(
                    "model_name",
                    "—",
                )
            ),
            helper="Runtime estimator",
            tone="info",
        )

    with c2:

        mini_metric(
            "Version",
            str(
                model.get(
                    "model_version",
                    "—",
                )
            ),
            helper="Deployment version",
        )

    with c3:

        mini_metric(
            "Inference",
            (
                "READY"
                if model_ready
                else "NOT READY"
            ),
            helper="End-to-end scoring",
            tone=(
                "success"
                if model_ready
                else "danger"
            ),
        )

    with c4:

        mini_metric(
            "Diagnostic Time",
            _format_latency(
                diagnostics.get(
                    "total_latency_ms"
                )
            ),
            helper="Health + contract",
        )

    st.write("")

    if model_ready:

        info_panel(
            "End-to-End Inference Ready",
            (
                "The API is reachable, /health reports a loaded "
                "model and /model-info exposes a valid deployment "
                "contract. Fraud-risk scoring is operational."
            ),
            tone="success",
        )

    elif online:

        info_panel(
            "Backend Reachable — Inference Not Ready",
            (
                "The backend responds, but the complete inference "
                "contract could not be validated."
            ),
            tone="danger",
        )

    else:

        info_panel(
            "Inference Service Unavailable",
            (
                "The Streamlit frontend cannot currently "
                "reach the FastAPI inference service."
            ),
            tone="danger",
        )

    errors = [
        (
            "Health",
            diagnostics.get(
                "health_error"
            ),
        ),
        (
            "Model info",
            diagnostics.get(
                "model_info_error"
            ),
        ),
    ]

    errors = [
        item
        for item in errors
        if item[1]
    ]

    if errors:

        with st.expander(
            "Runtime errors",
            expanded=False,
        ):

            for label, error in errors:

                st.markdown(
                    f"**{label}**"
                )

                st.code(
                    str(error),
                    language=None,
                )


# =============================================================================
# Model contract
# =============================================================================


def _render_model_contract(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """
    Render deployment identity and inference contract.
    """

    st.write("")
    st.write("")

    section_header(
        "Deployed Model Contract",
        (
            "Runtime model identity, source and transformed "
            "feature spaces, explainability and review policy."
        ),
        eyebrow="MODEL GOVERNANCE",
    )

    runtime = _nonempty_dict(
        diagnostics.get(
            "model"
        )
    )

    source = {
        **metadata,
        **runtime,
    }

    explainability = _nonempty_dict(
        runtime.get(
            "explainability"
        )
    )

    if not explainability:

        explainability = _nonempty_dict(
            metadata.get(
                "explainability"
            )
        )

    policy = _nonempty_dict(
        runtime.get(
            "review_policy"
        )
    )

    if not policy:

        policy = _nonempty_dict(
            metadata.get(
                "review_policy"
            )
        )

    fraction = _safe_float(
        policy.get(
            "fraction"
        )
    )

    left, right = st.columns(
        [
            1.3,
            1,
        ],
        gap="large",
    )

    with left:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Model Identity"
            )

            c1, c2 = st.columns(2)

            with c1:

                metric_card(
                    "Algorithm",
                    str(
                        source.get(
                            "model_name",
                            "—",
                        )
                    ),
                    "Runtime estimator",
                    tone="info",
                )

            with c2:

                metric_card(
                    "Version",
                    str(
                        source.get(
                            "model_version",
                            "—",
                        )
                    ),
                    "Frozen deployment version",
                )

            st.write("")

            c1, c2, c3 = st.columns(3)

            with c1:

                mini_metric(
                    "Source Features",
                    str(
                        source.get(
                            "feature_count",
                            "—",
                        )
                    ),
                    helper="Business input",
                )

            with c2:

                mini_metric(
                    "Transformed",
                    str(
                        source.get(
                            "transformed_feature_count",
                            "—",
                        )
                    ),
                    helper="Model feature space",
                    tone="info",
                )

            with c3:

                mini_metric(
                    "Target",
                    str(
                        source.get(
                            "target",
                            "—",
                        )
                    ),
                    helper="Prediction target",
                )

            st.write("")
            st.divider()

            key_value_row(
                "Probability method",
                str(
                    source.get(
                        "probability_method",
                        "—",
                    )
                ),
                monospace=True,
            )

            key_value_row(
                "Model artifact",
                str(
                    MODEL_PATH.name
                ),
                monospace=True,
            )

            key_value_row(
                "Preprocessor",
                str(
                    PREPROCESSOR_PATH.name
                ),
                monospace=True,
            )

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Operational Contract"
            )

            metric_card(
                "Review Capacity",
                (
                    f"{fraction:.0%}"
                    if fraction is not None
                    else "—"
                ),
                "Portfolio investigation policy",
                tone="info",
            )

            st.write("")

            key_value_row(
                "Policy",
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
                    "AVAILABLE"
                    if explainability.get(
                        "available"
                    )
                    is True
                    else "UNAVAILABLE"
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
                "Output space",
                str(
                    explainability.get(
                        "output_space",
                        "—",
                    )
                ),
                monospace=True,
            )

            key_value_row(
                "SHAP dimensions",
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
                    "Review capacity determines portfolio "
                    "prioritization. It is not an individual "
                    "fraud adjudication threshold."
                )
            )


# =============================================================================
# Contract consistency
# =============================================================================


def _render_contract_consistency(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """
    Verify frozen analytical metadata against runtime deployment.
    """

    st.write("")
    st.write("")

    section_header(
        "Contract Consistency",
        (
            "Verification that frozen analytical evidence "
            "belongs to the model currently served by the API."
        ),
        eyebrow="DEPLOYMENT VERIFICATION",
    )

    runtime = _nonempty_dict(
        diagnostics.get(
            "model"
        )
    )

    if not runtime:

        info_panel(
            "Runtime Contract Unavailable",
            (
                "The live model contract could not be "
                "retrieved, so deployment consistency "
                "cannot be verified."
            ),
            tone="warning",
        )

        return

    if not metadata:

        info_panel(
            "Frozen Metadata Unavailable",
            (
                "The runtime model is available, but local "
                "evaluation metadata is missing."
            ),
            tone="warning",
        )

        return

    (
        rows,
        comparisons,
        mismatches,
    ) = _contract_comparison(
        diagnostics,
        metadata,
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:

        metric_card(
            "Fields Checked",
            str(comparisons),
            "Shared deployment fields",
        )

    with c2:

        metric_card(
            "Matches",
            str(
                max(
                    comparisons
                    - mismatches,
                    0,
                )
            ),
            "Consistent values",
            tone="success",
        )

    with c3:

        metric_card(
            "Mismatches",
            str(mismatches),
            "Deployment discrepancies",
            tone=(
                "success"
                if mismatches == 0
                else "danger"
            ),
        )

    st.write("")

    if (
        comparisons > 0
        and mismatches == 0
    ):

        info_panel(
            "Deployment Contract Verified",
            (
                "The frozen metadata is consistent with "
                "the model currently exposed by the inference API."
            ),
            tone="success",
        )

    elif comparisons > 0:

        info_panel(
            "Deployment Contract Mismatch",
            (
                f"{mismatches} of {comparisons} checked fields "
                "differ. Frozen evaluation results must not be "
                "attributed to the current runtime model until "
                "the discrepancy is resolved."
            ),
            tone="danger",
        )

    else:

        info_panel(
            "Contract Verification Inconclusive",
            (
                "No shared contract fields are available "
                "for deployment verification."
            ),
            tone="warning",
        )

    if rows:

        with st.expander(
            "Contract comparison",
            expanded=False,
        ):

            st.dataframe(
                pd.DataFrame(
                    rows
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Field":
                        st.column_config.TextColumn(
                            "Field",
                            width="medium",
                        ),

                    "Local Artifact":
                        st.column_config.TextColumn(
                            "Frozen Metadata",
                            width="large",
                        ),

                    "Runtime API":
                        st.column_config.TextColumn(
                            "Runtime API",
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
# Artifact coverage
# =============================================================================


def _render_artifacts() -> None:
    """
    Render inference and analytical artifact readiness.
    """

    st.write("")
    st.write("")

    section_header(
        "Model Artifact Registry",
        (
            "Availability and traceability of inference, "
            "governance, explainability and evaluation assets."
        ),
        eyebrow="ARTIFACT COVERAGE",
    )

    inventory = _artifact_inventory()

    inference = [
        artifact
        for artifact in inventory
        if artifact.required_for_inference
    ]

    analytical = [
        artifact
        for artifact in inventory
        if not artifact.required_for_inference
    ]

    inference_available = sum(
        artifact.available
        for artifact in inference
    )

    analytical_available = sum(
        artifact.available
        for artifact in analytical
    )

    total_size = sum(
        artifact.size_bytes
        for artifact in inventory
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Inference Artifacts",
            (
                f"{inference_available}/"
                f"{len(inference)}"
            ),
            "Required for model loading",
            tone=(
                "success"
                if (
                    inference
                    and inference_available
                    == len(inference)
                )
                else "danger"
            ),
        )

    with c2:

        metric_card(
            "Analytical Assets",
            (
                f"{analytical_available}/"
                f"{len(analytical)}"
            ),
            "Explainability + evaluation",
            tone=(
                "success"
                if (
                    analytical
                    and analytical_available
                    == len(analytical)
                )
                else "warning"
            ),
        )

    with c3:

        metric_card(
            "Tracked Size",
            _format_bytes(
                total_size
            ),
            "Expected artifact inventory",
        )

    with c4:

        metric_card(
            "Artifacts Root",
            (
                "READY"
                if (
                    ARTIFACTS_ROOT.exists()
                    and ARTIFACTS_ROOT.is_dir()
                )
                else "MISSING"
            ),
            str(
                ARTIFACTS_ROOT
            ),
            tone=(
                "success"
                if ARTIFACTS_ROOT.exists()
                else "danger"
            ),
        )

    rows = []

    for artifact in inventory:

        rows.append(
            {
                "Category":
                    artifact.category,

                "Artifact":
                    artifact.name,

                "Required":
                    (
                        "YES"
                        if artifact.required_for_inference
                        else "NO"
                    ),

                "Status":
                    (
                        "AVAILABLE"
                        if artifact.available
                        else "MISSING"
                    ),

                "Size":
                    (
                        _format_bytes(
                            artifact.size_bytes
                        )
                        if artifact.available
                        else "—"
                    ),

                "Location":
                    str(
                        artifact.path
                    ),
            }
        )

    st.write("")

    with st.expander(
        "Artifact inventory",
        expanded=False,
    ):

        st.dataframe(
            pd.DataFrame(
                rows
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category":
                    st.column_config.TextColumn(
                        "Category",
                        width="medium",
                    ),

                "Artifact":
                    st.column_config.TextColumn(
                        "Artifact",
                        width="large",
                    ),

                "Required":
                    st.column_config.TextColumn(
                        "Inference Required",
                        width="small",
                    ),

                "Status":
                    st.column_config.TextColumn(
                        "Status",
                        width="small",
                    ),

                "Size":
                    st.column_config.TextColumn(
                        "Size",
                        width="small",
                    ),

                "Location":
                    st.column_config.TextColumn(
                        "Location",
                        width="large",
                    ),
            },
        )


# =============================================================================
# Architecture
# =============================================================================


def _render_architecture(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """
    Render readiness of principal application layers.
    """

    st.write("")
    st.write("")

    section_header(
        "System Architecture",
        (
            "Operational readiness of the complete "
            "fraud-risk decision-support stack."
        ),
        eyebrow="STACK READINESS",
    )

    frontend_ready = True

    api_ready = bool(
        diagnostics.get(
            "online"
        )
    )

    model_ready = (
        _api_model_ready(
            diagnostics
        )
    )

    explainability_ready = (
        _runtime_explainability_ready(
            diagnostics
        )
    )

    analytics_ready = (
        _analytics_ready(
            metadata
        )
    )

    stages = [
        (
            "01",
            "Frontend",
            frontend_ready,
            "Streamlit analyst workspace",
        ),
        (
            "02",
            "API",
            api_ready,
            "FastAPI inference gateway",
        ),
        (
            "03",
            "Model",
            model_ready,
            "Frozen XGBoost scoring stack",
        ),
        (
            "04",
            "TreeSHAP",
            explainability_ready,
            "Claim-level explainability",
        ),
        (
            "05",
            "Analytics",
            analytics_ready,
            "Evaluation and governance evidence",
        ),
    ]

    columns = st.columns(
        len(stages)
    )

    for column, (
        number,
        name,
        ready,
        description,
    ) in zip(
        columns,
        stages,
    ):

        with column:

            metric_card(
                f"Stage {number}",
                name,
                description,
                tone=(
                    "success"
                    if ready
                    else "danger"
                ),
            )

            st.caption(
                (
                    "READY"
                    if ready
                    else "NOT READY"
                )
            )

    st.write("")

    (
        status,
        tone,
        description,
    ) = _system_status(
        diagnostics,
        metadata,
    )

    info_panel(
        f"Stack Status — {status}",
        description,
        tone=tone,
    )


# =============================================================================
# Runtime environment
# =============================================================================


def _render_runtime(
    diagnostics: dict[str, Any],
) -> None:
    """
    Render technical frontend runtime information.
    """

    st.write("")
    st.write("")

    section_header(
        "Runtime Environment",
        (
            "Technical context of the current "
            "Streamlit application process."
        ),
        eyebrow="ENVIRONMENT",
    )

    api_url = os.getenv(
        "FRAUD_API_URL",
        "http://127.0.0.1:8000",
    )

    runtime_environment = os.getenv(
        "APP_ENV",
        os.getenv(
            "ENVIRONMENT",
            "development",
        ),
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Python",
            platform.python_version(),
            "Runtime version",
        )

    with c2:

        metric_card(
            "Operating System",
            platform.system(),
            platform.machine(),
        )

    with c3:

        metric_card(
            "Streamlit",
            st.__version__,
            "Frontend framework",
        )

    with c4:

        metric_card(
            "Environment",
            runtime_environment,
            "Application environment",
            tone="info",
        )

    st.write("")

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:

        mini_metric(
            "Health Latency",
            _format_latency(
                diagnostics.get(
                    "health_latency_ms"
                )
            ),
            helper="GET /health",
            tone="info",
        )

    with c2:

        mini_metric(
            "Contract Latency",
            _format_latency(
                diagnostics.get(
                    "model_info_latency_ms"
                )
            ),
            helper="GET /model-info",
            tone="info",
        )

    with c3:

        mini_metric(
            "Total Diagnostic",
            _format_latency(
                diagnostics.get(
                    "total_latency_ms"
                )
            ),
            helper="Sequential diagnostic",
        )

    st.write("")

    with st.container(
        border=True
    ):

        st.markdown(
            "### Runtime Configuration"
        )

        key_value_row(
            "Fraud API",
            api_url,
            monospace=True,
        )

        key_value_row(
            "Artifacts root",
            str(
                ARTIFACTS_ROOT
            ),
            monospace=True,
        )

        key_value_row(
            "Artifacts root status",
            _path_status(
                ARTIFACTS_ROOT
            ),
        )

        key_value_row(
            "Python executable",
            sys.executable,
            monospace=True,
        )

        key_value_row(
            "Process ID",
            str(
                os.getpid()
            ),
        )

        key_value_row(
            "Working directory",
            str(
                Path.cwd()
            ),
            monospace=True,
        )

        st.write("")

        st.caption(
            (
                "Status generated at "
                f"{_utc_now()}."
            )
        )


# =============================================================================
# Technical diagnostics
# =============================================================================


def _render_diagnostics(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """
    Render low-level contracts and diagnostic state.
    """

    st.write("")
    st.write("")

    section_header(
        "Technical Diagnostics",
        (
            "Raw runtime contracts and diagnostic state "
            "for verification and troubleshooting."
        ),
        eyebrow="DEBUG VIEW",
    )

    (
        health_tab,
        model_tab,
        metadata_tab,
        artifacts_tab,
        summary_tab,
    ) = st.tabs(
        [
            "Health",
            "Model Contract",
            "Metadata",
            "Artifacts",
            "Diagnostic Summary",
        ]
    )

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    with health_tab:

        health = _nonempty_dict(
            diagnostics.get(
                "health"
            )
        )

        if health:
            st.json(
                health
            )

        else:
            info_panel(
                "Health Payload Unavailable",
                (
                    "No valid health payload was returned "
                    "by the inference API."
                ),
                tone="warning",
            )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    with model_tab:

        model = _nonempty_dict(
            diagnostics.get(
                "model"
            )
        )

        if model:
            st.json(
                model
            )

        else:
            info_panel(
                "Model Contract Unavailable",
                (
                    "No valid model-info payload was "
                    "returned by the inference API."
                ),
                tone="warning",
            )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    with metadata_tab:

        if metadata:
            st.json(
                metadata
            )

        else:
            info_panel(
                "Frozen Metadata Unavailable",
                (
                    "The local model metadata artifact "
                    "could not be loaded."
                ),
                tone="warning",
            )

    # -------------------------------------------------------------------------
    # Artifacts
    # -------------------------------------------------------------------------

    with artifacts_tab:

        inventory = _artifact_inventory()

        payload = [
            {
                "name":
                    artifact.name,

                "category":
                    artifact.category,

                "required_for_inference":
                    artifact.required_for_inference,

                "available":
                    artifact.available,

                "size_bytes":
                    artifact.size_bytes,

                "path":
                    str(
                        artifact.path
                    ),
            }
            for artifact in inventory
        ]

        st.json(
            payload
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    with summary_tab:

        (
            status,
            _,
            status_description,
        ) = _system_status(
            diagnostics,
            metadata,
        )

        (
            analytical_available,
            analytical_total,
        ) = _analytics_coverage()

        (
            contract_rows,
            contract_checks,
            contract_mismatches,
        ) = _contract_comparison(
            diagnostics,
            metadata,
        )

        summary = {
            "generated_at":
                _utc_now(),

            "system_status":
                status,

            "system_status_description":
                status_description,

            "api_online":
                bool(
                    diagnostics.get(
                        "online"
                    )
                ),

            "health_endpoint_ok":
                bool(
                    diagnostics.get(
                        "health_ok"
                    )
                ),

            "semantic_health_ok":
                _health_reports_ready(
                    diagnostics
                ),

            "model_info_endpoint_ok":
                bool(
                    diagnostics.get(
                        "model_info_ok"
                    )
                ),

            "inference_ready":
                _api_model_ready(
                    diagnostics
                ),

            "runtime_explainability_ready":
                _runtime_explainability_ready(
                    diagnostics
                ),

            "local_inference_artifacts_ready":
                _inference_artifacts_ready(),

            "analytics_ready":
                _analytics_ready(
                    metadata
                ),

            "analytical_artifacts_available":
                analytical_available,

            "analytical_artifacts_expected":
                analytical_total,

            "contract_checks":
                contract_checks,

            "contract_mismatches":
                contract_mismatches,

            "contract_consistent":
                (
                    contract_mismatches == 0
                    if contract_checks > 0
                    else None
                ),

            "health_latency_ms":
                diagnostics.get(
                    "health_latency_ms"
                ),

            "model_info_latency_ms":
                diagnostics.get(
                    "model_info_latency_ms"
                ),

            "total_diagnostic_latency_ms":
                diagnostics.get(
                    "total_latency_ms"
                ),

            "metadata_path":
                str(
                    METADATA_PATH
                ),

            "artifacts_root":
                str(
                    ARTIFACTS_ROOT
                ),

            "fraud_api_url":
                os.getenv(
                    "FRAUD_API_URL",
                    "http://127.0.0.1:8000",
                ),

            "contract_comparison":
                contract_rows,
        }

        st.json(
            summary
        )


# =============================================================================
# Main page
# =============================================================================


def render(
    client,
) -> None:
    """
    Render the complete operational status and deployment
    verification workspace.
    """

    section_header(
        "System Status",
        (
            "Operational readiness, inference health, model "
            "contract verification, explainability availability "
            "and analytical artifact traceability."
        ),
    )

    # -------------------------------------------------------------------------
    # Evaluate runtime state exactly once per Streamlit render.
    # -------------------------------------------------------------------------

    diagnostics = _check_api(
        client
    )

    metadata = _read_metadata(
        str(
            METADATA_PATH
        )
    )

    # -------------------------------------------------------------------------
    # 1. Operational health
    # -------------------------------------------------------------------------

    _render_service_health(
        diagnostics,
        metadata,
    )

    # -------------------------------------------------------------------------
    # 2. Deployed model
    # -------------------------------------------------------------------------

    _render_model_contract(
        diagnostics,
        metadata,
    )

    # -------------------------------------------------------------------------
    # 3. Deployment verification
    # -------------------------------------------------------------------------

    _render_contract_consistency(
        diagnostics,
        metadata,
    )

    # -------------------------------------------------------------------------
    # 4. Artifact registry
    # -------------------------------------------------------------------------

    _render_artifacts()

    # -------------------------------------------------------------------------
    # 5. Stack readiness
    # -------------------------------------------------------------------------

    _render_architecture(
        diagnostics,
        metadata,
    )

    # -------------------------------------------------------------------------
    # 6. Runtime environment
    # -------------------------------------------------------------------------

    _render_runtime(
        diagnostics
    )

    # -------------------------------------------------------------------------
    # 7. Low-level diagnostics
    # -------------------------------------------------------------------------

    _render_diagnostics(
        diagnostics,
        metadata,
    )