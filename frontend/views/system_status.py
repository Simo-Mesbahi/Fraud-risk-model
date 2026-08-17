from __future__ import annotations

import json
import os
import platform
import sys
import time

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
    metric_card,
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
    Resolve artifact locations across local development,
    Codespaces and Docker runtime.
    """

    candidates: list[Path] = []

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


def _find_artifact_root() -> Path:
    """
    Return the first existing artifact root.

    Falls back to the configured/default project path so that
    status diagnostics can still display the expected location.
    """

    for path in (
        _candidate_artifact_roots()
    ):

        if (
            path.exists()
            and path.is_dir()
        ):

            return path

    configured = (
        os.getenv(
            "ARTIFACTS_ROOT"
        )
    )

    if configured:

        return (
            Path(
                configured
            )
            .expanduser()
        )

    return (
        PROJECT_ROOT
        / "artifacts"
    )


ARTIFACTS_ROOT = (
    _find_artifact_root()
)


METADATA_PATH = (
    ARTIFACTS_ROOT
    / "metadata"
    / "health_fraud_model_metadata.json"
)


EXPLAINABILITY_DIR = (
    ARTIFACTS_ROOT
    / "explainability"
)


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


def _safe_get(
    payload: Any,
    key: str,
    default: Any = "—",
) -> Any:
    """
    Read a dictionary field safely.
    """

    if isinstance(
        payload,
        dict,
    ):

        return payload.get(
            key,
            default,
        )

    return default


def _format_bytes(
    size: int,
) -> str:
    """
    Format byte counts for human-readable diagnostics.
    """

    value = float(
        max(
            int(
                size
            ),
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


def _count_files(
    path: Path,
) -> int:
    """
    Count files recursively without failing on inaccessible entries.
    """

    if not path.exists():

        return 0

    if path.is_file():

        return 1

    count = 0

    try:

        for item in path.rglob(
            "*"
        ):

            try:

                if item.is_file():

                    count += 1

            except OSError:

                continue

    except OSError:

        return 0

    return count


def _directory_size(
    path: Path,
) -> int:
    """
    Compute total directory size defensively.
    """

    if not path.exists():

        return 0

    if path.is_file():

        try:

            return (
                path.stat()
                .st_size
            )

        except OSError:

            return 0

    total = 0

    try:

        for item in path.rglob(
            "*"
        ):

            try:

                if item.is_file():

                    total += (
                        item.stat()
                        .st_size
                    )

            except OSError:

                continue

    except OSError:

        return total

    return total


def _utc_now() -> str:
    """
    Return current UTC timestamp.
    """

    return (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )


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


# =============================================================================
# API diagnostics
# =============================================================================


def _check_api(
    client,
) -> dict[str, Any]:
    """
    Perform one health request and one model-contract request.

    Diagnostics intentionally distinguish connectivity from
    model-readiness.
    """

    result: dict[
        str,
        Any,
    ] = {
        "online":
            False,

        "health_ok":
            False,

        "model_info_ok":
            False,

        "health":
            {},

        "model":
            {},

        "health_latency_ms":
            None,

        "model_info_latency_ms":
            None,

        "total_latency_ms":
            None,

        "error":
            None,
    }

    total_start = (
        time.perf_counter()
    )

    # -------------------------------------------------------------------------
    # Health endpoint
    # -------------------------------------------------------------------------

    try:

        start = (
            time.perf_counter()
        )

        health = (
            client.health()
        )

        health_latency = (
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        result[
            "health_latency_ms"
        ] = health_latency

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

            result[
                "online"
            ] = True

        else:

            result[
                "error"
            ] = (
                "Health endpoint returned "
                "an invalid payload."
            )

            return result

    except Exception as exc:

        result[
            "error"
        ] = str(
            exc
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
    # Model contract
    # -------------------------------------------------------------------------

    try:

        start = (
            time.perf_counter()
        )

        model = (
            client.model_info()
        )

        model_latency = (
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        result[
            "model_info_latency_ms"
        ] = model_latency

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
                "error"
            ] = (
                "Model-info endpoint returned "
                "an invalid payload."
            )

    except Exception as exc:

        result[
            "error"
        ] = str(
            exc
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
# Readiness interpretation
# =============================================================================


def _api_model_ready(
    diagnostics: dict[str, Any],
) -> bool:
    """
    Determine whether the backend confirms a usable model.
    """

    if not diagnostics.get(
        "online",
        False,
    ):

        return False

    health = (
        diagnostics.get(
            "health"
        )
        or {}
    )

    explicit = (
        health.get(
            "model_loaded"
        )
    )

    if explicit is not None:

        return bool(
            explicit
        )

    model = (
        diagnostics.get(
            "model"
        )
        or {}
    )

    return bool(
        model.get(
            "model_name"
        )
        and model.get(
            "model_version"
        )
    )


def _analytics_ready(
    metadata: dict[str, Any],
) -> bool:
    """
    Determine whether analytical dashboard assets are present.
    """

    return bool(
        metadata
        and _count_files(
            EXPLAINABILITY_DIR
        )
        > 0
    )


def _system_status(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[
    str,
    str,
]:
    """
    Derive one global application status.
    """

    api_ready = (
        diagnostics.get(
            "online",
            False,
        )
    )

    model_ready = (
        _api_model_ready(
            diagnostics
        )
    )

    analytics_ready = (
        _analytics_ready(
            metadata
        )
    )

    if (
        api_ready
        and model_ready
        and analytics_ready
    ):

        return (
            "READY",
            "success",
        )

    if (
        api_ready
        and model_ready
    ):

        return (
            "DEGRADED",
            "warning",
        )

    return (
        "NOT READY",
        "danger",
    )


# =============================================================================
# Service health
# =============================================================================


def _render_service_health(
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """
    Render top-level operational health.
    """

    section_header(
        "Inference Service",
        (
            "Live connectivity, latency and readiness "
            "of the deployed scoring backend."
        ),
        eyebrow="RUNTIME HEALTH",
    )

    online = bool(
        diagnostics.get(
            "online"
        )
    )

    model_ready = (
        _api_model_ready(
            diagnostics
        )
    )

    model = (
        diagnostics.get(
            "model"
        )
        or {}
    )

    (
        overall_status,
        overall_tone,
    ) = (
        _system_status(
            diagnostics,
            metadata,
        )
    )

    health_latency = (
        diagnostics.get(
            "health_latency_ms"
        )
    )

    total_latency = (
        diagnostics.get(
            "total_latency_ms"
        )
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "System Status",
            overall_status,
            "Application readiness",
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
                "Inference reachable"
                if online
                else "Connection failed"
            ),
            tone=(
                "success"
                if online
                else "danger"
            ),
        )

    with c3:

        metric_card(
            "Health Latency",
            (
                f"{health_latency:.1f} ms"
                if health_latency
                is not None
                else "—"
            ),
            "GET /health",
            tone="info",
        )

    with c4:

        metric_card(
            "Total Diagnostic",
            (
                f"{total_latency:.1f} ms"
                if total_latency
                is not None
                else "—"
            ),
            "Health + model contract",
            tone="info",
        )

    st.write("")

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        metric_card(
            "Deployed Model",
            str(
                model.get(
                    "model_name",
                    "—",
                )
            ),
            (
                "Backend estimator"
            ),
            tone="info",
        )

    with c2:

        metric_card(
            "Model Loaded",
            (
                "YES"
                if model_ready
                else "NO"
            ),
            (
                f"Version "
                f"{model.get('model_version', '—')}"
            ),
            tone=(
                "success"
                if model_ready
                else "danger"
            ),
        )

    st.write("")

    if (
        online
        and model_ready
    ):

        info_panel(
            "End-to-End Inference Ready",
            (
                "The API is reachable and the backend "
                "reports a usable frozen model."
            ),
            tone="success",
        )

    elif online:

        info_panel(
            "API Reachable — Model Not Ready",
            (
                "The inference API responds, but model readiness "
                "could not be confirmed."
            ),
            tone="warning",
        )

    else:

        info_panel(
            "Inference Service Unavailable",
            (
                "The frontend cannot currently reach "
                "the inference API."
            ),
            tone="danger",
        )

        error = (
            diagnostics.get(
                "error"
            )
        )

        if error:

            with st.expander(
                "Connection error",
                expanded=False,
            ):

                st.code(
                    str(
                        error
                    ),
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
    Render live model identity and policy.
    """

    st.write("")
    st.write("")

    section_header(
        "Deployed Model Contract",
        (
            "Runtime model identity, feature contract "
            "and operational review policy."
        ),
        eyebrow="MODEL GOVERNANCE",
    )

    api_model = (
        diagnostics.get(
            "model"
        )
        or {}
    )

    # Runtime API has priority for deployment identity.
    source = {
        **metadata,
        **api_model,
    }

    model_name = (
        source.get(
            "model_name",
            "—",
        )
    )

    model_version = (
        source.get(
            "model_version",
            "—",
        )
    )

    target = (
        source.get(
            "target",
            "—",
        )
    )

    feature_count = (
        source.get(
            "feature_count",
            "—",
        )
    )

    probability_method = (
        source.get(
            "probability_method"
        )
        or "—"
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

        with st.container(
            border=True
        ):

            st.markdown(
                "### Model Identity"
            )

            c1, c2 = (
                st.columns(2)
            )

            with c1:

                metric_card(
                    "Algorithm",
                    str(
                        model_name
                    ),
                    "Runtime estimator",
                    tone="info",
                )

            with c2:

                metric_card(
                    "Version",
                    str(
                        model_version
                    ),
                    "Frozen deployment version",
                )

            st.write("")

            c1, c2 = (
                st.columns(2)
            )

            with c1:

                metric_card(
                    "Feature Count",
                    str(
                        feature_count
                    ),
                    "Business input features",
                )

            with c2:

                metric_card(
                    "Probability Method",
                    str(
                        probability_method
                    ),
                    "Model output contract",
                )

            st.write("")

            st.caption(
                "PREDICTION TARGET"
            )

            st.code(
                str(
                    target
                ),
                language=None,
            )

    with right:

        with st.container(
            border=True
        ):

            st.markdown(
                "### Operational Policy"
            )

            policy = (
                source.get(
                    "review_policy",
                    {},
                )
            )

            if not isinstance(
                policy,
                dict,
            ):

                policy = {}

            fraction = (
                _safe_float(
                    policy.get(
                        "fraction"
                    )
                )
            )

            metric_card(
                "Review Capacity",
                (
                    f"{fraction:.0%}"
                    if fraction
                    is not None
                    else "—"
                ),
                "Portfolio selection policy",
                tone="info",
            )

            st.write("")

            st.write(
                (
                    "**Policy type:** "
                    f"{policy.get('type', '—')}"
                )
            )

            st.caption(
                (
                    "Review capacity controls model-driven "
                    "portfolio prioritization. Final investigation "
                    "decisions remain human."
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
    Compare local frozen metadata with live API contract.
    """

    st.write("")
    st.write("")

    section_header(
        "Contract Consistency",
        (
            "Comparison between analytical metadata "
            "and the model currently served by the API."
        ),
        eyebrow="DEPLOYMENT VERIFICATION",
    )

    api_model = (
        diagnostics.get(
            "model"
        )
        or {}
    )

    if not api_model:

        info_panel(
            "Runtime Contract Unavailable",
            (
                "The live model contract could not be retrieved."
            ),
            tone="warning",
        )

        return

    if not metadata:

        info_panel(
            "Local Metadata Unavailable",
            (
                "The runtime model is available, but no local "
                "metadata exists for comparison."
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

    rows: list[
        dict[str, str]
    ] = []

    mismatch_count = 0
    comparison_count = 0

    for (
        key,
        label,
    ) in fields:

        local_value = (
            metadata.get(
                key
            )
        )

        runtime_value = (
            api_model.get(
                key
            )
        )

        if (
            local_value is None
            and runtime_value is None
        ):

            continue

        match = (
            local_value is not None
            and runtime_value is not None
            and str(
                local_value
            )
            == str(
                runtime_value
            )
        )

        comparison_count += 1

        if not match:

            mismatch_count += 1

        rows.append(
            {
                "Field":
                    label,

                "Local Artifact":
                    (
                        "—"
                        if local_value
                        is None
                        else str(
                            local_value
                        )
                    ),

                "Runtime API":
                    (
                        "—"
                        if runtime_value
                        is None
                        else str(
                            runtime_value
                        )
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
        comparison_count
        and mismatch_count == 0
    ):

        info_panel(
            "Contracts Consistent",
            (
                "The core local metadata matches the "
                "model currently exposed by the API."
            ),
            tone="success",
        )

    elif comparison_count:

        info_panel(
            "Contract Mismatch Detected",
            (
                f"{mismatch_count} of {comparison_count} "
                "checked field(s) differ between local "
                "metadata and the runtime API."
            ),
            tone="warning",
        )

    else:

        info_panel(
            "No Comparable Fields",
            (
                "The current artifacts do not expose enough "
                "shared fields for comparison."
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
                            "Local Artifact",
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
# Frontend analytical assets
# =============================================================================


def _render_frontend_assets(
    metadata: dict[str, Any],
) -> None:
    """
    Render analytical artifact coverage.
    """

    st.write("")
    st.write("")

    section_header(
        "Frontend Analytical Assets",
        (
            "Artifacts available to the dashboard for "
            "evaluation reporting and explainability."
        ),
        eyebrow="ARTIFACT COVERAGE",
    )

    metadata_ready = bool(
        metadata
    )

    metadata_file_ready = (
        METADATA_PATH.exists()
        and METADATA_PATH.is_file()
    )

    explainability_files = (
        _count_files(
            EXPLAINABILITY_DIR
        )
    )

    explainability_ready = (
        explainability_files
        > 0
    )

    explainability_size = (
        _directory_size(
            EXPLAINABILITY_DIR
        )
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "Metadata",
            (
                "AVAILABLE"
                if metadata_ready
                else "MISSING"
            ),
            "Frozen evaluation contract",
            tone=(
                "success"
                if metadata_ready
                else "danger"
            ),
        )

    with c2:

        metric_card(
            "Explainability",
            (
                "AVAILABLE"
                if explainability_ready
                else "MISSING"
            ),
            (
                f"{explainability_files:,} file(s)"
            ),
            tone=(
                "success"
                if explainability_ready
                else "danger"
            ),
        )

    with c3:

        metric_card(
            "Explainability Size",
            _format_bytes(
                explainability_size
            ),
            "Mounted frontend assets",
        )

    with c4:

        metric_card(
            "Artifacts Root",
            (
                "READY"
                if ARTIFACTS_ROOT.exists()
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

    st.write("")

    status_rows = [
        {
            "Component":
                "Model Metadata",

            "Owner":
                "Frontend",

            "Status":
                (
                    "AVAILABLE"
                    if metadata_file_ready
                    else "MISSING"
                ),

            "Files":
                (
                    1
                    if metadata_file_ready
                    else 0
                ),

            "Location":
                str(
                    METADATA_PATH
                ),
        },
        {
            "Component":
                "Explainability",

            "Owner":
                "Frontend",

            "Status":
                (
                    "AVAILABLE"
                    if explainability_ready
                    else "MISSING"
                ),

            "Files":
                explainability_files,

            "Location":
                str(
                    EXPLAINABILITY_DIR
                ),
        },
    ]

    st.dataframe(
        pd.DataFrame(
            status_rows
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Component":
                st.column_config.TextColumn(
                    "Component",
                    width="medium",
                ),

            "Owner":
                st.column_config.TextColumn(
                    "Owner",
                    width="small",
                ),

            "Status":
                st.column_config.TextColumn(
                    "Status",
                    width="small",
                ),

            "Files":
                st.column_config.NumberColumn(
                    "Files",
                    format="%d",
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
# Architecture readiness
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
            "Readiness of the principal application "
            "layers and their operational responsibilities."
        ),
        eyebrow="STACK READINESS",
    )

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

    metadata_ready = bool(
        metadata
    )

    explainability_ready = (
        _count_files(
            EXPLAINABILITY_DIR
        )
        > 0
    )

    stages = [
        (
            "01",
            "Frontend",
            True,
            "Streamlit investigation console",
        ),
        (
            "02",
            "API",
            api_ready,
            "FastAPI inference service",
        ),
        (
            "03",
            "Model",
            model_ready,
            "Frozen fraud-risk estimator",
        ),
        (
            "04",
            "Analytics",
            (
                metadata_ready
                and explainability_ready
            ),
            "Evaluation and explainability assets",
        ),
    ]

    columns = (
        st.columns(
            4
        )
    )

    for (
        column,
        stage,
    ) in zip(
        columns,
        stages,
    ):

        (
            number,
            name,
            ready,
            description,
        ) = stage

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

    operational_ready = (
        api_ready
        and model_ready
    )

    analytical_ready = (
        metadata_ready
        and explainability_ready
    )

    st.write("")

    if (
        operational_ready
        and analytical_ready
    ):

        info_panel(
            "Complete Application Stack Ready",
            (
                "Frontend, API, model and analytical "
                "assets are all available."
            ),
            tone="success",
        )

    elif operational_ready:

        info_panel(
            "Inference Ready — Analytics Degraded",
            (
                "Live scoring is operational, but one or more "
                "analytical dashboard assets are unavailable."
            ),
            tone="warning",
        )

    else:

        info_panel(
            "Inference Stack Not Ready",
            (
                "The application cannot currently guarantee "
                "end-to-end fraud-risk inference."
            ),
            tone="danger",
        )


# =============================================================================
# Runtime
# =============================================================================


def _render_runtime(
    diagnostics: dict[str, Any],
) -> None:
    """
    Render technical frontend/runtime information.
    """

    st.write("")
    st.write("")

    section_header(
        "Runtime Environment",
        (
            "Technical context of the current "
            "Streamlit frontend process."
        ),
        eyebrow="ENVIRONMENT",
    )

    api_url = (
        os.getenv(
            "FRAUD_API_URL",
            "http://127.0.0.1:8000",
        )
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
            "Process ID",
            str(
                os.getpid()
            ),
            "Frontend process",
        )

    st.write("")

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        metric_card(
            "Health Request",
            (
                f"{diagnostics['health_latency_ms']:.1f} ms"
                if diagnostics.get(
                    "health_latency_ms"
                )
                is not None
                else "—"
            ),
            "API diagnostic latency",
            tone="info",
        )

    with c2:

        metric_card(
            "Model Contract Request",
            (
                f"{diagnostics['model_info_latency_ms']:.1f} ms"
                if diagnostics.get(
                    "model_info_latency_ms"
                )
                is not None
                else "—"
            ),
            "API contract latency",
            tone="info",
        )

    st.write("")

    with st.container(
        border=True
    ):

        st.markdown(
            "### Runtime Configuration"
        )

        st.caption(
            "FRAUD API"
        )

        st.code(
            api_url,
            language=None,
        )

        st.caption(
            "ARTIFACTS ROOT"
        )

        st.code(
            str(
                ARTIFACTS_ROOT
            ),
            language=None,
        )

        st.caption(
            "PYTHON EXECUTABLE"
        )

        st.code(
            sys.executable,
            language=None,
        )

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
    Render raw technical payloads for debugging.
    """

    st.write("")
    st.write("")

    section_header(
        "Technical Diagnostics",
        (
            "Raw runtime contracts for verification "
            "and troubleshooting."
        ),
        eyebrow="DEBUG VIEW",
    )

    health_tab, model_tab, metadata_tab, summary_tab = (
        st.tabs(
            [
                "Health",
                "Model Contract",
                "Metadata",
                "Diagnostic Summary",
            ]
        )
    )

    with health_tab:

        payload = (
            diagnostics.get(
                "health"
            )
        )

        if payload:

            st.json(
                payload
            )

        else:

            info_panel(
                "Health Payload Unavailable",
                (
                    "No health payload was returned "
                    "by the inference API."
                ),
                tone="warning",
            )

    with model_tab:

        payload = (
            diagnostics.get(
                "model"
            )
        )

        if payload:

            st.json(
                payload
            )

        else:

            info_panel(
                "Model Contract Unavailable",
                (
                    "No model-info payload was returned "
                    "by the inference API."
                ),
                tone="warning",
            )

    with metadata_tab:

        if metadata:

            st.json(
                metadata
            )

        else:

            info_panel(
                "Frontend Metadata Unavailable",
                (
                    "The frozen metadata artifact could not "
                    "be loaded from the frontend runtime."
                ),
                tone="warning",
            )

    with summary_tab:

        summary = {
            "generated_at":
                _utc_now(),

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

            "model_info_endpoint_ok":
                bool(
                    diagnostics.get(
                        "model_info_ok"
                    )
                ),

            "model_ready":
                _api_model_ready(
                    diagnostics
                ),

            "analytics_ready":
                _analytics_ready(
                    metadata
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

            "metadata_available":
                METADATA_PATH.exists(),

            "explainability_directory":
                str(
                    EXPLAINABILITY_DIR
                ),

            "explainability_file_count":
                _count_files(
                    EXPLAINABILITY_DIR
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
        }

        st.json(
            summary
        )


# =============================================================================
# Main
# =============================================================================


def render(
    client,
) -> None:
    """
    Render complete runtime / readiness dashboard.
    """

    section_header(
        "System Status",
        (
            "Operational readiness, API health, model contract, "
            "deployment consistency and analytical asset availability."
        ),
    )

    # -------------------------------------------------------------------------
    # Diagnostics are evaluated once per page render.
    # -------------------------------------------------------------------------

    diagnostics = (
        _check_api(
            client
        )
    )

    metadata = (
        _read_metadata(
            str(
                METADATA_PATH
            )
        )
    )

    # -------------------------------------------------------------------------
    # Runtime health
    # -------------------------------------------------------------------------

    _render_service_health(
        diagnostics,
        metadata,
    )

    # -------------------------------------------------------------------------
    # Model contract
    # -------------------------------------------------------------------------

    _render_model_contract(
        diagnostics,
        metadata,
    )

    # -------------------------------------------------------------------------
    # Runtime ↔ artifact verification
    # -------------------------------------------------------------------------

    _render_contract_consistency(
        diagnostics,
        metadata,
    )

    # -------------------------------------------------------------------------
    # Analytical artifacts
    # -------------------------------------------------------------------------

    _render_frontend_assets(
        metadata
    )

    # -------------------------------------------------------------------------
    # Architecture
    # -------------------------------------------------------------------------

    _render_architecture(
        diagnostics,
        metadata,
    )

    # -------------------------------------------------------------------------
    # Runtime environment
    # -------------------------------------------------------------------------

    _render_runtime(
        diagnostics
    )

    # -------------------------------------------------------------------------
    # Raw diagnostics
    # -------------------------------------------------------------------------

    _render_diagnostics(
        diagnostics,
        metadata,
    )