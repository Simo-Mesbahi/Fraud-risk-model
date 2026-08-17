from __future__ import annotations

import html
import os

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import streamlit as st

from frontend.api_client import (
    FraudAPIClient,
    FraudAPIError,
)

from frontend.components import (
    status_badge,
)

from frontend.styles import (
    CUSTOM_CSS,
)

from frontend.utils.state import (
    initialize_state,
)

from frontend.views.claim_analysis import (
    render as render_claim_analysis,
)

from frontend.views.investigation_queue import (
    render as render_investigation_queue,
)

from frontend.views.model_insights import (
    render as render_model_insights,
)

from frontend.views.overview import (
    render as render_overview,
)

from frontend.views.portfolio_scoring import (
    render as render_portfolio_scoring,
)

from frontend.views.system_status import (
    render as render_system_status,
)


# =============================================================================
# Product configuration
# =============================================================================


APP_TITLE: Final[str] = (
    "Health Fraud Intelligence"
)

APP_SHORT_TITLE: Final[str] = (
    "Fraud Intelligence"
)

APP_SUBTITLE: Final[str] = (
    "AI-assisted health insurance "
    "investigation prioritization"
)

APP_DESCRIPTION: Final[str] = (
    "Human-in-the-loop fraud risk scoring "
    "and investigation decision support."
)

APP_VERSION: Final[str] = (
    "5.0"
)

DEFAULT_API_URL: Final[str] = (
    "http://127.0.0.1:8000"
)

API_URL: Final[str] = (
    os.getenv(
        "FRAUD_API_URL",
        DEFAULT_API_URL,
    )
    .strip()
    .rstrip("/")
)

ENVIRONMENT: Final[str] = (
    os.getenv(
        "APP_ENV",
        "development",
    )
    .strip()
    .lower()
)


# =============================================================================
# Streamlit configuration
# =============================================================================


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":
            None,

        "Report a bug":
            None,

        "About":
            (
                f"{APP_TITLE}\n\n"
                f"{APP_DESCRIPTION}\n\n"
                f"Frontend v{APP_VERSION}"
            ),
    },
)


# =============================================================================
# Global styling
# =============================================================================


st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# =============================================================================
# Session state
# =============================================================================


initialize_state()


# =============================================================================
# API client
# =============================================================================


@st.cache_resource
def get_api_client(
    base_url: str,
) -> FraudAPIClient:
    """
    Build one persistent HTTP client.

    The client is reused across Streamlit reruns and keeps
    HTTP connections alive for lower inference latency.
    """

    return FraudAPIClient(
        base_url=base_url,
        connect_timeout=3.0,
        read_timeout=60.0,
        retry_total=2,
    )


client = (
    get_api_client(
        API_URL
    )
)


# =============================================================================
# Runtime probe
# =============================================================================


@st.cache_data(
    ttl=8,
    show_spinner=False,
)
def get_runtime_status(
    base_url: str,
) -> dict[str, Any]:
    """
    Perform a lightweight independent runtime probe.

    The probe intentionally uses a short-lived client so sidebar
    diagnostics do not interfere with scoring requests.

    API connectivity and model readiness are evaluated separately.
    """

    probe = FraudAPIClient(
        base_url=base_url,
        connect_timeout=1.5,
        read_timeout=3.0,
        retry_total=0,
    )

    result: dict[
        str,
        Any,
    ] = {
        "api_online":
            False,

        "model_ready":
            False,

        "health":
            {},

        "model":
            {},

        "error":
            None,
    }

    try:

        health = (
            probe.health()
        )

        result[
            "health"
        ] = (
            health
            if isinstance(
                health,
                dict,
            )
            else {}
        )

        result[
            "api_online"
        ] = True

        # ---------------------------------------------------------------------
        # Prefer explicit model_loaded from /health when available.
        # ---------------------------------------------------------------------

        explicit_ready = (
            result[
                "health"
            ]
            .get(
                "model_loaded"
            )
        )

        if explicit_ready is not None:

            result[
                "model_ready"
            ] = bool(
                explicit_ready
            )

        # ---------------------------------------------------------------------
        # Otherwise confirm readiness through /model-info.
        # ---------------------------------------------------------------------

        if explicit_ready is None:

            try:

                model = (
                    probe.model_info()
                )

                if isinstance(
                    model,
                    dict,
                ):

                    result[
                        "model"
                    ] = model

                    result[
                        "model_ready"
                    ] = bool(
                        model.get(
                            "model_name"
                        )
                        and model.get(
                            "model_version"
                        )
                    )

            except Exception as exc:

                result[
                    "error"
                ] = str(
                    exc
                )

        else:

            # model-info remains useful for sidebar identity,
            # but failure does not redefine explicit health readiness.

            try:

                model = (
                    probe.model_info()
                )

                if isinstance(
                    model,
                    dict,
                ):

                    result[
                        "model"
                    ] = model

            except Exception:
                pass

    except Exception as exc:

        result[
            "error"
        ] = str(
            exc
        )

    finally:

        probe.close()

    return result


# =============================================================================
# Page registry
# =============================================================================


PageRenderer = Callable[
    [FraudAPIClient],
    None,
]


@dataclass(
    frozen=True,
    slots=True,
)
class PageDefinition:
    """
    Definition of one application workspace.
    """

    name: str

    short_description: str

    renderer: PageRenderer

    category: str


PAGES: Final[
    tuple[
        PageDefinition,
        ...,
    ]
] = (
    PageDefinition(
        name="Overview",
        short_description=(
            "Executive risk and model overview"
        ),
        renderer=render_overview,
        category="MONITOR",
    ),

    PageDefinition(
        name="Claim Analysis",
        short_description=(
            "Individual claim risk assessment"
        ),
        renderer=render_claim_analysis,
        category="ANALYZE",
    ),

    PageDefinition(
        name="Portfolio Scoring",
        short_description=(
            "Batch portfolio risk scoring"
        ),
        renderer=render_portfolio_scoring,
        category="SCORE",
    ),

    PageDefinition(
        name="Investigation Queue",
        short_description=(
            "Prioritized human-review workflow"
        ),
        renderer=render_investigation_queue,
        category="INVESTIGATE",
    ),

    PageDefinition(
        name="Model Insights",
        short_description=(
            "Performance and explainability"
        ),
        renderer=render_model_insights,
        category="UNDERSTAND",
    ),

    PageDefinition(
        name="System Status",
        short_description=(
            "Inference and artifact readiness"
        ),
        renderer=render_system_status,
        category="OPERATE",
    ),
)


PAGE_BY_NAME: Final[
    dict[
        str,
        PageDefinition,
    ]
] = {
    page.name:
        page

    for page in PAGES
}


# =============================================================================
# Sidebar helpers
# =============================================================================


def _sidebar_model_identity(
    runtime: dict[str, Any],
) -> tuple[
    str,
    str,
]:
    """
    Resolve model identity from model-info first,
    then fall back to health payload.
    """

    model = (
        runtime.get(
            "model"
        )
        or {}
    )

    health = (
        runtime.get(
            "health"
        )
        or {}
    )

    model_name = (
        model.get(
            "model_name"
        )
        or health.get(
            "model_name"
        )
        or "Model"
    )

    model_version = (
        model.get(
            "model_version"
        )
        or health.get(
            "model_version"
        )
        or "—"
    )

    return (
        str(
            model_name
        ),
        str(
            model_version
        ),
    )


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar() -> str:
    """
    Render global navigation and compact runtime status.
    """

    with st.sidebar:

        # ---------------------------------------------------------------------
        # Product identity
        # ---------------------------------------------------------------------

        st.markdown(
            (
                '<div class="sidebar-brand">'
                '<div class="sidebar-brand-title">'
                "◇ Fraud Intelligence"
                "</div>"
                '<div class="sidebar-brand-subtitle">'
                "AI-Assisted Investigation Console"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.write("")

        # ---------------------------------------------------------------------
        # Navigation
        # ---------------------------------------------------------------------

        selected_page = (
            st.radio(
                "Navigation",
                options=[
                    page.name
                    for page
                    in PAGES
                ],
                index=0,
                label_visibility="collapsed",
                key="main_navigation",
            )
        )

        current_page = (
            PAGE_BY_NAME[
                selected_page
            ]
        )

        st.markdown(
            (
                '<div class="sidebar-page-context">'
                '<span class="sidebar-page-category">'
                f"{html.escape(current_page.category)}"
                "</span>"
                '<span class="sidebar-page-description">'
                f"{html.escape(current_page.short_description)}"
                "</span>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.write("")
        st.divider()

        # ---------------------------------------------------------------------
        # Runtime status
        # ---------------------------------------------------------------------

        runtime = (
            get_runtime_status(
                API_URL
            )
        )

        api_online = bool(
            runtime.get(
                "api_online"
            )
        )

        model_ready = bool(
            runtime.get(
                "model_ready"
            )
        )

        (
            model_name,
            model_version,
        ) = (
            _sidebar_model_identity(
                runtime
            )
        )

        if (
            api_online
            and model_ready
        ):

            status_badge(
                True,
                label="INFERENCE READY",
            )

            st.caption(
                (
                    f"{model_name} "
                    f"v{model_version}"
                )
            )

            st.caption(
                "Scoring service operational"
            )

        elif api_online:

            status_badge(
                False,
                label="MODEL NOT READY",
            )

            st.caption(
                "API reachable"
            )

            st.caption(
                "Model readiness unavailable"
            )

        else:

            status_badge(
                False,
                label="API OFFLINE",
            )

            st.caption(
                "Inference service unavailable"
            )

        # ---------------------------------------------------------------------
        # Product context
        # ---------------------------------------------------------------------

        st.write("")
        st.divider()

        st.markdown(
            (
                '<div class="sidebar-context-block">'
                '<div class="sidebar-context-line">'
                "Human-in-the-loop prioritization"
                "</div>"
                '<div class="sidebar-context-line">'
                "Synthetic health-insurance environment"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.write("")

        safe_environment = (
            html.escape(
                ENVIRONMENT.title()
            )
        )

        safe_version = (
            html.escape(
                APP_VERSION
            )
        )

        st.markdown(
            (
                '<div class="sidebar-product-meta">'
                f"Frontend v{safe_version}"
                "<br>"
                f"{safe_environment}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        return selected_page


# =============================================================================
# Global header
# =============================================================================


def render_header(
    selected_page: str,
) -> None:
    """
    Render persistent product and workspace context.
    """

    page_definition = (
        PAGE_BY_NAME[
            selected_page
        ]
    )

    safe_title = (
        html.escape(
            APP_TITLE
        )
    )

    safe_subtitle = (
        html.escape(
            APP_SUBTITLE
        )
    )

    safe_page = (
        html.escape(
            page_definition.name
        )
    )

    safe_description = (
        html.escape(
            page_definition
            .short_description
        )
    )

    safe_category = (
        html.escape(
            page_definition.category
        )
    )

    st.markdown(
        (
            '<div class="dashboard-header">'
            '<div class="dashboard-title">'
            f"{safe_title}"
            "</div>"
            '<div class="dashboard-subtitle">'
            f"{safe_subtitle}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="page-context-strip">'
            '<span class="page-context-category">'
            f"{safe_category}"
            "</span>"
            '<span class="page-context-divider">'
            "•"
            "</span>"
            '<strong>'
            f"{safe_page}"
            "</strong>"
            '<span class="page-context-divider">'
            "•"
            "</span>"
            '<span>'
            f"{safe_description}"
            "</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.write("")


# =============================================================================
# Router
# =============================================================================


def render_page(
    page_name: str,
) -> None:
    """
    Render one registered application workspace.

    Page-level failures are isolated so a failure in one view does
    not make the complete investigation console unusable.
    """

    definition = (
        PAGE_BY_NAME.get(
            page_name
        )
    )

    if definition is None:

        st.error(
            (
                "The requested application "
                "workspace does not exist."
            )
        )

        return

    try:

        definition.renderer(
            client
        )

    except FraudAPIError as exc:

        st.error(
            (
                "The inference service could not complete "
                "the requested operation."
            )
        )

        with st.expander(
            "API details",
            expanded=False,
        ):

            st.code(
                str(
                    exc
                ),
                language=None,
            )

    except Exception as exc:

        st.error(
            (
                "An unexpected application error occurred. "
                "The remaining workspaces are still available."
            )
        )

        with st.expander(
            "Technical details",
            expanded=False,
        ):

            st.exception(
                exc
            )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """
    Application entry point.
    """

    selected_page = (
        render_sidebar()
    )

    render_header(
        selected_page
    )

    render_page(
        selected_page
    )


if __name__ == "__main__":
    main()