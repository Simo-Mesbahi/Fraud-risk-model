from __future__ import annotations

import os
from collections.abc import Callable
from typing import Final

import streamlit as st

from api_client import FraudAPIClient
from components import status_badge
from styles import CUSTOM_CSS
from utils.state import initialize_state

from views.claim_analysis import (
    render as render_claim_analysis,
)
from views.investigation_queue import (
    render as render_investigation_queue,
)
from views.model_insights import (
    render as render_model_insights,
)
from views.overview import (
    render as render_overview,
)
from views.portfolio_scoring import (
    render as render_portfolio_scoring,
)
from views.system_status import (
    render as render_system_status,
)


# =============================================================================
# Application constants
# =============================================================================


APP_TITLE: Final[str] = (
    "Health Fraud Intelligence"
)

APP_SUBTITLE: Final[str] = (
    "AI-assisted health insurance "
    "investigation prioritization"
)

DEFAULT_API_URL: Final[str] = (
    "http://127.0.0.1:8000"
)

API_URL: Final[str] = os.getenv(
    "FRAUD_API_URL",
    DEFAULT_API_URL,
)


# =============================================================================
# Page configuration
# =============================================================================


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
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
    Build and cache a single API client for the Streamlit session.

    The API endpoint is injected through FRAUD_API_URL when running
    inside Docker and defaults to localhost during local development.
    """

    return FraudAPIClient(
        base_url=base_url
    )


client = get_api_client(
    API_URL
)


# =============================================================================
# Navigation configuration
# =============================================================================


PageRenderer = Callable[
    [FraudAPIClient],
    None,
]


PAGE_ROUTES: Final[
    dict[
        str,
        PageRenderer,
    ]
] = {
    "Overview":
        render_overview,

    "Claim Analysis":
        render_claim_analysis,

    "Portfolio Scoring":
        render_portfolio_scoring,

    "Investigation Queue":
        render_investigation_queue,

    "Model Insights":
        render_model_insights,

    "System Status":
        render_system_status,
}


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar() -> str:
    """
    Render the application navigation and live model status.

    Returns
    -------
    str
        Selected page name.
    """

    with st.sidebar:

        # ---------------------------------------------------------------------
        # Brand
        # ---------------------------------------------------------------------

        st.markdown(
            "## ◇ Fraud Intelligence"
        )

        st.caption(
            "AI-Assisted Investigation Console"
        )

        st.write("")

        # ---------------------------------------------------------------------
        # Navigation
        # ---------------------------------------------------------------------

        page = st.radio(
            "Navigation",
            options=list(
                PAGE_ROUTES
            ),
            index=0,
            label_visibility="collapsed",
            key="main_navigation",
        )

        st.write("")
        st.divider()

        # ---------------------------------------------------------------------
        # Live inference status
        # ---------------------------------------------------------------------

        try:
            health = client.health()

            status_badge(
                True
            )

            st.caption(
                (
                    f"{health['model_name']} "
                    f"v{health['model_version']}"
                )
            )

            st.caption(
                "Inference service available"
            )

        except Exception:
            status_badge(
                False
            )

            st.caption(
                "Inference service unavailable"
            )

        # ---------------------------------------------------------------------
        # Product context
        # ---------------------------------------------------------------------

        st.write("")
        st.divider()

        st.caption(
            "Human-in-the-loop fraud prioritization"
        )

        st.caption(
            "Synthetic health-insurance environment"
        )

        st.caption(
            "Decision-support prototype"
        )

        return page


# =============================================================================
# Main header
# =============================================================================


def render_header() -> None:
    """
    Render the global product header.
    """

    st.markdown(
        (
            '<div class="dashboard-title">'
            f"{APP_TITLE}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="dashboard-subtitle">'
            f"{APP_SUBTITLE}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# Application router
# =============================================================================


def render_page(
    page: str,
) -> None:
    """
    Render the selected application view.
    """

    renderer = PAGE_ROUTES.get(
        page
    )

    if renderer is None:
        st.error(
            "Unknown application page."
        )
        return

    try:
        renderer(
            client
        )

    except Exception as exc:
        st.error(
            (
                "An unexpected error occurred "
                "while rendering this page."
            )
        )

        with st.expander(
            "Technical details"
        ):
            st.exception(
                exc
            )


# =============================================================================
# Application entry point
# =============================================================================


def main() -> None:
    selected_page = (
        render_sidebar()
    )

    render_header()

    render_page(
        selected_page
    )


if __name__ == "__main__":
    main()