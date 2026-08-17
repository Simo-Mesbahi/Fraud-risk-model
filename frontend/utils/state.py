from __future__ import annotations

from copy import deepcopy
from typing import Any

import streamlit as st


# =============================================================================
# State groups
# =============================================================================


CLAIM_STATE_KEYS = (
    "single_prediction",
    "single_score",
    "single_claim",
    "single_source",
    "single_explanation",
)


BATCH_STATE_KEYS = (
    "batch_results",
    "batch_input",
    "batch_source",
    "batch_metadata",
    "batch_selected_claim_id",
)


QUEUE_STATE_KEYS = (
    "queue_results",
    "queue_metadata",
    "queue_source_claims",
    "queue_source_name",
    "queue_selected_claim_id",
    "queue_human_decisions",
)


APPLICATION_STATE_KEYS = (
    "last_error",
)


# =============================================================================
# Application state contract
# =============================================================================


DEFAULT_STATE: dict[
    str,
    Any,
] = {

    # -------------------------------------------------------------------------
    # Claim Analysis
    # -------------------------------------------------------------------------

    "single_prediction":
        None,

    "single_score":
        None,

    "single_claim":
        None,

    "single_source":
        None,

    "single_explanation":
        None,

    # -------------------------------------------------------------------------
    # Portfolio Scoring
    # -------------------------------------------------------------------------

    "batch_results":
        None,

    "batch_input":
        None,

    "batch_source":
        None,

    "batch_metadata":
        None,

    "batch_selected_claim_id":
        None,

    # -------------------------------------------------------------------------
    # Investigation Queue
    # -------------------------------------------------------------------------

    "queue_results":
        None,

    "queue_metadata":
        None,

    "queue_source_claims":
        None,

    "queue_source_name":
        None,

    "queue_selected_claim_id":
        None,

    "queue_human_decisions":
        {},

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------

    "last_error":
        None,
}


# =============================================================================
# Internal helpers
# =============================================================================


def _reset_keys(
    keys: tuple[
        str,
        ...,
    ],
) -> None:
    """
    Reset a defined state group to its declared defaults.

    deepcopy is intentionally used for mutable defaults such as
    queue_human_decisions so Streamlit sessions never share the same object.
    """

    for key in keys:

        if key not in DEFAULT_STATE:
            raise KeyError(
                (
                    "Unknown session-state key "
                    f"'{key}'."
                )
            )

        st.session_state[
            key
        ] = deepcopy(
            DEFAULT_STATE[
                key
            ]
        )


def _ensure_contract() -> None:
    """
    Ensure every declared default exists in Streamlit session state.
    """

    for (
        key,
        default,
    ) in DEFAULT_STATE.items():

        if key not in st.session_state:

            st.session_state[
                key
            ] = deepcopy(
                default
            )


# =============================================================================
# Initialization
# =============================================================================


def initialize_state() -> None:
    """
    Initialize the complete application session-state contract.

    Safe to call on every Streamlit rerun.
    """

    _ensure_contract()


# =============================================================================
# Claim Analysis
# =============================================================================


def clear_single_claim() -> None:
    """
    Clear the active single-claim workflow.

    This resets both the model prediction and its local TreeSHAP explanation.
    """

    _reset_keys(
        CLAIM_STATE_KEYS
    )


# =============================================================================
# Portfolio Scoring
# =============================================================================


def clear_batch() -> None:
    """
    Clear the active portfolio-scoring workflow.
    """

    _reset_keys(
        BATCH_STATE_KEYS
    )


# =============================================================================
# Investigation Queue
# =============================================================================


def clear_queue() -> None:
    """
    Clear the active investigation queue and analyst decisions.
    """

    _reset_keys(
        QUEUE_STATE_KEYS
    )


# =============================================================================
# Error state
# =============================================================================


def set_last_error(
    error: Exception | str | None,
) -> None:
    """
    Persist the latest application-level error message.

    Passing None clears the stored error.
    """

    if error is None:

        st.session_state[
            "last_error"
        ] = None

        return

    message = str(
        error
    ).strip()

    st.session_state[
        "last_error"
    ] = (
        message
        or error.__class__.__name__
        if isinstance(
            error,
            Exception,
        )
        else message
    )


def clear_last_error() -> None:
    """
    Clear the application-level error state.
    """

    _reset_keys(
        APPLICATION_STATE_KEYS
    )


# =============================================================================
# Global reset
# =============================================================================


def clear_all_workflows() -> None:
    """
    Reset all user workflows while keeping the application session alive.
    """

    clear_single_claim()
    clear_batch()
    clear_queue()
    clear_last_error()


# =============================================================================
# Diagnostics
# =============================================================================


def state_snapshot() -> dict[
    str,
    Any,
]:
    """
    Return a defensive copy of the application-managed state.

    Intended for debugging and controlled diagnostics.
    """

    return {
        key:
            deepcopy(
                st.session_state.get(
                    key,
                    default,
                )
            )

        for (
            key,
            default,
        ) in DEFAULT_STATE.items()
    }