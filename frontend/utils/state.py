from __future__ import annotations

from copy import deepcopy
from typing import Any

import streamlit as st


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
# Initialization
# =============================================================================


def initialize_state() -> None:

    for (
        key,
        default,
    ) in DEFAULT_STATE.items():

        if key not in (
            st.session_state
        ):

            st.session_state[
                key
            ] = deepcopy(
                default
            )


# =============================================================================
# Claim Analysis
# =============================================================================


def clear_single_claim() -> None:

    st.session_state.single_prediction = None
    st.session_state.single_score = None
    st.session_state.single_claim = None
    st.session_state.single_source = None


# =============================================================================
# Portfolio
# =============================================================================


def clear_batch() -> None:

    st.session_state.batch_results = None
    st.session_state.batch_input = None
    st.session_state.batch_source = None
    st.session_state.batch_metadata = None
    st.session_state.batch_selected_claim_id = None


# =============================================================================
# Queue
# =============================================================================


def clear_queue() -> None:

    st.session_state.queue_results = None
    st.session_state.queue_metadata = None
    st.session_state.queue_source_claims = None
    st.session_state.queue_source_name = None
    st.session_state.queue_selected_claim_id = None
    st.session_state.queue_human_decisions = {}


# =============================================================================
# Global reset
# =============================================================================


def clear_all_workflows() -> None:

    clear_single_claim()
    clear_batch()
    clear_queue()

    st.session_state.last_error = None