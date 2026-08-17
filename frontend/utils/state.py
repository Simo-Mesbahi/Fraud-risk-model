from __future__ import annotations

import streamlit as st


DEFAULT_STATE = {
    "single_prediction": None,
    "single_score": None,
    "single_claim": None,
    "single_source": None,

    "batch_results": None,
    "batch_input": None,

    "queue_results": None,
    "queue_metadata": None,

    "last_error": None,
}


def initialize_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_single_claim() -> None:
    st.session_state.single_prediction = None
    st.session_state.single_score = None
    st.session_state.single_claim = None
    st.session_state.single_source = None


def clear_batch() -> None:
    st.session_state.batch_results = None
    st.session_state.batch_input = None


def clear_queue() -> None:
    st.session_state.queue_results = None
    st.session_state.queue_metadata = None