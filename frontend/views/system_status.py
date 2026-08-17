from __future__ import annotations

import time

import streamlit as st

from components import (
    section_header,
)


def render(
    client,
) -> None:

    section_header(
        "System Status",
        (
            "Inference health and deployed "
            "model contract."
        ),
    )

    try:
        start = time.perf_counter()

        health = (
            client.health()
        )

        latency_ms = (
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        model = (
            client.model_info()
        )

        c1, c2, c3 = (
            st.columns(3)
        )

        with c1:
            st.metric(
                "API",
                "ONLINE",
            )

        with c2:
            st.metric(
                "Latency",
                f"{latency_ms:.1f} ms",
            )

        with c3:
            st.metric(
                "Model",
                model[
                    "model_name"
                ],
            )

        st.write("")
        st.json(
            model
        )

    except Exception as exc:

        st.error(
            str(exc)
        )