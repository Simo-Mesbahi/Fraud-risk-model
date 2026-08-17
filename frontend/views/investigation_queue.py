from __future__ import annotations

import pandas as pd
import streamlit as st

from components import (
    empty_state,
    metric_card,
    section_header,
)

from utils.data import (
    read_uploaded_file,
)

from utils.formatting import (
    risk_tier,
)


def render(
    client,
) -> None:

    section_header(
        "Investigation Queue",
        (
            "Convert model scores into "
            "an operational human-review queue."
        ),
    )

    capacity = (
        st.slider(
            "Investigation capacity",
            1,
            25,
            3,
        )
        / 100
    )

    uploaded = (
        st.file_uploader(
            "Upload claims",
            type=[
                "json",
                "csv",
                "parquet",
            ],
            key="queue_file",
        )
    )

    if uploaded is not None:

        try:
            claims = (
                read_uploaded_file(
                    uploaded
                )
            )

            if st.button(
                "Build Queue",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Prioritizing claims..."
                ):
                    response = (
                        client.top_review(
                            claims,
                            capacity,
                        )
                    )

                frame = pd.DataFrame(
                    response[
                        "predictions"
                    ]
                )

                frame[
                    "risk_tier"
                ] = (
                    frame[
                        "fraud_risk_score"
                    ]
                    .apply(
                        risk_tier
                    )
                )

                st.session_state.queue_results = (
                    frame
                )

                st.session_state.queue_metadata = {
                    "total_claims":
                        response[
                            "total_claims"
                        ],

                    "selected_claims":
                        response[
                            "selected_claims"
                        ],

                    "capacity":
                        capacity,
                }

        except Exception as exc:

            st.error(
                str(exc)
            )

    frame = (
        st.session_state.queue_results
    )

    metadata = (
        st.session_state.queue_metadata
    )

    if (
        frame is None
        or metadata is None
    ):

        empty_state(
            "No investigation queue",
            (
                "Upload a portfolio and generate "
                "a prioritized human-review queue."
            ),
        )

        return

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:
        metric_card(
            "Portfolio",
            f"{metadata['total_claims']:,}",
        )

    with c2:
        metric_card(
            "Selected",
            f"{metadata['selected_claims']:,}",
        )

    with c3:
        metric_card(
            "Capacity",
            f"{metadata['capacity']:.0%}",
        )

    with c4:
        metric_card(
            "Mean Risk",
            (
                f"{frame['fraud_risk_score'].mean():.2%}"
            ),
        )

    st.write("")
    st.write("")

    search = st.text_input(
        "Search claim ID",
        key="queue_search",
    )

    display = (
        frame.copy()
    )

    if search:
        display = (
            display[
                display[
                    "claim_id"
                ]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False,
                )
            ]
        )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "fraud_risk_score":
                st.column_config.ProgressColumn(
                    "Fraud Risk",
                    min_value=0,
                    max_value=1,
                ),
        },
    )

    st.download_button(
        "Download Investigation Queue",
        data=frame.to_csv(
            index=False
        ),
        file_name=(
            "investigation_queue.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )