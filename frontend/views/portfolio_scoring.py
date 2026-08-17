from __future__ import annotations

import altair as alt
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
        "Portfolio Scoring",
        (
            "Score JSON, CSV or Parquet portfolios "
            "through the deployed model."
        ),
    )

    uploaded = st.file_uploader(
        "Upload portfolio",
        type=[
            "json",
            "csv",
            "parquet",
        ],
    )

    if uploaded is not None:

        try:
            claims = (
                read_uploaded_file(
                    uploaded
                )
            )

            st.info(
                f"{len(claims):,} claims detected."
            )

            if st.button(
                "Score Portfolio",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Scoring portfolio..."
                ):
                    response = (
                        client.score_batch(
                            claims
                        )
                    )

                frame = (
                    pd.DataFrame(
                        response[
                            "predictions"
                        ]
                    )
                    .sort_values(
                        "fraud_risk_score",
                        ascending=False,
                    )
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

                st.session_state.batch_results = (
                    frame
                )

                st.session_state.batch_input = (
                    claims
                )

        except Exception as exc:

            st.error(
                str(exc)
            )

    frame = (
        st.session_state.batch_results
    )

    if frame is None:

        empty_state(
            "No portfolio scored",
            (
                "Upload a JSON, CSV or Parquet portfolio "
                "to generate fraud-risk scores."
            ),
        )

        return

    st.write("")
    st.write("")

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:
        metric_card(
            "Claims",
            f"{len(frame):,}",
        )

    with c2:
        metric_card(
            "Mean Risk",
            f"{frame['fraud_risk_score'].mean():.2%}",
        )

    with c3:
        metric_card(
            "Maximum Risk",
            f"{frame['fraud_risk_score'].max():.2%}",
        )

    with c4:
        metric_card(
            "High / Critical",
            str(
                int(
                    (
                        frame[
                            "fraud_risk_score"
                        ]
                        >= 0.20
                    ).sum()
                )
            ),
        )

    st.write("")
    st.write("")

    section_header(
        "Risk Distribution"
    )

    distribution = (
        frame[
            "risk_tier"
        ]
        .value_counts()
        .rename_axis(
            "Risk Tier"
        )
        .reset_index(
            name="Claims"
        )
    )

    chart = (
        alt.Chart(
            distribution
        )
        .mark_bar(
            cornerRadiusTopLeft=6,
            cornerRadiusTopRight=6,
        )
        .encode(
            x=alt.X(
                "Risk Tier:N",
                sort=[
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL",
                ],
            ),
            y="Claims:Q",
            tooltip=[
                "Risk Tier:N",
                "Claims:Q",
            ],
        )
        .properties(
            height=320
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    search = st.text_input(
        "Search claim ID"
    )

    displayed = (
        frame.copy()
    )

    if search:

        displayed = (
            displayed[
                displayed[
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
        displayed,
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
        "Download Scores",
        data=frame.to_csv(
            index=False
        ),
        file_name="fraud_scores.csv",
        mime="text/csv",
        use_container_width=True,
    )