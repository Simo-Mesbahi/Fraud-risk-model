from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from components import (
    metric_card,
    section_header,
)

from utils.formatting import (
    format_review_policy,
)


def render(
    client,
) -> None:

    section_header(
        "Executive Overview",
        (
            "Out-of-time model performance "
            "and live inference status."
        ),
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:
        metric_card(
            "Average Precision",
            "0.5520",
            "2026 test",
        )

    with c2:
        metric_card(
            "ROC-AUC",
            "0.8518",
            "Discrimination",
        )

    with c3:
        metric_card(
            "Recall @ 3%",
            "53.79%",
            "Fraud captured",
        )

    with c4:
        metric_card(
            "Lift @ 3%",
            "17.90×",
            "vs random review",
        )

    st.write("")

    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:
        metric_card(
            "Precision @ 3%",
            "51.64%",
            "Investigation yield",
        )

    with c2:
        metric_card(
            "Fraud Amount Captured",
            "55.15%",
            "At 3% capacity",
        )

    with c3:
        metric_card(
            "Test Fraud Prevalence",
            "2.885%",
            "409 fraud claims",
        )

    st.write("")
    st.write("")

    left, right = (
        st.columns(
            [
                1.7,
                1,
            ],
            gap="large",
        )
    )

    with left:

        section_header(
            "Investigation Capacity",
            (
                "How fraud capture evolves "
                "with investigation workload."
            ),
        )

        data = pd.DataFrame(
            {
                "capacity": [
                    0.005,
                    0.010,
                    0.020,
                    0.030,
                    0.050,
                    0.075,
                    0.100,
                    0.150,
                ],

                "Recall": [
                    0.1711,
                    0.3276,
                    0.4743,
                    0.5379,
                    0.5990,
                    0.6455,
                    0.6748,
                    0.7188,
                ],

                "Fraud Amount Capture": [
                    0.1693,
                    0.3283,
                    0.4866,
                    0.5515,
                    0.6080,
                    0.6874,
                    0.7244,
                    0.7669,
                ],
            }
        )

        chart_data = (
            data.melt(
                id_vars=[
                    "capacity"
                ],
                var_name="Metric",
                value_name="Value",
            )
        )

        chart = (
            alt.Chart(
                chart_data
            )
            .mark_line(
                point=True,
                strokeWidth=3,
            )
            .encode(
                x=alt.X(
                    "capacity:Q",
                    title="Investigation Capacity",
                    axis=alt.Axis(
                        format=".1%",
                    ),
                ),

                y=alt.Y(
                    "Value:Q",
                    title="Capture",
                    axis=alt.Axis(
                        format=".0%",
                    ),
                ),

                color=alt.Color(
                    "Metric:N",
                    title=None,
                ),

                tooltip=[
                    alt.Tooltip(
                        "capacity:Q",
                        format=".1%",
                    ),
                    "Metric:N",
                    alt.Tooltip(
                        "Value:Q",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=390
            )
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

    with right:

        section_header(
            "Live System",
            "Current production-style inference service.",
        )

        try:
            health = (
                client.health()
            )

            model = (
                client.model_info()
            )

            with st.container(
                border=True
            ):

                st.success(
                    "API ONLINE"
                )

                st.metric(
                    "Model",
                    model[
                        "model_name"
                    ],
                )

                st.metric(
                    "Version",
                    model[
                        "model_version"
                    ],
                )

                st.write(
                    "**Target**"
                )

                st.code(
                    model[
                        "target"
                    ]
                )

                st.write(
                    "**Business features:** "
                    f"{model['feature_count']}"
                )

                st.write(
                    "**Policy:** "
                    + format_review_policy(
                        model.get(
                            "review_policy"
                        )
                    )
                )

        except Exception as exc:

            st.error(
                str(exc)
            )

    st.write("")
    st.write("")

    section_header(
        "Main Risk Drivers",
        (
            "Business-level features identified "
            "by the final SHAP analysis."
        ),
    )

    drivers = pd.DataFrame(
        {
            "Rank": [
                1,
                2,
                3,
                4,
                5,
            ],

            "Feature": [
                "claim_to_service_median_ratio",
                "days_since_customer_previous_claim",
                "reimbursement_ratio",
                "provider_claims_30d",
                "submission_month",
            ],
        }
    )

    st.dataframe(
        drivers,
        use_container_width=True,
        hide_index=True,
    )