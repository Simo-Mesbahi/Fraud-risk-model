from __future__ import annotations

import json

import streamlit as st

from components import (
    risk_badge,
    risk_gauge,
    section_header,
)

from utils.data import (
    get_demo_claim,
    load_demo_claims,
)

from utils.formatting import (
    risk_tier,
)


def save_prediction(
    claim,
    response,
    source,
) -> None:
    prediction = (
        response[
            "prediction"
        ]
    )

    st.session_state.single_prediction = (
        prediction
    )

    st.session_state.single_score = float(
        prediction[
            "fraud_risk_score"
        ]
    )

    st.session_state.single_claim = (
        claim
    )

    st.session_state.single_source = (
        source
    )


def render(
    client,
) -> None:

    section_header(
        "Claim Analysis",
        (
            "Score an individual claim "
            "through the frozen inference pipeline."
        ),
    )

    demo_tab, json_tab = (
        st.tabs(
            [
                "Quick Demo",
                "Advanced JSON",
            ]
        )
    )

    with demo_tab:

        try:
            demo_claims = (
                load_demo_claims()
            )

            index = st.selectbox(
                "Demo claim",
                options=range(
                    min(
                        len(
                            demo_claims
                        ),
                        100,
                    )
                ),
                format_func=lambda i:
                    str(
                        demo_claims.iloc[
                            i
                        ][
                            "claim_id"
                        ]
                    ),
            )

            claim = (
                get_demo_claim(
                    index
                )
            )

            with st.expander(
                "View claim payload"
            ):
                st.json(
                    claim
                )

            if st.button(
                "Analyze Demo Claim",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Building features and scoring claim..."
                ):
                    response = (
                        client.score_claim(
                            claim
                        )
                    )

                save_prediction(
                    claim,
                    response,
                    "Quick Demo",
                )

        except Exception as exc:

            st.error(
                str(exc)
            )

    with json_tab:

        raw = st.text_area(
            "Complete claim JSON",
            height=400,
        )

        if st.button(
            "Analyze JSON",
            use_container_width=True,
            key="json_score",
        ):

            try:
                claim = json.loads(
                    raw
                )

                with st.spinner(
                    "Scoring claim..."
                ):
                    response = (
                        client.score_claim(
                            claim
                        )
                    )

                save_prediction(
                    claim,
                    response,
                    "Advanced JSON",
                )

            except Exception as exc:

                st.error(
                    str(exc)
                )

    prediction = (
        st.session_state.single_prediction
    )

    score = (
        st.session_state.single_score
    )

    if (
        prediction is None
        or score is None
    ):
        return

    st.write("")
    st.write("")

    section_header(
        "Risk Assessment"
    )

    left, right = (
        st.columns(
            [
                1,
                1.6,
            ],
            gap="large",
        )
    )

    with left:
        risk_gauge(
            score
        )

        risk_badge(
            score
        )

    with right:

        with st.container(
            border=True
        ):

            c1, c2 = (
                st.columns(2)
            )

            with c1:
                st.metric(
                    "Fraud Risk",
                    f"{score:.2%}",
                )

            with c2:
                st.metric(
                    "Risk Tier",
                    risk_tier(
                        score
                    ),
                )

            st.divider()

            st.write(
                "**Claim ID:** "
                f"`{prediction.get('claim_id')}`"
            )

            st.write(
                "**Model:** "
                f"{prediction['model_name']}"
            )

            st.write(
                "**Version:** "
                f"{prediction['model_version']}"
            )

            st.write(
                "**Source:** "
                f"{st.session_state.single_source}"
            )

            if score >= 0.50:
                st.error(
                    "Priority investigator review."
                )

            elif score >= 0.20:
                st.warning(
                    "Elevated risk — review recommended."
                )

            elif score >= 0.05:
                st.info(
                    "Moderate risk — capacity-dependent review."
                )

            else:
                st.success(
                    "Low individual model risk."
                )

            st.caption(
                (
                    "The risk score supports investigation "
                    "prioritization and does not prove fraud."
                )
            )