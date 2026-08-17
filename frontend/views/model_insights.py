from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from components import (
    section_header,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def render(
    client,
) -> None:

    section_header(
        "Model Insights",
        (
            "Explainability, fraud mechanisms "
            "and observed failure modes."
        ),
    )

    shap_global = (
        PROJECT_ROOT
        / "artifacts"
        / "explainability"
        / "figures"
        / "01_shap_global_bar.png"
    )

    shap_beeswarm = (
        PROJECT_ROOT
        / "artifacts"
        / "explainability"
        / "figures"
        / "02_shap_beeswarm.png"
    )

    left, right = (
        st.columns(2)
    )

    with left:

        st.markdown(
            "#### Global SHAP Importance"
        )

        if shap_global.exists():
            st.image(
                str(
                    shap_global
                ),
                use_container_width=True,
            )

    with right:

        st.markdown(
            "#### SHAP Distribution"
        )

        if shap_beeswarm.exists():
            st.image(
                str(
                    shap_beeswarm
                ),
                use_container_width=True,
            )

    st.write("")
    st.write("")

    section_header(
        "Recall by Fraud Mechanism"
    )

    mechanism = pd.DataFrame(
        {
            "Mechanism": [
                "Customer-provider pattern",
                "Repeated service",
                "Mixed pattern",
                "Frequency abuse",
                "Provider abnormality",
                "Amount inflation",
            ],

            "Recall @ 3%": [
                0.8030,
                0.6897,
                0.6176,
                0.5775,
                0.5352,
                0.0800,
            ],
        }
    )

    st.dataframe(
        mechanism,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Recall @ 3%":
                st.column_config.ProgressColumn(
                    "Recall @ 3%",
                    min_value=0,
                    max_value=1,
                ),
        },
    )

    st.warning(
        (
            "Primary observed weakness: "
            "amount-inflation fraud."
        )
    )