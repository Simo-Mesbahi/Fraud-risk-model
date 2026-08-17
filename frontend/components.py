from __future__ import annotations

import html

import streamlit as st

from utils.formatting import (
    risk_color,
    risk_tier,
)


def section_header(
    title: str,
    subtitle: str | None = None,
) -> None:

    st.markdown(
        f"### {title}"
    )

    if subtitle:
        st.caption(
            subtitle
        )


def metric_card(
    label: str,
    value: str,
    helper: str = "",
) -> None:

    safe_label = (
        html.escape(
            str(label)
        )
    )

    safe_value = (
        html.escape(
            str(value)
        )
    )

    safe_helper = (
        html.escape(
            str(helper)
        )
    )

    markup = (
        '<div class="glass-card">'
        '<div class="metric-label">'
        f"{safe_label}"
        "</div>"
        '<div class="metric-value">'
        f"{safe_value}"
        "</div>"
        '<div class="metric-helper">'
        f"{safe_helper}"
        "</div>"
        "</div>"
    )

    st.markdown(
        markup,
        unsafe_allow_html=True,
    )


def status_badge(
    online: bool,
) -> None:

    if online:

        st.markdown(
            (
                '<span class="status-ok">'
                "● API ONLINE"
                "</span>"
            ),
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            (
                '<span class="status-offline">'
                "● API OFFLINE"
                "</span>"
            ),
            unsafe_allow_html=True,
        )


def risk_badge(
    score: float,
) -> None:

    tier = (
        risk_tier(
            score
        )
    )

    color = (
        risk_color(
            score
        )
    )

    markup = (
        '<div style="'
        "display:flex;"
        "justify-content:center;"
        "margin-top:.65rem;"
        '">'

        '<span style="'
        f"color:{color};"
        f"border:1px solid {color}55;"
        f"background:{color}18;"
        "padding:.48rem .95rem;"
        "border-radius:999px;"
        "font-size:.78rem;"
        "font-weight:850;"
        "letter-spacing:.13em;"
        '">'

        f"{tier}"

        "</span>"
        "</div>"
    )

    st.markdown(
        markup,
        unsafe_allow_html=True,
    )


def risk_gauge(
    score: float,
) -> None:

    score = min(
        max(
            float(score),
            0.0,
        ),
        1.0,
    )

    tier = (
        risk_tier(
            score
        )
    )

    color = (
        risk_color(
            score
        )
    )

    degrees = (
        score
        * 360
    )

    markup = (
        '<div style="'
        "width:100%;"
        "display:flex;"
        "align-items:center;"
        "justify-content:center;"
        "padding:1.2rem 0;"
        '">'

        '<div style="'
        "width:235px;"
        "height:235px;"
        "border-radius:50%;"
        "display:flex;"
        "align-items:center;"
        "justify-content:center;"
        "background:"
        f"conic-gradient("
        f"{color} 0deg,"
        f"{color} {degrees:.1f}deg,"
        "rgba(255,255,255,.065) "
        f"{degrees:.1f}deg,"
        "rgba(255,255,255,.065) 360deg"
        ");"
        f"box-shadow:0 0 55px {color}25;"
        '">'

        '<div style="'
        "width:180px;"
        "height:180px;"
        "display:flex;"
        "flex-direction:column;"
        "align-items:center;"
        "justify-content:center;"
        "border-radius:50%;"
        "background:"
        "linear-gradient("
        "145deg,"
        "#0A1020,"
        "#070B14"
        ");"
        "border:"
        "1px solid "
        "rgba(255,255,255,.085);"
        '">'

        '<span style="'
        "color:#8794AA;"
        "font-size:.7rem;"
        "font-weight:700;"
        "letter-spacing:.13em;"
        '">'
        "FRAUD RISK"
        "</span>"

        '<strong style="'
        "margin-top:.15rem;"
        "color:#FFFFFF;"
        "font-size:2.35rem;"
        "font-weight:850;"
        "letter-spacing:-.04em;"
        '">'

        f"{score:.1%}"

        "</strong>"

        '<span style="'
        f"color:{color};"
        "margin-top:.15rem;"
        "font-size:.8rem;"
        "font-weight:850;"
        "letter-spacing:.13em;"
        '">'

        f"{tier}"

        "</span>"

        "</div>"
        "</div>"
        "</div>"
    )

    st.markdown(
        markup,
        unsafe_allow_html=True,
    )


def empty_state(
    title: str,
    message: str,
) -> None:

    safe_title = (
        html.escape(
            str(title)
        )
    )

    safe_message = (
        html.escape(
            str(message)
        )
    )

    markup = (
        '<div class="empty-state">'
        '<div class="empty-title">'
        f"{safe_title}"
        "</div>"
        '<div class="empty-message">'
        f"{safe_message}"
        "</div>"
        "</div>"
    )

    st.markdown(
        markup,
        unsafe_allow_html=True,
    )


def page_intro(
    title: str,
    subtitle: str,
) -> None:

    st.markdown(
        f"## {title}"
    )

    st.caption(
        subtitle
    )

    st.write("")


def human_review_notice() -> None:

    st.caption(
        (
            "Model scores support investigation "
            "prioritization only. "
            "They do not establish that fraud occurred."
        )
    )