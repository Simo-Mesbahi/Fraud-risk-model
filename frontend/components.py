from __future__ import annotations

import html
import textwrap

from typing import (
    Any,
    Literal,
)

import streamlit as st

from utils.formatting import (
    risk_color,
    risk_tier,
)


# =============================================================================
# Types
# =============================================================================


Tone = Literal[
    "neutral",
    "info",
    "success",
    "warning",
    "danger",
]


# =============================================================================
# Design tokens
# =============================================================================


TONE_COLORS: dict[
    Tone,
    str,
] = {
    "neutral":
        "#A6B2C7",

    "info":
        "#63A7FF",

    "success":
        "#61E7A6",

    "warning":
        "#FFD166",

    "danger":
        "#FF5C7A",
}


# =============================================================================
# Internal helpers
# =============================================================================


def _escape(
    value: Any,
) -> str:
    """
    Escape dynamic values before injecting them into HTML.
    """

    return html.escape(
        str(
            value
        ),
        quote=True,
    )


def _tone_color(
    tone: Tone,
) -> str:
    """
    Return the semantic design-system color associated with a tone.
    """

    return TONE_COLORS.get(
        tone,
        TONE_COLORS[
            "neutral"
        ],
    )


def _clamp_score(
    score: float,
) -> float:
    """
    Clamp a probability-like score to the closed interval [0, 1].
    """

    return min(
        max(
            float(
                score
            ),
            0.0,
        ),
        1.0,
    )


def _render_html(
    markup: str,
) -> None:
    """
    Render HTML through Streamlit.

    Multi-line markup is normalized before rendering so that readable
    Python HTML strings cannot accidentally become Markdown code blocks.
    """

    clean_markup = (
        textwrap.dedent(
            markup
        )
        .strip()
    )

    st.markdown(
        clean_markup,
        unsafe_allow_html=True,
    )


def _render_compact_html(
    markup: str,
) -> None:
    """
    Render compact single-line HTML.

    This renderer is used for components such as the risk gauge where
    Markdown whitespace interpretation must be eliminated completely.
    """

    clean_markup = (
        " ".join(
            markup
            .replace(
                "\n",
                " ",
            )
            .split()
        )
    )

    st.markdown(
        clean_markup,
        unsafe_allow_html=True,
    )


# =============================================================================
# Page headings
# =============================================================================


def section_header(
    title: str,
    subtitle: str | None = None,
    *,
    eyebrow: str | None = None,
) -> None:
    """
    Render a reusable section heading.
    """

    safe_title = (
        _escape(
            title
        )
    )

    safe_subtitle = (
        _escape(
            subtitle
        )
        if subtitle
        else ""
    )

    safe_eyebrow = (
        _escape(
            eyebrow
        )
        if eyebrow
        else ""
    )

    eyebrow_markup = (
        (
            '<div class="section-eyebrow">'
            f"{safe_eyebrow}"
            "</div>"
        )
        if safe_eyebrow
        else ""
    )

    subtitle_markup = (
        (
            '<div class="section-subtitle">'
            f"{safe_subtitle}"
            "</div>"
        )
        if safe_subtitle
        else ""
    )

    markup = (
        '<div class="section-heading">'
        f"{eyebrow_markup}"
        '<div class="section-title">'
        f"{safe_title}"
        "</div>"
        f"{subtitle_markup}"
        "</div>"
    )

    _render_html(
        markup
    )


def page_intro(
    title: str,
    subtitle: str,
) -> None:
    """
    Render a larger page-level introduction.
    """

    markup = (
        '<div class="page-intro">'
        '<div class="page-intro-title">'
        f"{_escape(title)}"
        "</div>"
        '<div class="page-intro-subtitle">'
        f"{_escape(subtitle)}"
        "</div>"
        "</div>"
    )

    _render_html(
        markup
    )


# =============================================================================
# Metric cards
# =============================================================================


def metric_card(
    label: str,
    value: str,
    helper: str = "",
    *,
    tone: Tone = "neutral",
) -> None:
    """
    Render a reusable premium KPI card.
    """

    safe_label = (
        _escape(
            label
        )
    )

    safe_value = (
        _escape(
            value
        )
    )

    safe_helper = (
        _escape(
            helper
        )
    )

    color = (
        _tone_color(
            tone
        )
    )

    helper_markup = (
        (
            '<div class="metric-helper">'
            f"{safe_helper}"
            "</div>"
        )
        if safe_helper
        else (
            '<div class="metric-helper">'
            "&nbsp;"
            "</div>"
        )
    )

    markup = (
        '<div class="glass-card metric-card-pro">'
        '<div class="metric-accent" '
        f'style="background:{color};">'
        "</div>"
        '<div class="metric-label">'
        f"{safe_label}"
        "</div>"
        '<div class="metric-value">'
        f"{safe_value}"
        "</div>"
        f"{helper_markup}"
        "</div>"
    )

    _render_html(
        markup
    )


# =============================================================================
# Generic badge
# =============================================================================


def badge(
    label: str,
    *,
    tone: Tone = "neutral",
    dot: bool = False,
) -> None:
    """
    Render a compact semantic badge.
    """

    color = (
        _tone_color(
            tone
        )
    )

    prefix = (
        "● "
        if dot
        else ""
    )

    markup = (
        '<span class="app-badge" '
        f'style="--badge-color:{color};">'
        f"{prefix}{_escape(label)}"
        "</span>"
    )

    _render_compact_html(
        markup
    )


# =============================================================================
# API status
# =============================================================================


def status_badge(
    online: bool,
    *,
    label: str | None = None,
) -> None:
    """
    Render the compact API availability indicator.
    """

    badge(
        (
            label
            or (
                "API ONLINE"
                if online
                else "API OFFLINE"
            )
        ),
        tone=(
            "success"
            if online
            else "danger"
        ),
        dot=True,
    )


# =============================================================================
# Risk badge
# =============================================================================


def risk_badge(
    score: float,
) -> None:
    """
    Render the categorical risk tier below the fraud-risk gauge.
    """

    score = (
        _clamp_score(
            score
        )
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

    markup = (
        '<div class="risk-badge-wrapper">'
        '<span class="risk-badge" '
        f'style="--risk-color:{color};">'
        f"{_escape(tier)}"
        "</span>"
        "</div>"
    )

    _render_compact_html(
        markup
    )


# =============================================================================
# Risk gauge
# =============================================================================


def risk_gauge(
    score: float,
) -> None:
    """
    Render a responsive circular fraud-risk gauge.

    The markup is deliberately built as compact HTML rather than a
    formatted multi-line block. This avoids Streamlit/Markdown treating
    indented tags such as <strong> and <span> as code blocks.
    """

    score = (
        _clamp_score(
            score
        )
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
        * 360.0
    )

    safe_tier = (
        _escape(
            tier
        )
    )

    aria_label = (
        _escape(
            (
                f"Fraud risk {score:.1%}, "
                f"risk tier {tier}"
            )
        )
    )

    markup = (
        '<div class="risk-gauge-wrapper">'
        '<div '
        'class="risk-gauge-ring" '
        'role="img" '
        f'aria-label="{aria_label}" '
        f'style="--risk-color:{color};'
        f'--risk-degrees:{degrees:.2f}deg;">'
        '<div class="risk-gauge-inner">'
        '<span class="risk-gauge-label">'
        "FRAUD RISK"
        "</span>"
        '<strong class="risk-gauge-value">'
        f"{score:.1%}"
        "</strong>"
        '<span '
        'class="risk-gauge-tier" '
        f'style="color:{color};">'
        f"{safe_tier}"
        "</span>"
        "</div>"
        "</div>"
        "</div>"
    )

    _render_compact_html(
        markup
    )


# =============================================================================
# Empty state
# =============================================================================


def empty_state(
    title: str,
    message: str,
    *,
    hint: str | None = None,
) -> None:
    """
    Render a visually consistent empty-state panel.
    """

    hint_markup = (
        (
            '<div class="empty-hint">'
            f"{_escape(hint)}"
            "</div>"
        )
        if hint
        else ""
    )

    markup = (
        '<div class="empty-state">'
        '<div class="empty-title">'
        f"{_escape(title)}"
        "</div>"
        '<div class="empty-message">'
        f"{_escape(message)}"
        "</div>"
        f"{hint_markup}"
        "</div>"
    )

    _render_html(
        markup
    )


# =============================================================================
# Information panels
# =============================================================================


def info_panel(
    title: str,
    message: str,
    *,
    tone: Tone = "info",
) -> None:
    """
    Render a semantic information panel.
    """

    color = (
        _tone_color(
            tone
        )
    )

    markup = (
        '<div class="info-panel" '
        f'style="--panel-color:{color};">'
        '<div class="info-panel-title">'
        f"{_escape(title)}"
        "</div>"
        '<div class="info-panel-message">'
        f"{_escape(message)}"
        "</div>"
        "</div>"
    )

    _render_html(
        markup
    )


# =============================================================================
# Human review / governance notice
# =============================================================================


def human_review_notice() -> None:
    """
    Render the application's standard human-in-the-loop notice.
    """

    info_panel(
        "HUMAN DECISION REQUIRED",
        (
            "Model scores support investigation "
            "prioritization only. They do not establish "
            "that fraud occurred. Final adjudication remains "
            "a human responsibility."
        ),
        tone="info",
    )