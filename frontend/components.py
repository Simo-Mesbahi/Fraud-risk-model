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


TONE_ICONS: dict[
    Tone,
    str,
] = {
    "neutral":
        "●",

    "info":
        "●",

    "success":
        "●",

    "warning":
        "●",

    "danger":
        "●",
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
    Clamp a probability-like score to [0, 1].
    """

    try:

        score = float(
            score
        )

    except (
        TypeError,
        ValueError,
    ):
        score = 0.0

    return min(
        max(
            score,
            0.0,
        ),
        1.0,
    )


def _render_html(
    markup: str,
) -> None:
    """
    Render HTML safely through Streamlit.

    Multi-line markup is normalized before rendering so readable Python
    strings cannot accidentally become Markdown code blocks.
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
    Render whitespace-normalized HTML.

    This is useful for components where Markdown whitespace interpretation
    must be completely eliminated.
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

    safe_title = _escape(
        title
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
    *,
    eyebrow: str | None = None,
) -> None:
    """
    Render a larger page-level introduction.
    """

    eyebrow_markup = (
        (
            '<div class="page-intro-eyebrow">'
            f"{_escape(eyebrow)}"
            "</div>"
        )
        if eyebrow
        else ""
    )

    markup = (
        '<div class="page-intro">'
        f"{eyebrow_markup}"
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
# Surface wrappers
# =============================================================================


def surface_card(
    title: str | None = None,
    subtitle: str | None = None,
    *,
    tone: Tone = "neutral",
) -> None:
    """
    Render a lightweight content-surface header.

    Intended for reusable visual grouping where a full Streamlit container
    is not necessary.
    """

    color = _tone_color(
        tone
    )

    title_markup = (
        (
            '<div class="surface-card-title">'
            f"{_escape(title)}"
            "</div>"
        )
        if title
        else ""
    )

    subtitle_markup = (
        (
            '<div class="surface-card-subtitle">'
            f"{_escape(subtitle)}"
            "</div>"
        )
        if subtitle
        else ""
    )

    markup = (
        '<div class="surface-card-heading" '
        f'style="--surface-accent:{color};">'
        f"{title_markup}"
        f"{subtitle_markup}"
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

    Designed to avoid Streamlit native metric truncation for long business
    values such as recommendations and status labels.
    """

    safe_label = _escape(
        label
    )

    safe_value = _escape(
        value
    )

    safe_helper = _escape(
        helper
    )

    color = _tone_color(
        tone
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


def mini_metric(
    label: str,
    value: str,
    *,
    helper: str | None = None,
    tone: Tone = "neutral",
) -> None:
    """
    Render a compact metric suitable for diagnostics and dense layouts.
    """

    color = _tone_color(
        tone
    )

    helper_markup = (
        (
            '<div class="mini-metric-helper">'
            f"{_escape(helper)}"
            "</div>"
        )
        if helper
        else ""
    )

    markup = (
        '<div class="mini-metric" '
        f'style="--mini-accent:{color};">'
        '<div class="mini-metric-label">'
        f"{_escape(label)}"
        "</div>"
        '<div class="mini-metric-value">'
        f"{_escape(value)}"
        "</div>"
        f"{helper_markup}"
        "</div>"
    )

    _render_html(
        markup
    )


# =============================================================================
# Generic badges
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

    color = _tone_color(
        tone
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


def risk_badge(
    score: float,
) -> None:
    """
    Render the categorical risk tier below the fraud-risk gauge.
    """

    score = _clamp_score(
        score
    )

    tier = risk_tier(
        score
    )

    color = risk_color(
        score
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

    Compact HTML is used deliberately to prevent Streamlit/Markdown from
    treating nested tags as code blocks.
    """

    score = _clamp_score(
        score
    )

    tier = risk_tier(
        score
    )

    color = risk_color(
        score
    )

    degrees = (
        score
        * 360.0
    )

    safe_tier = _escape(
        tier
    )

    aria_label = _escape(
        (
            f"Fraud risk {score:.1%}, "
            f"risk tier {tier}"
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

    color = _tone_color(
        tone
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


def decision_panel(
    title: str,
    message: str,
    *,
    tone: Tone = "info",
    caption: str | None = None,
) -> None:
    """
    Render a stronger action-oriented decision-support panel.
    """

    color = _tone_color(
        tone
    )

    caption_markup = (
        (
            '<div class="decision-panel-caption">'
            f"{_escape(caption)}"
            "</div>"
        )
        if caption
        else ""
    )

    markup = (
        '<div class="decision-panel" '
        f'style="--decision-color:{color};">'
        '<div class="decision-panel-header">'
        '<span class="decision-panel-dot"></span>'
        '<div class="decision-panel-title">'
        f"{_escape(title)}"
        "</div>"
        "</div>"
        '<div class="decision-panel-message">'
        f"{_escape(message)}"
        "</div>"
        f"{caption_markup}"
        "</div>"
    )

    _render_html(
        markup
    )


# =============================================================================
# SHAP / model-driver components
# =============================================================================


def driver_card(
    label: str,
    *,
    feature_value: str | None = None,
    contribution: float | None = None,
    direction: Literal[
        "increase",
        "decrease",
        "neutral",
    ] = "neutral",
) -> None:
    """
    Render one local model-driver card.

    Intended for TreeSHAP explanations and other signed feature attributions.
    """

    direction_map = {
        "increase":
            (
                "danger",
                "INCREASES MODEL RISK",
            ),

        "decrease":
            (
                "success",
                "REDUCES MODEL RISK",
            ),

        "neutral":
            (
                "neutral",
                "NEUTRAL CONTRIBUTION",
            ),
    }

    tone, direction_label = (
        direction_map[
            direction
        ]
    )

    color = _tone_color(
        tone
    )

    value_markup = (
        (
            '<div class="driver-value">'
            '<span>Model value</span>'
            f"<strong>{_escape(feature_value)}</strong>"
            "</div>"
        )
        if feature_value is not None
        else ""
    )

    contribution_markup = ""

    if contribution is not None:

        try:

            contribution_value = float(
                contribution
            )

            contribution_text = (
                f"{contribution_value:+.4f}"
            )

        except (
            TypeError,
            ValueError,
        ):

            contribution_text = (
                str(
                    contribution
                )
            )

        contribution_markup = (
            '<div class="driver-contribution">'
            '<span>SHAP contribution</span>'
            f"<strong>{_escape(contribution_text)}</strong>"
            "</div>"
        )

    markup = (
        '<div class="driver-card" '
        f'style="--driver-color:{color};">'
        '<div class="driver-direction">'
        f"{_escape(direction_label)}"
        "</div>"
        '<div class="driver-label">'
        f"{_escape(label)}"
        "</div>"
        '<div class="driver-meta">'
        f"{value_markup}"
        f"{contribution_markup}"
        "</div>"
        "</div>"
    )

    _render_html(
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
# Key/value list
# =============================================================================


def key_value_row(
    label: str,
    value: str,
    *,
    monospace: bool = False,
) -> None:
    """
    Render one dense key/value diagnostic row.
    """

    value_class = (
        " key-value-monospace"
        if monospace
        else ""
    )

    markup = (
        '<div class="key-value-row">'
        '<div class="key-value-label">'
        f"{_escape(label)}"
        "</div>"
        f'<div class="key-value-value{value_class}">'
        f"{_escape(value)}"
        "</div>"
        "</div>"
    )

    _render_html(
        markup
    )


# =============================================================================
# Section separator
# =============================================================================


def soft_divider(
    label: str | None = None,
) -> None:
    """
    Render a visually light separator.

    Optional labels are useful inside dense operational panels.
    """

    label_markup = (
        (
            '<span class="soft-divider-label">'
            f"{_escape(label)}"
            "</span>"
        )
        if label
        else ""
    )

    markup = (
        '<div class="soft-divider">'
        '<div class="soft-divider-line"></div>'
        f"{label_markup}"
        '<div class="soft-divider-line"></div>'
        "</div>"
    )

    _render_compact_html(
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