from __future__ import annotations

import math

from numbers import Real
from typing import Any


# =============================================================================
# Risk presentation contract
# =============================================================================


RISK_LOW_THRESHOLD = 0.05
RISK_MEDIUM_THRESHOLD = 0.20
RISK_HIGH_THRESHOLD = 0.50


RISK_TIERS = (
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
)


RISK_COLORS: dict[str, str] = {
    "LOW": "#61E7A6",
    "MEDIUM": "#FFD166",
    "HIGH": "#FF8A5B",
    "CRITICAL": "#FF5C7A",
}


# =============================================================================
# Generic numeric helpers
# =============================================================================


def safe_float(
    value: Any,
    *,
    default: float | None = None,
) -> float | None:
    """
    Convert a scalar-like value into a finite float.

    Invalid, missing and non-finite values return ``default``.
    """

    if value is None:
        return default

    if isinstance(
        value,
        bool,
    ):
        return default

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return default

    if not math.isfinite(
        result
    ):
        return default

    return result


def safe_int(
    value: Any,
    *,
    default: int | None = None,
) -> int | None:
    """
    Convert a finite integer-like value safely.

    Decimal values are accepted only when they represent
    an exact integer.
    """

    numeric = safe_float(
        value
    )

    if numeric is None:
        return default

    if not numeric.is_integer():
        return default

    return int(
        numeric
    )


def clamp_probability(
    value: Any,
) -> float:
    """
    Validate and return a probability in [0, 1].

    Risk scores and fractions are contracts, not arbitrary
    numerical values, so invalid values fail explicitly.
    """

    numeric = safe_float(
        value
    )

    if numeric is None:

        raise ValueError(
            "Probability must be a finite numeric value."
        )

    if not (
        0.0
        <= numeric
        <= 1.0
    ):

        raise ValueError(
            (
                "Probability must be between "
                "0 and 1 inclusive."
            )
        )

    return numeric


# =============================================================================
# Risk classification
# =============================================================================


def risk_tier(
    score: float,
) -> str:
    """
    Convert a fraud-risk probability into a presentation tier.

    This is a frontend interpretation layer only.

    It must not be interpreted as an automatic fraud decision,
    claim-rejection rule or investigation-capacity threshold.
    """

    probability = clamp_probability(
        score
    )

    if probability < RISK_LOW_THRESHOLD:
        return "LOW"

    if probability < RISK_MEDIUM_THRESHOLD:
        return "MEDIUM"

    if probability < RISK_HIGH_THRESHOLD:
        return "HIGH"

    return "CRITICAL"


def risk_color(
    score: float,
) -> str:
    """
    Return the presentation color associated with a risk tier.
    """

    tier = risk_tier(
        score
    )

    return RISK_COLORS[
        tier
    ]


def risk_label(
    score: float,
) -> str:
    """
    Return a human-readable risk label.
    """

    tier = risk_tier(
        score
    )

    labels = {
        "LOW": "Low Risk",
        "MEDIUM": "Medium Risk",
        "HIGH": "High Risk",
        "CRITICAL": "Critical Risk",
    }

    return labels[
        tier
    ]


# =============================================================================
# Percentage formatting
# =============================================================================


def percent(
    value: Any,
    digits: int = 2,
) -> str:
    """
    Format a decimal value as a percentage.

    Example
    -------
    0.0314 -> "3.14%"
    """

    numeric = safe_float(
        value
    )

    if numeric is None:
        return "—"

    try:
        precision = int(
            digits
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "digits must be an integer."
        ) from exc

    if precision < 0:

        raise ValueError(
            "digits must be non-negative."
        )

    return (
        f"{numeric:.{precision}%}"
    )


def probability_percent(
    value: Any,
    digits: int = 2,
) -> str:
    """
    Format a validated probability as a percentage.
    """

    probability = clamp_probability(
        value
    )

    return percent(
        probability,
        digits=digits,
    )


# =============================================================================
# Number formatting
# =============================================================================


def format_number(
    value: Any,
    digits: int = 2,
) -> str:
    """
    Format a finite numerical value for UI display.
    """

    numeric = safe_float(
        value
    )

    if numeric is None:
        return "—"

    precision = safe_int(
        digits
    )

    if (
        precision is None
        or precision < 0
    ):

        raise ValueError(
            "digits must be a non-negative integer."
        )

    return (
        f"{numeric:,.{precision}f}"
    )


def format_integer(
    value: Any,
) -> str:
    """
    Format an integer-like value with thousands separators.
    """

    numeric = safe_float(
        value
    )

    if numeric is None:
        return "—"

    return (
        f"{numeric:,.0f}"
    )


def format_score(
    value: Any,
    digits: int = 4,
) -> str:
    """
    Format a model score without converting it to percentage form.
    """

    numeric = safe_float(
        value
    )

    if numeric is None:
        return "—"

    precision = safe_int(
        digits
    )

    if (
        precision is None
        or precision < 0
    ):

        raise ValueError(
            "digits must be a non-negative integer."
        )

    return (
        f"{numeric:.{precision}f}"
    )


# =============================================================================
# Text formatting
# =============================================================================


def humanize_identifier(
    value: Any,
) -> str:
    """
    Convert snake_case identifiers into readable UI labels.

    Example
    -------
    fraud_risk_score -> Fraud Risk Score
    """

    if value is None:
        return "—"

    text = (
        str(value)
        .strip()
    )

    if not text:
        return "—"

    return (
        text
        .replace(
            "_",
            " ",
        )
        .strip()
        .title()
    )


def display_value(
    value: Any,
    *,
    missing: str = "—",
) -> str:
    """
    Convert a generic scalar value into a safe UI string.
    """

    if value is None:
        return missing

    if isinstance(
        value,
        bool,
    ):
        return (
            "Yes"
            if value
            else "No"
        )

    if isinstance(
        value,
        Real,
    ):

        numeric = safe_float(
            value
        )

        if numeric is None:
            return missing

    text = str(
        value
    ).strip()

    return (
        text
        if text
        else missing
    )


# =============================================================================
# Review policy
# =============================================================================


def format_review_policy(
    policy: Any,
) -> str:
    """
    Convert the model review-policy contract into a readable label.

    Expected contract
    -----------------
    {
        "type": "top_fraction",
        "fraction": 0.03
    }
    """

    if not policy:
        return "Not configured"

    if not isinstance(
        policy,
        dict,
    ):

        return str(
            policy
        )

    policy_type = (
        str(
            policy.get(
                "type",
                "",
            )
        )
        .strip()
        .lower()
    )

    fraction = safe_float(
        policy.get(
            "fraction"
        )
    )

    if (
        policy_type
        == "top_fraction"
    ):

        if fraction is None:

            return (
                "Top-fraction review policy "
                "(fraction unavailable)"
            )

        if not (
            0.0
            < fraction
            <= 1.0
        ):

            return (
                "Top-fraction review policy "
                "(invalid fraction)"
            )

        return (
            f"Top {fraction:.0%} "
            "highest-risk claims"
        )

    if policy_type:

        return humanize_identifier(
            policy_type
        )

    return "Configured"


# =============================================================================
# Review selection
# =============================================================================


def review_status(
    selected: Any,
) -> str:
    """
    Format investigation-selection state.
    """

    if selected is True:
        return "Selected for Review"

    if selected is False:
        return "Not Selected"

    return "Unknown"


# =============================================================================
# SHAP contribution formatting
# =============================================================================


def contribution_direction(
    value: Any,
    *,
    tolerance: float = 1e-12,
) -> str:
    """
    Classify a SHAP contribution by direction.

    Positive contributions increase the model raw margin.
    Negative contributions decrease it.
    """

    numeric = safe_float(
        value
    )

    if numeric is None:
        return "UNKNOWN"

    threshold = abs(
        float(
            tolerance
        )
    )

    if numeric > threshold:
        return "INCREASES RISK"

    if numeric < -threshold:
        return "DECREASES RISK"

    return "NEUTRAL"


def format_contribution(
    value: Any,
    digits: int = 4,
) -> str:
    """
    Format a signed SHAP contribution.
    """

    numeric = safe_float(
        value
    )

    if numeric is None:
        return "—"

    precision = safe_int(
        digits
    )

    if (
        precision is None
        or precision < 0
    ):

        raise ValueError(
            "digits must be a non-negative integer."
        )

    return (
        f"{numeric:+.{precision}f}"
    )