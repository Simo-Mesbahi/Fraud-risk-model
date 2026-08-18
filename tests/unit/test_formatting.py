from __future__ import annotations

import math

import pytest

from frontend.utils.formatting import (
    clamp_probability,
    contribution_direction,
    display_value,
    format_contribution,
    format_integer,
    format_number,
    format_review_policy,
    format_score,
    humanize_identifier,
    percent,
    probability_percent,
    review_status,
    risk_color,
    risk_label,
    risk_tier,
    safe_float,
    safe_int,
)


# =============================================================================
# safe_float
# =============================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (1.25, 1.25),
        ("1.5", 1.5),
        ("0", 0.0),
    ],
)
def test_safe_float_valid_values(
    value,
    expected,
) -> None:
    assert (
        safe_float(value)
        == expected
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        "abc",
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_safe_float_invalid_values_return_none(
    value,
) -> None:
    assert (
        safe_float(value)
        is None
    )


def test_safe_float_uses_default() -> None:
    assert (
        safe_float(
            "invalid",
            default=7.5,
        )
        == 7.5
    )


# =============================================================================
# safe_int
# =============================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (1.0, 1),
        ("2", 2),
        ("3.0", 3),
    ],
)
def test_safe_int_accepts_integer_like_values(
    value,
    expected,
) -> None:
    assert (
        safe_int(value)
        == expected
    )


@pytest.mark.parametrize(
    "value",
    [
        1.2,
        "3.5",
        None,
        float("nan"),
    ],
)
def test_safe_int_rejects_non_integer_values(
    value,
) -> None:
    assert (
        safe_int(value)
        is None
    )


# =============================================================================
# probability validation
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        0.5,
        1.0,
    ],
)
def test_clamp_probability_accepts_valid_range(
    value: float,
) -> None:
    assert (
        clamp_probability(value)
        == value
    )


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
        None,
        "invalid",
        float("nan"),
        float("inf"),
    ],
)
def test_clamp_probability_rejects_invalid_values(
    value,
) -> None:
    with pytest.raises(
        ValueError,
        match="Probability",
    ):
        clamp_probability(value)


# =============================================================================
# risk tiers
# =============================================================================


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "LOW"),
        (0.049999, "LOW"),
        (0.05, "MEDIUM"),
        (0.199999, "MEDIUM"),
        (0.20, "HIGH"),
        (0.499999, "HIGH"),
        (0.50, "CRITICAL"),
        (1.0, "CRITICAL"),
    ],
)
def test_risk_tier_boundaries(
    score: float,
    expected: str,
) -> None:
    assert (
        risk_tier(score)
        == expected
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.01, "#61E7A6"),
        (0.10, "#FFD166"),
        (0.30, "#FF8A5B"),
        (0.70, "#FF5C7A"),
    ],
)
def test_risk_color_mapping(
    score: float,
    expected: str,
) -> None:
    assert (
        risk_color(score)
        == expected
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.01, "Low Risk"),
        (0.10, "Medium Risk"),
        (0.30, "High Risk"),
        (0.70, "Critical Risk"),
    ],
)
def test_risk_label_mapping(
    score: float,
    expected: str,
) -> None:
    assert (
        risk_label(score)
        == expected
    )


# =============================================================================
# percentage formatting
# =============================================================================


def test_percent_default_precision() -> None:
    assert (
        percent(0.0314)
        == "3.14%"
    )


def test_percent_custom_precision() -> None:
    assert (
        percent(
            0.0314,
            digits=1,
        )
        == "3.1%"
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        float("nan"),
        float("inf"),
        "invalid",
    ],
)
def test_percent_missing_values(
    value,
) -> None:
    assert (
        percent(value)
        == "—"
    )


def test_percent_rejects_negative_precision() -> None:
    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        percent(
            0.10,
            digits=-1,
        )


def test_probability_percent_validates_range() -> None:
    assert (
        probability_percent(
            0.42,
            digits=1,
        )
        == "42.0%"
    )


def test_probability_percent_rejects_invalid_probability() -> None:
    with pytest.raises(
        ValueError
    ):
        probability_percent(
            1.5
        )


# =============================================================================
# number formatting
# =============================================================================


def test_format_number() -> None:
    assert (
        format_number(
            1234.567,
            digits=2,
        )
        == "1,234.57"
    )


def test_format_integer() -> None:
    assert (
        format_integer(
            1234.4
        )
        == "1,234"
    )


def test_format_score() -> None:
    assert (
        format_score(
            0.123456,
            digits=4,
        )
        == "0.1235"
    )


@pytest.mark.parametrize(
    "function",
    [
        format_number,
        format_score,
    ],
)
def test_numeric_formatters_reject_invalid_precision(
    function,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-negative integer",
    ):
        function(
            1.23,
            digits=-1,
        )


# =============================================================================
# text formatting
# =============================================================================


def test_humanize_identifier() -> None:
    assert (
        humanize_identifier(
            "fraud_risk_score"
        )
        == "Fraud Risk Score"
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
    ],
)
def test_humanize_identifier_missing(
    value,
) -> None:
    assert (
        humanize_identifier(value)
        == "—"
    )


def test_display_value_boolean() -> None:
    assert (
        display_value(True)
        == "Yes"
    )

    assert (
        display_value(False)
        == "No"
    )


def test_display_value_missing() -> None:
    assert (
        display_value(None)
        == "—"
    )


def test_display_value_custom_missing() -> None:
    assert (
        display_value(
            None,
            missing="N/A",
        )
        == "N/A"
    )


# =============================================================================
# review policy
# =============================================================================


def test_format_review_policy_top_fraction() -> None:
    policy = {
        "type": "top_fraction",
        "fraction": 0.03,
    }

    assert (
        format_review_policy(policy)
        == "Top 3% highest-risk claims"
    )


def test_format_review_policy_missing() -> None:
    assert (
        format_review_policy(None)
        == "Not configured"
    )


def test_format_review_policy_missing_fraction() -> None:
    assert (
        format_review_policy(
            {
                "type":
                    "top_fraction"
            }
        )
        == (
            "Top-fraction review policy "
            "(fraction unavailable)"
        )
    )


@pytest.mark.parametrize(
    "fraction",
    [
        0,
        -0.1,
        1.1,
    ],
)
def test_format_review_policy_invalid_fraction(
    fraction: float,
) -> None:
    result = (
        format_review_policy(
            {
                "type": "top_fraction",
                "fraction": fraction,
            }
        )
    )

    assert (
        result
        == (
            "Top-fraction review policy "
            "(invalid fraction)"
        )
    )


def test_format_review_policy_unknown_type() -> None:
    assert (
        format_review_policy(
            {
                "type": "manual_review"
            }
        )
        == "Manual Review"
    )


# =============================================================================
# review status
# =============================================================================


def test_review_status() -> None:
    assert (
        review_status(True)
        == "Selected for Review"
    )

    assert (
        review_status(False)
        == "Not Selected"
    )

    assert (
        review_status(None)
        == "Unknown"
    )


# =============================================================================
# SHAP formatting
# =============================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            0.1,
            "INCREASES RISK",
        ),
        (
            -0.1,
            "DECREASES RISK",
        ),
        (
            0.0,
            "NEUTRAL",
        ),
    ],
)
def test_contribution_direction(
    value: float,
    expected: str,
) -> None:
    assert (
        contribution_direction(value)
        == expected
    )


def test_contribution_direction_respects_tolerance() -> None:
    assert (
        contribution_direction(
            1e-13,
            tolerance=1e-12,
        )
        == "NEUTRAL"
    )


def test_contribution_direction_unknown_value() -> None:
    assert (
        contribution_direction(
            None
        )
        == "UNKNOWN"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            0.12345,
            "+0.1235",
        ),
        (
            -0.12345,
            "-0.1235",
        ),
        (
            0.0,
            "+0.0000",
        ),
    ],
)
def test_format_contribution(
    value: float,
    expected: str,
) -> None:
    assert (
        format_contribution(value)
        == expected
    )


def test_format_contribution_missing() -> None:
    assert (
        format_contribution(None)
        == "—"
    )