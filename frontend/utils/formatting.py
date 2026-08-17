from __future__ import annotations


def risk_tier(
    score: float,
) -> str:
    score = float(score)

    if score < 0.05:
        return "LOW"

    if score < 0.20:
        return "MEDIUM"

    if score < 0.50:
        return "HIGH"

    return "CRITICAL"


def risk_color(
    score: float,
) -> str:
    tier = risk_tier(score)

    mapping = {
        "LOW": "#61E7A6",
        "MEDIUM": "#FFD166",
        "HIGH": "#FF8A5B",
        "CRITICAL": "#FF5C7A",
    }

    return mapping[tier]


def percent(
    value: float,
    digits: int = 2,
) -> str:
    return f"{float(value):.{digits}%}"


def format_review_policy(
    policy,
) -> str:
    if not policy:
        return "Not configured"

    if isinstance(policy, dict):
        policy_type = policy.get(
            "type",
            "unknown",
        )

        fraction = policy.get(
            "fraction"
        )

        if (
            policy_type == "top_fraction"
            and fraction is not None
        ):
            return (
                f"Top {float(fraction):.0%} "
                "highest-risk claims"
            )

        return str(policy)

    return str(policy)