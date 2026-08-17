from __future__ import annotations


def validate_claims(
    claims: list[dict],
) -> tuple[
    bool,
    list[str],
]:
    errors = []

    if not claims:
        errors.append(
            "No claims were provided."
        )

        return (
            False,
            errors,
        )

    for index, claim in enumerate(
        claims
    ):
        if not isinstance(
            claim,
            dict,
        ):
            errors.append(
                f"Claim #{index + 1} "
                "is not a valid object."
            )

    return (
        len(errors) == 0,
        errors,
    )