from __future__ import annotations

import pytest

from frontend.utils.validation import (
    validate_claims,
)


def test_validate_claims_rejects_empty_list() -> None:
    valid, errors = validate_claims([])

    assert valid is False
    assert errors == [
        "No claims were provided."
    ]


def test_validate_claims_accepts_single_claim() -> None:
    valid, errors = validate_claims(
        [
            {
                "claim_id": "CLM_TEST_001"
            }
        ]
    )

    assert valid is True
    assert errors == []


def test_validate_claims_accepts_multiple_claims() -> None:
    valid, errors = validate_claims(
        [
            {
                "claim_id": "CLM_1"
            },
            {
                "claim_id": "CLM_2"
            },
        ]
    )

    assert valid is True
    assert errors == []


@pytest.mark.parametrize(
    "invalid_claim",
    [
        None,
        [],
        "claim",
        42,
        3.14,
    ],
)
def test_validate_claims_rejects_non_dict_claim(
    invalid_claim,
) -> None:
    valid, errors = validate_claims(
        [
            invalid_claim
        ]
    )

    assert valid is False
    assert len(errors) == 1

    assert (
        "Claim #1 is not a valid object."
        in errors
    )


def test_validate_claims_reports_correct_position() -> None:
    valid, errors = validate_claims(
        [
            {
                "claim_id": "CLM_OK"
            },
            "invalid",
        ]
    )

    assert valid is False

    assert errors == [
        "Claim #2 is not a valid object."
    ]


def test_validate_claims_collects_multiple_errors() -> None:
    valid, errors = validate_claims(
        [
            None,
            {
                "claim_id": "CLM_OK"
            },
            "invalid",
        ]
    )

    assert valid is False

    assert errors == [
        "Claim #1 is not a valid object.",
        "Claim #3 is not a valid object.",
    ]