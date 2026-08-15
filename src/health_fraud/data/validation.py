from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# Validation result containers
# =============================================================================


@dataclass
class ValidationIssue:
    rule: str
    severity: str
    count: int
    description: str
    sample_indices: list[int] = field(default_factory=list)


@dataclass
class ValidationReport:
    dataset_name: str
    n_rows: int
    n_columns: int
    issues: list[ValidationIssue]

    @property
    def error_count(self) -> int:
        return sum(
            issue.count
            for issue in self.issues
            if issue.severity == "error"
        )

    @property
    def warning_count(self) -> int:
        return sum(
            issue.count
            for issue in self.issues
            if issue.severity == "warning"
        )

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "rule": issue.rule,
                    "severity": issue.severity,
                    "count": issue.count,
                    "description": issue.description,
                    "sample_indices": issue.sample_indices,
                }
                for issue in self.issues
            ]
        )


# =============================================================================
# Generic helpers
# =============================================================================


def _sample_indices(
    mask: pd.Series | np.ndarray,
    max_samples: int = 5,
) -> list[int]:
    mask = np.asarray(mask, dtype=bool)

    return (
        np.flatnonzero(mask)
        [:max_samples]
        .astype(int)
        .tolist()
    )


def _add_issue(
    issues: list[ValidationIssue],
    rule: str,
    severity: str,
    mask: pd.Series | np.ndarray,
    description: str,
) -> None:
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())

    if count == 0:
        return

    issues.append(
        ValidationIssue(
            rule=rule,
            severity=severity,
            count=count,
            description=description,
            sample_indices=_sample_indices(mask),
        )
    )


def _require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
) -> None:
    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )


# =============================================================================
# Claims validation
# =============================================================================


CLAIM_REQUIRED_COLUMNS = {
    "claim_id",
    "customer_id",
    "policy_id",
    "provider_id",
    "service_category",
    "service_code",
    "service_units",
    "service_date",
    "claim_submission_date",
    "claim_submission_timestamp",
    "claim_amount",
    "requested_reimbursement",
    "coverage_limit",
    "submission_channel",
    "document_count",
    "has_invoice",
    "has_prescription",
    "customer_age",
    "customer_tenure_months",
    "coverage_level",
    "policy_tenure_months",
    "provider_type",
    "is_fraud",
}


ALLOWED_SERVICE_CATEGORIES = {
    "consultation",
    "dental",
    "optical",
    "physiotherapy",
    "pharmacy",
    "medical_device",
    "diagnostic",
    "other",
}


ALLOWED_COVERAGE_LEVELS = {
    "basic",
    "standard",
    "premium",
}


ALLOWED_SUBMISSION_CHANNELS = {
    "mobile_app",
    "web",
    "email",
    "paper",
    "provider_direct",
}


def validate_claims(
    claims: pd.DataFrame,
) -> ValidationReport:
    _require_columns(
        claims,
        CLAIM_REQUIRED_COLUMNS,
    )

    issues: list[ValidationIssue] = []

    n_rows = len(claims)

    # -------------------------------------------------------------------------
    # Identifier integrity
    # -------------------------------------------------------------------------

    _add_issue(
        issues,
        rule="claim_id_not_null",
        severity="error",
        mask=claims["claim_id"].isna(),
        description="claim_id must never be missing.",
    )

    _add_issue(
        issues,
        rule="claim_id_unique",
        severity="error",
        mask=claims["claim_id"].duplicated(
            keep=False
        ),
        description="claim_id must be globally unique.",
    )

    _add_issue(
        issues,
        rule="customer_id_not_null",
        severity="error",
        mask=claims["customer_id"].isna(),
        description="customer_id must not be missing.",
    )

    _add_issue(
        issues,
        rule="policy_id_not_null",
        severity="error",
        mask=claims["policy_id"].isna(),
        description="policy_id must not be missing.",
    )

    # Provider missingness is intentionally allowed.
    _add_issue(
        issues,
        rule="provider_id_missing",
        severity="warning",
        mask=claims["provider_id"].isna(),
        description=(
            "provider_id may be missing for a small "
            "subset of synthetic claims."
        ),
    )

    # -------------------------------------------------------------------------
    # Financial constraints
    # -------------------------------------------------------------------------

    _add_issue(
        issues,
        rule="claim_amount_positive",
        severity="error",
        mask=(
            claims["claim_amount"].isna()
            | (claims["claim_amount"] <= 0)
        ),
        description="claim_amount must be strictly positive.",
    )

    _add_issue(
        issues,
        rule="requested_reimbursement_non_negative",
        severity="error",
        mask=(
            claims["requested_reimbursement"].isna()
            | (claims["requested_reimbursement"] < 0)
        ),
        description=(
            "requested_reimbursement must be "
            "greater than or equal to zero."
        ),
    )

    _add_issue(
        issues,
        rule="coverage_limit_positive",
        severity="error",
        mask=(
            claims["coverage_limit"].isna()
            | (claims["coverage_limit"] <= 0)
        ),
        description="coverage_limit must be strictly positive.",
    )

    _add_issue(
        issues,
        rule="requested_reimbursement_within_limit",
        severity="error",
        mask=(
            claims["requested_reimbursement"]
            > claims["coverage_limit"]
        ),
        description=(
            "requested_reimbursement must not exceed "
            "the configured coverage limit."
        ),
    )

    _add_issue(
        issues,
        rule="requested_reimbursement_not_above_claim",
        severity="error",
        mask=(
            claims["requested_reimbursement"]
            > claims["claim_amount"]
        ),
        description=(
            "requested_reimbursement must not exceed "
            "claim_amount."
        ),
    )

    # -------------------------------------------------------------------------
    # Service quantities
    # -------------------------------------------------------------------------

    _add_issue(
        issues,
        rule="service_units_positive",
        severity="error",
        mask=(
            claims["service_units"].isna()
            | (claims["service_units"] < 1)
        ),
        description="service_units must be at least 1.",
    )

    # -------------------------------------------------------------------------
    # Date consistency
    # -------------------------------------------------------------------------

    _add_issue(
        issues,
        rule="service_date_not_after_submission",
        severity="error",
        mask=(
            claims["service_date"]
            > claims["claim_submission_date"]
        ),
        description=(
            "service_date must be on or before "
            "claim_submission_date."
        ),
    )

    _add_issue(
        issues,
        rule="submission_date_matches_timestamp",
        severity="error",
        mask=(
            claims["claim_submission_date"]
            != claims[
                "claim_submission_timestamp"
            ].dt.normalize()
        ),
        description=(
            "claim_submission_date must match the "
            "calendar date of claim_submission_timestamp."
        ),
    )

    _add_issue(
        issues,
        rule="days_service_to_submission_non_negative",
        severity="error",
        mask=(
            claims[
                "days_service_to_submission"
            ].isna()
            | (
                claims[
                    "days_service_to_submission"
                ]
                < 0
            )
        ),
        description=(
            "days_service_to_submission must be "
            "greater than or equal to zero."
        ),
    )

    # -------------------------------------------------------------------------
    # Categorical domains
    # -------------------------------------------------------------------------

    _add_issue(
        issues,
        rule="service_category_valid",
        severity="error",
        mask=~claims[
            "service_category"
        ].isin(ALLOWED_SERVICE_CATEGORIES),
        description="Unknown service_category detected.",
    )

    _add_issue(
        issues,
        rule="coverage_level_valid",
        severity="error",
        mask=~claims[
            "coverage_level"
        ].isin(ALLOWED_COVERAGE_LEVELS),
        description="Unknown coverage_level detected.",
    )

    _add_issue(
        issues,
        rule="submission_channel_valid",
        severity="error",
        mask=~claims[
            "submission_channel"
        ].isin(ALLOWED_SUBMISSION_CHANNELS),
        description="Unknown submission_channel detected.",
    )

    # -------------------------------------------------------------------------
    # Target integrity
    # -------------------------------------------------------------------------

    _add_issue(
        issues,
        rule="fraud_target_not_null",
        severity="error",
        mask=claims["is_fraud"].isna(),
        description="is_fraud must not be missing.",
    )

    _add_issue(
        issues,
        rule="fraud_target_binary",
        severity="error",
        mask=~claims["is_fraud"].isin([0, 1]),
        description="is_fraud must contain only 0 or 1.",
    )

    # -------------------------------------------------------------------------
    # Missingness expected by design
    # -------------------------------------------------------------------------

    _add_issue(
        issues,
        rule="document_count_missing",
        severity="warning",
        mask=claims["document_count"].isna(),
        description=(
            "document_count contains intentionally "
            "simulated missing values."
        ),
    )

    _add_issue(
        issues,
        rule="prescription_missing",
        severity="warning",
        mask=claims["has_prescription"].isna(),
        description=(
            "has_prescription contains intentionally "
            "simulated missing values."
        ),
    )

    # -------------------------------------------------------------------------
    # Historical temporal consistency
    # -------------------------------------------------------------------------

    history_columns = [
        "customer_claims_7d",
        "customer_claims_30d",
        "customer_claims_90d",
        "customer_claims_365d",
    ]

    if all(
        column in claims.columns
        for column in history_columns
    ):
        history = claims[
            history_columns
        ].fillna(0)

        _add_issue(
            issues,
            rule="customer_claim_windows_monotonic",
            severity="error",
            mask=~(
                (
                    history[
                        "customer_claims_7d"
                    ]
                    <= history[
                        "customer_claims_30d"
                    ]
                )
                & (
                    history[
                        "customer_claims_30d"
                    ]
                    <= history[
                        "customer_claims_90d"
                    ]
                )
                & (
                    history[
                        "customer_claims_90d"
                    ]
                    <= history[
                        "customer_claims_365d"
                    ]
                )
            ),
            description=(
                "Customer claim counts must be "
                "non-decreasing as the historical "
                "window becomes larger."
            ),
        )

    # -------------------------------------------------------------------------
    # Probability / ratio checks
    # -------------------------------------------------------------------------

    if "reimbursement_ratio" in claims.columns:
        _add_issue(
            issues,
            rule="reimbursement_ratio_range",
            severity="error",
            mask=(
                claims[
                    "reimbursement_ratio"
                ].isna()
                | (
                    claims[
                        "reimbursement_ratio"
                    ]
                    < 0
                )
                | (
                    claims[
                        "reimbursement_ratio"
                    ]
                    > 1
                )
            ),
            description=(
                "reimbursement_ratio must lie in [0, 1]."
            ),
        )

    if (
        "synthetic_fraud_probability"
        in claims.columns
    ):
        _add_issue(
            issues,
            rule="synthetic_probability_range",
            severity="error",
            mask=(
                claims[
                    "synthetic_fraud_probability"
                ].isna()
                | (
                    claims[
                        "synthetic_fraud_probability"
                    ]
                    < 0
                )
                | (
                    claims[
                        "synthetic_fraud_probability"
                    ]
                    > 1
                )
            ),
            description=(
                "synthetic_fraud_probability must lie "
                "in [0, 1]."
            ),
        )

    return ValidationReport(
        dataset_name="claims",
        n_rows=n_rows,
        n_columns=claims.shape[1],
        issues=issues,
    )


# =============================================================================
# Entity-table validation
# =============================================================================


def validate_customers(
    customers: pd.DataFrame,
) -> ValidationReport:
    required = {
        "customer_id",
        "customer_age",
        "customer_tenure_months",
        "coverage_level",
    }

    _require_columns(
        customers,
        required,
    )

    issues: list[ValidationIssue] = []

    _add_issue(
        issues,
        rule="customer_id_unique",
        severity="error",
        mask=customers[
            "customer_id"
        ].duplicated(keep=False),
        description="customer_id must be unique.",
    )

    _add_issue(
        issues,
        rule="customer_age_valid",
        severity="error",
        mask=(
            customers["customer_age"].isna()
            | (customers["customer_age"] < 18)
            | (customers["customer_age"] > 100)
        ),
        description=(
            "customer_age must be between 18 and 100."
        ),
    )

    _add_issue(
        issues,
        rule="customer_tenure_positive",
        severity="error",
        mask=(
            customers[
                "customer_tenure_months"
            ].isna()
            | (
                customers[
                    "customer_tenure_months"
                ]
                < 1
            )
        ),
        description=(
            "customer_tenure_months must be positive."
        ),
    )

    return ValidationReport(
        dataset_name="customers",
        n_rows=len(customers),
        n_columns=customers.shape[1],
        issues=issues,
    )


def validate_providers(
    providers: pd.DataFrame,
) -> ValidationReport:
    required = {
        "provider_id",
        "provider_type",
        "provider_region",
        "provider_tenure_months",
    }

    _require_columns(
        providers,
        required,
    )

    issues: list[ValidationIssue] = []

    _add_issue(
        issues,
        rule="provider_id_unique",
        severity="error",
        mask=providers[
            "provider_id"
        ].duplicated(keep=False),
        description="provider_id must be unique.",
    )

    _add_issue(
        issues,
        rule="provider_tenure_positive",
        severity="error",
        mask=(
            providers[
                "provider_tenure_months"
            ].isna()
            | (
                providers[
                    "provider_tenure_months"
                ]
                < 1
            )
        ),
        description=(
            "provider_tenure_months must be positive."
        ),
    )

    return ValidationReport(
        dataset_name="providers",
        n_rows=len(providers),
        n_columns=providers.shape[1],
        issues=issues,
    )


def validate_policies(
    policies: pd.DataFrame,
) -> ValidationReport:
    required = {
        "policy_id",
        "customer_id",
        "coverage_level",
        "policy_start_date",
    }

    _require_columns(
        policies,
        required,
    )

    issues: list[ValidationIssue] = []

    _add_issue(
        issues,
        rule="policy_id_unique",
        severity="error",
        mask=policies[
            "policy_id"
        ].duplicated(keep=False),
        description="policy_id must be unique.",
    )

    _add_issue(
        issues,
        rule="policy_start_date_not_null",
        severity="error",
        mask=policies[
            "policy_start_date"
        ].isna(),
        description=(
            "policy_start_date must not be missing."
        ),
    )

    return ValidationReport(
        dataset_name="policies",
        n_rows=len(policies),
        n_columns=policies.shape[1],
        issues=issues,
    )


# =============================================================================
# Cross-table integrity
# =============================================================================


def validate_referential_integrity(
    customers: pd.DataFrame,
    providers: pd.DataFrame,
    policies: pd.DataFrame,
    claims: pd.DataFrame,
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    customer_ids = set(
        customers[
            "customer_id"
        ].dropna()
    )

    policy_ids = set(
        policies[
            "policy_id"
        ].dropna()
    )

    provider_ids = set(
        providers[
            "provider_id"
        ].dropna()
    )

    _add_issue(
        issues,
        rule="claim_customer_exists",
        severity="error",
        mask=~claims[
            "customer_id"
        ].isin(customer_ids),
        description=(
            "Every claim customer_id must exist "
            "in customers."
        ),
    )

    _add_issue(
        issues,
        rule="claim_policy_exists",
        severity="error",
        mask=~claims[
            "policy_id"
        ].isin(policy_ids),
        description=(
            "Every claim policy_id must exist "
            "in policies."
        ),
    )

    provider_not_null = claims[
        "provider_id"
    ].notna()

    provider_invalid = (
        provider_not_null
        & ~claims[
            "provider_id"
        ].isin(provider_ids)
    )

    _add_issue(
        issues,
        rule="claim_provider_exists",
        severity="error",
        mask=provider_invalid,
        description=(
            "Every non-missing claim provider_id "
            "must exist in providers."
        ),
    )

    policy_customer_map = (
        policies
        .set_index("policy_id")[
            "customer_id"
        ]
    )

    expected_customer = (
        claims[
            "policy_id"
        ]
        .map(policy_customer_map)
    )

    _add_issue(
        issues,
        rule="policy_customer_consistency",
        severity="error",
        mask=(
            expected_customer
            != claims[
                "customer_id"
            ]
        ),
        description=(
            "Each claim's policy must belong to "
            "the same customer as the claim."
        ),
    )

    return ValidationReport(
        dataset_name="referential_integrity",
        n_rows=len(claims),
        n_columns=0,
        issues=issues,
    )


# =============================================================================
# Complete validation
# =============================================================================


def validate_dataset_bundle(
    customers: pd.DataFrame,
    providers: pd.DataFrame,
    policies: pd.DataFrame,
    claims: pd.DataFrame,
) -> dict[str, ValidationReport]:
    return {
        "customers": validate_customers(
            customers
        ),
        "providers": validate_providers(
            providers
        ),
        "policies": validate_policies(
            policies
        ),
        "claims": validate_claims(
            claims
        ),
        "referential_integrity": (
            validate_referential_integrity(
                customers=customers,
                providers=providers,
                policies=policies,
                claims=claims,
            )
        ),
    }


def print_validation_report(
    reports: dict[
        str,
        ValidationReport,
    ],
) -> None:
    print(
        "\n"
        + "=" * 72
    )

    print(
        "DATA VALIDATION REPORT"
    )

    print(
        "=" * 72
    )

    total_errors = 0
    total_warnings = 0

    for (
        name,
        report,
    ) in reports.items():

        print(
            f"\n[{name.upper()}]"
        )

        print(
            f"Rows: {report.n_rows:,}"
        )

        print(
            f"Errors: {report.error_count:,}"
        )

        print(
            f"Warnings: {report.warning_count:,}"
        )

        if not report.issues:
            print(
                "No validation issues."
            )

            continue

        for issue in report.issues:
            print(
                f"  "
                f"{issue.severity.upper():<7} "
                f"{issue.rule:<42} "
                f"{issue.count:>7,}"
            )

            print(
                f"          "
                f"{issue.description}"
            )

            if issue.sample_indices:
                print(
                    "          "
                    "Sample rows: "
                    + ", ".join(
                        map(
                            str,
                            issue.sample_indices,
                        )
                    )
                )

        total_errors += (
            report.error_count
        )

        total_warnings += (
            report.warning_count
        )

    print(
        "\n"
        + "-" * 72
    )

    print(
        f"TOTAL ERRORS:   {total_errors:,}"
    )

    print(
        f"TOTAL WARNINGS: {total_warnings:,}"
    )

    print(
        "=" * 72
    )
    