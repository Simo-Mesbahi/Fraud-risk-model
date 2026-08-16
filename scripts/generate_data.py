from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Project imports
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from health_fraud.data.generate import (  # noqa: E402
    SyntheticDataBundle,
    generate_synthetic_data,
    load_config,
)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the synthetic health insurance fraud dataset."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data.yaml",
        help="Path to the data-generation YAML configuration.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing synthetic output files.",
    )

    return parser.parse_args()


# =============================================================================
# Output helpers
# =============================================================================


def _resolve_output_paths(
    config: dict,
) -> dict[str, Path]:
    output_cfg = config["output"]

    output_dir = (
        PROJECT_ROOT
        / output_cfg["synthetic_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        name: output_dir / filename
        for name, filename
        in output_cfg["files"].items()
    }


def _check_existing_files(
    paths: dict[str, Path],
    overwrite: bool,
) -> None:
    existing = [
        path
        for path in paths.values()
        if path.exists()
    ]

    if existing and not overwrite:
        formatted = "\n".join(
            f"  - {path.relative_to(PROJECT_ROOT)}"
            for path in existing
        )

        raise FileExistsError(
            "Synthetic data files already exist:\n"
            f"{formatted}\n\n"
            "Run again with --overwrite "
            "if replacement is intentional."
        )


def _save_parquet(
    bundle: SyntheticDataBundle,
    paths: dict[str, Path],
) -> None:
    tables = {
        "customers": bundle.customers,
        "providers": bundle.providers,
        "policies": bundle.policies,
        "claims": bundle.claims,
    }

    for name, dataframe in tables.items():
        dataframe.to_parquet(
            paths[name],
            index=False,
            engine="pyarrow",
        )


# =============================================================================
# General reporting
# =============================================================================


def _print_table_summary(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    memory_mb = (
        dataframe
        .memory_usage(
            index=True,
            deep=True,
        )
        .sum()
        / 1024**2
    )

    print(
        f"{name:<12}"
        f"{len(dataframe):>10,} rows   "
        f"{dataframe.shape[1]:>3} columns   "
        f"{memory_mb:>8.2f} MB"
    )


def _print_claim_summary(
    claims: pd.DataFrame,
) -> None:
    print("\n" + "=" * 72)
    print("CLAIMS QUALITY SUMMARY")
    print("=" * 72)

    print(
        f"Claims:                  "
        f"{len(claims):,}"
    )

    print(
        f"Unique claims:           "
        f"{claims['claim_id'].nunique():,}"
    )

    print(
        f"Unique customers:        "
        f"{claims['customer_id'].nunique():,}"
    )

    print(
        f"Unique providers:        "
        f"{claims['provider_id'].nunique(dropna=True):,}"
    )

    fraud_count = int(
        claims["is_fraud"].sum()
    )

    fraud_rate = float(
        claims["is_fraud"].mean()
    )

    print(
        f"Fraud cases:             "
        f"{fraud_count:,}"
    )

    print(
        f"Fraud prevalence:        "
        f"{fraud_rate:.3%}"
    )

    print(
        f"Mean claim amount:       "
        f"EUR {claims['claim_amount'].mean():,.2f}"
    )

    print(
        f"Median claim amount:     "
        f"EUR {claims['claim_amount'].median():,.2f}"
    )

    print(
        f"Missing provider IDs:    "
        f"{claims['provider_id'].isna().mean():.3%}"
    )

    print(
        f"Missing prescriptions:   "
        f"{claims['has_prescription'].isna().mean():.3%}"
    )

    print(
        f"Missing document counts: "
        f"{claims['document_count'].isna().mean():.3%}"
    )

    duplicate_claim_ids = int(
        claims["claim_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate claim IDs:     "
        f"{duplicate_claim_ids:,}"
    )

    invalid_amounts = int(
        (
            claims["claim_amount"]
            <= 0
        ).sum()
    )

    invalid_units = int(
        (
            claims["service_units"]
            < 1
        ).sum()
    )

    invalid_dates = int(
        (
            claims["service_date"]
            > claims[
                "claim_submission_date"
            ]
        ).sum()
    )

    print(
        f"Invalid amounts:         "
        f"{invalid_amounts:,}"
    )

    print(
        f"Invalid service units:   "
        f"{invalid_units:,}"
    )

    print(
        f"Invalid service dates:   "
        f"{invalid_dates:,}"
    )


def _print_service_distribution(
    claims: pd.DataFrame,
) -> None:
    print("\n" + "=" * 72)
    print("SERVICE DISTRIBUTION")
    print("=" * 72)

    distribution = (
        claims["service_category"]
        .value_counts(normalize=True)
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    for (
        service,
        percentage,
    ) in distribution.items():
        print(
            f"{service:<22} "
            f"{percentage:>7.2f}%"
        )


def _print_fraud_distribution(
    claims: pd.DataFrame,
) -> None:
    print("\n" + "=" * 72)
    print("SYNTHETIC FRAUD MECHANISMS")
    print("=" * 72)

    fraud = claims.loc[
        claims["is_fraud"] == 1
    ]

    if fraud.empty:
        print(
            "No fraudulent claims generated."
        )
        return

    mechanisms = (
        fraud[
            "fraud_mechanism"
        ]
        .value_counts()
    )

    for (
        mechanism,
        count,
    ) in mechanisms.items():

        percentage = (
            count / len(fraud)
        )

        print(
            f"{mechanism:<30}"
            f"{count:>7,}   "
            f"{percentage:>7.2%}"
        )

    print("\nFraud difficulty:")

    difficulty = (
        fraud[
            "fraud_difficulty"
        ]
        .value_counts(
            normalize=True
        )
    )

    for (
        level,
        percentage,
    ) in difficulty.items():

        print(
            f"{level:<12} "
            f"{percentage:>7.2%}"
        )


def _print_temporal_summary(
    claims: pd.DataFrame,
) -> None:
    print("\n" + "=" * 72)
    print("TEMPORAL COVERAGE")
    print("=" * 72)

    minimum = claims[
        "claim_submission_timestamp"
    ].min()

    maximum = claims[
        "claim_submission_timestamp"
    ].max()

    print(
        f"First claim: {minimum}"
    )

    print(
        f"Last claim:  {maximum}"
    )


# =============================================================================
# Historical feature sanity
# =============================================================================


def _print_history_sanity(
    claims: pd.DataFrame,
) -> None:
    print("\n" + "=" * 72)
    print("HISTORICAL FEATURE SANITY CHECK")
    print("=" * 72)

    count_columns = [
        "customer_claims_7d",
        "customer_claims_30d",
        "customer_claims_90d",
        "customer_claims_365d",
        "customer_provider_claims_30d",
        "same_service_claims_30d",
        "provider_claims_30d",
        "provider_claims_90d",
    ]

    available = [
        column
        for column in count_columns
        if column in claims.columns
    ]

    if not available:
        print(
            "Historical count features "
            "not found."
        )
        return

    total_missing = 0
    total_negative = 0

    for column in available:
        missing = int(
            claims[column]
            .isna()
            .sum()
        )

        negative = int(
            (
                claims[column]
                .fillna(0)
                < 0
            ).sum()
        )

        total_missing += missing
        total_negative += negative

        status = (
            "PASS"
            if (
                missing == 0
                and negative == 0
            )
            else "FAIL"
        )

        print(
            f"{column:<38} "
            f"missing={missing:>6,}   "
            f"negative={negative:>4,}   "
            f"{status}"
        )

    if (
        total_missing == 0
        and total_negative == 0
    ):
        print(
            "\nPASS - count histories use "
            "zero when no prior event exists."
        )
    else:
        print(
            "\nFAIL - historical count "
            "features require investigation."
        )


# =============================================================================
# Fraud signal sanity
# =============================================================================


def _safe_median(
    dataframe: pd.DataFrame,
    feature: str,
) -> float:
    if (
        feature
        not in dataframe.columns
    ):
        return np.nan

    return float(
        dataframe[
            feature
        ].median()
    )


def _signal_ratio(
    mechanism_value: float,
    legitimate_value: float,
) -> float:
    if (
        not np.isfinite(
            mechanism_value
        )
        or not np.isfinite(
            legitimate_value
        )
        or legitimate_value == 0
    ):
        return np.nan

    return (
        mechanism_value
        / legitimate_value
    )


def _print_fraud_signal_sanity(
    claims: pd.DataFrame,
) -> None:
    """
    Check whether each synthetic fraud mechanism produces
    the business signal it is supposed to represent.

    These are sanity checks rather than model-performance
    requirements. Fraud and legitimate distributions should
    still overlap.
    """

    print("\n" + "=" * 72)
    print("FRAUD SIGNAL SANITY CHECK")
    print("=" * 72)

    required = {
        "is_fraud",
        "fraud_mechanism",
    }

    if not required.issubset(
        claims.columns
    ):
        print(
            "Required fraud metadata "
            "is not available."
        )
        return

    legitimate = claims.loc[
        claims["is_fraud"] == 0
    ]

    checks = {
        "amount_inflation": {
            "feature": (
                "claim_to_service_median_ratio"
            ),
            "label": (
                "claim/service amount ratio"
            ),
        },
        "frequency_abuse": {
            "feature": (
                "customer_claims_30d"
            ),
            "label": (
                "customer claims 30d"
            ),
        },
        "repeated_service": {
            "feature": (
                "same_service_claims_30d"
            ),
            "label": (
                "same-service claims 30d"
            ),
        },
        "provider_abnormality": {
            "feature": (
                "provider_claims_30d"
            ),
            "label": (
                "provider claims 30d"
            ),
        },
        "customer_provider_pattern": {
            "feature": (
                "customer_provider_claims_30d"
            ),
            "label": (
                "customer-provider claims 30d"
            ),
        },
    }

    pass_count = 0
    evaluated_count = 0

    for (
        mechanism,
        specification,
    ) in checks.items():

        feature = specification[
            "feature"
        ]

        label = specification[
            "label"
        ]

        mechanism_rows = claims.loc[
            (
                claims[
                    "is_fraud"
                ]
                == 1
            )
            & (
                claims[
                    "fraud_mechanism"
                ]
                == mechanism
            )
        ]

        if mechanism_rows.empty:
            print(
                f"\n{mechanism}"
            )

            print(
                "  No observations generated."
            )

            continue

        fraud_median = (
            _safe_median(
                mechanism_rows,
                feature,
            )
        )

        legitimate_median = (
            _safe_median(
                legitimate,
                feature,
            )
        )

        ratio = _signal_ratio(
            fraud_median,
            legitimate_median,
        )

        passed = (
            np.isfinite(
                fraud_median
            )
            and np.isfinite(
                legitimate_median
            )
            and (
                fraud_median
                > legitimate_median
            )
        )

        evaluated_count += 1

        if passed:
            pass_count += 1

        status = (
            "PASS"
            if passed
            else "REVIEW"
        )

        print(
            f"\n{mechanism}"
        )

        print(
            f"  Signal:              "
            f"{label}"
        )

        print(
            f"  Fraud median:        "
            f"{fraud_median:,.4f}"
        )

        print(
            f"  Legitimate median:   "
            f"{legitimate_median:,.4f}"
        )

        if np.isfinite(ratio):
            print(
                f"  Fraud / legitimate:  "
                f"{ratio:,.3f}x"
            )
        else:
            print(
                "  Fraud / legitimate:  "
                "n/a"
            )

        print(
            f"  Status:              "
            f"{status}"
        )

    # -------------------------------------------------------------------------
    # Mixed fraud diagnostic
    # -------------------------------------------------------------------------

    mixed = claims.loc[
        (
            claims["is_fraud"]
            == 1
        )
        & (
            claims[
                "fraud_mechanism"
            ]
            == "mixed_pattern"
        )
    ]

    if not mixed.empty:
        mixed_features = [
            "claim_to_service_median_ratio",
            "customer_claims_30d",
            "same_service_claims_30d",
            "customer_provider_claims_30d",
        ]

        active_signals = 0

        for feature in mixed_features:
            if feature not in claims.columns:
                continue

            mixed_median = mixed[
                feature
            ].median()

            legitimate_median = (
                legitimate[
                    feature
                ].median()
            )

            if (
                pd.notna(mixed_median)
                and pd.notna(
                    legitimate_median
                )
                and (
                    mixed_median
                    > legitimate_median
                )
            ):
                active_signals += 1

        print(
            "\nmixed_pattern"
        )

        print(
            "  Observable elevated "
            f"signals:       {active_signals}"
            f"/{len(mixed_features)}"
        )

        print(
            "  Status:              "
            + (
                "PASS"
                if active_signals >= 2
                else "REVIEW"
            )
        )

    print("\n" + "-" * 72)

    if (
        evaluated_count > 0
        and pass_count
        == evaluated_count
    ):
        print(
            "OVERALL SIGNAL STATUS: PASS"
        )

        print(
            "All primary fraud mechanisms "
            "show the expected directional signal."
        )

    else:
        print(
            "OVERALL SIGNAL STATUS: REVIEW"
        )

        print(
            f"{pass_count}/{evaluated_count} "
            "primary mechanisms passed "
            "the directional sanity check."
        )

        print(
            "Review weak mechanisms before "
            "starting model experiments."
        )


# =============================================================================
# Split-level fraud signal sanity
# =============================================================================


def _print_split_signal_summary(
    claims: pd.DataFrame,
    config: dict,
) -> None:
    print("\n" + "=" * 72)
    print("TEMPORAL FRAUD SIGNAL STABILITY")
    print("=" * 72)

    split_cfg = config.get(
        "splits",
        {},
    )

    if not split_cfg:
        print(
            "No temporal split "
            "configuration found."
        )
        return

    feature_checks = {
        "amount_inflation": (
            "claim_to_service_median_ratio"
        ),
        "frequency_abuse": (
            "customer_claims_30d"
        ),
        "repeated_service": (
            "same_service_claims_30d"
        ),
        "provider_abnormality": (
            "provider_claims_30d"
        ),
        "customer_provider_pattern": (
            "customer_provider_claims_30d"
        ),
    }

    for (
        split_name,
        period,
    ) in split_cfg.items():

        start = pd.Timestamp(
            period[
                "start_date"
            ]
        )

        end = (
            pd.Timestamp(
                period[
                    "end_date"
                ]
            )
            + pd.Timedelta(
                days=1
            )
            - pd.Timedelta(
                microseconds=1
            )
        )

        split = claims.loc[
            (
                claims[
                    "claim_submission_timestamp"
                ]
                >= start
            )
            & (
                claims[
                    "claim_submission_timestamp"
                ]
                <= end
            )
        ]

        if split.empty:
            continue

        print(
            f"\n[{split_name.upper()}]"
        )

        print(
            f"Rows:              "
            f"{len(split):,}"
        )

        print(
            f"Fraud prevalence:  "
            f"{split['is_fraud'].mean():.3%}"
        )

        legitimate = split.loc[
            split["is_fraud"] == 0
        ]

        for (
            mechanism,
            feature,
        ) in feature_checks.items():

            mechanism_rows = split.loc[
                (
                    split[
                        "is_fraud"
                    ]
                    == 1
                )
                & (
                    split[
                        "fraud_mechanism"
                    ]
                    == mechanism
                )
            ]

            if mechanism_rows.empty:
                continue

            fraud_median = (
                mechanism_rows[
                    feature
                ].median()
            )

            legit_median = (
                legitimate[
                    feature
                ].median()
            )

            status = (
                "PASS"
                if (
                    pd.notna(
                        fraud_median
                    )
                    and pd.notna(
                        legit_median
                    )
                    and (
                        fraud_median
                        > legit_median
                    )
                )
                else "REVIEW"
            )

            print(
                f"  {mechanism:<28} "
                f"{feature:<34} "
                f"{status}"
            )


# =============================================================================
# Complete report
# =============================================================================


def print_generation_report(
    bundle: SyntheticDataBundle,
    config: dict,
) -> None:
    print("\n" + "=" * 72)
    print("GENERATED TABLES")
    print("=" * 72)

    _print_table_summary(
        "customers",
        bundle.customers,
    )

    _print_table_summary(
        "providers",
        bundle.providers,
    )

    _print_table_summary(
        "policies",
        bundle.policies,
    )

    _print_table_summary(
        "claims",
        bundle.claims,
    )

    _print_claim_summary(
        bundle.claims
    )

    _print_service_distribution(
        bundle.claims
    )

    _print_fraud_distribution(
        bundle.claims
    )

    _print_history_sanity(
        bundle.claims
    )

    _print_fraud_signal_sanity(
        bundle.claims
    )

    _print_split_signal_summary(
        bundle.claims,
        config,
    )

    _print_temporal_summary(
        bundle.claims
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    config_path = (
        args.config.resolve()
    )

    print("=" * 72)
    print(
        "HEALTH FRAUD SYNTHETIC "
        "DATA GENERATION"
    )
    print("=" * 72)

    try:
        display_config = (
            config_path.relative_to(
                PROJECT_ROOT
            )
        )
    except ValueError:
        display_config = (
            config_path
        )

    print(
        f"Configuration: "
        f"{display_config}"
    )

    config = load_config(
        config_path
    )

    output_paths = (
        _resolve_output_paths(
            config
        )
    )

    _check_existing_files(
        output_paths,
        overwrite=args.overwrite,
    )

    print(
        f"Random seed:   "
        f"{config['project']['random_seed']}"
    )

    print(
        f"Target claims: "
        f"{config['simulation']['target_claims']:,}"
    )

    print(
        f"Target fraud:  "
        f"{config['fraud']['target_prevalence']:.3%}"
    )

    print(
        "\nGenerating synthetic data..."
    )

    bundle = (
        generate_synthetic_data(
            config_path=config_path
        )
    )

    print_generation_report(
        bundle,
        config,
    )

    print(
        "\nSaving Parquet files..."
    )

    _save_parquet(
        bundle,
        output_paths,
    )

    print(
        "\nGenerated files:"
    )

    for (
        name,
        path,
    ) in output_paths.items():

        size_mb = (
            path.stat().st_size
            / 1024**2
        )

        print(
            f"  {name:<12} "
            f"{path.relative_to(PROJECT_ROOT)} "
            f"({size_mb:.2f} MB)"
        )

    print(
        "\nGeneration completed "
        "successfully."
    )


if __name__ == "__main__":
    main()
