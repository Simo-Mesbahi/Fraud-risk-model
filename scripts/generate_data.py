from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    output_dir = PROJECT_ROOT / output_cfg["synthetic_dir"]
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        name: output_dir / filename
        for name, filename in output_cfg["files"].items()
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
            "Run again with --overwrite if replacement is intentional."
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
        path = paths[name]

        dataframe.to_parquet(
            path,
            index=False,
            engine="pyarrow",
        )


# =============================================================================
# Quality summary
# =============================================================================

def _print_table_summary(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    memory_mb = (
        dataframe.memory_usage(
            index=True,
            deep=True,
        ).sum()
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
        f"Claims:                  {len(claims):,}"
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
        f"Fraud cases:             {fraud_count:,}"
    )

    print(
        f"Fraud prevalence:        {fraud_rate:.3%}"
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
        claims["claim_id"].duplicated().sum()
    )

    print(
        f"Duplicate claim IDs:     "
        f"{duplicate_claim_ids:,}"
    )

    invalid_amounts = int(
        (claims["claim_amount"] <= 0).sum()
    )

    invalid_units = int(
        (claims["service_units"] < 1).sum()
    )

    invalid_dates = int(
        (
            claims["service_date"]
            > claims["claim_submission_date"]
        ).sum()
    )

    print(
        f"Invalid amounts:         {invalid_amounts:,}"
    )

    print(
        f"Invalid service units:   {invalid_units:,}"
    )

    print(
        f"Invalid service dates:   {invalid_dates:,}"
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
        .sort_values(ascending=False)
    )

    for service, percentage in distribution.items():
        print(
            f"{service:<22} {percentage:>7.2f}%"
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
        print("No fraudulent claims generated.")
        return

    mechanisms = (
        fraud["fraud_mechanism"]
        .value_counts()
    )

    for mechanism, count in mechanisms.items():
        percentage = count / len(fraud)

        print(
            f"{mechanism:<30}"
            f"{count:>7,}   "
            f"{percentage:>7.2%}"
        )

    print("\nFraud difficulty:")

    difficulty = (
        fraud["fraud_difficulty"]
        .value_counts(normalize=True)
    )

    for level, percentage in difficulty.items():
        print(
            f"{level:<12} {percentage:>7.2%}"
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


def print_generation_report(
    bundle: SyntheticDataBundle,
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
        bundle.claims,
    )

    _print_service_distribution(
        bundle.claims,
    )

    _print_fraud_distribution(
        bundle.claims,
    )

    _print_temporal_summary(
        bundle.claims,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    config_path = args.config.resolve()

    print("=" * 72)
    print("HEALTH FRAUD SYNTHETIC DATA GENERATION")
    print("=" * 72)

    print(
        f"Configuration: "
        f"{config_path.relative_to(PROJECT_ROOT)}"
    )

    config = load_config(
        config_path
    )

    output_paths = _resolve_output_paths(
        config
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

    print("\nGenerating synthetic data...")

    bundle = generate_synthetic_data(
        config_path=config_path
    )

    print_generation_report(
        bundle
    )

    print("\nSaving Parquet files...")

    _save_parquet(
        bundle,
        output_paths,
    )

    print("\nGenerated files:")

    for name, path in output_paths.items():
        size_mb = (
            path.stat().st_size
            / 1024**2
        )

        print(
            f"  {name:<12} "
            f"{path.relative_to(PROJECT_ROOT)} "
            f"({size_mb:.2f} MB)"
        )

    print("\nGeneration completed successfully.")


if __name__ == "__main__":
    main()