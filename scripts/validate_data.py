from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from health_fraud.data.validation import (  # noqa: E402
    print_validation_report,
    validate_dataset_bundle,
)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate health insurance fraud datasets."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic",
        help=(
            "Directory containing customers.parquet, "
            "providers.parquet, policies.parquet "
            "and claims.parquet."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Loading
# =============================================================================


def load_tables(
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    required_files = {
        "customers": "customers.parquet",
        "providers": "providers.parquet",
        "policies": "policies.parquet",
        "claims": "claims.parquet",
    }

    tables: dict[str, pd.DataFrame] = {}

    for name, filename in required_files.items():
        path = data_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {path}"
            )

        tables[name] = pd.read_parquet(path)

    return tables


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir

    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    data_dir = data_dir.resolve()

    print("=" * 72)
    print("HEALTH FRAUD DATA VALIDATION")
    print("=" * 72)

    try:
        display_path = data_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = data_dir

    print(f"Dataset directory: {display_path}")

    print("\nLoading datasets...")

    tables = load_tables(
        data_dir=data_dir,
    )

    print(
        f"Customers: {len(tables['customers']):,}"
    )

    print(
        f"Providers: {len(tables['providers']):,}"
    )

    print(
        f"Policies:  {len(tables['policies']):,}"
    )

    print(
        f"Claims:    {len(tables['claims']):,}"
    )

    reports = validate_dataset_bundle(
        customers=tables["customers"],
        providers=tables["providers"],
        policies=tables["policies"],
        claims=tables["claims"],
    )

    print_validation_report(
        reports
    )

    total_errors = sum(
        report.error_count
        for report in reports.values()
    )

    total_warnings = sum(
        report.warning_count
        for report in reports.values()
    )

    print("\nVALIDATION STATUS")

    if total_errors == 0:
        print("PASS - no blocking data-quality errors detected.")
    else:
        print(
            "FAIL - "
            f"{total_errors:,} blocking validation errors detected."
        )

    if total_warnings:
        print(
            f"{total_warnings:,} non-blocking warnings detected."
        )


if __name__ == "__main__":
    main()