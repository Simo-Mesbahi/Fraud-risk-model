from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from health_fraud.data.clean import clean_claims
from health_fraud.data.load import (
    load_synthetic_tables,
    save_interim_tables,
)


def main() -> None:
    tables = load_synthetic_tables(
        PROJECT_ROOT / "data" / "synthetic"
    )

    claims_clean, claims_rejected = clean_claims(
        tables["claims"]
    )

    interim_tables = {
        "customers": tables["customers"],
        "providers": tables["providers"],
        "policies": tables["policies"],
        "claims": claims_clean,
        "claims_rejected": claims_rejected,
    }

    save_interim_tables(
        interim_tables,
        PROJECT_ROOT / "data" / "interim",
    )

    print("=" * 72)
    print("DATA CLEANING REPORT")
    print("=" * 72)

    print(f"Input claims:       {len(tables['claims']):,}")
    print(f"Clean claims:       {len(claims_clean):,}")
    print(f"Rejected claims:    {len(claims_rejected):,}")

    print("\nMissing values intentionally preserved:")

    print(
        f"provider_id:        "
        f"{claims_clean['provider_id'].isna().sum():,}"
    )

    print(
        f"has_prescription:   "
        f"{claims_clean['has_prescription'].isna().sum():,}"
    )

    print(
        f"document_count:     "
        f"{claims_clean['document_count'].isna().sum():,}"
    )

    print("\nSaved to data/interim/")

    print("=" * 72)


if __name__ == "__main__":
    main()