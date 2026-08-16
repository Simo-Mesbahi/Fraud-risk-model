from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from health_fraud.models.predict import FraudScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score health insurance claims "
            "with the frozen fraud model."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "interim"
            / "claims.parquet"
        ),
        help="Input Parquet file containing claims.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "predictions"
            / "claim_scores.parquet"
        ),
        help="Output Parquet file for fraud scores.",
    )

    parser.add_argument(
        "--review-fraction",
        type=float,
        default=0.03,
        help=(
            "Fraction of highest-risk claims "
            "to select for review."
        ),
    )

    parser.add_argument(
        "--top-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "predictions"
            / "top_review_claims.parquet"
        ),
        help=(
            "Output file containing the "
            "highest-risk claims."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = (
        PROJECT_ROOT
        / "artifacts"
        / "models"
        / "health_fraud_xgboost.joblib"
    )

    preprocessor_path = (
        PROJECT_ROOT
        / "artifacts"
        / "preprocessors"
        / "health_fraud_preprocessor.joblib"
    )

    metadata_path = (
        PROJECT_ROOT
        / "artifacts"
        / "metadata"
        / "health_fraud_model_metadata.json"
    )

    input_path = args.input

    if not input_path.is_absolute():
        input_path = (
            PROJECT_ROOT
            / input_path
        )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: "
            f"{input_path}"
        )

    print("=" * 72)
    print("HEALTH FRAUD CLAIM SCORING")
    print("=" * 72)

    print(
        f"Input dataset: "
        f"{input_path}"
    )

    claims = pd.read_parquet(
        input_path
    )

    print(
        f"Claims loaded: "
        f"{len(claims):,}"
    )

    scorer = FraudScorer(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        metadata_path=metadata_path,
    )

    print(
        f"Model: "
        f"{scorer.metadata['model_name']}"
    )

    print(
        f"Version: "
        f"{scorer.metadata['model_version']}"
    )

    scored = scorer.rank(
        dataframe=claims,
        claim_id_column="claim_id",
    )

    output_path = args.output

    if not output_path.is_absolute():
        output_path = (
            PROJECT_ROOT
            / output_path
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored.to_parquet(
        output_path,
        index=False,
    )

    top_review = scorer.select_top_fraction(
        dataframe=claims,
        review_fraction=args.review_fraction,
        claim_id_column="claim_id",
    )

    top_output_path = args.top_output

    if not top_output_path.is_absolute():
        top_output_path = (
            PROJECT_ROOT
            / top_output_path
        )

    top_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    top_review.to_parquet(
        top_output_path,
        index=False,
    )

    print("\n" + "=" * 72)
    print("SCORING SUMMARY")
    print("=" * 72)

    print(
        f"Claims scored:           "
        f"{len(scored):,}"
    )

    print(
        f"Mean fraud risk:         "
        f"{scored['fraud_risk_score'].mean():.3%}"
    )

    print(
        f"Median fraud risk:       "
        f"{scored['fraud_risk_score'].median():.3%}"
    )

    print(
        f"Maximum fraud risk:      "
        f"{scored['fraud_risk_score'].max():.3%}"
    )

    print(
        f"Review fraction:         "
        f"{args.review_fraction:.2%}"
    )

    print(
        f"Claims selected:         "
        f"{len(top_review):,}"
    )

    print(
        f"\nScores saved to:\n"
        f"{output_path}"
    )

    print(
        f"\nReview queue saved to:\n"
        f"{top_output_path}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()