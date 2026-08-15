from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# Data containers
# =============================================================================


@dataclass(frozen=True)
class SyntheticDataBundle:
    customers: pd.DataFrame
    providers: pd.DataFrame
    policies: pd.DataFrame
    claims: pd.DataFrame


# =============================================================================
# Configuration
# =============================================================================


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")

    return config


# =============================================================================
# Generic helpers
# =============================================================================


def _validate_probabilities(
    probabilities: dict[str, float],
    name: str,
) -> None:
    values = np.asarray(
        list(probabilities.values()),
        dtype=float,
    )

    if np.any(values < 0):
        raise ValueError(
            f"{name}: probabilities cannot be negative."
        )

    total = float(values.sum())

    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"{name}: probabilities must sum to 1.0, "
            f"got {total:.8f}"
        )


def _sample_categories(
    rng: np.random.Generator,
    probabilities: dict[str, float],
    size: int,
    name: str,
) -> np.ndarray:
    _validate_probabilities(
        probabilities,
        name,
    )

    categories = np.asarray(
        list(probabilities.keys())
    )

    probs = np.asarray(
        list(probabilities.values()),
        dtype=float,
    )

    return rng.choice(
        categories,
        size=size,
        p=probs,
    )


def _sigmoid(
    x: np.ndarray,
) -> np.ndarray:
    x = np.clip(
        x,
        -30.0,
        30.0,
    )

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def _logit(
    p: float,
) -> float:
    p = float(
        np.clip(
            p,
            1e-8,
            1.0 - 1e-8,
        )
    )

    return float(
        np.log(
            p / (1.0 - p)
        )
    )


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    minimum: float,
    maximum: float,
    size: int,
) -> np.ndarray:
    values = rng.normal(
        loc=mean,
        scale=std,
        size=size,
    )

    return np.clip(
        values,
        minimum,
        maximum,
    )


def _normalize_positive_signal(
    values: np.ndarray,
    percentile: float = 95.0,
) -> np.ndarray:
    """
    Robustly normalize a non-negative fraud signal.

    This avoids one mechanism dominating only because
    its numerical scale is larger than another mechanism.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    values = np.maximum(
        values,
        0.0,
    )

    positive = values[
        values > 0
    ]

    if len(positive) == 0:
        return np.zeros_like(
            values,
            dtype=float,
        )

    scale = np.percentile(
        positive,
        percentile,
    )

    if scale <= 0:
        return np.zeros_like(
            values,
            dtype=float,
        )

    normalized = (
        values / scale
    )

    return np.clip(
        normalized,
        0.0,
        3.0,
    )


# =============================================================================
# Customers
# =============================================================================


def generate_customers(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_customers = int(
        config["simulation"]["n_customers"]
    )

    cfg = config["entities"]["customers"]

    age_cfg = cfg["age"]

    if (
        age_cfg.get("distribution")
        == "truncated_normal"
    ):
        ages = _truncated_normal(
            rng=rng,
            mean=float(
                age_cfg["mean"]
            ),
            std=float(
                age_cfg["std"]
            ),
            minimum=float(
                age_cfg["min"]
            ),
            maximum=float(
                age_cfg["max"]
            ),
            size=n_customers,
        ).round().astype(int)

    else:
        ages = rng.integers(
            age_cfg["min"],
            age_cfg["max"] + 1,
            size=n_customers,
        )

    tenure_months = rng.integers(
        cfg["tenure_months"]["min"],
        cfg["tenure_months"]["max"] + 1,
        size=n_customers,
    )

    coverage_level = _sample_categories(
        rng=rng,
        probabilities=cfg[
            "coverage_levels"
        ],
        size=n_customers,
        name="customer coverage levels",
    )

    behavior_segment = _sample_categories(
        rng=rng,
        probabilities=cfg[
            "behavior_segments"
        ],
        size=n_customers,
        name="customer behavior segments",
    )

    multiplier_map = cfg[
        "claim_frequency_multiplier"
    ]

    claim_frequency_multiplier = np.asarray(
        [
            multiplier_map[segment]
            for segment in behavior_segment
        ],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "customer_id": [
                f"CUST_{i:06d}"
                for i in range(
                    1,
                    n_customers + 1,
                )
            ],
            "customer_age": ages,
            "customer_tenure_months": tenure_months,
            "coverage_level": coverage_level,
            "customer_behavior_segment": behavior_segment,
            "claim_frequency_multiplier": (
                claim_frequency_multiplier
            ),
        }
    )


# =============================================================================
# Providers
# =============================================================================


def generate_providers(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_providers = int(
        config["simulation"]["n_providers"]
    )

    cfg = config["entities"]["providers"]

    provider_type = _sample_categories(
        rng=rng,
        probabilities=cfg[
            "provider_types"
        ],
        size=n_providers,
        name="provider types",
    )

    provider_region = _sample_categories(
        rng=rng,
        probabilities=cfg["regions"],
        size=n_providers,
        name="provider regions",
    )

    behavior_segment = _sample_categories(
        rng=rng,
        probabilities=cfg[
            "behavior_segments"
        ],
        size=n_providers,
        name="provider behavior segments",
    )

    multiplier_map = cfg[
        "claim_volume_multiplier"
    ]

    provider_volume_multiplier = (
        np.asarray(
            [
                multiplier_map[segment]
                for segment
                in behavior_segment
            ],
            dtype=float,
        )
    )

    provider_tenure_months = (
        rng.integers(
            1,
            241,
            size=n_providers,
        )
    )

    return pd.DataFrame(
        {
            "provider_id": [
                f"PROV_{i:05d}"
                for i in range(
                    1,
                    n_providers + 1,
                )
            ],
            "provider_type": provider_type,
            "provider_region": provider_region,
            "provider_tenure_months": (
                provider_tenure_months
            ),
            "provider_behavior_segment": (
                behavior_segment
            ),
            "provider_volume_multiplier": (
                provider_volume_multiplier
            ),
        }
    )


# =============================================================================
# Policies
# =============================================================================


def generate_policies(
    customers: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate policies that already exist at the beginning
    of the simulation.

    Claim-specific policy tenure will later be calculated
    using the actual claim timestamp.
    """

    n = len(customers)

    simulation_start = pd.Timestamp(
        config["simulation"]["start_date"]
    )

    max_prior_tenure = np.minimum(
        customers[
            "customer_tenure_months"
        ].to_numpy(),
        120,
    )

    prior_policy_tenure = np.asarray(
        [
            rng.integers(
                1,
                int(max_tenure) + 1,
            )
            for max_tenure
            in max_prior_tenure
        ]
    )

    policy_start_date = (
        simulation_start
        - pd.to_timedelta(
            prior_policy_tenure * 30,
            unit="D",
        )
    )

    return pd.DataFrame(
        {
            "policy_id": [
                f"POL_{i:06d}"
                for i in range(
                    1,
                    n + 1,
                )
            ],
            "customer_id": (
                customers[
                    "customer_id"
                ].to_numpy()
            ),
            "coverage_level": (
                customers[
                    "coverage_level"
                ].to_numpy()
            ),
            "policy_start_date": (
                policy_start_date
            ),
            "policy_end_date": pd.NaT,
        }
    )


# =============================================================================
# Claim dates
# =============================================================================


def _build_daily_sampling_weights(
    config: dict[str, Any],
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
]:
    start = pd.Timestamp(
        config["simulation"]["start_date"]
    )

    end = pd.Timestamp(
        config["simulation"]["end_date"]
    )

    dates = pd.date_range(
        start=start,
        end=end,
        freq="D",
    )

    weights = np.ones(
        len(dates),
        dtype=float,
    )

    seasonality = (
        config["simulation"].get(
            "seasonality",
            {},
        )
    )

    if seasonality.get(
        "enabled",
        False,
    ):
        monthly_factors = (
            seasonality[
                "monthly_factors"
            ]
        )

        weights = np.asarray(
            [
                float(
                    monthly_factors[
                        f"{date.month:02d}"
                    ]
                )
                for date in dates
            ],
            dtype=float,
        )

    weights /= weights.sum()

    return dates, weights


def _generate_submission_dates(
    config: dict[str, Any],
    rng: np.random.Generator,
    size: int,
) -> pd.Series:
    dates, probabilities = (
        _build_daily_sampling_weights(
            config
        )
    )

    sampled_dates = rng.choice(
        dates.to_numpy(),
        size=size,
        p=probabilities,
    )

    timestamps = pd.to_datetime(
        sampled_dates
    )

    seconds = rng.integers(
        0,
        24 * 60 * 60,
        size=size,
    )

    timestamps = (
        timestamps
        + pd.to_timedelta(
            seconds,
            unit="s",
        )
    )

    return pd.Series(
        timestamps
    )


# =============================================================================
# Services
# =============================================================================


def _sample_service_categories(
    config: dict[str, Any],
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    probabilities = {
        service: service_cfg[
            "probability"
        ]
        for service, service_cfg
        in config["services"].items()
    }

    return _sample_categories(
        rng=rng,
        probabilities=probabilities,
        size=size,
        name="service probabilities",
    )


def _generate_service_codes(
    service_categories: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    codes = {
        "consultation": [
            "CONS_GP",
            "CONS_SPEC",
            "CONS_FOLLOWUP",
        ],
        "dental": [
            "DENT_CHECK",
            "DENT_CROWN",
            "DENT_PROSTHESIS",
            "DENT_SURGERY",
        ],
        "optical": [
            "OPT_FRAME",
            "OPT_LENSES",
            "OPT_CONTACT",
        ],
        "physiotherapy": [
            "PHYSIO_STANDARD",
            "PHYSIO_REHAB",
            "PHYSIO_SPECIAL",
        ],
        "pharmacy": [
            "PHARM_RX",
            "PHARM_DEVICE",
            "PHARM_OTHER",
        ],
        "medical_device": [
            "DEVICE_ORTHO",
            "DEVICE_HEARING",
            "DEVICE_OTHER",
        ],
        "diagnostic": [
            "DIAG_LAB",
            "DIAG_IMAGING",
            "DIAG_OTHER",
        ],
        "other": [
            "OTHER_01",
            "OTHER_02",
        ],
    }

    return np.asarray(
        [
            rng.choice(
                codes[category]
            )
            for category
            in service_categories
        ]
    )


def _generate_claim_amounts(
    service_categories: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    amounts = np.zeros(
        len(service_categories),
        dtype=float,
    )

    noise_std = float(
        config["claims"].get(
            "claim_amount_noise_std",
            0.0,
        )
    )

    for (
        service_name,
        service_cfg,
    ) in config[
        "services"
    ].items():

        mask = (
            service_categories
            == service_name
        )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        cfg = service_cfg[
            "amount"
        ]

        if (
            cfg["distribution"]
            != "lognormal"
        ):
            raise ValueError(
                "Unsupported distribution "
                f"for {service_name}: "
                f"{cfg['distribution']}"
            )

        raw = rng.lognormal(
            mean=np.log(
                float(
                    cfg["median"]
                )
            ),
            sigma=float(
                cfg["sigma"]
            ),
            size=count,
        )

        if noise_std > 0:
            multiplier = (
                rng.normal(
                    loc=1.0,
                    scale=noise_std,
                    size=count,
                )
            )

            multiplier = np.clip(
                multiplier,
                0.5,
                1.5,
            )

            raw *= multiplier

        raw = np.clip(
            raw,
            cfg["min"],
            cfg["max"],
        )

        amounts[mask] = raw

    return np.round(
        amounts,
        2,
    )


def _generate_service_units(
    service_categories: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    units = np.zeros(
        len(service_categories),
        dtype=int,
    )

    for (
        service_name,
        service_cfg,
    ) in config[
        "services"
    ].items():

        mask = (
            service_categories
            == service_name
        )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        units[mask] = (
            rng.integers(
                service_cfg[
                    "units"
                ]["min"],
                service_cfg[
                    "units"
                ]["max"] + 1,
                size=count,
            )
        )

    return units


# =============================================================================
# Entity sampling
# =============================================================================


def _sample_customer_indices(
    customers: pd.DataFrame,
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    weights = customers[
        "claim_frequency_multiplier"
    ].to_numpy(
        dtype=float
    )

    weights /= weights.sum()

    return rng.choice(
        np.arange(
            len(customers)
        ),
        size=size,
        p=weights,
    )


SERVICE_PROVIDER_TYPES = {
    "consultation": {
        "general_practitioner",
        "specialist",
    },
    "dental": {
        "dentist",
    },
    "optical": {
        "optician",
    },
    "physiotherapy": {
        "physiotherapist",
    },
    "pharmacy": {
        "pharmacy",
    },
    "medical_device": {
        "medical_supplier",
        "pharmacy",
    },
    "diagnostic": {
        "laboratory",
        "specialist",
    },
    "other": {
        "other",
        "general_practitioner",
        "specialist",
    },
}


def _sample_provider_indices(
    providers: pd.DataFrame,
    service_categories: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample healthcare providers conditional on service type.

    This prevents unrealistic combinations such as
    a dental crown being billed by a pharmacy.
    """

    result = np.empty(
        len(service_categories),
        dtype=int,
    )

    provider_types = providers[
        "provider_type"
    ].to_numpy()

    provider_weights = providers[
        "provider_volume_multiplier"
    ].to_numpy(
        dtype=float
    )

    for service in np.unique(
        service_categories
    ):
        claim_mask = (
            service_categories
            == service
        )

        allowed_types = (
            SERVICE_PROVIDER_TYPES[
                service
            ]
        )

        provider_mask = np.isin(
            provider_types,
            list(allowed_types),
        )

        candidate_indices = (
            np.flatnonzero(
                provider_mask
            )
        )

        if len(
            candidate_indices
        ) == 0:
            raise ValueError(
                "No providers available "
                f"for service {service}."
            )

        weights = (
            provider_weights[
                candidate_indices
            ]
        )

        weights = (
            weights / weights.sum()
        )

        result[
            claim_mask
        ] = rng.choice(
            candidate_indices,
            size=int(
                claim_mask.sum()
            ),
            p=weights,
        )

    return result


# =============================================================================
# Claims
# =============================================================================


def generate_claims(
    customers: pd.DataFrame,
    providers: pd.DataFrame,
    policies: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_claims = int(
        config["simulation"][
            "target_claims"
        ]
    )

    customer_idx = (
        _sample_customer_indices(
            customers=customers,
            rng=rng,
            size=n_claims,
        )
    )

    selected_customers = (
        customers
        .iloc[customer_idx]
        .reset_index(
            drop=True
        )
    )

    selected_policies = (
        policies
        .iloc[customer_idx]
        .reset_index(
            drop=True
        )
    )

    service_category = (
        _sample_service_categories(
            config=config,
            rng=rng,
            size=n_claims,
        )
    )

    provider_idx = (
        _sample_provider_indices(
            providers=providers,
            service_categories=(
                service_category
            ),
            rng=rng,
        )
    )

    selected_providers = (
        providers
        .iloc[provider_idx]
        .reset_index(
            drop=True
        )
    )

    service_code = (
        _generate_service_codes(
            service_categories=(
                service_category
            ),
            rng=rng,
        )
    )

    claim_amount = (
        _generate_claim_amounts(
            service_categories=(
                service_category
            ),
            config=config,
            rng=rng,
        )
    )

    service_units = (
        _generate_service_units(
            service_categories=(
                service_category
            ),
            config=config,
            rng=rng,
        )
    )

    submission_timestamp = (
        _generate_submission_dates(
            config=config,
            rng=rng,
            size=n_claims,
        )
    )

    delay_cfg = config[
        "claims"
    ][
        "service_to_submission_days"
    ]

    delays = rng.gamma(
        shape=float(
            delay_cfg["shape"]
        ),
        scale=float(
            delay_cfg["scale"]
        ),
        size=n_claims,
    )

    delays = np.minimum(
        np.floor(
            delays
        ).astype(int),
        int(
            delay_cfg[
                "max_days"
            ]
        ),
    )

    service_date = (
        submission_timestamp
        .dt.normalize()
        - pd.to_timedelta(
            delays,
            unit="D",
        )
    )

    channels = (
        _sample_categories(
            rng=rng,
            probabilities=config[
                "claims"
            ][
                "submission_channels"
            ],
            size=n_claims,
            name=(
                "submission channels"
            ),
        )
    )

    coverage_cfg = config[
        "policy"
    ][
        "coverage_limits"
    ]

    coverage_limits = np.asarray(
        [
            coverage_cfg[
                coverage
            ][
                service
            ]
            for coverage, service
            in zip(
                selected_policies[
                    "coverage_level"
                ],
                service_category,
            )
        ],
        dtype=float,
    )

    reimbursement_rate = (
        rng.beta(
            a=8,
            b=2,
            size=n_claims,
        )
    )

    requested_reimbursement = (
        np.minimum(
            claim_amount
            * reimbursement_rate,
            coverage_limits,
        )
    )

    document_cfg = config[
        "claims"
    ][
        "document_count"
    ]

    document_count = (
        rng.integers(
            document_cfg["min"],
            document_cfg["max"]
            + 1,
            size=n_claims,
        )
        .astype(float)
    )

    has_invoice = (
        rng.random(
            n_claims
        )
        < float(
            config[
                "claims"
            ][
                "has_invoice_probability"
            ]
        )
    )

    prescription_probs = (
        np.asarray(
            [
                config[
                    "claims"
                ][
                    "has_prescription_probability"
                ][service]
                for service
                in service_category
            ],
            dtype=float,
        )
    )

    has_prescription = (
        rng.random(
            n_claims
        )
        < prescription_probs
    ).astype(object)

    policy_start_dates = (
        pd.to_datetime(
            selected_policies[
                "policy_start_date"
            ]
        )
    )

    policy_tenure_months = (
        (
            submission_timestamp
            - policy_start_dates
        )
        .dt.days
        .div(30.4375)
        .clip(lower=1)
        .astype(int)
    )

    # Claim-time policy change.
    policy_cfg = config[
        "policy"
    ]

    recent_policy_change = (
        rng.random(
            n_claims
        )
        < float(
            policy_cfg[
                "recent_change_probability"
            ]
        )
    )

    change_window = int(
        policy_cfg[
            "recent_change_window_days"
        ]
    )

    days_since_policy_change = (
        np.where(
            recent_policy_change,
            rng.integers(
                1,
                change_window + 1,
                size=n_claims,
            ),
            np.nan,
        )
    )

    claims = pd.DataFrame(
        {
            "claim_id": [
                f"CLM_{i:08d}"
                for i in range(
                    1,
                    n_claims + 1,
                )
            ],

            "customer_id": (
                selected_customers[
                    "customer_id"
                ].to_numpy()
            ),

            "policy_id": (
                selected_policies[
                    "policy_id"
                ].to_numpy()
            ),

            "provider_id": (
                selected_providers[
                    "provider_id"
                ].to_numpy()
            ),

            "service_category": (
                service_category
            ),

            "service_code": (
                service_code
            ),

            "service_units": (
                service_units
            ),

            "service_date": (
                service_date
            ),

            "claim_submission_date": (
                submission_timestamp
                .dt.normalize()
            ),

            "claim_submission_timestamp": (
                submission_timestamp
            ),

            "claim_amount": (
                claim_amount
            ),

            "requested_reimbursement": (
                np.round(
                    requested_reimbursement,
                    2,
                )
            ),

            "coverage_limit": (
                coverage_limits
            ),

            "submission_channel": (
                channels
            ),

            "document_count": (
                document_count
            ),

            "has_invoice": (
                has_invoice
            ),

            "has_prescription": (
                has_prescription
            ),

            "customer_age": (
                selected_customers[
                    "customer_age"
                ].to_numpy()
            ),

            "customer_tenure_months": (
                selected_customers[
                    "customer_tenure_months"
                ].to_numpy()
            ),

            "coverage_level": (
                selected_customers[
                    "coverage_level"
                ].to_numpy()
            ),

            "customer_behavior_segment": (
                selected_customers[
                    "customer_behavior_segment"
                ].to_numpy()
            ),

            "policy_tenure_months": (
                policy_tenure_months
                .to_numpy()
            ),

            "recent_policy_change": (
                recent_policy_change
            ),

            "days_since_policy_change": (
                days_since_policy_change
            ),

            "provider_type": (
                selected_providers[
                    "provider_type"
                ].to_numpy()
            ),

            "provider_region": (
                selected_providers[
                    "provider_region"
                ].to_numpy()
            ),

            "provider_tenure_months": (
                selected_providers[
                    "provider_tenure_months"
                ].to_numpy()
            ),

            "provider_behavior_segment": (
                selected_providers[
                    "provider_behavior_segment"
                ].to_numpy()
            ),
        }
    )

    claims[
        "days_service_to_submission"
    ] = (
        claims[
            "claim_submission_date"
        ]
        - claims[
            "service_date"
        ]
    ).dt.days

    claims[
        "reimbursement_ratio"
    ] = (
        claims[
            "requested_reimbursement"
        ]
        / claims[
            "claim_amount"
        ]
    ).clip(
        0,
        1,
    )

    claims = (
        claims
        .sort_values(
            "claim_submission_timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return claims


# =============================================================================
# Historical features
# =============================================================================


def _rolling_feature(
    df: pd.DataFrame,
    group_columns: str | list[str],
    value_column: str,
    window: str,
    aggregation: str,
) -> pd.Series:
    """
    Compute a strict-past rolling feature while preserving
    the original dataframe row indexes.
    """

    output = pd.Series(
        index=df.index,
        dtype=float,
    )

    grouped = df.groupby(
        group_columns,
        sort=False,
        dropna=False,
    )

    for _, group in grouped:
        ordered = (
            group
            .sort_values(
                "claim_submission_timestamp"
            )
        )

        values = (
            ordered
            .set_index(
                "claim_submission_timestamp"
            )[value_column]
            .rolling(
                window,
                closed="left",
            )
        )

        if aggregation == "count":
            result = (
                values.count()
            )

        elif aggregation == "sum":
            result = (
                values.sum()
            )

        elif aggregation == "mean":
            result = (
                values.mean()
            )

        elif aggregation == "median":
            result = (
                values.median()
            )

        else:
            raise ValueError(
                "Unsupported rolling "
                f"aggregation: {aggregation}"
            )

        output.loc[
            ordered.index
        ] = result.to_numpy()

    return output


def add_historical_features(
    claims: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    df = (
        claims
        .sort_values(
            "claim_submission_timestamp"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    # Customer counts.
    for window in (
        7,
        30,
        90,
        365,
    ):
        df[
            f"customer_claims_{window}d"
        ] = _rolling_feature(
            df=df,
            group_columns="customer_id",
            value_column="claim_id",
            window=f"{window}D",
            aggregation="count",
        )

    df[
        "customer_amount_30d"
    ] = _rolling_feature(
        df=df,
        group_columns="customer_id",
        value_column="claim_amount",
        window="30D",
        aggregation="sum",
    )

    df[
        "customer_amount_365d"
    ] = _rolling_feature(
        df=df,
        group_columns="customer_id",
        value_column="claim_amount",
        window="365D",
        aggregation="sum",
    )

    df[
        "customer_avg_claim_amount_365d"
    ] = _rolling_feature(
        df=df,
        group_columns="customer_id",
        value_column="claim_amount",
        window="365D",
        aggregation="mean",
    )

    # Previous customer claim.
    df[
        "days_since_customer_previous_claim"
    ] = (
        df.groupby(
            "customer_id"
        )[
            "claim_submission_timestamp"
        ]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    # Customer-provider history.
    df[
        "days_since_same_provider_claim"
    ] = (
        df.groupby(
            [
                "customer_id",
                "provider_id",
            ]
        )[
            "claim_submission_timestamp"
        ]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    df[
        "customer_provider_claims_30d"
    ] = _rolling_feature(
        df=df,
        group_columns=[
            "customer_id",
            "provider_id",
        ],
        value_column="claim_id",
        window="30D",
        aggregation="count",
    )

    # Customer / service.
    df[
        "same_service_claims_30d"
    ] = _rolling_feature(
        df=df,
        group_columns=[
            "customer_id",
            "service_category",
        ],
        value_column="claim_id",
        window="30D",
        aggregation="count",
    )

    # Provider.
    df[
        "provider_claims_30d"
    ] = _rolling_feature(
        df=df,
        group_columns="provider_id",
        value_column="claim_id",
        window="30D",
        aggregation="count",
    )

    df[
        "provider_claims_90d"
    ] = _rolling_feature(
        df=df,
        group_columns="provider_id",
        value_column="claim_id",
        window="90D",
        aggregation="count",
    )

    df[
        "provider_avg_claim_amount_90d"
    ] = _rolling_feature(
        df=df,
        group_columns="provider_id",
        value_column="claim_amount",
        window="90D",
        aggregation="mean",
    )

    # ---------------------------------------------------------
    # Service baseline
    #
    # IMPORTANT:
    # never use a full-dataset median here because that would
    # contain future information.
    # ---------------------------------------------------------

    historical_service_median = (
        df.groupby(
            "service_code"
        )[
            "claim_amount"
        ]
        .transform(
            lambda series:
            series
            .shift(1)
            .expanding()
            .median()
        )
    )

    configured_service_prior = {
        service: float(
            cfg["amount"]["median"]
        )
        for service, cfg
        in config["services"].items()
    }

    service_prior = (
        df[
            "service_category"
        ]
        .map(
            configured_service_prior
        )
        .astype(float)
    )

    df[
        "service_typical_amount"
    ] = (
        historical_service_median
        .fillna(
            service_prior
        )
    )

    df[
        "claim_to_service_median_ratio"
    ] = (
        df["claim_amount"]
        / df[
            "service_typical_amount"
        ]
    )

    df[
        "claim_to_customer_avg_ratio"
    ] = (
        df["claim_amount"]
        / df[
            "customer_avg_claim_amount_365d"
        ]
    )

    df[
        "claim_to_provider_avg_ratio"
    ] = (
        df["claim_amount"]
        / df[
            "provider_avg_claim_amount_90d"
        ]
    )

    ratio_columns = [
        "claim_to_service_median_ratio",
        "claim_to_customer_avg_ratio",
        "claim_to_provider_avg_ratio",
    ]

    df[
        ratio_columns
    ] = (
        df[
            ratio_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    return df


# =============================================================================
# Legitimate anomalies
# =============================================================================


def add_legitimate_anomalies(
    claims: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Mark naturally unusual legitimate-looking observations.

    We deliberately do not mutate historical aggregates:
    that would make the synthetic data internally inconsistent.

    These rows become hard negatives during target generation.
    """

    df = claims.copy()

    cfg = config[
        "legitimate_anomalies"
    ]

    df[
        "legitimate_anomaly"
    ] = False

    df[
        "legitimate_anomaly_type"
    ] = "none"

    if not cfg.get(
        "enabled",
        False,
    ):
        return df

    n_target = int(
        round(
            len(df)
            * float(
                cfg["prevalence"]
            )
        )
    )

    if n_target <= 0:
        return df

    amount_signal = (
        df[
            "claim_to_service_median_ratio"
        ]
        .fillna(1.0)
        .clip(lower=1.0)
        .to_numpy()
        - 1.0
    )

    frequency_signal = (
        df[
            "customer_claims_30d"
        ]
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    provider_signal = (
        df[
            "provider_claims_30d"
        ]
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    repeated_signal = (
        df[
            "same_service_claims_30d"
        ]
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    amount_signal = (
        _normalize_positive_signal(
            amount_signal
        )
    )

    frequency_signal = (
        _normalize_positive_signal(
            frequency_signal
        )
    )

    provider_signal = (
        _normalize_positive_signal(
            provider_signal
        )
    )

    repeated_signal = (
        _normalize_positive_signal(
            repeated_signal
        )
    )

    combined = (
        0.30 * amount_signal
        + 0.30 * frequency_signal
        + 0.20 * provider_signal
        + 0.20 * repeated_signal
        + 0.05
    )

    probability = (
        combined
        / combined.sum()
    )

    selected = rng.choice(
        np.arange(
            len(df)
        ),
        size=min(
            n_target,
            len(df),
        ),
        replace=False,
        p=probability,
    )

    pattern_probabilities = cfg[
        "patterns"
    ]

    anomaly_types = (
        _sample_categories(
            rng=rng,
            probabilities=(
                pattern_probabilities
            ),
            size=len(selected),
            name=(
                "legitimate anomaly "
                "patterns"
            ),
        )
    )

    df.loc[
        selected,
        "legitimate_anomaly",
    ] = True

    df.loc[
        selected,
        "legitimate_anomaly_type",
    ] = anomaly_types

    return df


# =============================================================================
# Fraud generation
# =============================================================================


def _calibrate_probability_mean(
    latent_without_intercept: np.ndarray,
    target_mean: float,
    minimum_probability: float,
    maximum_probability: float,
) -> tuple[
    np.ndarray,
    float,
]:
    """
    Find an intercept through binary search so the mean
    predicted synthetic fraud probability matches the target.
    """

    low = -20.0
    high = 5.0

    for _ in range(
        100
    ):
        intercept = (
            low + high
        ) / 2.0

        probabilities = (
            _sigmoid(
                latent_without_intercept
                + intercept
            )
        )

        probabilities = np.clip(
            probabilities,
            minimum_probability,
            maximum_probability,
        )

        mean_probability = float(
            probabilities.mean()
        )

        if (
            mean_probability
            < target_mean
        ):
            low = intercept
        else:
            high = intercept

    intercept = (
        low + high
    ) / 2.0

    probabilities = np.clip(
        _sigmoid(
            latent_without_intercept
            + intercept
        ),
        minimum_probability,
        maximum_probability,
    )

    return (
        probabilities,
        intercept,
    )


def _sample_fraud_mechanisms(
    df: pd.DataFrame,
    target: np.ndarray,
    signals: dict[
        str,
        np.ndarray,
    ],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Assign a synthetic ground-truth mechanism to positive
    fraud cases.

    Propensities depend both on configured mechanism weights
    and observed signal intensity.

    This prevents one mechanism from trivially accounting
    for ~90% of fraud.
    """

    result = np.full(
        len(df),
        "none",
        dtype=object,
    )

    fraud_indices = np.flatnonzero(
        target == 1
    )

    if len(
        fraud_indices
    ) == 0:
        return result

    mechanisms_cfg = config[
        "fraud"
    ][
        "mechanisms"
    ]

    base_names = [
        "frequency_abuse",
        "amount_inflation",
        "repeated_service",
        "provider_abnormality",
        "customer_provider_pattern",
    ]

    mechanism_names = (
        base_names
        + ["mixed_pattern"]
    )

    propensities = []

    for mechanism in base_names:
        base_weight = float(
            mechanisms_cfg[
                mechanism
            ][
                "prevalence_weight"
            ]
        )

        signal = signals[
            mechanism
        ][
            fraud_indices
        ]

        # Every mechanism remains possible,
        # while stronger evidence increases
        # its assignment probability.
        propensity = (
            base_weight
            * (
                0.40
                + signal
            )
        )

        propensities.append(
            propensity
        )

    stacked_base = np.column_stack(
        [
            signals[name][
                fraud_indices
            ]
            for name in base_names
        ]
    )

    sorted_signals = np.sort(
        stacked_base,
        axis=1,
    )

    strongest_two = (
        sorted_signals[
            :,
            -1
        ]
        + sorted_signals[
            :,
            -2
        ]
    )

    mixed_weight = float(
        mechanisms_cfg[
            "mixed_pattern"
        ][
            "prevalence_weight"
        ]
    )

    mixed_propensity = (
        mixed_weight
        * (
            0.50
            + strongest_two
        )
    )

    propensities.append(
        mixed_propensity
    )

    propensity_matrix = (
        np.column_stack(
            propensities
        )
    )

    propensity_matrix = (
        propensity_matrix
        / propensity_matrix.sum(
            axis=1,
            keepdims=True,
        )
    )

    cumulative = np.cumsum(
        propensity_matrix,
        axis=1,
    )

    draws = rng.random(
        len(fraud_indices)
    )

    selected_position = (
        (
            draws[
                :,
                None
            ]
            > cumulative
        )
        .sum(
            axis=1
        )
    )

    selected_mechanism = (
        np.asarray(
            mechanism_names,
            dtype=object,
        )[
            selected_position
        ]
    )

    result[
        fraud_indices
    ] = selected_mechanism

    return result


def add_fraud_target(
    claims: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = claims.copy()

    fraud_cfg = config[
        "fraud"
    ]

    thresholds = fraud_cfg[
        "thresholds"
    ]

    n = len(df)

    # ---------------------------------------------------------
    # Raw risk signals
    # ---------------------------------------------------------

    frequency_raw = np.maximum(
        (
            df[
                "customer_claims_30d"
            ]
            .fillna(0)
            .to_numpy(
                dtype=float
            )
            - float(
                thresholds[
                    "high_recent_claims_30d"
                ]
            )
        ),
        0,
    )

    amount_raw = np.maximum(
        (
            df[
                "claim_to_service_median_ratio"
            ]
            .fillna(1.0)
            .to_numpy(
                dtype=float
            )
            - float(
                thresholds[
                    "high_claim_to_service_ratio"
                ]
            )
        ),
        0,
    )

    repeated_raw = np.maximum(
        (
            df[
                "same_service_claims_30d"
            ]
            .fillna(0)
            .to_numpy(
                dtype=float
            )
            - float(
                thresholds[
                    "high_same_service_claims_30d"
                ]
            )
        ),
        0,
    )

    provider_recent = (
        df[
            "provider_claims_30d"
        ]
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    provider_90 = (
        df[
            "provider_claims_90d"
        ]
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    expected_30 = (
        provider_90 / 3.0
    )

    provider_growth = (
        provider_recent
        + 1.0
    ) / (
        expected_30
        + 1.0
    )

    provider_raw = np.maximum(
        provider_growth
        - float(
            thresholds[
                "high_provider_growth_ratio"
            ]
        ),
        0,
    )

    pair_raw = np.maximum(
        (
            df[
                "customer_provider_claims_30d"
            ]
            .fillna(0)
            .to_numpy(
                dtype=float
            )
            - float(
                thresholds[
                    "high_customer_provider_claims_30d"
                ]
            )
        ),
        0,
    )

    # ---------------------------------------------------------
    # Normalize each mechanism independently.
    #
    # This is the key fix preventing count-based signals from
    # numerically overwhelming the other fraud mechanisms.
    # ---------------------------------------------------------

    frequency_signal = (
        _normalize_positive_signal(
            np.log1p(
                frequency_raw
            )
        )
    )

    amount_signal = (
        _normalize_positive_signal(
            np.log1p(
                amount_raw
            )
        )
    )

    repeated_signal = (
        _normalize_positive_signal(
            np.log1p(
                repeated_raw
            )
        )
    )

    provider_signal = (
        _normalize_positive_signal(
            np.log1p(
                provider_raw
            )
        )
    )

    pair_signal = (
        _normalize_positive_signal(
            np.log1p(
                pair_raw
            )
        )
    )

    signals = {
        "frequency_abuse": (
            frequency_signal
        ),
        "amount_inflation": (
            amount_signal
        ),
        "repeated_service": (
            repeated_signal
        ),
        "provider_abnormality": (
            provider_signal
        ),
        "customer_provider_pattern": (
            pair_signal
        ),
    }

    # ---------------------------------------------------------
    # Difficulty
    # ---------------------------------------------------------

    difficulty = (
        _sample_categories(
            rng=rng,
            probabilities=fraud_cfg[
                "difficulty_mix"
            ],
            size=n,
            name="fraud difficulty mix",
        )
    )

    mechanisms_cfg = fraud_cfg[
        "mechanisms"
    ]

    contribution_arrays: dict[
        str,
        np.ndarray,
    ] = {}

    for (
        mechanism,
        signal,
    ) in signals.items():

        cfg = mechanisms_cfg[
            mechanism
        ]

        strengths = cfg[
            "signal_strength"
        ]

        row_strength = np.asarray(
            [
                float(
                    strengths[level]
                )
                for level
                in difficulty
            ],
            dtype=float,
        )

        prevalence_weight = float(
            cfg[
                "prevalence_weight"
            ]
        )

        contribution_arrays[
            mechanism
        ] = (
            prevalence_weight
            * row_strength
            * signal
        )

    # ---------------------------------------------------------
    # Concept drift by mechanism
    # ---------------------------------------------------------

    drift_cfg = config[
        "concept_drift"
    ]

    if drift_cfg.get(
        "enabled",
        False,
    ):
        drift_start = pd.Timestamp(
            drift_cfg[
                "start_date"
            ]
        )

        drift_mask = (
            df[
                "claim_submission_timestamp"
            ]
            >= drift_start
        ).to_numpy()

        for mechanism, drift_spec in (
            drift_cfg[
                "mechanisms"
            ].items()
        ):
            if (
                mechanism
                in contribution_arrays
            ):
                multiplier = float(
                    drift_spec[
                        "multiplier"
                    ]
                )

                contribution_arrays[
                    mechanism
                ][drift_mask] *= (
                    multiplier
                )

    # ---------------------------------------------------------
    # Combine mechanism contributions
    # ---------------------------------------------------------

    latent = np.zeros(
        n,
        dtype=float,
    )

    for contribution in (
        contribution_arrays.values()
    ):
        latent += contribution

    # ---------------------------------------------------------
    # Interactions
    # ---------------------------------------------------------

    interaction_cfg = (
        fraud_cfg[
            "interactions"
        ]
    )

    if interaction_cfg.get(
        "enabled",
        False,
    ):
        definitions = (
            interaction_cfg[
                "definitions"
            ]
        )

        latent += (
            float(
                definitions[
                    "amount_and_frequency"
                ]["weight"]
            )
            * amount_signal
            * frequency_signal
        )

        latent += (
            float(
                definitions[
                    "provider_and_customer"
                ]["weight"]
            )
            * provider_signal
            * pair_signal
        )

        latent += (
            float(
                definitions[
                    "repeated_service_and_amount"
                ]["weight"]
            )
            * repeated_signal
            * amount_signal
        )

        policy_signal = (
            df[
                "recent_policy_change"
            ]
            .fillna(False)
            .astype(float)
            .to_numpy()
        )

        latent += (
            float(
                definitions[
                    "policy_change_and_high_amount"
                ]["weight"]
            )
            * policy_signal
            * amount_signal
        )

    # ---------------------------------------------------------
    # Legitimate anomalies = hard negatives
    # ---------------------------------------------------------

    legitimate_anomaly = (
        df[
            "legitimate_anomaly"
        ]
        .fillna(False)
        .to_numpy(
            dtype=bool
        )
    )

    latent[
        legitimate_anomaly
    ] -= 1.75

    # ---------------------------------------------------------
    # Concept drift prevalence shift
    # ---------------------------------------------------------

    if drift_cfg.get(
        "enabled",
        False,
    ):
        drift_start = pd.Timestamp(
            drift_cfg[
                "start_date"
            ]
        )

        drift_mask = (
            df[
                "claim_submission_timestamp"
            ]
            >= drift_start
        ).to_numpy()

        latent[
            drift_mask
        ] += np.log(
            float(
                drift_cfg[
                    "prevalence_multiplier"
                ]
            )
        )

    # ---------------------------------------------------------
    # Latent uncertainty
    # ---------------------------------------------------------

    latent += rng.normal(
        loc=0.0,
        scale=float(
            fraud_cfg[
                "noise"
            ][
                "latent_score_std"
            ]
        ),
        size=n,
    )

    # ---------------------------------------------------------
    # Calibration
    #
    # Label flips are applied later. If q is the flip rate:
    #
    # final prevalence
    # = p(1-q) + (1-p)q
    #
    # therefore solve for the required pre-flip prevalence.
    # ---------------------------------------------------------

    target_final = float(
        fraud_cfg[
            "target_prevalence"
        ]
    )

    flip_probability = float(
        fraud_cfg[
            "noise"
        ][
            "label_flip_probability"
        ]
    )

    if flip_probability >= 0.5:
        raise ValueError(
            "label_flip_probability "
            "must be < 0.5."
        )

    target_before_flip = (
        (
            target_final
            - flip_probability
        )
        / (
            1.0
            - 2.0
            * flip_probability
        )
    )

    target_before_flip = float(
        np.clip(
            target_before_flip,
            0.0001,
            0.9999,
        )
    )

    probabilities, intercept = (
        _calibrate_probability_mean(
            latent_without_intercept=(
                latent
            ),
            target_mean=(
                target_before_flip
            ),
            minimum_probability=float(
                fraud_cfg[
                    "probability_clip"
                ]["min"]
            ),
            maximum_probability=float(
                fraud_cfg[
                    "probability_clip"
                ]["max"]
            ),
        )
    )

    latent_calibrated = (
        latent
        + intercept
    )

    target = (
        rng.random(
            n
        )
        < probabilities
    ).astype(int)

    # ---------------------------------------------------------
    # Label noise
    # ---------------------------------------------------------

    flip_mask = (
        rng.random(
            n
        )
        < flip_probability
    )

    target[
        flip_mask
    ] = (
        1
        - target[
            flip_mask
        ]
    )

    # ---------------------------------------------------------
    # Mechanism metadata
    # ---------------------------------------------------------

    fraud_mechanism = (
        _sample_fraud_mechanisms(
            df=df,
            target=target,
            signals=signals,
            config=config,
            rng=rng,
        )
    )

    df[
        "latent_fraud_score"
    ] = latent_calibrated

    df[
        "synthetic_fraud_probability"
    ] = probabilities

    df[
        "fraud_difficulty"
    ] = np.where(
        target == 1,
        difficulty,
        "none",
    )

    df[
        "fraud_mechanism"
    ] = fraud_mechanism

    df[
        "is_fraud"
    ] = target

    return df


# =============================================================================
# Missingness
# =============================================================================


def inject_missingness(
    claims: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = claims.copy()

    cfg = config[
        "missingness"
    ]

    if not cfg.get(
        "enabled",
        False,
    ):
        return df

    provider_mask = (
        rng.random(
            len(df)
        )
        < float(
            cfg[
                "provider_id_probability"
            ]
        )
    )

    df.loc[
        provider_mask,
        "provider_id",
    ] = pd.NA

    prescription_mask = (
        rng.random(
            len(df)
        )
        < float(
            cfg[
                "prescription_missing_probability"
            ]
        )
    )

    df.loc[
        prescription_mask,
        "has_prescription",
    ] = pd.NA

    document_mask = (
        rng.random(
            len(df)
        )
        < float(
            cfg[
                "document_count_missing_probability"
            ]
        )
    )

    df.loc[
        document_mask,
        "document_count",
    ] = np.nan

    return df


# =============================================================================
# Data quality perturbations
# =============================================================================


def inject_quality_issues(
    claims: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Inject a small controlled number of invalid records.

    These records intentionally exist so validation.py
    has real data-quality problems to detect.
    """

    df = claims.copy()

    cfg = config[
        "quality"
    ]

    if not cfg.get(
        "inject_invalid_records",
        False,
    ):
        return df

    invalid_mask = (
        rng.random(
            len(df)
        )
        < float(
            cfg[
                "invalid_record_probability"
            ]
        )
    )

    invalid_indices = (
        np.flatnonzero(
            invalid_mask
        )
    )

    for idx in invalid_indices:
        issue_type = rng.choice(
            [
                "negative_amount",
                "service_after_submission",
                "zero_units",
            ]
        )

        if (
            issue_type
            == "negative_amount"
        ):
            df.loc[
                idx,
                "claim_amount",
            ] *= -1

        elif (
            issue_type
            == "service_after_submission"
        ):
            df.loc[
                idx,
                "service_date",
            ] = (
                df.loc[
                    idx,
                    "claim_submission_date",
                ]
                + pd.Timedelta(
                    days=2
                )
            )

        elif (
            issue_type
            == "zero_units"
        ):
            df.loc[
                idx,
                "service_units",
            ] = 0

    return df


# =============================================================================
# Complete pipeline
# =============================================================================


def generate_synthetic_data(
    config_path: str | Path = (
        "configs/data.yaml"
    ),
) -> SyntheticDataBundle:
    config = load_config(
        config_path
    )

    seed = int(
        config[
            "project"
        ][
            "random_seed"
        ]
    )

    rng = np.random.default_rng(
        seed
    )

    customers = (
        generate_customers(
            config=config,
            rng=rng,
        )
    )

    providers = (
        generate_providers(
            config=config,
            rng=rng,
        )
    )

    policies = (
        generate_policies(
            customers=customers,
            config=config,
            rng=rng,
        )
    )

    claims = generate_claims(
        customers=customers,
        providers=providers,
        policies=policies,
        config=config,
        rng=rng,
    )

    claims = (
        add_historical_features(
            claims=claims,
            config=config,
        )
    )

    claims = (
        add_legitimate_anomalies(
            claims=claims,
            config=config,
            rng=rng,
        )
    )

    claims = (
        add_fraud_target(
            claims=claims,
            config=config,
            rng=rng,
        )
    )

    claims = inject_missingness(
        claims=claims,
        config=config,
        rng=rng,
    )

    claims = (
        inject_quality_issues(
            claims=claims,
            config=config,
            rng=rng,
        )
    )

    return SyntheticDataBundle(
        customers=customers,
        providers=providers,
        policies=policies,
        claims=claims,
    )