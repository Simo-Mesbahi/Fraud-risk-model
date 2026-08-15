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
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

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
    values = np.asarray(list(probabilities.values()), dtype=float)

    if np.any(values < 0):
        raise ValueError(f"{name}: probabilities cannot be negative.")

    total = values.sum()

    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"{name}: probabilities must sum to 1.0, got {total:.8f}"
        )


def _sample_categories(
    rng: np.random.Generator,
    probabilities: dict[str, float],
    size: int,
    name: str,
) -> np.ndarray:
    _validate_probabilities(probabilities, name)

    categories = np.asarray(list(probabilities.keys()))
    probs = np.asarray(list(probabilities.values()), dtype=float)

    return rng.choice(
        categories,
        size=size,
        p=probs,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


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


# =============================================================================
# Customers
# =============================================================================

def generate_customers(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_customers = int(config["simulation"]["n_customers"])
    cfg = config["entities"]["customers"]

    age_cfg = cfg["age"]

    if age_cfg.get("distribution") == "truncated_normal":
        ages = _truncated_normal(
            rng=rng,
            mean=age_cfg["mean"],
            std=age_cfg["std"],
            minimum=age_cfg["min"],
            maximum=age_cfg["max"],
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
        rng,
        cfg["coverage_levels"],
        n_customers,
        "customer coverage levels",
    )

    behavior_segment = _sample_categories(
        rng,
        cfg["behavior_segments"],
        n_customers,
        "customer behavior segments",
    )

    frequency_multiplier_map = cfg["claim_frequency_multiplier"]

    frequency_multiplier = np.array(
        [
            frequency_multiplier_map[segment]
            for segment in behavior_segment
        ],
        dtype=float,
    )

    customers = pd.DataFrame(
        {
            "customer_id": [
                f"CUST_{i:06d}"
                for i in range(1, n_customers + 1)
            ],
            "customer_age": ages,
            "customer_tenure_months": tenure_months,
            "coverage_level": coverage_level,
            "customer_behavior_segment": behavior_segment,
            "claim_frequency_multiplier": frequency_multiplier,
        }
    )

    return customers


# =============================================================================
# Providers
# =============================================================================

def generate_providers(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_providers = int(config["simulation"]["n_providers"])
    cfg = config["entities"]["providers"]

    provider_type = _sample_categories(
        rng,
        cfg["provider_types"],
        n_providers,
        "provider types",
    )

    provider_region = _sample_categories(
        rng,
        cfg["regions"],
        n_providers,
        "provider regions",
    )

    behavior_segment = _sample_categories(
        rng,
        cfg["behavior_segments"],
        n_providers,
        "provider behavior segments",
    )

    volume_multiplier_map = cfg["claim_volume_multiplier"]

    volume_multiplier = np.array(
        [
            volume_multiplier_map[segment]
            for segment in behavior_segment
        ],
        dtype=float,
    )

    provider_tenure_months = rng.integers(
        1,
        241,
        size=n_providers,
    )

    providers = pd.DataFrame(
        {
            "provider_id": [
                f"PROV_{i:05d}"
                for i in range(1, n_providers + 1)
            ],
            "provider_type": provider_type,
            "provider_region": provider_region,
            "provider_tenure_months": provider_tenure_months,
            "provider_behavior_segment": behavior_segment,
            "provider_volume_multiplier": volume_multiplier,
        }
    )

    return providers


# =============================================================================
# Policies
# =============================================================================

def generate_policies(
    customers: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(customers)

    simulation_start = pd.Timestamp(
        config["simulation"]["start_date"]
    )
    simulation_end = pd.Timestamp(
        config["simulation"]["end_date"]
    )

    max_policy_tenure = np.minimum(
        customers["customer_tenure_months"].to_numpy(),
        120,
    )

    policy_tenure_months = np.array(
        [
            rng.integers(1, int(max_tenure) + 1)
            for max_tenure in max_policy_tenure
        ]
    )

    latest_possible_start = (
        simulation_end
        - pd.to_timedelta(policy_tenure_months * 30, unit="D")
    )

    policy_start_date = pd.to_datetime(
        np.maximum(
            latest_possible_start.values.astype("datetime64[ns]"),
            simulation_start.to_datetime64(),
        )
    )

    policy_cfg = config["policy"]

    recent_change_probability = float(
        policy_cfg["recent_change_probability"]
    )

    recent_policy_change = (
        rng.random(n) < recent_change_probability
    )

    change_window = int(
        policy_cfg["recent_change_window_days"]
    )

    days_since_policy_change = np.where(
        recent_policy_change,
        rng.integers(
            1,
            change_window + 1,
            size=n,
        ),
        np.nan,
    )

    policies = pd.DataFrame(
        {
            "policy_id": [
                f"POL_{i:06d}"
                for i in range(1, n + 1)
            ],
            "customer_id": customers["customer_id"].to_numpy(),
            "coverage_level": customers["coverage_level"].to_numpy(),
            "policy_start_date": policy_start_date,
            "policy_end_date": pd.NaT,
            "policy_tenure_months": policy_tenure_months,
            "recent_policy_change": recent_policy_change,
            "days_since_policy_change": days_since_policy_change,
        }
    )

    return policies


# =============================================================================
# Claim date generation
# =============================================================================

def _build_daily_sampling_weights(
    config: dict[str, Any],
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    start = pd.Timestamp(config["simulation"]["start_date"])
    end = pd.Timestamp(config["simulation"]["end_date"])

    dates = pd.date_range(
        start=start,
        end=end,
        freq="D",
    )

    weights = np.ones(len(dates), dtype=float)

    seasonality = config["simulation"].get("seasonality", {})

    if seasonality.get("enabled", False):
        monthly = seasonality["monthly_factors"]

        weights = np.array(
            [
                float(monthly[f"{date.month:02d}"])
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
    dates, probabilities = _build_daily_sampling_weights(config)

    sampled = rng.choice(
        dates.to_numpy(),
        size=size,
        p=probabilities,
    )

    timestamps = pd.to_datetime(sampled)

    # Add time-of-day variability.
    seconds = rng.integers(
        0,
        24 * 60 * 60,
        size=size,
    )

    timestamps = timestamps + pd.to_timedelta(
        seconds,
        unit="s",
    )

    return pd.Series(timestamps)


# =============================================================================
# Service generation
# =============================================================================

def _sample_service_categories(
    config: dict[str, Any],
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    probabilities = {
        service: cfg["probability"]
        for service, cfg in config["services"].items()
    }

    return _sample_categories(
        rng,
        probabilities,
        size,
        "service probabilities",
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

    global_noise_std = float(
        config["claims"].get(
            "claim_amount_noise_std",
            0.0,
        )
    )

    for service_name, service_cfg in config["services"].items():
        mask = service_categories == service_name
        count = int(mask.sum())

        if count == 0:
            continue

        amount_cfg = service_cfg["amount"]

        if amount_cfg["distribution"] != "lognormal":
            raise ValueError(
                f"Unsupported distribution for {service_name}: "
                f"{amount_cfg['distribution']}"
            )

        median = float(amount_cfg["median"])
        sigma = float(amount_cfg["sigma"])

        raw = rng.lognormal(
            mean=np.log(median),
            sigma=sigma,
            size=count,
        )

        if global_noise_std > 0:
            noise = rng.normal(
                loc=1.0,
                scale=global_noise_std,
                size=count,
            )

            raw *= np.clip(
                noise,
                0.5,
                1.5,
            )

        raw = np.clip(
            raw,
            amount_cfg["min"],
            amount_cfg["max"],
        )

        amounts[mask] = raw

    return np.round(amounts, 2)


def _generate_service_units(
    service_categories: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    units = np.zeros(
        len(service_categories),
        dtype=int,
    )

    for service_name, service_cfg in config["services"].items():
        mask = service_categories == service_name

        units[mask] = rng.integers(
            service_cfg["units"]["min"],
            service_cfg["units"]["max"] + 1,
            size=int(mask.sum()),
        )

    return units


def _generate_service_codes(
    service_categories: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    codes_by_category = {
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

    result: list[str] = []

    for category in service_categories:
        result.append(
            rng.choice(codes_by_category[category])
        )

    return np.asarray(result)


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
    ].to_numpy(dtype=float)

    weights /= weights.sum()

    return rng.choice(
        np.arange(len(customers)),
        size=size,
        p=weights,
    )


def _sample_provider_indices(
    providers: pd.DataFrame,
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    weights = providers[
        "provider_volume_multiplier"
    ].to_numpy(dtype=float)

    weights /= weights.sum()

    return rng.choice(
        np.arange(len(providers)),
        size=size,
        p=weights,
    )


# =============================================================================
# Core claims
# =============================================================================

def generate_claims(
    customers: pd.DataFrame,
    providers: pd.DataFrame,
    policies: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_claims = int(config["simulation"]["target_claims"])

    customer_idx = _sample_customer_indices(
        customers,
        rng,
        n_claims,
    )

    provider_idx = _sample_provider_indices(
        providers,
        rng,
        n_claims,
    )

    selected_customers = (
        customers
        .iloc[customer_idx]
        .reset_index(drop=True)
    )

    selected_policies = (
        policies
        .iloc[customer_idx]
        .reset_index(drop=True)
    )

    selected_providers = (
        providers
        .iloc[provider_idx]
        .reset_index(drop=True)
    )

    service_category = _sample_service_categories(
        config,
        rng,
        n_claims,
    )

    service_code = _generate_service_codes(
        service_category,
        rng,
    )

    claim_amount = _generate_claim_amounts(
        service_category,
        config,
        rng,
    )

    service_units = _generate_service_units(
        service_category,
        config,
        rng,
    )

    submission_timestamp = _generate_submission_dates(
        config,
        rng,
        n_claims,
    )

    delay_cfg = config["claims"]["service_to_submission_days"]

    delays = rng.gamma(
        shape=delay_cfg["shape"],
        scale=delay_cfg["scale"],
        size=n_claims,
    )

    delays = np.minimum(
        np.floor(delays).astype(int),
        delay_cfg["max_days"],
    )

    service_date = (
        submission_timestamp.dt.normalize()
        - pd.to_timedelta(delays, unit="D")
    )

    channels = _sample_categories(
        rng,
        config["claims"]["submission_channels"],
        n_claims,
        "submission channels",
    )

    coverage_limit_cfg = config["policy"]["coverage_limits"]

    coverage_limit = np.array(
        [
            coverage_limit_cfg[coverage][service]
            for coverage, service in zip(
                selected_policies["coverage_level"],
                service_category,
            )
        ],
        dtype=float,
    )

    base_reimbursement_rate = rng.beta(
        a=8,
        b=2,
        size=n_claims,
    )

    requested_reimbursement = np.minimum(
        claim_amount * base_reimbursement_rate,
        coverage_limit,
    )

    document_cfg = config["claims"]["document_count"]

    document_count = rng.integers(
        document_cfg["min"],
        document_cfg["max"] + 1,
        size=n_claims,
    ).astype(float)

    has_invoice = (
        rng.random(n_claims)
        < config["claims"]["has_invoice_probability"]
    )

    prescription_probs = np.array(
        [
            config["claims"][
                "has_prescription_probability"
            ][service]
            for service in service_category
        ],
        dtype=float,
    )

    has_prescription = (
        rng.random(n_claims)
        < prescription_probs
    ).astype(object)

    claims = pd.DataFrame(
        {
            "claim_id": [
                f"CLM_{i:08d}"
                for i in range(1, n_claims + 1)
            ],
            "customer_id": selected_customers["customer_id"].to_numpy(),
            "policy_id": selected_policies["policy_id"].to_numpy(),
            "provider_id": selected_providers["provider_id"].to_numpy(),
            "service_category": service_category,
            "service_code": service_code,
            "service_units": service_units,
            "service_date": service_date,
            "claim_submission_date": submission_timestamp.dt.normalize(),
            "claim_submission_timestamp": submission_timestamp,
            "claim_amount": claim_amount,
            "requested_reimbursement": np.round(
                requested_reimbursement,
                2,
            ),
            "coverage_limit": coverage_limit,
            "submission_channel": channels,
            "document_count": document_count,
            "has_invoice": has_invoice,
            "has_prescription": has_prescription,
            "customer_age": selected_customers[
                "customer_age"
            ].to_numpy(),
            "customer_tenure_months": selected_customers[
                "customer_tenure_months"
            ].to_numpy(),
            "coverage_level": selected_customers[
                "coverage_level"
            ].to_numpy(),
            "customer_behavior_segment": selected_customers[
                "customer_behavior_segment"
            ].to_numpy(),
            "policy_tenure_months": selected_policies[
                "policy_tenure_months"
            ].to_numpy(),
            "recent_policy_change": selected_policies[
                "recent_policy_change"
            ].to_numpy(),
            "days_since_policy_change": selected_policies[
                "days_since_policy_change"
            ].to_numpy(),
            "provider_type": selected_providers[
                "provider_type"
            ].to_numpy(),
            "provider_region": selected_providers[
                "provider_region"
            ].to_numpy(),
            "provider_tenure_months": selected_providers[
                "provider_tenure_months"
            ].to_numpy(),
            "provider_behavior_segment": selected_providers[
                "provider_behavior_segment"
            ].to_numpy(),
        }
    )

    claims["days_service_to_submission"] = (
        claims["claim_submission_date"]
        - claims["service_date"]
    ).dt.days

    claims["reimbursement_ratio"] = (
        claims["requested_reimbursement"]
        / claims["claim_amount"]
    ).clip(0, 1)

    claims = (
        claims
        .sort_values("claim_submission_timestamp")
        .reset_index(drop=True)
    )

    return claims


# =============================================================================
# Historical feature generation
# =============================================================================

def add_historical_features(
    claims: pd.DataFrame,
) -> pd.DataFrame:
    df = claims.copy()

    df = df.sort_values(
        "claim_submission_timestamp"
    ).reset_index(drop=True)

    timestamp = "claim_submission_timestamp"

    # -------------------------------------------------------------------------
    # Customer history
    # -------------------------------------------------------------------------

    for window in [7, 30, 90, 365]:
        count_col = f"customer_claims_{window}d"

        df[count_col] = (
            df.groupby("customer_id", group_keys=False)
            .apply(
                lambda group: (
                    group.set_index(timestamp)["claim_id"]
                    .rolling(
                        f"{window}D",
                        closed="left",
                    )
                    .count()
                    .reset_index(drop=True)
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )

    df["customer_amount_30d"] = (
        df.groupby("customer_id", group_keys=False)
        .apply(
            lambda group: (
                group.set_index(timestamp)["claim_amount"]
                .rolling("30D", closed="left")
                .sum()
                .reset_index(drop=True)
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    df["customer_amount_365d"] = (
        df.groupby("customer_id", group_keys=False)
        .apply(
            lambda group: (
                group.set_index(timestamp)["claim_amount"]
                .rolling("365D", closed="left")
                .sum()
                .reset_index(drop=True)
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    df["customer_avg_claim_amount_365d"] = (
        df.groupby("customer_id", group_keys=False)
        .apply(
            lambda group: (
                group.set_index(timestamp)["claim_amount"]
                .rolling("365D", closed="left")
                .mean()
                .reset_index(drop=True)
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Previous-claim timings
    # -------------------------------------------------------------------------

    df["days_since_customer_previous_claim"] = (
        df.groupby("customer_id")[timestamp]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    df["days_since_same_provider_claim"] = (
        df.groupby(
            ["customer_id", "provider_id"]
        )[timestamp]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    # -------------------------------------------------------------------------
    # Pair interaction counts
    # -------------------------------------------------------------------------

    df["customer_provider_claims_30d"] = (
        df.groupby(
            ["customer_id", "provider_id"],
            group_keys=False,
        )
        .apply(
            lambda group: (
                group.set_index(timestamp)["claim_id"]
                .rolling("30D", closed="left")
                .count()
                .reset_index(drop=True)
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Provider history
    # -------------------------------------------------------------------------

    df["provider_claims_30d"] = (
        df.groupby("provider_id", group_keys=False)
        .apply(
            lambda group: (
                group.set_index(timestamp)["claim_id"]
                .rolling("30D", closed="left")
                .count()
                .reset_index(drop=True)
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    df["provider_claims_90d"] = (
        df.groupby("provider_id", group_keys=False)
        .apply(
            lambda group: (
                group.set_index(timestamp)["claim_id"]
                .rolling("90D", closed="left")
                .count()
                .reset_index(drop=True)
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    df["provider_avg_claim_amount_90d"] = (
        df.groupby("provider_id", group_keys=False)
        .apply(
            lambda group: (
                group.set_index(timestamp)["claim_amount"]
                .rolling("90D", closed="left")
                .mean()
                .reset_index(drop=True)
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Same-service history
    # -------------------------------------------------------------------------

    df["same_service_claims_30d"] = (
        df.groupby(
            ["customer_id", "service_category"],
            group_keys=False,
        )
        .apply(
            lambda group: (
                group.set_index(timestamp)["claim_id"]
                .rolling("30D", closed="left")
                .count()
                .reset_index(drop=True)
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Service baselines
    # -------------------------------------------------------------------------

    expanding_median = (
        df.groupby("service_code")["claim_amount"]
        .transform(
            lambda x: x.shift(1).expanding().median()
        )
    )

    service_fallback = (
        df.groupby("service_code")["claim_amount"]
        .transform("median")
    )

    df["service_typical_amount"] = (
        expanding_median
        .fillna(service_fallback)
    )

    df["claim_to_service_median_ratio"] = (
        df["claim_amount"]
        / df["service_typical_amount"]
    )

    df["claim_to_customer_avg_ratio"] = (
        df["claim_amount"]
        / df["customer_avg_claim_amount_365d"]
    )

    df["claim_to_provider_avg_ratio"] = (
        df["claim_amount"]
        / df["provider_avg_claim_amount_90d"]
    )

    # Prevent infinities.
    ratio_cols = [
        "claim_to_service_median_ratio",
        "claim_to_customer_avg_ratio",
        "claim_to_provider_avg_ratio",
    ]

    df[ratio_cols] = (
        df[ratio_cols]
        .replace([np.inf, -np.inf], np.nan)
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
    df = claims.copy()

    cfg = config["legitimate_anomalies"]

    if not cfg.get("enabled", False):
        df["legitimate_anomaly"] = False
        df["legitimate_anomaly_type"] = "none"
        return df

    prevalence = float(cfg["prevalence"])

    anomaly_mask = (
        rng.random(len(df))
        < prevalence
    )

    df["legitimate_anomaly"] = anomaly_mask
    df["legitimate_anomaly_type"] = "none"

    pattern_probs = cfg["patterns"]

    anomaly_indices = np.flatnonzero(anomaly_mask)

    if len(anomaly_indices) == 0:
        return df

    anomaly_types = _sample_categories(
        rng,
        pattern_probs,
        len(anomaly_indices),
        "legitimate anomaly patterns",
    )

    df.loc[
        anomaly_indices,
        "legitimate_anomaly_type",
    ] = anomaly_types

    for idx, anomaly_type in zip(
        anomaly_indices,
        anomaly_types,
    ):
        if anomaly_type == "high_amount_legitimate":
            df.loc[idx, "claim_amount"] *= rng.uniform(1.8, 3.5)

        elif anomaly_type == "high_frequency_legitimate":
            df.loc[idx, "customer_claims_30d"] += rng.integers(3, 8)

        elif anomaly_type == "unusual_provider_legitimate":
            df.loc[idx, "provider_claims_30d"] += rng.integers(10, 30)

        elif anomaly_type == "repeated_service_legitimate":
            df.loc[idx, "same_service_claims_30d"] += rng.integers(2, 6)

    return df


# =============================================================================
# Fraud generation
# =============================================================================

def add_fraud_target(
    claims: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = claims.copy()

    cfg = config["fraud"]
    thresholds = cfg["thresholds"]

    n = len(df)

    # -------------------------------------------------------------------------
    # Individual fraud signals
    # -------------------------------------------------------------------------

    frequency_signal = np.maximum(
        (
            df["customer_claims_30d"].fillna(0).to_numpy()
            - thresholds["high_recent_claims_30d"]
        )
        / 3.0,
        0,
    )

    amount_signal = np.maximum(
        (
            df["claim_to_service_median_ratio"].fillna(1).to_numpy()
            - thresholds["high_claim_to_service_ratio"]
        ),
        0,
    )

    repeated_service_signal = np.maximum(
        (
            df["same_service_claims_30d"].fillna(0).to_numpy()
            - thresholds["high_same_service_claims_30d"]
        )
        / 2.0,
        0,
    )

    provider_recent = (
        df["provider_claims_30d"]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    provider_long = (
        df["provider_claims_90d"]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    expected_30d = provider_long / 3.0

    provider_growth = (
        provider_recent + 1
    ) / (
        expected_30d + 1
    )

    provider_signal = np.maximum(
        provider_growth
        - thresholds["high_provider_growth_ratio"],
        0,
    )

    pair_signal = np.maximum(
        (
            df["customer_provider_claims_30d"]
            .fillna(0)
            .to_numpy()
            - thresholds["high_customer_provider_claims_30d"]
        ),
        0,
    )

    # -------------------------------------------------------------------------
    # Fraud difficulty
    # -------------------------------------------------------------------------

    difficulty = _sample_categories(
        rng,
        cfg["difficulty_mix"],
        n,
        "fraud difficulty mix",
    )

    mechanisms = cfg["mechanisms"]

    def strength(
        mechanism: str,
    ) -> np.ndarray:
        levels = mechanisms[mechanism]["signal_strength"]

        return np.array(
            [
                levels[level]
                for level in difficulty
            ],
            dtype=float,
        )

    latent = np.full(
        n,
        np.log(
            cfg["base_rate"]
            / (1 - cfg["base_rate"])
        ),
        dtype=float,
    )

    latent += strength("frequency_abuse") * frequency_signal
    latent += strength("amount_inflation") * amount_signal
    latent += strength("repeated_service") * repeated_service_signal
    latent += strength("provider_abnormality") * provider_signal
    latent += strength("customer_provider_pattern") * pair_signal

    # -------------------------------------------------------------------------
    # Interactions
    # -------------------------------------------------------------------------

    if cfg["interactions"].get("enabled", False):
        interactions = cfg["interactions"]["definitions"]

        latent += (
            interactions["amount_and_frequency"]["weight"]
            * amount_signal
            * frequency_signal
        )

        latent += (
            interactions["provider_and_customer"]["weight"]
            * provider_signal
            * pair_signal
        )

        latent += (
            interactions["repeated_service_and_amount"]["weight"]
            * repeated_service_signal
            * amount_signal
        )

        policy_signal = (
            df["recent_policy_change"]
            .fillna(False)
            .astype(float)
            .to_numpy()
        )

        latent += (
            interactions["policy_change_and_high_amount"]["weight"]
            * policy_signal
            * amount_signal
        )

    # -------------------------------------------------------------------------
    # Concept drift
    # -------------------------------------------------------------------------

    drift_cfg = config["concept_drift"]

    if drift_cfg.get("enabled", False):
        drift_start = pd.Timestamp(
            drift_cfg["start_date"]
        )

        drift_mask = (
            df["claim_submission_timestamp"]
            >= drift_start
        ).to_numpy()

        prevalence_multiplier = float(
            drift_cfg["prevalence_multiplier"]
        )

        latent[drift_mask] += np.log(
            prevalence_multiplier
        )

    # -------------------------------------------------------------------------
    # Noise
    # -------------------------------------------------------------------------

    latent += rng.normal(
        loc=0.0,
        scale=cfg["noise"]["latent_score_std"],
        size=n,
    )

    probabilities = _sigmoid(latent)

    probabilities = np.clip(
        probabilities,
        cfg["probability_clip"]["min"],
        cfg["probability_clip"]["max"],
    )

    # -------------------------------------------------------------------------
    # Approximate prevalence calibration
    # -------------------------------------------------------------------------

    target_prevalence = float(
        cfg["target_prevalence"]
    )

    for _ in range(30):
        current_mean = probabilities.mean()

        if abs(
            current_mean - target_prevalence
        ) < 1e-4:
            break

        adjustment = np.log(
            target_prevalence
            / max(current_mean, 1e-8)
        )

        latent += adjustment

        probabilities = np.clip(
            _sigmoid(latent),
            cfg["probability_clip"]["min"],
            cfg["probability_clip"]["max"],
        )

    target = (
        rng.random(n)
        < probabilities
    ).astype(int)

    # -------------------------------------------------------------------------
    # Legitimate hard negatives
    # -------------------------------------------------------------------------

    legitimate_anomaly_mask = (
        df["legitimate_anomaly"]
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    # Most intentionally generated legitimate anomalies remain legitimate.
    keep_legitimate = (
        legitimate_anomaly_mask
        & (rng.random(n) < 0.90)
    )

    target[keep_legitimate] = 0

    # -------------------------------------------------------------------------
    # Label noise
    # -------------------------------------------------------------------------

    label_flip_probability = float(
        cfg["noise"]["label_flip_probability"]
    )

    flip_mask = (
        rng.random(n)
        < label_flip_probability
    )

    target[flip_mask] = 1 - target[flip_mask]

    # -------------------------------------------------------------------------
    # Fraud mechanism metadata
    # -------------------------------------------------------------------------

    signal_matrix = np.column_stack(
        [
            frequency_signal,
            amount_signal,
            repeated_service_signal,
            provider_signal,
            pair_signal,
        ]
    )

    mechanism_names = np.array(
        [
            "frequency_abuse",
            "amount_inflation",
            "repeated_service",
            "provider_abnormality",
            "customer_provider_pattern",
        ]
    )

    dominant = mechanism_names[
        np.argmax(signal_matrix, axis=1)
    ]

    active_signal_count = (
        signal_matrix > 0
    ).sum(axis=1)

    fraud_mechanism = np.where(
        target == 0,
        "none",
        np.where(
            active_signal_count >= 2,
            "mixed_pattern",
            dominant,
        ),
    )

    df["latent_fraud_score"] = latent
    df["synthetic_fraud_probability"] = probabilities
    df["fraud_difficulty"] = np.where(
        target == 1,
        difficulty,
        "none",
    )
    df["fraud_mechanism"] = fraud_mechanism
    df["is_fraud"] = target

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

    cfg = config["missingness"]

    if not cfg.get("enabled", False):
        return df

    provider_mask = (
        rng.random(len(df))
        < cfg["provider_id_probability"]
    )

    df.loc[
        provider_mask,
        "provider_id",
    ] = pd.NA

    prescription_mask = (
        rng.random(len(df))
        < cfg["prescription_missing_probability"]
    )

    df.loc[
        prescription_mask,
        "has_prescription",
    ] = pd.NA

    document_mask = (
        rng.random(len(df))
        < cfg["document_count_missing_probability"]
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
    df = claims.copy()

    cfg = config["quality"]

    if not cfg.get(
        "inject_invalid_records",
        False,
    ):
        return df

    n = len(df)

    invalid_mask = (
        rng.random(n)
        < cfg["invalid_record_probability"]
    )

    invalid_indices = np.flatnonzero(
        invalid_mask
    )

    for idx in invalid_indices:
        issue_type = rng.choice(
            [
                "negative_amount",
                "service_after_submission",
                "zero_units",
            ]
        )

        if issue_type == "negative_amount":
            df.loc[idx, "claim_amount"] *= -1

        elif issue_type == "service_after_submission":
            df.loc[idx, "service_date"] = (
                df.loc[idx, "claim_submission_date"]
                + pd.Timedelta(days=2)
            )

        elif issue_type == "zero_units":
            df.loc[idx, "service_units"] = 0

    return df


# =============================================================================
# Complete generation pipeline
# =============================================================================

def generate_synthetic_data(
    config_path: str | Path = "configs/data.yaml",
) -> SyntheticDataBundle:
    config = load_config(config_path)

    seed = int(
        config["project"]["random_seed"]
    )

    rng = np.random.default_rng(seed)

    customers = generate_customers(
        config,
        rng,
    )

    providers = generate_providers(
        config,
        rng,
    )

    policies = generate_policies(
        customers,
        config,
        rng,
    )

    claims = generate_claims(
        customers,
        providers,
        policies,
        config,
        rng,
    )

    claims = add_historical_features(
        claims,
    )

    claims = add_legitimate_anomalies(
        claims,
        config,
        rng,
    )

    claims = add_fraud_target(
        claims,
        config,
        rng,
    )

    claims = inject_missingness(
        claims,
        config,
        rng,
    )

    claims = inject_quality_issues(
        claims,
        config,
        rng,
    )

    return SyntheticDataBundle(
        customers=customers,
        providers=providers,
        policies=policies,
        claims=claims,
    )