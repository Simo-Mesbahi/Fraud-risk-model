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


def _validate_probabilities(probabilities: dict[str, float], name: str) -> None:
    values = np.asarray(list(probabilities.values()), dtype=float)

    if np.any(values < 0):
        raise ValueError(f"{name}: probabilities cannot be negative.")

    total = float(values.sum())

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

    return rng.choice(categories, size=size, p=probs)


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    minimum: float,
    maximum: float,
    size: int,
) -> np.ndarray:
    values = rng.normal(loc=mean, scale=std, size=size)
    return np.clip(values, minimum, maximum)


def _sample_int_range(
    rng: np.random.Generator,
    specification: dict[str, Any],
) -> int:
    return int(
        rng.integers(
            int(specification["min"]),
            int(specification["max"]) + 1,
        )
    )


def _sample_float_range(
    rng: np.random.Generator,
    specification: dict[str, Any],
) -> float:
    return float(
        rng.uniform(
            float(specification["min"]),
            float(specification["max"]),
        )
    )


def _clip_timestamp(
    timestamp: pd.Timestamp,
    minimum: pd.Timestamp,
    maximum: pd.Timestamp,
) -> pd.Timestamp:
    return min(max(timestamp, minimum), maximum)


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
            mean=float(age_cfg["mean"]),
            std=float(age_cfg["std"]),
            minimum=float(age_cfg["min"]),
            maximum=float(age_cfg["max"]),
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

    multiplier_map = cfg["claim_frequency_multiplier"]
    claim_frequency_multiplier = np.asarray(
        [multiplier_map[segment] for segment in behavior_segment],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "customer_id": [
                f"CUST_{i:06d}" for i in range(1, n_customers + 1)
            ],
            "customer_age": ages,
            "customer_tenure_months": tenure_months,
            "coverage_level": coverage_level,
            "customer_behavior_segment": behavior_segment,
            "claim_frequency_multiplier": claim_frequency_multiplier,
        }
    )


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

    multiplier_map = cfg["claim_volume_multiplier"]
    provider_volume_multiplier = np.asarray(
        [multiplier_map[segment] for segment in behavior_segment],
        dtype=float,
    )

    provider_tenure_months = rng.integers(1, 241, size=n_providers)

    return pd.DataFrame(
        {
            "provider_id": [
                f"PROV_{i:05d}" for i in range(1, n_providers + 1)
            ],
            "provider_type": provider_type,
            "provider_region": provider_region,
            "provider_tenure_months": provider_tenure_months,
            "provider_behavior_segment": behavior_segment,
            "provider_volume_multiplier": provider_volume_multiplier,
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
    n = len(customers)
    simulation_start = pd.Timestamp(config["simulation"]["start_date"])

    max_prior_tenure = np.minimum(
        customers["customer_tenure_months"].to_numpy(),
        120,
    )

    prior_policy_tenure = np.asarray(
        [
            rng.integers(1, int(max_tenure) + 1)
            for max_tenure in max_prior_tenure
        ]
    )

    policy_start_date = simulation_start - pd.to_timedelta(
        prior_policy_tenure * 30,
        unit="D",
    )

    return pd.DataFrame(
        {
            "policy_id": [f"POL_{i:06d}" for i in range(1, n + 1)],
            "customer_id": customers["customer_id"].to_numpy(),
            "coverage_level": customers["coverage_level"].to_numpy(),
            "policy_start_date": policy_start_date,
            "policy_end_date": pd.NaT,
        }
    )


# =============================================================================
# Claim dates
# =============================================================================


def _build_daily_sampling_weights(
    config: dict[str, Any],
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    start = pd.Timestamp(config["simulation"]["start_date"])
    end = pd.Timestamp(config["simulation"]["end_date"])
    dates = pd.date_range(start=start, end=end, freq="D")
    weights = np.ones(len(dates), dtype=float)

    seasonality = config["simulation"].get("seasonality", {})

    if seasonality.get("enabled", False):
        monthly_factors = seasonality["monthly_factors"]
        weights = np.asarray(
            [float(monthly_factors[f"{date.month:02d}"]) for date in dates],
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
    sampled_dates = rng.choice(dates.to_numpy(), size=size, p=probabilities)
    timestamps = pd.to_datetime(sampled_dates)
    seconds = rng.integers(0, 24 * 60 * 60, size=size)
    timestamps = timestamps + pd.to_timedelta(seconds, unit="s")
    return pd.Series(timestamps)


# =============================================================================
# Services
# =============================================================================


SERVICE_CODES = {
    "consultation": ["CONS_GP", "CONS_SPEC", "CONS_FOLLOWUP"],
    "dental": ["DENT_CHECK", "DENT_CROWN", "DENT_PROSTHESIS", "DENT_SURGERY"],
    "optical": ["OPT_FRAME", "OPT_LENSES", "OPT_CONTACT"],
    "physiotherapy": ["PHYSIO_STANDARD", "PHYSIO_REHAB", "PHYSIO_SPECIAL"],
    "pharmacy": ["PHARM_RX", "PHARM_DEVICE", "PHARM_OTHER"],
    "medical_device": ["DEVICE_ORTHO", "DEVICE_HEARING", "DEVICE_OTHER"],
    "diagnostic": ["DIAG_LAB", "DIAG_IMAGING", "DIAG_OTHER"],
    "other": ["OTHER_01", "OTHER_02"],
}


SERVICE_PROVIDER_TYPES = {
    "consultation": {"general_practitioner", "specialist"},
    "dental": {"dentist"},
    "optical": {"optician"},
    "physiotherapy": {"physiotherapist"},
    "pharmacy": {"pharmacy"},
    "medical_device": {"medical_supplier", "pharmacy"},
    "diagnostic": {"laboratory", "specialist"},
    "other": {"other", "general_practitioner", "specialist"},
}


PROVIDER_TYPE_SERVICES: dict[str, list[str]] = {}
for _service, _provider_types in SERVICE_PROVIDER_TYPES.items():
    for _provider_type in _provider_types:
        PROVIDER_TYPE_SERVICES.setdefault(_provider_type, []).append(_service)


def _sample_service_categories(
    config: dict[str, Any],
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    probabilities = {
        service: service_cfg["probability"]
        for service, service_cfg in config["services"].items()
    }
    return _sample_categories(
        rng,
        probabilities,
        size,
        "service probabilities",
    )


def _generate_service_codes(
    service_categories: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.asarray(
        [rng.choice(SERVICE_CODES[category]) for category in service_categories]
    )


def _generate_claim_amounts(
    service_categories: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    amounts = np.zeros(len(service_categories), dtype=float)
    noise_std = float(config["claims"].get("claim_amount_noise_std", 0.0))

    for service_name, service_cfg in config["services"].items():
        mask = service_categories == service_name
        count = int(mask.sum())

        if count == 0:
            continue

        cfg = service_cfg["amount"]

        if cfg["distribution"] != "lognormal":
            raise ValueError(
                f"Unsupported distribution for {service_name}: {cfg['distribution']}"
            )

        raw = rng.lognormal(
            mean=np.log(float(cfg["median"])),
            sigma=float(cfg["sigma"]),
            size=count,
        )

        if noise_std > 0:
            multiplier = rng.normal(loc=1.0, scale=noise_std, size=count)
            raw *= np.clip(multiplier, 0.5, 1.5)

        raw = np.clip(raw, cfg["min"], cfg["max"])
        amounts[mask] = raw

    return np.round(amounts, 2)


def _generate_service_units(
    service_categories: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    units = np.zeros(len(service_categories), dtype=int)

    for service_name, service_cfg in config["services"].items():
        mask = service_categories == service_name
        count = int(mask.sum())

        if count == 0:
            continue

        units[mask] = rng.integers(
            service_cfg["units"]["min"],
            service_cfg["units"]["max"] + 1,
            size=count,
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
    weights = customers["claim_frequency_multiplier"].to_numpy(dtype=float)
    weights /= weights.sum()
    return rng.choice(np.arange(len(customers)), size=size, p=weights)


def _sample_provider_indices(
    providers: pd.DataFrame,
    service_categories: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty(len(service_categories), dtype=int)
    provider_types = providers["provider_type"].to_numpy()
    provider_weights = providers["provider_volume_multiplier"].to_numpy(dtype=float)

    for service in np.unique(service_categories):
        claim_mask = service_categories == service
        allowed_types = SERVICE_PROVIDER_TYPES[service]
        provider_mask = np.isin(provider_types, list(allowed_types))
        candidate_indices = np.flatnonzero(provider_mask)

        if len(candidate_indices) == 0:
            raise ValueError(f"No providers available for service {service}.")

        weights = provider_weights[candidate_indices]
        weights = weights / weights.sum()

        result[claim_mask] = rng.choice(
            candidate_indices,
            size=int(claim_mask.sum()),
            p=weights,
        )

    return result


# =============================================================================
# Baseline claims
# =============================================================================


def generate_claims(
    customers: pd.DataFrame,
    providers: pd.DataFrame,
    policies: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_claims = int(config["simulation"]["target_claims"])

    customer_idx = _sample_customer_indices(customers, rng, n_claims)
    selected_customers = customers.iloc[customer_idx].reset_index(drop=True)
    selected_policies = policies.iloc[customer_idx].reset_index(drop=True)

    service_category = _sample_service_categories(config, rng, n_claims)
    provider_idx = _sample_provider_indices(providers, service_category, rng)
    selected_providers = providers.iloc[provider_idx].reset_index(drop=True)

    service_code = _generate_service_codes(service_category, rng)
    claim_amount = _generate_claim_amounts(service_category, config, rng)
    service_units = _generate_service_units(service_category, config, rng)
    submission_timestamp = _generate_submission_dates(config, rng, n_claims)

    delay_cfg = config["claims"]["service_to_submission_days"]
    delays = rng.gamma(
        shape=float(delay_cfg["shape"]),
        scale=float(delay_cfg["scale"]),
        size=n_claims,
    )
    delays = np.minimum(
        np.floor(delays).astype(int),
        int(delay_cfg["max_days"]),
    )

    service_date = submission_timestamp.dt.normalize() - pd.to_timedelta(
        delays,
        unit="D",
    )

    channels = _sample_categories(
        rng,
        config["claims"]["submission_channels"],
        n_claims,
        "submission channels",
    )

    coverage_cfg = config["policy"]["coverage_limits"]
    coverage_limits = np.asarray(
        [
            coverage_cfg[coverage][service]
            for coverage, service in zip(
                selected_policies["coverage_level"],
                service_category,
            )
        ],
        dtype=float,
    )

    reimbursement_rate = rng.beta(a=8, b=2, size=n_claims)
    requested_reimbursement = np.minimum(
        claim_amount * reimbursement_rate,
        coverage_limits,
    )

    document_cfg = config["claims"]["document_count"]
    document_count = rng.integers(
        document_cfg["min"],
        document_cfg["max"] + 1,
        size=n_claims,
    ).astype(float)

    has_invoice = rng.random(n_claims) < float(
        config["claims"]["has_invoice_probability"]
    )

    prescription_probs = np.asarray(
        [
            config["claims"]["has_prescription_probability"][service]
            for service in service_category
        ],
        dtype=float,
    )
    has_prescription = (rng.random(n_claims) < prescription_probs).astype(object)

    policy_start_dates = pd.to_datetime(selected_policies["policy_start_date"])
    policy_tenure_months = (
        (submission_timestamp - policy_start_dates)
        .dt.days.div(30.4375)
        .clip(lower=1)
        .astype(int)
    )

    policy_cfg = config["policy"]
    recent_policy_change = rng.random(n_claims) < float(
        policy_cfg["recent_change_probability"]
    )
    change_window = int(policy_cfg["recent_change_window_days"])
    days_since_policy_change = np.where(
        recent_policy_change,
        rng.integers(1, change_window + 1, size=n_claims),
        np.nan,
    )

    claims = pd.DataFrame(
        {
            "claim_id": [f"CLM_{i:08d}" for i in range(1, n_claims + 1)],
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
            "requested_reimbursement": np.round(requested_reimbursement, 2),
            "coverage_limit": coverage_limits,
            "submission_channel": channels,
            "document_count": document_count,
            "has_invoice": has_invoice,
            "has_prescription": has_prescription,
            "customer_age": selected_customers["customer_age"].to_numpy(),
            "customer_tenure_months": selected_customers[
                "customer_tenure_months"
            ].to_numpy(),
            "coverage_level": selected_customers["coverage_level"].to_numpy(),
            "customer_behavior_segment": selected_customers[
                "customer_behavior_segment"
            ].to_numpy(),
            "policy_tenure_months": policy_tenure_months.to_numpy(),
            "recent_policy_change": recent_policy_change,
            "days_since_policy_change": days_since_policy_change,
            "provider_type": selected_providers["provider_type"].to_numpy(),
            "provider_region": selected_providers["provider_region"].to_numpy(),
            "provider_tenure_months": selected_providers[
                "provider_tenure_months"
            ].to_numpy(),
            "provider_behavior_segment": selected_providers[
                "provider_behavior_segment"
            ].to_numpy(),
        }
    )

    claims["days_service_to_submission"] = (
        claims["claim_submission_date"] - claims["service_date"]
    ).dt.days
    claims["reimbursement_ratio"] = (
        claims["requested_reimbursement"] / claims["claim_amount"]
    ).clip(0, 1)

    return claims.sort_values("claim_submission_timestamp").reset_index(drop=True)


# =============================================================================
# Behaviour injection helpers
# =============================================================================


def _customer_lookup(
    customers: pd.DataFrame,
    policies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    customer_lookup = customers.set_index("customer_id", drop=False)
    policy_lookup = policies.set_index("customer_id", drop=False)
    return customer_lookup, policy_lookup


def _provider_lookup(providers: pd.DataFrame) -> pd.DataFrame:
    return providers.set_index("provider_id", drop=False)


def _set_customer(
    df: pd.DataFrame,
    indices: np.ndarray,
    customer_id: str,
    customer_lookup: pd.DataFrame,
    policy_lookup: pd.DataFrame,
) -> None:
    customer = customer_lookup.loc[customer_id]
    policy = policy_lookup.loc[customer_id]

    df.loc[indices, "customer_id"] = customer_id
    df.loc[indices, "policy_id"] = policy["policy_id"]
    df.loc[indices, "customer_age"] = customer["customer_age"]
    df.loc[indices, "customer_tenure_months"] = customer[
        "customer_tenure_months"
    ]
    df.loc[indices, "coverage_level"] = customer["coverage_level"]
    df.loc[indices, "customer_behavior_segment"] = customer[
        "customer_behavior_segment"
    ]

    timestamps = pd.to_datetime(df.loc[indices, "claim_submission_timestamp"])
    tenure = (
        (timestamps - pd.Timestamp(policy["policy_start_date"]))
        .dt.days.div(30.4375)
        .clip(lower=1)
        .astype(int)
    )
    df.loc[indices, "policy_tenure_months"] = tenure.to_numpy()


def _set_provider(
    df: pd.DataFrame,
    indices: np.ndarray,
    provider_id: str,
    provider_lookup: pd.DataFrame,
) -> None:
    provider = provider_lookup.loc[provider_id]

    df.loc[indices, "provider_id"] = provider_id
    df.loc[indices, "provider_type"] = provider["provider_type"]
    df.loc[indices, "provider_region"] = provider["provider_region"]
    df.loc[indices, "provider_tenure_months"] = provider[
        "provider_tenure_months"
    ]
    df.loc[indices, "provider_behavior_segment"] = provider[
        "provider_behavior_segment"
    ]


def _sample_provider_for_service(
    providers: pd.DataFrame,
    service: str,
    rng: np.random.Generator,
) -> str:
    candidates = providers.loc[
        providers["provider_type"].isin(SERVICE_PROVIDER_TYPES[service])
    ].copy()
    weights = candidates["provider_volume_multiplier"].to_numpy(dtype=float)
    weights /= weights.sum()
    chosen = rng.choice(candidates.index.to_numpy(), p=weights)
    return str(providers.loc[chosen, "provider_id"])


def _sample_service_for_provider_type(
    provider_type: str,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> str:
    allowed = PROVIDER_TYPE_SERVICES[provider_type]
    raw_weights = np.asarray(
        [float(config["services"][service]["probability"]) for service in allowed]
    )
    raw_weights /= raw_weights.sum()
    return str(rng.choice(np.asarray(allowed), p=raw_weights))


def _set_service(
    df: pd.DataFrame,
    indices: np.ndarray,
    service: str,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    df.loc[indices, "service_category"] = service
    df.loc[indices, "service_code"] = [
        rng.choice(SERVICE_CODES[service]) for _ in range(len(indices))
    ]

    units_cfg = config["services"][service]["units"]
    df.loc[indices, "service_units"] = rng.integers(
        int(units_cfg["min"]),
        int(units_cfg["max"]) + 1,
        size=len(indices),
    )

    prescription_probability = float(
        config["claims"]["has_prescription_probability"][service]
    )
    df.loc[indices, "has_prescription"] = (
        rng.random(len(indices)) < prescription_probability
    ).astype(object)


def _recompute_coverage_and_reimbursement(
    df: pd.DataFrame,
    indices: np.ndarray,
    config: dict[str, Any],
    reimbursement_rates: np.ndarray | None = None,
) -> None:
    coverage_cfg = config["policy"]["coverage_limits"]

    limits = np.asarray(
        [
            float(coverage_cfg[coverage][service])
            for coverage, service in zip(
                df.loc[indices, "coverage_level"],
                df.loc[indices, "service_category"],
            )
        ],
        dtype=float,
    )

    df.loc[indices, "coverage_limit"] = limits

    if reimbursement_rates is None:
        reimbursement_rates = (
            df.loc[indices, "reimbursement_ratio"]
            .fillna(0.80)
            .clip(0.50, 1.00)
            .to_numpy(dtype=float)
        )

    requested = np.minimum(
        df.loc[indices, "claim_amount"].to_numpy(dtype=float)
        * reimbursement_rates,
        limits,
    )

    df.loc[indices, "requested_reimbursement"] = np.round(requested, 2)
    df.loc[indices, "reimbursement_ratio"] = (
        df.loc[indices, "requested_reimbursement"].to_numpy(dtype=float)
        / df.loc[indices, "claim_amount"].to_numpy(dtype=float)
    )


def _set_episode_timestamps(
    df: pd.DataFrame,
    indices: np.ndarray,
    anchor_timestamp: pd.Timestamp,
    window_days: int,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    simulation_start = pd.Timestamp(config["simulation"]["start_date"])
    simulation_end = pd.Timestamp(config["simulation"]["end_date"]) + pd.Timedelta(
        hours=23,
        minutes=59,
        seconds=59,
    )

    max_seconds = max(1, int(window_days * 24 * 60 * 60))
    offsets = np.sort(rng.integers(0, max_seconds + 1, size=len(indices)))

    latest_anchor = simulation_end - pd.Timedelta(seconds=int(offsets[-1]))
    anchor_timestamp = _clip_timestamp(
        anchor_timestamp,
        simulation_start,
        latest_anchor,
    )

    timestamps = pd.to_datetime(
        [anchor_timestamp + pd.Timedelta(seconds=int(offset)) for offset in offsets]
    )

    old_delay = (
        df.loc[indices, "days_service_to_submission"]
        .fillna(0)
        .clip(lower=0)
        .astype(int)
        .to_numpy()
    )

    df.loc[indices, "claim_submission_timestamp"] = timestamps.to_numpy()
    df.loc[indices, "claim_submission_date"] = timestamps.normalize().to_numpy()
    df.loc[indices, "service_date"] = (
        timestamps.normalize() - pd.to_timedelta(old_delay, unit="D")
    ).to_numpy()
    df.loc[indices, "days_service_to_submission"] = old_delay


def _inflate_amounts(
    df: pd.DataFrame,
    indices: np.ndarray,
    factor_spec: dict[str, Any],
    config: dict[str, Any],
    rng: np.random.Generator,
    reimbursement_pressure_spec: dict[str, Any] | None = None,
) -> None:
    factors = rng.uniform(
        float(factor_spec["min"]),
        float(factor_spec["max"]),
        size=len(indices),
    )

    inflated = df.loc[indices, "claim_amount"].to_numpy(dtype=float) * factors

    # Allow fraudulent claims to exceed the ordinary simulation maximum slightly,
    # while preventing extreme unrealistic values.
    caps = np.asarray(
        [
            float(config["services"][service]["amount"]["max"]) * 1.25
            for service in df.loc[indices, "service_category"]
        ],
        dtype=float,
    )

    df.loc[indices, "claim_amount"] = np.round(np.minimum(inflated, caps), 2)

    if reimbursement_pressure_spec is None:
        reimbursement_rates = None
    else:
        reimbursement_rates = rng.uniform(
            float(reimbursement_pressure_spec["min"]),
            float(reimbursement_pressure_spec["max"]),
            size=len(indices),
        )

    _recompute_coverage_and_reimbursement(
        df,
        indices,
        config,
        reimbursement_rates=reimbursement_rates,
    )


def _available_indices(
    used: np.ndarray,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = ~used
    if eligible_mask is not None:
        mask &= eligible_mask
    return np.flatnonzero(mask)


def _choose_episode_rows(
    used: np.ndarray,
    size: int,
    rng: np.random.Generator,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    candidates = _available_indices(used, eligible_mask)

    if len(candidates) < size:
        raise RuntimeError(
            f"Not enough unused claims to build episode of size {size}."
        )

    return np.asarray(rng.choice(candidates, size=size, replace=False), dtype=int)


# =============================================================================
# Fraud behaviour injection
# =============================================================================


def _fraud_target_before_label_noise(
    config: dict[str, Any],
    n_rows: int,
) -> int:
    target = float(config["fraud"]["target_prevalence"])
    flip_probability = float(config["fraud"]["label_flip_probability"])

    if flip_probability >= 0.5:
        raise ValueError("label_flip_probability must be < 0.5.")

    pre_flip_rate = (target - flip_probability) / (1.0 - 2.0 * flip_probability)
    pre_flip_rate = float(np.clip(pre_flip_rate, 0.0, 1.0))
    return int(round(pre_flip_rate * n_rows))


def _allocate_counts(
    total: int,
    probabilities: dict[str, float],
) -> dict[str, int]:
    _validate_probabilities(probabilities, "allocation probabilities")

    names = list(probabilities)
    expected = np.asarray([probabilities[name] * total for name in names])
    counts = np.floor(expected).astype(int)
    remainder = total - int(counts.sum())

    if remainder > 0:
        fractional = expected - counts
        order = np.argsort(-fractional)
        counts[order[:remainder]] += 1

    return {name: int(count) for name, count in zip(names, counts)}


def _difficulty_for_episode(
    config: dict[str, Any],
    rng: np.random.Generator,
) -> str:
    return str(
        _sample_categories(
            rng,
            config["fraud"]["difficulty_mix"],
            1,
            "fraud difficulty mix",
        )[0]
    )


def _effective_mechanism_mix(
    config: dict[str, Any],
    after_drift: bool,
) -> dict[str, float]:
    mix = {
        key: float(value)
        for key, value in config["fraud"]["mechanism_mix"].items()
    }

    if after_drift and config["concept_drift"].get("enabled", False):
        multipliers = config["concept_drift"]["mechanism_weight_multiplier"]
        for mechanism in mix:
            mix[mechanism] *= float(multipliers.get(mechanism, 1.0))

    total = sum(mix.values())
    return {key: value / total for key, value in mix.items()}


def inject_fraud_behaviors(
    claims: pd.DataFrame,
    customers: pd.DataFrame,
    providers: pd.DataFrame,
    policies: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = claims.copy()
    n = len(df)

    df["_behavior_is_fraud"] = False
    df["_fraud_mechanism"] = "none"
    df["_fraud_difficulty"] = "none"

    customer_lookup, policy_lookup = _customer_lookup(customers, policies)
    provider_lookup = _provider_lookup(providers)

    used = np.zeros(n, dtype=bool)

    fraud_cfg = config["fraud"]
    episode_cfg = fraud_cfg["episode_generation"]
    warmup_days = int(episode_cfg["warmup_days"])

    simulation_start = pd.Timestamp(config["simulation"]["start_date"])
    warmup_cutoff = simulation_start + pd.Timedelta(days=warmup_days)
    drift_start = pd.Timestamp(config["concept_drift"]["start_date"])

    eligible = (
        df["claim_submission_timestamp"] >= warmup_cutoff
    ).to_numpy()

    total_target = _fraud_target_before_label_noise(config, n)

    before_drift_available = int(
        ((df["claim_submission_timestamp"] < drift_start) & (eligible)).sum()
    )
    after_drift_available = int(
        ((df["claim_submission_timestamp"] >= drift_start) & (eligible)).sum()
    )
    total_eligible = before_drift_available + after_drift_available

    if total_eligible <= 0:
        raise RuntimeError("No claims available after fraud warmup period.")

    prevalence_multiplier = 1.0
    if config["concept_drift"].get("enabled", False):
        prevalence_multiplier = float(
            config["concept_drift"].get("prevalence_multiplier", 1.0)
        )

    before_weight = before_drift_available
    after_weight = after_drift_available * prevalence_multiplier
    after_target = int(
        round(total_target * after_weight / (before_weight + after_weight))
    )
    before_target = total_target - after_target

    period_specs = [
        (
            "before_drift",
            before_target,
            eligible
            & (df["claim_submission_timestamp"] < drift_start).to_numpy(),
            _effective_mechanism_mix(config, after_drift=False),
        ),
        (
            "after_drift",
            after_target,
            eligible
            & (df["claim_submission_timestamp"] >= drift_start).to_numpy(),
            _effective_mechanism_mix(config, after_drift=True),
        ),
    ]

    for period_name, period_target, period_mask, mechanism_mix in period_specs:
        mechanism_counts = _allocate_counts(period_target, mechanism_mix)

        for mechanism, mechanism_target in mechanism_counts.items():
            remaining = mechanism_target
            mechanism_cfg = fraud_cfg["mechanisms"][mechanism]

            if not mechanism_cfg.get("enabled", True):
                continue

            while remaining > 0:
                difficulty = _difficulty_for_episode(config, rng)

                if mechanism == "amount_inflation":
                    episode_size = 1
                else:
                    episode_size = _sample_int_range(
                        rng,
                        mechanism_cfg["episode_size"][difficulty],
                    )

                episode_size = min(episode_size, remaining)
                indices = _choose_episode_rows(
                    used,
                    episode_size,
                    rng,
                    eligible_mask=period_mask,
                )
                used[indices] = True

                anchor_index = int(indices[0])
                anchor_timestamp = pd.Timestamp(
                    df.loc[anchor_index, "claim_submission_timestamp"]
                )

                if mechanism == "amount_inflation":
                    _inflate_amounts(
                        df,
                        indices,
                        mechanism_cfg["inflation_factor"][difficulty],
                        config,
                        rng,
                        reimbursement_pressure_spec=mechanism_cfg[
                            "reimbursement_pressure"
                        ][difficulty],
                    )

                elif mechanism == "frequency_abuse":
                    window_days = int(mechanism_cfg["window_days"][difficulty])
                    _set_episode_timestamps(
                        df,
                        indices,
                        anchor_timestamp,
                        window_days,
                        config,
                        rng,
                    )

                    anchor_customer = str(df.loc[anchor_index, "customer_id"])
                    anchor_service = str(df.loc[anchor_index, "service_category"])
                    anchor_provider = str(df.loc[anchor_index, "provider_id"])

                    _set_customer(
                        df,
                        indices,
                        anchor_customer,
                        customer_lookup,
                        policy_lookup,
                    )

                    same_service_probability = float(
                        mechanism_cfg["same_service_probability"][difficulty]
                    )
                    same_provider_probability = float(
                        mechanism_cfg["same_provider_probability"][difficulty]
                    )

                    for idx in indices:
                        if rng.random() < same_service_probability:
                            service = anchor_service
                        else:
                            service = str(
                                _sample_service_categories(config, rng, 1)[0]
                            )

                        _set_service(df, np.asarray([idx]), service, config, rng)

                        anchor_provider_type = str(
                            provider_lookup.loc[anchor_provider, "provider_type"]
                        )
                        can_use_anchor = (
                            anchor_provider_type in SERVICE_PROVIDER_TYPES[service]
                        )

                        if can_use_anchor and rng.random() < same_provider_probability:
                            provider_id = anchor_provider
                        else:
                            provider_id = _sample_provider_for_service(
                                providers,
                                service,
                                rng,
                            )

                        _set_provider(
                            df,
                            np.asarray([idx]),
                            provider_id,
                            provider_lookup,
                        )

                    _recompute_coverage_and_reimbursement(df, indices, config)

                elif mechanism == "repeated_service":
                    window_days = int(mechanism_cfg["window_days"][difficulty])
                    _set_episode_timestamps(
                        df,
                        indices,
                        anchor_timestamp,
                        window_days,
                        config,
                        rng,
                    )

                    anchor_customer = str(df.loc[anchor_index, "customer_id"])
                    anchor_service = str(df.loc[anchor_index, "service_category"])
                    anchor_provider = str(df.loc[anchor_index, "provider_id"])

                    _set_customer(
                        df,
                        indices,
                        anchor_customer,
                        customer_lookup,
                        policy_lookup,
                    )
                    _set_service(df, indices, anchor_service, config, rng)

                    same_provider_probability = float(
                        mechanism_cfg["same_provider_probability"][difficulty]
                    )
                    anchor_provider_type = str(
                        provider_lookup.loc[anchor_provider, "provider_type"]
                    )
                    anchor_compatible = (
                        anchor_provider_type in SERVICE_PROVIDER_TYPES[anchor_service]
                    )

                    for idx in indices:
                        if anchor_compatible and rng.random() < same_provider_probability:
                            provider_id = anchor_provider
                        else:
                            provider_id = _sample_provider_for_service(
                                providers,
                                anchor_service,
                                rng,
                            )
                        _set_provider(
                            df,
                            np.asarray([idx]),
                            provider_id,
                            provider_lookup,
                        )

                    _recompute_coverage_and_reimbursement(df, indices, config)

                elif mechanism == "provider_abnormality":
                    window_days = int(mechanism_cfg["window_days"][difficulty])
                    _set_episode_timestamps(
                        df,
                        indices,
                        anchor_timestamp,
                        window_days,
                        config,
                        rng,
                    )

                    anchor_provider = str(df.loc[anchor_index, "provider_id"])
                    provider_type = str(
                        provider_lookup.loc[anchor_provider, "provider_type"]
                    )
                    _set_provider(df, indices, anchor_provider, provider_lookup)

                    allowed_services = PROVIDER_TYPE_SERVICES[provider_type]
                    for idx in indices:
                        service = _sample_service_for_provider_type(
                            provider_type,
                            config,
                            rng,
                        )
                        _set_service(df, np.asarray([idx]), service, config, rng)

                    # Existing rows generally already correspond to distinct customers.
                    # If duplicates are excessive, reassign some rows to random customers.
                    desired_distinct = max(
                        1,
                        int(
                            np.ceil(
                                len(indices)
                                * float(
                                    mechanism_cfg["distinct_customer_fraction"][
                                        difficulty
                                    ]
                                )
                            )
                        ),
                    )
                    current_customers = df.loc[indices, "customer_id"].nunique()
                    if current_customers < desired_distinct:
                        replacement_count = desired_distinct - current_customers
                        replacement_rows = indices[-replacement_count:]
                        replacement_customers = rng.choice(
                            customers["customer_id"].to_numpy(),
                            size=replacement_count,
                            replace=False,
                        )
                        for idx, customer_id in zip(
                            replacement_rows,
                            replacement_customers,
                        ):
                            _set_customer(
                                df,
                                np.asarray([idx]),
                                str(customer_id),
                                customer_lookup,
                                policy_lookup,
                            )

                    _inflate_amounts(
                        df,
                        indices,
                        mechanism_cfg["amount_multiplier"][difficulty],
                        config,
                        rng,
                    )

                elif mechanism == "customer_provider_pattern":
                    window_days = int(mechanism_cfg["window_days"][difficulty])
                    _set_episode_timestamps(
                        df,
                        indices,
                        anchor_timestamp,
                        window_days,
                        config,
                        rng,
                    )

                    anchor_customer = str(df.loc[anchor_index, "customer_id"])
                    anchor_provider = str(df.loc[anchor_index, "provider_id"])
                    provider_type = str(
                        provider_lookup.loc[anchor_provider, "provider_type"]
                    )
                    anchor_service = str(df.loc[anchor_index, "service_category"])

                    _set_customer(
                        df,
                        indices,
                        anchor_customer,
                        customer_lookup,
                        policy_lookup,
                    )
                    _set_provider(df, indices, anchor_provider, provider_lookup)

                    same_service_probability = float(
                        mechanism_cfg["same_service_probability"][difficulty]
                    )
                    for idx in indices:
                        if (
                            provider_type in SERVICE_PROVIDER_TYPES[anchor_service]
                            and rng.random() < same_service_probability
                        ):
                            service = anchor_service
                        else:
                            service = _sample_service_for_provider_type(
                                provider_type,
                                config,
                                rng,
                            )
                        _set_service(df, np.asarray([idx]), service, config, rng)

                    _recompute_coverage_and_reimbursement(df, indices, config)

                elif mechanism == "mixed_pattern":
                    window_days = int(mechanism_cfg["window_days"][difficulty])
                    _set_episode_timestamps(
                        df,
                        indices,
                        anchor_timestamp,
                        window_days,
                        config,
                        rng,
                    )

                    anchor_customer = str(df.loc[anchor_index, "customer_id"])
                    anchor_provider = str(df.loc[anchor_index, "provider_id"])
                    provider_type = str(
                        provider_lookup.loc[anchor_provider, "provider_type"]
                    )
                    anchor_service = str(df.loc[anchor_index, "service_category"])

                    _set_customer(
                        df,
                        indices,
                        anchor_customer,
                        customer_lookup,
                        policy_lookup,
                    )

                    same_service_probability = float(
                        mechanism_cfg["same_service_probability"][difficulty]
                    )
                    same_provider_probability = float(
                        mechanism_cfg["same_provider_probability"][difficulty]
                    )

                    for idx in indices:
                        if (
                            provider_type in SERVICE_PROVIDER_TYPES[anchor_service]
                            and rng.random() < same_service_probability
                        ):
                            service = anchor_service
                        else:
                            service = _sample_service_for_provider_type(
                                provider_type,
                                config,
                                rng,
                            )

                        _set_service(df, np.asarray([idx]), service, config, rng)

                        can_anchor = provider_type in SERVICE_PROVIDER_TYPES[service]
                        if can_anchor and rng.random() < same_provider_probability:
                            provider_id = anchor_provider
                        else:
                            provider_id = _sample_provider_for_service(
                                providers,
                                service,
                                rng,
                            )

                        _set_provider(
                            df,
                            np.asarray([idx]),
                            provider_id,
                            provider_lookup,
                        )

                    _inflate_amounts(
                        df,
                        indices,
                        mechanism_cfg["inflation_factor"][difficulty],
                        config,
                        rng,
                    )

                else:
                    raise ValueError(f"Unsupported fraud mechanism: {mechanism}")

                df.loc[indices, "_behavior_is_fraud"] = True
                df.loc[indices, "_fraud_mechanism"] = mechanism
                df.loc[indices, "_fraud_difficulty"] = difficulty

                remaining -= len(indices)

    actual = int(df["_behavior_is_fraud"].sum())
    if actual != total_target:
        raise RuntimeError(
            f"Fraud injection count mismatch: expected {total_target}, got {actual}."
        )

    return df


# =============================================================================
# Legitimate hard-negative behaviour injection
# =============================================================================


def inject_legitimate_anomaly_behaviors(
    claims: pd.DataFrame,
    customers: pd.DataFrame,
    providers: pd.DataFrame,
    policies: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = claims.copy()
    cfg = config["legitimate_anomalies"]

    df["legitimate_anomaly"] = False
    df["legitimate_anomaly_type"] = "none"

    if not cfg.get("enabled", False):
        return df

    customer_lookup, policy_lookup = _customer_lookup(customers, policies)
    provider_lookup = _provider_lookup(providers)

    eligible = ~df["_behavior_is_fraud"].to_numpy(dtype=bool)
    selected_used = np.zeros(len(df), dtype=bool)
    selected_used[~eligible] = True

    total_target = int(round(len(df) * float(cfg["prevalence"])))
    pattern_mix = cfg["pattern_mix"]
    counts = _allocate_counts(total_target, pattern_mix)

    for pattern, target_count in counts.items():
        remaining = target_count

        while remaining > 0:
            if pattern == "high_amount_legitimate":
                episode_size = 1
            else:
                episode_size = _sample_int_range(
                    rng,
                    cfg[pattern]["episode_size"],
                )
            episode_size = min(episode_size, remaining)

            indices = _choose_episode_rows(
                selected_used,
                episode_size,
                rng,
                eligible_mask=eligible,
            )
            selected_used[indices] = True
            anchor_index = int(indices[0])
            anchor_timestamp = pd.Timestamp(
                df.loc[anchor_index, "claim_submission_timestamp"]
            )

            if pattern == "high_amount_legitimate":
                _inflate_amounts(
                    df,
                    indices,
                    cfg[pattern]["multiplier"],
                    config,
                    rng,
                )

            elif pattern == "high_frequency_legitimate":
                window_spec = cfg[pattern]["window_days"]
                window_days = int(
                    rng.integers(
                        int(window_spec["min"]),
                        int(window_spec["max"]) + 1,
                    )
                )
                _set_episode_timestamps(
                    df,
                    indices,
                    anchor_timestamp,
                    window_days,
                    config,
                    rng,
                )
                customer_id = str(df.loc[anchor_index, "customer_id"])
                _set_customer(
                    df,
                    indices,
                    customer_id,
                    customer_lookup,
                    policy_lookup,
                )

            elif pattern == "unusual_provider_legitimate":
                window_spec = cfg[pattern]["window_days"]
                window_days = int(
                    rng.integers(
                        int(window_spec["min"]),
                        int(window_spec["max"]) + 1,
                    )
                )
                _set_episode_timestamps(
                    df,
                    indices,
                    anchor_timestamp,
                    window_days,
                    config,
                    rng,
                )
                provider_id = str(df.loc[anchor_index, "provider_id"])
                provider_type = str(provider_lookup.loc[provider_id, "provider_type"])
                _set_provider(df, indices, provider_id, provider_lookup)
                for idx in indices:
                    service = _sample_service_for_provider_type(
                        provider_type,
                        config,
                        rng,
                    )
                    _set_service(df, np.asarray([idx]), service, config, rng)
                _recompute_coverage_and_reimbursement(df, indices, config)

            elif pattern == "repeated_service_legitimate":
                window_spec = cfg[pattern]["window_days"]
                window_days = int(
                    rng.integers(
                        int(window_spec["min"]),
                        int(window_spec["max"]) + 1,
                    )
                )
                _set_episode_timestamps(
                    df,
                    indices,
                    anchor_timestamp,
                    window_days,
                    config,
                    rng,
                )
                customer_id = str(df.loc[anchor_index, "customer_id"])
                service = str(df.loc[anchor_index, "service_category"])
                _set_customer(
                    df,
                    indices,
                    customer_id,
                    customer_lookup,
                    policy_lookup,
                )
                _set_service(df, indices, service, config, rng)
                for idx in indices:
                    provider_id = _sample_provider_for_service(
                        providers,
                        service,
                        rng,
                    )
                    _set_provider(
                        df,
                        np.asarray([idx]),
                        provider_id,
                        provider_lookup,
                    )
                _recompute_coverage_and_reimbursement(df, indices, config)

            else:
                raise ValueError(f"Unsupported legitimate anomaly pattern: {pattern}")

            df.loc[indices, "legitimate_anomaly"] = True
            df.loc[indices, "legitimate_anomaly_type"] = pattern
            remaining -= len(indices)

    return df


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
    output = pd.Series(index=df.index, dtype=float)

    grouped = df.groupby(
        group_columns,
        sort=False,
        dropna=False,
    )

    for _, group in grouped:
        ordered = group.sort_values("claim_submission_timestamp")
        rolling = (
            ordered.set_index("claim_submission_timestamp")[value_column]
            .rolling(window, closed="left")
        )

        if aggregation == "count":
            result = rolling.count()
        elif aggregation == "sum":
            result = rolling.sum()
        elif aggregation == "mean":
            result = rolling.mean()
        elif aggregation == "median":
            result = rolling.median()
        else:
            raise ValueError(f"Unsupported rolling aggregation: {aggregation}")

        output.loc[ordered.index] = result.to_numpy()

    if aggregation in {"count", "sum"}:
        output = output.fillna(0.0)

    return output


def add_historical_features(
    claims: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    df = claims.sort_values("claim_submission_timestamp").reset_index(drop=True).copy()

    for window in (7, 30, 90, 365):
        df[f"customer_claims_{window}d"] = _rolling_feature(
            df,
            "customer_id",
            "claim_id",
            f"{window}D",
            "count",
        )

    df["customer_amount_30d"] = _rolling_feature(
        df,
        "customer_id",
        "claim_amount",
        "30D",
        "sum",
    )
    df["customer_amount_365d"] = _rolling_feature(
        df,
        "customer_id",
        "claim_amount",
        "365D",
        "sum",
    )
    df["customer_avg_claim_amount_365d"] = _rolling_feature(
        df,
        "customer_id",
        "claim_amount",
        "365D",
        "mean",
    )

    df["days_since_customer_previous_claim"] = (
        df.groupby("customer_id")["claim_submission_timestamp"]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    df["days_since_same_provider_claim"] = (
        df.groupby(["customer_id", "provider_id"])["claim_submission_timestamp"]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    df["customer_provider_claims_30d"] = _rolling_feature(
        df,
        ["customer_id", "provider_id"],
        "claim_id",
        "30D",
        "count",
    )

    df["same_service_claims_30d"] = _rolling_feature(
        df,
        ["customer_id", "service_category"],
        "claim_id",
        "30D",
        "count",
    )

    df["provider_claims_30d"] = _rolling_feature(
        df,
        "provider_id",
        "claim_id",
        "30D",
        "count",
    )
    df["provider_claims_90d"] = _rolling_feature(
        df,
        "provider_id",
        "claim_id",
        "90D",
        "count",
    )
    df["provider_avg_claim_amount_90d"] = _rolling_feature(
        df,
        "provider_id",
        "claim_amount",
        "90D",
        "mean",
    )

    historical_service_median = (
        df.groupby("service_code")["claim_amount"]
        .transform(lambda series: series.shift(1).expanding().median())
    )

    configured_service_prior = {
        service: float(service_cfg["amount"]["median"])
        for service, service_cfg in config["services"].items()
    }
    service_prior = (
        df["service_category"].map(configured_service_prior).astype(float)
    )

    df["service_typical_amount"] = historical_service_median.fillna(service_prior)
    df["claim_to_service_median_ratio"] = (
        df["claim_amount"] / df["service_typical_amount"]
    )
    df["claim_to_customer_avg_ratio"] = (
        df["claim_amount"] / df["customer_avg_claim_amount_365d"]
    )
    df["claim_to_provider_avg_ratio"] = (
        df["claim_amount"] / df["provider_avg_claim_amount_90d"]
    )

    ratio_columns = [
        "claim_to_service_median_ratio",
        "claim_to_customer_avg_ratio",
        "claim_to_provider_avg_ratio",
    ]
    df[ratio_columns] = df[ratio_columns].replace([np.inf, -np.inf], np.nan)

    return df


# =============================================================================
# Final fraud labels and synthetic diagnostic metadata
# =============================================================================


def finalize_fraud_labels(
    claims: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = claims.copy()
    target = df["_behavior_is_fraud"].astype(int).to_numpy()

    flip_probability = float(config["fraud"]["label_flip_probability"])
    flip_mask = rng.random(len(df)) < flip_probability
    target[flip_mask] = 1 - target[flip_mask]

    mechanism = df["_fraud_mechanism"].astype(object).to_numpy(copy=True)
    difficulty = df["_fraud_difficulty"].astype(object).to_numpy(copy=True)

    # Positive labels created only by label noise are intentionally treated as
    # very hard mixed cases. This preserves the six-mechanism public schema.
    noise_positive = flip_mask & (target == 1) & (mechanism == "none")
    mechanism[noise_positive] = "mixed_pattern"
    difficulty[noise_positive] = "hard"

    became_legitimate = target == 0
    mechanism[became_legitimate] = "none"
    difficulty[became_legitimate] = "none"

    # Diagnostic score only: never used by the predictive model.
    difficulty_uplift = {
        "none": 0.0,
        "hard": 1.0,
        "medium": 1.8,
        "easy": 2.7,
    }
    mechanism_uplift = {
        "none": 0.0,
        "amount_inflation": 0.45,
        "frequency_abuse": 0.40,
        "provider_abnormality": 0.40,
        "repeated_service": 0.35,
        "customer_provider_pattern": 0.40,
        "mixed_pattern": 0.60,
    }

    base_logit = np.log(
        float(config["fraud"]["target_prevalence"])
        / (1.0 - float(config["fraud"]["target_prevalence"]))
    )

    latent = np.full(len(df), base_logit, dtype=float)
    latent += np.asarray([difficulty_uplift[str(value)] for value in difficulty])
    latent += np.asarray([mechanism_uplift[str(value)] for value in mechanism])

    hard_negative = df["legitimate_anomaly"].fillna(False).to_numpy(dtype=bool)
    latent[hard_negative & (target == 0)] += 0.55
    latent += rng.normal(0.0, 0.35, size=len(df))

    probability = 1.0 / (1.0 + np.exp(-np.clip(latent, -30, 30)))
    probability = np.clip(probability, 0.0005, 0.95)

    df["latent_fraud_score"] = latent
    df["synthetic_fraud_probability"] = probability
    df["fraud_difficulty"] = difficulty
    df["fraud_mechanism"] = mechanism
    df["is_fraud"] = target

    return df.drop(
        columns=[
            "_behavior_is_fraud",
            "_fraud_mechanism",
            "_fraud_difficulty",
        ]
    )


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

    provider_mask = rng.random(len(df)) < float(cfg["provider_id_probability"])
    df.loc[provider_mask, "provider_id"] = pd.NA

    prescription_mask = rng.random(len(df)) < float(
        cfg["prescription_missing_probability"]
    )
    df.loc[prescription_mask, "has_prescription"] = pd.NA

    document_mask = rng.random(len(df)) < float(
        cfg["document_count_missing_probability"]
    )
    df.loc[document_mask, "document_count"] = np.nan

    return df


# =============================================================================
# Controlled data-quality perturbations
# =============================================================================


def inject_quality_issues(
    claims: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = claims.copy()
    cfg = config["quality"]

    if not cfg.get("inject_invalid_records", False):
        return df

    invalid_mask = rng.random(len(df)) < float(cfg["invalid_record_probability"])
    invalid_indices = np.flatnonzero(invalid_mask)

    issue_probabilities = cfg.get(
        "invalid_record_types",
        {
            "negative_amount": 1 / 3,
            "zero_units": 1 / 3,
            "service_after_submission": 1 / 3,
        },
    )

    issue_types = _sample_categories(
        rng,
        issue_probabilities,
        len(invalid_indices),
        "invalid record types",
    )

    for idx, issue_type in zip(invalid_indices, issue_types):
        if issue_type == "negative_amount":
            df.loc[idx, "claim_amount"] *= -1

        elif issue_type == "service_after_submission":
            df.loc[idx, "service_date"] = (
                df.loc[idx, "claim_submission_date"] + pd.Timedelta(days=2)
            )

        elif issue_type == "zero_units":
            df.loc[idx, "service_units"] = 0

        else:
            raise ValueError(f"Unsupported invalid record type: {issue_type}")

    return df


# =============================================================================
# Complete pipeline
# =============================================================================


def generate_synthetic_data(
    config_path: str | Path = "configs/data.yaml",
) -> SyntheticDataBundle:
    config = load_config(config_path)
    seed = int(config["project"]["random_seed"])
    rng = np.random.default_rng(seed)

    customers = generate_customers(config, rng)
    providers = generate_providers(config, rng)
    policies = generate_policies(customers, config, rng)

    claims = generate_claims(
        customers,
        providers,
        policies,
        config,
        rng,
    )

    # Fraud and legitimate hard-negative behaviour MUST be materialized before
    # historical features are computed. Otherwise the model sees labels without
    # the corresponding observable behaviour.
    claims = inject_fraud_behaviors(
        claims,
        customers,
        providers,
        policies,
        config,
        rng,
    )

    claims = inject_legitimate_anomaly_behaviors(
        claims,
        customers,
        providers,
        policies,
        config,
        rng,
    )

    claims = add_historical_features(claims, config)
    claims = finalize_fraud_labels(claims, config, rng)
    claims = inject_missingness(claims, config, rng)
    claims = inject_quality_issues(claims, config, rng)

    claims = claims.sort_values("claim_submission_timestamp").reset_index(drop=True)

    return SyntheticDataBundle(
        customers=customers,
        providers=providers,
        policies=policies,
        claims=claims,
    )
