"""Synthetic oncology cohort.

Each row is a patient, the treatment they received, and whether they
responded. Response depends on patient-specific parameters in ways that mirror
well-known clinical patterns (HER2 status drives response to targeted therapy,
PD-L1 expression and tumor mutational burden drive response to immunotherapy,
proliferation drives response to chemotherapy, performance status lowers
response to everything). Treatment assignment is confounded by those same
parameters, as it is in real observational data.

This data is synthetic. It exists so the agent can be run and tested without
access to protected health information. Nothing here is clinical guidance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TREATMENTS = ("chemotherapy", "targeted", "immunotherapy")
FEATURES = [
    "age",
    "stage",
    "tumor_size_cm",
    "ecog",
    "her2_positive",
    "er_positive",
    "ki67_pct",
    "pdl1_tps",
    "tmb_mut_mb",
    "prior_lines",
    "ldh_ratio",
]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_cohort(n: int = 4000, seed: int = 0, shift: float = 0.0, new_regimen: bool = False) -> pd.DataFrame:
    """Generate a cohort.

    `shift` > 0 simulates covariate drift (older, sicker, higher-TMB patients).
    `new_regimen` simulates concept drift: a newer immunotherapy combination that
    works for PD-L1-low patients and changes how biomarkers relate to response.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": np.clip(rng.normal(61 + 6 * shift, 11, n), 25, 92).round(),
            "stage": rng.choice([1, 2, 3, 4], n, p=[0.15, 0.3, 0.3, 0.25]),
            "tumor_size_cm": np.clip(rng.lognormal(1.0, 0.5, n), 0.3, 15).round(1),
            "ecog": rng.choice([0, 1, 2, 3], n, p=[0.35 - 0.1 * shift, 0.4, 0.18 + 0.05 * shift, 0.07 + 0.05 * shift]),
            "her2_positive": rng.binomial(1, 0.22, n),
            "er_positive": rng.binomial(1, 0.6, n),
            "ki67_pct": np.clip(rng.beta(2, 4, n) * 100, 1, 95).round(),
            "pdl1_tps": np.clip(rng.beta(0.8, 2.2, n) * 100, 0, 100).round(),
            "tmb_mut_mb": np.clip(rng.lognormal(1.6 + 0.4 * shift, 0.8, n), 0.2, 80).round(1),
            "prior_lines": rng.choice([0, 1, 2, 3], n, p=[0.45, 0.3, 0.17, 0.08]),
            "ldh_ratio": np.clip(rng.lognormal(0.0, 0.35, n), 0.4, 5).round(2),
        }
    )

    # Confounded assignment: clinicians already steer patients toward plausible therapies.
    logits = np.stack(
        [
            0.02 * df["ki67_pct"] - 0.3 * df["ecog"],
            2.0 * df["her2_positive"] - 0.5,
            0.025 * df["pdl1_tps"] + 0.03 * df["tmb_mut_mb"] - 1.2,
        ],
        axis=1,
    )
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    choice = np.array([rng.choice(3, p=p) for p in probs])
    df["treatment"] = np.array(TREATMENTS)[choice]

    df["response"] = rng.binomial(1, response_probability(df, df["treatment"], new_regimen))
    return df


def response_probability(patients: pd.DataFrame, treatment, new_regimen: bool = False) -> np.ndarray:
    """Ground-truth response probability. Used only by the generator and the tests."""
    p = patients
    treatment = np.broadcast_to(np.asarray(treatment), (len(p),))
    base = 0.4 - 0.55 * p["ecog"] - 0.25 * (p["stage"] - 2) - 0.25 * p["prior_lines"] - 0.6 * (p["ldh_ratio"] - 1)
    chemo = 0.035 * (p["ki67_pct"] - 30) - 0.02 * (p["age"] - 60)
    targeted = 2.6 * p["her2_positive"] - 1.3 + 0.4 * p["er_positive"]
    # Immunotherapy benefit saturates with TMB and depends on PD-L1 nonlinearly.
    immuno = 1.8 * np.tanh((p["pdl1_tps"] - 40) / 20) + 1.1 * np.log1p(p["tmb_mut_mb"]) - 2.6
    if new_regimen:
        # Combination regimen: PD-L1 matters far less, and ER+ disease now responds as well.
        immuno = 0.4 * np.tanh((p["pdl1_tps"] - 40) / 20) + 0.8 * np.log1p(p["tmb_mut_mb"]) - 0.9 + 0.9 * p["er_positive"]
    effect = np.select(
        [treatment == "chemotherapy", treatment == "targeted", treatment == "immunotherapy"],
        [chemo, targeted, immuno],
    )
    return _sigmoid(base + effect).to_numpy()


EXAMPLE_PATIENTS = {
    "her2": dict(age=54, stage=3, tumor_size_cm=3.1, ecog=1, her2_positive=1, er_positive=1,
                 ki67_pct=28, pdl1_tps=5, tmb_mut_mb=3.2, prior_lines=0, ldh_ratio=1.0),
    "high-tmb": dict(age=67, stage=4, tumor_size_cm=4.5, ecog=1, her2_positive=0, er_positive=0,
                     ki67_pct=35, pdl1_tps=70, tmb_mut_mb=22.0, prior_lines=1, ldh_ratio=1.2),
    "proliferative": dict(age=45, stage=2, tumor_size_cm=2.2, ecog=0, her2_positive=0, er_positive=0,
                          ki67_pct=75, pdl1_tps=10, tmb_mut_mb=4.0, prior_lines=0, ldh_ratio=0.9),
}
