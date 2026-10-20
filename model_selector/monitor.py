"""Self-adaptive monitoring: detect drift, let the reasoner decide, act on the decision.

For every new batch of labeled outcomes the monitor computes feature drift
(population stability index) and the deployed model's current AUC, hands that
report to the reasoner, and then does what the reasoner chose:

* ``none``        keep serving the current model
* ``recalibrate`` refit a Platt scaler on recent outcomes (cheap; fixes shifted base rates)
* ``retrain``     retrain on reference + recent data with AutoGluon (or a scikit-learn
                  fallback when AutoGluon is not installed), then promote it only if it
                  beats the current model on a held-out slice of the recent data
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .cohort import FEATURES
from .graph import design_matrix
from .reasoner import Reasoner, get_reasoner

LABEL = "response"


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index using reference-quantile bins."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: fall back to category shares
        cats = np.union1d(reference, current)
        ref = np.array([(reference == c).mean() for c in cats])
        cur = np.array([(current == c).mean() for c in cats])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        ref = np.histogram(reference, edges)[0] / len(reference)
        cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


class SklearnBackend:
    name = "sklearn"

    def fit(self, df: pd.DataFrame):
        self.model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)
        self.model.fit(design_matrix(df), df[LABEL])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(design_matrix(df))[:, 1]


class AutoGluonBackend:
    name = "autogluon"

    def __init__(self, time_limit: int = 60, presets: str = "medium_quality"):
        self.time_limit, self.presets = time_limit, presets

    def fit(self, df: pd.DataFrame):
        from autogluon.tabular import TabularPredictor

        train = design_matrix(df).assign(**{LABEL: df[LABEL].to_numpy()})
        self.predictor = TabularPredictor(label=LABEL, eval_metric="roc_auc", path=tempfile.mkdtemp(prefix="ag_"),
                                          verbosity=0)
        self.predictor.fit(train, time_limit=self.time_limit, presets=self.presets)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.predictor.predict_proba(design_matrix(df), as_multiclass=False).to_numpy()


def default_backend():
    """AutoGluon when it is installed, otherwise the scikit-learn fallback."""
    try:
        import autogluon.tabular  # noqa: F401

        return AutoGluonBackend()
    except ImportError:
        return SklearnBackend()


@dataclass
class Monitor:
    reference: pd.DataFrame
    new_backend: Callable[[], object] = default_backend  # factory: returns an unfitted backend
    reasoner: Reasoner = field(default_factory=get_reasoner)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.calibrator = None
        # Reference AUC is measured out-of-sample: score a held-out split with a model fit on the rest.
        fit, hold = train_test_split(self.reference, test_size=0.25, random_state=0, stratify=self.reference[LABEL])
        probe = self.new_backend().fit(fit)
        self.auc_reference = float(roc_auc_score(hold[LABEL], probe.predict_proba(hold)))
        self.model = self.new_backend().fit(self.reference)
        self.backend_name = self.model.name

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.model.predict_proba(df)
        if self.calibrator is not None:
            p = self.calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return p

    def check(self, batch: pd.DataFrame) -> dict:
        report = {
            "n_batch": len(batch),
            "psi": {f: round(psi(self.reference[f].to_numpy(), batch[f].to_numpy()), 3) for f in FEATURES},
            "auc_reference": round(self.auc_reference, 3),
            "auc_current": round(float(roc_auc_score(batch[LABEL], self.predict_proba(batch))), 3),
            "base_rate_reference": round(float(self.reference[LABEL].mean()), 3),
            "base_rate_current": round(float(batch[LABEL].mean()), 3),
        }
        decision = self.reasoner.monitor_action(report)
        outcome = self._act(decision["action"], batch)
        entry = {**report, **decision, **outcome}
        self.history.append(entry)
        return entry

    def _act(self, action: str, batch: pd.DataFrame) -> dict:
        if action == "recalibrate":
            raw = self.model.predict_proba(batch)
            self.calibrator = LogisticRegression().fit(_logit(raw).reshape(-1, 1), batch[LABEL])
            return {"result": "recalibrated on recent outcomes"}
        if action == "retrain":
            fit, hold = train_test_split(batch, test_size=0.3, random_state=0, stratify=batch[LABEL])
            challenger = self.new_backend().fit(pd.concat([self.reference, fit], ignore_index=True))
            auc_new = float(roc_auc_score(hold[LABEL], challenger.predict_proba(hold)))
            auc_old = float(roc_auc_score(hold[LABEL], self.predict_proba(hold)))
            if auc_new > auc_old:
                self.model, self.calibrator = challenger, None
                self.reference = pd.concat([self.reference, batch], ignore_index=True)
                self.auc_reference = auc_new
                return {"result": f"promoted challenger ({auc_old:.3f} -> {auc_new:.3f} AUC on held-out recent data)"}
            return {"result": f"kept incumbent (challenger {auc_new:.3f} <= incumbent {auc_old:.3f})"}
        return {"result": "no change"}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
