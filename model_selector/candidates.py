"""Registry of candidate model families the agent can choose from.

Each entry carries the properties the reasoner uses to decide whether it is a
sensible candidate for a given dataset: how much data it needs, whether it
captures nonlinear interactions, and whether a clinician can inspect it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class Candidate:
    name: str
    build: Callable[[], object]
    min_rows: int
    nonlinear: bool
    interpretable: bool
    note: str


def _xgb():
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss", verbosity=0)


REGISTRY: dict[str, Candidate] = {
    c.name: c
    for c in [
        Candidate("logistic_regression",
                  lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                  min_rows=50, nonlinear=False, interpretable=True,
                  note="Linear log-odds; stable on small samples and easy to audit."),
        Candidate("logistic_interactions",
                  lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, interaction_only=True, include_bias=False),
                                        StandardScaler(), LogisticRegression(C=0.1, max_iter=4000)),
                  min_rows=300, nonlinear=True, interpretable=True,
                  note="Pairwise interactions (e.g. treatment x biomarker) with strong regularization."),
        Candidate("random_forest",
                  lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0),
                  min_rows=200, nonlinear=True, interpretable=False,
                  note="Robust to feature scale and outliers; captures interactions."),
        Candidate("hist_gradient_boosting",
                  lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0),
                  min_rows=400, nonlinear=True, interpretable=False,
                  note="Strong default for tabular data; handles missing values natively."),
        Candidate("xgboost", _xgb, min_rows=400, nonlinear=True, interpretable=False,
                  note="Regularized boosting; competitive when interactions are strong."),
        Candidate("knn",
                  lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance")),
                  min_rows=150, nonlinear=True, interpretable=True,
                  note="Case-based reasoning over similar patients; degrades with many features."),
    ]
}
