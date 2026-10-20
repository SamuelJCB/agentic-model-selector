"""LangGraph workflow that selects a model for a specific patient and uses it.

    find_similar ──> profile ──> propose ──> evaluate ──┬──> select_and_recommend ──> END
                                    ^                   │
                                    └──── widen search ─┘  (best AUC below target)

The agent does not pick one global model. For each patient it builds a cohort
of clinically similar patients, characterizes that data, lets the reasoner
choose candidate model families, cross-validates them, and widens the search
if nothing clears the quality bar. The winning model then estimates the
patient's probability of response under each treatment option.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .candidates import REGISTRY
from .cohort import FEATURES, TREATMENTS
from .reasoner import Reasoner, get_reasoner


class AgentState(TypedDict, total=False):
    patient: dict
    cohort: pd.DataFrame
    reasoner: Reasoner
    k_neighbors: int
    target_auc: float
    max_rounds: int
    similar: pd.DataFrame
    profile: dict
    round: int
    candidates: list[str]
    tried: list[str]
    leaderboard: list[dict]
    selection: dict
    recommendation: dict[str, float]
    summary: str
    trace: list[str]


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Patient features plus one-hot treatment, so one model can score every option (an S-learner)."""
    X = df[FEATURES].astype(float).copy()
    for t in TREATMENTS:
        X[f"tx_{t}"] = (df["treatment"] == t).astype(float)
    return X


def find_similar(state: AgentState) -> AgentState:
    cohort, k = state["cohort"], min(state.get("k_neighbors", 1500), len(state["cohort"]))
    scaler = StandardScaler().fit(cohort[FEATURES])
    nn = NearestNeighbors(n_neighbors=k).fit(scaler.transform(cohort[FEATURES]))
    query = scaler.transform(pd.DataFrame([state["patient"]])[FEATURES])
    _, idx = nn.kneighbors(query)
    similar = cohort.iloc[idx[0]].reset_index(drop=True)
    return {"similar": similar, "trace": state.get("trace", []) + [f"find_similar: {k} nearest patients"]}


def profile(state: AgentState) -> AgentState:
    df = state["similar"]
    prof = {
        "n_rows": len(df),
        "n_features": len(FEATURES) + len(TREATMENTS),
        "positive_rate": float(df["response"].mean()),
        "treatment_counts": df["treatment"].value_counts().to_dict(),
        "response_rate_by_treatment": df.groupby("treatment")["response"].mean().round(3).to_dict(),
        "min_class_count": int(min(df["response"].sum(), (1 - df["response"]).sum())),
    }
    return {"profile": prof, "tried": [], "leaderboard": [], "round": 0,
            "trace": state["trace"] + [f"profile: n={prof['n_rows']}, response rate {prof['positive_rate']:.1%}"]}


def propose(state: AgentState) -> AgentState:
    expand = state["round"] > 0
    out = state["reasoner"].propose(state["profile"], state["tried"], expand)
    return {"candidates": out["candidates"], "round": state["round"] + 1,
            "trace": state["trace"] + [f"propose[{state['reasoner'].name}]: {out['candidates']} | {out['rationale']}"]}


def evaluate(state: AgentState) -> AgentState:
    df = state["similar"]
    X, y = design_matrix(df), df["response"].to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    board = list(state["leaderboard"])
    for name in state["candidates"]:
        proba = cross_val_predict(REGISTRY[name].build(), X, y, cv=folds, method="predict_proba")[:, 1]
        board.append({"model": name, "auc": float(roc_auc_score(y, proba)),
                      "brier": float(brier_score_loss(y, proba)), "round": state["round"]})
    board.sort(key=lambda r: (-r["auc"], r["brier"]))
    best = board[0]
    return {"leaderboard": board, "tried": state["tried"] + state["candidates"],
            "trace": state["trace"] + [f"evaluate: best so far {best['model']} AUC {best['auc']:.3f}"]}


def route(state: AgentState) -> str:
    remaining = [c for c in REGISTRY.values()
                 if c.name not in state["tried"] and c.min_rows <= state["profile"]["n_rows"]]
    below_target = state["leaderboard"][0]["auc"] < state.get("target_auc", 0.80)
    if below_target and remaining and state["round"] < state.get("max_rounds", 2):
        return "propose"
    return "select_and_recommend"


def select_and_recommend(state: AgentState) -> AgentState:
    best = state["leaderboard"][0]
    df = state["similar"]
    model = REGISTRY[best["model"]].build().fit(design_matrix(df), df["response"])
    rows = pd.DataFrame([{**state["patient"], "treatment": t} for t in TREATMENTS])
    probs = model.predict_proba(design_matrix(rows))[:, 1]
    rec = {t: float(p) for t, p in zip(TREATMENTS, probs)}
    selection = {**best, "leaderboard": state["leaderboard"]}
    summary = state["reasoner"].explain(state["patient"], selection, rec)
    top = max(rec, key=rec.get)
    return {"selection": selection, "recommendation": rec, "summary": summary,
            "trace": state["trace"] + [f"select: {best['model']} -> highest estimated response: {top}"]}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("find_similar", find_similar)
    g.add_node("profile", profile)
    g.add_node("propose", propose)
    g.add_node("evaluate", evaluate)
    g.add_node("select_and_recommend", select_and_recommend)
    g.add_edge(START, "find_similar")
    g.add_edge("find_similar", "profile")
    g.add_edge("profile", "propose")
    g.add_edge("propose", "evaluate")
    g.add_conditional_edges("evaluate", route, {"propose": "propose", "select_and_recommend": "select_and_recommend"})
    g.add_edge("select_and_recommend", END)
    return g.compile()


def run(patient: dict, cohort: pd.DataFrame, reasoner: str | Reasoner = "auto", **options: Any) -> AgentState:
    r = get_reasoner(reasoner) if isinstance(reasoner, str) else reasoner
    return build_graph().invoke({"patient": patient, "cohort": cohort, "reasoner": r, "trace": [], **options})
