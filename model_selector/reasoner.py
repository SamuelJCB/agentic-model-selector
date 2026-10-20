"""Reasoning backends for the agent's decision points.

`LLMReasoner` asks Claude to make each decision and return structured JSON.
`HeuristicReasoner` makes the same decisions with explicit rules, so the graph
runs deterministically offline and in CI. Both expose the same three methods,
and the LLM backend falls back to the heuristic one if a call fails or returns
something invalid.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from .candidates import REGISTRY


class Reasoner(Protocol):
    name: str

    def propose(self, profile: dict, tried: list[str], expand: bool) -> dict: ...

    def explain(self, patient: dict, selection: dict, recommendation: dict) -> str: ...

    def monitor_action(self, report: dict) -> dict: ...


class HeuristicReasoner:
    name = "heuristic"

    def propose(self, profile: dict, tried: list[str], expand: bool) -> dict:
        n, pos_rate = profile["n_rows"], profile["positive_rate"]
        pool = [c for c in REGISTRY.values() if c.name not in tried and c.min_rows <= n]
        if not expand:
            # First pass: one interpretable baseline plus the strongest nonlinear learners that fit the sample size.
            picks = [c.name for c in pool if c.interpretable and not c.nonlinear][:1]
            picks += [c.name for c in pool if c.nonlinear and not c.interpretable][:2]
            why = (f"{n} similar patients with a {pos_rate:.0%} response rate. Starting with an auditable linear "
                   "baseline and two tree ensembles that can capture treatment-by-biomarker interactions.")
        else:
            picks = [c.name for c in pool]
            why = "Best cross-validated AUC is below target; widening the search to every remaining family."
        return {"candidates": picks, "rationale": why}

    def explain(self, patient: dict, selection: dict, recommendation: dict) -> str:
        ranked = sorted(recommendation.items(), key=lambda kv: -kv[1])
        top, p_top = ranked[0]
        runner, p_run = ranked[1]
        return (f"Selected {selection['model']} (CV AUC {selection['auc']:.3f}, Brier {selection['brier']:.3f}) "
                f"over {len(selection['leaderboard']) - 1} alternative(s). Estimated response is highest for "
                f"{top} ({p_top:.0%}), ahead of {runner} ({p_run:.0%}). Estimates come from patients similar on "
                "stage, performance status, and biomarker profile and should be read as decision support only.")

    def monitor_action(self, report: dict) -> dict:
        drifted = [f for f, v in report["psi"].items() if v > 0.25]
        decay = report["auc_reference"] - report["auc_current"]
        if decay > 0.05 or len(drifted) >= 3:
            action = "retrain"
        elif decay > 0.035 or drifted:
            action = "recalibrate"
        else:
            action = "none"
        return {"action": action,
                "rationale": f"AUC change {-decay:+.3f}; features with PSI > 0.25: {drifted or 'none'}."}


class LLMReasoner:
    """Claude-backed reasoner. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    name = "llm"

    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("MODEL_SELECTOR_LLM", "claude-sonnet-4-5")
        self.fallback = HeuristicReasoner()

    def _ask(self, system: str, payload: dict) -> dict:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=system + " Respond with a single JSON object and nothing else.",
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        )
        text = msg.content[0].text
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group(0) if match else text)

    def propose(self, profile: dict, tried: list[str], expand: bool) -> dict:
        options = {c.name: {"min_rows": c.min_rows, "nonlinear": c.nonlinear,
                            "interpretable": c.interpretable, "note": c.note}
                   for c in REGISTRY.values() if c.name not in tried}
        try:
            out = self._ask(
                "You are an ML engineer choosing candidate models for a clinical response-prediction task. "
                "Pick 2-4 candidates from `options` suited to the data profile. If `expand` is true, earlier "
                "candidates underperformed, so prefer families with different inductive biases. Return "
                '{"candidates": [names], "rationale": "one or two sentences"}.',
                {"profile": profile, "options": options, "already_tried": tried, "expand": expand},
            )
            picks = [c for c in out["candidates"] if c in options and REGISTRY[c].min_rows <= profile["n_rows"]]
            if picks:
                return {"candidates": picks, "rationale": out.get("rationale", "")}
        except Exception:
            pass
        return self.fallback.propose(profile, tried, expand)

    def explain(self, patient: dict, selection: dict, recommendation: dict) -> str:
        try:
            out = self._ask(
                "Summarize this model-selection result for a clinical research team in 3 sentences: which model "
                "was chosen and why, what the per-treatment response estimates are, and one caveat. Do not give "
                'medical advice. Return {"summary": "..."}.',
                {"patient": patient, "selection": selection, "estimates": recommendation},
            )
            return out["summary"]
        except Exception:
            return self.fallback.explain(patient, selection, recommendation)

    def monitor_action(self, report: dict) -> dict:
        try:
            out = self._ask(
                "You monitor a deployed clinical prediction model. Given feature PSI values and AUC on recent "
                'labeled data versus the reference, choose an action from "none", "recalibrate", "retrain". '
                'Return {"action": "...", "rationale": "..."}.',
                report,
            )
            if out.get("action") in {"none", "recalibrate", "retrain"}:
                return out
        except Exception:
            pass
        return self.fallback.monitor_action(report)


def get_reasoner(kind: str = "auto") -> Reasoner:
    if kind == "heuristic" or (kind == "auto" and not os.environ.get("ANTHROPIC_API_KEY")):
        return HeuristicReasoner()
    try:
        return LLMReasoner()
    except ImportError:
        if kind == "llm":
            raise
        return HeuristicReasoner()
