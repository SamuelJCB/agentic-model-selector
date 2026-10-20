"""Command-line entry point.

    python -m model_selector recommend --example her2
    python -m model_selector recommend --patient '{"age": 60, "stage": 3, ...}'
    python -m model_selector benchmark --patients 25
    python -m model_selector monitor
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .candidates import REGISTRY
from .cohort import EXAMPLE_PATIENTS, FEATURES, TREATMENTS, generate_cohort, response_probability
from .graph import design_matrix, run


def cmd_recommend(args) -> None:
    patient = EXAMPLE_PATIENTS[args.example] if args.example else json.loads(args.patient)
    missing = set(FEATURES) - set(patient)
    if missing:
        raise SystemExit(f"patient is missing: {sorted(missing)}")
    state = run(patient, generate_cohort(args.cohort_size, seed=args.seed), args.reasoner, target_auc=args.target_auc)

    print("Agent trace")
    for step in state["trace"]:
        print("  -", step)
    print("\nLeaderboard (5-fold CV on similar patients)")
    for row in state["leaderboard"]:
        print(f"  {row['model']:<24} AUC {row['auc']:.3f}  Brier {row['brier']:.3f}  (round {row['round']})")
    print("\nEstimated probability of response")
    for t, p in sorted(state["recommendation"].items(), key=lambda kv: -kv[1]):
        print(f"  {t:<14} {p:6.1%}  {'#' * int(p * 40)}")
    print("\n" + state["summary"])


def cmd_benchmark(args) -> None:
    """Compare per-patient model selection with fixed global models against the known ground truth."""
    cohort = generate_cohort(args.cohort_size, seed=args.seed)
    test = generate_cohort(args.patients, seed=args.seed + 100)[FEATURES]
    truth = np.stack([response_probability(test, t) for t in TREATMENTS], axis=1)
    best_true = truth.argmax(axis=1)

    def score(choice_idx):
        acc = float((choice_idx == best_true).mean())
        regret = float((truth.max(axis=1) - truth[np.arange(len(test)), choice_idx]).mean())
        return acc, regret

    results = {}
    for name in ("logistic_regression", "hist_gradient_boosting"):
        model = REGISTRY[name].build().fit(design_matrix(cohort), cohort["response"])
        probs = np.stack([model.predict_proba(design_matrix(test.assign(treatment=t)))[:, 1] for t in TREATMENTS], axis=1)
        results[f"global {name}"] = score(probs.argmax(axis=1))

    picks, chosen_models = [], []
    for _, row in test.iterrows():
        state = run(row.to_dict(), cohort, args.reasoner, target_auc=args.target_auc)
        picks.append(TREATMENTS.index(max(state["recommendation"], key=state["recommendation"].get)))
        chosen_models.append(state["selection"]["model"])
    results["agent (per-patient selection)"] = score(np.array(picks))

    print(f"{len(test)} held-out synthetic patients, cohort of {len(cohort)}\n")
    print(f"{'approach':<34} {'best-treatment match':>21} {'mean regret':>12}")
    for name, (acc, regret) in results.items():
        print(f"{name:<34} {acc:>20.0%} {regret:>12.3f}")
    print("\nmodels the agent selected:", pd.Series(chosen_models).value_counts().to_dict())


def cmd_monitor(args) -> None:
    from .monitor import Monitor, SklearnBackend, default_backend
    from .reasoner import get_reasoner

    backend = SklearnBackend if args.backend == "sklearn" else default_backend
    mon = Monitor(generate_cohort(4000, seed=0), new_backend=backend, reasoner=get_reasoner(args.reasoner))
    print(f"backend: {mon.backend_name} | reasoner: {mon.reasoner.name} | reference AUC {mon.auc_reference:.3f}\n")
    batches = [
        ("month 1: stable population", generate_cohort(1500, seed=1)),
        ("month 2: older, sicker referrals", generate_cohort(1500, seed=2, shift=1.5)),
        ("month 3: new immunotherapy regimen", generate_cohort(2000, seed=3, new_regimen=True)),
        ("month 4: new regimen continues", generate_cohort(1500, seed=4, new_regimen=True)),
    ]
    for label, batch in batches:
        e = mon.check(batch)
        print(f"{label}\n  AUC {e['auc_current']:.3f} (ref {e['auc_reference']:.3f}) -> {e['action'].upper()}: "
              f"{e['rationale']}\n  {e['result']}\n")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="model_selector")
    p.add_argument("--reasoner", choices=["auto", "heuristic", "llm"], default="auto",
                   help="auto uses Claude when ANTHROPIC_API_KEY is set, otherwise the heuristic reasoner")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cohort-size", type=int, default=6000)
    p.add_argument("--target-auc", type=float, default=0.75)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recommend")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--example", choices=sorted(EXAMPLE_PATIENTS))
    g.add_argument("--patient", help="JSON object with every feature in cohort.FEATURES")
    r.set_defaults(func=cmd_recommend)
    b = sub.add_parser("benchmark")
    b.add_argument("--patients", type=int, default=25)
    b.set_defaults(func=cmd_benchmark)
    mo = sub.add_parser("monitor")
    mo.add_argument("--backend", choices=["auto", "sklearn"], default="auto")
    mo.set_defaults(func=cmd_monitor)
    args = p.parse_args(argv)
    args.func(args)
