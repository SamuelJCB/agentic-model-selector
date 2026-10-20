import numpy as np
import pandas as pd
import pytest

from model_selector.candidates import REGISTRY
from model_selector.cohort import EXAMPLE_PATIENTS, TREATMENTS, generate_cohort, response_probability
from model_selector.graph import build_graph, run
from model_selector.monitor import Monitor, SklearnBackend, psi
from model_selector.reasoner import HeuristicReasoner, LLMReasoner


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(3000, seed=0)


def test_graph_has_expected_nodes():
    nodes = set(build_graph().get_graph().nodes)
    assert {"find_similar", "profile", "propose", "evaluate", "select_and_recommend"} <= nodes


@pytest.mark.parametrize("example,expected", [("her2", "targeted"), ("high-tmb", "immunotherapy"),
                                              ("proliferative", "chemotherapy")])
def test_recommendation_matches_ground_truth(cohort, example, expected):
    patient = EXAMPLE_PATIENTS[example]
    truth = {t: response_probability(pd.DataFrame([patient]), t)[0] for t in TREATMENTS}
    assert max(truth, key=truth.get) == expected  # sanity check on the generator itself
    state = run(patient, cohort, "heuristic", k_neighbors=1000)
    assert max(state["recommendation"], key=state["recommendation"].get) == expected
    assert state["selection"]["model"] in REGISTRY


def test_search_widens_when_below_target(cohort):
    state = run(EXAMPLE_PATIENTS["her2"], cohort, "heuristic", k_neighbors=800, target_auc=0.99)
    assert state["round"] == 2
    assert len(state["leaderboard"]) > 3


def test_heuristic_respects_sample_size():
    out = HeuristicReasoner().propose({"n_rows": 250, "positive_rate": 0.4}, tried=[], expand=True)
    assert all(REGISTRY[c].min_rows <= 250 for c in out["candidates"])


def test_llm_reasoner_falls_back_on_error():
    class Broken:
        class messages:
            @staticmethod
            def create(**_):
                raise RuntimeError("network down")

    r = LLMReasoner.__new__(LLMReasoner)
    r.client, r.model, r.fallback = Broken(), "x", HeuristicReasoner()
    out = r.propose({"n_rows": 1000, "positive_rate": 0.4}, tried=[], expand=False)
    assert out["candidates"]


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1, 1, 5000)) > 0.25


def test_monitor_retrains_on_concept_drift(cohort):
    mon = Monitor(cohort, new_backend=SklearnBackend, reasoner=HeuristicReasoner())
    assert mon.check(generate_cohort(1500, seed=1))["action"] in {"none", "recalibrate"}
    assert mon.check(generate_cohort(2000, seed=3, new_regimen=True))["action"] == "retrain"
