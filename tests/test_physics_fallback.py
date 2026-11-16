"""When the classifier is unsure but the physics evidence is strong, the physics verdict is reported (marked);
a confident but contradicted classifier still gives 'Unknown'."""
import numpy as np

from conftest import needs_index, needs_model
from intellimaintain.agent.agent import DiagnosisAgent
from intellimaintain.agent.llm import LLM
from intellimaintain.ml.physics import defect_frequencies

pytestmark = [needs_model, needs_index]


def _synthetic_outer_race(fs=12000, seconds=2.0, rpm=1772.0, seed=0):
    """Impacts at BPFO ringing a 3 kHz resonance - unlike CWRU, so the classifier is unsure."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = 0.03 * rng.normal(size=n) + 0.02 * np.sin(2 * np.pi * rpm / 60 * t)
    ring = np.exp(-900 * t[:120]) * np.sin(2 * np.pi * 3000 * t[:120])
    for k in np.arange(0, seconds, 1 / defect_frequencies(rpm)["bpfo_hz"]):
        i = int(k * fs)
        x[i:i + 120] += 0.8 * ring[: n - i]
    return x


def test_unsure_classifier_strong_physics_reports_physics():
    d = DiagnosisAgent(LLM("offline")).run("What is the condition of this bearing?",
                                           signal=_synthetic_outer_race(), rpm=1772)
    assert d.fault_type == "Outer race"  # never "Unknown" when physics is clear and the classifier unsure
    if d.ml_result.fault_type != "Outer race":
        assert d.ml_result.low_confidence and d.fault_source == "physics check (classifier unsure)"
        assert d.answer.startswith("Most likely outer race fault [PHYS]")


def test_confident_but_contradicted_label_still_abstains(demo_signal):
    # the stress test injects a confident wrong label; the governance rule must still abstain
    d = DiagnosisAgent(LLM("offline")).run("What is wrong?", signal=demo_signal, rpm=1797, override_label="IR_007")
    if d.fault_frequencies.consistency == "conflict":
        assert d.fault_type == "Unknown" and d.fault_source == "undecided (conflict)"
