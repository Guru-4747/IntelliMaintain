"""Copilot evaluation: retrieval metrics, 2x3 ablation, C0 vs C1, wrong-ML stress test.

All metrics are computed automatically and deterministically (no LLM judge needed):
  fault_correct      final fault type == ground truth (signal cases)
  key_fact_recall    share of reference key facts present in the answer
  citation_coverage  share of answer sentences that carry a valid citation
  faithfulness       share of answer sentences supported by their cited source
                     (content-word overlap >= 0.5 with the cited passage / tool output; uncited = unsupported)
  context_precision  share of retrieved passages that come from a relevant source
  context_recall     share of relevant sources that were retrieved
  insufficient_ok    out-of-scope questions answered with "insufficient evidence"
An optional LLM-as-judge score (RAGAS-style faithfulness/relevance, 1-5) is added with --judge.
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402
from ..agent.agent import DiagnosisAgent  # noqa: E402
from ..agent.llm import LLM, parse_json  # noqa: E402
from ..agent.prompts import format_ml, format_physics  # noqa: E402
from ..agent.verify import CITE_RE, split_sentences  # noqa: E402
from ..ml import classifier as clf_module  # noqa: E402
from ..ml import cwru  # noqa: E402
from ..rag.retriever import load_retriever, tokenize  # noqa: E402
from ..schemas import Diagnosis  # noqa: E402

CASES_PATH = config.EVAL_DIR / "test_cases.csv"
SIGNAL_SECONDS = 4.0
CONFIGS = list(product([False, True], ["none", "basic", "improved"]))  # (use_ml, retrieval)


def load_cases(path: Path = CASES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")
    required = {"case_id", "fault_type", "question", "reference_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test case sheet is missing columns: {missing}")
    return df


def case_signal(row) -> tuple[np.ndarray | None, float | None]:
    if not row.get("cwru_file"):
        return None, None
    c = cwru.BY_NUMBER[int(float(row["cwru_file"]))]
    x = cwru.load_file(c)
    start = int(1.0 * config.FS)
    return x[start:start + int(SIGNAL_SECONDS * config.FS)], float(row.get("rpm") or c.rpm)


# ------------------------------------------------------------------ metrics
def answer_text(d: Diagnosis) -> str:
    return " ".join([d.summary, d.explanation] + d.recommended_actions)


def key_fact_recall(d: Diagnosis, facts: str) -> float:
    keys = [k.strip().lower() for k in facts.split(";") if k.strip()]
    if not keys:
        return float("nan")
    text = answer_text(d).lower()
    return sum(k in text for k in keys) / len(keys)


def faithfulness(d: Diagnosis) -> float:
    sents = [s for t in [d.explanation] + d.recommended_actions for s in split_sentences(t)]
    if not sents:
        return float("nan")
    sources = {str(i): set(tokenize(e.text)) for i, e in enumerate(d.evidence, 1)}
    sources["ML"] = set(tokenize(format_ml(d.ml_result))) if d.ml_result else set()
    sources["PHYS"] = set(tokenize(format_physics(d.fault_frequencies))) if d.fault_frequencies else set()
    supported = 0
    for s in sents:
        tags = [t.upper() for t in CITE_RE.findall(s)]
        words = set(tokenize(CITE_RE.sub("", s)))
        if not tags or not words:
            continue
        cited = set().union(*(sources.get(t, set()) for t in tags))
        if len(words & cited) / len(words) >= 0.5:
            supported += 1
    return supported / len(sents)


def context_scores(d: Diagnosis, relevant: str) -> tuple[float, float]:
    rel = {s.strip() for s in relevant.split(";") if s.strip()}
    if not rel or not d.evidence:
        return (float("nan"), float("nan")) if not rel else (0.0, 0.0)
    got = [e.source for e in d.evidence]
    precision = sum(s in rel for s in got) / len(got)
    recall = len(rel & set(got)) / len(rel)
    return precision, recall


def llm_judge(llm: LLM, d: Diagnosis, reference: str) -> dict:
    """Optional RAGAS-style judge (1-5). Only used with --judge and a real LLM."""
    ctx = "\n".join(f"[{i}] {e.text[:600]}" for i, e in enumerate(d.evidence, 1))
    prompt = (f"QUESTION: {d.question}\nREFERENCE ANSWER: {reference}\nCONTEXT:\n{ctx or '(none)'}\n"
              f"ML: {format_ml(d.ml_result)}\nANSWER: {answer_text(d)}\n\nRate 1-5: faithfulness (claims supported "
              'by context/ML), relevance (answers the question), correctness (agrees with reference). JSON only: '
              '{"faithfulness": n, "relevance": n, "correctness": n}')
    try:
        j = parse_json(llm.chat("You are a strict evaluator of maintenance answers.", prompt, json_mode=True,
                                max_tokens=100))
        return {f"judge_{k}": float(j.get(k, np.nan)) for k in ("faithfulness", "relevance", "correctness")}
    except Exception:  # noqa: BLE001
        return {}


def score(d: Diagnosis, row) -> dict:
    truth = row["fault_type"]
    cp, cr = context_scores(d, row.get("relevant_sources", ""))
    out = {
        "case_id": row["case_id"], "config": d.config_name, "truth": truth, "pred": d.fault_type,
        "fault_correct": float(d.fault_type == truth) if truth not in ("N/A", "Unknown") else np.nan,
        "key_fact_recall": key_fact_recall(d, row.get("key_facts", "")),
        "citation_coverage": d.citation_coverage, "faithfulness": faithfulness(d),
        "context_precision": cp, "context_recall": cr, "insufficient": float(d.insufficient_evidence),
        "insufficient_ok": float(d.insufficient_evidence) if truth == "Unknown" else np.nan,
        "n_warnings": len(d.warnings), "n_tool_calls": len(d.tool_trace),
        "grounding": d.grounding_score, "llm_answer_verified": float("(verified)" in d.answer_source),
    }
    return out


# ------------------------------------------------------------------ experiments
def run_ablation(llm: LLM, cases: pd.DataFrame, judge: bool = False, verbose: bool = True,
                 classifier: str = "rf") -> pd.DataFrame:
    rows = []
    signals = {r["case_id"]: case_signal(r) for _, r in cases.iterrows()}
    for use_ml, retrieval in CONFIGS:
        agent = DiagnosisAgent(llm=llm, use_ml=use_ml, retrieval=retrieval, classifier=classifier)
        t0 = time.time()
        for _, r in cases.iterrows():
            x, rpm = signals[r["case_id"]]
            d = agent.run(r["question"], signal=x, rpm=rpm)
            s = score(d, r)
            if judge and not llm.is_offline:
                s.update(llm_judge(llm, d, r["reference_answer"]))
            s["answer"] = answer_text(d)[:600]
            rows.append(s)
        if verbose:
            print(f"  config {agent.config_name:<14} done in {time.time() - t0:5.1f} s")
    return pd.DataFrame(rows)


def retrieval_benchmark(cases: pd.DataFrame) -> pd.DataFrame:
    """Hit@k and MRR of the knowledge questions for each retrieval mode (question text only)."""
    r = load_retriever()
    out = []
    kq = cases[cases["relevant_sources"].str.len() > 0]
    for mode in ("bm25", "basic", "improved"):
        hits, rr, prec = [], [], []
        for _, c in kq.iterrows():
            rel = set(c["relevant_sources"].split(";"))
            ev = r.search(c["question"], top_k=config.TOP_K, mode=mode)
            srcs = [e.source for e in ev]
            first = next((i for i, s in enumerate(srcs, 1) if s in rel), None)
            hits.append(first is not None)
            rr.append(1 / first if first else 0.0)
            prec.append(np.mean([s in rel for s in srcs]) if srcs else 0.0)
        out.append({"mode": mode, f"hit@{config.TOP_K}": np.mean(hits), "MRR": np.mean(rr),
                    f"precision@{config.TOP_K}": np.mean(prec), "n_queries": len(kq)})
    return pd.DataFrame(out)


def casual_benchmark() -> tuple[pd.DataFrame, dict]:
    """Informal questions (typos, short forms, everyday words): intent accuracy and retrieval with vs without
    the language-understanding layer (agent/nlu.py)."""
    from ..agent.nlu import understand

    path = config.EVAL_DIR / "casual_questions.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    r = load_retriever()
    rows = []
    for _, c in df.iterrows():
        q = understand(c["question"], has_signal=c["has_signal"] == "1")
        rel = {x for x in c["relevant_sources"].split(";") if x}
        row = {"question": c["question"], "expected_intent": c["expected_intent"], "intent": q.intent,
               "intent_ok": float(q.intent == c["expected_intent"]), "interpreted_as": q.search_query[:80]}
        if rel:
            for name, text in (("raw", c["question"]), ("normalised", q.search_query)):
                srcs = [e.source for e in r.search(text, top_k=config.TOP_K, mode="improved")]
                row[f"hit@5_{name}"] = float(bool(rel & set(srcs)))
                row[f"top1_{name}"] = float(bool(srcs) and srcs[0] in rel)
        rows.append(row)
    out = pd.DataFrame(rows)
    summary = {"n_questions": int(len(out)), "intent_accuracy": round(float(out.intent_ok.mean()), 3)}
    for k in ("hit@5_raw", "hit@5_normalised", "top1_raw", "top1_normalised"):
        if k in out:
            summary[k] = round(float(out[k].dropna().mean()), 3)
    return out, summary


def stress_test(llm: LLM, cases: pd.DataFrame, verbose: bool = True, classifier: str = "rf") -> pd.DataFrame:
    """Inject WRONG ML labels and measure how often the system flags the problem."""
    wrong_for = {"Normal": ["IR_014", "OR_014", "B_014"], "Inner race": ["OR_014", "B_014", "Normal"],
                 "Outer race": ["IR_014", "B_014", "Normal"], "Ball": ["IR_014", "OR_014", "Normal"]}
    rows = []
    agents = {p: DiagnosisAgent(llm=llm, use_ml=True, retrieval="improved", conflict_policy=p,
                                 classifier=classifier)
              for p in ("trust_ml", "abstain")}
    for (_, r), (policy, agent) in product(cases[cases["cwru_file"] != ""].iterrows(), agents.items()):
        x, rpm = case_signal(r)
        for wrong in [None] + wrong_for[r["fault_type"]]:
            d = agent.run(r["question"], signal=x, rpm=rpm, override_label=wrong)
            ff = d.fault_frequencies
            rows.append({
                "policy": policy, "injected_error": wrong is not None,
                "case_id": r["case_id"], "truth": r["fault_type"],
                "injected": cwru.fault_type_of(wrong) if wrong else "",
                "abstained": float(d.fault_type == "Unknown"),
                "final_fault": d.fault_type, "correct": float(d.fault_type == r["fault_type"]),
                # a confidently WRONG final answer (abstaining with "Unknown" is not counted as an error)
                "error_propagated": float(d.fault_type not in (r["fault_type"], "Unknown")),
                "physics_status": ff.consistency if ff else "not_run",
                "flagged": float(bool(ff and ff.consistency in ("conflict", "inconclusive")) or
                                 bool(d.ml_result and d.ml_result.low_confidence)),
                "conflict_flag": float(bool(ff and ff.consistency == "conflict")),
                "citation_coverage": d.citation_coverage, "faithfulness": faithfulness(d),
            })
    df = pd.DataFrame(rows)
    if verbose and len(df):
        w = df[df.injected_error & (df.policy == "abstain")]
        print(f"  stress test: {len(w)} injected errors, flagged {w.flagged.mean():.0%}, "
              f"hard conflict {w.conflict_flag.mean():.0%}")
    return df


# ------------------------------------------------------------------ reporting
METRICS = ["fault_correct", "key_fact_recall", "citation_coverage", "faithfulness", "context_precision",
           "context_recall", "insufficient_ok", "grounding", "llm_answer_verified"]


def summarise(abl: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in METRICS + [c for c in abl.columns if c.startswith("judge_")] if c in abl.columns]
    order = [f"{'ML' if m else 'noML'}+{r}" for m, r in CONFIGS]
    return abl.groupby("config")[cols].mean().reindex(order).round(3)


def plot_ablation(summary: pd.DataFrame, path: Path) -> None:
    metrics = ["fault_correct", "key_fact_recall", "citation_coverage", "faithfulness"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.13
    for i, cfg in enumerate(summary.index):
        vals = summary.loc[cfg, metrics].astype(float).fillna(0).values
        ax.bar(np.arange(len(metrics)) + (i - 2.5) * w, vals, w, label=cfg)
    ax.set_xticks(range(len(metrics)), [m.replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.3)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_title("2 x 3 ablation: (no ML / ML) x (no retrieval / basic RAG / improved RAG)")
    ax.legend(fontsize=7, ncol=6, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stress(st_df: pd.DataFrame, path: Path) -> None:
    w = st_df[st_df.injected_error & (st_df.policy == "abstain")]
    g = w.groupby("truth")[["flagged", "conflict_flag", "error_propagated"]].mean()
    ax = g.plot.bar(figsize=(7, 4), rot=0, color=["#2f6f9f", "#c0392b", "#7f8c8d"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Wrong-ML stress test: injected wrong labels by true fault type")
    ax.set_ylabel("Rate")
    ax.legend(["flagged for review", "physics conflict", "wrong label propagated"], fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_all(provider: str | None = None, model: str | None = None, judge: bool = False, classifier: str = "rf",
            verbose: bool = True) -> dict:
    llm = LLM(provider, model)
    cases = load_cases()
    if not clf_module.CROSSLOAD_MODEL_PATH.exists():
        raise FileNotFoundError("Run python scripts/train_ml.py first")
    clf_module.ACTIVE_MODEL_PATH = clf_module.CROSSLOAD_MODEL_PATH  # 3 HP test recordings unseen by the model
    try:
        if verbose:
            print(f"Evaluating {len(cases)} test cases with LLM = {llm.describe()}")
        retr = retrieval_benchmark(cases)
        casual_df, casual = casual_benchmark()
        abl = run_ablation(llm, cases, judge=judge, verbose=verbose, classifier=classifier)
        st_df = stress_test(llm, cases, verbose=verbose, classifier=classifier)
    finally:
        clf_module.ACTIVE_MODEL_PATH = clf_module.MODEL_PATH

    tag = llm.provider if llm.is_offline else f"{llm.provider}_{llm.model.replace(':', '-').replace('/', '-')}"
    if classifier != "rf":
        tag += f"_{classifier}"
    out_dir = config.REPORTS_DIR / "eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarise(abl)
    abl.to_csv(out_dir / "ablation_per_case.csv", index=False)
    summary.to_csv(out_dir / "ablation_summary.csv")
    retr.to_csv(out_dir / "retrieval_benchmark.csv", index=False)
    casual_df.to_csv(out_dir / "casual_questions_results.csv", index=False)
    st_df.to_csv(out_dir / "stress_test.csv", index=False)
    plot_ablation(summary, config.FIGURES_DIR / "ablation.png")
    if len(st_df):
        plot_stress(st_df, config.FIGURES_DIR / "stress_test.png")

    c0c1 = summary.loc[["noML+none", "noML+basic"]].rename(index={"noML+none": "C0 (LLM only)",
                                                                  "noML+basic": "C1 (basic RAG)"})
    lines = [f"# Copilot evaluation (auto-generated by `python scripts/run_eval.py`)", "",
             f"LLM: `{llm.describe()}` · classifier: `{classifier}` · test cases: {len(cases)} (`eval/test_cases.csv`) · classifier: cross-load "
             f"model (trained 0-2 HP, test recordings at 3 HP are unseen) · embedder: "
             f"`{load_retriever().meta['embed_model']}`", "",
             "## Retrieval benchmark (knowledge questions)", "", retr.round(3).to_markdown(index=False), "",
             "## Informal questions (typos, short forms, everyday words)", "",
             f"{casual['n_questions']} chat-style questions (`eval/casual_questions.csv`). Intent recognised correctly: "
             f"{casual['intent_accuracy']:.0%}.", "",
             "| Retrieval on informal questions | Without language layer | With language layer |", "|---|---|---|",
             f"| Relevant document in top 5 | {casual.get('hit@5_raw', float('nan')):.0%} | "
             f"{casual.get('hit@5_normalised', float('nan')):.0%} |",
             f"| Relevant document ranked first | {casual.get('top1_raw', float('nan')):.0%} | "
             f"{casual.get('top1_normalised', float('nan')):.0%} |", "",
             "## Preliminary result: C0 (LLM-only) vs C1 (basic RAG)", "", c0c1.to_markdown(), "",
             "## 2 x 3 ablation (use of ML output x retrieval)", "", summary.to_markdown(), "",
             "## Wrong-ML stress test", "",
             f"{int((st_df.injected_error & (st_df.policy == 'abstain')).sum())} wrong labels injected into the agent "
             "(classifier output replaced by a confident wrong class), each run under two conflict policies.", ""]
    if len(st_df):
        w = st_df[st_df.injected_error & (st_df.policy == "abstain")]
        g = w.groupby("truth")[["flagged", "conflict_flag", "error_propagated"]].mean().round(3)
        pol = st_df.groupby(["policy", "injected_error"])[["correct", "error_propagated", "abstained", "flagged"]].mean()
        pol = pol.round(3).reset_index().replace({"injected_error": {True: "wrong ML label", False: "true ML label"}})
        lines += ["Detection of injected errors (by true fault type):", "", g.to_markdown(), "",
                  f"Overall: {w.flagged.mean():.1%} of wrong predictions were flagged for review "
                  f"(physics conflict {w.conflict_flag.mean():.1%}).", "",
                  "Conflict policy trade-off (`trust_ml` = always report the ML label; `abstain` = report "
                  "'Unknown - analyst verification required' when physics contradicts the ML label):", "",
                  pol.to_markdown(index=False), "",
                  "`error_propagated` = share of runs whose final answer is a confidently wrong fault type; "
                  "`abstained` = share answered 'Unknown - verify'."]
    lines += ["", "Metric definitions: see `intellimaintain/evaluation/evaluate.py`. Figures: "
              "`reports/figures/ablation.png`, `reports/figures/stress_test.png`."]
    md = "\n".join(lines) + "\n"
    (out_dir / "eval_results.md").write_text(md)
    (config.REPORTS_DIR / "eval_results.md").write_text(md)
    if verbose:
        print("\n" + md)
    return {"summary": json.loads(summary.to_json()), "retrieval": json.loads(retr.to_json()), "casual": casual,
            "out_dir": str(out_dir)}
