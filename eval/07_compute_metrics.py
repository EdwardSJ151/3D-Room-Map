"""Compute all 4 metrics and write summary.json + summary.md.

Usage:
    python eval/07_compute_metrics.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.config import SCENES_JSON
from eval.lib.paths import RESULTS_DIR, scene_result_file

CATEGORIES = [
    "positive_object_existence",
    "negative_object_existence",
    "attribute_grounding",
    "category_retrieval",
    "affordance_retrieval",
    "spatial_or_local_relation",
]

FAILURE_REASONS = [
    "missing_memory_record",
    "semantic_record_error",
    "retrieval_error",
    "unsupported_generation",
    "failed_abstention",
    "incomplete_answer",
    "ambiguous_question",
]

N_BOOTSTRAP = 10_000
CI_ALPHA = 0.95


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def std(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2



def bootstrap_ci(scene_rates: list[float], n: int = N_BOOTSTRAP) -> tuple[float, float]:
    if not scene_rates:
        return (0.0, 0.0)
    lo_idx = int((1 - CI_ALPHA) / 2 * n)
    hi_idx = int((1 + CI_ALPHA) / 2 * n)
    samples = sorted(
        sum(random.choices(scene_rates, k=len(scene_rates))) / len(scene_rates)
        for _ in range(n)
    )
    return (samples[lo_idx], samples[hi_idx])


def load_scenes() -> list:
    return json.loads(SCENES_JSON.read_text())


def compute_metric1(scenes: list) -> dict:
    """Mapping performance timing."""
    timing_keys = [
        "cutr_inference_time_s",
        "crop_generation_time_s",
        "qwen_labeling_time_s",
        "embedding_indexing_time_s",
        "total_capture_to_memory_time_s",
    ]
    raw: dict[str, list] = {k: [] for k in timing_keys}
    per_record: dict[str, list] = {k: [] for k in timing_keys if k != "cutr_inference_time_s"}
    num_records_list = []

    for scene in scenes:
        sid = scene["scene_id"]
        p = scene_result_file(sid, "mapping.json")
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        n = m.get("num_generated_records", 0)
        num_records_list.append(n)

        for k in timing_keys:
            v = m.get(k)
            if v is not None:
                raw[k].append(v)
                if k != "cutr_inference_time_s" and n > 0:
                    per_record[k].append(v / n)

    def stat(vals):
        return {"mean": mean(vals), "median": median(vals), "std": std(vals)}

    result = {"num_records": stat(num_records_list)}
    for k in timing_keys:
        result[k] = stat(raw[k])
    for k in per_record:
        result[f"{k}_per_record"] = stat(per_record[k])
    return result


def compute_metric2(scenes: list) -> dict:
    """Query performance timing."""
    rows = collect_timing_rows(scenes)

    retrieval_ms_all = [r["retrieval_ms"]  for r in rows if r["retrieval_ms"]  is not None]
    gen_ms_all       = [r["generation_ms"] for r in rows if r["generation_ms"] is not None]
    total_ms_all     = [r["total_ms"]      for r in rows if r["total_ms"]      is not None]
    num_records_all  = [r["num_records"]   for r in rows if r["retrieval_ms"]  is not None]

    retrieval_per_100 = [
        (ret / n) * 100 for ret, n in zip(retrieval_ms_all, num_records_all) if n > 0
    ]

    def stat(vals):
        return {"mean": mean(vals), "median": median(vals), "std": std(vals)}

    return {
        "retrieval_time_ms":                 stat(retrieval_ms_all),
        "retrieval_time_ms_per_100_records": stat(retrieval_per_100),
        "response_generation_time_ms":       stat(gen_ms_all),
        "total_query_to_answer_time_ms":     stat(total_ms_all),
    }


def compute_metric3(scenes: list) -> dict:
    """Usability rates from sampled human review."""
    total_u = total_p = total_x = total = 0
    missing_scenes = []

    for scene in scenes:
        sid = scene["scene_id"]
        p = scene_result_file(sid, "usability.json")
        if not p.exists():
            missing_scenes.append(sid)
            continue
        data = json.loads(p.read_text())
        for r in data.get("reviews", []):
            j = r.get("judgment")
            if j == "usable":   total_u += 1
            elif j == "partial": total_p += 1
            elif j == "unusable": total_x += 1
            total += 1

    if missing_scenes:
        print(f"[metric3] WARNING: no usability.json for: {missing_scenes}", file=sys.stderr)

    return {
        "note": "Sampled human audit — not all scenes/records may be included",
        "reviewed_records": total,
        "usable_rate": total_u / total if total else None,
        "partial_rate": total_p / total if total else None,
        "unusable_rate": total_x / total if total else None,
    }


_GROUNDING_APPLICABLE_CATEGORIES = {
    "positive_object_existence",
    "attribute_grounding",
    "category_retrieval",
    "affordance_retrieval",
    "spatial_or_local_relation",
}


def _grounding_rate(counts: dict) -> dict:
    denom = counts["true"] + counts["false"]
    return {
        "true_count": counts["true"],
        "false_count": counts["false"],
        "not_applicable_count": counts["not_applicable"],
        "denominator": denom,
        "rate": counts["true"] / denom if denom else None,
    }


def compute_metric4(scenes: list) -> dict:
    """Grounded QA quality — three independent submetrics."""
    # Part 1: Strict QA Success
    overall_scene_rates: list[float] = []
    cat_scene_rates: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    failure_counts: dict[str, int] = {r: 0 for r in FAILURE_REASONS}
    total_failures = 0

    # Part 2 & 3: grounding metrics per category
    er_total = {"true": 0, "false": 0, "not_applicable": 0}
    bi_total = {"true": 0, "false": 0, "not_applicable": 0}
    er_by_cat: dict[str, dict] = {c: {"true": 0, "false": 0, "not_applicable": 0} for c in _GROUNDING_APPLICABLE_CATEGORIES}
    bi_by_cat: dict[str, dict] = {c: {"true": 0, "false": 0, "not_applicable": 0} for c in _GROUNDING_APPLICABLE_CATEGORIES}

    for scene in scenes:
        sid = scene["scene_id"]
        j_path = scene_result_file(sid, "judgments.json")
        q_path = scene_result_file(sid, "questions.json")
        if not j_path.exists() or not q_path.exists():
            continue

        judgments = json.loads(j_path.read_text()).get("judgments", [])
        questions  = json.loads(q_path.read_text()).get("questions", [])
        q_map = {q["question_id"]: q for q in questions}

        scene_successes = 0
        scene_total = len(judgments)
        cat_results: dict[str, list[int]] = {c: [] for c in CATEGORIES}

        for j in judgments:
            success = 1 if j.get("judgment") == "success" else 0
            scene_successes += success

            if not success:
                reason = j.get("failure_reason") or "ambiguous_question"
                if reason in failure_counts:
                    failure_counts[reason] += 1
                total_failures += 1

            qid = j.get("question_id")
            cat = q_map.get(qid, {}).get("category", "")
            if cat in cat_results:
                cat_results[cat].append(success)

            # Grounding metrics — applicable categories only
            if cat in _GROUNDING_APPLICABLE_CATEGORIES:
                for field, totals, by_cat in [
                    ("evidence_retrieval_at_5", er_total, er_by_cat),
                    ("best_idx_accuracy", bi_total, bi_by_cat),
                ]:
                    val = str(j.get(field, "not_applicable")).lower()
                    bucket = val if val in ("true", "false") else "not_applicable"
                    totals[bucket] += 1
                    by_cat[cat][bucket] += 1

        if scene_total > 0:
            overall_scene_rates.append(scene_successes / scene_total)

        for cat in CATEGORIES:
            vals = cat_results[cat]
            if vals:
                cat_scene_rates[cat].append(sum(vals) / len(vals))

    overall_rate = mean(overall_scene_rates)
    overall_ci   = bootstrap_ci(overall_scene_rates)

    cat_metrics = {}
    for cat in CATEGORIES:
        rates = cat_scene_rates[cat]
        cat_metrics[cat] = {"rate": mean(rates), "ci_95": bootstrap_ci(rates)}

    failure_dist = {}
    for reason in FAILURE_REASONS:
        cnt = failure_counts[reason]
        failure_dist[reason] = {
            "count": cnt,
            "pct": (cnt / total_failures * 100) if total_failures else 0.0,
        }

    return {
        "part1_strict_qa": {
            "overall": {"rate": overall_rate, "ci_95": list(overall_ci)},
            "per_category": {k: {"rate": v["rate"], "ci_95": list(v["ci_95"])} for k, v in cat_metrics.items()},
            "failure_distribution": failure_dist,
        },
        "part2_evidence_retrieval_at_5": {
            "overall": _grounding_rate(er_total),
            "per_category": {cat: _grounding_rate(er_by_cat[cat]) for cat in _GROUNDING_APPLICABLE_CATEGORIES},
        },
        "part3_best_idx_accuracy": {
            "overall": _grounding_rate(bi_total),
            "per_category": {cat: _grounding_rate(bi_by_cat[cat]) for cat in _GROUNDING_APPLICABLE_CATEGORIES},
        },
    }


def format_table1(m1: dict, m2: dict) -> str:
    rows = [
        ("CuTR inference time (s)",                   m1.get("cutr_inference_time_s")),
        ("Crop generation time/record (s)",            m1.get("crop_generation_time_s_per_record")),
        ("Qwen labeling time/record (s)",              m1.get("qwen_labeling_time_s_per_record")),
        ("Embedding/indexing time/record (s)",         m1.get("embedding_indexing_time_s_per_record")),
        ("Total capture-to-memory time (s)",           m1.get("total_capture_to_memory_time_s")),
        ("Total capture-to-memory time/record (s)",    m1.get("total_capture_to_memory_time_s_per_record")),
        ("Retrieval time/query (ms)",                  m2.get("retrieval_time_ms")),
        ("Retrieval time per 100 records (ms)",        m2.get("retrieval_time_ms_per_100_records")),
        ("Response generation time/query (ms)",        m2.get("response_generation_time_ms")),
        ("Total query-to-answer time/query (ms)",      m2.get("total_query_to_answer_time_ms")),
    ]
    lines = ["## Table 1: System Performance\n",
             "| Metric | Mean | Median | Std |",
             "|--------|------|--------|-----|"]
    for name, vals in rows:
        m  = f"{vals['mean']:.3f}"   if vals and vals.get("mean")   is not None else "—"
        md = f"{vals['median']:.3f}" if vals and vals.get("median") is not None else "—"
        s  = f"{vals['std']:.3f}"    if vals and vals.get("std")    is not None else "—"
        lines.append(f"| {name} | {m} | {md} | {s} |")
    return "\n".join(lines)


def format_table2(m1: dict, m3: dict, m4: dict) -> str:
    def fmt(rate, ci=None):
        if rate is None:
            return "—", "—"
        r_str = f"{rate*100:.1f}%"
        ci_str = f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]" if ci and None not in ci else "—"
        return r_str, ci_str

    p1 = m4.get("part1_strict_qa", {})
    n_mean = m1.get("num_records", {}).get("mean")
    u_per_scene = (
        m3.get("usable_rate", 0) * n_mean
        if m3.get("usable_rate") is not None and n_mean
        else None
    )

    rows_raw = [
        ("Generated records/scene",     (f"{n_mean:.1f}" if n_mean else "—"), "—"),
        ("Usable records/scene (est.)",  (f"{u_per_scene:.1f}" if u_per_scene else "—"), "—"),
        ("Usable object-record rate",    *fmt(m3.get("usable_rate"))),
        ("Partial object-record rate",   *fmt(m3.get("partial_rate"))),
        ("Unusable object-record rate",  *fmt(m3.get("unusable_rate"))),
        ("Strict grounded QA success",   *fmt(p1.get("overall", {}).get("rate"), p1.get("overall", {}).get("ci_95"))),
    ]
    for cat in CATEGORIES:
        cd = p1.get("per_category", {}).get(cat, {})
        rows_raw.append((cat.replace("_", " ").title(), *fmt(cd.get("rate"), cd.get("ci_95"))))

    lines = ["## Table 2: Memory and QA Quality\n",
             "| Metric | Score | 95% CI |",
             "|--------|-------|--------|"]
    for name, score, ci in rows_raw:
        lines.append(f"| {name} | {score} | {ci} |")
    return "\n".join(lines)


def format_table4(m4: dict) -> str:
    def fmt_grounding(g: dict) -> str:
        rate = g.get("rate")
        denom = g.get("denominator", 0)
        if rate is None or denom == 0:
            return "—"
        return f"{rate*100:.1f}% (n={denom})"

    p2 = m4.get("part2_evidence_retrieval_at_5", {})
    p3 = m4.get("part3_best_idx_accuracy", {})

    lines = ["## Table 4: Grounding Metrics\n",
             "| Category | Evidence Retrieval@5 | Best-idx Accuracy |",
             "|----------|---------------------|-------------------|"]

    lines.append(f"| **Overall** | {fmt_grounding(p2.get('overall', {}))} | {fmt_grounding(p3.get('overall', {}))} |")
    for cat in _GROUNDING_APPLICABLE_CATEGORIES:
        label = cat.replace("_", " ").title()
        er = fmt_grounding(p2.get("per_category", {}).get(cat, {}))
        bi = fmt_grounding(p3.get("per_category", {}).get(cat, {}))
        lines.append(f"| {label} | {er} | {bi} |")

    return "\n".join(lines)


def format_table3(m4: dict) -> str:
    fd = m4.get("part1_strict_qa", {}).get("failure_distribution", {})
    total_f = sum(v["count"] for v in fd.values())
    lines = [f"## Table 3: Failure Analysis (total failed={total_f})\n",
             "| Failure Reason | Count | % of Failed |",
             "|---------------|-------|-------------|"]
    for reason in FAILURE_REASONS:
        v = fd.get(reason, {"count": 0, "pct": 0.0})
        lines.append(f"| {reason.replace('_', ' ')} | {v['count']} | {v['pct']:.1f}% |")
    return "\n".join(lines)


def collect_timing_rows(scenes: list) -> list[dict]:
    """Return one dict per QA result that had a tool call, with timing fields + metadata."""
    rows = []
    for scene in scenes:
        sid = scene["scene_id"]
        p = scene_result_file(sid, "qa_results.json")
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        n = data.get("num_records_in_scene", 0)
        for r in data.get("results", []):
            if not r.get("tool_was_called"):
                continue
            rows.append({
                "scene_id": sid,
                "question_id": r.get("question_id", ""),
                "num_records": n,
                "retrieval_ms": r.get("retrieval_time_ms"),
                "generation_ms": r.get("response_generation_time_ms"),
                "total_ms": r.get("total_query_to_answer_time_ms"),
            })
    return rows


def plot_timings(scenes: list, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[plot] matplotlib not installed, skipping timing plots", file=sys.stderr)
        return

    rows = collect_timing_rows(scenes)
    if not rows:
        print("[plot] no timing data to plot", file=sys.stderr)
        return

    retrieval  = [r["retrieval_ms"]  for r in rows if r["retrieval_ms"]  is not None]
    generation = [r["generation_ms"] for r in rows if r["generation_ms"] is not None]
    total      = [r["total_ms"]      for r in rows if r["total_ms"]      is not None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Query Timing Distribution", fontsize=14)

    # --- Box plot ---
    ax = axes[0]
    labels = ["Retrieval", "Generation", "Total"]
    data   = [retrieval, generation, total]
    bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    colors = ["#4C9BE8", "#E87C4C", "#4CE87C"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Box plot (outliers shown)")
    ax.grid(axis="y", alpha=0.3)

    # --- Scatter: retrieval vs total, coloured by generation ---
    ax2 = axes[1]
    xs = [r["retrieval_ms"]  for r in rows if None not in (r["retrieval_ms"], r["total_ms"], r["generation_ms"])]
    ys = [r["total_ms"]      for r in rows if None not in (r["retrieval_ms"], r["total_ms"], r["generation_ms"])]
    cs = [r["generation_ms"] for r in rows if None not in (r["retrieval_ms"], r["total_ms"], r["generation_ms"])]

    sc = ax2.scatter(xs, ys, c=cs, cmap="plasma", alpha=0.7, s=30)
    fig.colorbar(sc, ax=ax2, label="Generation time (ms)")
    ax2.set_xlabel("Retrieval time (ms)")
    ax2.set_ylabel("Total time (ms)")
    ax2.set_title("Retrieval vs Total (colour = generation)")
    ax2.grid(alpha=0.3)

    # Annotate top-5 outliers by total time
    if ys:
        threshold = sorted(ys, reverse=True)[min(4, len(ys) - 1)]
        for r in rows:
            if None in (r["retrieval_ms"], r["total_ms"], r["generation_ms"]):
                continue
            if r["total_ms"] >= threshold:
                ax2.annotate(
                    r["question_id"].split("_")[-1],
                    (r["retrieval_ms"], r["total_ms"]),
                    fontsize=7, alpha=0.8,
                    xytext=(4, 4), textcoords="offset points",
                )

    plt.tight_layout()
    out_path = out_dir / "timing_plots.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"timing_plots.png → {out_path}")


def write_incorrect(scenes: list, out_dir: Path) -> None:
    incorrect = []
    er_failures = []
    bi_failures = []

    for scene in scenes:
        sid = scene["scene_id"]
        j_path  = scene_result_file(sid, "judgments.json")
        q_path  = scene_result_file(sid, "questions.json")
        qa_path = scene_result_file(sid, "qa_results.json")
        if not j_path.exists() or not q_path.exists() or not qa_path.exists():
            continue

        judgments  = {j["question_id"]: j for j in json.loads(j_path.read_text()).get("judgments", [])}
        questions  = {q["question_id"]: q for q in json.loads(q_path.read_text()).get("questions", [])}
        qa_results = {r["question_id"]: r for r in json.loads(qa_path.read_text()).get("results", [])}

        for qid, j in judgments.items():
            q  = questions.get(qid, {})
            qa = qa_results.get(qid, {})

            base = {
                "scene_id":                    sid,
                "question_id":                 qid,
                "category":                    q.get("category", ""),
                "question":                    q.get("question", ""),
                "expected_answer":             q.get("expected_answer", ""),
                "expected_visible_evidence":   q.get("expected_visible_evidence", ""),
                "retrieved_records":           (qa.get("retrieved_records") or [])[:5],
                "best_idx":                    qa.get("best_idx"),
                "assistant_answer":            qa.get("assistant_answer", ""),
                "tool_was_called":             qa.get("tool_was_called", False),
                "judgment":                    j.get("judgment"),
                "failure_reason":              j.get("failure_reason"),
                "judge_explanation":           j.get("judge_explanation", ""),
                "evidence_retrieval_at_5":     j.get("evidence_retrieval_at_5"),
                "evidence_retrieval_explanation": j.get("evidence_retrieval_explanation", ""),
                "best_idx_accuracy":           j.get("best_idx_accuracy"),
                "best_idx_explanation":        j.get("best_idx_explanation", ""),
            }

            if j.get("judgment") != "success":
                incorrect.append(base)

            if j.get("evidence_retrieval_at_5") == "false":
                er_failures.append(base)

            if j.get("best_idx_accuracy") == "false":
                bi_failures.append(base)

    (out_dir / "incorrect.json").write_text(json.dumps(incorrect, indent=2))
    print(f"incorrect.json → {out_dir / 'incorrect.json'}  ({len(incorrect)} failures)")

    (out_dir / "evidence_retrieval_failures.json").write_text(json.dumps(er_failures, indent=2))
    print(f"evidence_retrieval_failures.json → {out_dir / 'evidence_retrieval_failures.json'}  ({len(er_failures)} failures)")

    (out_dir / "best_idx_failures.json").write_text(json.dumps(bi_failures, indent=2))
    print(f"best_idx_failures.json → {out_dir / 'best_idx_failures.json'}  ({len(bi_failures)} failures)")


def main():
    scenes = load_scenes()
    print(f"Computing metrics for {len(scenes)} scenes...", flush=True)

    m1 = compute_metric1(scenes)
    m2 = compute_metric2(scenes)
    m3 = compute_metric3(scenes)
    m4 = compute_metric4(scenes)

    summary = {"metric1": m1, "metric2": m2, "metric3": m3, "metric4": m4}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"summary.json → {json_path}")

    md_parts = [
        "# Eval Summary\n",
        format_table1(m1, m2),
        "\n",
        format_table2(m1, m3, m4),
        "\n",
        format_table3(m4),
        "\n",
        format_table4(m4),
    ]
    md_path = RESULTS_DIR / "summary.md"
    md_path.write_text("\n".join(md_parts))
    print(f"summary.md  → {md_path}")

    plot_timings(scenes, RESULTS_DIR)
    write_incorrect(scenes, RESULTS_DIR)

    print("\n--- Quick view ---")
    p1 = m4.get("part1_strict_qa", {}).get("overall", {})
    p2 = m4.get("part2_evidence_retrieval_at_5", {}).get("overall", {})
    p3 = m4.get("part3_best_idx_accuracy", {}).get("overall", {})
    rate = p1.get("rate")
    ci   = p1.get("ci_95")
    if rate is not None:
        ci_str = f" (95% CI [{ci[0]*100:.1f}%–{ci[1]*100:.1f}%])" if ci else ""
        print(f"Strict QA success:       {rate*100:.1f}%{ci_str}")
    if p2.get("rate") is not None:
        print(f"Evidence Retrieval@5:    {p2['rate']*100:.1f}%  (n={p2.get('denominator')})")
    if p3.get("rate") is not None:
        print(f"Best-idx Accuracy:       {p3['rate']*100:.1f}%  (n={p3.get('denominator')})")


if __name__ == "__main__":
    main()
