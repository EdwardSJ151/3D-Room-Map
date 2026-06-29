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

    result = {"num_records": {"mean": mean(num_records_list), "std": std(num_records_list)}}
    for k in timing_keys:
        result[k] = {"mean": mean(raw[k]), "std": std(raw[k])}
    for k in per_record:
        result[f"{k}_per_record"] = {"mean": mean(per_record[k]), "std": std(per_record[k])}
    return result


def compute_metric2(scenes: list) -> dict:
    """Query performance timing."""
    retrieval_ms_all = []
    gen_ms_all = []
    total_ms_all = []
    num_records_all = []

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
            ret = r.get("retrieval_time_ms")
            gen = r.get("response_generation_time_ms")
            tot = r.get("total_query_to_answer_time_ms")
            if ret is not None:
                retrieval_ms_all.append(ret)
                num_records_all.append(n)
            if gen is not None:
                gen_ms_all.append(gen)
            if tot is not None:
                total_ms_all.append(tot)

    retrieval_per_100 = [
        (ret / n) * 100 for ret, n in zip(retrieval_ms_all, num_records_all) if n > 0
    ]

    return {
        "retrieval_time_ms": {"mean": mean(retrieval_ms_all), "std": std(retrieval_ms_all)},
        "retrieval_time_ms_per_100_records": {"mean": mean(retrieval_per_100), "std": std(retrieval_per_100)},
        "response_generation_time_ms": {"mean": mean(gen_ms_all), "std": std(gen_ms_all)},
        "total_query_to_answer_time_ms": {"mean": mean(total_ms_all), "std": std(total_ms_all)},
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


def compute_metric4(scenes: list) -> dict:
    """Grounded QA success rates with bootstrap CI."""
    overall_scene_rates: list[float] = []
    cat_scene_rates: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    failure_counts: dict[str, int] = {r: 0 for r in FAILURE_REASONS}
    total_failures = 0

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
        cat_metrics[cat] = {
            "rate": mean(rates),
            "ci_95": bootstrap_ci(rates),
        }

    failure_dist = {}
    for reason in FAILURE_REASONS:
        cnt = failure_counts[reason]
        failure_dist[reason] = {
            "count": cnt,
            "pct": (cnt / total_failures * 100) if total_failures else 0.0,
        }

    return {
        "overall": {"rate": overall_rate, "ci_95": list(overall_ci)},
        "per_category": {k: {"rate": v["rate"], "ci_95": list(v["ci_95"])} for k, v in cat_metrics.items()},
        "failure_distribution": failure_dist,
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
             "| Metric | Mean | Std |",
             "|--------|------|-----|"]
    for name, vals in rows:
        m = f"{vals['mean']:.3f}" if vals and vals.get("mean") is not None else "—"
        s = f"{vals['std']:.3f}"  if vals and vals.get("std")  is not None else "—"
        lines.append(f"| {name} | {m} | {s} |")
    return "\n".join(lines)


def format_table2(m1: dict, m3: dict, m4: dict) -> str:
    def fmt(rate, ci):
        if rate is None:
            return "—", "—"
        r_str = f"{rate*100:.1f}%"
        ci_str = f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]" if ci and None not in ci else "—"
        return r_str, ci_str

    n_mean = m1.get("num_records", {}).get("mean")
    u_per_scene = (
        m3.get("usable_rate", 0) * n_mean
        if m3.get("usable_rate") is not None and n_mean
        else None
    )

    rows_raw = [
        ("Generated records/scene",    (f"{n_mean:.1f}" if n_mean else "—"), "—"),
        ("Usable records/scene (est.)", (f"{u_per_scene:.1f}" if u_per_scene else "—"), "—"),
        ("Usable object-record rate",   *fmt(m3.get("usable_rate"), None)),
        ("Partial object-record rate",  *fmt(m3.get("partial_rate"), None)),
        ("Unusable object-record rate", *fmt(m3.get("unusable_rate"), None)),
        ("Strict grounded QA success",  *fmt(m4["overall"]["rate"], m4["overall"]["ci_95"])),
    ]
    for cat in CATEGORIES:
        cd = m4["per_category"].get(cat, {})
        rows_raw.append((cat.replace("_", " ").title(), *fmt(cd.get("rate"), cd.get("ci_95"))))

    lines = ["## Table 2: Memory and QA Quality\n",
             "| Metric | Score | 95% CI |",
             "|--------|-------|--------|"]
    for name, score, ci in rows_raw:
        lines.append(f"| {name} | {score} | {ci} |")
    return "\n".join(lines)


def format_table3(m4: dict) -> str:
    fd = m4.get("failure_distribution", {})
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
        for r in data.get("results", []):
            if not r.get("tool_was_called"):
                continue
            rows.append({
                "scene_id": sid,
                "question_id": r.get("question_id", ""),
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
    ]
    md_path = RESULTS_DIR / "summary.md"
    md_path.write_text("\n".join(md_parts))
    print(f"summary.md  → {md_path}")

    plot_timings(scenes, RESULTS_DIR)

    print("\n--- Quick view ---")
    overall = m4.get("overall", {})
    rate = overall.get("rate")
    ci   = overall.get("ci_95")
    if rate is not None:
        ci_str = f" (95% CI [{ci[0]*100:.1f}%–{ci[1]*100:.1f}%])" if ci else ""
        print(f"Grounded QA success: {rate*100:.1f}%{ci_str}")


if __name__ == "__main__":
    main()
