"""Recompute website chart values from raw ECCV benchmark JSON and compare
against src/data/paper-figures.ts (values recovered from the paper's PDFs).

Metric definitions replicate OxyGen/experiments/analysis/aggregate_metrics.py:
  - warmup frames (is_warmup) excluded
  - action_frequency_hz = 10 * n_frames / sum(frame_ms) * 1000
  - language_throughput_tps = mean over frames of total_tokens / frame_ms * 1000
Series selection replicates plot_tradeoff.py / compute_speedup_ablation.py:
  num_denoise_steps=10, CB uses uniform_arrivals(rate=1); ablation "ours" is
  k=5, "no_batching" is shared_kv, upper bound is shared_kv at N=5.

Writes scripts/data/raw_verification.json. Read-only on everything else.
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEBSITE = HERE.parents[1]
WS = WEBSITE.parent
RAW = WS / "results" / "eccv_submission_raw" / "results_eccv"
TS = WEBSITE / "src" / "data" / "paper-figures.ts"
OUT = HERE / "raw_verification.json"

H = 10
S = 10
NS = [5, 10, 15, 20, 30]
NS_K10 = [10, 20, 30]
CONFIGS = ["libero", "droid", "aloha"]
TOL_HZ = 0.15  # figure values are given to 1 decimal
TOL_TPS = 0.15


def metrics(path: Path):
    d = json.load(open(path))
    fr = [f for f in d["frames"] if not f.get("is_warmup", False)]
    tot = sum(f["frame_ms"] for f in fr)
    hz = H * len(fr) / tot * 1000
    tps = [f["total_tokens_this_frame"] / f["frame_ms"] * 1000 for f in fr if f["frame_ms"] > 0]
    return hz, sum(tps) / len(tps), d


def raw_file(setting, cfg, n, k=None):
    base = RAW / setting / f"pi05_o2_{cfg}"
    if k is None:
        return base / f"denoise{S}_decode{n}.json"
    return base / f"denoise{S}_decode{n}_step{k}_uniform_1.json"


def parse_ts():
    """Pull figure values out of paper-figures.ts (simple, format-specific)."""
    src = TS.read_text()
    trade = {}
    head = src.split("export const ablationN")[0]
    cur_cfg = cur_dev = cur_ser = None
    for line in head.splitlines():
        s = line.strip()
        m = re.match(r"^(libero|droid|aloha): \{", s)
        if m:
            cur_cfg = m.group(1); trade[cur_cfg] = {}; continue
        m = re.match(r"^(rtx4090|thor): \{", s)
        if m and cur_cfg:
            cur_dev = m.group(1); trade[cur_cfg][cur_dev] = {}; continue
        m = re.match(r"^(baseline|parallel|k1|k5|k10): \[", s)
        if m and cur_dev:
            cur_ser = m.group(1); trade[cur_cfg][cur_dev][cur_ser] = []; continue
        m = re.match(r"^\[(\d+), ([\d.]+), ([\d.]+)\],?$", s)
        if m and cur_ser:
            trade[cur_cfg][cur_dev][cur_ser].append([int(m.group(1)), float(m.group(2)), float(m.group(3))])
    abl = {}
    tail = src.split("export const ablation:")[1]
    for dev in ["rtx4090", "thor"]:
        blk = tail.split(f"{dev}: {{")[1].split("}")[0]
        abl[dev] = {}
        for key in ["baseline", "parallel", "no_batching", "ours"]:
            arr = re.search(rf"{key}: \[([^\]]*)\]", blk).group(1)
            abl[dev][key] = [float(x) for x in arr.split(",")]
        abl[dev]["upperBound"] = float(re.search(r"upperBound: ([\d.]+)", blk).group(1))
    return trade, abl


def reextract_parallel_bbox_center():
    """Re-read parallel (triangle) markers from the paper PDFs using the bbox
    centre. matplotlib's '^' marker path is (0,1),(-1,-1),(1,-1), i.e. anchored
    at its bbox centre; extract_paper_figures.py instead used 2/3 height, which
    biases tok/s low. Needs pymupdf (e.g. `uv run --no-project --with pymupdf`)."""
    try:
        import pymupdf
        import extract_paper_figures as ex
    except ImportError:
        return None
    root = WS / "NeurIPS-2026---OxyGen" / "figures" / "experiments"
    res = {}
    for cfg in CONFIGS:
        page = pymupdf.open(root / f"tradeoff_pi05_o2_{cfg}.pdf")[0]
        res[cfg] = {}
        for dev, (xp, yp, inside) in zip(["rtx4090", "thor"], ex.axis_maps(page)):
            fx, fy = ex.linear(xp), ex.linear(yp)
            pts = []
            for d in page.get_drawings():
                if ex.COLORS.get(ex.rgb(d.get("fill"))) != "parallel" or ex.rgb(d.get("color")) != (1.0, 1.0, 1.0):
                    continue
                r = d["rect"]
                cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
                if not inside(cx) or r.width > 12:
                    continue
                pts.append((fx(cx), fy(cy)))
            pts.sort(key=lambda q: -q[0])
            res[cfg][dev] = [[n, round(hz, 2), round(t, 2)] for n, (hz, t) in zip(NS, pts)]
    return res


def main():
    fig_trade, fig_abl = parse_ts()
    reext = reextract_parallel_bbox_center()
    rel = lambda p: str(p.relative_to(WS))
    out = {
        "_description": (
            "Website chart values recomputed from raw benchmark JSON using the "
            "exact aggregation of OxyGen/experiments/analysis/aggregate_metrics.py "
            "(warmup excluded; Hz = 10*frames/sum(frame_ms); tok/s = mean per-frame "
            "tokens/frame_ms). Same structure as src/data/paper-figures.ts. "
            "Values rounded to 2 decimals. Figure values (1 decimal) compared with "
            f"tolerance {TOL_HZ} Hz / {TOL_TPS} tok/s."
        ),
        "generated_by": rel(Path(__file__)),
        "raw_root": rel(RAW),
        "raw_root_note": (
            "Identical byte-for-byte to results_eccv_merged.zip contents. "
            "metadata.gpu = NVIDIA GeForce RTX 4090, JAX 0.5.3."
        ),
        "num_denoise_steps": S,
        "tradeoff": {},
        "ablationN": NS,
        "ablation": {},
        "comparison": {},
    }
    series_spec = {
        "baseline": ("baseline", None, NS),
        "parallel": ("parallel_mps", None, NS),
        "k1": ("continuous_batching", 1, NS),
        "k5": ("continuous_batching", 5, NS),
        "k10": ("continuous_batching", 10, NS_K10),
    }
    gpus = set()
    for cfg in CONFIGS:
        out["tradeoff"][cfg] = {}
        # RTX 4090
        dev = "rtx4090"
        panel, sources, diffs = {}, {}, {}
        for ser, (setting, k, ns) in series_spec.items():
            pts, srcs = [], []
            for n in ns:
                p = raw_file(setting, cfg, n, k)
                hz, tps, d = metrics(p)
                gpus.add(d.get("metadata", {}).get("gpu"))
                pts.append([n, round(hz, 2), round(tps, 2)])
                srcs.append(rel(p))
            panel[ser] = pts
            sources[ser] = srcs
            fpts = {q[0]: q for q in fig_trade[cfg][dev][ser]}
            dh = [abs(q[1] - fpts[q[0]][1]) for q in pts]
            dt = [abs(q[2] - fpts[q[0]][2]) for q in pts]
            dhr = [abs(q[1] - fpts[q[0]][1]) / fpts[q[0]][1] for q in pts]
            diffs[ser] = {"max_abs_hz": round(max(dh), 3), "max_abs_tps": round(max(dt), 3),
                          "max_rel_hz": round(max(dhr), 4)}
        ok = all(v["max_abs_hz"] <= TOL_HZ and v["max_abs_tps"] <= TOL_TPS for v in diffs.values())
        out["tradeoff"][cfg][dev] = {"available": True, "matches_figure": ok, "series": panel,
                                     "sources": sources, "diff_vs_figure": diffs}
        thor = {
            "available": False, "matches_figure": None,
            "note": "No Jetson AGX Thor raw results found (expected experiments/results_thor_pytorch); figure values retained.",
        }
        if reext:
            r4090 = reext[cfg]["rtx4090"]
            chk = max(abs(a[2] - b[2]) for a, b in zip(r4090, panel["parallel"]))
            thor["parallel_pdf_reextracted_bbox_center"] = reext[cfg]["thor"]
            thor["parallel_reextraction_note"] = (
                "Parallel markers re-read from the PDF with bbox-centre anchoring; "
                f"the same method on the RTX 4090 panel matches raw tok/s within {chk:.2f}. "
                "Use these in place of the website's Thor parallel tok/s.")
        out["tradeoff"][cfg]["thor"] = thor

    # Ablation (LIBERO, k=5)
    cfg = "libero"
    abl = {}
    srcs = {}
    for key, (setting, k) in {"baseline": ("baseline", None), "parallel": ("parallel_mps", None),
                              "no_batching": ("shared_kv", None), "ours": ("continuous_batching", 5)}.items():
        vals, ss = [], []
        for n in NS:
            p = raw_file(setting, cfg, n, k)
            vals.append(round(metrics(p)[0], 2)); ss.append(rel(p))
        abl[key] = vals; srcs[key] = ss
    ub_path = raw_file("shared_kv", cfg, 5)
    abl["upperBound"] = round(metrics(ub_path)[0], 2)
    srcs["upperBound"] = rel(ub_path)
    f = fig_abl["rtx4090"]
    adiff = {k: round(max(abs(a - b) for a, b in zip(abl[k], f[k])), 3)
             for k in ["baseline", "parallel", "no_batching", "ours"]}
    adiff["upperBound"] = round(abs(abl["upperBound"] - f["upperBound"]), 3)
    out["ablation"]["rtx4090"] = {
        "available": True, "matches_figure": all(v <= TOL_HZ for v in adiff.values()),
        **abl, "sources": srcs, "max_abs_diff_hz_vs_figure": adiff,
        "upperBound_definition": "shared_kv action Hz at N=5 (compute_speedup_ablation.py shared_kv_5_action_freq_hz)",
    }
    out["ablation"]["thor"] = {"available": False, "matches_figure": None,
                               "note": "No Thor raw results found; figure values retained."}

    out["gpus_in_metadata"] = sorted(g for g in gpus if g)
    summary = {}
    for cfg in CONFIGS:
        summary[f"tradeoff.{cfg}.rtx4090"] = out["tradeoff"][cfg]["rtx4090"]["matches_figure"]
        summary[f"tradeoff.{cfg}.thor"] = "raw not found"
    summary["ablation.rtx4090"] = out["ablation"]["rtx4090"]["matches_figure"]
    summary["ablation.thor"] = "raw not found"
    out["comparison"] = summary
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    for cfg in CONFIGS:
        print(cfg, json.dumps(out["tradeoff"][cfg]["rtx4090"]["diff_vs_figure"]))
    print("ablation", adiff, abl)


if __name__ == "__main__":
    main()
