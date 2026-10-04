"""Recover plotted values from the paper's vector PDF figures.

The paper's tradeoff and ablation figures are matplotlib vector PDFs. Marker
centers and bar extents are mapped back to data coordinates through the axis
tick marks, giving the plotted values to within ~0.1 units.
Usage: python extract_paper_figures.py <paper_repo>/figures/experiments out.json
"""
import json
import sys
from pathlib import Path

import pymupdf

COLORS = {
    (0.77, 0.31, 0.32): "baseline",
    (0.87, 0.52, 0.32): "parallel",
    (0.12, 0.29, 0.56): "k1",
    (0.27, 0.45, 0.77): "k5",
    (0.56, 0.7, 0.88): "k10",
    (0.94, 0.78, 0.37): "no_batching",
}
N_VALUES = {"baseline": [5, 10, 15, 20, 30], "parallel": [5, 10, 15, 20, 30],
            "k1": [5, 10, 15, 20, 30], "k5": [5, 10, 15, 20, 30], "k10": [10, 20, 30]}


def rgb(c):
    return tuple(round(v, 2) for v in c) if c else ()


def linear(pairs):
    (p0, v0), (p1, v1) = pairs[0], pairs[-1]
    return lambda p: v0 + (p - p0) * (v1 - v0) / (p1 - p0)


def axis_maps(page):
    """Panels are split at the page midline; returns [(fx, fy, title)] left to right."""
    words = page.get_text("words")
    xt, yt = [], []
    for d in page.get_drawings():
        if rgb(d.get("color")) != (0.0, 0.0, 0.0) or rgb(d.get("fill")) != (0.0, 0.0, 0.0):
            continue
        (_, a, b) = d["items"][0]
        if abs(a.x - b.x) < 0.01 and abs(abs(a.y - b.y) - 3.5) < 0.1:
            xt.append((a.x, max(a.y, b.y)))
        elif abs(a.y - b.y) < 0.01 and abs(abs(a.x - b.x) - 3.5) < 0.1:
            yt.append((min(a.x, b.x), a.y))
    def num(s):
        try:
            return float(s)
        except ValueError:
            return None
    panels = []
    frames = sorted((d["rect"] for d in page.get_drawings()
                     if rgb(d.get("fill")) == (1.0, 1.0, 1.0) and not d.get("color")
                     and d["rect"].width < page.rect.width * 0.6), key=lambda r: r.x0)
    for frame in frames:
        inside = (lambda f: lambda x: f.x0 - 1 <= x <= f.x1 + 1)(frame)
        xp = []
        for x, y in xt:
            if not inside(x):
                continue
            cands = [w for w in words if abs((w[0] + w[2]) / 2 - x) < 6 and 0 < w[1] - y < 8 and num(w[4]) is not None]
            if cands:
                xp.append((x, num(cands[0][4])))
        yp = []
        for x, y in yt:
            if not inside(x + 4):
                continue
            cands = [w for w in words if abs((w[1] + w[3]) / 2 - y) < 4 and 0 < x - w[2] < 8 and num(w[4]) is not None]
            if cands:
                yp.append((y, num(cands[0][4])))
        xp.sort(); yp.sort()
        panels.append((xp, yp, inside))
    return panels


def tradeoff(path):
    page = pymupdf.open(path)[0]
    panels = axis_maps(page)
    out = []
    for xp, yp, inside in panels:
        fx, fy = linear(xp), linear(yp)
        series = {}
        for d in page.get_drawings():
            name = COLORS.get(rgb(d.get("fill")))
            if not name or rgb(d.get("color")) != (1.0, 1.0, 1.0):
                continue
            r = d["rect"]
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if not inside(cx) or r.width > 12:
                continue
            series.setdefault(name, []).append((round(fx(cx), 2), round(fy(cy), 1)))
        panel = {}
        for name, pts in series.items():
            pts.sort(key=lambda p: -p[0])  # N increases as action frequency drops
            panel[name] = [{"N": n, "hz": hz, "tps": tps} for n, (hz, tps) in zip(N_VALUES[name], pts)]
        out.append(panel)
    return out


def ablation(path):
    page = pymupdf.open(path)[0]
    panels = axis_maps(page)
    out = []
    names = {(0.27, 0.45, 0.77): "ours", (0.94, 0.78, 0.37): "no_batching",
             (0.87, 0.52, 0.32): "parallel", (0.77, 0.31, 0.32): "baseline"}
    for xp, yp, inside in panels:
        fx = linear(xp)
        rows = sorted({round(d["rect"].y0, 1) for d in page.get_drawings()
                       if rgb(d.get("fill")) in names and inside(d["rect"].x0 + 1) and d["rect"].y0 < 120}, reverse=True)
        n_for_row = dict(zip(rows, [5, 10, 15, 20, 30]))
        panel = {v: [] for v in names.values()}
        for d in page.get_drawings():
            name = names.get(rgb(d.get("fill")))
            r = d["rect"]
            if not name or not inside(r.x0 + 1) or r.y0 > 120:
                continue
            panel[name].append({"N": n_for_row[round(r.y0, 1)], "hz": round(fx(r.x1), 2)})
        for v in panel.values():
            v.sort(key=lambda e: e["N"])
        bound = [d for d in page.get_drawings() if d.get("dashes") and rgb(d.get("color")) == (0.25, 0.25, 0.25)
                 and inside(d["rect"].x0) and d["rect"].y1 < 125]
        if bound:
            panel["upper_bound_hz"] = round(fx(bound[0]["rect"].x0), 2)
        out.append(panel)
    return out


def main():
    root = Path(sys.argv[1])
    devices = ["rtx4090", "thor"]
    data = {"tradeoff": {}, "ablation": {}}
    for cfg in ("libero", "droid", "aloha"):
        panels = tradeoff(root / f"tradeoff_pi05_o2_{cfg}.pdf")
        data["tradeoff"][cfg] = dict(zip(devices, panels))
    data["ablation"]["libero"] = dict(zip(devices, ablation(root / "combined_ablation_pi05_o2_libero.pdf")))
    Path(sys.argv[2]).write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
