"""Generate the memory/energy overhead table from run_overhead.py output."""

import argparse
import csv
from pathlib import Path


DISPLAY_LABELS = {
    "Ours (batch≈2)": "Ours (batch: 2)",
    "Ours (batch≈4)": "Ours (batch: 4)",
    "Ours (batch≈10)": "Ours (batch: 10)",
    "Ours (batch≈20)": "Ours (batch: 20)",
    "Ours w/o batching": "Ours w/o batch",
    "Parallel (MPS)": "Parallel",
}


def _fmt_delta(delta_pct: float) -> str:
    if abs(delta_pct) < 0.05:
        return "0"
    sign = "+" if delta_pct > 0 else "-"
    return f"${sign}${abs(delta_pct):.1f}"


def _load_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _compute_rows(rows: list[dict]) -> list[dict]:
    baseline = next((row for row in rows if row["label"] == "Baseline"), None)
    if baseline is None:
        raise ValueError("Could not find Baseline row in overhead CSV.")
    baseline_energy = float(baseline["energy_per_request_mj"])

    computed = []
    for row in rows:
        energy = float(row["energy_per_request_mj"])
        delta_pct = (energy / baseline_energy - 1.0) * 100.0
        computed.append({
            "setting": DISPLAY_LABELS.get(row["label"], row["label"]),
            "peak_memory_gb": float(row["peak_memory_gb"]),
            "avg_power_w": float(row["avg_power_w"]),
            "energy_per_request_mj": energy,
            "delta_energy_per_request_pct": delta_pct,
        })
    return computed


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _latex_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Comparison of memory and energy cost. \nickname adds modest memory overhead while achieving substantial energy savings.}",
        r"  \label{tab:overhead}",
        r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{l|rrrr}",
        r"    \toprule",
        r"    \shortstack[c]{Setting\\~} & \shortstack{Peak\\Mem. (GB)} & \shortstack{Avg.\\Power (W)} & \shortstack{Energy\\/Req. (mJ)} & \shortstack{$\Delta$ Energy\\ /Req. (\%)} \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            f"{row['setting']} & "
            f"{row['peak_memory_gb']:.2f} & "
            f"{row['avg_power_w']:.1f} & "
            f"{row['energy_per_request_mj']:.1f} & "
            f"{_fmt_delta(row['delta_energy_per_request_pct'])} \\\\"
        )
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Root directory passed to experiments.run_overhead.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Override input CSV path. Defaults to {results-dir}/analysis/overhead/overhead.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory. Defaults to {results-dir}/analysis/overhead.",
    )
    args = parser.parse_args()

    input_csv = args.input_csv or args.results_dir / "analysis" / "overhead" / "overhead.csv"
    output_dir = args.output_dir or args.results_dir / "analysis" / "overhead"

    rows = _compute_rows(_load_rows(input_csv))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "overhead_table.csv")
    latex = _latex_table(rows)
    (output_dir / "overhead_table.tex").write_text(latex + "\n")

    print(latex)
    print(f"\nWrote {output_dir / 'overhead_table.csv'}")
    print(f"Wrote {output_dir / 'overhead_table.tex'}")


if __name__ == "__main__":
    main()
