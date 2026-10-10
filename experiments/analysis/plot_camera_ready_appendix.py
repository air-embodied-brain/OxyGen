"""Publication figures from archived camera-ready experiment data.

Run as a module from OxyGen; --results-root points to the workspace results.
All outputs have a fixed 8.5-inch canvas; insert at full text width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import numpy as np

from experiments.analysis.plot_utils import PlotColors as C, setup_style

GREEN = '#1F6B2E'
GREEN_BG = '#C9EACA'


def axis(ax):
    ax.grid(alpha=.5, linestyle='--', linewidth=.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(.8)
        spine.set_color('black')


def save(fig, out):
    fig.canvas.draw()
    # Fixed canvas preserves font sizes when each figure is included at linewidth.
    with plt.rc_context({'savefig.bbox': None, 'savefig.pad_inches': 0}):
        fig.savefig(out.with_suffix('.pdf'))
        fig.savefig(out.with_suffix('.png'), dpi=160)
    plt.close(fig)


def plot_manager(root, out):
    data = json.loads((root / 'kv_manager_fullpath_aligned_20260729/aggregate.json').read_text())
    points = [data['jax']['points'][f'N30_k{k}']['synchronized_full_path'] for k in (5, 1)]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.9), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.subplots_adjust(left=.14, right=.98, bottom=.36, top=.80, wspace=.24)
    ax = axes[0]
    stages = [('Prefix', 'prefix_ms', C.OURS_LIGHT), ('Denoising', 'denoise_ms', C.VARIANT_GREEN),
              ('Language decode', 'decode_ms', C.OURS_PRIMARY), ('KV manager', 'manager_ms', C.VARIANT_GOLD)]
    left = np.zeros(2)
    for label, key, color in stages:
        vals = np.array([p[key] for p in points])
        ax.barh([1, 0], vals, left=left, height=.30, color=color, label=label,
                edgecolor=C.BAR_EDGE, linewidth=.5)
        left += vals
    other = [p['mean_ms'] - v for p, v in zip(points, left)]
    ax.barh([1, 0], other, left=left, height=.30, color='#D3D3D3', label='Other frame work',
            edgecolor=C.BAR_EDGE, linewidth=.5)
    for y, p in zip([1, 0], points):
        ax.text(p['mean_ms']+3, y, f"{p['mean_ms']:.1f}", va='center', fontsize=11)
    frame_limit = max(240, max(p['mean_ms'] for p in points) * 1.18)
    ax.set(yticks=[1, 0], yticklabels=['Batch 6\n($k=5$)', 'Batch 30\n($k=1$)'],
           xlim=(0, frame_limit), ylim=(-.5, 1.5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_title('(a) Full frame', fontweight='bold')
    ax.set_xlabel('Frame Time (ms)')
    axis(ax)
    ax = axes[1]
    left = np.zeros(2)
    for label, key, color, hatch in [('KV stack', 'kv_stack_ms', '#F8E6B0', ''),
                                    ('KV split', 'kv_split_ms', C.VARIANT_GOLD, '///'),
                                    ('Bookkeeping', 'crud_ms', C.NEUTRAL_DARK, '')]:
        vals = [p[key] for p in points]
        ax.barh([1, 0], vals, left=left, height=.30, color=color, hatch=hatch,
                edgecolor=C.BAR_EDGE, linewidth=.5, label=label)
        left += vals
    for y, p in zip([1, 0], points):
        ax.text(p['manager_ms']+2, y, f"{100*p['manager_fraction']:.1f}%", va='center', fontsize=11,
                fontweight='bold')
    manager_limit = max(95, max(p['manager_ms'] for p in points) * 1.35)
    split_ratio = points[1]['kv_split_ms'] / points[0]['kv_split_ms']
    ax.annotate(f'KV split: {split_ratio:.1f}×', xy=(points[1]['kv_stack_ms'] + points[1]['kv_split_ms']*.6, .05), xytext=(manager_limit*.42, .68), ha='center',
                fontsize=11, color=GREEN, fontweight='bold',
                bbox=dict(boxstyle='round,pad=.2', fc=GREEN_BG, ec='none'),
                arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=1.2))
    ax.set(xlim=(0, manager_limit), ylim=(-.5, 1.5), yticks=[0, 1], yticklabels=[])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set_title('(b) KV manager', fontweight='bold')
    ax.set_xlabel('Manager Time (ms)')
    axis(ax)
    for ax, center in zip(axes, [.35, .82]):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(center, .01),
                   ncol=3, fontsize=9, frameon=False, columnspacing=.8, handlelength=1.5)
    fig.suptitle(r'GeForce RTX 4090 · JAX · $N=30$, $S=10$', y=.99, fontsize=12)
    save(fig, out / 'manager_overhead_breakdown')


def plot_scaling(root, out):
    data = json.loads((root / 'qwen35_scaling_tp_formal_20260728/aggregated.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.25), sharey=True)
    fig.subplots_adjust(left=.085, right=.75, bottom=.23, top=.88, wspace=.12)
    sizes = ['0.8B', '2B', '4B', '9B']
    styles = [
        ('Baseline', 'separate|N={N}|k=None', C.BASELINE_PRIMARY, 's', '-'),
        ('Ours w/o Batching', 'shared_kv|N={N}|k=None', C.VARIANT_GOLD, 'D', '--'),
        (r'Ours ($k=1$)', 'continuous_batching|N={N}|k=1', C.OURS_DARK, 'o', '-'),
        (r'Ours ($k=5$)', 'continuous_batching|N={N}|k=5', C.OURS_PRIMARY, 'o', '-'),
        (r'Ours ($k=10$)', 'continuous_batching|N={N}|k=10', C.OURS_LIGHT, 'o', '-'),
    ]
    ymax = max(data[f'{size}-TP1'][pattern.format(N=N)]['latency_ms']
               for size in sizes for _, pattern, *_ in styles for N in [10, 30])
    ylimit = max(1900, ymax * 1.12)
    for ax, N in zip(axes, [10, 30]):
        x = np.arange(len(sizes))
        for label, pattern, color, marker, ls in styles:
            vals = [data[f'{size}-TP1'][pattern.format(N=N)]['latency_ms'] for size in sizes]
            ax.plot(x, vals, color=color, marker=marker, linestyle=ls, lw=1.7,
                    markersize=5.5, markeredgecolor='white', markeredgewidth=.8,
                    label=label)
        ax.set(xticks=x, xticklabels=sizes, xlim=(-.25, 3.25), ylim=(0, ylimit))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel('Qwen3.5 Backbone')
        ax.set_title(f'$N={N}$ tokens', fontweight='bold')
        axis(ax)
    axes[0].set_ylabel('Frame Latency (ms)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(.765, .555), ncol=1,
               fontsize=10, frameon=True, facecolor='#EEEEEE', edgecolor='#D0D0D0',
               labelspacing=1.1, handlelength=1.8, borderpad=.8)
    save(fig, out / 'model_size_scaling')

def trimmean(values):
    values = np.sort(values)
    n = int(len(values)*.05)
    return np.mean(values[n:len(values)-n] if n else values)


def plot_stability(root, out):
    folder = root / 'eccv_submission_raw/kv_manager_longrun_20260726/raw/bs32'
    fig, ax = plt.subplots(figsize=(8.5, 3.35))
    # Keep the plotting area narrow and centered when included at text width.
    fig.subplots_adjust(left=.20, right=.80, bottom=.20, top=.80)
    ax_mem = ax.twinx()
    colors = {'jax': (C.OURS_DARK, C.OURS_LIGHT), 'pytorch': (C.BASELINE_PRIMARY, C.BASELINE_LIGHT)}
    latency_handles, memory_handles = [], []
    all_latency, all_memory = [], []
    for backend in ['jax', 'pytorch']:
        color_latency, color_memory = colors[backend]
        rows = [json.loads(s) for s in (folder / f'{backend}_fluctuating_bs32/frames.jsonl').read_text().splitlines()]
        rows = [r for r in rows if not r['is_warmup']]
        assert len(rows) == 3000
        memkey = 'bytes_in_use' if backend == 'jax' else 'allocated_bytes'
        samples = [r for r in rows if r.get('memory')]
        xmem = np.array([r['frame'] - rows[0]['frame'] + 1 for r in samples])
        mem = np.array([r['memory'][memkey] / 2**30 for r in samples])
        # Per-window envelope makes allocation variation visible without a noisy trace.
        bins = np.arange(0, 3001, 50)
        centers, lo, hi, mid = [], [], [], []
        for start, end in zip(bins[:-1], bins[1:]):
            vals = mem[(xmem >= start + 1) & (xmem <= end)]
            if len(vals):
                centers.append((start + end) / 2); lo.append(np.min(vals)); hi.append(np.max(vals)); mid.append(np.mean(vals))
        centers, lo, hi, mid = map(np.asarray, (centers, lo, hi, mid))
        ax_mem.fill_between(centers, lo, hi, color=color_memory, alpha=.28, linewidth=0)
        mh, = ax_mem.plot(centers, mid, color=color_memory, lw=1.5, linestyle='--', label=f'{backend.upper()} memory')
        memory_handles.append((Patch(facecolor=color_memory, alpha=.28, edgecolor='none'), mh))
        byframe = {r['frame']: r for r in rows}
        cycles = [[byframe[f] for f in range(start, start + 119)] for start in range(119, 3094, 119)]
        assert len(cycles) == 25
        x = [np.mean([r['frame'] for r in c]) - rows[0]['frame'] + 1 for c in cycles]
        means = [np.mean([r['frame_ms'] for r in c]) for c in cycles]
        all_latency.extend(means); all_memory.extend(mem)
        lh, = ax.plot(x, means, color=color_latency, lw=1.7, marker='o' if backend == 'jax' else '^',
                      markersize=4.5, markeredgecolor='white', markeredgewidth=.5,
                      label=f'{backend.upper()} latency')
        latency_handles.append(lh)
        first = np.mean([r['frame_ms'] for c in cycles[:5] for r in c])
        last = np.mean([r['frame_ms'] for c in cycles[-5:] for r in c])
        print(f'{backend}: memory {min(mem):.3f}–{max(mem):.3f} GiB; latency {first:.1f}→{last:.1f} ms')
    def limits(values, pad):
        delta = max(max(values)-min(values), pad)
        return max(0, min(values)-delta*.2), max(values)+delta*.2
    ax.set(xlim=(0, 3000), xticks=[0, 1000, 2000, 3000], ylim=limits(all_latency, 10))
    ax_mem.set(ylim=limits(all_memory, .2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_mem.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel('Measured Frame')
    ax.set_ylabel('Frame Latency (ms)')
    ax_mem.set_ylabel('Active Memory (GiB)')
    axis(ax)
    for spine in ax_mem.spines.values(): spine.set_color('black'); spine.set_linewidth(.8)
    ax_mem.grid(False)
    handles = latency_handles + memory_handles
    labels = ['JAX Latency', 'PyTorch Latency', 'JAX Memory Range', 'PyTorch Memory Range']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .96), ncol=4,
               handler_map={tuple: HandlerTuple(ndivide=1)}, columnspacing=1.2,
               frameon=True, facecolor='#EEEEEE', edgecolor='#D0D0D0', fontsize=11)
    save(fig, out / 'longrun_stability')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/results_camera_ready/plot_appendix'))
    parser.add_argument('--only', choices=['all', 'stability'], default='all',
                        help='Use stability for the sole supplementary figure adopted in the reviewed paper.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    names = {f.name for f in font_manager.fontManager.ttflist}
    available = [f for f in ['Times New Roman', 'Liberation Serif'] if f in names]
    if not available:
        raise RuntimeError('Install Times New Roman, Liberation Serif or Tinos; refusing silent font substitution.')
    plt.rcParams.update({'font.serif': available, 'mathtext.fontset': 'stix'})
    print('Figure font:', available[0])
    if args.only == 'all':
        plot_manager(args.results_root, args.output_dir)
        plot_scaling(args.results_root, args.output_dir)
    plot_stability(args.results_root, args.output_dir)


if __name__ == '__main__':
    main()
