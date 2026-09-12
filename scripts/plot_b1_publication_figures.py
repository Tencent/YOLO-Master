"""Render English PR figures from the retained controlled-intervention CSV files."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'reports/text_router_analysis/controlled_interventions'

def rows(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))
C = {(int(r['seed']), r['condition']): r for r in rows('conditions.csv')}
B = {(int(r['seed']), r['contrast'], r['metric']): r for r in rows('paired_bootstrap.csv')}
O = {int(r['seed']): r for r in rows('opportunity_summary.csv')}
BLUE, ORANGE, GRAY = ('#0072B2', '#D55E00', '#d9dce1')
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'path', 'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'savefig.facecolor': 'white'})

def setup(ax, cn):
    ax.set_yticks(range(3), [f'Seed {i}' for i in range(3)])
    ax.set_ylim(2.6, -0.6)
    ax.grid(axis='x', color='#e8e8e8', lw=0.6, zorder=0)
    ax.set_axisbelow(True)

def save(fig, stem, cn):
    suffix = 'en'
    for ext in ['svg']:
        fig.savefig(DATA / 'figures' / f'{stem}.{ext}', dpi=190, bbox_inches='tight')
    plt.close(fig)
for cn in [False]:
    plt.rcParams['font.family'] = ['DejaVu Sans']
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.55), gridspec_kw={'width_ratios': [1.15, 1.15, 1]}, layout='constrained')
    a, b, c = axs
    for ax in axs:
        setup(ax, cn)
    for label, offset, col, marker in [('GZ-T', -0.12, BLUE, 'o'), ('GW-T', 0.12, ORANGE, 's')]:
        for seed in range(3):
            r = B[seed, label, 'AP']
            point = 100 * float(r['point_delta'])
            lo = 100 * float(r['bootstrap_p2_5'])
            hi = 100 * float(r['bootstrap_p97_5'])
            a.plot([lo, hi], [seed + offset] * 2, color=col, lw=1.4)
            a.plot(point, seed + offset, marker, color=col, ms=4)
    a.axvline(0, color='#555', ls='--', lw=0.8)
    a.set_xlim(-0.18, 0.045)
    a.set_title('(a) Gate substitution')
    a.set_xlabel('New17 AP difference (points)')
    a.legend([Patch(color=BLUE), Patch(color=ORANGE)], ['Zero text', 'Other-class text'], loc='lower left', fontsize=8, frameon=False)
    for seed in range(3):
        x = [100 * float(C[seed, k]['new17_AP']) for k in ['E0', 'E1']]
        b.plot(x, [seed] * 2, color='#999', lw=1)
        b.scatter(x, [seed] * 2, c=[BLUE, ORANGE], s=26, zorder=3)
        for expert, value in enumerate(x):
            b.annotate(f'{value:.3f}', (value, seed), xytext=(0, 10 if expert == 0 else -16), textcoords='offset points', ha='center', fontsize=8, color=BLUE if expert == 0 else ORANGE)
    b.set_title('(b) Fixed-expert quality')
    b.set_xlabel('New17 AP (points)')
    b.set_xlim(25.35, 31.4)
    b.legend([Patch(color=BLUE), Patch(color=ORANGE)], ['Expert 0', 'Expert 1'], loc='upper center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.015))
    for seed in range(3):
        n = int(C[seed, 'T']['expert0_count'])
        share = n / 50
        dominant = max(n, 5000 - n) / 50
        c.plot(dominant, seed, 'o', color=BLUE if n > 2500 else ORANGE, ms=5)
        c.annotate(f'{dominant:.2f}%', (dominant, seed), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
    c.set_xlim(99.8, 100.04)
    c.set_xticks([99.8, 99.9, 100])
    c.set_title('(c) Dominant expert use (zoom)')
    c.set_xlabel('More-used expert share (%)')
    c.axvline(100, color='#777', ls=':', lw=0.8)
    save(fig, 'fig5_controlled_interventions', cn)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw={'width_ratios': [1.35, 1.1]}, layout='constrained')
    a, c = axs
    for ax in axs:
        setup(ax, cn)
    for seed in range(3):
        r = O[seed]
        left = 0
        for key, col in [('images_E0_better', BLUE), ('images_E1_better', ORANGE), ('images_tie', GRAY)]:
            value = int(r[key]) / 50
            a.barh(seed, value, left=left, height=0.4, color=col)
            a.text(left + value / 2, seed, f'{value:.1f}', ha='center', va='center', fontsize=8, color='#333' if col == GRAY else 'white')
            left += value
        delta = 100 * float(r['oracle_minus_best_fixed_new17_AP'])
        c.hlines(seed, 0, delta, color='#596574', lw=1.3)
        c.plot(delta, seed, 'o', color='#334155', ms=5)
        c.annotate(f'{delta:+.3f}', (delta, seed), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
    a.set_title('(a) Per-image F1 comparison')
    a.set_xlim(0, 100)
    a.set_xticks([0, 25, 50, 75, 100])
    a.set_xlabel('Share of all images (%)')
    a.legend([Patch(color=x) for x in [BLUE, ORANGE, GRAY]], ['E0 higher', 'E1 higher', 'Tie'], fontsize=8, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.01))
    c.set_title('(b) Complete-prediction outcome')
    c.axvline(0, ls='--', color='#666', lw=0.8)
    c.set_xlim(-0.25, 0.62)
    c.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
    c.set_xlabel('New17 AP vs. fixed E0 (points)')
    save(fig, 'fig6_local_opportunity', cn)
print('Rendered two English SVG figures.')

# Normalize SVG whitespace for readable repository diffs.
for svg in (DATA / 'figures').glob('fig[56]_*.svg'):
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
