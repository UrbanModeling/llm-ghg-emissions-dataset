"""
plot_temporal.py — Temporal distribution figure
  a) Single-year records: bar chart, year up to / after the paper's publication year
  b) Multi-year range records: 2D heatmap (start year × end year, log color)

Output: figures/temporal_distribution.png

Reads the year column of output/full/emission_records.csv (records without a year are not drawn).

Usage:
    python plots/plot_temporal.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.stdout.reconfigure(encoding='utf-8')

BASE    = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / 'output' / 'full'
FIG_DIR = BASE / 'figures'

mpl.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':        8,
    'axes.titlesize':   9,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.dpi':      300,
    'pdf.fonttype':     42,
})

WOS_RAW = BASE / 'data' / 'raw'   # Web of Science export (publication year)

df = pd.read_csv(OUT_DIR / 'emission_records.csv', encoding='utf-8-sig', dtype={'year': str})
yr = df['year'].dropna().astype(str)

# publication year of each paper: a single year after it is a projection or target
ids = pd.read_csv(BASE / 'output' / 'prepared.csv', usecols=['uid', 'wos_id'])
wos = pd.concat([pd.read_excel(f, usecols=['UT (Unique WOS ID)', 'Publication Year'])
                 for f in sorted(WOS_RAW.glob('*.xls'))])
pub = ids.merge(wos, left_on='wos_id', right_on='UT (Unique WOS ID)').set_index('uid')['Publication Year']

# ── Separate single-year vs range ────────────────────────────────
import re
RANGE_RE = re.compile(r'^(\d{4})-(\d{4})$')
SINGLE_RE = re.compile(r'^\d{4}$')

singles, range_starts, range_ends = [], [], []
for v in yr:
    m = RANGE_RE.match(v.strip())
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 <= y2:
            range_starts.append(y1)
            range_ends.append(y2)
    elif SINGLE_RE.match(v.strip()):
        singles.append(int(v.strip()))

singles = pd.Series(singles)
range_starts = np.array(range_starts)
range_ends   = np.array(range_ends)

# ── Figure ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')
fig.subplots_adjust(wspace=0.35)

# ── Panel a: single-year bar chart ───────────────────────────────
ax = axes[0]

one = df[df['year'].astype(str).str.fullmatch(r'\d{4}')].copy()
one['y'] = one['year'].astype(int)
one['projected'] = one['y'] > one['uid'].map(pub)
print(f"single-year {len(one)} | reported {int((~one.projected).sum())} | projected {int(one.projected.sum())}")
one = one[one['y'].between(1988, 2100)]
c = one.groupby(['y', 'projected']).size().unstack(fill_value=0).reindex(
    columns=[False, True], fill_value=0).reindex(range(1988, 2101), fill_value=0)

ax.bar(c.index, c[False], color='#9B1C1C', width=0.8, label='Year ≤ publication year (reported)')
ax.bar(c.index, c[True], bottom=c[False], color='#F4A9A8', width=0.8,
       label='Year > publication year (projected)')
ax.set_ylim(0, (c[False] + c[True]).max() * 1.3)      # room for the legend above the tallest bar

ax.set_xlabel('Year', fontsize=8.5)
ax.set_ylabel('Number of records', fontsize=8.5)
ax.set_title('Single-year records', fontsize=9, pad=5, loc='left')
ax.text(-0.08, 1.05, 'a', transform=ax.transAxes, fontsize=10, fontweight='bold')

ax.legend(fontsize=7.5, frameon=False, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=7.5)
ax.set_xlim(1987, 2101)

# ── Panel b: range records — horizontal rectangles ───────────────
ax = axes[1]

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# Bin to 5-year intervals for cleaner display
def snap(y, step=5):
    return (y // step) * step

starts_s = np.array([snap(y) for y in range_starts])
ends_s   = np.array([snap(y) for y in range_ends])

# Count records per (start_bin, end_bin) pair
pair_counts = {}
for s, e in zip(starts_s, ends_s):
    if s < e:                        # skip zero-length after snapping
        pair_counts[(s, e)] = pair_counts.get((s, e), 0) + 1

max_count = max(pair_counts.values())
norm = mcolors.LogNorm(vmin=1, vmax=max_count)
cmap = plt.cm.YlOrRd

patches, colors = [], []
for (s, e), cnt in pair_counts.items():
    rel_freq = cnt / max_count
    patches.append(Rectangle((s, 0), e - s, rel_freq))
    colors.append(cnt)

pc = PatchCollection(patches, cmap=cmap, norm=norm,
                     edgecolor='none', alpha=0.85)
pc.set_array(np.array(colors))
ax.add_collection(pc)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
cbar.set_label('Number of records', fontsize=7.5)
cbar.ax.tick_params(labelsize=7)

ax.set_xlim(1950, 2105)
ax.set_ylim(0, 1.05)
ax.set_xlabel('Year', fontsize=8.5)
ax.set_ylabel('Relative frequency', fontsize=8.5)
ax.set_title('Multi-year range records', fontsize=9, pad=5, loc='left')
ax.text(-0.10, 1.05, 'b', transform=ax.transAxes, fontsize=10, fontweight='bold')
ax.tick_params(labelsize=7.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

FIG_DIR.mkdir(exist_ok=True)
out = FIG_DIR / 'temporal_distribution.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved -> {out.name}')
