"""
Figure 4: sectoral distribution. Per-group counts are built from output/full/emission_records.csv (records without a sector group are not drawn;
total_papers = distinct papers per group).

Usage:
    python plots/plot_sector.py
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
mpl.rcParams.update({            # same as plot_temporal.py
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':        8,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.dpi':      300,
    'pdf.fonttype':     42,
})

# 1. 每个 sector group 的记录数（按空间层级拆分）和论文数
_rec = pd.read_csv(BASE / "output" / "full" / "emission_records.csv", encoding="utf-8-sig",
                   dtype=str).fillna("")
_rec = _rec[_rec["sector_group"] != ""]
_lvl = {"country": "Country_records", "subnational": "Subnational_records",
        "city": "City_records", "local": "Unknown_records"}   # named regions (local) -> Other
df = (_rec.groupby(["sector_group", "spatial_level"]).size().unstack(fill_value=0)
          .rename(columns=_lvl).reindex(columns=list(_lvl.values()), fill_value=0))
df["total_records"] = df.sum(axis=1)
df["total_papers"] = _rec.groupby("sector_group")["uid"].nunique()
df = df.rename_axis("sector_type").reset_index()

# 2. 计算强度（Y轴）：每篇论文的 records 数，并按此升序排序
df['records_per_paper'] = df['total_records'] / df['total_papers']
df = df.sort_values('records_per_paper', ascending=True).reset_index(drop=True)

# 3. 计算绘图参数
# X轴宽度：该 sector_type 的论文总数 (Volume)
widths    = df['total_papers'].values
x_edges   = np.append([0], np.cumsum(widths))
x_centers = x_edges[:-1] + widths / 2

# Y轴各堆叠部分的高度 (每篇论文的 records 强度)
h_country     = df['Country_records']     / df['total_papers']
h_subnational = df['Subnational_records'] / df['total_papers']
h_city        = df['City_records']        / df['total_papers']
h_unknown     = df['Unknown_records']     / df['total_papers']

components = [h_country, h_subnational, h_city, h_unknown]
sp_labels  = ['Country', 'Subnational', 'City', 'Local']
colors     = ['#EEDD77', '#77AB77', '#EE7777', '#55AADD']

# 4. 绘图
fig, ax = plt.subplots(figsize=(10, 4.5))

bottom = np.zeros(len(widths))
for data, color, label in zip(components, colors, sp_labels):
    ax.bar(x_centers, data, width=widths, bottom=bottom, label=label,
           color=color, edgecolor='black', linewidth=0.5)
    bottom += data

# Sector type 标签置于柱顶
for i, sector in enumerate(df['sector_type']):
    rotation = 90 if widths[i] < 400 else 0
    ax.text(x_centers[i], bottom[i] + 0.05, sector,
            ha='center', va='bottom', fontsize=7.5, rotation=rotation)

# 5. 坐标轴
ax.set_xlim(0, x_edges[-1])
ax.set_ylim(0, bottom.max() + 0.8)
ax.set_xlabel('Papers (cumulative)', fontsize=8.5)
ax.set_ylabel('Records per paper', fontsize=8.5)
ax.tick_params(labelsize=7.5)

ax.set_axisbelow(True)
ax.grid(True, axis='both', linestyle='-', color='#DDDDDD', linewidth=0.5)

ax.legend(loc='upper left', ncol=2, fontsize=7.5, framealpha=1, edgecolor='black')

plt.tight_layout()
(BASE / "figures").mkdir(exist_ok=True)
plt.savefig(BASE / "figures" / "sector_distribution.png",
            dpi=300, bbox_inches='tight')
