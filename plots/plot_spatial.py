"""
plot_spatial.py

3-panel spatial distribution map (Nature style, Natural Earth projection).
  a) Country-level  — choropleth
  b) Subnational    — scatter points (centroids from ACM1)
  c) City-level     — scatter points (lat/lon from geonamescache)

Output: figures/spatial_distribution.png

Counts come from output/full/emission_records.csv. Named regions (spatial_level 'local') are not drawn.

Usage:
    python plots/plot_spatial.py
"""
import re, sys, unicodedata, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from pyproj import Transformer

warnings.filterwarnings('ignore')

BASE      = Path(__file__).resolve().parent.parent
OUT_DIR   = BASE / 'output' / 'full'
FIG_DIR   = BASE / 'figures'
ADM0_SHP  = BASE / 'data' / 'external' / 'geoBoundaries' / 'geoBoundariesCGAZ_ADM0.shp'
ADM1_SHP  = BASE / 'data' / 'external' / 'geoBoundaries' / 'geoBoundariesCGAZ_ADM1.shp'
NE_CRS    = '+proj=natearth +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

mpl.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':        7.5,
    'axes.titlesize':   8.5,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.dpi':      300,
    'pdf.fonttype':     42,
})

# ── Load classification results ───────────────────────────────────
rec = pd.read_csv(OUT_DIR / 'emission_records.csv', encoding='utf-8-sig', low_memory=False)
lvl = rec['spatial_level']

country_data = (rec[lvl == 'country'].groupby('country_iso3').size()
                .rename('record_count').reset_index())

sub_data = (rec[lvl == 'subnational'].groupby(['subnational', 'country_iso3']).size()
            .rename('record_count').reset_index())

city_data = (rec[(lvl == 'city') & rec['lat'].notna()].groupby(['city', 'country_iso3']).agg(
                 lat=('lat', 'first'), lon=('lon', 'first'), record_count=('uid', 'size'))
             .reset_index())

print(f'Country entries : {len(country_data)}')
print(f'Subnational     : {len(sub_data)}')
print(f'City            : {len(city_data)}')

# ── Build ACM1 centroid index ─────────────────────────────────────
_SUB_SUFFIXES = re.compile(
    r'\s+(Sheng|Shi|SAR|Zizhiqu|Province|Provincial|State|Region|Prefecture|'
    r'District|Territory|County|Oblast|Krai|Republic|Autonomous|Municipality|'
    r'Governorate|Department|Division|Special Administrative Region).*$',
    re.IGNORECASE
)
# geonames names carry accents and the CGAZ file has long, garbled or wrong names
_ETHNIC = re.compile(r'\s+(Uyghur|Zhuang|Hui)\b.*$', re.IGNORECASE)
_ALIAS  = {('guangzhou', 'CHN'): 'guangdong'}          # CGAZ labels Guangdong 'Guangzhou Province'

def sub_key(name, iso3):
    # the CGAZ .dbf stores '?' for characters outside Latin-1 (Karn?taka): degrade names the same way
    n = str(name).encode('latin1', 'replace').decode('latin1').replace('?', '')
    n = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode()
    n = _ETHNIC.sub('', _SUB_SUFFIXES.sub('', n.strip(" '‘’"))).strip().lower()
    n = re.sub(r'^(\w+) \1\b', r'\1', n)                # 'Ningxia Ningxia Hui ...' -> 'ningxia'
    return (_ALIAS.get((n, iso3), n), iso3)

print('Loading ACM1...')
adm1 = gpd.read_file(ADM1_SHP).to_crs('EPSG:3857')
adm1['lat'] = adm1.geometry.centroid.to_crs('EPSG:4326').y
adm1['lon'] = adm1.geometry.centroid.to_crs('EPSG:4326').x
adm1 = adm1.dropna(subset=['shapeName'])

adm1_idx = {}
for _, row in adm1.iterrows():
    raw   = str(row['shapeName']).strip()
    clean = _SUB_SUFFIXES.sub('', raw).strip()
    iso3  = row['shapeGroup']
    adm1_idx.setdefault(sub_key(raw, iso3), (row['lat'], row['lon']))

# Attach coordinates to subnational data
def get_sub_coords(row):
    key = sub_key(row['subnational'], row['country_iso3'])
    return pd.Series(adm1_idx.get(key, (np.nan, np.nan)), index=['lat', 'lon'])

sub_data[['lat', 'lon']] = sub_data.apply(get_sub_coords, axis=1)
_miss = sub_data['lat'].isna()
print(f'Subnational not drawn: {int(_miss.sum())} units, {int(sub_data.loc[_miss, "record_count"].sum())} '
      f'of {int(sub_data["record_count"].sum())} records')
sub_data = sub_data.dropna(subset=['lat', 'lon'])
print(f'Subnational with coords: {len(sub_data)}')

# ── Load world boundaries ─────────────────────────────────────────
print('Loading ADM0...')
world = gpd.read_file(ADM0_SHP)
world = world[world['shapeName'] != 'Antarctica']
world = world.merge(country_data, left_on='shapeGroup', right_on='country_iso3', how='left')
world['record_count'] = world['record_count'].fillna(0)
world_ne = world.to_crs(NE_CRS)

# ── Build map boundary ────────────────────────────────────────────
tr      = Transformer.from_crs('EPSG:4326', NE_CRS, always_xy=True)
lons_g  = np.linspace(-180, 180, 361)
LAT_MIN = -60
n_arc   = 200
lats_arc = np.linspace(LAT_MIN, 90, n_arc)
xl, yl = tr.transform([-180]*n_arc,  lats_arc)
xt, yt = tr.transform(lons_g,        [90]*361)
xr, yr = tr.transform([180]*n_arc,   lats_arc[::-1])
xb, yb = tr.transform(lons_g[::-1],  [LAT_MIN]*361)
bnd_x  = np.concatenate([xl, xt, xr, xb])
bnd_y  = np.concatenate([yl, yt, yr, yb])
verts  = np.column_stack([bnd_x, bnd_y])
codes  = [MPath.MOVETO] + [MPath.LINETO]*(len(verts)-2) + [MPath.CLOSEPOLY]

lats_g = np.linspace(LAT_MIN, 90, 181)

def reproject_points(lats, lons):
    rx, ry = tr.transform(lons, lats)
    return rx, ry

# ── Color setup ───────────────────────────────────────────────────
cmap = cm.YlOrRd

def make_norm(data_series):
    vmin = max(data_series.min(), 1)
    vmax = data_series.max()
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)

norm_country = make_norm(country_data['record_count'])
norm_sub     = make_norm(sub_data['record_count'])
norm_city    = make_norm(city_data['record_count'])

# ── Draw helper ───────────────────────────────────────────────────
def setup_ax(ax):
    ax.add_patch(PathPatch(MPath(verts, codes),
                           facecolor='white', edgecolor='none', zorder=0))
    for lon in range(-150, 181, 30):
        xs, ys = tr.transform([lon]*181, lats_g)
        ax.plot(xs, ys, color='white', lw=0.25, zorder=1)
    for lat in range(-30, 61, 30):
        xs, ys = tr.transform(lons_g, [lat]*361)
        ax.plot(xs, ys, color='white', lw=0.25, zorder=1)
    world_ne[world_ne['record_count'] == 0].plot(
        ax=ax, color='#e8e8e8', edgecolor='#bbbbbb', linewidth=0.2, zorder=2)
    xs_eq, ys_eq = tr.transform(lons_g, [0]*361)
    ax.plot(xs_eq, ys_eq, color='#999', lw=0.4, ls=':', zorder=4)
    ax.plot(bnd_x, bnd_y, color='#555', lw=0.5, zorder=6)
    X_MAX = np.nanmax(np.abs(bnd_x)) * 1.01
    Y_TOP = np.nanmax(bnd_y) * 1.02
    Y_BOT = np.nanmin(bnd_y) * 1.02
    ax.set_xlim(-X_MAX, X_MAX)
    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_aspect('equal')
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

def add_colorbar(fig, ax, norm, label):
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        fraction=0.025, pad=0.02, aspect=40, shrink=0.45)
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    cbar.ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    return cbar

def set_title(ax, i):
    ax.set_title(f'{PANEL_LABELS[i]}   {TITLES[i]}', fontsize=8.5, pad=4, loc='left', fontweight='bold')

# ── Figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(7, 10), facecolor='white')
fig.subplots_adjust(hspace=0.28)

PANEL_LABELS = ['a', 'b', 'c']
TITLES = [
    'Country-level records',
    'Subnational-level records',
    'City-level records',
]

# ── Panel a: Country choropleth ───────────────────────────────────
ax = axes[0]
setup_ax(ax)
world_ne[world_ne['record_count'] > 0].plot(
    ax=ax, column='record_count', cmap=cmap, norm=norm_country,
    edgecolor='#bbbbbb', linewidth=0.2, zorder=3, aspect=None)
add_colorbar(fig, ax, norm_country, 'Number of records')
set_title(ax, 0)

# ── Panel b: Subnational choropleth ──────────────────────────────
ax = axes[1]
setup_ax(ax)

# Merge record counts into ACM1 shapefile
adm1_ne = gpd.read_file(ADM1_SHP).to_crs(NE_CRS)
adm1_ne = adm1_ne[adm1_ne['shapeName'].notna()].copy()
adm1_ne['name_clean'] = adm1_ne['shapeName'].apply(
    lambda x: _SUB_SUFFIXES.sub('', str(x)).strip())

# Match sub_data to ACM1 by name + country
sub_lookup = {}
for n, i, c in zip(sub_data.subnational, sub_data.country_iso3, sub_data.record_count):
    k = sub_key(n, i)
    sub_lookup[k] = sub_lookup.get(k, 0) + c

def match_records(row):
    return sub_lookup.get(sub_key(row['shapeName'], row['shapeGroup']), 0)

adm1_ne['record_count'] = adm1_ne.apply(match_records, axis=1)

world_ne.plot(ax=ax, color='#e8e8e8', edgecolor='#bbbbbb', linewidth=0.2, zorder=2)
adm1_ne[adm1_ne['record_count'] == 0].plot(
    ax=ax, color='#e8e8e8', edgecolor='#cccccc', linewidth=0.1, zorder=3)
adm1_ne[adm1_ne['record_count'] > 0].plot(
    ax=ax, column='record_count', cmap=cmap, norm=norm_sub,
    edgecolor='#bbbbbb', linewidth=0.1, zorder=4, aspect=None)
add_colorbar(fig, ax, norm_sub, 'Number of records')
set_title(ax, 1)

# ── Panel c: City scatter ─────────────────────────────────────────
ax = axes[2]
setup_ax(ax)
world_ne.plot(ax=ax, color='#e8e8e8', edgecolor='#bbbbbb', linewidth=0.2, zorder=2)
# one colour; marker area grows with the square root of the count; large cities drawn first
CITY_COLOR = '#b2182b'
city_size = lambda n: 3 + 0.9 * np.asarray(n, dtype=float)     # marker area proportional to count
cd = city_data.sort_values('record_count', ascending=False)
rx, ry = reproject_points(cd['lat'].values, cd['lon'].values)
ax.scatter(rx, ry, s=city_size(cd['record_count'].values), color=CITY_COLOR, alpha=0.75,
           edgecolors='white', linewidths=0.3, zorder=5, rasterized=True)
handles = [ax.scatter([], [], s=city_size(n), color=CITY_COLOR, alpha=0.75, edgecolors='white',
                      linewidths=0.3, label=f'{n}') for n in (1, 10, 50, 200)]
add_colorbar(fig, ax, norm_city, '').ax.set_visible(False)   # same map size as panels a and b
ax.legend(handles=handles, title='Number of records', title_fontsize=7, fontsize=6, ncol=4,
          loc='upper center', bbox_to_anchor=(0.5, -0.01), frameon=False, columnspacing=1.2,
          handletextpad=0.3, borderaxespad=0)
set_title(ax, 2)

# ── Save ──────────────────────────────────────────────────────────
FIG_DIR.mkdir(exist_ok=True)
out_png = FIG_DIR / 'spatial_distribution.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f'\nSaved -> {out_png.name}')
