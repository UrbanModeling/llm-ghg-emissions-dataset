"""
Figure 1: the extraction method.
  a) papers / values / records kept at each step of the full run
  b) worked example (uid 7467): abstract -> Step 2 (LLM) -> Step 3 (LLM) -> Step 4 (code) -> dataset records
  c) Step 4 rules: how places are resolved and what the checks against the text change
  d) accuracy on the random 100-paper reference set (eval/gold/, 95% Wilson intervals)
Counts and the example are read from the pipeline outputs (output/full/steps.csv, eval/scores/full_summary.csv).

Output: figures/framework_workflow.png

Usage:
    python plots/plot_method.py
"""
import json
import sys
from math import sqrt
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "output"
FIG_DIR = BASE / "figures"
EXAMPLE_UID = 7467
sys.path.insert(0, str(BASE / "pipeline"))
from build import POPULATION  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 7,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
# field kinds (colours from the other figures: sector figure palette, temporal figure reds)
CHECK, COPY, ENUM, WORDS = "#9FCBEA", "#F4A9A8", "#F2E5A0", "#A9CDA9"
CODE, LLM, FINAL, BAD = "#EFEFEF", "#F4A9A8", "#9B1C1C", "#9B1C1C"
EDGE, INK, INK_2 = "#444444", "#111111", "#555555"
MONO = "DejaVu Sans Mono"


def rd(p, **k):
    return pd.read_csv(p, dtype=str, encoding="utf-8-sig", **k).fillna("")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


# ── helpers ─────────────────────────────────────────────────────────────────────

def rich_text(ax, x, y, width, segments, size=6.6, line_h=0.46):
    """Wrap (text, highlight colour or None) segments into `width` data units, highlighting spans."""
    r = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    cx, cy = x, y
    space = None
    for text, col in segments:
        for word in text.split(" "):
            if word == "":
                continue
            t = ax.text(cx, cy, word, fontsize=size, va="baseline", ha="left", color=INK, zorder=3)
            bb = t.get_window_extent(r)
            (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            if space is None:                                   # width of one space, measured once
                sp, one = ax.text(0, 0, "a a", fontsize=size), ax.text(0, 0, "aa", fontsize=size)
                space = (inv.transform([(sp.get_window_extent(r).x1, 0)])[0][0]
                         - inv.transform([(one.get_window_extent(r).x1, 0)])[0][0])
                sp.remove()
                one.remove()
            w = x1 - x0
            if cx + w > x + width and cx > x:
                cx, cy = x, cy - line_h
                t.set_position((cx, cy))
            if col:
                ax.add_patch(Rectangle((cx - 0.02, cy - 0.12), w + 0.04, 0.4, facecolor=col, edgecolor="none",
                                       zorder=2))
            cx += w + space
    return cy


def card(ax, x, y, w, h, title, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", facecolor=fill,
                                edgecolor=EDGE, linewidth=0.7, zorder=1))
    ax.text(x + 0.12, y + h - 0.12, title, ha="left", va="top", fontsize=7.2, fontweight="bold", color=INK, zorder=3)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=EDGE, lw=0.9, zorder=4))


def panel(ax, letter, title):
    ax.text(0, 1.0, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")
    ax.text(0.035, 1.0, title, transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left")


# ── panels ──────────────────────────────────────────────────────────────────────

def panel_a(ax):
    n = rd(OUT / "full" / "steps.csv").set_index("step")["values"].astype(int)
    rec = rd(OUT / "full" / "emission_records.csv", usecols=["uid"])
    stages = [
        ("Web of Science\nabstracts", len(rd(OUT / "prepared.csv", usecols=["uid"])), "papers", CODE),
        ("Step 2 (LLM):\nvalues extracted", n["extracted"], "values", LLM),
        ("Step 3 (LLM): emissions,\nnumbers in the text", n["an emission (Step 3)"], "values", LLM),
        ("Greenhouse gas, quantity\nand unit; not a year", n["not money, area, energy, people"], "values", CODE),
        ("With a country", n["has a country"], "records", CODE),
        ("Value parsed:\ndataset", len(rec), f"records\n{rec.uid.nunique():,} papers", FINAL),
    ]
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (name, cnt, unit, fill) in enumerate(stages):
        tip = 0.12
        pts = [(i + 0.02, 0.28), (i + 0.98 - tip, 0.28), (i + 0.98, 0.58), (i + 0.98 - tip, 0.88), (i + 0.02, 0.88)]
        if i:
            pts.append((i + 0.02 + tip, 0.58))
        ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=EDGE, lw=0.7))
        white = fill == FINAL
        dx = 0 if i == 0 else 0.03
        ax.text(i + 0.5 + dx, 0.66, f"{cnt:,}", ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white" if white else INK)
        ax.text(i + 0.5 + dx, 0.43, unit.split("\n")[0], ha="center", va="center", fontsize=6.5,
                color="white" if white else INK_2)
        ax.text(i + 0.47, 0.2, name + ("\n" + unit.split("\n")[1] if "\n" in unit else ""), ha="center",
                va="top", fontsize=6.3, color=INK_2, linespacing=1.2)
    panel(ax, "a", "Steps and what each keeps (full run)")


def panel_b(ax):
    ext = next(d for d in json.loads((OUT / "full" / "extracted.json").read_text("utf-8")) if d["uid"] == EXAMPLE_UID)
    ver = next(d for d in json.loads((OUT / "full" / "verified.json").read_text("utf-8")) if d["uid"] == EXAMPLE_UID)
    vals = ext["response"]["values"]
    chk = ver["response"] if isinstance(ver["response"], list) else ver["response"]["checks"]
    v1, v2, v3, v4 = vals
    rec = rd(OUT / "full" / "emission_records.csv")
    rec = rec[rec.uid == str(EXAMPLE_UID)]
    total = rec[rec.is_total == "True"].iloc[0]
    per_cap = float(total.value_num) * 1e6 / POPULATION[total.country_iso3]

    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 10)
    ax.axis("off")
    top, h = 2.55, 7.3

    # abstract
    card(ax, 0, top, 2.55, h, f"Abstract (uid {EXAMPLE_UID}, excerpt)")
    segs = [("… South Korea must reduce its gross CO2 emission level from its", None), ("2015", COPY),
            ("level of", None), ("680 Mt.", COPY),
            ("… This study calculated the level of CO2 emissions by agricultural tractors, heaters, and dryers "
             "which are accounted for consuming", None), ("88.6%", COPY),
            ("of the tax-free fuels allocated for agricultural use … The total annual amount of CO2 emissions "
             "produced by these agricultural machines was estimated to be", None), ("4,334.2 kt", COPY),
            ("which is", None), ("less than 1%", COPY), ("of the gross CO2 emissions in South Korea in", None),
            ("2015.", COPY)]
    rich_text(ax, 0.12, 9.25, 2.35, segs, size=6.2, line_h=0.43)
    ax.text(0.12, 2.75, "pink: values and years\nthe model copied", fontsize=5.8, color=INK_2, style="italic",
            va="bottom", linespacing=1.3)

    # Step 2
    card(ax, 2.75, top, 3.05, h, "Step 2: extraction (LLM)", fill="#FBFBFB")
    lines = [
        (None, "values: [ {"),
        (COPY, f'  value_text: "{v1["value_text"]}"  unit: "{v1["unit"]}"'),
        (ENUM, f'  quantity_type: {v1["quantity_type"]}  gas: {v1["gas"]}'),
        (WORDS, f'  country: "{v1["country"]}"'),
        (WORDS, f'  year: "{v1["year"]}"  sector: "{v1["sector"]}"'),
        (None, "}, {"),
        (COPY, f'  value_text: "{v2["value_text"]}"'),
        (ENUM, f'  quantity_type: {v2["quantity_type"]}  gas: {v2["gas"]}'),
        (WORDS, f'  sector: "{v2["sector"]}"'),
        (None, "}, {"),
        (COPY, f'  value_text: "{v3["value_text"]}"  unit: "{v3["unit"]}"'),
        (ENUM, f'  quantity_type: {v3["quantity_type"]}  gas: {v3["gas"]}'),
        (WORDS, f'  year: "{v3["year"]}"'),
        (WORDS, f'  sector: "{v3["sector"]}"'),
        (None, "}, {"),
        (COPY, f'  value_text: "{v4["value_text"]}"  …'),
        (None, "} ]   + evidence sentence per value"),
    ]
    y = 9.25
    for col, s in lines:
        if col:
            ax.add_patch(Rectangle((2.83, y - 0.11), 2.89, 0.29, facecolor=col, edgecolor="none", zorder=2))
        ax.text(2.87, y, s, fontsize=5.1, family=MONO, va="baseline", color=INK, zorder=3)
        y -= 0.33
    for i, (col, lab) in enumerate([(COPY, "copied verbatim"), (ENUM, "fixed list"), (WORDS, "model's words")]):
        yy = 3.4 - i * 0.28
        ax.add_patch(Rectangle((2.87, yy - 0.08), 0.22, 0.2, facecolor=col, edgecolor="none"))
        ax.text(3.17, yy, lab, fontsize=5.8, va="center", color=INK_2)

    # Step 3
    card(ax, 6.0, top, 1.95, h, "Step 3: review (LLM)", fill="#FBFBFB")
    ax.text(6.12, 9.25, "text + numbered values →\nper value, from fixed lists:", fontsize=5.6, color=INK_2,
            va="top", linespacing=1.25)
    y = 8.55
    for i, (v, c) in enumerate(zip(vals, chk), 1):
        ax.text(6.12, y, f'{i}. {v["value_text"]}', fontsize=5.8, fontweight="bold", color=INK, va="baseline")
        for s in [f'emission: {c["is_emission"]}', f'coverage: {c["coverage"]}', f'basis: {c["basis"]}',
                  f'group: {c["group"]}']:
            y -= 0.27
            ax.add_patch(Rectangle((6.2, y - 0.09), 1.66, 0.25, facecolor=ENUM, edgecolor="none", zorder=2))
            ax.text(6.24, y, s, fontsize=5.1, family=MONO, va="baseline", color=INK, zorder=3)
        y -= 0.36

    # Step 4
    card(ax, 8.15, top, 2.25, h, "Step 4: checks (code)", fill=CODE)
    ok, no, to = "✓", "✗", "→"
    checks = [
        (ok, "values and years found in text"),
        (to, f'"{v1["country"]}" → {total.country_iso3}, country'),
        (no, f'{v2["value_text"]}: not an emission'),
        ("", "   (Step 3) → dropped"),
        (ok, f'{v1["value_text"]}: total (Step 3) and'),
        ("", f'   "{v1["sector"]}" (Step 2);'),
        ("", f"   {per_cap:.1f} t CO2 per person"),
        ("", "   → is_total"),
        (no, f'{v3["value_text"]}: sector value,'),
        ("", "   not a total"),
        (to, f'"{v4["value_text"]}" → 1, qualifier <'),
        (to, "group → sector group"),
    ]
    y = 9.2
    for sym, s in checks:
        col = "#2E7D32" if sym == ok else BAD if sym == no else INK_2
        ax.text(8.25, y, sym, fontsize=6.8, color=col, fontweight="bold", va="baseline", family="DejaVu Sans")
        ax.text(8.45, y, s, fontsize=5.8, color=INK, va="baseline")
        y -= 0.42

    for x0, x1 in [(2.57, 2.73), (5.82, 5.98), (7.97, 8.13)]:
        arrow(ax, (x0, 6.2), (x1, 6.2))

    # records
    arrow(ax, (9.3, 2.53), (9.3, 2.17))
    cols = [("value_num", "value_num"), ("value_qualifier", "qual."), ("unit", "unit"), ("quantity_type", "quantity"),
            ("gas", "gas"), ("country", "country"), ("spatial_level", "level"), ("year", "year"),
            ("sector_group", "sector"), ("is_total", "is_total")]
    xs = [0.15, 1.1, 1.75, 2.45, 3.35, 4.05, 5.2, 6.0, 6.75, 7.9]
    ax.add_patch(FancyBboxPatch((0, 0.05), 10.4, 2.08, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=FINAL, edgecolor=EDGE, lw=0.7))
    ax.text(0.12, 1.95, f"Dataset records (emission_records.csv, {len(rec)} from this paper; plus evidence, Step 3 "
                        "answers and check flags)", fontsize=6.5, color="white", va="top", fontweight="bold")
    for x, (_, lab) in zip(xs, cols):
        ax.text(x, 1.45, lab, fontsize=6, color="#F8D7D7", va="center")
    for j, (_, row) in enumerate(rec.iterrows()):
        for x, (c, _) in zip(xs, cols):
            v = f"{float(row[c]):g}" if c == "value_num" else row[c]
            ax.text(x, 1.05 - j * 0.36, v, fontsize=6.2, color="white", va="center")
    panel(ax, "b", "Worked example")


def panel_c(ax):
    rec = rd(OUT / "full" / "emission_records.csv",
             usecols=["place_how", "spatial_level", "chk_year", "chk_place", "chk_magnitude"])
    how = rec.place_how.value_counts()
    rules = [
        ("Places (first rule that applies)", None, None, None),
        ("Country only", "no place within the country", int((rec.place_how == "").sum()), False),
        ("Gazetteer name", "“Seoul” → Seoul", int(how.get("exact", 0)), False),
        ("Longest name inside", "“Mato Grosso (MT) state” → Mato Grosso", int(how.get("contained", 0)), False),
        ("Kept as written (local)", "“Yangtze River Delta”", int(how.get("kept", 0)), False),
        ("Checks", None, None, None),
        ("Year not in the text (Step 1)", "year cleared", int((rec.chk_year == "False").sum()), True),
        ("Place not in the text (Step 1)", "widened to the country", int((rec.chk_place == "False").sum()), True),
        ("Total outside 0.05–60 t per person", "not a total", int((rec.chk_magnitude == "False").sum()), True),
    ]
    n = len(rules)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, n + 0.2)
    ax.axis("off")
    mx = max(r[2] for r in rules if r[2] is not None)
    for i, (name, ex, cnt, changed) in enumerate(rules):
        y = n - 1 - i
        if cnt is None:
            ax.text(0, y, name, fontsize=6.8, fontweight="bold", color=INK_2, va="center")
            continue
        ax.text(0.2, y + 0.18, name, fontsize=6.6, fontweight="bold", color=INK, va="center")
        ax.text(0.35, y - 0.2, ex, fontsize=6, color=INK_2, va="center")
        bw = 3.2 * cnt / mx
        ax.add_patch(Rectangle((6.3, y - 0.28), max(bw, 0.03), 0.56, facecolor=LLM if changed else "#BDBDBD",
                               edgecolor=EDGE, lw=0.5))
        ax.text(6.3 + max(bw, 0.03) + 0.08, y, f"{cnt:,}", fontsize=6.3, va="center", color=INK)
    ax.text(6.3, n - 0.35, "records", fontsize=6, color=INK_2)
    panel(ax, "c", "Rules in code (Step 4)")


def panel_d(ax):
    s = pd.read_csv(BASE / "eval" / "scores" / "full_summary.csv").set_index("metric")
    rows = [("Value", "value"), ("Location", "location"), ("Year", "year"), ("Gas", "gas"),
            ("Quantity type", "quantity_type"), ("Sector", "sector"), ("Total", "total"), ("Recall¹", "recall")]
    est = [wilson(int(s.loc[m, "k"]), int(s.loc[m, "n"])) for _, m in rows][::-1]
    labels = [r[0] for r in rows][::-1]
    pct, lo, hi = zip(*est)
    y = range(len(rows))
    ax.barh(y, pct, color=["#9B1C1C"] + ["#F4A9A8"] * (len(rows) - 1), edgecolor=EDGE, lw=0.5, height=0.6)
    ax.errorbar(pct, y, xerr=[[p - l for p, l in zip(pct, lo)], [h - p for p, h in zip(pct, hi)]], fmt="none",
                ecolor=INK, elinewidth=0.7, capsize=1.5)
    for yy, p in zip(y, pct):
        ax.text(2, yy, f"{p:.1f}", va="center", fontsize=6.3, color="white" if yy == 0 else INK)
    ax.set_yticks(list(y), labels, fontsize=6.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("%  (95% CI)", fontsize=6.6)
    ax.tick_params(labelsize=6.3, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0, -0.2, f"{int(s.loc['value', 'n'])} matched records, 100 random papers.\n¹ reference "
                     "emission values with a location that were found.", transform=ax.transAxes, fontsize=5.8,
            color=INK_2, va="top")
    ax.text(-0.42, 1.03, "d", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.text(-0.33, 1.03, "Accuracy on the reference set", transform=ax.transAxes, fontsize=8.5, va="bottom")


def main():
    fig = plt.figure(figsize=(7.2, 9.6))
    ax_a = fig.add_axes([0.02, 0.885, 0.96, 0.085])
    ax_b = fig.add_axes([0.02, 0.37, 0.96, 0.47])
    ax_c = fig.add_axes([0.02, 0.03, 0.55, 0.29])
    ax_d = fig.add_axes([0.74, 0.08, 0.24, 0.235])
    fig.canvas.draw()
    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    panel_d(ax_d)
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "framework_workflow.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved -> {out.name}")


if __name__ == "__main__":
    main()
