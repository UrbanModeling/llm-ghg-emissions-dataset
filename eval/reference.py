"""
National reference emissions for validating flagged totals, in Mt/yr on the basis used by owid.record_mt
(CO2; GHG in CO2-eq; CH4 and N2O in CO2-eq with AR6 GWP100 27.9 / 273). All series exclude LULUCF.

Sources (data/external):
  GCB     Global Carbon Budget, fossil CO2 incl. cement; taken from owid-co2-data.csv (column co2)
  PRIMAP  PRIMAP-hist v2.7, scenario HISTCR (country-reported data prioritised), no extrapolation,
          category M.0.EL (national total excluding LULUCF); GHG = KYOTOGHG (AR6GWP100)
  EDGAR   EDGAR_2025_GHG, sheet TOTALS BY COUNTRY (all sectors excluding LULUCF); GHG = GWP_100_AR5_GHG
"""
from pathlib import Path

import pandas as pd

EXT = Path(__file__).resolve().parent.parent / "data" / "external"
PRIMAP = EXT / "primap-hist" / "PRIMAP-hist_v2.7_final_no_extrap.csv"
EDGAR = EXT / "edgar2025"
OWID = EXT / "owid-co2-data.csv"

# record gas -> reference basis
GAS_BASIS = {"CO2": "co2", "carbon (C)": "co2", "GHG (CO2-eq)": "ghg", "CH4": "ch4", "N2O": "n2o"}
GWP = {"ch4": 27.9, "n2o": 273.0}
SOURCES = ["PRIMAP", "EDGAR", "GCB"]


def _gcb() -> pd.DataFrame:
    o = pd.read_csv(OWID, usecols=["iso_code", "year", "co2"])
    o = o[o.iso_code.notna() & ~o.iso_code.str.startswith("OWID", na=False)].dropna(subset=["co2"])
    return pd.DataFrame({"country_iso3": o.iso_code, "year": o.year, "basis": "co2", "ref_mt": o.co2, "source": "GCB"})


def _primap() -> pd.DataFrame:
    p = pd.read_csv(PRIMAP)
    p = p[(p["scenario (PRIMAP-hist)"] == "HISTCR") & (p["category (IPCC2006_PRIMAP)"] == "M.0.EL")]
    entity = {"CO2": "co2", "KYOTOGHG (AR6GWP100)": "ghg", "CH4": "ch4", "N2O": "n2o"}
    p = p[p.entity.isin(entity)]
    years = [c for c in p.columns if c.isdigit()]
    m = p.melt(id_vars=["area (ISO3)", "entity"], value_vars=years, var_name="year", value_name="gg").dropna()
    m["basis"] = m.entity.map(entity)
    m["ref_mt"] = m.gg / 1e3 * m.basis.map(GWP).fillna(1.0)   # Gg -> Mt, CH4/N2O -> CO2-eq
    return pd.DataFrame({"country_iso3": m["area (ISO3)"], "year": m.year.astype(int), "basis": m.basis,
                         "ref_mt": m.ref_mt, "source": "PRIMAP"})


def _edgar() -> pd.DataFrame:
    files = {"co2": "IEA_EDGAR_CO2_1970_2024", "ghg": "EDGAR_AR5_GHG_1970_2024",
             "ch4": "EDGAR_CH4_1970_2024", "n2o": "EDGAR_N2O_1970_2024"}
    out = []
    for basis, f in files.items():
        e = pd.read_excel(EDGAR / f"{f}.xlsx", sheet_name="TOTALS BY COUNTRY", header=9)
        years = [c for c in e.columns if str(c).startswith("Y_")]
        m = e.melt(id_vars=["Country_code_A3"], value_vars=years, var_name="year", value_name="gg").dropna()
        out.append(pd.DataFrame({"country_iso3": m.Country_code_A3, "year": m.year.str[2:].astype(int),
                                 "basis": basis, "ref_mt": m.gg / 1e3 * GWP.get(basis, 1.0), "source": "EDGAR"}))
    return pd.concat(out)


def reference_table() -> pd.DataFrame:
    """Long table: country_iso3, year, basis, source, ref_mt (Mt/yr, CO2-eq for CH4/N2O/GHG)."""
    t = pd.concat([_primap(), _edgar(), _gcb()], ignore_index=True)
    return t[t.ref_mt > 0]
