import os
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL

# ---------------------------------------------------------------------------
# 1. Daten laden (mit lokalem Cache, da Apple die Quelle selbst nicht mehr hostet)
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/ActiveConclusion/"
    "COVID19_mobility/master/apple_reports/applemobilitytrends.csv"
)
DATA_FILE = "applemobilitytrends.csv" #lokal speichern

if not os.path.exists(DATA_FILE):
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

raw = pd.read_csv(DATA_FILE, low_memory=False)

# ---------------------------------------------------------------------------
# 2. FILTER auf CH/DE
# ---------------------------------------------------------------------------
COUNTRIES = ["Germany", "Switzerland"]
MODES = ["driving", "walking", "transit"]

date_cols = [c for c in raw.columns if c.startswith("20")]

subset = raw[
    (raw["geo_type"] == "country/region")
    & (raw["region"].isin(COUNTRIES))
    & (raw["transportation_type"].isin(MODES))
]

long_df = subset.melt(
    id_vars=["region", "transportation_type"],
    value_vars=date_cols,
    var_name="date",
    value_name="index_value",
)
long_df["date"] = pd.to_datetime(long_df["date"])
long_df = long_df.sort_values(["region", "transportation_type", "date"])

# 7-Tage gleitender Durchschnitt pro Land/Modus berechnen
long_df["rolling_7d"] = (
    long_df.groupby(["region", "transportation_type"])["index_value"]
    .transform(lambda s: s.rolling(7, center=True, min_periods=1).mean())
)

# ---------------------------------------------------------------------------
# 3. Bekannte Lockdown-Meilensteine (CH und DE, erste + zweite Welle)
# ---------------------------------------------------------------------------
LOCKDOWN_EVENTS = {
    "DE: Lockdown 1": "2020-03-22",
    "CH: Notlage": "2020-03-16",
    "DE: Lockdown 2": "2020-12-16",
    "CH: Lockdown 2": "2021-01-18",
}

# ---------------------------------------------------------------------------
# 4. Hauptgrafik: Zeitreihen CH vs. DE, je ein Verlauf pro Verkehrsmittel
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(MODES), 1, figsize=(11, 9), sharex=True)

colors = {"Germany": "#1f4e79", "Switzerland": "#c00000"}

for ax, mode in zip(axes, MODES):
    for country in COUNTRIES:
        sel = long_df[
            (long_df["region"] == country) & (long_df["transportation_type"] == mode)
        ]
        ax.plot(sel["date"], sel["rolling_7d"], label=country, color=colors[country])

    ax.axhline(100, color="grey", linestyle=":", linewidth=1)  # Baseline 13.01.2020
    for label, date in LOCKDOWN_EVENTS.items():
        ax.axvline(pd.Timestamp(date), color="black", linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_title(f"Mobility Index - {mode}")
    ax.set_ylabel("Index (Baseline=100)")

axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Datum")
fig.suptitle("Apple Mobility Trends: Deutschland vs. Schweiz (7-Tage-Durchschnitt)", y=1.02)
fig.tight_layout()
fig.savefig("m00_03_mobility_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. STL-Decomposition: Kommt wohl nicht auf die Folie... aber interessant (Beispiel: Driving Schweiz)
# STL = Seasonal-Trend decomposition using Loess --> trennt eine Zeitreihe in drei Abschnitte: Trend, Saisonalität, Rest
# ---------------------------------------------------------------------------
example = long_df[
    (long_df["region"] == "Switzerland") & (long_df["transportation_type"] == "driving")
].set_index("date")["index_value"].asfreq("D").interpolate()

stl_result = STL(example, period=7, robust=True).fit()

fig2 = stl_result.plot()
fig2.set_size_inches(10, 7)
fig2.suptitle("STL-Decomposition: Driving Index Schweiz", y=1.02)
fig2.tight_layout()
fig2.savefig("m08_03_stl_decomposition.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# ---------------------------------------------------------------------------
# 6. Kennzahlen berechnen
# ---------------------------------------------------------------------------
def summary_stats(df: pd.DataFrame, country: str, mode: str) -> dict:
    sel = df[(df["region"] == country) & (df["transportation_type"] == mode)].copy()
    sel = sel[sel["date"] < "2020-07-01"]  # erste Welle
    min_row = sel.loc[sel["rolling_7d"].idxmin()]
    max_drop_pct = 100 - min_row["rolling_7d"]

    after_min = sel[sel["date"] >= min_row["date"]]
    recovery = after_min[after_min["rolling_7d"] >= 100]
    recovery_days = (
        (recovery.iloc[0]["date"] - min_row["date"]).days if not recovery.empty else None
    )

    return {
        "country": country,
        "mode": mode,
        "min_date": min_row["date"].date(),
        "max_drop_pct": round(max_drop_pct, 1),
        "recovery_days": recovery_days,
    }


results = [
    summary_stats(long_df, country, mode)
    for country in COUNTRIES
    for mode in MODES
]

summary_df = pd.DataFrame(results)
summary_df.to_csv("m08_03_summary_stats.csv", index=False)
print(summary_df.to_string(index=False))
