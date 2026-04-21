import pandas as pd
import plotly.graph_objects as go
from taipy import Gui

from fraud_scoring import score_transactions

# pip install taipy pandas scikit-learn joblib numpy

# python app.py

# =========================
# CONFIG
# =========================
DATA_DIR = "data"

# =========================
# LOAD DATA + SCORE
# =========================
df_raw = pd.read_csv(DATA_DIR + "/synthetic_fraud_transactions.csv")
df = score_transactions(df_raw)

# fix risk level column
df["risk_level"] = df["risk_level"].astype(str)

# -------------------------
# MAP Integration
# source: https://taipy-designer.readthedocs.io/wdg/wdg-geo-time/
# -------------------------
# Move the dictionary out of the Markdown to avoid the SyntaxError
chart_options = {
    "colorscale": "Reds",
        "geo": {
            "showframe": False,
            "showcoastlines": True,
            "projection": {"type": "equirectangular"}
    }
}

# Mapping for specific synthetic transactions with iso3: only include countries that are in the data
iso2_to_iso3 = {
    "US": "USA", "CA": "CAN", "GB": "GBR", "FR": "FRA", "DE": "DEU",
    "CH": "CHE", "IN": "IND", "AU": "AUS", "BR": "BRA", "CN": "CHN"
}

table_data = df.copy().fillna("")
selected_row = table_data.iloc[0]

# 3. CRITICAL: Initialize the DF with the column Taipy is looking for
# small hack: when init the GUI user did not click an a row, so default to USA or similar, could be improved but works :)
first_iso3 = iso2_to_iso3.get(selected_row['country'], "USA") # Default to USA or similar
selected_row_df = pd.DataFrame([selected_row])
selected_row_df['country_iso3'] = first_iso3
selected_row_df['fraud_probability'] = float(selected_row['fraud_probability'])


# -------------------------
# KPIs
# -------------------------
kpi_total = len(table_data)
kpi_fraud = df["is_fraud"].sum()
kpi_avg_risk = round(df["fraud_probability"].mean(), 3)
kpi_fraud_rate = round((kpi_fraud / kpi_total) * 100, 2) if kpi_total else 0.0
kpi_total_text = f"{kpi_total:,}"
kpi_fraud_text = f"{int(kpi_fraud):,}"
kpi_avg_risk_text = f"{kpi_avg_risk:.3f}"
kpi_fraud_rate_text = f"{kpi_fraud_rate:.2f}%"


# -------------------------
# Sankey Diagram Functions
# source: https://plotly.com/python/sankey-diagram/
# -------------------------
def build_sankey_figure(sankey_mode, country_hits_only=False):
    if sankey_mode == "country":
        country_source = df
        if country_hits_only:
            country_source = df[df["is_fraud"] == 1]
        country_counts = country_source["country"].astype(str).value_counts()
        if country_counts.empty:
            labels = ["No hits found"]
            values = [0]
            title = "Transaction Flow by Country (Hits Only)"
            all_labels = ["All Transactions"] + labels
            sources = [0]
            targets = [1]
            link_colors = ["rgba(199,139,209,0.84)"]
            fig = go.Figure(
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=16,
                        line=dict(color="rgba(148,163,184,0.4)", width=1),
                        label=all_labels,
                        color=["#a4b1f9"] + ["#cbb5ef"] * len(labels),
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                    ),
                )
            )
            fig.update_layout(
                title=title,
                font=dict(color="#e2e8f0", size=12),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                margin=dict(l=10, r=10, t=35, b=10),
            )
            return fig
        top_country_counts = country_counts.head(8)
        other_count = int(country_counts.iloc[8:].sum())
        labels = list(top_country_counts.index)
        values = list(top_country_counts.values)
        if other_count > 0:
            labels.append("Other Countries")
            values.append(other_count)
        title = "Transaction Flow by Country (Hits Only)" if country_hits_only else "Transaction Flow by Country"
    elif sankey_mode == "risk":
        risk_source = df
        if country_hits_only:
            risk_source = df[df["is_fraud"] == 1]
        risk_counts = risk_source["risk_level"].astype(str).value_counts()
        labels = list(risk_counts.index)
        values = list(risk_counts.values)
        title = "Transaction Flow by Risk Level (Hits Only)" if country_hits_only else "Transaction Flow by Risk Level"
    else:
        hit_count = int(df["is_fraud"].sum())
        no_hit_count = int(len(df) - hit_count)
        labels = ["Hits", "Not Hits"]
        values = [hit_count, no_hit_count]
        title = "Transaction Flow: Hits vs Not Hits"

    all_labels = ["All Transactions"] + labels
    sources = [0] * len(labels)
    targets = list(range(1, len(labels) + 1))
    # indigo/lilac palette, could be improved..
    link_colors = [
        "rgba(199,139,209,0.84)" if "Hit" in label or "high" in label.lower()
        else "rgba(133,137,249,0.80)"
        for label in labels
    ]

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=16,
                line=dict(color="rgba(148,163,184,0.4)", width=1),
                label=all_labels,
                color=["#a4b1f9"] + ["#cbb5ef"] * len(labels),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            ),
        )
    )
    fig.update_layout(
        title=title,
        font=dict(color="#e2e8f0", size=12),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=10, r=10, t=35, b=10),
    )
    return fig

# -------------------------
# Sankey Diagram Initial State
# ------------------------- 
sankey_mode = "hits" #default
sankey_mode_text = "Hits / Not Hits"
country_hits_only = False
sankey_fig = build_sankey_figure(sankey_mode, country_hits_only)


# -------------------------
# Sankey Diagram State Setter
# -------------------------
def set_sankey_mode(state, mode):
    if mode not in {"hits", "country", "risk"}:
        return
    state.sankey_mode = mode
    if mode == "country":
        state.sankey_mode_text = "Country"
    elif mode == "risk":
        state.sankey_mode_text = "Risk Level"
    else:
        state.sankey_mode_text = "Hits / Not Hits"
    state.sankey_fig = build_sankey_figure(mode, state.country_hits_only)


# -------------------------
# Country Hits Only
# -------------------------
def on_country_hits_only_change(state, var_name, value):
    state.country_hits_only = bool(value)
    if state.sankey_mode in {"country", "risk"}:
        state.sankey_fig = build_sankey_figure(state.sankey_mode, state.country_hits_only)

# Setters for sankey buttons
def on_sankey_hits(state, var_name, payload):
    set_sankey_mode(state, "hits")


def on_sankey_country(state, var_name, payload):
    set_sankey_mode(state, "country")


def on_sankey_risk(state, var_name, payload):
    set_sankey_mode(state, "risk")


def format_selected_row(row):
    if row is None:
        return "", "", "", "", ""
    return (
        str(row.get("transaction_id", "")),
        f"${float(row.get('amount', 0)):,.2f}",
        str(row.get("country", "")),
        str(row.get("risk_level", "")),
        f"{float(row.get('fraud_probability', 0)):.2f}%",
    )


(
    selected_tx_id,
    selected_amount_text,
    selected_country_text,
    selected_risk_text,
    selected_probability_text,
) = format_selected_row(selected_row)

def on_row_select(state, var_name, payload):
    idx = payload.get('index')
    if idx is not None:
        row = state.table_data.iloc[idx].copy()
        state.selected_row = row
        
        # Convert ISO-2 to ISO-3
        iso2 = row['country']
        iso3 = iso2_to_iso3.get(iso2, iso2) # Fallback to original if not found
        
        # Create the DF for the chart
        new_df = pd.DataFrame([row])
        new_df['country_iso3'] = iso3
        new_df['fraud_probability'] = float(row['fraud_probability'])
        
        state.selected_row_df = new_df
        (
            state.selected_tx_id,
            state.selected_amount_text,
            state.selected_country_text,
            state.selected_risk_text,
            state.selected_probability_text,
        ) = format_selected_row(row)

# chatgpt was a big help with the formatting of the containers using css
# otherwise documented here: https://docs.taipy.io/en/release-4.0/userman/gui/styling/
page = """
<style>
.hero {
  background: linear-gradient(120deg, #1e3a8a 0%, #991b1b 100%);
  color: white;
  padding: 1rem 1.2rem;
  border-radius: 12px;
  margin-bottom: 1rem;
}
.subtitle {
  color: #334155;
  margin: 0.3rem 0 0.9rem 0;
}
.table-panel {
  background: #0f172a;
  color: #f8fafc;
  border: 1px solid #1e293b;
  border-radius: 12px;
  padding: 1rem;
}
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(120px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.kpi-card {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 12px;
  padding: 0.75rem 0.85rem;
}
.kpi-label {
  color: #93c5fd;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.kpi-value {
  color: #f8fafc;
  font-size: 1.2rem;
  font-weight: 700;
  margin-top: 0.1rem;
}
.sankey-left {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 12px;
  padding: 1rem;
}
.sankey-counter-label {
  color: #93c5fd;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.sankey-counter-value {
  color: #f8fafc;
  font-size: 2rem;
  font-weight: 700;
}
.sankey-right {
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 12px;
  padding: 1rem;
}
.details-panel {
  background: #0f172a;
  color: #f8fafc;
  border-radius: 12px;
  padding: 1rem;
}
.details-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.7rem;
  margin: 0.8rem 0 1rem 0;
}
.detail-card {
  background: #1e293b;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 0.65rem 0.75rem;
}
.detail-label {
  color: #93c5fd;
  font-size: 0.8rem;
  margin-bottom: 0.15rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.detail-value {
  font-size: 1rem;
  font-weight: 600;
  color: #f8fafc;
}
.risk-pill {
  display: inline-block;
  background: #7f1d1d;
  color: #fee2e2;
  border: 1px solid #ef4444;
  border-radius: 999px;
  padding: 0.15rem 0.55rem;
  font-size: 0.8rem;
  margin-left: 0.45rem;
}
</style>

<|part|class_name=hero|
## Fraud Detection Dashboard
Real-time view of suspicious transactions and risk patterns.
|>

<|part|class_name=subtitle|
Select a transaction to inspect risk details and location intelligence.
|>

<|part|class_name=kpi-grid|
<|part|class_name=kpi-card|
<|Total Transactions|text|class_name=kpi-label|>
<|{kpi_total_text}|text|class_name=kpi-value|>
|>
<|part|class_name=kpi-card|
<|Fraud Detected|text|class_name=kpi-label|>
<|{kpi_fraud_text}|text|class_name=kpi-value|>
|>
<|part|class_name=kpi-card|
<|Average Risk Score|text|class_name=kpi-label|>
<|{kpi_avg_risk_text}|text|class_name=kpi-value|>
|>
<|part|class_name=kpi-card|
<|Fraud Rate|text|class_name=kpi-label|>
<|{kpi_fraud_rate_text}|text|class_name=kpi-value|>
|>
|>

<|layout|columns=3 2|
<|part|class_name=table-panel|
## Transactions
<|{table_data}|table|on_action=on_row_select|page_size=14|height=70vh|>
|>

<|part|class_name=details-panel|render={selected_row is not None}|
## Investigation: <|{selected_tx_id}|>
Click any row in the table to refresh this panel.

<|part|class_name=details-grid|
<|part|class_name=detail-card|
<|Amount|text|class_name=detail-label|>
<|{selected_amount_text}|text|class_name=detail-value|>
|>
<|part|class_name=detail-card|
<|Country|text|class_name=detail-label|>
<|{selected_country_text}|text|class_name=detail-value|>
|>
<|part|class_name=detail-card|
<|Risk Level|text|class_name=detail-label|>
<|{selected_risk_text}|text|class_name=detail-value|>
<|Active Case|text|class_name=risk-pill|>
|>
<|part|class_name=detail-card|
<|Fraud Probability|text|class_name=detail-label|>
<|{selected_probability_text}|text|class_name=detail-value|>
|>
|>

### Location Analysis
<|{selected_row_df}|chart|type=choropleth|locations=country_iso3|z=fraud_probability|locationmode=ISO-3|zmin=0|zmax=100|options={chart_options}|height=400px|>
|>
|>

<|layout|columns=1 4|
<|part|class_name=sankey-left|
### Transaction Flow
<|Total Transactions|text|class_name=sankey-counter-label|>
<|{kpi_total_text}|text|class_name=sankey-counter-value|>

Flow mode: **<|{sankey_mode_text}|>**
|>

<|part|class_name=sankey-right|
### Sankey Analysis
<|layout|columns=1 1 1|
<|Hits / Not Hits|button|on_action=on_sankey_hits|>
<|Country|button|on_action=on_sankey_country|>
<|Risk Level|button|on_action=on_sankey_risk|>
|>
<|{country_hits_only}|toggle|label=Show hits only|on_change=on_country_hits_only_change|render={sankey_mode in ["country", "risk"]}|>
<|chart|figure={sankey_fig}|height=420px|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page).run(
        table_data=table_data,
        kpi_total=kpi_total,
        kpi_fraud=kpi_fraud,
        kpi_avg_risk=kpi_avg_risk,
        kpi_fraud_rate=kpi_fraud_rate,
        kpi_total_text=kpi_total_text,
        kpi_fraud_text=kpi_fraud_text,
        kpi_avg_risk_text=kpi_avg_risk_text,
        kpi_fraud_rate_text=kpi_fraud_rate_text,
        sankey_mode=sankey_mode,
        sankey_mode_text=sankey_mode_text,
        country_hits_only=country_hits_only,
        sankey_fig=sankey_fig,
        selected_row=selected_row,
        selected_tx_id=selected_tx_id,
        selected_amount_text=selected_amount_text,
        selected_country_text=selected_country_text,
        selected_risk_text=selected_risk_text,
        selected_probability_text=selected_probability_text,
        selected_row_df=selected_row_df,
        chart_options=chart_options
)