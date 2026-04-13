import pandas as pd
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
# CLEAN FOR TAIPY
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

# Mapping for your specific data
iso2_to_iso3 = {
    "US": "USA", "CA": "CAN", "GB": "GBR", "FR": "FRA", "DE": "DEU",
    "CH": "CHE", "IN": "IND", "AU": "AUS", "BR": "BRA", "CN": "CHN"
}

# ... (Your data loading code remains the same) ...
# For demonstration, I'm assuming 'df' is already loaded and scored

table_data = df.copy().fillna("")
selected_row = table_data.iloc[0]

# 3. CRITICAL: Initialize the DF with the column Taipy is looking for
first_iso3 = iso2_to_iso3.get(selected_row['country'], "USA") # Default to USA or similar
selected_row_df = pd.DataFrame([selected_row])
selected_row_df['country_iso3'] = first_iso3
selected_row_df['fraud_probability'] = float(selected_row['fraud_probability'])

kpi_total = len(table_data)
kpi_fraud = df["is_fraud"].sum()
kpi_avg_risk = round(df["fraud_probability"].mean(), 3)

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

page = """
# Fraud Detection Dashboard

<|layout|columns=1 1 1|
<|{kpi_total}|indicator|value={kpi_total}|title=Total Transactions|>
<|{kpi_fraud}|indicator|value={kpi_fraud}|title=Fraud Detected|>
<|{kpi_avg_risk}|indicator|value={kpi_avg_risk}|title=Avg Risk Score|>
|>

---

<|layout|columns=1 2|
<|part|
## Transactions
<|{table_data}|table|on_action=on_row_select|page_size=10|>
|>

<|part|render={selected_row is not None}|
## Investigation: <|{selected_row['transaction_id'] if selected_row is not None else ''}|>

<|layout|columns=1 1|
<|part|
### Details
- **Amount:** $<|{selected_row['amount'] if selected_row is not None else ''}|>
- **Country:** <|{selected_row['country'] if selected_row is not None else ''}|>
- **Risk Level:** <|{selected_row['risk_level'] if selected_row is not None else ''}|>
- **Probability:** <|{selected_row['fraud_probability'] if selected_row is not None else ''}|>%
|>

<|part|
### Location Analysis
<|{selected_row_df}|chart|type=choropleth|locations=country_iso3|z=fraud_probability|locationmode=ISO-3|zmin=0|zmax=100|options={chart_options}|height=400px|>
|>
|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page).run(
        table_data=table_data,
        kpi_total=kpi_total,
        kpi_fraud=kpi_fraud,
        kpi_avg_risk=kpi_avg_risk,
        selected_row=selected_row,
        selected_row_df=selected_row_df,
        chart_options=chart_options  # Pass it here!
)