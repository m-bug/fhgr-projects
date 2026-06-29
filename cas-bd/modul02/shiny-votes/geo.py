import pandas as pd
import plotly.express as px
from shiny import App, render, ui
from shinywidgets import output_widget, render_widget

CSV_FILE_PATH = "voting-data-2026-06-14/data.csv"

try:
    df = pd.read_csv(CSV_FILE_PATH, sep=";", encoding="utf-8")
except FileNotFoundError:
    raise FileNotFoundError(f"Die Datei '{CSV_FILE_PATH}' wurde nicht gefunden.")

# 1. Nur Gemeinden filtern
df_mun = df[df["region_type"] == "municipality"].copy()

# ➡️ FIX: Daten nach Gemeinde gruppieren, um Duplikate sauber zusammenzurechnen!
# Wir gruppieren nach region_name UND parent_canton_name, damit Gemeinden mit gleichem Namen 
# in verschiedenen Kantonen (z.B. "Buch") nicht fälschlicherweise verschmolzen werden.
df_mun = df_mun.groupby(["region_name", "parent_canton_name"], as_index=False).agg({
    "yes_votes_count": "sum",
    "no_votes_count": "sum",
    "cast_votes_count": "sum",
    "eligible_voters_count": "sum"
})

# ➡️ ERST JETZT: Prozentwerte auf den sauber summierten Gesamtzahlen berechnen
df_mun["Ja-Stimmen %"] = (df_mun["yes_votes_count"] / df_mun["cast_votes_count"] * 100).round(1)
df_mun["Stimmbeteiligung %"] = (df_mun["cast_votes_count"] / df_mun["eligible_voters_count"] * 100).round(1)

cantons = sorted(df_mun["parent_canton_name"].dropna().unique())

app_ui = ui.page_fluid(
    ui.panel_title("Schweizer Abstimmungs-Auswertung (Bereinigt)"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select("canton_select", "1. Kanton filtern:", choices=["Alle Kantone"] + cantons),
            ui.input_radio_buttons("metric_select", "2. Anzuzeigende Kennzahl:",
                                   choices={"Ja-Stimmen %": "Ja-Stimmen-Anteil (%)", "Stimmbeteiligung %": "Stimmbeteiligung (%)"}),
            ui.hr(),
            ui.input_radio_buttons("sort_order", "3. Sortierung:",
                                   choices={"highest": "Höchste Werte zuerst (Top X)", "lowest": "Niedrigste Werte zuerst"}),
            ui.input_slider("top_n", "4. Anzahl Gemeinden anzeigen:", min=5, max=50, value=10, step=1),
            ui.input_checkbox("show_all", "Alle Gemeinden anzeigen (ignoriert Limit)", value=False),
        ),
        ui.card(
            ui.card_header("Resultate im Vergleich"),
            output_widget("vote_plot"),
        ),
    ),
)

def server(input, output, session):

    @render_widget
    def vote_plot():
        filtered_df = df_mun.copy()
        if input.canton_select() != "Alle Kantone":
            filtered_df = filtered_df[filtered_df["parent_canton_name"] == input.canton_select()]

        metric = input.metric_select()
        
        if input.sort_order() == "highest":
            filtered_df = filtered_df.sort_values(by=metric, ascending=True)
        else:
            filtered_df = filtered_df.sort_values(by=metric, ascending=False)

        if not input.show_all():
            filtered_df = filtered_df.tail(input.top_n())

        if metric == "Ja-Stimmen %":
            color_scale = [(0, "#e63946"), (0.5, "#f1faee"), (1, "#457b9d")]
            range_color = [0, 100]
        else:
            color_scale = "Viridis"
            range_color = [0, 100] # Stimmbeteiligung macht mathematisch auch nur Sinn zwischen 0 und 100%

        fig = px.bar(
            filtered_df,
            x=metric,
            y="region_name",
            orientation="h",
            title=f"{metric} – Auswahl der Gemeinden",
            labels={metric: metric, "region_name": "Gemeinde", "parent_canton_name": "Kanton"},
            color=metric,
            color_continuous_scale=color_scale,
            range_color=range_color,
            hover_data={"yes_votes_count": True, "no_votes_count": True, "cast_votes_count": True, "eligible_voters_count": True},
        )

        dynamic_height = max(350, len(filtered_df) * 32)
        fig.update_layout(
            xaxis_title=metric,
            yaxis_title="Gemeinde",
            height=dynamic_height,
            margin=dict(l=150, r=20, t=40, b=40),
        )

        fig.update_yaxes(categoryorder="array", categoryarray=filtered_df["region_name"])

        if metric == "Ja-Stimmen %":
            fig.add_vline(x=50, line_dash="dash", line_color="#1d3557", annotation_text="Mehrheit (50%)", annotation_position="top right")

        return fig

app = App(app_ui, server)