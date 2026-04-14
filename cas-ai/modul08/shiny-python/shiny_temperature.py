from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from shiny import App, ui, render, reactive

# this script uses the local file "/data/AWEL_Sensors_LoRa_202501.csv"
# overview here: https://opendata.swiss/de/dataset/lufttemperatur-und-luftfeuchte-lora-sensor-messwerte
# download here: https://opendata.swiss/de/dataset/lufttemperatur-und-luftfeuchte-lora-sensor-messwerte/resource/a086efc4-5329-4646-9282-d49ce790014d

# GET STARTED:
# pip install shiny pandas matplotlib

# =========================
# CONFIG
# =========================
DATA_DIR = "data"

# =========================
# DATA LOADING
# =========================
def load_temperature_data(data_dir):
    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.csv"))

    dfs = []

    usecols = ["starttime", "site", "temperature", "humidity", "masl"]
    dtypes = {
        "site": "category",
        "temperature": "float32",
        "humidity": "float32",
        "masl": "float32",
    }

    for file in files:
        print(f"Loading {file.name}...")

        for chunk in pd.read_csv(
            file,
            sep=";",
            usecols=usecols,
            dtype=dtypes,
            parse_dates=["starttime"],
            chunksize=200_000,
        ):
            dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    return df


def aggregate_data(df, freq):
    df = df.set_index("starttime")

    agg = (
        df.groupby("site")
        .resample(freq)[["temperature", "humidity"]]
        .mean()
        .reset_index()
    )

    return agg


# =========================
# LOAD DATA
# =========================
df_raw = load_temperature_data(DATA_DIR)

df_hourly = aggregate_data(df_raw, "H")
df_daily = aggregate_data(df_raw, "D")

# Feature Engineering
df_daily["ma_7"] = df_daily.groupby("site")["temperature"].transform(
    lambda x: x.rolling(7).mean()
)

df_daily["is_hot"] = df_daily["temperature"] > 30
df_daily["is_frost"] = df_daily["temperature"] < 0

# =========================
# UI
# =========================
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            "sites",
            "Messstationen",
            choices=sorted(df_raw["site"].unique().tolist()),
            multiple=True,
        ),

        ui.input_date_range(
            "date_range",
            "Zeitraum",
            start=str(df_raw["starttime"].min().date()),
            end=str(df_raw["starttime"].max().date()),
        ),

        ui.input_select(
            "aggregation",
            "Aggregation",
            choices={"hourly": "Stündlich", "daily": "Täglich"},
            selected="daily",
        ),

        ui.input_checkbox("show_ma", "Moving Average anzeigen", True),
    ),

    ui.h2("Advanced Temperature & Humidity Dashboard"),

    ui.navset_tab(
        ui.nav_panel("Zeitreihe", ui.output_plot("temp_plot")),
        ui.nav_panel("Tagesprofil", ui.output_plot("day_profile")),
        ui.nav_panel("Temp vs Height", ui.output_plot("height_plot")),
        ui.nav_panel("Temp vs Humidity", ui.output_plot("humidity_plot")),
        ui.nav_panel("Ranking", ui.output_plot("ranking_plot")),
    ),
)

# =========================
# SERVER
# =========================
def server(input, output, session):

    @reactive.calc
    def base_df():
        return df_hourly if input.aggregation() == "hourly" else df_daily

    @reactive.calc
    def filtered_df():
        df = base_df()

        if input.sites():
            df = df[df["site"].isin(input.sites())]

        start, end = input.date_range()
        df = df[
            (df["starttime"] >= pd.to_datetime(start)) &
            (df["starttime"] <= pd.to_datetime(end))
        ]

        return df

    # =========================
    # 1. Zeitreihe
    # =========================
    @output
    @render.plot
    def temp_plot():
        df = filtered_df()

        fig, ax = plt.subplots()

        for site in df["site"].unique():
            d = df[df["site"] == site]
            ax.plot(d["starttime"], d["temperature"], label=site)

            if input.show_ma() and "ma_7" in d:
                ax.plot(d["starttime"], d["ma_7"], linestyle="--")

        ax.legend()
        ax.set_title("Temperatur Zeitreihe")
        return fig

    # =========================
    # 2. Tagesprofil
    # =========================
    @output
    @render.plot
    def day_profile():
        df = filtered_df().copy()
        df["hour"] = df["starttime"].dt.hour

        agg = df.groupby("hour")[["temperature", "humidity"]].mean()

        fig, ax = plt.subplots()
        ax.plot(agg.index, agg["temperature"], label="Temp")
        ax.plot(agg.index, agg["humidity"], label="Humidity")

        ax.set_title("Durchschnittlicher Tagesverlauf")
        ax.legend()
        return fig

    # =========================
    # 3. Temp vs Height
    # =========================
    @output
    @render.plot
    def height_plot():
        df = filtered_df()

        grouped = df.groupby("site").mean(numeric_only=True)

        fig, ax = plt.subplots()
        ax.scatter(grouped["masl"], grouped["temperature"])

        ax.set_xlabel("Höhe (m)")
        ax.set_ylabel("Temperatur")
        ax.set_title("Temperatur vs Höhe")

        return fig

    # =========================
    # 4. Temp vs Humidity
    # =========================
    @output
    @render.plot
    def humidity_plot():
        df = filtered_df()

        fig, ax = plt.subplots()
        ax.scatter(df["temperature"], df["humidity"], alpha=0.3)

        ax.set_xlabel("Temperatur")
        ax.set_ylabel("Humidity")
        ax.set_title("Temp vs Humidity")

        return fig

    # =========================
    # 5. Ranking
    # =========================
    @output
    @render.plot
    def ranking_plot():
        df = filtered_df()

        ranking = (
            df.groupby("site")["temperature"]
            .mean()
            .sort_values()
        )

        n = len(ranking)

        # adjust y height
        fig_height = max(4, n * 1.9)

        fig, ax = plt.subplots(figsize=(15, fig_height))

        ranking.plot(kind="barh", ax=ax)

        ax.set_title("Ø Temperatur pro Station")
        ax.set_xlabel("Temperatur (°C)")

        ## small hacks: somehow the x-axis is still not properly shown.. fix later..
        ax.xaxis.set_major_locator(plt.MaxNLocator(15))
        xmin = ranking.min() - 1
        xmax = ranking.max() + 1
        ax.set_xlim(xmin, xmax)

        plt.tight_layout()

        return fig


# =========================
# APP
# =========================
app = App(app_ui, server)