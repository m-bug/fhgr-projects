import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
import json
import os

st.set_page_config(layout="wide")
st.title("🗺️ SBB Abonnement-Dichte in der Schweiz")

@st.cache_data
def load_and_process_data(ausgewaehltes_jahr):
    gpkg_path = "swissBOUNDARIES3D_1_5_LV95_LN02.gpkg"
    gdf_gemeinden = None
    
    # 1. Versuche, die lokalen swisstopo-Daten zu laden
    if os.path.exists(gpkg_path):
        try:
            gdf_gemeinden = gpd.read_file(gpkg_path, layer="TLM_HOHEITSGEBIET")
            gdf_gemeinden.columns = gdf_gemeinden.columns.str.upper()
            
            if "GEOMETRY" in gdf_gemeinden.columns:
                gdf_gemeinden = gdf_gemeinden.set_geometry("GEOMETRY")
            gdf_gemeinden["GEOMETRY"] = gdf_gemeinden["GEOMETRY"].force_2d()
            gdf_gemeinden = gdf_gemeinden.to_crs(epsg=4326)
            
            if "NAME" not in gdf_gemeinden.columns:
                possible_names = [c for c in gdf_gemeinden.columns if "NAME" in c or "GMD" in c]
                if possible_names:
                    gdf_gemeinden = gdf_gemeinden.rename(columns={possible_names[0]: "NAME"})
        except Exception as e:
            st.sidebar.warning(f"Lokales GPKG konnte nicht voll gelesen werden: {e}")
            gdf_gemeinden = None

    # Fallback Online-Karte
    if gdf_gemeinden is None or len(gdf_gemeinden) == 0:
        st.sidebar.info("Nutze Online-Ersatzkarte (ch-boundaries)...")
        url = "https://raw.githubusercontent.com/stefanolderog/geopandas-swiss-boundaries/master/data/GEN_A4_GEMEINDEN_2019.geojson"
        gdf_gemeinden = gpd.read_file(url)
        gdf_gemeinden.columns = gdf_gemeinden.columns.str.upper()
        gdf_gemeinden = gdf_gemeinden.rename(columns={"GMDNAME": "NAME", "GMDNR": "BFS_NUMMER"})

    # BFS-Nummer vereinheitlichen
    if "BFS_NUMMER" not in gdf_gemeinden.columns:
        for alt in ["GMDNR", "BFS_ID", "GMD_NR", "BFS"]:
            if alt in gdf_gemeinden.columns:
                gdf_gemeinden = gdf_gemeinden.rename(columns={alt: "BFS_NUMMER"})
                break
                
    gdf_gemeinden["BFS_NUMMER"] = pd.to_numeric(gdf_gemeinden["BFS_NUMMER"], errors='coerce').fillna(0).astype(int).astype(str)

    # 2. SBB-Daten laden
    df_sbb = pd.read_csv("generalabo-halbtax.csv", sep=";")
    df_sbb.columns = df_sbb.columns.str.upper()
    df_sbb = df_sbb[df_sbb["JAHR"] == ausgewaehltes_jahr]
    
    sbb_bfs_col = None
    for col in df_sbb.columns:
        if "BFS" in col or "GMD" in col:
            sbb_bfs_col = col
            break
            
    if sbb_bfs_col:
        df_sbb = df_sbb.rename(columns={sbb_bfs_col: "BFS_NUMMER"})
    else:
        df_sbb["BFS_NUMMER"] = df_sbb["PLZ"]

    df_sbb["BFS_NUMMER"] = pd.to_numeric(df_sbb["BFS_NUMMER"], errors='coerce').fillna(0).astype(int).astype(str)

    # Abos aggregieren
    df_sbb_grouped = df_sbb.groupby("BFS_NUMMER")[["GENERALABONNEMENT", "HALBTAXABONNEMENT"]].sum().reset_index()

    # 4. Zusammenführen (how="left", damit alle 2136 Gemeinden auf der Karte bleiben)
    merged = gdf_gemeinden.merge(df_sbb_grouped, on="BFS_NUMMER", how="left")
    merged["GENERALABONNEMENT"] = merged["GENERALABONNEMENT"].fillna(0)
    merged["HALBTAXABONNEMENT"] = merged["HALBTAXABONNEMENT"].fillna(0)
    
    # --- FEHLERBEHEBUNG FÜR DATUMSOBJEKTE (TIMESTAMP FIX) ---
    # Wir wandeln jede Spalte, die Datumsangaben enthält, in Text (String) um.
    for col in merged.columns:
        if pd.api.types.is_datetime64_any_dtype(merged[col]) or merged[col].dtype == "object":
            try:
                # Prüfen, ob Timestamps drinstecken und konvertieren
                if merged[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                    merged[col] = merged[col].astype(str)
            except:
                pass
                
    return merged

# --- Streamlit Steuerung ---
st.sidebar.header("Filter-Optionen")
jahr = st.sidebar.selectbox("Jahr auswählen", options=[2021, 2022, 2023, 2024])
abo_typ = st.sidebar.selectbox(
    "Welches Abonnement möchtest du anzeigen?",
    options=["GENERALABONNEMENT", "HALBTAXABONNEMENT"]
)

# Daten laden
df_map = load_and_process_data(jahr)

# Kontroll-Anzeige
st.success(f"Geometrien geladen: {len(df_map)} Gemeinden auf der Karte.")
aktive_gemeinden = len(df_map[df_map["GENERALABONNEMENT"] + df_map["HALBTAXABONNEMENT"] > 0])
st.info(f"Davon erfolgreich mit SBB-Daten gefärbt: {aktive_gemeinden}")

# GeoJSON extrahieren (jetzt ohne Timestamp-Absturz!)
geojson_data = json.loads(df_map.to_json())
for feature in geojson_data["features"]:
    feature["id"] = str(feature["properties"]["BFS_NUMMER"])

color_scale = "Reds" if abo_typ == "GENERALABONNEMENT" else "Blues"

# --- Plotly Choroplethenkarte ---
fig = px.choropleth_map(
    df_map,
    geojson=geojson_data,        
    locations="BFS_NUMMER",       
    color=abo_typ,                           
    color_continuous_scale=color_scale,      
    map_style="carto-positron",
    zoom=7.5,
    center={"lat": 46.8182, "lon": 8.2275},  
    opacity=0.6,
    hover_name="NAME",                       
    labels={"GENERALABONNEMENT": "GAs", "HALBTAXABONNEMENT": "Halbtax"}
)

fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=750)
st.plotly_chart(fig, width="stretch")