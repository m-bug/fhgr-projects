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
    
    # 1. Geodaten laden (swisstopo Hoheitsgebiete)
    if os.path.exists(gpkg_path):
        try:
            gdf_gemeinden = gpd.read_file(gpkg_path, layer="TLM_HOHEITSGEBIET")
            gdf_gemeinden.columns = gdf_gemeinden.columns.str.upper()
            
            # Performance-Optimierung für Plotly
            if "GEOMETRY" in gdf_gemeinden.columns:
                gdf_gemeinden = gdf_gemeinden.set_geometry("GEOMETRY")
            gdf_gemeinden["GEOMETRY"] = gdf_gemeinden["GEOMETRY"].simplify(10, preserve_topology=True)
            gdf_gemeinden["GEOMETRY"] = gdf_gemeinden["GEOMETRY"].force_2d()
            gdf_gemeinden = gdf_gemeinden.to_crs(epsg=4326)
            
            # BFS-ID Spalte vereinheitlichen
            for alt in ["BFS_NUMMER", "GMDNR", "BFS_ID", "GMD_NR", "BFS"]:
                if alt in gdf_gemeinden.columns:
                    gdf_gemeinden = gdf_gemeinden.rename(columns={alt: "BFS_NUMMER"})
                    break
        except Exception as e:
            st.sidebar.warning(f"GPKG-Fehler: {e}")
            gdf_gemeinden = None

    # Fallback Online-Karte
    if gdf_gemeinden is None or len(gdf_gemeinden) == 0:
        url = "https://raw.githubusercontent.com/stefanolderog/geopandas-swiss-boundaries/master/data/GEN_A4_GEMEINDEN_2019.geojson"
        gdf_gemeinden = gpd.read_file(url)
        gdf_gemeinden.columns = gdf_gemeinden.columns.str.upper()
        gdf_gemeinden = gdf_gemeinden.rename(columns={"GMDNAME": "NAME", "GMDNR": "BFS_NUMMER"})

    # BFS_NUMMER der Karte als String-ID säubern
    gdf_gemeinden["BFS_NUMMER"] = pd.to_numeric(gdf_gemeinden["BFS_NUMMER"], errors='coerce').fillna(0).astype(int).astype(str)

    # 2. SBB-Daten laden
    df_sbb = pd.read_csv("generalabo-halbtax.csv", sep=";")
    df_sbb.columns = df_sbb.columns.str.upper()
    df_sbb = df_sbb[df_sbb["JAHR"] == ausgewaehltes_jahr]
    df_sbb["PLZ"] = pd.to_numeric(df_sbb["PLZ"], errors='coerce').fillna(0).astype(int)

    # 3. LOKALES PLZ-MAPPING
    plz_file = "AMTOVZ_CSV_LV95.csv"
    if os.path.exists(plz_file):
        try:
            try:
                df_plz = pd.read_csv(plz_file, sep=";", encoding='utf-8-sig', on_bad_lines='skip')
            except Exception:
                df_plz = pd.read_csv(plz_file, sep=";", encoding='latin1', on_bad_lines='skip')
            
            # Spalten exakt aus deinem File matchen
            df_plz_mapping = df_plz[["PLZ4", "BFS-Nr"]].drop_duplicates()
            df_plz_mapping = df_plz_mapping.rename(columns={"PLZ4": "PLZ", "BFS-Nr": "BFS_NUMMER"})
            
            df_plz_mapping["PLZ"] = pd.to_numeric(df_plz_mapping["PLZ"], errors='coerce').fillna(0).astype(int)
            df_plz_mapping["BFS_NUMMER"] = pd.to_numeric(df_plz_mapping["BFS_NUMMER"], errors='coerce').fillna(0).astype(int).astype(str)
            
            # SBB-Daten mit echten BFS-Nummern anreichern
            df_sbb = df_sbb.merge(df_plz_mapping, on="PLZ", how="left")
            
        except Exception as e:
            st.sidebar.error(f"Fehler beim Einlesen der neuen PLZ-Datei: {e}")
            df_sbb["BFS_NUMMER"] = df_sbb["PLZ"].astype(str)
    else:
        st.sidebar.error(f"Datei '{plz_file}' nicht gefunden!")
        df_sbb["BFS_NUMMER"] = df_sbb["PLZ"].astype(str)

    # SBB Abos pro offizieller BFS-Gemeindenummer aggregieren
    df_sbb["BFS_NUMMER"] = df_sbb["BFS_NUMMER"].fillna("0")
    df_sbb_grouped = df_sbb.groupby("BFS_NUMMER")[["GENERALABONNEMENT", "HALBTAXABONNEMENT"]].sum().reset_index()

    # 4. Karte mit aggregierten SBB-Daten verknüpfen via BFS_NUMMER
    merged = gdf_gemeinden.merge(df_sbb_grouped, on="BFS_NUMMER", how="left")
    merged["GENERALABONNEMENT"] = merged["GENERALABONNEMENT"].fillna(0)
    merged["HALBTAXABONNEMENT"] = merged["HALBTAXABONNEMENT"].fillna(0)
    
    # Datetime-Fix für die GeoJSON-Konvertierung
    for col in merged.columns:
        if pd.api.types.is_datetime64_any_dtype(merged[col]) or merged[col].dtype == "object":
            try:
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

# Kontroll-Anzeigen in der App
st.success(f"Geometrien geladen: {len(df_map)} Gemeinden auf der Karte.")
aktive_gemeinden = len(df_map[df_map["GENERALABONNEMENT"] + df_map["HALBTAXABONNEMENT"] > 0])
st.info(f"Davon erfolgreich mit SBB-Daten gefärbt: {aktive_gemeinden}")

# GeoJSON extrahieren für Plotly Choropleth
geojson_data = json.loads(df_map.to_json())
for feature in geojson_data["features"]:
    feature["id"] = str(feature["properties"]["BFS_NUMMER"])

# --- OPTIMIERUNG DER FARBSKALA (95%-Quantil gegen extreme Ausreisser-Städte) ---
non_zero_data = df_map[df_map[abo_typ] > 0][abo_typ]
if not non_zero_data.empty:
    max_val = non_zero_data.quantile(0.95)  # Schützt ländliche Gebiete vor dem Ausbleichen
else:
    max_val = 10

color_scale = "Reds" if abo_typ == "GENERALABONNEMENT" else "Blues"

# --- Plotly Choroplethenkarte (OPTIMIERTE FARBKRAFT & TRANSPARENZ) ---
fig = px.choropleth_map(
    df_map,
    geojson=geojson_data,        
    locations="BFS_NUMMER",       
    color=abo_typ,                           
    color_continuous_scale=color_scale,      
    map_style="carto-positron",
    zoom=7.5,
    center={"lat": 46.8182, "lon": 8.2275},  
    opacity=0.85,                           # Erhöhte Deckkraft für kräftigere Farben
    range_color=[0, max_val],               # Skala dynamisch gestrafft
    hover_name="NAME",                       
    labels={"GENERALABONNEMENT": "GAs", "HALBTAXABONNEMENT": "Halbtax"}
)

fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=750)
st.plotly_chart(fig, width="stretch")