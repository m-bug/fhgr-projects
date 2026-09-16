import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re

# source: https://www.bfe.admin.ch/de/gesamtenergiestatistik
file_path = 'ogd115_gest_bilanz.csv' 
df = pd.read_csv(file_path, on_bad_lines='skip')

# ich baue hier filter, da ich nicht alles darstellen kann.. zu grosses datenset..
# jahr: 2012
jahr_ziel = 2012
df_jahr = df[df['Jahr'] == jahr_ziel].copy()
df_jahr['TJ'] = pd.to_numeric(df_jahr['TJ'], errors='coerce').fillna(0)
df_jahr['TJ_abs'] = df_jahr['TJ'].abs()

erlaubte_rubriken = [
    'Endverbrauch - Haushalte',
    'Endverbrauch - Industrie',
    'Endverbrauch - Verkehr',
    'Endverbrauch - Dienstleistungen',
    'Endverbrauch - Statistische Differenz inkl. Landwirtschaft',
    'Nichtenergetischer Verbrauch',
    'Eigenverbrauch des Energiesektors, Netzverluste, Verbrauch der Speicherungen'
]
df_clean = df_jahr[df_jahr['Rubrik'].isin(erlaubte_rubriken)].copy()
df_clean = df_clean[df_clean['TJ_abs'] > 100]

energietraeger = df_clean['Energietraeger'].unique().tolist()
rubriken = df_clean['Rubrik'].unique().tolist()
all_nodes = energietraeger + rubriken
node_dict = {node: i for i, node in enumerate(all_nodes)}

# Farb-Hilfsfunktion (verarbeitet Hex, RGB und RGBA)
def color_to_rgba(color_str, alpha=0.9):
    # Falls es schon ein rgb/rgba-String ist, Zahlen extrahieren
    if 'rgb' in str(color_str):
        nums = re.findall(r'[\d\.]+', str(color_str))
        if len(nums) >= 3:
            return f'rgba({nums[0]}, {nums[1]}, {nums[2]}, {alpha})'
    
    # sonst: als Hex-Code
    hex_str = str(color_str).lstrip('#')
    if len(hex_str) == 6:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
        
    return f'rgba(150, 150, 150, {alpha})' # Fallback

# Palette für die Energieträger zuweisen
palette = px.colors.qualitative.Safe
source_color_dict = {}
for i, source in enumerate(energietraeger):
    source_color_dict[source] = palette[i % len(palette)]

# Knoten-Farben aufbauen
node_colors = []
for node in all_nodes:
    if node in source_color_dict:
        node_colors.append(color_to_rgba(source_color_dict[node], alpha=0.9))
    else:
        node_colors.append('rgba(180, 180, 180, 0.8)') # Dezentes Grau für Ziele rechts

# Quellen, Ziele und Werte aufbauen
sources = []
targets = []
values = []
link_colors = []

for _, row in df_clean.iterrows():
    src_name = row['Energietraeger']
    tgt_name = row['Rubrik']
    val = row['TJ_abs']
    
    source_idx = node_dict.get(src_name)
    target_idx = node_dict.get(tgt_name)
    
    if source_idx is not None and target_idx is not None and val > 0:
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(val)
        
        # Bänder erhalten die Quellfarbe mit leichter Transparenz (alpha=0.4)
        if src_name in source_color_dict:
            link_colors.append(color_to_rgba(source_color_dict[src_name], alpha=0.4))
        else:
            link_colors.append('rgba(200, 200, 200, 0.4)')

# 8. Plotly Sankey-Diagramm erstellen
# sehr einfach, siehe hier: https://plotly.com/python/sankey-diagram/
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = all_nodes,
      color = node_colors
    ),
    link = dict(
      source = sources,
      target = targets,
      value = values,
      color = link_colors
  ))])

fig.update_layout(
    title_text=f"Schweizer Energieströme nach Herkunft ({jahr_ziel})",
    font_size=12,
    width=1200,
    height=700
)

fig.show()