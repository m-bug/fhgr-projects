import pandas as pd
import holoviews as hv
from holoviews import opts, dim
from bokeh.plotting import show

hv.extension('bokeh')

# 1. CSV einlesen
df = pd.read_csv('welthandel.csv', encoding='latin1')

# 2. Exakte Top-10-Namen
top_10_namen = [
    'USA', 'China', 'Germany', 'Japan', 
    'United Kingdom', 'India', 'France', 
    'Italy', 'Russian Federation', 'Brazil'
]

# 3. Grundfilter anwenden
df_filtered = df[
    df['reporterISO'].isin(top_10_namen) & 
    df['partnerISO'].isin(top_10_namen)
].copy()

# HIER GEÄNDERT: Wir nehmen 'fobvalue' statt 'primaryValue'
df_chord = df_filtered[['reporterISO', 'partnerISO', 'fobvalue']].dropna()
df_chord.columns = ['source', 'target', 'value']
df_chord = df_chord[df_chord['value'] > 0]

# 4. Selbsthandel und fremde Partner ausschließen
df_chord = df_chord[df_chord['source'] != df_chord['target']]
df_chord = df_chord[df_chord['target'].isin(top_10_namen)]

df_chord = df_chord.groupby(['source', 'target'], as_index=False)['value'].sum()

print(f"Anzahl eindeutiger Handelsbeziehungen nach Aggregieren: {len(df_chord)}")

# 5. Integer-Mapping für HoloViews vorbereiten
unique_nodes = sorted(list(set(df_chord['source']).union(set(df_chord['target']))))
node_mapping = {name: i for i, name in enumerate(unique_nodes)}

df_chord['source'] = df_chord['source'].map(node_mapping)
df_chord['target'] = df_chord['target'].map(node_mapping)

# 6. Node-Dataset erstellen
nodes_df = pd.DataFrame({'index': range(len(unique_nodes)), 'name': unique_nodes})
nodes = hv.Dataset(nodes_df, 'index')

# 7. Chord-Diagramm initialisieren
chord = hv.Chord((df_chord, nodes), ['source', 'target'], ['value'])

# 8. Styling
chord.opts(
    opts.Chord(
        width=700,
        height=700,
        labels='name',
        cmap='Category10',
        edge_cmap='Category10',
        edge_color=dim('source').str(),
        node_color=dim('index').str()
    )
)

# 9. Rendern und anzeigen
bokeh_plot = hv.render(chord, backend='bokeh')
show(bokeh_plot)