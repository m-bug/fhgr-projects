# ==========================================
# Knowledge Graph: Organigramm
# ==========================================

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

###
# Getting started:
#
# pip install pandas matplotlib
#
###

# ------------------------------------------
# 1. Knowledge-Graph-Daten definieren
# ------------------------------------------

data = [
    ("Marc", "leads", "Company"),
    ("Tim", "reports_to", "Marc"),
    ("Kevin", "reports_to", "Marc"),
    ("David", "reports_to", "Tim"),
    ("Daniela", "reports_to", "Tim"),
    ("Tim", "works_in", "IT"),
    ("Kevin", "works_in", "Finance"),
    ("David", "works_in", "IT"),
    ("Daniela", "works_in", "IT")
]

df = pd.DataFrame(data, columns=["head", "relation", "tail"])
print(df)

# ------------------------------------------
# 2. Knowledge Graph aufbauen
# ------------------------------------------

G = nx.DiGraph()

for _, row in df.iterrows():
    G.add_edge(row["head"], row["tail"], relation=row["relation"])

# ------------------------------------------
# 3. Visualisierung
# ------------------------------------------

# Reproduzierbares Layout
pos = nx.spring_layout(G, seed=42, k=1.2)

# Knotentypen
persons = {"Marc", "Tim", "Kevin", "David", "Daniela"}
departments = {"IT", "Finance"}
company = {"Company"}

plt.figure(figsize=(12, 10))

# Company
nx.draw_networkx_nodes(
    G, pos,
    nodelist=company,
    node_shape="s",
    node_size=3000,
    node_color="lightgrey"
)

# Departments
nx.draw_networkx_nodes(
    G, pos,
    nodelist=departments,
    node_shape="s",
    node_size=2500,
    node_color="lightgreen"
)

# Persons
nx.draw_networkx_nodes(
    G, pos,
    nodelist=persons,
    node_shape="o",
    node_size=2000,
    node_color="lightblue"
)

# Kanten
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")

# Labels
nx.draw_networkx_labels(G, pos, font_size=16)

# Kantenbeschriftungen (Relationen)
edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=14,
    label_pos=0.5
)

plt.title("Organizational Knowledge Graph of Fantasy-IT Gmbh")
plt.axis("off")

plt.savefig('m02_02_knowledge-graph.png', dpi=300)

plt.show()



# ------------------------------------------
# 4. Knowledge-Graph-Abfragen
# ------------------------------------------

# Wo arbeitet eine Person?
def where_does_person_work(person):
    return [
        target
        for _, target, data in G.out_edges(person, data=True)
        if data["relation"] == "works_in"
    ]

# Wem berichtet eine Person?
def reports_to(person):
    return [
        target
        for _, target, data in G.out_edges(person, data=True)
        if data["relation"] == "reports_to"
    ]

# Wer arbeitet in einer Abteilung?
def who_works_in(department):
    return [
        source
        for source, _, data in G.in_edges(department, data=True)
        if data["relation"] == "works_in"
    ]

# Wer berichtet direkt oder indirekt an eine Person?
def all_reports_to(manager):
    return list(nx.ancestors(G, manager))

# ------------------------------------------
# 5. Beispiel-Ausgaben
# ------------------------------------------

print("Where does David work?:", where_does_person_work("David"))
print("Who does David report to?:", reports_to("David"))
print("Who works in IT?:", who_works_in("IT"))
print("All employees under Marc:", all_reports_to("Marc"))
