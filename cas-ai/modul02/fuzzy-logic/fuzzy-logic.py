import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

###
# Getting started:
#
# python3 -m venv venv
# source venv/bin/activate
# pip install numpy matplotlib scikit-fuzzy scipy networkx
#
###

# -------------------------------------------------
# 1. Fuzzy-Variablen
# -------------------------------------------------

alter = ctrl.Antecedent(np.arange(18, 71, 1), 'alter')
einkommen = ctrl.Antecedent(np.arange(20000, 150001, 1000), 'einkommen')
beschaeftigungsdauer = ctrl.Antecedent(np.arange(0, 41, 1), 'beschaeftigungsdauer')
kontohistorie = ctrl.Antecedent(np.arange(0, 11, 1), 'kontohistorie')

kreditangebot = ctrl.Consequent(np.arange(0, 101, 1), 'kreditangebot')

# -------------------------------------------------
# 2. Membership Functions
# -------------------------------------------------

alter['jung'] = fuzz.trapmf(alter.universe, [18, 18, 25, 32])
alter['mittel'] = fuzz.trimf(alter.universe, [28, 40, 55])
alter['alt'] = fuzz.trapmf(alter.universe, [50, 60, 70, 70])

einkommen['niedrig'] = fuzz.trapmf(einkommen.universe, [20000, 20000, 35000, 50000])
einkommen['mittel'] = fuzz.trapmf(einkommen.universe, [45000, 60000, 80000, 100000])
einkommen['hoch'] = fuzz.trapmf(einkommen.universe, [80000, 100000, 150000, 150000])

beschaeftigungsdauer['kurz'] = fuzz.trapmf(beschaeftigungsdauer.universe, [0, 0, 1, 3])
beschaeftigungsdauer['mittel'] = fuzz.trapmf(beschaeftigungsdauer.universe, [2, 5, 8, 12])
beschaeftigungsdauer['lang'] = fuzz.trapmf(beschaeftigungsdauer.universe, [10, 15, 40, 40])

kontohistorie['schlecht'] = fuzz.gaussmf(kontohistorie.universe, 2, 1.5)
kontohistorie['normal'] = fuzz.gaussmf(kontohistorie.universe, 5, 1.2)
kontohistorie['gut'] = fuzz.gaussmf(kontohistorie.universe, 8.5, 1.2)

kreditangebot['kein_kredit'] = fuzz.trapmf(kreditangebot.universe, [0, 0, 25, 45])
kreditangebot['standard'] = fuzz.trapmf(kreditangebot.universe, [40, 55, 70, 85])
kreditangebot['premium'] = fuzz.trapmf(kreditangebot.universe, [75, 90, 100, 100])

# -------------------------------------------------
# 3. Fuzzy-Regeln (JETZT KONSISTENT)
# -------------------------------------------------

rules = [

    # PREMIUM
    ctrl.Rule(
        einkommen['hoch'] &
        kontohistorie['gut'] &
        beschaeftigungsdauer['lang'],
        kreditangebot['premium']
    ),

    # STANDARD
    ctrl.Rule(
        einkommen['mittel'] &
        kontohistorie['normal'] &
        beschaeftigungsdauer['mittel'],
        kreditangebot['standard']
    ),

    ctrl.Rule(
        alter['jung'] &
        einkommen['mittel'] &
        kontohistorie['gut'],
        kreditangebot['standard']
    ),

    ctrl.Rule(
        alter['alt'] &
        kontohistorie['gut'],
        kreditangebot['standard']
    ),

    # KEIN KREDIT (harte Negativkriterien)
    ctrl.Rule(
        einkommen['niedrig'] |
        kontohistorie['schlecht'] |
        beschaeftigungsdauer['kurz'],
        kreditangebot['kein_kredit']
    )
]

# -------------------------------------------------
# 4. Control System
# -------------------------------------------------

credit_ctrl = ctrl.ControlSystem(rules)

# -------------------------------------------------
# 5. Kundendaten
# -------------------------------------------------

kunden = [
    {"id": 1, "alter": 23, "einkommen": 42000, "beschaeftigungsdauer": 1, "kontohistorie": 4},
    {"id": 2, "alter": 35, "einkommen": 75000, "beschaeftigungsdauer": 6, "kontohistorie": 7},
    {"id": 3, "alter": 52, "einkommen": 120000, "beschaeftigungsdauer": 15, "kontohistorie": 9},
    {"id": 4, "alter": 29, "einkommen": 55000, "beschaeftigungsdauer": 3, "kontohistorie": 6},
    {"id": 5, "alter": 61, "einkommen": 90000, "beschaeftigungsdauer": 25, "kontohistorie": 8},
    {"id": 6, "alter": 45, "einkommen": 65000, "beschaeftigungsdauer": 12, "kontohistorie": 5}
]

# -------------------------------------------------
# 6. Evaluation
# -------------------------------------------------

scores = []
labels = []

for kunde in kunden:
    sim = ctrl.ControlSystemSimulation(credit_ctrl)

    sim.input['alter'] = kunde['alter']
    sim.input['einkommen'] = kunde['einkommen']
    sim.input['beschaeftigungsdauer'] = kunde['beschaeftigungsdauer']
    sim.input['kontohistorie'] = kunde['kontohistorie']

    sim.compute()
    score = sim.output['kreditangebot']

    scores.append(score)
    labels.append(f"Kunde {kunde['id']}")

    print(f"Kunde {kunde['id']}: Kredit-Score = {score:.2f}")

# -------------------------------------------------
# 7. Visualisierung
# -------------------------------------------------

plt.figure(figsize=(10, 5))
plt.bar(labels, scores)

plt.axhline(45, color='red', linestyle='--', linewidth=2, label='Kein Kredit / Standard')
plt.axhline(80, color='green', linestyle='--', linewidth=2, label='Standard / Premium')

plt.ylabel("Kredit-Score")
plt.title("Fuzzy Kredit-Pre-Scoring (BKreditangebot)")
plt.legend()
plt.tight_layout()

plt.savefig('m02_01_fuzzy_logic.png', dpi=300)

plt.show()


# -------------------------------------------------
# 8. Optional: Membership Functions visualisieren
# -------------------------------------------------
# Einkommen und Alter sind besonders präsentationsstark

#einkommen.view()
#alter.view()
#kontohistorie.view()
#plt.show()
