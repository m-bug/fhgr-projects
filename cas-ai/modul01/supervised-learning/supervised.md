
# Supervised Learning – Kreditwürdigkeitsprüfung (Banking Use Case)

## Überblick
Dieses Projekt demonstriert **Supervised Learning** anhand eines realistischen Bank-Use-Cases: der Vorhersage, ob ein Kredit voraussichtlich zurückgezahlt wird.

Supervised Learning eignet sich in diesem fiktiven Beispiel, da historische Daten mit **bekannten Labels**
(zurückgezahlt / nicht zurückgezahlt) vorliegen.

## Use Case
**Fragestellung:** 
Soll ein Kredit basierend auf Kundenmerkmalen bewilligt werden?

**Ziel:**
Vorhersage einer ja/nein Zielvariable:
- 1 = Kredit wird zurückgezahlt
- 0 = Kredit wird nicht zurückgezahlt

## Verwendete Features
| Feature | Beschreibung |
|-------|--------------|
| Einkommen | Monatliches Einkommen (CHF) |
| Schulden | Bestehende Schulden (CHF) |

**Label:**
`Kredit_ok` (0 oder 1)

## Warum Supervised Learning?
- Historische Daten mit bekannten Ergebnissen
- Klare Zielvariable
- Ziel ist **Vorhersage**, nicht Exploration

Geeignete mögliche Algorithmen:
- Logistic Regression
- Decision Tree
- Random Forest

## Python-Abhängigkeiten
- numpy
- pandas
- matplotlib
- scikit-learn

Installation:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Skript
Das Python-Skript:
- generiert synthetische, realistische Daten
- simuliert ein Label
- erzeugt eine einfache Visualisierung

Die erzeugte Grafik:
```
supervised_example.png
```

## Ergebnis
Das Modell lernt eine Entscheidungsgrenze zwischen:
- hohem Einkommen & niedrigen Schulden → Kredit bewilligt
- niedrigem Einkommen & hohen Schulden → Kredit abgelehnt

## Typische Einsatzgebiete
- Kreditvergabe
- Betrugserkennung
- Risiko-Scoring
- Prognosemodelle
