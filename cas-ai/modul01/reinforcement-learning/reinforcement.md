
# Reinforcement Learning – Dynamische Angebotssteuerung im Banking

## Überblick
Dieses Projekt demonstriert **Reinforcement Learning (RL)** anhand eines realistischen
Bank-Use-Cases: der dynamischen Steuerung von Angeboten im E-Banking.

Im Gegensatz zu Supervised und Unsupervised Learning lernt ein Agent hier durch
**Interaktion mit der Umgebung** und **Feedback (Rewards)**.

---

## Use Case
**Fragestellung:**
Welches Produktangebot soll welchem Kunden zu welchem Zeitpunkt angezeigt werden,
um langfristig den Geschäftserfolg zu maximieren?

**Ziel:**
Optimierung einer Angebotsstrategie basierend auf Kundenreaktionen.

---

## Modellierung des Problems

### Zustand (State)
- Kundentyp (z.B. aus vorherigem Clustering)
- Kontext (vereinfacht: Kundengruppe)

### Aktionen (Actions)
- Angebot A (z.B. Kredit)
- Angebot B (z.B. Sparplan)
- Kein Angebot

### Reward
| Kundenreaktion | Reward |
|--------------|--------|
| Angebot akzeptiert | +1 |
| Ignoriert | 0 |
| Negativ / Abmeldung | -1 |

---

## Warum Reinforcement Learning?
- Keine vorab bekannten optimalen Entscheidungen
- Entscheidungen beeinflussen zukünftiges Verhalten
- Ziel ist **langfristige Optimierung**, nicht Einzelfall-Vorhersage

Geeignete mögliche Algorithmen:
- Q-Learning
- SARSA
- Policy Gradient Methoden

---

## Python-Abhängigkeiten
- numpy
- matplotlib

Installation:
```bash
pip install numpy matplotlib
```

---

## Skript
Das Python-Skript:
- simuliert Kundeninteraktionen
- lernt eine Q-Tabelle mittels Q-Learning
- visualisiert den Lernfortschritt

Die erzeugte Grafik:
```
reinforcement_learning_progress.png
```

---

## Ergebnis
Der Agent lernt schrittweise:
- welche Angebote für welche Kundentypen funktionieren
- Exploration am Anfang, Exploitation im späteren Verlauf

Typisches Lernverhalten:
- anfänglich niedriger Reward
- kontinuierliche Verbesserung über Zeit

---

## Typische Einsatzgebiete im Banking
- Personalisierte Produktangebote
- Dynamische Preisgestaltung
- Next-Best-Action Systeme
- Kampagnenoptimierung

---

## Governance- & Risiko-Hinweis
Reinforcement Learning:
- ist schwerer erklärbar als klassische Modelle
- benötigt klare Guardrails (z.B. Angebotsfrequenzen)
- sollte initial in kontrollierten Umgebungen getestet werden
