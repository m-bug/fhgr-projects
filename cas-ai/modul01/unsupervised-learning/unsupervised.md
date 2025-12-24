
# Unsupervised Learning – Zahlungsbasierte Kundensegmentierung (Banking Use Case)

## Überblick
Dieses Projekt demonstriert **Unsupervised Learning** anhand von Zahlungsverkehrsdaten einer Bank.

Ziel ist es, **Verhaltensmuster** zu erkennen, ohne vorab definierte Kundentypen.

## Use Case
**Fragestellung:** 
Welche natürlichen Kundensegmente existieren im ausgehenden Zahlungsverkehr einer fiktiven Retail Bank?

**Ziel:** 
Explorative Segmentierung von Kunden zur:
- Produktentwicklung
- Personalisierung
- strategischen Analyse
- gezielte Kommunikation

## Verwendete Features
| Feature | Beschreibung |
|-------|--------------|
| Alter | Alter des Kontoinhabers |
| Anzahl_Zahlungen | Anzahl ausgehender Zahlungen pro Monat |
| Durchschnittsbetrag | Durchschnittlicher Zahlungsbetrag |
| Durchschnittliche_Uhrzeit | Mittlere Tageszeit der Zahlungen |

**Keine Labels vorhanden**


## Warum Unsupervised Learning?
- Keine bekannte Zielvariable
- Ziel ist **Erkenntnisgewinn**
- Cluster werden erst **nachträglich** interpretiert

Geeignete mögliche Algorithmen:
- K-Means
- DBSCAN
- Hierarchisches Clustering


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
- simuliert realistische Zahlungsprofile
- clustert Kunden mit K-Means (scikit-learn)
- erzeugt eine einfache Visualisierung

Die erzeugte Grafik:
```
bank_unsupervised_clusters.png
```

## Mögliche Interpretationen (nachträglich)
Beispielhafte Cluster:
- viele kleine Zahlungen → digitale Vielnutzer / mobile User
- mittlere Aktivität → klassische Kunden
- wenige hohe Zahlungen → vermögende Kunden / klassische E-Banking User am Desktop

⚠️ Diese Bedeutungen entstehen **erst nach dem Clustering**.

## Typische Einsatzgebiete
- Kundensegmentierung
- Produktstrategie
- Anomalieerkennung
- Verhaltensanalyse

## Allgemeine Hinweis
Unsupervised Modelle:
- liefern keine „richtigen Antworten“
- benötigen fachliche Interpretation
- sind erklärungsbedürftig gegenüber Management & Regulierung
