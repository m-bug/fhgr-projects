import json
import time
from datetime import datetime
import pandas as pd
from pytrends.request import TrendReq

# pip install pytrends pandas requests
from datetime import datetime
import json
import xml.etree.ElementTree as ET
import requests


def fetch_google_news_trends():
  """Ruft die aktuellen Top-Schlagzeilen/Trends über den offiziellen

  Google News RSS-Feed für die USA ab (100% stabil & kein 404-Fehler).
  """
  print(f"[{datetime.now()}] Hole aktuelle Trends via Google News RSS (US)...")

  # Offizieller Google News RSS-Feed für die USA (Englisch)
  url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"

  try:
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
      print(f" Fehler: Google hat mit Statuscode {response.status_code} geantwortet.")
      return []

    # XML-Daten parsen
    root = ET.fromstring(response.content)
    items = root.findall(".//item")

    trend_data = []
    # Wir nehmen die Top 10 aktuellen Themen
    for index, item in enumerate(items[:10], 1):
      title = item.find("title").text if item.find("title") is not None else "Unknown Trend"
      
      # Bereinigen des Titels (Google News hängt oft den Verlagsnamen an, z.B. " - CNN")
      clean_title = title.split(" - ")[0]

      trend_item = {
          "rank": index,
          "keyword": clean_title,
          "timestamp": datetime.now().isoformat(),
          "status": "ready_for_script",
          "generated_script": {
              "hook": f"Du wirst nicht glauben, was gerade in den USA passiert: {clean_title}!",
              "body": f"Hier sind die wichtigsten Hintergründe dazu...",
              "call_to_action": "Was ist deine Meinung dazu? Schreib es in die Kommentare!",
          },
      }
      trend_data.append(trend_item)

    print(f" Erfolgreich {len(trend_data)} Trends extrahiert.")
    return trend_data

  except Exception as e:
    print(f" Fehler beim Abrufen der News: {e}")
    return []


def save_to_json(data, filename="trends_queue.json"):
  """Speichert die Trends als JSON für den nächsten Pipeline-Schritt."""
  with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
  print(f" Daten erfolgreich in '{filename}' gespeichert.")


if __name__ == "__main__":
  current_trends = fetch_google_news_trends()

  if current_trends:
    save_to_json(current_trends)
    print(" Nächster Schritt: Dein LLM-Skript kann jetzt 'trends_queue.json' verarbeiten!")
  else:
    print(" Keine Daten erhalten.")