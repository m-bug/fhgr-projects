## usage:
# python cinema_a.py
# → https://www.kinoaarau.ch/#programm

# python cinema_a.py 2026-04-10
# → https://www.kinoaarau.ch/#programm-2026-04-11

import argparse
import sqlite3
from datetime import datetime
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

HTML_FILE = "kino.html"
DB_FILE = "kino.db"
BASE_URL = "https://www.kinoaarau.ch/#programm"


def parse_args():
    parser = argparse.ArgumentParser(description="Kino Aarau Scraper")

    parser.add_argument(
        "date",
        nargs="?",
        help="Datum im Format YYYY-MM-DD (optional)"
    )

    return parser.parse_args()


def build_url(target_date: str | None) -> str:
    """URL je nach Datum bauen"""
    if target_date:
        return f"{BASE_URL}-{target_date}"
    return BASE_URL


def fetch_html(url: str):
    """HTML vom Kino holen und speichern"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        # easy: determine selector
        page.wait_for_selector(".movie-teaser")

        # download static content
        content = page.content()
        with open(HTML_FILE, "w", encoding="utf-8") as f:
            f.write(content)

        browser.close()


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS showtimes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cinema TEXT,
            movie TEXT,
            date TEXT,
            time TEXT,
            room TEXT,
            UNIQUE(cinema, movie, date, time, room)
        )
    """)

    conn.commit()
    conn.close()


def parse_and_store(cinema_name: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    with open(HTML_FILE, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    movies = soup.select(".movie-teaser")

    for movie in movies:
        title_tag = movie.select_one("h5.ticket-buy-day-titel.small")
        title = title_tag.get_text(strip=True) if title_tag else "UNKNOWN"

        date_tag = movie.select_one("h5.ticket-buy-day-titel.light")
        date = date_tag.get_text(strip=True) if date_tag else ""

        showtimes = movie.select(".showtime")

        for show in showtimes:
            raw_text = show.get_text("\n", strip=True)
            lines = raw_text.split("\n")

            time_info = lines[0] if len(lines) > 0 else ""
            room = lines[1] if len(lines) > 1 else ""

            cursor.execute("""
                INSERT OR IGNORE INTO showtimes (cinema, movie, date, time, room)
                VALUES (?, ?, ?, ?, ?)
            """, (cinema_name, title, date, time_info, room))

            print(f"{title} | {date} {time_info} | {room}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    args = parse_args()

    # Optional: Datum validieren
    if args.date:
        try:
            datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print("❌ Ungültiges Datum! Format: YYYY-MM-DD")
            exit(1)

    cinema_name = "Kino Aarau"
    url = build_url(args.date)

    print(f"🌐 Verwende URL: {url}")

    fetch_html(url)
    init_db()
    parse_and_store(cinema_name)

    print("✅ Daten wurden in kino.db gespeichert")