import argparse
from datetime import date
from playwright.sync_api import sync_playwright
import sqlite3

## usage:
## python cinema_s.py
## python cinema_s.py 2026-04-12

DB_FILE = "kino.db"
CINEMA_NAME = "Cinema 8"
URL = "https://cinema8.ch/programmuebersicht/?time=tomorrow"


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


def parse_args():
    parser = argparse.ArgumentParser(description="Cinema Scraper")

    parser.add_argument(
        "date",
        nargs="?",
        default=date.today().isoformat(),
        help="Datum im Format YYYY-MM-DD (default: heute)"
    )

    return parser.parse_args()


def scrape_and_store(target_date: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL)

        try:
            page.click("text=Ich akzeptiere alle", timeout=5000)
        except:
            pass

        page.wait_for_timeout(3000)

        data = page.evaluate("() => pmkinoFrontVars.apiData.movies.items")

        for movie_id, movie in data.items():
            title = movie.get("title", "UNKNOWN")

            for perf in movie.get("performances", []):
                perf_date = perf.get("date")

                if perf_date != target_date:
                    continue

                time = perf.get("showtime")
                room = perf.get("theatreName")

                cursor.execute("""
                    INSERT OR IGNORE INTO showtimes (cinema, movie, date, time, room)
                    VALUES (?, ?, ?, ?, ?)
                """, (CINEMA_NAME, title, perf_date, time, room))

                print(f"{title} | {perf_date} {time} | {room}")

        browser.close()

    conn.commit()
    conn.close()


if __name__ == "__main__":
    args = parse_args()

    print(f"📅 Verwende Datum: {args.date}")

    init_db()
    scrape_and_store(args.date)