"""
pull_odds_api.py
─────────────────
pulls historical college basketball odds (spread + total) from The Odds API
for seasons 2016-17 through 2023-24.

usage:
  export ODDS_API_KEY="your_key_here"
  python pull_odds_api.py

  or pass the key directly:
  python pull_odds_api.py --api-key YOUR_KEY

get a free api key at: https://the-odds-api.com  (free tier = 500 req/month)

the free tier is tight for 8 seasons. strategy to stay within quota:
  - pull one season at a time and cache results to disk immediately
  - re-running the script skips dates already saved (idempotent)
  - you can spread pulls across multiple days

outputs (saved to ../data_raw/):
  odds_api_{season}.csv — one row per game-bookmaker line
  odds_api_all.csv      — combined across all seasons

api details:
  endpoint : GET /v4/historical/sports/{sport}/odds
  sport key: basketball_ncaab
  markets  : spreads, totals (h2h/moneyline not needed for this project)
  regions  : us
  docs     : https://the-odds-api.com/lh/api/#get-/v4/historical-sports/-sport--odds
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import date, timedelta

import requests
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────

API_BASE     = "https://api.the-odds-api.com/v4"
SPORT        = "basketball_ncaab"
REGIONS      = "us"
MARKETS      = "spreads,totals"      # we only need spread and over/under total
ODDS_FORMAT  = "american"
DATE_FORMAT  = "%Y-%m-%dT%H:%M:%SZ"

DELAY_S      = 1.5    # seconds between requests (api is not rate-limited harshly, but be safe)
DATA_RAW_DIR = Path("../data_raw")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# ncaa basketball season date windows (approximate — regular season + postseason)
SEASON_WINDOWS = {
    2017: (date(2016, 11, 1), date(2017, 4, 5)),
    2018: (date(2017, 11, 1), date(2018, 4, 4)),
    2019: (date(2018, 11, 1), date(2019, 4, 9)),
    2020: (date(2019, 11, 1), date(2020, 3, 12)),   # covid ended the season early
    2021: (date(2020, 11, 25), date(2021, 4, 5)),   # covid-shortened schedule
    2022: (date(2021, 11, 1), date(2022, 4, 5)),
    2023: (date(2022, 11, 1), date(2023, 4, 4)),
    2024: (date(2023, 11, 1), date(2024, 4, 9)),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── api helpers ───────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """get key from env var or command-line argument."""
    parser = argparse.ArgumentParser(description="pull historical ncaab odds from the odds api")
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY", ""),
                        help="your odds api key (or set ODDS_API_KEY env var)")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(SEASON_WINDOWS.keys()),
                        help="which seasons to pull (e.g. --seasons 2022 2023 2024)")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit(
            "error: no api key found.\n"
            "  set the ODDS_API_KEY environment variable or pass --api-key YOUR_KEY\n"
            "  get a free key at https://the-odds-api.com"
        )
    return args.api_key, args.seasons


def fetch_odds_for_date(api_key: str, snapshot_date: date) -> list[dict]:
    """
    fetch all ncaab game odds for a specific date snapshot.

    the odds api historical endpoint returns the odds as they were at the
    snapshot timestamp. we use noon UTC to capture pre-game lines.
    """
    timestamp = f"{snapshot_date.isoformat()}T12:00:00Z"
    url = f"{API_BASE}/historical/sports/{SPORT}/odds"

    params = {
        "apiKey":     api_key,
        "regions":    REGIONS,
        "markets":    MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "date":       timestamp,
    }

    resp = requests.get(url, params=params, timeout=30)

    # log remaining api quota from response headers
    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")

    if resp.status_code == 401:
        sys.exit("error: invalid api key. check your ODDS_API_KEY.")
    elif resp.status_code == 422:
        log.warning(f"  {snapshot_date}: no data available (422) — skipping")
        return []
    elif resp.status_code == 429:
        log.warning(f"  rate limited (429). waiting 60s...")
        time.sleep(60)
        return fetch_odds_for_date(api_key, snapshot_date)
    elif resp.status_code != 200:
        log.warning(f"  {snapshot_date}: status {resp.status_code} — skipping")
        return []

    games_raw = resp.json().get("data", [])

    log.info(
        f"  {snapshot_date}: {len(games_raw)} games | "
        f"api quota: {used} used / {remaining} remaining"
    )

    if remaining != "?" and int(remaining) < 20:
        log.warning("  !! fewer than 20 api requests remaining. pausing to avoid exhaustion.")
        log.warning("  !! resume tomorrow or upgrade your plan.")
        sys.exit(1)

    return games_raw


def parse_odds_response(games_raw: list[dict], snapshot_date: date) -> list[dict]:
    """
    parse the raw api response into flat rows (one per game).
    we take the consensus line: average spread and total across all bookmakers.
    """
    rows = []

    for game in games_raw:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        commence  = game.get("commence_time", "")

        # collect spread and total values across all bookmakers
        spreads = []
        totals  = []

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home_team:
                            # positive point = home is favored (they give points)
                            # odds api uses negative spread for the favorite, so flip sign
                            spreads.append(-outcome["point"])
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == "Over":
                            totals.append(outcome["point"])

        if not spreads or not totals:
            continue   # skip games with no spread or total data

        rows.append({
            "date":            snapshot_date.isoformat(),
            "home_team":       home_team,
            "away_team":       away_team,
            "commence_time":   commence,
            "spread_line":     round(sum(spreads) / len(spreads) * 2) / 2,  # consensus, rounded to 0.5
            "total_line":      round(sum(totals)  / len(totals)  * 2) / 2,
            "n_bookmakers":    len(game.get("bookmakers", [])),
        })

    return rows


# ── main loop ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key, seasons_to_pull = get_api_key()

    all_rows = []

    for season in seasons_to_pull:
        if season not in SEASON_WINDOWS:
            log.warning(f"season {season} not in SEASON_WINDOWS, skipping.")
            continue

        out_path = DATA_RAW_DIR / f"odds_api_{season}.csv"

        if out_path.exists():
            log.info(f"season {season} already cached at {out_path}, loading from disk.")
            season_df = pd.read_csv(out_path)
            all_rows.append(season_df)
            continue

        start_date, end_date = SEASON_WINDOWS[season]
        log.info(f"\n── season {season} ({start_date} → {end_date}) ──────────────")

        # track which dates we've already pulled so restarts are safe
        cache_dir  = DATA_RAW_DIR / f"odds_cache_{season}"
        cache_dir.mkdir(exist_ok=True)

        season_rows = []
        current     = start_date

        while current <= end_date:
            cache_file = cache_dir / f"{current.isoformat()}.json"

            if cache_file.exists():
                with open(cache_file) as f:
                    day_rows = json.load(f)
                log.info(f"  {current}: loaded {len(day_rows)} games from cache")
            else:
                games_raw = fetch_odds_for_date(api_key, current)
                day_rows  = parse_odds_response(games_raw, current)
                with open(cache_file, "w") as f:
                    json.dump(day_rows, f)
                time.sleep(DELAY_S)

            season_rows.extend(day_rows)
            current += timedelta(days=1)

        if season_rows:
            season_df = pd.DataFrame(season_rows)
            season_df["season"] = season
            season_df.to_csv(out_path, index=False)
            log.info(f"  saved {len(season_df):,} games → {out_path}")
            all_rows.append(season_df)
        else:
            log.warning(f"  no data collected for season {season}")

    # combine all seasons
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(DATA_RAW_DIR / "odds_api_all.csv", index=False)
        log.info(f"\ncombined: {len(combined):,} game-lines → odds_api_all.csv")
    else:
        log.warning("no odds data collected at all. check your api key and quota.")

    log.info("\ndone. run build_merged_dataset.py next.")
