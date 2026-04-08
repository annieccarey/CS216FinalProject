"""
scrape_espn.py
──────────────
pulls ncaa d1 men's basketball game results from espn's public scoreboard api
for seasons 2016-17 through 2023-24. no api key needed.

outputs (saved to data_raw/):
  espn_games_all.csv        — one row per completed game
  espn_team_stats_all.csv   — per-team season efficiency ratings (derived from results)

usage:
  python scrape_espn.py
"""

import time
import json
import logging
from pathlib import Path
from datetime import date, timedelta

import requests
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_RAW  = Path("data_raw")
DATA_RAW.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# season date windows: (start, end) inclusive
SEASONS = {
    2017: (date(2016, 11, 1),  date(2017, 4, 10)),
    2018: (date(2017, 11, 1),  date(2018, 4, 5)),
    2019: (date(2018, 11, 1),  date(2019, 4, 9)),
    2020: (date(2019, 11, 1),  date(2020, 3, 15)),
    2021: (date(2020, 11, 25), date(2021, 4, 5)),
    2022: (date(2021, 11, 1),  date(2022, 4, 5)),
    2023: (date(2022, 11, 1),  date(2023, 4, 4)),
    2024: (date(2023, 11, 1),  date(2024, 4, 9)),
}

ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/scoreboard"
)


def fetch_day(game_date: date, retries: int = 3) -> list:
    """
    fetch all completed ncaa men's basketball games for a given date.
    returns a list of parsed game dicts.
    """
    date_str = game_date.strftime("%Y%m%d")
    params   = {"dates": date_str, "limit": 300}

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(ESPN_URL, params=params,
                                headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return parse_games(resp.json(), game_date)
            else:
                log.warning(f"  {game_date}: status {resp.status_code}, retry {attempt}")
                time.sleep(2 * attempt)
        except Exception as exc:
            log.warning(f"  {game_date}: error {exc}, retry {attempt}")
            time.sleep(2 * attempt)

    return []


def parse_games(data: dict, game_date: date) -> list:
    """
    parse the espn scoreboard json and return flat game records.
    drops neutral-site games and unfinished games.
    """
    rows = []

    for event in data.get("events", []):
        for comp in event.get("competitions", []):

            # skip games that haven't finished
            status = comp.get("status", {}).get("type", {})
            if not status.get("completed", False):
                continue

            # skip neutral-site games (no true home team)
            if comp.get("neutralSite", False):
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            # sort into home and away
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)

            if not home or not away:
                continue

            try:
                home_score = int(home.get("score", 0))
                away_score = int(away.get("score", 0))
            except (ValueError, TypeError):
                continue

            # skip 0-0 games (data errors)
            if home_score == 0 and away_score == 0:
                continue

            home_team = home.get("team", {}).get("displayName", "")
            away_team = away.get("team", {}).get("displayName", "")

            if not home_team or not away_team:
                continue

            rows.append({
                "date":       game_date.isoformat(),
                "home_team":  home_team,
                "away_team":  away_team,
                "home_score": home_score,
                "away_score": away_score,
            })

    return rows


def scrape_season(season: int) -> pd.DataFrame:
    """
    loop through every date in the season and collect all games.
    results are cached per-date so the script is safe to restart.
    """
    start, end   = SEASONS[season]
    cache_dir    = DATA_RAW / f"espn_cache_{season}"
    cache_dir.mkdir(exist_ok=True)

    all_rows = []
    current  = start
    n_days   = (end - start).days + 1
    day_num  = 0

    while current <= end:
        day_num += 1
        cache_file = cache_dir / f"{current.isoformat()}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                rows = json.load(f)
        else:
            rows = fetch_day(current)
            with open(cache_file, "w") as f:
                json.dump(rows, f)
            time.sleep(0.8)   # polite delay between requests

        if rows:
            log.info(f"  {current}: {len(rows)} games")

        all_rows.extend(rows)
        current += timedelta(days=1)

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(
        columns=["date","home_team","away_team","home_score","away_score"]
    )
    df["season"] = season
    log.info(f"season {season}: {len(df):,} games total")
    return df


def build_team_stats(games: pd.DataFrame) -> pd.DataFrame:
    """
    derive per-team season efficiency ratings from game results.

    offensive efficiency proxy: points scored per game (scaled to pts/100 poss)
    defensive efficiency proxy: points allowed per game (scaled similarly)
    pace proxy: average total points / 2 (correlated with possessions)

    these are raw approximations — close enough for the ml features.
    """
    rows = []

    for (season, team), grp in games.groupby(["season", "home_team"]):
        # collect all games for this team (home + away)
        home_games = games[(games["season"] == season) & (games["home_team"] == team)]
        away_games = games[(games["season"] == season) & (games["away_team"] == team)]

        pts_scored  = list(home_games["home_score"]) + list(away_games["away_score"])
        pts_allowed = list(home_games["away_score"]) + list(away_games["home_score"])

        if len(pts_scored) < 5:
            continue

        avg_scored  = np.mean(pts_scored)
        avg_allowed = np.mean(pts_allowed)
        avg_total   = avg_scored + avg_allowed

        # scale to pts/100 possessions using the approximate conversion:
        # D1 average: 71 pts scored per game ≈ 103 pts/100 poss
        # scale factor ≈ 103 / 71 = 1.45
        SCALE = 103 / 71
        pace  = avg_total / 2 * (70 / 71)   # approximate possessions

        rows.append({
            "season":    season,
            "team":      team,
            "games":     len(pts_scored),
            "off_eff":   round(avg_scored  * SCALE, 2),
            "def_eff":   round(avg_allowed * SCALE, 2),
            "pace":      round(pace, 2),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    log.info("starting espn scrape for all 8 seasons...")
    log.info("(cached per-date — safe to restart if interrupted)\n")

    all_seasons = []

    for season in SEASONS:
        season_path = DATA_RAW / f"espn_games_{season}.csv"

        if season_path.exists():
            log.info(f"season {season} already cached, loading from disk.")
            df = pd.read_csv(season_path)
        else:
            df = scrape_season(season)
            df.to_csv(season_path, index=False)

        all_seasons.append(df)

    # combine all seasons
    games = pd.concat(all_seasons, ignore_index=True)
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values(["season", "date"]).reset_index(drop=True)

    out_path = DATA_RAW / "espn_games_all.csv"
    games.to_csv(out_path, index=False)
    log.info(f"\nall games saved → {out_path}  ({len(games):,} rows)")

    # build team efficiency stats
    log.info("computing team efficiency ratings from results...")
    stats = build_team_stats(games)
    stats_path = DATA_RAW / "espn_team_stats_all.csv"
    stats.to_csv(stats_path, index=False)
    log.info(f"team stats saved → {stats_path}  ({len(stats):,} team-seasons)")

    # quick sanity check
    log.info(f"\n── sanity check ──────────────────────────────────────")
    log.info(f"total games   : {len(games):,}")
    log.info(f"seasons       : {sorted(games['season'].unique())}")
    log.info(f"date range    : {games['date'].min().date()} → {games['date'].max().date()}")
    log.info(f"avg home score: {games['home_score'].mean():.1f}")
    log.info(f"avg away score: {games['away_score'].mean():.1f}")
    log.info(f"avg total     : {(games['home_score']+games['away_score']).mean():.1f}")
    log.info(f"\ndone. run pull_odds_api.py next, then build_merged_dataset.py")
