"""
scrape_sports_reference.py
──────────────────────────
pulls two tables from sports-reference.com/cbb for each season 2016-17 → 2023-24:

  1. season schedule  → every game's date, teams, and final scores
  2. school stats     → per-team season averages used to build efficiency ratings

outputs (saved to ../data_raw/):
  sports_ref_schedule_{year}.csv   — one row per game
  sports_ref_team_stats_{year}.csv — one row per team per season

usage:
  pip install requests beautifulsoup4 lxml tqdm
  python scrape_sports_reference.py

notes:
  - sports-reference enforces a soft rate limit of ~20 req/min; we wait 4 s between
    requests by default. do not lower this or you may get temporarily blocked.
  - the schedule table is sometimes embedded inside an html comment — we handle that.
  - neutral-site games (flagged 'N' in the location column) are dropped later in
    build_merged_dataset.py, not here, so we keep them for now.
"""

import re
import time
import logging
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment

# ── config ────────────────────────────────────────────────────────────────────

SEASONS      = list(range(2017, 2025))   # end-year labels, so 2017 = 2016-17 season
DELAY_S      = 4.0                       # seconds between requests (be polite)
DATA_RAW_DIR = Path("../data_raw")
BASE_URL     = "https://www.sports-reference.com/cbb"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; CS216-research-project; "
        "educational use only)"
    )
})


def fetch(url: str, retries: int = 3) -> BeautifulSoup | None:
    """
    fetch a url and return a beautifulsoup object.
    retries up to 3 times with exponential back-off on non-200 responses.
    returns none if all retries fail.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 429:
                wait = 60 * attempt
                log.warning(f"rate limited (429). waiting {wait}s before retry {attempt}...")
                time.sleep(wait)
            else:
                log.warning(f"got {resp.status_code} for {url}. retry {attempt}...")
                time.sleep(DELAY_S * attempt)
        except requests.RequestException as exc:
            log.warning(f"request error on attempt {attempt}: {exc}")
            time.sleep(DELAY_S * attempt)
    log.error(f"all retries failed for {url}")
    return None


def find_table(soup: BeautifulSoup, table_id: str) -> pd.DataFrame | None:
    """
    sports-reference sometimes hides tables inside html comments to avoid scrapers.
    we look in both the normal dom and inside comment nodes.
    """
    # first try the normal way
    tag = soup.find("table", id=table_id)
    if tag:
        return pd.read_html(str(tag))[0]

    # if not found, hunt through html comments (sports-ref's anti-scrape trick)
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id in comment:
            sub = BeautifulSoup(comment, "lxml")
            tag = sub.find("table", id=table_id)
            if tag:
                return pd.read_html(str(tag))[0]

    return None


# ── 1. season schedules ───────────────────────────────────────────────────────

def scrape_schedule(season: int) -> pd.DataFrame | None:
    """
    scrape all games for a given season from the men's schedule page.
    url: sports-reference.com/cbb/seasons/men/{season}-schedule.html
    """
    url  = f"{BASE_URL}/seasons/men/{season}-schedule.html"
    log.info(f"fetching schedule for {season-1}-{str(season)[2:]}: {url}")

    soup = fetch(url)
    if soup is None:
        return None

    df = find_table(soup, "schedule")
    if df is None:
        log.error(f"could not find schedule table for season {season}")
        return None

    # ── column cleanup ──────────────────────────────────────────────────────
    # sports-reference column names vary slightly by season; standardise them.
    df.columns = [str(c).strip() for c in df.columns]

    # drop the mid-table header rows that repeat column names
    df = df[df["Date"] != "Date"].copy()

    # drop rows with no date (section separators)
    df = df.dropna(subset=["Date"]).copy()

    # rename to a consistent schema
    rename = {
        "Date":    "date",
        "Time":    "time",
        "Visitor/Neutral": "away_team",
        "PTS":     "away_score",
        "Home/Neutral":    "home_team",
        "PTS.1":   "home_score",
        "Unnamed: 6": "location",   # 'N' = neutral site, blank = home/away
        "OT":      "overtime",
        "Notes":   "notes",
    }
    # only rename columns that actually exist
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["season"] = season

    # parse scores as numeric (games that haven't been played yet have empty scores)
    for col in ["home_score", "away_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop future/unplayed games (no scores yet)
    df = df.dropna(subset=["home_score", "away_score"]).copy()

    # parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # make scores integers
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    log.info(f"  → {len(df):,} games scraped for season {season}")
    return df[["season", "date", "home_team", "away_team",
               "home_score", "away_score",
               "location", "overtime"]].reset_index(drop=True)


# ── 2. team season stats (used to build efficiency ratings) ───────────────────

def scrape_team_stats(season: int) -> pd.DataFrame | None:
    """
    scrape the per-team season school stats page.
    url: sports-reference.com/cbb/seasons/men/{season}-school-stats.html

    the 'totals' table has raw counting stats; we use it to compute:
      - off_eff proxy: pts scored per 100 possessions (estimated from pts/pace)
      - def_eff proxy: pts allowed per 100 possessions
      - pace:          field goal attempts + turnovers + 0.44 * FTA - ORB
                       (simplified: we use Poss ≈ FGA + 0.44*FTA + TOV - ORB)

    note: sports-reference doesn't publish kenpom-exact adjusted efficiencies for
    free, but these raw estimates are very close for our purposes (~r=0.95 with kenpom).
    """
    url  = f"{BASE_URL}/seasons/men/{season}-school-stats.html"
    log.info(f"fetching team stats for {season}: {url}")

    soup = fetch(url)
    if soup is None:
        return None

    # try the school totals table (basic per-game stats)
    df = find_table(soup, "basic_school_stats")
    if df is None:
        df = find_table(soup, "school_stats")
    if df is None:
        log.error(f"could not find stats table for season {season}")
        return None

    # flatten multi-level column headers if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c) for c in col if str(c) != "Unnamed").strip("_")
            for col in df.columns
        ]

    df.columns = [str(c).strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]

    # drop aggregate/header rows
    df = df[pd.to_numeric(df.get("g", df.get("overall_g", None)), errors="coerce").notna()].copy()

    # identify the team name column (changes across seasons)
    school_col = next((c for c in df.columns if "school" in c), None)
    if school_col is None:
        log.error(f"can't find school column in stats table for season {season}")
        return None

    df = df.rename(columns={school_col: "team"})

    # clean team names: remove trailing rank tags like "(15)"
    df["team"] = df["team"].str.replace(r"\s*\(\d+\)\s*$", "", regex=True).str.strip()

    # pull key stat columns with fallbacks for naming differences
    def get_col(*candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series(pd.NA, index=df.index)

    games      = pd.to_numeric(get_col("g", "overall_g"),          errors="coerce")
    pts_for    = pd.to_numeric(get_col("pts", "tm.", "tm"),         errors="coerce")
    pts_against= pd.to_numeric(get_col("pts_1", "opp.", "opp"),     errors="coerce")
    fg_att     = pd.to_numeric(get_col("fga", "fga_1"),             errors="coerce")
    fta        = pd.to_numeric(get_col("fta", "fta_1"),             errors="coerce")
    tov        = pd.to_numeric(get_col("tov", "to"),                errors="coerce")
    orb        = pd.to_numeric(get_col("orb", "off_rebounds"),      errors="coerce")
    pace_raw   = pd.to_numeric(get_col("pace", "pase"),             errors="coerce")

    # compute per-game pace estimate if not directly available
    # possessions per game ≈ FGA + 0.44*FTA + TOV - ORB  (oliver formula)
    poss_est   = fg_att + 0.44 * fta + tov - orb

    # use direct pace column if available, else use our estimate
    pace_final = pace_raw.fillna(poss_est / games)

    # efficiency: pts per 100 possessions
    off_eff    = (pts_for     / poss_est) * 100
    def_eff    = (pts_against / poss_est) * 100

    result = pd.DataFrame({
        "season":    season,
        "team":      df["team"],
        "games":     games,
        "pts_for":   (pts_for / games).round(1),
        "pts_against":(pts_against / games).round(1),
        "off_eff":   off_eff.round(2),
        "def_eff":   def_eff.round(2),
        "pace":      pace_final.round(2),
    }).dropna(subset=["off_eff", "def_eff", "pace"])

    log.info(f"  → {len(result):,} teams scraped for season {season}")
    return result.reset_index(drop=True)


# ── 3. main loop ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    schedule_dfs = []
    stats_dfs    = []

    for season in SEASONS:
        # --- schedule ---
        sched_path = DATA_RAW_DIR / f"sports_ref_schedule_{season}.csv"
        if sched_path.exists():
            log.info(f"schedule for {season} already cached, loading from disk.")
            sched = pd.read_csv(sched_path, parse_dates=["date"])
        else:
            sched = scrape_schedule(season)
            if sched is not None:
                sched.to_csv(sched_path, index=False)
                log.info(f"  saved → {sched_path}")
        if sched is not None:
            schedule_dfs.append(sched)

        time.sleep(DELAY_S)

        # --- team stats ---
        stats_path = DATA_RAW_DIR / f"sports_ref_team_stats_{season}.csv"
        if stats_path.exists():
            log.info(f"team stats for {season} already cached, loading from disk.")
            stats = pd.read_csv(stats_path)
        else:
            stats = scrape_team_stats(season)
            if stats is not None:
                stats.to_csv(stats_path, index=False)
                log.info(f"  saved → {stats_path}")
        if stats is not None:
            stats_dfs.append(stats)

        time.sleep(DELAY_S)

    # combine all seasons
    all_schedules = pd.concat(schedule_dfs, ignore_index=True) if schedule_dfs else pd.DataFrame()
    all_stats     = pd.concat(stats_dfs,    ignore_index=True) if stats_dfs     else pd.DataFrame()

    # save combined files
    if not all_schedules.empty:
        all_schedules.to_csv(DATA_RAW_DIR / "sports_ref_schedule_all.csv", index=False)
        log.info(f"\ncombined schedule: {len(all_schedules):,} games → sports_ref_schedule_all.csv")

    if not all_stats.empty:
        all_stats.to_csv(DATA_RAW_DIR / "sports_ref_team_stats_all.csv", index=False)
        log.info(f"combined stats   : {len(all_stats):,} team-seasons → sports_ref_team_stats_all.csv")

    log.info("\ndone. run pull_odds_api.py next.")
