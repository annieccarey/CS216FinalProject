"""
build_merged_dataset.py
────────────────────────
joins sports reference game results with the odds api betting lines,
normalizes team names, adds team efficiency features, and writes the
final merged_games.csv that the analysis notebook loads.

inputs  (from ../data_raw/):
  sports_ref_schedule_all.csv   — from scrape_sports_reference.py
  sports_ref_team_stats_all.csv — from scrape_sports_reference.py
  odds_api_all.csv              — from pull_odds_api.py

output (to ../data_processed/):
  merged_games.csv

usage:
  python build_merged_dataset.py
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from difflib import get_close_matches

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_RAW  = Path("../data_raw")
DATA_PROC = Path("../data_processed")
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ── 1. team name normalization dictionary ─────────────────────────────────────
# sports-reference and the odds api use different spellings/abbreviations.
# this maps the odds-api variants → the sports-reference canonical name.
# extend this dict as you find mismatches in the audit log below.

NAME_MAP = {
    # odds api variant            : sports-ref canonical
    "UCONN":                       "Connecticut",
    "UConn":                       "Connecticut",
    "Connecticut Huskies":         "Connecticut",
    "UNC":                         "North Carolina",
    "North Carolina Tar Heels":    "North Carolina",
    "NC State":                    "NC State",
    "North Carolina State":        "NC State",
    "Kansas Jayhawks":             "Kansas",
    "Duke Blue Devils":            "Duke",
    "Kentucky Wildcats":           "Kentucky",
    "Villanova Wildcats":          "Villanova",
    "Gonzaga Bulldogs":            "Gonzaga",
    "Baylor Bears":                "Baylor",
    "Houston Cougars":             "Houston",
    "Michigan State Spartans":     "Michigan State",
    "Texas Tech Red Raiders":      "Texas Tech",
    "Virginia Cavaliers":          "Virginia",
    "Wisconsin Badgers":           "Wisconsin",
    "Purdue Boilermakers":         "Purdue",
    "UCLA Bruins":                 "UCLA",
    "Arizona Wildcats":            "Arizona",
    "Tennessee Volunteers":        "Tennessee",
    "Alabama Crimson Tide":        "Alabama",
    "Auburn Tigers":               "Auburn",
    "Iowa Hawkeyes":               "Iowa",
    "Illinois Fighting Illini":    "Illinois",
    "Arkansas Razorbacks":         "Arkansas",
    "Providence Friars":           "Providence",
    "Creighton Bluejays":          "Creighton",
    "Miami (FL)":                  "Miami FL",
    "Miami Florida":               "Miami FL",
    "Miami (Ohio)":                "Miami OH",
    "Ohio Bobcats":                "Ohio",
    "Buffalo Bulls":               "Buffalo",
    "South Florida Bulls":         "South Florida",
    "FAU":                         "Florida Atlantic",
    "Florida Atlantic Owls":       "Florida Atlantic",
    "Saint Mary's":                "Saint Mary's",
    "Saint Mary's (CA)":           "Saint Mary's",
    "SMC":                         "Saint Mary's",
    "BYU":                         "BYU",
    "Brigham Young":               "BYU",
    "USF":                         "South Florida",
    "San Diego St":                "San Diego State",
    "San Diego State Aztecs":      "San Diego State",
    "SDSU":                        "San Diego State",
    "VCU Rams":                    "VCU",
    "Virginia Commonwealth":       "VCU",
    "LSU Tigers":                  "LSU",
    "Ole Miss Rebels":             "Ole Miss",
    "Mississippi":                 "Ole Miss",
    "Mississippi State Bulldogs":  "Mississippi State",
    "Texas A&M Aggies":            "Texas A&M",
    "Texas A&M":                   "Texas A&M",
    "Penn State Nittany Lions":    "Penn State",
    "Ohio State Buckeyes":         "Ohio State",
    "Michigan Wolverines":         "Michigan",
    "Indiana Hoosiers":            "Indiana",
    "Minnesota Golden Gophers":    "Minnesota",
    "Northwestern Wildcats":       "Northwestern",
    "Nebraska Cornhuskers":        "Nebraska",
    "Rutgers Scarlet Knights":     "Rutgers",
    "Maryland Terrapins":          "Maryland",
    "Iowa State Cyclones":         "Iowa State",
    "Oklahoma Sooners":            "Oklahoma",
    "Oklahoma State Cowboys":      "Oklahoma State",
    "West Virginia Mountaineers":  "West Virginia",
    "Kansas State Wildcats":       "Kansas State",
    "TCU Horned Frogs":            "TCU",
    "Texas Longhorns":             "Texas",
    "Florida Gators":              "Florida",
    "Georgia Bulldogs":            "Georgia",
    "South Carolina Gamecocks":    "South Carolina",
    "Missouri Tigers":             "Missouri",
    "Vanderbilt Commodores":       "Vanderbilt",
    "Wichita St":                  "Wichita State",
    "Wichita State Shockers":      "Wichita State",
    "Memphis Tigers":              "Memphis",
    "Cincinnati Bearcats":         "Cincinnati",
    "SMU Mustangs":                "SMU",
    "UCF Knights":                 "UCF",
    "Central Florida":             "UCF",
    "Temple Owls":                 "Temple",
    "Tulsa Golden Hurricane":      "Tulsa",
    "East Carolina Pirates":       "East Carolina",
    "Tulane Green Wave":           "Tulane",
    "Loyola (Chi)":                "Loyola Chicago",
    "Loyola Chicago Ramblers":     "Loyola Chicago",
    "Loyola-Chicago":              "Loyola Chicago",
    "Nevada Wolf Pack":            "Nevada",
    "Utah State Aggies":           "Utah State",
    "Boise State Broncos":         "Boise State",
    "New Mexico Lobos":            "New Mexico",
    "Colorado State Rams":         "Colorado State",
    "Wyoming Cowboys":             "Wyoming",
    "Fresno State Bulldogs":       "Fresno State",
    "Air Force Falcons":           "Air Force",
    "UNLV Rebels":                 "UNLV",
    "Hawaii Warriors":             "Hawaii",
    "San Jose State Spartans":     "San Jose State",
    "Georgetown Hoyas":            "Georgetown",
    "Marquette Golden Eagles":     "Marquette",
    "St. John's Red Storm":        "St. John's",
    "Seton Hall Pirates":          "Seton Hall",
    "Xavier Musketeers":           "Xavier",
    "Butler Bulldogs":             "Butler",
    "DePaul Blue Demons":          "DePaul",
    "Dayton Flyers":               "Dayton",
    "Rhode Island Rams":           "Rhode Island",
    "Saint Louis Billikens":       "Saint Louis",
    "George Mason Patriots":       "George Mason",
    "Davidson Wildcats":           "Davidson",
    "Fordham Rams":                "Fordham",
    "La Salle Explorers":          "La Salle",
    "UMass":                       "Massachusetts",
    "Massachusetts Minutemen":     "Massachusetts",
    "Richmond Spiders":            "Richmond",
    "St. Bonaventure Bonnies":     "St. Bonaventure",
    "St. Joseph's Hawks":          "St. Joseph's",
    "George Washington Colonials": "George Washington",
    "Duquesne Dukes":              "Duquesne",
    "Pitt":                        "Pittsburgh",
    "Pittsburgh Panthers":         "Pittsburgh",
    "Boston College Eagles":       "Boston College",
    "Wake Forest Demon Deacons":   "Wake Forest",
    "Clemson Tigers":              "Clemson",
    "Virginia Tech Hokies":        "Virginia Tech",
    "Georgia Tech Yellow Jackets": "Georgia Tech",
    "Syracuse Orange":             "Syracuse",
    "Louisville Cardinals":        "Louisville",
    "Notre Dame Fighting Irish":   "Notre Dame",
    "Florida State Seminoles":     "Florida State",
    "Miami (Fla.)":                "Miami FL",
    "Oregon Ducks":                "Oregon",
    "Oregon State Beavers":        "Oregon State",
    "Washington Huskies":          "Washington",
    "Washington State Cougars":    "Washington State",
    "USC Trojans":                 "USC",
    "Utah Utes":                   "Utah",
    "Colorado Buffaloes":          "Colorado",
    "California Golden Bears":     "California",
    "Stanford Cardinal":           "Stanford",
    "Arizona State Sun Devils":    "Arizona State",
    "Old Dominion Monarchs":       "Old Dominion",
    "UAB Blazers":                 "UAB",
    "Florida Intl":                "FIU",
    "Florida International":       "FIU",
    "Middle Tenn":                 "Middle Tennessee",
    "Middle Tennessee Blue Raiders":"Middle Tennessee",
    "Western Ky":                  "Western Kentucky",
    "Western Kentucky Hilltoppers":"Western Kentucky",
    "North Texas Mean Green":      "North Texas",
    "Louisiana Tech Bulldogs":     "Louisiana Tech",
    "Marshall Thundering Herd":    "Marshall",
    "Rice Owls":                   "Rice",
    "UTEP Miners":                 "UTEP",
    "UTSA Roadrunners":            "UTSA",
    "UL Monroe":                   "Louisiana Monroe",
    "Louisiana Monroe Warhawks":   "Louisiana Monroe",
    "Louisiana Ragin' Cajuns":     "Louisiana",
    "Arkansas State Red Wolves":   "Arkansas State",
    "Troy Trojans":                "Troy",
    "South Alabama Jaguars":       "South Alabama",
    "Appalachian State Mountaineers":"Appalachian State",
    "Georgia State Panthers":      "Georgia State",
    "Georgia Southern Eagles":     "Georgia Southern",
    "Texas State Bobcats":         "Texas State",
    "Little Rock Trojans":         "Little Rock",
    "Akron Zips":                  "Akron",
    "Buffalo Bulls":               "Buffalo",
    "Ball State Cardinals":        "Ball State",
    "Bowling Green Falcons":       "Bowling Green",
    "Central Michigan Chippewas":  "Central Michigan",
    "Eastern Michigan Eagles":     "Eastern Michigan",
    "Kent State Golden Flashes":   "Kent State",
    "Northern Illinois Huskies":   "Northern Illinois",
    "Toledo Rockets":              "Toledo",
    "Western Michigan Broncos":    "Western Michigan",
    "Pepperdine Waves":            "Pepperdine",
    "BYU Cougars":                 "BYU",
    "Pacific Tigers":              "Pacific",
    "LMU":                         "Loyola Marymount",
    "Loyola Marymount Lions":      "Loyola Marymount",
    "Portland Pilots":             "Portland",
    "Santa Clara Broncos":         "Santa Clara",
    "San Diego Toreros":           "San Diego",
    "Gonzaga Bulldogs":            "Gonzaga",
    "Belmont Bruins":              "Belmont",
    "Murray State Racers":         "Murray State",
    "Jacksonville State Gamecocks":"Jacksonville State",
    "Morehead State Eagles":       "Morehead State",
    "FGCU":                        "FGCU",
    "Florida Gulf Coast":          "FGCU",
    "Florida Gulf Coast Eagles":   "FGCU",
    "Lipscomb Bisons":             "Lipscomb",
    "Winthrop Eagles":             "Winthrop",
    "College of Charleston Cougars":"College of Charleston",
    "Hofstra Pride":               "Hofstra",
    "Northeastern Huskies":        "Northeastern",
    "Drexel Dragons":              "Drexel",
    "James Madison Dukes":         "James Madison",
    "UNC Wilmington Seahawks":     "UNC Wilmington",
    "Wright State Raiders":        "Wright State",
    "Drake Bulldogs":              "Drake",
    "Northern Iowa Panthers":      "Northern Iowa",
    "Southern Illinois Salukis":   "Southern Illinois",
    "Illinois State Redbirds":     "Illinois State",
    "Indiana State Sycamores":     "Indiana State",
    "Bradley Braves":              "Bradley",
    "Evansville Purple Aces":      "Evansville",
    "Charlotte 49ers":             "Charlotte",
    "UAB Blazers":                 "UAB",
    "Florida Intl Panthers":       "FIU",
}


def normalize_name(name: str, fallback: bool = True) -> str:
    """
    apply the name map, then optionally do fuzzy matching as a fallback.
    returns the canonical name or the original name if no match found.
    """
    if not isinstance(name, str):
        return str(name)

    name = name.strip()

    # direct lookup first
    if name in NAME_MAP:
        return NAME_MAP[name]

    # try removing common suffixes that appear in one dataset but not the other
    cleaned = (name
               .replace(" Wildcats", "").replace(" Bulldogs", "").replace(" Tigers", "")
               .replace(" Bears", "").replace(" Eagles", "").replace(" Hawks", "")
               .strip())
    if cleaned in NAME_MAP:
        return NAME_MAP[cleaned]

    return name   # give up gracefully; the merge audit will catch residuals


def fuzzy_match_names(
    unmatched: list[str],
    canonical_names: list[str],
    cutoff: float = 0.82
) -> dict[str, str]:
    """
    use difflib to find close matches for names that didn't match exactly.
    returns a dict of {unmatched_name: best_canonical_match}.
    """
    extra_map = {}
    for name in unmatched:
        matches = get_close_matches(name, canonical_names, n=1, cutoff=cutoff)
        if matches:
            extra_map[name] = matches[0]
            log.info(f"  fuzzy match: '{name}' → '{matches[0]}'")
    return extra_map


# ── 2. load raw data ──────────────────────────────────────────────────────────

def load_and_clean():
    log.info("loading raw data files...")

    schedule_path = DATA_RAW / "sports_ref_schedule_all.csv"
    stats_path    = DATA_RAW / "sports_ref_team_stats_all.csv"
    odds_path     = DATA_RAW / "odds_api_all.csv"

    for p in [schedule_path, stats_path, odds_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"missing file: {p}\n"
                f"make sure you ran scrape_sports_reference.py and pull_odds_api.py first."
            )

    games = pd.read_csv(schedule_path, parse_dates=["date"])
    stats = pd.read_csv(stats_path)
    odds  = pd.read_csv(odds_path,     parse_dates=["date"])

    log.info(f"  games loaded : {len(games):,}")
    log.info(f"  team stats   : {len(stats):,}")
    log.info(f"  odds rows    : {len(odds):,}")

    # drop neutral-site games from the game results
    # sports-reference uses 'N' in the location/notes column for neutral sites
    if "location" in games.columns:
        n_before = len(games)
        games = games[games["location"].fillna("").str.strip() != "N"].copy()
        log.info(f"  dropped {n_before - len(games):,} neutral-site games")

    return games, stats, odds


# ── 3. normalize names and merge ─────────────────────────────────────────────

def build_merged(games: pd.DataFrame, stats: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    log.info("normalizing team names...")

    # normalize names in both sources
    for col in ["home_team", "away_team"]:
        games[f"{col}_canon"] = games[col].apply(normalize_name)
        odds[f"{col}_canon"]  = odds[col].apply(normalize_name)

    # fuzzy-match any odds names that still differ from the sports-ref names
    sr_names   = set(games["home_team_canon"]).union(games["away_team_canon"])
    odds_names = set(odds["home_team_canon"]).union(odds["away_team_canon"])
    unmatched  = [n for n in odds_names if n not in sr_names]

    if unmatched:
        log.info(f"  {len(unmatched)} odds team names not in sports-ref name set — fuzzy matching...")
        extra = fuzzy_match_names(unmatched, list(sr_names))
        for col in ["home_team_canon", "away_team_canon"]:
            odds[col] = odds[col].replace(extra)

    # ── merge games + odds on (home, away, date ±1 day) ───────────────────
    log.info("merging game results with odds...")

    # add a date-tolerance key: we allow odds date to be ±1 day off from game date
    # (timezone differences between the two sources)
    games["date_key"] = games["date"].dt.normalize()
    odds["date_key"]  = odds["date"].dt.normalize()

    # create shifted versions for the ±1 day tolerance
    odds_minus1 = odds.copy(); odds_minus1["date_key"] += pd.Timedelta(days=1)
    odds_plus1  = odds.copy(); odds_plus1["date_key"]  -= pd.Timedelta(days=1)
    odds_expanded = pd.concat([odds, odds_minus1, odds_plus1], ignore_index=True)

    merge_keys = ["home_team_canon", "away_team_canon", "date_key"]

    merged = games.merge(
        odds_expanded[merge_keys + ["spread_line", "total_line", "n_bookmakers"]],
        on=merge_keys,
        how="inner"
    )

    # if a game got multiple odds rows (from different date offsets), keep first
    merged = merged.sort_values("date_key").drop_duplicates(
        subset=["home_team_canon", "away_team_canon", "season", "date_key"]
    ).copy()

    log.info(f"  merged: {len(merged):,} games matched out of {len(games):,} total")
    unmatched_count = len(games) - len(merged)
    log.info(f"  unmatched game records dropped: {unmatched_count:,}")

    # ── attach team efficiency features ───────────────────────────────────
    log.info("joining team efficiency ratings...")

    stats["team_canon"] = stats["team"].apply(normalize_name)

    # helper: join efficiency for home/away team
    for side in ["home", "away"]:
        merged = merged.merge(
            stats.rename(columns={
                "team_canon": f"{side}_team_canon",
                "off_eff":    f"{side}_off_eff",
                "def_eff":    f"{side}_def_eff",
                "pace":       f"{side}_pace",
            })[[f"{side}_team_canon", "season", f"{side}_off_eff",
                f"{side}_def_eff", f"{side}_pace"]],
            on=[f"{side}_team_canon", "season"],
            how="left"
        )

    # ── final cleanup ──────────────────────────────────────────────────────
    # fill missing efficiency ratings with league averages for that season
    for season, grp in merged.groupby("season"):
        avg_off = grp["home_off_eff"].mean()
        avg_def = grp["home_def_eff"].mean()
        avg_pace= grp["home_pace"].mean()

        for col, avg in [("home_off_eff", avg_off), ("away_off_eff", avg_off),
                         ("home_def_eff", avg_def), ("away_def_eff", avg_def),
                         ("home_pace", avg_pace),   ("away_pace",    avg_pace)]:
            merged.loc[(merged["season"] == season) & merged[col].isna(), col] = avg

    # drop rows still missing spread/total
    n_before = len(merged)
    merged = merged.dropna(subset=["spread_line", "total_line"]).copy()
    log.info(f"  dropped {n_before - len(merged):,} rows with missing odds after fill")

    # select and rename final columns
    merged = merged.rename(columns={
        "home_score": "actual_home",
        "away_score": "actual_away",
        "home_team_canon": "home_team",
        "away_team_canon": "away_team",
    })

    merged["actual_total"] = merged["actual_home"] + merged["actual_away"]
    merged["game_id"]      = range(len(merged))

    final_cols = [
        "game_id", "season", "date",
        "home_team", "away_team",
        "actual_home", "actual_away", "actual_total",
        "total_line", "spread_line",
        "home_off_eff", "home_def_eff",
        "away_off_eff", "away_def_eff",
        "home_pace", "away_pace",
    ]
    return merged[final_cols].sort_values(["season", "date"]).reset_index(drop=True)


# ── 4. validation report ──────────────────────────────────────────────────────

def validate(df: pd.DataFrame):
    log.info("\n── validation report ────────────────────────────────────────────")
    log.info(f"  total games    : {len(df):,}")
    log.info(f"  seasons        : {sorted(df['season'].unique())}")
    log.info(f"  unique teams   : {df['home_team'].nunique()}")
    log.info(f"  date range     : {df['date'].min().date()} → {df['date'].max().date()}")

    log.info(f"\n  actual total   : mean={df['actual_total'].mean():.1f}  "
             f"std={df['actual_total'].std():.1f}  "
             f"[{df['actual_total'].min()}, {df['actual_total'].max()}]")
    log.info(f"  total_line     : mean={df['total_line'].mean():.1f}  "
             f"std={df['total_line'].std():.1f}")

    df["error_total"] = df["actual_total"] - df["total_line"]
    log.info(f"\n  total line bias: mean error = {df['error_total'].mean():.3f} pts  "
             f"(should be ~0)")
    log.info(f"  total line std : {df['error_total'].std():.2f} pts  "
             f"(expect ~12 for D1)")

    # flag suspicious rows for the team to inspect
    suspicious = df[
        (df["actual_total"] < 90) |
        (df["actual_total"] > 220) |
        (df["total_line"]   < 100) |
        (df["total_line"]   > 180) |
        (df["spread_line"].abs() > 45)
    ]
    if len(suspicious):
        log.warning(f"\n  !! {len(suspicious)} suspicious rows flagged — inspect before analysis:")
        log.warning(suspicious[["season","date","home_team","away_team",
                                 "actual_total","total_line","spread_line"]].to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    games, stats, odds = load_and_clean()
    merged = build_merged(games, stats, odds)
    validate(merged)

    out_path = DATA_PROC / "merged_games.csv"
    merged.to_csv(out_path, index=False)
    log.info(f"\nfinal dataset saved → {out_path}  |  shape: {merged.shape}")
    log.info("copy merged_games.csv into the notebook directory and run CS216_Final_Project.ipynb")
