import json
from pathlib import Path

import pandas as pd

from .config import STATSBOMB_DATA_DIR


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_competitions(data_dir: Path = STATSBOMB_DATA_DIR) -> pd.DataFrame:
    return pd.DataFrame(load_json(data_dir / "competitions.json"))


def get_competition_season(
    competition_name: str = "FIFA World Cup",
    season_name: str = "2018",
    data_dir: Path = STATSBOMB_DATA_DIR,
) -> tuple[int, int]:
    competitions = load_competitions(data_dir)
    selected = competitions[
        (competitions["competition_name"] == competition_name)
        & (competitions["season_name"] == season_name)
    ]
    if selected.empty:
        raise ValueError(f"No competition found for {competition_name} {season_name}")

    row = selected.iloc[0]
    return int(row["competition_id"]), int(row["season_id"])


def load_matches(
    competition_id: int,
    season_id: int,
    data_dir: Path = STATSBOMB_DATA_DIR,
) -> pd.DataFrame:
    path = data_dir / "matches" / str(competition_id) / f"{season_id}.json"
    return pd.DataFrame(load_json(path))


def load_world_cup_2018_matches(data_dir: Path = STATSBOMB_DATA_DIR) -> pd.DataFrame:
    competition_id, season_id = get_competition_season(data_dir=data_dir)
    return load_matches(competition_id, season_id, data_dir)


def load_events(match_id: int, data_dir: Path = STATSBOMB_DATA_DIR) -> pd.DataFrame:
    path = data_dir / "events" / f"{match_id}.json"
    return pd.json_normalize(load_json(path))


def load_lineups(match_id: int, data_dir: Path = STATSBOMB_DATA_DIR):
    path = data_dir / "lineups" / f"{match_id}.json"
    return load_json(path)


def team_name(match_row: pd.Series, column: str) -> str:
    value = match_row[column]
    if isinstance(value, dict):
        return value.get(f"{column}_name")
    return value

