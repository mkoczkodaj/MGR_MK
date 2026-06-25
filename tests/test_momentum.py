from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mgr_mk.data_loader import load_events, load_world_cup_2018_matches
from mgr_mk.datasets import build_player_match_dataset
from mgr_mk.momentum import (
    build_match_momentum,
    goal_events,
    player_momentum_by_window,
    player_momentum_contributions,
)


def test_brazil_belgium_detects_all_goals():
    match_id = 8650
    matches = load_world_cup_2018_matches()
    events = load_events(match_id)
    match_info = matches.loc[matches["match_id"].eq(match_id)].iloc[0]

    _, _, momentum_events = build_match_momentum(events, match_info)
    goals = goal_events(momentum_events)

    assert len(goals) == 3
    assert goals["team.name"].tolist() == ["Belgium", "Belgium", "Brazil"]


def test_player_match_dataset_contains_match_context():
    matches = load_world_cup_2018_matches()
    dataset = build_player_match_dataset(matches, match_ids=[8650])

    assert not dataset.empty
    assert dataset["match_id"].nunique() == 1
    assert {"opponent", "result", "team_momentum", "team_xg"}.issubset(dataset.columns)
    assert dataset["opponent"].notna().all()
    assert dataset["result"].isin(["win", "draw", "loss"]).all()


def test_player_momentum_contributions_are_available():
    match_id = 8650
    matches = load_world_cup_2018_matches()
    events = load_events(match_id)
    match_info = matches.loc[matches["match_id"].eq(match_id)].iloc[0]

    _, _, momentum_events = build_match_momentum(events, match_info)
    player_totals = player_momentum_contributions(momentum_events)
    player_windows = player_momentum_by_window(momentum_events)

    assert not player_totals.empty
    assert not player_windows.empty
    assert {"player.name", "player_momentum"}.issubset(player_totals.columns)
