from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mgr_mk.data_loader import load_events, load_world_cup_2018_matches
from mgr_mk.momentum import build_match_momentum, goal_events


def test_brazil_belgium_detects_all_goals():
    match_id = 8650
    matches = load_world_cup_2018_matches()
    events = load_events(match_id)
    match_info = matches.loc[matches["match_id"].eq(match_id)].iloc[0]

    _, _, momentum_events = build_match_momentum(events, match_info)
    goals = goal_events(momentum_events)

    assert len(goals) == 3
    assert goals["team.name"].tolist() == ["Belgium", "Belgium", "Brazil"]
