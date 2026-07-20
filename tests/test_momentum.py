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
from mgr_mk.offensive_actions import (
    build_offensive_sequences,
    build_player_offensive_features,
)
from mgr_mk.mvp import (
    add_player_momentum_to_dataset,
    score_mvp_candidates,
    select_match_mvps,
)
from mgr_mk.ml import (
    apply_pca_impact_model,
    compare_manual_and_ml_mvps,
    fit_pca_impact_model,
    select_ml_mvps,
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
    assert {
        "progressive_actions_per90",
        "final_third_entries_per90",
        "box_entries_per90",
        "xg_chain_per90",
        "xg_buildup_per90",
    }.issubset(dataset.columns)


def test_offensive_action_features_are_available():
    match_id = 8650
    events = load_events(match_id)

    sequences = build_offensive_sequences(events, match_id)
    player_features = build_player_offensive_features(events, match_id)

    assert not sequences.empty
    assert not player_features.empty
    assert {"sequence_xg", "ends_with_shot"}.issubset(sequences.columns)
    assert {
        "progressive_actions",
        "final_third_entries",
        "box_entries",
        "xg_chain",
        "xg_buildup",
    }.issubset(player_features.columns)
    assert player_features["progressive_actions"].sum() > 0


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


def test_mvp_candidates_can_be_ranked_for_match():
    matches = load_world_cup_2018_matches()
    dataset = build_player_match_dataset(matches, match_ids=[8650])
    dataset = add_player_momentum_to_dataset(dataset, matches)

    scored = score_mvp_candidates(dataset)
    mvps = select_match_mvps(scored)

    assert not scored.empty
    assert not mvps.empty
    assert scored["mvp_score"].notna().all()
    assert mvps["match_id"].tolist() == [8650]
    assert mvps["is_mvp_eligible"].all()


def test_pca_ml_mvp_iteration_runs_for_match():
    matches = load_world_cup_2018_matches()
    dataset = build_player_match_dataset(matches, match_ids=[8650])
    dataset = add_player_momentum_to_dataset(dataset, matches)
    scored = score_mvp_candidates(dataset)
    manual_mvps = select_match_mvps(scored)

    model = fit_pca_impact_model(scored)
    ml_candidates = apply_pca_impact_model(scored, model)
    ml_mvps = select_ml_mvps(ml_candidates)
    comparison = compare_manual_and_ml_mvps(manual_mvps, ml_mvps)

    assert not ml_candidates.empty
    assert not ml_mvps.empty
    assert ml_candidates["ml_pca_score"].between(0, 1).all()
    assert comparison["match_id"].tolist() == [8650]
