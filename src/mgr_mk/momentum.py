import numpy as np
import pandas as pd

from .config import MomentumWeights
from .data_loader import team_name


def safe_series(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def x_coordinate(value):
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return np.nan


def build_match_momentum(
    events: pd.DataFrame,
    match_info: pd.Series,
    window: int = 5,
    weights: MomentumWeights = MomentumWeights(),
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    momentum = events.copy()
    momentum["match_minute"] = (
        momentum["minute"] + safe_series(momentum, "second", 0).fillna(0) / 60
    )
    momentum["shot_xg"] = safe_series(momentum, "shot.statsbomb_xg", 0).fillna(0)
    momentum["location_x"] = safe_series(momentum, "location").apply(x_coordinate)
    momentum["carry_end_x"] = safe_series(momentum, "carry.end_location").apply(
        x_coordinate
    )
    momentum["pass_end_x"] = safe_series(momentum, "pass.end_location").apply(
        x_coordinate
    )

    completed_pass = safe_series(momentum, "pass.outcome.name").isna()
    shot_goal = momentum["type.name"].eq("Shot") & safe_series(
        momentum, "shot.outcome.name"
    ).eq("Goal")
    shot_score = np.where(
        momentum["type.name"].eq("Shot"),
        weights.shot_base + weights.shot_xg * momentum["shot_xg"],
        0,
    )
    shot_score = np.where(shot_goal, weights.goal, shot_score)

    pass_into_final_third = (
        momentum["type.name"].eq("Pass")
        & completed_pass
        & momentum["location_x"].lt(80)
        & momentum["pass_end_x"].ge(80)
    )
    carry_into_final_third = (
        momentum["type.name"].eq("Carry")
        & momentum["location_x"].lt(80)
        & momentum["carry_end_x"].ge(80)
    )
    complete_dribble = momentum["type.name"].eq("Dribble") & safe_series(
        momentum, "dribble.outcome.name"
    ).eq("Complete")
    recovery = momentum["type.name"].eq("Ball Recovery")
    own_goal_for = momentum["type.name"].eq("Own Goal For")

    momentum["momentum_score"] = (
        shot_score
        + own_goal_for.astype(float) * weights.own_goal
        + pass_into_final_third.astype(float) * weights.final_third_pass
        + carry_into_final_third.astype(float) * weights.final_third_carry
        + complete_dribble.astype(float) * weights.complete_dribble
        + recovery.astype(float) * weights.recovery
    )
    momentum["time_bin"] = (momentum["match_minute"] // window * window).astype(int)

    teams = [team_name(match_info, "home_team"), team_name(match_info, "away_team")]
    bins = np.arange(
        0,
        int(np.ceil(momentum["match_minute"].max() / window) * window) + window,
        window,
    )

    grouped = (
        momentum.groupby(["time_bin", "team.name"], as_index=False)["momentum_score"]
        .sum()
        .pivot(index="time_bin", columns="team.name", values="momentum_score")
        .reindex(index=bins, columns=teams)
        .fillna(0)
    )
    grouped["momentum_diff"] = grouped[teams[0]] - grouped[teams[1]]
    return grouped, teams, momentum


def goal_events(momentum_events: pd.DataFrame) -> pd.DataFrame:
    shot_goals = momentum_events[
        momentum_events["type.name"].eq("Shot")
        & safe_series(momentum_events, "shot.outcome.name").eq("Goal")
    ][["match_minute", "team.name"]].copy()
    own_goals = momentum_events[momentum_events["type.name"].eq("Own Goal For")][
        ["match_minute", "team.name"]
    ].copy()
    return (
        pd.concat([shot_goals, own_goals], ignore_index=True)
        .sort_values("match_minute")
        .reset_index(drop=True)
    )


def cumulative_momentum(momentum_events: pd.DataFrame) -> pd.DataFrame:
    timeline = (
        momentum_events[momentum_events["momentum_score"].gt(0)]
        .groupby(["match_minute", "team.name"], as_index=False)["momentum_score"]
        .sum()
        .sort_values("match_minute")
    )
    timeline["cum_momentum"] = timeline.groupby("team.name")["momentum_score"].cumsum()
    return timeline


def player_momentum_contributions(momentum_events: pd.DataFrame) -> pd.DataFrame:
    player_events = momentum_events[
        momentum_events["momentum_score"].gt(0)
        & momentum_events["player.name"].notna()
    ].copy()

    if player_events.empty:
        return pd.DataFrame(
            columns=[
                "team.name",
                "player.name",
                "player_momentum",
                "momentum_events",
            ]
        )

    return (
        player_events.groupby(["team.name", "player.name"], as_index=False)
        .agg(
            player_momentum=("momentum_score", "sum"),
            momentum_events=("momentum_score", "count"),
        )
        .sort_values("player_momentum", ascending=False)
        .reset_index(drop=True)
    )


def player_momentum_by_window(momentum_events: pd.DataFrame) -> pd.DataFrame:
    player_events = momentum_events[
        momentum_events["momentum_score"].gt(0)
        & momentum_events["player.name"].notna()
    ].copy()

    if player_events.empty:
        return pd.DataFrame(
            columns=[
                "time_bin",
                "team.name",
                "player.name",
                "player_momentum",
            ]
        )

    return (
        player_events.groupby(
            ["time_bin", "team.name", "player.name"],
            as_index=False,
        )["momentum_score"]
        .sum()
        .rename(columns={"momentum_score": "player_momentum"})
        .sort_values(
            ["time_bin", "team.name", "player_momentum"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )


def cumulative_xg(events: pd.DataFrame) -> pd.DataFrame:
    xg_events = events.copy()
    xg_events["match_minute"] = (
        xg_events["minute"] + safe_series(xg_events, "second", 0).fillna(0) / 60
    )
    xg_events["shot_xg"] = safe_series(xg_events, "shot.statsbomb_xg", 0).fillna(0)
    shots = xg_events[xg_events["type.name"].eq("Shot")].copy()
    timeline = (
        shots.groupby(["match_minute", "team.name"], as_index=False)["shot_xg"]
        .sum()
        .sort_values("match_minute")
    )
    timeline["cum_xg"] = timeline.groupby("team.name")["shot_xg"].cumsum()
    return timeline
