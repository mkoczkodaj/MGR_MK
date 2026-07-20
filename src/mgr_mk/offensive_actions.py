import numpy as np
import pandas as pd


PITCH_LENGTH = 120
PITCH_WIDTH = 80
FINAL_THIRD_X = 80
PENALTY_BOX_X = 102
PENALTY_BOX_Y_MIN = 18
PENALTY_BOX_Y_MAX = 62


def _empty_player_offensive_features() -> pd.DataFrame:
    columns = [
        "match_id",
        "player.id",
        "player.name",
        "team.name",
        "offensive_sequence_involvements",
        "shot_ending_sequence_involvements",
        "progressive_passes",
        "progressive_carries",
        "final_third_entries",
        "box_entries",
        "passes_into_box",
        "carries_into_box",
        "shot_assists",
        "xg_chain",
        "xg_buildup",
        "offensive_sequences_xg",
        "progressive_actions",
    ]
    return pd.DataFrame(columns=columns)


def _is_location(value) -> bool:
    return isinstance(value, list) and len(value) >= 2


def _x(value):
    return value[0] if _is_location(value) else np.nan


def _y(value):
    return value[1] if _is_location(value) else np.nan


def _in_penalty_box(x: float, y: float) -> bool:
    return (
        pd.notna(x)
        and pd.notna(y)
        and x >= PENALTY_BOX_X
        and PENALTY_BOX_Y_MIN <= y <= PENALTY_BOX_Y_MAX
    )


def _distance_to_goal(x: float, y: float) -> float:
    if pd.isna(x) or pd.isna(y):
        return np.nan
    return float(np.hypot(PITCH_LENGTH - x, (PITCH_WIDTH / 2) - y))


def _is_progressive_action(start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
    """Return whether an action meaningfully moves the ball toward goal.

    The threshold follows a common event-data idea: the required progress is
    bigger in a team's own half and smaller near the opponent's goal, because
    five meters close to the box can be more valuable than five meters near the
    halfway line.
    """

    if pd.isna(start_x) or pd.isna(start_y) or pd.isna(end_x) or pd.isna(end_y):
        return False
    if end_x <= start_x:
        return False

    start_distance = _distance_to_goal(start_x, start_y)
    end_distance = _distance_to_goal(end_x, end_y)
    progress = start_distance - end_distance

    if start_x < 60 and end_x < 60:
        threshold = 30
    elif start_x < 60 <= end_x:
        threshold = 15
    else:
        threshold = 10
    return progress >= threshold


def _prepare_offensive_events(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    df = events.copy()
    df["match_id"] = match_id

    for column in [
        "pass.outcome.name",
        "pass.shot_assist",
        "pass.goal_assist",
        "shot.statsbomb_xg",
    ]:
        if column not in df:
            df[column] = np.nan

    df["start_x"] = df["location"].apply(_x)
    df["start_y"] = df["location"].apply(_y)
    df["pass_end_x"] = df["pass.end_location"].apply(_x)
    df["pass_end_y"] = df["pass.end_location"].apply(_y)
    df["carry_end_x"] = df["carry.end_location"].apply(_x)
    df["carry_end_y"] = df["carry.end_location"].apply(_y)
    df["end_x"] = np.where(df["type.name"].eq("Pass"), df["pass_end_x"], df["carry_end_x"])
    df["end_y"] = np.where(df["type.name"].eq("Pass"), df["pass_end_y"], df["carry_end_y"])

    df["is_completed_pass"] = df["type.name"].eq("Pass") & df["pass.outcome.name"].isna()
    df["is_carry"] = df["type.name"].eq("Carry")
    df["is_successful_ball_progression"] = df["is_completed_pass"] | df["is_carry"]
    df["is_progressive_action"] = df.apply(
        lambda row: _is_progressive_action(
            row["start_x"],
            row["start_y"],
            row["end_x"],
            row["end_y"],
        ),
        axis=1,
    )
    df["enters_final_third"] = (
        df["is_successful_ball_progression"]
        & df["start_x"].lt(FINAL_THIRD_X)
        & df["end_x"].ge(FINAL_THIRD_X)
    )
    df["starts_in_box"] = df.apply(
        lambda row: _in_penalty_box(row["start_x"], row["start_y"]),
        axis=1,
    )
    df["ends_in_box"] = df.apply(
        lambda row: _in_penalty_box(row["end_x"], row["end_y"]),
        axis=1,
    )
    df["enters_box"] = (
        df["is_successful_ball_progression"]
        & ~df["starts_in_box"]
        & df["ends_in_box"]
    )
    df["is_shot_assist"] = (
        df["type.name"].eq("Pass")
        & (
            df["pass.shot_assist"].fillna(False).astype(bool)
            | df["pass.goal_assist"].fillna(False).astype(bool)
        )
    )
    df["shot_xg"] = df["shot.statsbomb_xg"].fillna(0)
    return df


def build_offensive_sequences(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Build one row per team possession with shot and xG context.

    In this project an offensive sequence is treated as one StatsBomb possession.
    This is a practical definition: it is stable, available for every match, and
    lets us ask who touched or progressed the ball before a shot appeared.
    """

    df = _prepare_offensive_events(events, match_id)
    if df.empty or "possession" not in df:
        return pd.DataFrame()

    possession_team_events = df[df["team.name"].eq(df["possession_team.name"])].copy()
    if possession_team_events.empty:
        return pd.DataFrame()

    sequence_rows = []
    for (possession, team), group in possession_team_events.groupby(
        ["possession", "possession_team.name"],
        sort=False,
    ):
        shots = group[group["type.name"].eq("Shot")]
        involved_players = sorted(group["player.name"].dropna().unique().tolist())
        sequence_rows.append(
            {
                "match_id": match_id,
                "possession": int(possession),
                "team.name": team,
                "start_minute": int(group["minute"].min()),
                "end_minute": int(group["minute"].max()),
                "event_count": int(len(group)),
                "player_count": int(len(involved_players)),
                "involved_players": involved_players,
                "shot_count": int(len(shots)),
                "sequence_xg": float(shots["shot_xg"].sum()),
                "ends_with_shot": bool(len(shots) > 0),
            }
        )

    return pd.DataFrame(sequence_rows)


def build_player_offensive_features(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Aggregate offensive action-building features to player-match rows.

    The most important columns:
    - `progressive_passes` / `progressive_carries`: actions moving the ball
      meaningfully closer to goal,
    - `final_third_entries` / `box_entries`: actions that enter valuable zones,
    - `xg_chain`: sequence xG credited to every player involved before a shot,
    - `xg_buildup`: sequence xG credited to buildup players, excluding the
      shooter and the direct shot-assist passer.
    """

    df = _prepare_offensive_events(events, match_id)
    if df.empty:
        return _empty_player_offensive_features()

    possession_team_events = df[df["team.name"].eq(df["possession_team.name"])].copy()
    possession_team_events = possession_team_events[
        possession_team_events["player.name"].notna()
    ]
    if possession_team_events.empty:
        return _empty_player_offensive_features()

    action_rows = possession_team_events.copy()
    action_rows["progressive_passes"] = (
        action_rows["is_completed_pass"] & action_rows["is_progressive_action"]
    ).astype(int)
    action_rows["progressive_carries"] = (
        action_rows["is_carry"] & action_rows["is_progressive_action"]
    ).astype(int)
    action_rows["final_third_entries"] = action_rows["enters_final_third"].astype(int)
    action_rows["box_entries"] = action_rows["enters_box"].astype(int)
    action_rows["passes_into_box"] = (
        action_rows["is_completed_pass"] & action_rows["enters_box"]
    ).astype(int)
    action_rows["carries_into_box"] = (
        action_rows["is_carry"] & action_rows["enters_box"]
    ).astype(int)
    action_rows["shot_assists"] = action_rows["is_shot_assist"].astype(int)

    player_actions = (
        action_rows.groupby(["match_id", "player.id", "player.name", "team.name"])
        .agg(
            progressive_passes=("progressive_passes", "sum"),
            progressive_carries=("progressive_carries", "sum"),
            final_third_entries=("final_third_entries", "sum"),
            box_entries=("box_entries", "sum"),
            passes_into_box=("passes_into_box", "sum"),
            carries_into_box=("carries_into_box", "sum"),
            shot_assists=("shot_assists", "sum"),
        )
        .reset_index()
    )

    sequence_rows = []
    for (possession, team), group in possession_team_events.groupby(
        ["possession", "possession_team.name"],
        sort=False,
    ):
        sequence_xg = float(group["shot_xg"].sum())
        has_shot = sequence_xg > 0 or group["type.name"].eq("Shot").any()
        players = group[
            ["match_id", "player.id", "player.name", "team.name"]
        ].drop_duplicates()

        shooters = set(group.loc[group["type.name"].eq("Shot"), "player.name"].dropna())
        direct_creators = set(group.loc[group["is_shot_assist"], "player.name"].dropna())

        for _, player in players.iterrows():
            player_name = player["player.name"]
            is_buildup_player = player_name not in shooters and player_name not in direct_creators
            sequence_rows.append(
                {
                    "match_id": match_id,
                    "player.id": player["player.id"],
                    "player.name": player_name,
                    "team.name": player["team.name"],
                    "offensive_sequence_involvements": 1,
                    "shot_ending_sequence_involvements": int(has_shot),
                    "xg_chain": sequence_xg if has_shot else 0.0,
                    "xg_buildup": sequence_xg if has_shot and is_buildup_player else 0.0,
                    "offensive_sequences_xg": sequence_xg,
                }
            )

    sequence_features = pd.DataFrame(sequence_rows)
    if sequence_features.empty:
        sequence_features = _empty_player_offensive_features()
    else:
        sequence_features = (
            sequence_features.groupby(["match_id", "player.id", "player.name", "team.name"])
            .agg(
                offensive_sequence_involvements=("offensive_sequence_involvements", "sum"),
                shot_ending_sequence_involvements=(
                    "shot_ending_sequence_involvements",
                    "sum",
                ),
                xg_chain=("xg_chain", "sum"),
                xg_buildup=("xg_buildup", "sum"),
                offensive_sequences_xg=("offensive_sequences_xg", "sum"),
            )
            .reset_index()
        )

    features = sequence_features.merge(
        player_actions,
        on=["match_id", "player.id", "player.name", "team.name"],
        how="outer",
    )
    features["progressive_actions"] = (
        features["progressive_passes"].fillna(0)
        + features["progressive_carries"].fillna(0)
    )

    for column in _empty_player_offensive_features().columns:
        if column not in features:
            features[column] = 0

    numeric_columns = [
        column
        for column in features.columns
        if column not in ["player.name", "team.name"]
    ]
    features[numeric_columns] = features[numeric_columns].fillna(0)
    return features[_empty_player_offensive_features().columns]
