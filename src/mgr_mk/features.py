import pandas as pd


def build_player_match_features(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    df = events.copy()
    df["match_id"] = match_id
    df["is_pass"] = df["type.name"] == "Pass"
    df["is_shot"] = df["type.name"] == "Shot"
    df["is_carry"] = df["type.name"] == "Carry"
    df["is_duel"] = df["type.name"].isin(["Duel", "Interception"])
    df["completed_pass"] = df["is_pass"] & df["pass.outcome.name"].isna()
    df["shot_xg"] = df["shot.statsbomb_xg"].fillna(0)
    df["is_pressure"] = df["type.name"] == "Pressure"
    df["is_interception"] = df["type.name"] == "Interception"

    player_match = (
        df.groupby(["match_id", "player.id", "player.name", "team.name"])
        .agg(
            passes=("is_pass", "sum"),
            shots=("is_shot", "sum"),
            carries=("is_carry", "sum"),
            duels=("is_duel", "sum"),
            events=("type.name", "count"),
            passes_completed=("completed_pass", "sum"),
            total_xg=("shot_xg", "sum"),
            pressures=("is_pressure", "sum"),
            interceptions=("is_interception", "sum"),
            first_minute=("minute", "min"),
            last_minute=("minute", "max"),
        )
        .reset_index()
    )

    player_match["minutes_played"] = (
        player_match["last_minute"] - player_match["first_minute"] + 1
    )
    player_match["pass_accuracy"] = (
        player_match["passes_completed"] / player_match["passes"] * 100
    )
    player_match["pass_accuracy"] = player_match["pass_accuracy"].fillna(0)

    per90_cols = [
        "passes",
        "shots",
        "carries",
        "duels",
        "events",
        "total_xg",
        "pressures",
        "interceptions",
    ]
    for col in per90_cols:
        player_match[f"{col}_per90"] = (
            player_match[col] / player_match["minutes_played"] * 90
        )

    return player_match[
        [
            "match_id",
            "player.id",
            "player.name",
            "team.name",
            "minutes_played",
            "passes_per90",
            "shots_per90",
            "carries_per90",
            "duels_per90",
            "pressures_per90",
            "interceptions_per90",
            "total_xg_per90",
            "pass_accuracy",
            "events_per90",
        ]
    ]

