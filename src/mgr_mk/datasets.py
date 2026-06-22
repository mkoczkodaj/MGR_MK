import pandas as pd

from .data_loader import load_events, team_name
from .features import build_player_match_features
from .momentum import build_match_momentum, cumulative_xg


def _dict_name(value):
    if isinstance(value, dict):
        return value.get("name")
    return value


def _result(team_score: int, opponent_score: int) -> str:
    if team_score > opponent_score:
        return "win"
    if team_score < opponent_score:
        return "loss"
    return "draw"


def _team_context(match_info: pd.Series, team: str) -> dict:
    home_team = team_name(match_info, "home_team")
    away_team = team_name(match_info, "away_team")

    if team == home_team:
        opponent = away_team
        team_score = int(match_info["home_score"])
        opponent_score = int(match_info["away_score"])
        venue = "home"
    elif team == away_team:
        opponent = home_team
        team_score = int(match_info["away_score"])
        opponent_score = int(match_info["home_score"])
        venue = "away"
    else:
        opponent = None
        team_score = None
        opponent_score = None
        venue = None

    return {
        "opponent": opponent,
        "team_score": team_score,
        "opponent_score": opponent_score,
        "venue": venue,
        "result": _result(team_score, opponent_score)
        if team_score is not None and opponent_score is not None
        else None,
    }


def _team_match_metrics(events: pd.DataFrame, match_info: pd.Series, window: int) -> pd.DataFrame:
    momentum_table, teams, _ = build_match_momentum(events, match_info, window=window)
    momentum_totals = momentum_table[teams].sum()

    xg_timeline = cumulative_xg(events)
    if xg_timeline.empty:
        xg_totals = pd.Series(0.0, index=teams)
    else:
        xg_totals = xg_timeline.groupby("team.name")["cum_xg"].max().reindex(teams).fillna(0)

    rows = []
    for team in teams:
        opponent = teams[1] if team == teams[0] else teams[0]
        rows.append(
            {
                "team.name": team,
                "team_momentum": float(momentum_totals.get(team, 0)),
                "opponent_momentum": float(momentum_totals.get(opponent, 0)),
                "team_xg": float(xg_totals.get(team, 0)),
                "opponent_xg": float(xg_totals.get(opponent, 0)),
            }
        )

    return pd.DataFrame(rows)


def build_player_match_dataset(
    matches: pd.DataFrame,
    match_ids: list[int] | None = None,
    window: int = 5,
) -> pd.DataFrame:
    selected_matches = matches.copy()
    if match_ids is not None:
        selected_matches = selected_matches[selected_matches["match_id"].isin(match_ids)]

    datasets = []
    for _, match_info in selected_matches.iterrows():
        match_id = int(match_info["match_id"])
        events = load_events(match_id)
        player_features = build_player_match_features(events, match_id)
        team_metrics = _team_match_metrics(events, match_info, window=window)

        player_features = player_features.merge(team_metrics, on="team.name", how="left")
        player_features["match_date"] = match_info["match_date"]
        player_features["match_week"] = match_info.get("match_week")
        player_features["competition_stage"] = _dict_name(
            match_info.get("competition_stage")
        )
        player_features["home_team"] = team_name(match_info, "home_team")
        player_features["away_team"] = team_name(match_info, "away_team")
        player_features["home_score"] = int(match_info["home_score"])
        player_features["away_score"] = int(match_info["away_score"])

        contexts = player_features["team.name"].apply(lambda team: _team_context(match_info, team))
        context_df = pd.DataFrame(contexts.tolist(), index=player_features.index)
        player_features = pd.concat([player_features, context_df], axis=1)
        datasets.append(player_features)

    if not datasets:
        return pd.DataFrame()

    dataset = pd.concat(datasets, ignore_index=True)
    ordered_columns = [
        "match_id",
        "match_date",
        "match_week",
        "competition_stage",
        "player.id",
        "player.name",
        "team.name",
        "opponent",
        "venue",
        "result",
        "team_score",
        "opponent_score",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
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
        "team_momentum",
        "opponent_momentum",
        "team_xg",
        "opponent_xg",
    ]
    return dataset[ordered_columns]
