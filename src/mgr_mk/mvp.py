from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_loader import load_events
from .momentum import build_match_momentum, player_momentum_contributions


@dataclass(frozen=True)
class MVPWeights:

    momentum: float = 0.30
    attacking: float = 0.20
    offensive_building: float = 0.20
    involvement: float = 0.15
    defending: float = 0.10
    result: float = 0.05


RATE_FEATURES = [
    "shots_per90",
    "total_xg_per90",
    "carries_per90",
    "duels_per90",
    "pressures_per90",
    "interceptions_per90",
    "passes_per90",
    "events_per90",
    "progressive_actions_per90",
    "final_third_entries_per90",
    "box_entries_per90",
    "shot_ending_sequence_involvements_per90",
    "xg_chain_per90",
    "xg_buildup_per90",
]


def _minmax(series: pd.Series) -> pd.Series:

    series = series.fillna(0).astype(float)
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def _result_score(result: str) -> float:
    """Convert match result into a small contextual bonus.

    This bonus is intentionally weak. The MVP should mostly come from player
    actions, but the result gives a small nudge to players whose contribution
    translated into a win.
    """

    if result == "win":
        return 1.0
    if result == "draw":
        return 0.5
    return 0.0


def add_player_momentum_to_dataset(
    player_match_dataset: pd.DataFrame,
    matches: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """Attach individual player momentum to the player-match dataset.

    The existing dataset already contains team-level momentum, but MVP is about
    individual contribution. For every match, this function:
    1. loads the event data,
    2. rebuilds match momentum,
    3. sums every player's positive momentum events,
    4. merges that value back into the player-match table.

    The merge uses `match_id`, `team.name`, and `player.name`, because the
    StatsBomb event data and the aggregated dataset both expose these columns.
    """

    all_contributions = []
    for match_id in player_match_dataset["match_id"].dropna().unique():
        match_id = int(match_id)
        match_info = matches.loc[matches["match_id"].eq(match_id)].iloc[0]
        events = load_events(match_id)
        _, _, momentum_events = build_match_momentum(events, match_info, window=window)
        contributions = player_momentum_contributions(momentum_events)
        contributions["match_id"] = match_id
        all_contributions.append(contributions)

    enriched = player_match_dataset.copy()
    if not all_contributions:
        enriched["player_momentum"] = 0.0
        enriched["momentum_events"] = 0
        return enriched

    contribution_table = pd.concat(all_contributions, ignore_index=True)
    enriched = enriched.merge(
        contribution_table,
        on=["match_id", "team.name", "player.name"],
        how="left",
    )
    enriched["player_momentum"] = enriched["player_momentum"].fillna(0)
    enriched["momentum_events"] = enriched["momentum_events"].fillna(0).astype(int)
    return enriched


def score_mvp_candidates(
    player_match_dataset: pd.DataFrame,
    min_minutes: int = 30,
    full_minutes_reference: int = 60,
    weights: MVPWeights = MVPWeights(),
) -> pd.DataFrame:
    """Calculate an MVP ranking for every match.

    How substitutes are handled:
    - A substitute is not removed from the table.
    - `is_mvp_eligible` marks whether the player crossed `min_minutes`.
    - Rate features such as `shots_per90` are multiplied by `minute_factor`.

    Why the minute correction matters:
    A player who plays 5 minutes and takes 1 shot can have a huge per90 value.
    That is mathematically correct, but not fair for MVP ranking. Therefore:

        minute_factor = min(minutes_played / full_minutes_reference, 1)

    If a player played 30 minutes and `full_minutes_reference=60`, their per90
    features count at 50% strength. If they played 60+ minutes, they count fully.
    This keeps super-subs visible, but prevents tiny samples from dominating.
    """

    scored = player_match_dataset.copy()
    if "player_momentum" not in scored.columns:
        scored["player_momentum"] = 0.0
    if "momentum_events" not in scored.columns:
        scored["momentum_events"] = 0
    for feature in RATE_FEATURES:
        if feature not in scored.columns:
            scored[feature] = 0.0

    scored["minutes_played"] = scored["minutes_played"].fillna(0).astype(float)
    scored["minute_factor"] = (
        scored["minutes_played"] / full_minutes_reference
    ).clip(lower=0, upper=1)
    scored["is_mvp_eligible"] = scored["minutes_played"] >= min_minutes
    scored["result_score"] = scored["result"].apply(_result_score)

    for feature in RATE_FEATURES:
        adjusted_feature = f"{feature}_minute_adjusted"
        scored[adjusted_feature] = scored[feature].fillna(0) * scored["minute_factor"]

    scored["pass_accuracy_adjusted"] = (
        scored["pass_accuracy"].fillna(0) / 100
    ) * scored["minute_factor"]

    normalized_parts = []
    for match_id, match_rows in scored.groupby("match_id", sort=False):
        match_rows = match_rows.copy()

        # Normalize individual features inside the match. This means MVP score is
        # a within-match ranking, not a cross-tournament absolute rating.
        match_rows["norm_player_momentum"] = _minmax(match_rows["player_momentum"])
        match_rows["norm_total_xg"] = _minmax(
            match_rows["total_xg_per90_minute_adjusted"]
        )
        match_rows["norm_shots"] = _minmax(match_rows["shots_per90_minute_adjusted"])
        match_rows["norm_carries"] = _minmax(
            match_rows["carries_per90_minute_adjusted"]
        )
        match_rows["norm_passes"] = _minmax(match_rows["passes_per90_minute_adjusted"])
        match_rows["norm_events"] = _minmax(match_rows["events_per90_minute_adjusted"])
        match_rows["norm_progressive_actions"] = _minmax(
            match_rows["progressive_actions_per90_minute_adjusted"]
        )
        match_rows["norm_final_third_entries"] = _minmax(
            match_rows["final_third_entries_per90_minute_adjusted"]
        )
        match_rows["norm_box_entries"] = _minmax(
            match_rows["box_entries_per90_minute_adjusted"]
        )
        match_rows["norm_shot_sequence_involvement"] = _minmax(
            match_rows["shot_ending_sequence_involvements_per90_minute_adjusted"]
        )
        match_rows["norm_xg_chain"] = _minmax(match_rows["xg_chain_per90_minute_adjusted"])
        match_rows["norm_xg_buildup"] = _minmax(
            match_rows["xg_buildup_per90_minute_adjusted"]
        )
        match_rows["norm_pressures"] = _minmax(
            match_rows["pressures_per90_minute_adjusted"]
        )
        match_rows["norm_interceptions"] = _minmax(
            match_rows["interceptions_per90_minute_adjusted"]
        )
        match_rows["norm_duels"] = _minmax(match_rows["duels_per90_minute_adjusted"])

        # Components are intentionally broad and explainable. They are easier to
        # defend in a thesis than a single opaque number.
        match_rows["attacking_component"] = match_rows[
            ["norm_total_xg", "norm_shots", "norm_carries"]
        ].mean(axis=1)
        match_rows["offensive_building_component"] = match_rows[
            [
                "norm_progressive_actions",
                "norm_final_third_entries",
                "norm_box_entries",
                "norm_shot_sequence_involvement",
                "norm_xg_chain",
                "norm_xg_buildup",
            ]
        ].mean(axis=1)
        match_rows["involvement_component"] = match_rows[
            ["norm_passes", "norm_events", "pass_accuracy_adjusted"]
        ].mean(axis=1)
        match_rows["defending_component"] = match_rows[
            ["norm_pressures", "norm_interceptions", "norm_duels"]
        ].mean(axis=1)

        match_rows["mvp_score"] = (
            weights.momentum * match_rows["norm_player_momentum"]
            + weights.attacking * match_rows["attacking_component"]
            + weights.offensive_building * match_rows["offensive_building_component"]
            + weights.involvement * match_rows["involvement_component"]
            + weights.defending * match_rows["defending_component"]
            + weights.result * match_rows["result_score"]
        )

        # Rank all players and eligible players separately. The all-player rank
        # lets us spot explosive substitute performances; the eligible rank is a
        # safer default MVP ranking.
        match_rows["mvp_rank_all_players"] = match_rows["mvp_score"].rank(
            ascending=False,
            method="first",
        )
        eligible_scores = match_rows["mvp_score"].where(match_rows["is_mvp_eligible"])
        match_rows["mvp_rank_eligible"] = eligible_scores.rank(
            ascending=False,
            method="first",
        )
        normalized_parts.append(match_rows)

    return pd.concat(normalized_parts, ignore_index=True)


def select_match_mvps(scored_candidates: pd.DataFrame) -> pd.DataFrame:
    """Return one MVP per match using the safer eligible-player ranking.

    If a match somehow has no eligible player, the function falls back to the
    best player from the all-player ranking.
    """

    mvps = []
    for match_id, match_rows in scored_candidates.groupby("match_id", sort=False):
        eligible = match_rows[match_rows["mvp_rank_eligible"].eq(1)]
        if eligible.empty:
            winner = match_rows.sort_values("mvp_rank_all_players").head(1)
        else:
            winner = eligible.head(1)
        mvps.append(winner)

    columns = [
        "match_id",
        "match_date",
        "competition_stage",
        "player.name",
        "team.name",
        "opponent",
        "result",
        "minutes_played",
        "player_momentum",
        "mvp_score",
        "mvp_rank_all_players",
        "mvp_rank_eligible",
        "is_mvp_eligible",
    ]
    return pd.concat(mvps, ignore_index=True)[columns]
