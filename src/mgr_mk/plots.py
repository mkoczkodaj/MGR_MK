import numpy as np
import matplotlib.pyplot as plt

from .data_loader import team_name
from .momentum import (
    build_match_momentum,
    cumulative_momentum,
    cumulative_xg,
    goal_events,
    player_momentum_by_window,
)


HOME_COLOR = "#2f80ed"
AWAY_COLOR = "#37e119"


def _step_values(timeline, team, value_col, max_minute):
    team_timeline = timeline[timeline["team.name"].eq(team)].copy()
    if team_timeline.empty:
        return [0, max_minute], [0, 0]

    x_values = [0, *team_timeline["match_minute"].tolist(), max_minute]
    y_values = [0, *team_timeline[value_col].tolist(), team_timeline[value_col].iloc[-1]]
    return x_values, y_values


def plot_match_momentum(
    events,
    matches,
    match_id,
    window=5,
    annotate_players=True,
    max_player_labels=10,
):
    match_info = matches.loc[matches["match_id"].eq(match_id)].iloc[0]
    momentum_table, teams, momentum_events = build_match_momentum(
        events, match_info, window=window
    )
    home_team, away_team = teams
    max_minute = max(momentum_table.index) + window
    home_score = match_info["home_score"]
    away_score = match_info["away_score"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    colors = np.where(momentum_table["momentum_diff"].ge(0), HOME_COLOR, AWAY_COLOR)
    axes[0].bar(
        momentum_table.index + window / 2,
        momentum_table["momentum_diff"],
        width=window * 0.85,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].axhline(0, color="#222222", linewidth=1)
    axes[0].set_ylabel("Momentum")
    axes[0].set_title(
        f"Match momentum: {home_team} {home_score} - {away_score} {away_team}"
    )
    axes[0].grid(axis="y", alpha=0.25)
    if annotate_players:
        _annotate_momentum_players(
            axes[0],
            momentum_table,
            momentum_events,
            teams,
            window=window,
            max_labels=max_player_labels,
        )

    timeline = cumulative_momentum(momentum_events)
    for team, color in [(home_team, HOME_COLOR), (away_team, AWAY_COLOR)]:
        x_values, y_values = _step_values(timeline, team, "cum_momentum", max_minute)
        axes[1].step(
            x_values,
            y_values,
            where="post",
            label=team,
            color=color,
            linewidth=2,
        )

    _draw_goal_markers(axes, goal_events(momentum_events), home_team)
    axes[1].set_ylabel("Skumulowane momentum")
    axes[1].set_xlabel("Minuta meczu")
    axes[1].legend(loc="upper left", frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xlim(0, max_minute)

    plt.tight_layout()
    return fig, momentum_table


def _short_player_name(player_name):
    parts = str(player_name).split()
    if len(parts) <= 1:
        return str(player_name)
    return parts[-1]


def _top_player_for_bar(player_windows, time_bin, team):
    candidates = player_windows[
        player_windows["time_bin"].eq(time_bin)
        & player_windows["team.name"].eq(team)
    ]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _annotate_momentum_players(
    ax,
    momentum_table,
    momentum_events,
    teams,
    window,
    max_labels,
):
    player_windows = player_momentum_by_window(momentum_events)
    if player_windows.empty:
        return

    home_team, away_team = teams
    label_candidates = momentum_table.copy()
    label_candidates["abs_momentum_diff"] = label_candidates["momentum_diff"].abs()
    label_candidates = label_candidates[label_candidates["abs_momentum_diff"].gt(0)]
    label_candidates = label_candidates.sort_values(
        "abs_momentum_diff",
        ascending=False,
    ).head(max_labels)

    top = ax.get_ylim()[1]
    bottom = ax.get_ylim()[0]
    y_padding = (top - bottom) * 0.035

    for time_bin, row in label_candidates.iterrows():
        dominant_team = home_team if row["momentum_diff"] >= 0 else away_team
        player_row = _top_player_for_bar(player_windows, time_bin, dominant_team)
        if player_row is None:
            continue

        x = time_bin + window / 2
        y = row["momentum_diff"]
        va = "bottom" if y >= 0 else "top"
        y_text = y + y_padding if y >= 0 else y - y_padding
        ax.text(
            x,
            y_text,
            _short_player_name(player_row["player.name"]),
            ha="center",
            va=va,
            fontsize=8,
            rotation=90,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#d5d9e0",
                "alpha": 0.85,
            },
        )


def plot_cumulative_xg(events, matches, match_id, window=5):
    match_info = matches.loc[matches["match_id"].eq(match_id)].iloc[0]
    home_team = team_name(match_info, "home_team")
    away_team = team_name(match_info, "away_team")
    home_score = match_info["home_score"]
    away_score = match_info["away_score"]

    _, _, momentum_events = build_match_momentum(events, match_info, window=window)
    max_minute = int(np.ceil(momentum_events["match_minute"].max() / window) * window)
    timeline = cumulative_xg(events)

    fig, ax = plt.subplots(figsize=(14, 5))
    for team, color in [(home_team, HOME_COLOR), (away_team, AWAY_COLOR)]:
        x_values, y_values = _step_values(timeline, team, "cum_xg", max_minute)
        final_xg = y_values[-1]
        ax.step(
            x_values,
            y_values,
            where="post",
            label=f"{team}: {final_xg:.2f} xG",
            color=color,
            linewidth=2.5,
        )

    _draw_goal_markers([ax], goal_events(momentum_events), home_team)
    ax.set_title(f"Skumulowane xG: {home_team} {home_score} - {away_score} {away_team}")
    ax.set_xlabel("Minuta meczu")
    ax.set_ylabel("Skumulowane xG")
    ax.set_xlim(0, max_minute)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()
    return fig


def _draw_goal_markers(axes, goals, home_team):
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    for _, goal in goals.iterrows():
        minute = goal["match_minute"]
        color = HOME_COLOR if goal["team.name"] == home_team else AWAY_COLOR
        for ax in axes:
            ax.axvline(minute, color=color, linestyle="--", linewidth=1, alpha=0.65)
            top = ax.get_ylim()[1]
            ax.text(
                minute,
                top * 0.92,
                "GOL",
                rotation=90,
                color=color,
                ha="right",
                va="top",
                fontsize=9,
                fontweight="bold",
            )
