import numpy as np
import pandas as pd


# These features come from `score_mvp_candidates` in mvp.py. They are already
# normalized or composed in an interpretable way inside a match, so they are a
# good starting point for a first ML iteration.
#
# Important methodological note:
# We do not have official labels saying "this player was objectively the MVP".
# Therefore this module does not train a supervised classifier/regressor. The
# first ML step is unsupervised PCA, which looks for the main axis of variation
# in player profiles and turns it into an alternative MVP-like score.
DEFAULT_ML_FEATURES = [
    "norm_player_momentum",
    "attacking_component",
    "offensive_building_component",
    "involvement_component",
    "defending_component",
    "result_score",
    "minute_factor",
]


def _safe_standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features before PCA.

    PCA is sensitive to scale. Even though most inputs are already in a 0-1
    range, standardization makes each feature comparable around its own mean.

    If a column has zero standard deviation, it contains no information for PCA.
    We set its std to 1 to avoid division by zero; after centering, that column
    becomes all zeros and does not influence the components.
    """

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    std = np.where(std == 0, 1, std)
    standardized = (matrix - mean) / std
    return standardized, mean, std


def _minmax_per_group(frame: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Normalize a score to 0-1 inside each match.

    We rank MVP candidates inside a match, not across the full tournament. This
    helper rescales the ML score per `match_id`, so the best candidate in a match
    is close to 1 and weaker candidates are closer to 0.
    """

    def scale(group: pd.Series) -> pd.Series:
        min_value = group.min()
        max_value = group.max()
        if max_value == min_value:
            return pd.Series(0.0, index=group.index)
        return (group - min_value) / (max_value - min_value)

    return frame.groupby(group_col)[value_col].transform(scale)


def fit_pca_impact_model(
    scored_candidates: pd.DataFrame,
    features: list[str] | None = None,
    n_components: int = 2,
) -> dict:
    """Fit a small PCA model on scored player-match rows.

    What happens here:
    1. We select numerical player-profile features.
    2. Missing values are replaced with 0.
    3. Features are standardized.
    4. SVD is used to calculate PCA components.
    5. The first component is oriented so that higher values generally mean a
       stronger player profile.

    Why SVD instead of scikit-learn:
    This keeps the project lightweight and avoids adding another dependency just
    for a first PCA iteration. Mathematically this is the same core operation
    used by common PCA implementations.
    """

    features = features or DEFAULT_ML_FEATURES
    available_features = [feature for feature in features if feature in scored_candidates]
    if not available_features:
        raise ValueError("None of the requested ML features are available.")

    matrix = scored_candidates[available_features].fillna(0).astype(float).to_numpy()
    standardized, mean, std = _safe_standardize(matrix)

    # SVD factorizes the standardized matrix into directions of greatest
    # variance. Rows are player-match observations, columns are features.
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    components = vt[:n_components]

    total_variance = (singular_values**2).sum()
    explained_variance_ratio = (
        (singular_values[:n_components] ** 2) / total_variance
        if total_variance > 0
        else np.zeros(n_components)
    )

    # PCA signs are arbitrary: a component can point in the positive or negative
    # direction and still be mathematically valid. Because all selected features
    # are "more is better" signals, we flip PC1 if its average loading is
    # negative. This makes high PC1 easier to interpret as high impact.
    orientation = 1
    if components[0].mean() < 0:
        components[0] = -components[0]
        orientation = -1

    return {
        "features": available_features,
        "mean": mean,
        "std": std,
        "components": components,
        "explained_variance_ratio": explained_variance_ratio,
        "orientation": orientation,
    }


def apply_pca_impact_model(
    scored_candidates: pd.DataFrame,
    model: dict,
) -> pd.DataFrame:
    """Apply a fitted PCA impact model and rank candidates per match.

    Output columns:
    - `ml_pc1`: raw first principal component,
    - `ml_pc2`: raw second principal component if available,
    - `ml_pca_score`: PC1 scaled 0-1 inside each match,
    - `ml_rank_all_players`: ML rank among all players in the match,
    - `ml_rank_eligible`: ML rank among players above the minutes threshold.
    """

    result = scored_candidates.copy()
    features = model["features"]
    matrix = result[features].fillna(0).astype(float).to_numpy()
    standardized = (matrix - model["mean"]) / model["std"]
    scores = standardized @ model["components"].T

    result["ml_pc1"] = scores[:, 0]
    if scores.shape[1] > 1:
        result["ml_pc2"] = scores[:, 1]
    else:
        result["ml_pc2"] = 0.0

    result["ml_pca_score"] = _minmax_per_group(result, "match_id", "ml_pc1")
    result["ml_rank_all_players"] = result.groupby("match_id")["ml_pca_score"].rank(
        ascending=False,
        method="first",
    )

    eligible_scores = result["ml_pca_score"].where(result["is_mvp_eligible"])
    result["ml_rank_eligible"] = eligible_scores.groupby(result["match_id"]).rank(
        ascending=False,
        method="first",
    )
    return result


def select_ml_mvps(ml_candidates: pd.DataFrame) -> pd.DataFrame:
    """Select one ML-based MVP per match.

    The default is the best eligible player. If every player in a match is below
    the minutes threshold, the function falls back to the best all-player rank.
    """

    winners = []
    for _, match_rows in ml_candidates.groupby("match_id", sort=False):
        eligible = match_rows[match_rows["ml_rank_eligible"].eq(1)]
        if eligible.empty:
            winner = match_rows.sort_values("ml_rank_all_players").head(1)
        else:
            winner = eligible.head(1)
        winners.append(winner)

    columns = [
        "match_id",
        "match_date",
        "competition_stage",
        "player.name",
        "team.name",
        "opponent",
        "result",
        "minutes_played",
        "mvp_score",
        "ml_pca_score",
        "mvp_rank_eligible",
        "ml_rank_eligible",
        "is_mvp_eligible",
    ]
    return pd.concat(winners, ignore_index=True)[columns]


def compare_manual_and_ml_mvps(
    manual_mvps: pd.DataFrame,
    ml_mvps: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the hand-built MVP score with the ML/PCA MVP score.

    This is useful for interpretation. If both methods select the same player,
    the match is stable under two different assumptions. If they disagree, that
    match becomes a good case study for the thesis.
    """

    manual = manual_mvps[
        ["match_id", "player.name", "team.name", "mvp_score"]
    ].rename(
        columns={
            "player.name": "manual_mvp_player",
            "team.name": "manual_mvp_team",
            "mvp_score": "manual_mvp_score",
        }
    )
    ml = ml_mvps[["match_id", "player.name", "team.name", "ml_pca_score"]].rename(
        columns={
            "player.name": "ml_mvp_player",
            "team.name": "ml_mvp_team",
            "ml_pca_score": "ml_mvp_score",
        }
    )
    comparison = manual.merge(ml, on="match_id", how="inner")
    comparison["same_player"] = (
        comparison["manual_mvp_player"] == comparison["ml_mvp_player"]
    )
    return comparison
