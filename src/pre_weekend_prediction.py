"""
pre_weekend_prediction.py — Previsioni Pre-Weekend e Post-FP
============================================================
Abilita previsioni per round futuri (nessuna qualifica/gara corsa):
1. Pre-weekend: solo storico (Elo, circuit_history, recent_form)
2. Post-FP: storico + dati prove libere quando disponibili

Il flusso: stub → feature → predict quali → predict race
"""

import pandas as pd
import numpy as np
import fastf1
import warnings
import sys

sys.path.append("..")
from config import (
    PROCESSED_DATA_DIR,
    ROLLING_WINDOW,
    RANDOM_SEED,
    ELO_K_FACTOR,
    ELO_INITIAL_RATING,
    ELO_FP_BOOTSTRAP_SCALE,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def get_future_round_stub(
    year: int, round_num: int, df_past: pd.DataFrame
) -> pd.DataFrame | None:
    """
    Crea uno stub per un round futuro (driver, team, race_name).

    Usa fastf1 per il calendario e l'ultima gara in df_past per il lineup.
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        races = schedule[schedule["RoundNumber"] == round_num]
        if races.empty:
            return None
        race_name = races.iloc[0]["EventName"]
    except Exception:
        return None

    # Lineup: ultima gara nella stessa stagione, altrimenti stagione precedente
    past_same_season = df_past[df_past["year"] == year]
    past_earlier = df_past[df_past["year"] < year]

    if not past_same_season.empty:
        last_round = past_same_season["round"].max()
        lineup = df_past[
            (df_past["year"] == year) & (df_past["round"] == last_round)
        ][["driver", "team"]].drop_duplicates()
    elif not past_earlier.empty:
        last_year = past_earlier["year"].max()
        last_round = past_earlier[past_earlier["year"] == last_year]["round"].max()
        lineup = df_past[
            (df_past["year"] == last_year) & (df_past["round"] == last_round)
        ][["driver", "team"]].drop_duplicates()
    else:
        return None

    if lineup.empty:
        return None

    stub = pd.DataFrame(
        {
            "year": year,
            "round": round_num,
            "race_name": race_name,
            "driver": lineup["driver"].values,
            "team": lineup["team"].values,
        }
    )
    return stub


def _compute_elo_up_to(
    df_past: pd.DataFrame, year: int, round_num: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Calcola Elo piloti e team fino a (year, round) escluso."""
    from config import REGULATION_RESET_YEARS
    from src.elo import EloSystem

    past = df_past[
        (df_past["year"] < year)
        | ((df_past["year"] == year) & (df_past["round"] < round_num))
    ]
    if past.empty:
        return {}, {}

    elo_system = EloSystem()
    elo_system.process_season(past)
    driver_elo = dict(elo_system.ratings)

    team_elo = EloSystem(k_factor=ELO_K_FACTOR * 0.7)
    last_year = None
    for (y, r), race_df in past.groupby(["year", "round"]):
        if y in REGULATION_RESET_YEARS and last_year is not None and last_year < y:
            team_elo.reset_all_ratings()

            # ── BOOTSTRAP FP (stessa logica di compute_team_elo) ──
            fp_summary_file = PROCESSED_DATA_DIR / "fp_summary.csv"
            if fp_summary_file.exists():
                fp_summary = pd.read_csv(fp_summary_file)
                first_round_fp = fp_summary[fp_summary["year"] == y]
                if not first_round_fp.empty:
                    first_round_num = first_round_fp["round"].min()
                    first_round_fp = first_round_fp[
                        first_round_fp["round"] == first_round_num
                    ]
                    driver_team = dict(
                        zip(race_df["driver"], race_df["team"])
                    )
                    if "fp_best_lap_delta" in first_round_fp.columns:
                        fp_with_team = first_round_fp.copy()
                        fp_with_team["team"] = fp_with_team["driver"].map(
                            driver_team
                        )
                        fp_with_team = fp_with_team.dropna(
                            subset=["team", "fp_best_lap_delta"]
                        )
                        if not fp_with_team.empty:
                            team_pace = fp_with_team.groupby("team")[
                                "fp_best_lap_delta"
                            ].mean()
                            for team, delta in team_pace.items():
                                boost = -delta * ELO_FP_BOOTSTRAP_SCALE
                                team_elo.ratings[team] = (
                                    ELO_INITIAL_RATING + boost
                                )

        last_year = y
        team_results = race_df.groupby("team")["finish_position"].min().reset_index()
        result_list = list(zip(team_results["team"], team_results["finish_position"]))
        team_elo.update_after_race(result_list)
    team_elo_dict = dict(team_elo.ratings)

    return driver_elo, team_elo_dict


def _add_recent_form_for_stub(
    stub: pd.DataFrame, df_past: pd.DataFrame, year: int, round_num: int
) -> pd.DataFrame:
    """Calcola recent_form e recent_dnf_rate per ogni pilota nello stub."""
    past = df_past[
        (df_past["year"] < year)
        | ((df_past["year"] == year) & (df_past["round"] < round_num))
    ].sort_values(["driver", "year", "round"])

    form_map = {}
    dnf_map = {}
    for driver in stub["driver"].unique():
        driver_past = past[past["driver"] == driver]
        if len(driver_past) >= 1:
            last_n = driver_past.tail(ROLLING_WINDOW)
            valid_pos = last_n["finish_position"].dropna()
            form_map[driver] = valid_pos.mean() if len(valid_pos) >= 1 else np.nan
            dnf_map[driver] = last_n["dnf"].mean() if "dnf" in last_n.columns else 0.0
        else:
            form_map[driver] = np.nan
            dnf_map[driver] = 0.0

    stub["recent_form"] = stub["driver"].map(form_map)
    stub["recent_dnf_rate"] = stub["driver"].map(dnf_map)
    return stub


def _add_circuit_history_for_stub(
    stub: pd.DataFrame, df_past: pd.DataFrame, year: int, round_num: int
) -> pd.DataFrame:
    """Calcola circuit_avg_position e circuit_experience per ogni pilota."""
    race_name = stub["race_name"].iloc[0]
    past = df_past[
        (df_past["year"] < year)
        | ((df_past["year"] == year) & (df_past["round"] < round_num))
    ]
    past_at_circuit = past[past["race_name"] == race_name]

    circ_avg = {}
    circ_exp = {}
    for driver in stub["driver"].unique():
        driver_at_circuit = past_at_circuit[past_at_circuit["driver"] == driver]
        valid_pos = driver_at_circuit["finish_position"].dropna()
        if len(valid_pos) > 0:
            circ_avg[driver] = valid_pos.mean()
            circ_exp[driver] = len(valid_pos)
        else:
            circ_avg[driver] = np.nan
            circ_exp[driver] = 0

    stub["circuit_avg_position"] = stub["driver"].map(circ_avg)
    stub["circuit_experience"] = stub["driver"].map(circ_exp)
    return stub


def compute_features_for_future_round(
    stub: pd.DataFrame,
    df_past: pd.DataFrame,
    year: int,
    round_num: int,
    train_data: pd.DataFrame,
    fp_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calcola le feature per un round futuro.
    grid_position resta NaN (verrà predetto da predict_quali_positions).
    """
    driver_elo, team_elo = _compute_elo_up_to(df_past, year, round_num)

    stub["elo_pre_race"] = stub["driver"].map(
        lambda d: driver_elo.get(d, 1500.0)
    )
    stub["team_elo_pre_race"] = stub["team"].map(
        lambda t: team_elo.get(t, 1500.0)
    )
    stub["grid_position"] = np.nan
    stub["finish_position"] = np.nan
    stub["dnf"] = False

    stub = _add_recent_form_for_stub(stub, df_past, year, round_num)
    stub = _add_circuit_history_for_stub(stub, df_past, year, round_num)

    if fp_summary is not None:
        fp_round = fp_summary[
            (fp_summary["year"] == year) & (fp_summary["round"] == round_num)
        ]
        if not fp_round.empty:
            fp_cols = [
                c
                for c in fp_summary.columns
                if c not in ["year", "round", "driver"]
                and c.startswith(("fp_", "fp2_", "fp3_"))
            ]
            available = [c for c in fp_cols if c in fp_round.columns]
            if available:
                stub = stub.merge(
                    fp_round[["year", "round", "driver"] + available],
                    on=["year", "round", "driver"],
                    how="left",
                )

    # Grid features (placeholder, aggiornate dopo predict quali)
    stub["is_front_row"] = 0
    stub["is_top5"] = 0
    stub["is_top10"] = 0

    # Fill NaN con mediana del training
    median_cols = [
        "recent_form",
        "recent_dnf_rate",
        "circuit_avg_position",
        "circuit_experience",
    ]
    for col in median_cols:
        if col in stub.columns and col in train_data.columns:
            stub[col] = stub[col].fillna(train_data[col].median())
    stub["recent_dnf_rate"] = stub["recent_dnf_rate"].fillna(0)
    stub["circuit_experience"] = stub["circuit_experience"].fillna(0)
    stub["recent_form"] = stub["recent_form"].fillna(
        train_data["finish_position"].median() if "finish_position" in train_data.columns else 10.0
    )
    stub["circuit_avg_position"] = stub["circuit_avg_position"].fillna(
        train_data["finish_position"].median() if "finish_position" in train_data.columns else 10.0
    )

    return stub


def predict_quali_positions(
    round_data: pd.DataFrame,
    train_data: pd.DataFrame,
    feature_columns: list | None = None,
) -> pd.DataFrame:
    """
    Predice grid_position per ogni driver usando un modello qualifica.
    Aggiorna is_front_row, is_top5, is_top10 di conseguenza.
    """
    from sklearn.ensemble import RandomForestRegressor

    if feature_columns is None:
        feature_columns = [
            c for c in train_data.columns
            if c not in ("grid_position", "is_front_row", "is_top5", "is_top10", "finish_position")
            and pd.api.types.is_numeric_dtype(train_data[c])
        ]
    quali_feature_cols = [
        c for c in feature_columns
        if c not in ("grid_position", "is_front_row", "is_top5", "is_top10")
        and c in train_data.columns
        and c in round_data.columns
    ]

    if len(quali_feature_cols) < 3:
        # Fallback: ordina per elo_pre_race
        round_data = round_data.sort_values(
            "elo_pre_race", ascending=False
        ).reset_index(drop=True)
        round_data["grid_position"] = range(1, len(round_data) + 1)
    else:
        quali_train = train_data.dropna(subset=["grid_position"])
        if len(quali_train) < 50:
            round_data = round_data.sort_values(
                "elo_pre_race", ascending=False
            ).reset_index(drop=True)
            round_data["grid_position"] = range(1, len(round_data) + 1)
        else:
            X_train = quali_train[quali_feature_cols].copy()
            for col in X_train.columns:
                X_train[col] = X_train[col].fillna(X_train[col].median())
            y_train = quali_train["grid_position"]

            quali_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            quali_model.fit(X_train, y_train)

            X_round = round_data[quali_feature_cols].copy()
            for col in X_round.columns:
                if X_round[col].isna().any():
                    X_round[col] = X_round[col].fillna(X_train[col].median())
            pred_grid = quali_model.predict(X_round)
            round_data["grid_position"] = np.clip(pred_grid, 1, 22)

    round_data["is_front_row"] = (round_data["grid_position"] <= 2).astype(int)
    round_data["is_top5"] = (round_data["grid_position"] <= 5).astype(int)
    round_data["is_top10"] = (round_data["grid_position"] <= 10).astype(int)

    return round_data


def build_future_round_features(
    year: int,
    round_num: int,
    features: pd.DataFrame,
    df_all_races: pd.DataFrame,
) -> tuple[pd.DataFrame | None, bool]:
    """
    Orchestrazione: crea stub, feature, predice quali, restituisce round_data pronto.

    Ritorna (round_data, has_fp).
    round_data è None se fallisce.
    """
    stub = get_future_round_stub(year, round_num, df_all_races)
    if stub is None:
        return None, False

    train_data = features[
        (features["year"] < year)
        | ((features["year"] == year) & (features["round"] < round_num))
    ]
    if len(train_data) < 50:
        return None, False

    fp_summary = None
    fp_file = PROCESSED_DATA_DIR / "fp_summary.csv"
    if fp_file.exists():
        fp_summary = pd.read_csv(fp_file)
        has_fp_data = not fp_summary[
            (fp_summary["year"] == year) & (fp_summary["round"] == round_num)
        ].empty
    else:
        has_fp_data = False

    if not has_fp_data:
        fp_summary = None

    round_data = compute_features_for_future_round(
        stub, df_all_races, year, round_num, train_data, fp_summary
    )

    round_data = predict_quali_positions(round_data, train_data)

    # Add missing columns from train_data (for advanced features)
    for col in train_data.columns:
        if col not in round_data.columns and col not in (
            "finish_position",
            "classified_position",
        ):
            if pd.api.types.is_numeric_dtype(train_data[col]):
                round_data[col] = train_data[col].median()
            elif len(train_data[col].dropna()) > 0:
                round_data[col] = train_data[col].mode().iloc[0]

    return round_data, has_fp_data


def predict_future_round_race(
    round_data: pd.DataFrame,
    predictor,
    train_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predice la classifica di gara per round_data (senza finish_position).
    Non usa predictor.predict() perché droppa righe con target NaN.
    """
    feat_cols = [c for c in predictor.feature_columns if c in round_data.columns]
    if len(feat_cols) < len(predictor.feature_columns):
        missing = set(predictor.feature_columns) - set(feat_cols)
        for col in missing:
            if col in train_data.columns:
                round_data[col] = train_data[col].median()
        feat_cols = predictor.feature_columns

    X = round_data[feat_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(train_data[col].median())

    preds = predictor.model.predict(X)
    result = round_data.copy()
    result["predicted_position"] = preds
    result = result.sort_values("predicted_position").reset_index(drop=True)
    result["predicted_rank"] = range(1, len(result) + 1)
    return result
