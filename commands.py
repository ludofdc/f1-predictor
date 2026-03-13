"""
commands.py — F1 Predictor Command Center
==========================================
Tutte le analisi e simulazioni disponibili.

Uso:
  python commands.py          → Menu interattivo
  python -c "from commands import *; show_elo_rankings()"  → Comando singolo
"""

import pandas as pd
import numpy as np
import io
import sys

from config import PROCESSED_DATA_DIR, SEASONS
from src.data_loader import load_all_data
from src.elo import EloSystem, compute_team_elo
from src.feature_engineering import build_feature_matrix, FEATURE_COLUMNS
from src.model import F1Predictor
from src.evaluation import (
    evaluate_predictions,
    print_evaluation_report,
    per_race_analysis,
)

# Import avanzati (opzionali)
try:
    from src.advanced_features import build_advanced_features, ADVANCED_FEATURE_COLUMNS
    from src.circuit_data import CIRCUIT_DATA, DOWNFORCE_MAP
    ADVANCED = True
except ImportError:
    ADVANCED = False

# Import visualizzazioni FP (opzionali)
try:
    from src.fp_visualizations import (
        plot_long_runs,
        plot_long_run_traces,
        plot_telemetry_top3,
        plot_fp_weekend,
    )
    FP_PLOTS = True
except ImportError:
    FP_PLOTS = False


# ============================================================
# HELPER: carica le feature (avanzate se disponibili, altrimenti base)
# ============================================================

def _load_features() -> pd.DataFrame:
    """Carica il miglior file di feature disponibile."""
    adv_file = PROCESSED_DATA_DIR / "advanced_features.csv"
    base_file = PROCESSED_DATA_DIR / "features.csv"

    if adv_file.exists():
        return pd.read_csv(adv_file)
    elif base_file.exists():
        return pd.read_csv(base_file)
    else:
        raise FileNotFoundError(
            "Nessun file feature trovato. Esegui prima:\n"
            "  python -m src.feature_engineering\n"
            "  oppure: python -m src.advanced_features"
        )


def _train_silent(df: pd.DataFrame, use_advanced: bool = True) -> F1Predictor:
    """Allena un modello senza stampare nulla."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    predictor = F1Predictor(use_advanced=use_advanced)
    predictor.train(df)
    sys.stdout = old
    return predictor


# ============================================================
# 1. ESPLORA I DATI
# ============================================================

def explore_data():
    """Panoramica completa dei dati disponibili."""
    df = load_all_data()

    print("=" * 60)
    print("📊 PANORAMICA DATI")
    print("=" * 60)

    print(f"\nStagioni: {sorted(df['year'].unique())}")
    print(f"Gare totali: {df.groupby(['year', 'round']).ngroups}")
    print(f"Piloti unici: {df['driver'].nunique()}")
    print(f"Team unici: {df['team'].nunique()}")
    print(f"Righe totali: {len(df)}")

    print("\n📅 Gare per stagione:")
    for year, count in df.groupby("year")["round"].nunique().items():
        print(f"  {year}: {count} gare")

    print("\n👤 Piloti nel dataset:")
    drivers = df.groupby("driver")["year"].agg(["min", "max", "count"])
    drivers.columns = ["first_year", "last_year", "total_entries"]
    print(drivers.sort_values("total_entries", ascending=False).to_string())

    return df


def grid_vs_finish():
    """Quanto conta la posizione di partenza?"""
    df = load_all_data()

    print("\n📍 GRIGLIA vs RISULTATO FINALE")
    print("=" * 60)

    grid_analysis = df.groupby("grid_position")["finish_position"].agg(
        ["mean", "std", "count"]
    ).round(2)
    grid_analysis.columns = ["avg_finish", "std", "races"]
    grid_analysis = grid_analysis[grid_analysis["races"] >= 10]

    for grid_pos, row in grid_analysis.head(20).iterrows():
        if pd.isna(grid_pos):
            continue
        gained = grid_pos - row["avg_finish"]
        arrow = "↑" if gained > 0 else "↓" if gained < 0 else "="
        bar_len = int(row["avg_finish"])
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(
            f"  Griglia P{int(grid_pos):2d} → Finish P{row['avg_finish']:5.1f} "
            f"({arrow}{abs(gained):.1f})  {bar}"
        )

    pole_df = df[df["grid_position"] == 1]
    pole_wins = (pole_df["finish_position"] == 1).mean() * 100
    print(f"\n  🏁 Win rate dalla pole: {pole_wins:.1f}%")


def dnf_analysis():
    """Analisi ritiri per pilota e team."""
    df = load_all_data()

    print("\n💥 ANALISI DNF")
    print("=" * 60)

    dnf_by_driver = df.groupby("driver").agg(
        races=("dnf", "count"), dnfs=("dnf", "sum"),
    )
    dnf_by_driver["dnf_rate"] = (dnf_by_driver["dnfs"] / dnf_by_driver["races"] * 100).round(1)

    print("\nDNF rate per pilota:")
    for driver, row in dnf_by_driver.sort_values("dnf_rate", ascending=False).head(15).iterrows():
        bar = "█" * int(row["dnf_rate"] / 2)
        print(f"  {driver:4s}: {row['dnf_rate']:5.1f}% ({int(row['dnfs'])}/{int(row['races'])})  {bar}")

    print("\nDNF rate per team:")
    dnf_by_team = df.groupby("team").agg(
        races=("dnf", "count"), dnfs=("dnf", "sum"),
    )
    dnf_by_team["dnf_rate"] = (dnf_by_team["dnfs"] / dnf_by_team["races"] * 100).round(1)
    for team, row in dnf_by_team.sort_values("dnf_rate", ascending=False).iterrows():
        print(f"  {team:<20s}: {row['dnf_rate']:5.1f}%")


def best_overtakers():
    """Chi guadagna più posizioni in gara?"""
    df = load_all_data()
    df["positions_gained"] = df["grid_position"] - df["finish_position"]

    print("\n🚀 MIGLIORI RIMONTATORI")
    print("=" * 60)

    overtakers = df.groupby("driver")["positions_gained"].agg(["mean", "count"]).round(2)
    overtakers.columns = ["avg_gained", "races"]
    overtakers = overtakers[overtakers["races"] >= 20].sort_values("avg_gained", ascending=False)

    for driver, row in overtakers.iterrows():
        if row["avg_gained"] > 0:
            bar = "🟢" * int(row["avg_gained"])
            print(f"  {driver:4s}: +{row['avg_gained']:.2f} pos/gara  {bar}")
        else:
            bar = "🔴" * int(abs(row["avg_gained"]))
            print(f"  {driver:4s}: {row['avg_gained']:.2f} pos/gara  {bar}")


def team_performance_over_time():
    """Performance media dei team per stagione."""
    df = load_all_data()

    print("\n🏎️ PERFORMANCE MEDIA TEAM PER STAGIONE")
    print("=" * 60)

    pivot = df.pivot_table(
        values="finish_position", index="team", columns="year", aggfunc="mean"
    ).round(1)
    print(pivot.to_string())

    if len(pivot.columns) >= 2:
        first_year = pivot.columns[0]
        last_year = pivot.columns[-1]
        pivot["change"] = pivot[first_year] - pivot[last_year]

        print(f"\n📈 Miglioramento {first_year} → {last_year}:")
        for team, row in pivot.sort_values("change", ascending=False).iterrows():
            if pd.notna(row["change"]):
                arrow = "↑" if row["change"] > 0 else "↓"
                print(f"  {arrow} {team}: {row['change']:+.1f} posizioni")


# ============================================================
# 2. ELO RATINGS
# ============================================================

def show_elo_rankings():
    """Classifica Elo attuale."""
    df = load_all_data()
    elo = EloSystem()
    elo.process_season(df)

    print("=" * 60)
    print("🏆 CLASSIFICA ELO PILOTI")
    print("=" * 60)
    rankings = elo.get_current_ratings()
    for i, row in rankings.iterrows():
        bar = "█" * int((row["elo_rating"] - 1300) / 10)
        print(f"  {i+1:2d}. {row['driver']:4s} {row['elo_rating']:7.1f}  {bar}")
    return rankings


def show_driver_elo_history(driver_code: str):
    """Storico Elo di un pilota. Uso: show_driver_elo_history("VER")"""
    df = load_all_data()
    elo = EloSystem()
    history = elo.process_season(df)

    driver_history = history[history["driver"] == driver_code].copy()
    if driver_history.empty:
        print(f"❌ '{driver_code}' non trovato! Disponibili: {sorted(history['driver'].unique())}")
        return

    print(f"\n📈 Storico Elo di {driver_code}:")
    print(f"{'Gara':<35s} {'Elo Pre':>8s} {'Elo Post':>9s} {'Pos':>4s} {'Δ':>7s}")
    print("-" * 65)

    for _, row in driver_history.iterrows():
        delta = row["elo_post_race"] - row["elo_pre_race"]
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        pos = int(row["finish_position"]) if pd.notna(row["finish_position"]) else "DNF"
        print(
            f"  {row['year']:.0f} R{row['round']:02.0f} {row['race_name']:<25s}"
            f" {row['elo_pre_race']:7.1f} → {row['elo_post_race']:7.1f}"
            f"  P{pos:<3}  {arrow} {delta:+.1f}"
        )

    print(f"\n  Attuale: {driver_history['elo_post_race'].iloc[-1]:.1f}")
    print(f"  Massimo: {driver_history['elo_post_race'].max():.1f}")
    print(f"  Minimo:  {driver_history['elo_post_race'].min():.1f}")
    return driver_history


def compare_drivers(*driver_codes):
    """Confronta Elo di più piloti. Uso: compare_drivers("VER", "HAM", "NOR")"""
    df = load_all_data()
    elo = EloSystem()
    history = elo.process_season(df)

    print(f"\n⚔️ Confronto Elo: {' vs '.join(driver_codes)}")
    print("-" * 50)

    for code in driver_codes:
        data = history[history["driver"] == code]
        if data.empty:
            print(f"  {code}: non trovato")
            continue
        current = data["elo_post_race"].iloc[-1]
        peak = data["elo_post_race"].max()
        low = data["elo_post_race"].min()
        print(f"  {code}: attuale={current:.0f}  picco={peak:.0f}  minimo={low:.0f}")


def head_to_head(driver_a: str, driver_b: str):
    """Testa a testa. Uso: head_to_head("VER", "NOR")"""
    df = load_all_data()
    wins_a, wins_b, total = 0, 0, 0
    results = []

    for (year, rnd), race_df in df.groupby(["year", "round"]):
        a = race_df[race_df["driver"] == driver_a]
        b = race_df[race_df["driver"] == driver_b]
        if a.empty or b.empty:
            continue
        pos_a, pos_b = a["finish_position"].iloc[0], b["finish_position"].iloc[0]
        if pd.isna(pos_a) or pd.isna(pos_b):
            continue
        total += 1
        winner = driver_a if pos_a < pos_b else driver_b
        if pos_a < pos_b:
            wins_a += 1
        else:
            wins_b += 1
        results.append({
            "race": race_df["race_name"].iloc[0], "year": year,
            f"{driver_a}": int(pos_a), f"{driver_b}": int(pos_b), "winner": winner,
        })

    print(f"\n🥊 {driver_a} vs {driver_b}")
    print("=" * 50)
    print(f"  Gare in comune: {total}")
    print(f"  {driver_a} davanti: {wins_a} ({wins_a/total*100:.0f}%)")
    print(f"  {driver_b} davanti: {wins_b} ({wins_b/total*100:.0f}%)")
    print(f"\n  Ultime 10:")
    for r in results[-10:]:
        marker = "←" if r["winner"] == driver_a else "→"
        print(f"    {r['year']} {r['race']:<25s} "
              f"{driver_a} P{r[driver_a]} {marker} P{r[driver_b]} {driver_b}")


# ============================================================
# 3. PREDICTIONS
# ============================================================

def predict_specific_race(year: int, round_num: int):
    """Previsione per una gara specifica. Uso: predict_specific_race(2023, 1)"""
    features = _load_features()

    train_data = features[
        (features["year"] < year)
        | ((features["year"] == year) & (features["round"] < round_num))
    ]
    if len(train_data) < 100:
        print("⚠️ Non abbastanza dati di training")
        return

    race_data = features[
        (features["year"] == year) & (features["round"] == round_num)
    ]
    is_pre_quali = False
    if race_data.empty:
        try:
            from src.pre_weekend_prediction import (
                build_future_round_features,
                predict_future_round_race,
            )
            df_all = load_all_data()
            race_data, has_fp = build_future_round_features(
                year, round_num, features, df_all
            )
            if race_data is None:
                print(f"❌ Gara non trovata: {year} Round {round_num}")
                return
            is_pre_quali = True
            predictor = _train_silent(train_data)
            prediction = predict_future_round_race(
                race_data, predictor, train_data
            )
            print(f"  ℹ️  Previsione PRE-QUALI/QU (round non in dataset, uso storico"
                  f"{' + FP' if has_fp else ''})")
        except Exception as e:
            print(f"❌ Gara non trovata: {year} Round {round_num} ({e})")
            return
    else:
        predictor = _train_silent(features)
        prediction = predictor.predict_race(race_data)

    race_name = prediction["race_name"].iloc[0]
    print(f"\n🏁 Previsione: {race_name} {year}")
    print("=" * 70)
    print(f"{'Rank':>4s}  {'Pilota':<5s} {'Team':<20s} "
          f"{'Griglia':>7s}  {'Previsto':>8s}  {'Reale':>5s}  {'Errore':>6s}")
    print("-" * 70)

    for _, row in prediction.iterrows():
        actual = int(row["finish_position"]) if pd.notna(row["finish_position"]) else "-"
        error = abs(row["predicted_position"] - row["finish_position"]) if pd.notna(row["finish_position"]) else "-"
        if isinstance(error, float):
            error = f"{error:.1f}"
        print(
            f"  {row['predicted_rank']:2.0f}   {row['driver']:<5s} {row['team']:<20s}"
            f"  P{row['grid_position']:<5.0f}  {row['predicted_position']:7.1f}"
            f"  P{actual:<4}  {error:>5s}"
        )
    return prediction


def simulate_season(year: int):
    """Simula un'intera stagione gara per gara. Uso: simulate_season(2023)"""
    features = _load_features()
    season_data = features[features["year"] == year]
    rounds = sorted(season_data["round"].unique())

    print(f"\n🏆 SIMULAZIONE STAGIONE {year}")
    print("=" * 60)

    race_scores = []

    for rnd in rounds:
        train_data = features[
            (features["year"] < year)
            | ((features["year"] == year) & (features["round"] < rnd))
        ]
        if len(train_data) < 50:
            continue

        predictor = _train_silent(train_data)
        race_data = season_data[season_data["round"] == rnd]
        prediction = predictor.predict_race(race_data)

        mae = abs(prediction["predicted_position"] - prediction["finish_position"]).mean()
        pred_winner = prediction.iloc[0]["driver"]
        actual_winner = prediction.loc[prediction["finish_position"].idxmin(), "driver"]
        correct = "✅" if pred_winner == actual_winner else "❌"
        race_name = prediction["race_name"].iloc[0]

        print(f"  R{rnd:2.0f}: {race_name:<30s}  MAE={mae:.1f}  "
              f"Vincitore: {correct} (prev: {pred_winner}, reale: {actual_winner})")
        race_scores.append({"round": rnd, "race": race_name, "mae": mae,
                           "correct_winner": pred_winner == actual_winner})

    scores_df = pd.DataFrame(race_scores)
    print(f"\n📊 Riepilogo {year}:")
    print(f"  MAE medio: {scores_df['mae'].mean():.2f} posizioni")
    print(f"  Vincitore corretto: {scores_df['correct_winner'].sum()}"
          f"/{len(scores_df)} ({scores_df['correct_winner'].mean()*100:.0f}%)")
    print(f"  Meglio prevista: {scores_df.loc[scores_df['mae'].idxmin(), 'race']}")
    print(f"  Peggio prevista: {scores_df.loc[scores_df['mae'].idxmax(), 'race']}")
    return scores_df


# ============================================================
# 3b. PREVISIONE CON DATI FP (PRE-GARA)
# ============================================================

def predict_with_fp(year: int, round_num: int):
    """
    Previsione per una gara usando i dati delle prove libere.
    Questa è la previsione più accurata perché usa i dati reali del weekend.

    Uso: predict_with_fp(2024, 5)

    Mostra:
    - Classifica prevista basata su FP + storico
    - Indicatori di forma del weekend (FP delta, long run)
    - Confronto con il modello senza FP
    """
    features = _load_features()

    # Verifica che ci siano dati FP
    has_fp = "fp_best_lap_delta" in features.columns
    if not has_fp:
        print("⚠️ Dati FP non presenti nella feature matrix.")
        print("   Esegui: python -m src.fp_data_loader")
        print("   Poi:    python -m src.advanced_features")
        print("   Uso il modello standard...\n")
        return predict_specific_race(year, round_num)

    race_data = features[
        (features["year"] == year) & (features["round"] == round_num)
    ]
    from_future_round = False
    if race_data.empty:
        try:
            from src.pre_weekend_prediction import (
                build_future_round_features,
                predict_future_round_race,
            )
            df_all = load_all_data()
            train_data = features[
                (features["year"] < year)
                | ((features["year"] == year) & (features["round"] < round_num))
            ]
            race_data, has_fp_data = build_future_round_features(
                year, round_num, features, df_all
            )
            if race_data is None:
                print(f"❌ Gara non trovata: {year} Round {round_num}")
                return
            predictor_fp = _train_silent(train_data, use_advanced=True)
            predictor_base = _train_silent(train_data, use_advanced=False)
            prediction_fp = predict_future_round_race(
                race_data, predictor_fp, train_data
            )
            prediction_base = predict_future_round_race(
                race_data, predictor_base, train_data
            )
            from_future_round = True
            print(f"  ℹ️  Previsione PRE-QUALI (round non in dataset, uso storico"
                  f"{' + FP' if has_fp_data else ''})")
        except Exception as e:
            print(f"❌ Gara non trovata: {year} Round {round_num} ({e})")
            return
    else:
        has_fp_data = race_data["fp_best_lap_delta"].notna().any() if "fp_best_lap_delta" in race_data.columns else False

    race_name = race_data["race_name"].iloc[0]

    if not from_future_round:
        predictor_fp = _train_silent(features, use_advanced=True)
        prediction_fp = predictor_fp.predict_race(race_data)
        predictor_base = _train_silent(features, use_advanced=False)
        prediction_base = predictor_base.predict_race(race_data)

    print(f"\n🏁 PREVISIONE CON DATI FP: {race_name} {year}")
    print("=" * 85)
    print(f"{'Rank':>4s}  {'Pilota':<5s} {'Team':<20s} "
          f"{'Griglia':>7s}  {'Prev FP':>7s}  {'Prev Base':>9s}  "
          f"{'Reale':>5s}  {'FP Delta':>8s}")
    print("-" * 85)

    # Merge le due previsioni
    base_map = dict(zip(prediction_base["driver"], prediction_base["predicted_position"]))

    for _, row in prediction_fp.iterrows():
        actual = int(row["finish_position"]) if pd.notna(row["finish_position"]) else "DNF"
        base_pred = base_map.get(row["driver"], np.nan)
        fp_delta = row.get("fp_best_lap_delta", np.nan)

        fp_delta_str = f"{fp_delta:+.3f}s" if pd.notna(fp_delta) else "  n/a"

        print(
            f"  {row['predicted_rank']:2.0f}   {row['driver']:<5s} {row['team']:<20s}"
            f"  P{row['grid_position']:<5.0f}  {row['predicted_position']:6.1f}"
            f"  {base_pred:8.1f}"
            f"  P{actual:<4}  {fp_delta_str:>8s}"
        )

    # Riepilogo FP
    print(f"\n📊 INDICATORI WEEKEND:")
    fp_cols = ["fp_best_lap_delta", "fp2_long_run_deg", "fp_long_run_consistency"]
    for col in fp_cols:
        if col in race_data.columns:
            valid = race_data[[col, "driver"]].dropna(subset=[col])
            if not valid.empty:
                best_driver = valid.loc[valid[col].idxmin(), "driver"]
                best_val = valid[col].min()
                label = {
                    "fp_best_lap_delta": "Più veloce in FP",
                    "fp2_long_run_deg": "Minor degradazione FP2",
                    "fp_long_run_consistency": "Più costante long run",
                }.get(col, col)
                print(f"  {label}: {best_driver} ({best_val:.3f})")

    # Calcolo errore se abbiamo i risultati reali
    if race_data["finish_position"].notna().any():
        mae_fp = abs(prediction_fp["predicted_position"] - prediction_fp["finish_position"]).mean()
        mae_base_vals = []
        for _, row in prediction_fp.iterrows():
            bp = base_map.get(row["driver"], np.nan)
            if pd.notna(bp) and pd.notna(row["finish_position"]):
                mae_base_vals.append(abs(bp - row["finish_position"]))
        mae_base = np.mean(mae_base_vals) if mae_base_vals else np.nan

        print(f"\n📈 ACCURACY:")
        print(f"  MAE con FP:    {mae_fp:.2f} posizioni")
        if pd.notna(mae_base):
            print(f"  MAE senza FP:  {mae_base:.2f} posizioni")
            improvement = (mae_base - mae_fp) / mae_base * 100
            if improvement > 0:
                print(f"  Miglioramento: {improvement:.1f}%")
            else:
                print(f"  Differenza:    {improvement:.1f}%")

    return prediction_fp


def predict_next_race():
    """
    Previsione per la PROSSIMA gara (l'ultima nel dataset).
    Usa automaticamente i dati FP se disponibili.
    """
    features = _load_features()
    last_year = features["year"].max()
    last_round = features[features["year"] == last_year]["round"].max()

    print(f"🔮 Previsione per l'ultima gara disponibile: {last_year} R{int(last_round)}")

    has_fp = "fp_best_lap_delta" in features.columns
    if has_fp:
        return predict_with_fp(last_year, int(last_round))
    else:
        return predict_specific_race(last_year, int(last_round))


def fp_weekend_analysis(year: int, round_num: int):
    """
    Analisi dettagliata delle prove libere di un weekend specifico.
    Mostra: best laps, long run pace, degradazione, confronto sessioni.

    Uso: fp_weekend_analysis(2024, 5)
    """
    fp_file = PROCESSED_DATA_DIR / "fp_data.csv"
    if not fp_file.exists():
        print("❌ Dati FP non disponibili. Esegui: python -m src.fp_data_loader")
        return

    fp_data = pd.read_csv(fp_file)
    weekend = fp_data[(fp_data["year"] == year) & (fp_data["round"] == round_num)]

    if weekend.empty:
        print(f"❌ Nessun dato FP per {year} R{round_num}")
        return

    print(f"\n🏎️ ANALISI PROVE LIBERE: {year} Round {round_num}")
    print("=" * 70)

    for session in ["FP1", "FP2", "FP3"]:
        sess_data = weekend[weekend["session"] == session]
        if sess_data.empty:
            continue

        print(f"\n  --- {session} ---")

        # Best laps (top 10)
        if "best_lap" in sess_data.columns:
            top = sess_data.nsmallest(10, "best_lap")
            leader_time = top["best_lap"].iloc[0]
            print(f"  {'Pos':>3s}  {'Pilota':<5s}  {'Best Lap':>9s}  {'Gap':>7s}")
            for i, (_, row) in enumerate(top.iterrows()):
                gap = row["best_lap"] - leader_time
                gap_str = f"+{gap:.3f}" if gap > 0 else "LEADER"
                mins = int(row["best_lap"] // 60)
                secs = row["best_lap"] % 60
                print(f"  {i+1:3d}  {row['driver']:<5s}  {mins}:{secs:06.3f}  {gap_str:>7s}")

        # Long run (se disponibili)
        lr_data = sess_data[sess_data["long_run_pace"].notna()]
        if not lr_data.empty:
            print(f"\n  Long Run ({session}):")
            print(f"  {'Pilota':<5s}  {'Pace':>8s}  {'Deg':>7s}  {'Consist':>8s}  {'Giri':>5s}")
            for _, row in lr_data.nsmallest(10, "long_run_pace").iterrows():
                deg_str = f"{row['long_run_deg']:.3f}" if pd.notna(row.get('long_run_deg')) else "n/a"
                cons_str = f"{row['long_run_consistency']:.3f}" if pd.notna(row.get('long_run_consistency')) else "n/a"
                laps_str = f"{int(row['long_run_laps'])}" if pd.notna(row.get('long_run_laps')) else "n/a"
                mins = int(row["long_run_pace"] // 60)
                secs = row["long_run_pace"] % 60
                print(f"  {row['driver']:<5s}  {mins}:{secs:06.3f}  {deg_str:>7s}  {cons_str:>8s}  {laps_str:>5s}")

    return weekend


# ============================================================
# 3c. PREVISIONI WEEKEND (QUALIFICA + GARA + CONFIDENCE)
# ============================================================

def predict_weekend(year: int, round_num: int):
    """
    Previsione completa del weekend: qualifica E gara.

    Produce due tabelle:
    1. QUALIFICA PREVISTA — chi sarà in pole, top 3, ecc.
    2. GARA PREVISTA — classifica prevista con confidence

    La confidence viene calcolata così:
    - Il modello Random Forest è un ensemble di 200 alberi
    - Ogni albero produce una previsione diversa
    - La varianza tra le previsioni = incertezza
    - Bassa varianza = alta confidence (il modello è sicuro)
    - Alta varianza = bassa confidence (molti scenari possibili)

    Uso: predict_weekend(2024, 5)
    """
    features = _load_features()

    race_data = features[
        (features["year"] == year) & (features["round"] == round_num)
    ]
    from_future_round = False
    if race_data.empty:
        try:
            from src.pre_weekend_prediction import build_future_round_features
            df_all = load_all_data()
            race_data, has_fp_data = build_future_round_features(
                year, round_num, features, df_all
            )
            if race_data is None:
                print(f"❌ Gara non trovata: {year} Round {round_num}")
                return
            from_future_round = True
            print(f"  ℹ️  Previsione PRE-QUALI (round non in dataset, uso storico"
                  f"{' + FP' if has_fp_data else ''})")
        except Exception as e:
            print(f"❌ Gara non trovata: {year} Round {round_num} ({e})")
            return

    race_name = race_data["race_name"].iloc[0]

    # Dati di training: tutto prima di questa gara
    train_data = features[
        (features["year"] < year)
        | ((features["year"] == year) & (features["round"] < round_num))
    ]
    if len(train_data) < 100:
        print("⚠️ Non abbastanza dati di training, uso tutto il dataset")
        train_data = features

    # =======================================================
    # CALCOLO PESO FP — dinamico in base all'era regolamentare
    # =======================================================
    # A inizio era (R1 2026) le FP pesano 70% perché sono l'unica info
    # sulla forza delle macchine. Più gare ci sono, più lo storico pesa.
    # Le FP pesano SEMPRE almeno 25% perché danno info sul weekend specifico.
    from config import REGULATION_RESET_YEARS, get_regulation_era_start

    era_start = get_regulation_era_start(year)
    races_in_era = train_data[train_data["year"] >= era_start]
    races_completed = races_in_era.groupby(["year", "round"]).ngroups
    w_fp = max(0.25, 0.70 - races_completed * 0.045)

    # Controlliamo se ci sono dati FP per questa gara
    fp_cols = [c for c in race_data.columns if c.startswith("fp_") or c == "fp2_delta" or c == "fp3_delta"]
    fp_values = race_data[fp_cols].iloc[0] if fp_cols else pd.Series()
    # Se tutte le feature FP sono la stessa (mediana) → non ci sono dati FP reali
    has_real_fp = False
    if len(fp_values) > 0:
        unique_vals = fp_values.dropna().nunique()
        has_real_fp = unique_vals > 1

    if not has_real_fp:
        w_fp = 0.0

    if year in REGULATION_RESET_YEARS or year >= era_start:
        if has_real_fp:
            print(f"  ⚖️  Peso FP: {w_fp*100:.0f}% | Peso storico: {(1-w_fp)*100:.0f}%")
            print(f"     ({races_completed} gare nell'era {era_start}+ → {'FP dominano' if w_fp > 0.5 else 'bilanciato' if w_fp > 0.3 else 'storico domina'})")
        else:
            print(f"  ⚠️ Nuova era {year}: nessun dato FP reale — previsioni solo su Elo pilota e storico.")

    # =======================================================
    # MODELLO COMPLETO (predice finish_position con tutte le feature)
    # =======================================================
    predictor = F1Predictor(use_advanced=True)

    # Training silenzioso
    old = sys.stdout
    sys.stdout = io.StringIO()
    results = predictor.train(train_data)
    sys.stdout = old

    # Previsione gara con modello completo
    # NOTA: non usiamo prepare_data() perché droppa righe con finish_position=NaN.
    clean_race = race_data.copy()
    X_race = clean_race[predictor.feature_columns].copy()
    for col in X_race.columns:
        if X_race[col].isna().any():
            train_median = train_data[col].median() if col in train_data.columns else 0
            X_race[col] = X_race[col].fillna(train_median)
    pred_completo = predictor.model.predict(X_race)

    # =======================================================
    # MODELLO FP+PILOTA (solo feature FP + elo pilota + griglia)
    # =======================================================
    # Questo modello cattura la "forza del weekend" senza storico team.
    # Essenziale a inizio era regolamentare.
    pred_fp_model = None

    if has_real_fp and w_fp > 0:
        from sklearn.ensemble import RandomForestRegressor
        from config import RANDOM_SEED

        # Feature per il modello FP: solo cose legate alle FP + pilota
        fp_model_features = [
            c for c in predictor.feature_columns
            if c.startswith("fp_") or c.startswith("fp2_") or c.startswith("fp3_")
            or c in ["grid_position", "elo_pre_race", "is_front_row", "is_top5", "is_top10"]
        ]
        # Verifica che almeno qualche feature FP esista nei dati
        fp_model_features = [c for c in fp_model_features if c in train_data.columns]

        if len(fp_model_features) >= 3:
            # Training sul dataset storico (dove FP sono disponibili)
            fp_train = train_data.dropna(subset=["finish_position"]).copy()
            X_fp_train = fp_train[fp_model_features].copy()
            for col in X_fp_train.columns:
                if X_fp_train[col].isna().any():
                    X_fp_train[col] = X_fp_train[col].fillna(X_fp_train[col].median())
            y_fp_train = fp_train["finish_position"]

            fp_rf = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
            fp_rf.fit(X_fp_train, y_fp_train)

            # Previsione con modello FP
            X_fp_race = clean_race[fp_model_features].copy()
            for col in X_fp_race.columns:
                if X_fp_race[col].isna().any():
                    fallback = fp_train[col].median() if col in fp_train.columns else 0
                    X_fp_race[col] = X_fp_race[col].fillna(fallback)

            pred_fp_model = fp_rf.predict(X_fp_race)

    # =======================================================
    # COMBINAZIONE: ensemble pesato
    # =======================================================
    if pred_fp_model is not None and w_fp > 0:
        race_pred = w_fp * pred_fp_model + (1 - w_fp) * pred_completo
    else:
        race_pred = pred_completo

    # === CONFIDENCE: varianza tra i singoli alberi ===
    if hasattr(predictor.model, "estimators_"):
        tree_predictions = np.array([
            tree.predict(X_race) for tree in predictor.model.estimators_
        ])
        pred_std = tree_predictions.std(axis=0)
        max_std = pred_std.max() if pred_std.max() > 0 else 1.0
        confidence_race = ((1 - pred_std / max_std) * 60 + 40).clip(40, 99)
    else:
        confidence_race = np.full(len(X_race), 70.0)

    # =======================================================
    # MODELLO QUALIFICA (predice grid_position)
    # =======================================================
    # Usa le stesse feature ma con target diverso
    # Rimuoviamo grid_position e derivate dalle feature (sarebbero leakage!)
    quali_feature_cols = [
        c for c in predictor.feature_columns
        if c not in ["grid_position", "is_front_row", "is_top5", "is_top10"]
    ]

    if len(quali_feature_cols) >= 5:
        from sklearn.ensemble import RandomForestRegressor
        from config import RANDOM_SEED

        # Prepara dati qualifica
        quali_train = train_data.dropna(subset=["grid_position"])
        X_quali_train = quali_train[quali_feature_cols].copy()
        for col in X_quali_train.columns:
            if X_quali_train[col].isna().any():
                X_quali_train[col] = X_quali_train[col].fillna(X_quali_train[col].median())
        y_quali_train = quali_train["grid_position"]

        quali_model = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=RANDOM_SEED, n_jobs=-1,
        )
        quali_model.fit(X_quali_train, y_quali_train)

        X_quali_race = clean_race[quali_feature_cols].copy()
        for col in X_quali_race.columns:
            if X_quali_race[col].isna().any():
                fallback = X_quali_train[col].median() if col in X_quali_train.columns else 0
                X_quali_race[col] = X_quali_race[col].fillna(fallback)

        pred_quali_completo = quali_model.predict(X_quali_race)

        # === MODELLO QUALIFICA FP (best laps FP3 + FP + elo) ===
        pred_quali_fp = None
        if has_real_fp and w_fp > 0:
            # Per la qualifica, le feature FP più rilevanti sono i best laps
            # (FP3 in particolare = qualifying simulation)
            fp_quali_features = [
                c for c in quali_feature_cols
                if c.startswith("fp_") or c.startswith("fp2_") or c.startswith("fp3_")
                or c in ["elo_pre_race"]
            ]
            fp_quali_features = [c for c in fp_quali_features if c in train_data.columns]

            if len(fp_quali_features) >= 3:
                X_fq_train = quali_train[fp_quali_features].copy()
                for col in X_fq_train.columns:
                    if X_fq_train[col].isna().any():
                        X_fq_train[col] = X_fq_train[col].fillna(X_fq_train[col].median())

                fp_quali_rf = RandomForestRegressor(
                    n_estimators=200, max_depth=10, min_samples_leaf=5,
                    random_state=RANDOM_SEED, n_jobs=-1,
                )
                fp_quali_rf.fit(X_fq_train, y_quali_train)

                X_fq_race = clean_race[fp_quali_features].copy()
                for col in X_fq_race.columns:
                    if X_fq_race[col].isna().any():
                        fallback = X_fq_train[col].median() if col in X_fq_train.columns else 0
                        X_fq_race[col] = X_fq_race[col].fillna(fallback)

                pred_quali_fp = fp_quali_rf.predict(X_fq_race)

        # Combinazione ensemble qualifica
        if pred_quali_fp is not None and w_fp > 0:
            quali_pred = w_fp * pred_quali_fp + (1 - w_fp) * pred_quali_completo
        else:
            quali_pred = pred_quali_completo

        # Confidence qualifica
        if hasattr(quali_model, "estimators_"):
            tree_preds_q = np.array([
                tree.predict(X_quali_race) for tree in quali_model.estimators_
            ])
            pred_std_q = tree_preds_q.std(axis=0)
            max_std_q = pred_std_q.max() if pred_std_q.max() > 0 else 1.0
            confidence_quali = ((1 - pred_std_q / max_std_q) * 60 + 40).clip(40, 99)
        else:
            confidence_quali = np.full(len(X_quali_race), 70.0)

        has_quali_model = True
    else:
        has_quali_model = False

    # =======================================================
    # ACCURATEZZA STORICA (basata su gare simili passate)
    # =======================================================
    # Simula le ultime N gare per capire quanto è affidabile il modello
    recent_errors = []
    test_rounds = train_data.groupby(["year", "round"]).ngroups
    eval_races = train_data.groupby(["year", "round"]).tail(1).tail(min(10, test_rounds // 4))

    for (y, r), _ in eval_races.groupby(["year", "round"]):
        eval_train = features[
            (features["year"] < y)
            | ((features["year"] == y) & (features["round"] < r))
        ]
        eval_race = features[(features["year"] == y) & (features["round"] == r)]

        if len(eval_train) < 50 or eval_race.empty:
            continue

        try:
            eval_predictor = F1Predictor(use_advanced=True)
            old2 = sys.stdout
            sys.stdout = io.StringIO()
            eval_predictor.train(eval_train)
            sys.stdout = old2

            eval_pred = eval_predictor.predict(eval_race)
            mae = abs(eval_pred["predicted_position"] - eval_pred["finish_position"]).mean()
            recent_errors.append(mae)

            # Winner accuracy
            pred_winner = eval_pred.loc[eval_pred["predicted_position"].idxmin(), "driver"]
            real_winner = eval_pred.loc[eval_pred["finish_position"].idxmin(), "driver"]
            recent_errors.append(mae)
        except Exception:
            continue

    # =======================================================
    # OUTPUT
    # =======================================================

    print(f"\n{'=' * 80}")
    print(f"🏁 PREVISIONI WEEKEND: {race_name} {year}")
    print(f"{'=' * 80}")

    # --- TABELLA QUALIFICA ---
    if has_quali_model:
        print(f"\n📋 QUALIFICA PREVISTA")
        print(f"{'─' * 65}")
        print(f"  {'Pos':>3s}  {'Pilota':<5s}  {'Team':<22s}  {'Prevista':>8s}  {'Reale':>5s}  {'Conf':>5s}")
        print(f"{'─' * 65}")

        quali_df = clean_race.copy()
        quali_df["quali_predicted"] = quali_pred
        quali_df["quali_confidence"] = confidence_quali
        quali_df = quali_df.sort_values("quali_predicted").reset_index(drop=True)

        for i, (_, row) in enumerate(quali_df.iterrows()):
            pos = i + 1
            real_grid = int(row["grid_position"]) if pd.notna(row.get("grid_position")) else "?"
            conf = row["quali_confidence"]

            # Barra confidence
            conf_bar = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))

            print(
                f"  {pos:3d}  {row['driver']:<5s}  {row['team']:<22s}"
                f"  P{row['quali_predicted']:5.1f}  P{real_grid:<4}"
                f"  {conf_bar} {conf:.0f}%"
            )

        # Accuracy qualifica
        if quali_df["grid_position"].notna().any():
            mae_q = abs(quali_df["quali_predicted"] - quali_df["grid_position"]).mean()
            top3_real = set(quali_df.nsmallest(3, "grid_position")["driver"])
            top3_pred = set(quali_df.head(3)["driver"])
            top3_hit = len(top3_real & top3_pred)
            print(f"\n  📊 MAE qualifica: {mae_q:.2f} posizioni | Top 3 corretti: {top3_hit}/3")

    # --- TABELLA GARA ---
    print(f"\n📋 GARA PREVISTA")
    print(f"{'─' * 75}")
    print(f"  {'Pos':>3s}  {'Pilota':<5s}  {'Team':<22s}  {'Griglia':>7s}  {'Prevista':>8s}  {'Reale':>5s}  {'Conf':>5s}")
    print(f"{'─' * 75}")

    race_df = clean_race.copy()
    race_df["race_predicted"] = race_pred
    race_df["race_confidence"] = confidence_race
    race_df = race_df.sort_values("race_predicted").reset_index(drop=True)

    for i, (_, row) in enumerate(race_df.iterrows()):
        pos = i + 1
        grid = int(row["grid_position"]) if pd.notna(row.get("grid_position")) else "?"
        actual = int(row["finish_position"]) if pd.notna(row.get("finish_position")) else "?"
        conf = row["race_confidence"]

        conf_bar = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))

        print(
            f"  {pos:3d}  {row['driver']:<5s}  {row['team']:<22s}"
            f"  P{grid:<6}  P{row['race_predicted']:5.1f}  P{actual:<4}"
            f"  {conf_bar} {conf:.0f}%"
        )

    # --- ACCURATEZZA GARA ---
    if race_df["finish_position"].notna().any():
        mae_r = abs(race_df["race_predicted"] - race_df["finish_position"]).mean()
        top3_real_r = set(race_df.nsmallest(3, "finish_position")["driver"] if "finish_position" in race_df else [])
        top3_pred_r = set(race_df.head(3)["driver"])
        top3_hit_r = len(top3_real_r & top3_pred_r)

        winner_pred = race_df.iloc[0]["driver"]
        winner_real = race_df.loc[race_df["finish_position"].idxmin(), "driver"] if race_df["finish_position"].notna().any() else "?"
        winner_ok = "✅" if winner_pred == winner_real else "❌"

        print(f"\n  📊 MAE gara: {mae_r:.2f} posizioni | Top 3 corretti: {top3_hit_r}/3 | Vincitore: {winner_ok}")

    # --- ACCURATEZZA STIMATA DEL MODELLO ---
    print(f"\n{'─' * 75}")
    print(f"📈 AFFIDABILITÀ STIMATA DEL MODELLO")
    print(f"{'─' * 75}")

    cv_mae = results.get("cv_mae_mean", 0)
    cv_std = results.get("cv_mae_std", 0)

    # Traduci MAE in "accuratezza" interpretabile
    # MAE = 3.0 su 20 posizioni → circa 85% accurato
    accuracy_pct = max(0, min(100, (1 - cv_mae / 10) * 100))

    print(f"  Errore medio (CV):     {cv_mae:.2f} ± {cv_std:.2f} posizioni")
    print(f"  Accuratezza stimata:   {accuracy_pct:.0f}%")
    print(f"  Feature utilizzate:    {results.get('num_features', '?')}")

    if recent_errors:
        avg_recent = np.mean(recent_errors)
        print(f"  MAE ultime gare:       {avg_recent:.2f} posizioni")

    # Interpretazione
    if accuracy_pct >= 80:
        print(f"\n  🟢 Modello AFFIDABILE — errore contenuto, previsioni solide")
    elif accuracy_pct >= 65:
        print(f"\n  🟡 Modello BUONO — margine d'errore moderato, da verificare")
    else:
        print(f"\n  🔴 Modello INCERTO — alta variabilità, prendere con cautela")

    # Nota FP
    if has_real_fp and w_fp > 0:
        print(f"  ℹ️  Dati FP reali disponibili — peso FP: {w_fp*100:.0f}%")
        print(f"     Ensemble: modello FP+Pilota ({w_fp*100:.0f}%) + modello completo ({(1-w_fp)*100:.0f}%)")
    elif has_real_fp:
        print(f"  ℹ️  Dati FP disponibili — usati come feature nel modello completo")
    else:
        print(f"  ℹ️  Dati FP non disponibili — basato solo su storico e Elo")
    has_fp = has_real_fp

    # =======================================================
    # GENERA PDF
    # =======================================================
    pdf_path = _generate_weekend_pdf(
        race_name, year,
        quali_df if has_quali_model else None,
        race_df,
        cv_mae, cv_std, accuracy_pct,
        recent_errors,
        has_fp,
    )
    print(f"\n  📄 PDF salvato: {pdf_path}")

    return race_df


# ──────────────────────────────────────────────────────────────
# PDF HELPER: genera il documento con le tabelle di previsione
# ──────────────────────────────────────────────────────────────

def _generate_weekend_pdf(
    race_name: str,
    year: int,
    quali_df,          # può essere None
    race_df,
    cv_mae: float,
    cv_std: float,
    accuracy_pct: float,
    recent_errors: list,
    has_fp: bool,
):
    """
    Genera un PDF con le tabelle previsioni qualifica + gara.
    Stile dark-theme coerente con i grafici FP.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # ── colori tema ──
    DARK_BG   = "#1a1a2e"
    PANEL_BG  = "#16213e"
    TEXT_CLR  = "#e0e0e0"
    ACCENT    = "#00d4ff"
    GOLD      = "#FFD700"
    SILVER    = "#C0C0C0"
    BRONZE    = "#CD7F32"
    HEADER_BG = "#0f3460"
    ROW_ALT   = "#1c2a4a"
    ROW_NORM  = "#16213e"
    GREEN     = "#27ae60"
    ORANGE    = "#e67e22"
    RED       = "#e74c3c"

    # Colori team F1
    TEAM_COLORS = {
        "Red Bull Racing": "#3671C6", "Ferrari": "#E8002D",
        "McLaren": "#FF8000", "Mercedes": "#27F4D2",
        "Aston Martin": "#229971", "Alpine": "#FF87BC",
        "Williams": "#64C4FF", "RB": "#6692FF",
        "Kick Sauber": "#52E252", "Haas F1 Team": "#B6BABD",
        "Alfa Romeo": "#C92D4B", "AlphaTauri": "#5E8FAA",
    }

    def _team_color(team_name):
        for key, color in TEAM_COLORS.items():
            if key.lower() in str(team_name).lower():
                return color
        return "#888888"

    def _conf_color(conf):
        if conf >= 80:
            return GREEN
        elif conf >= 60:
            return ORANGE
        return RED

    def _draw_table(ax, df, title, is_quali=False):
        """Disegna una tabella di previsioni su un Axes."""
        ax.set_facecolor(DARK_BG)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(df) + 3.5)
        ax.axis("off")

        # Titolo
        ax.text(5, len(df) + 2.8, title, fontsize=16, fontweight="bold",
                color=ACCENT, ha="center", va="center",
                fontfamily="monospace")

        # Header
        if is_quali:
            headers = ["POS", "PILOTA", "TEAM", "PREVISTA", "REALE", "CONFIDENCE"]
            x_pos = [0.3, 1.2, 3.0, 5.5, 6.7, 8.0]
        else:
            headers = ["POS", "PILOTA", "TEAM", "GRIGLIA", "PREVISTA", "REALE", "CONFIDENCE"]
            x_pos = [0.3, 1.2, 3.0, 5.0, 6.0, 7.0, 8.2]

        header_y = len(df) + 1.5
        for i, (h, x) in enumerate(zip(headers, x_pos)):
            ax.text(x, header_y, h, fontsize=8, fontweight="bold",
                    color=GOLD, va="center", fontfamily="monospace")

        # Linea sotto header
        ax.plot([0.1, 9.9], [header_y - 0.4, header_y - 0.4],
                color=ACCENT, linewidth=0.8, alpha=0.5)

        # Righe
        for i, (_, row) in enumerate(df.iterrows()):
            pos = i + 1
            y = len(df) - i

            # Sfondo alternato
            bg_color = ROW_ALT if i % 2 == 0 else ROW_NORM
            ax.fill_between([0.1, 9.9], y - 0.4, y + 0.4,
                            color=bg_color, alpha=0.8)

            # Podio: evidenzia
            if pos <= 3:
                podium_colors = {1: GOLD, 2: SILVER, 3: BRONZE}
                pos_color = podium_colors[pos]
            else:
                pos_color = TEXT_CLR

            # Posizione
            ax.text(x_pos[0], y, f"P{pos}", fontsize=9, fontweight="bold",
                    color=pos_color, va="center", fontfamily="monospace")

            # Pilota
            ax.text(x_pos[1], y, str(row["driver"]), fontsize=9,
                    color=TEXT_CLR, va="center", fontfamily="monospace",
                    fontweight="bold")

            # Team (con colore scuderia)
            team_name = str(row["team"])
            team_short = team_name[:18]
            tc = _team_color(team_name)
            ax.text(x_pos[2], y, team_short, fontsize=7.5,
                    color=tc, va="center", fontfamily="monospace")

            if is_quali:
                # Prevista
                pred_val = row["quali_predicted"]
                ax.text(x_pos[3], y, f"P{pred_val:.1f}", fontsize=9,
                        color=TEXT_CLR, va="center", fontfamily="monospace")
                # Reale
                real_grid = row.get("grid_position", None)
                real_str = f"P{int(real_grid)}" if pd.notna(real_grid) else "?"
                ax.text(x_pos[4], y, real_str, fontsize=9,
                        color=TEXT_CLR, va="center", fontfamily="monospace")
                # Confidence
                conf = row["quali_confidence"]
                conf_x = x_pos[5]
            else:
                # Griglia
                grid = row.get("grid_position", None)
                grid_str = f"P{int(grid)}" if pd.notna(grid) else "?"
                ax.text(x_pos[3], y, grid_str, fontsize=9,
                        color=TEXT_CLR, va="center", fontfamily="monospace")
                # Prevista
                pred_val = row["race_predicted"]
                ax.text(x_pos[4], y, f"P{pred_val:.1f}", fontsize=9,
                        color=TEXT_CLR, va="center", fontfamily="monospace")
                # Reale
                actual = row.get("finish_position", None)
                act_str = f"P{int(actual)}" if pd.notna(actual) else "?"
                ax.text(x_pos[5], y, act_str, fontsize=9,
                        color=TEXT_CLR, va="center", fontfamily="monospace")
                # Confidence
                conf = row["race_confidence"]
                conf_x = x_pos[6]

            # Barra confidence visuale
            cc = _conf_color(conf)
            bar_width = conf / 100 * 1.5
            ax.barh(y, bar_width, left=conf_x, height=0.35,
                    color=cc, alpha=0.7, edgecolor="none")
            ax.text(conf_x + 1.6, y, f"{conf:.0f}%", fontsize=7,
                    color=cc, va="center", fontfamily="monospace",
                    fontweight="bold")

    # ── Costruzione PDF ──
    plots_dir = PROCESSED_DATA_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    safe_name = race_name.replace(" ", "_").replace("/", "-")
    pdf_path = plots_dir / f"previsioni_{safe_name}_{year}.pdf"

    n_drivers_q = len(quali_df) if quali_df is not None else 0
    n_drivers_r = len(race_df)

    with PdfPages(str(pdf_path)) as pdf:
        # ── PAGINA 1: QUALIFICA ──
        if quali_df is not None:
            fig_height = max(8, n_drivers_q * 0.42 + 3)
            fig_q, ax_q = plt.subplots(figsize=(12, fig_height))
            fig_q.patch.set_facecolor(DARK_BG)
            _draw_table(ax_q, quali_df, f"📋 QUALIFICA PREVISTA — {race_name} {year}",
                        is_quali=True)

            # Accuracy qualifica in fondo
            if quali_df["grid_position"].notna().any():
                mae_q = abs(quali_df["quali_predicted"] - quali_df["grid_position"]).mean()
                top3_real = set(quali_df.nsmallest(3, "grid_position")["driver"])
                top3_pred = set(quali_df.head(3)["driver"])
                top3_hit = len(top3_real & top3_pred)
                ax_q.text(5, 0.2,
                          f"MAE: {mae_q:.2f} posizioni  |  Top 3 corretti: {top3_hit}/3",
                          fontsize=10, color=ACCENT, ha="center", va="center",
                          fontfamily="monospace",
                          bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG,
                                    edgecolor=ACCENT, alpha=0.8))

            plt.tight_layout(pad=0.5)
            pdf.savefig(fig_q, facecolor=DARK_BG)
            plt.close(fig_q)

        # ── PAGINA 2: GARA ──
        fig_height = max(8, n_drivers_r * 0.42 + 3)
        fig_r, ax_r = plt.subplots(figsize=(12, fig_height))
        fig_r.patch.set_facecolor(DARK_BG)
        _draw_table(ax_r, race_df, f"📋 GARA PREVISTA — {race_name} {year}",
                    is_quali=False)

        # Accuracy gara in fondo
        if race_df["finish_position"].notna().any():
            mae_r = abs(race_df["race_predicted"] - race_df["finish_position"]).mean()
            top3_real_r = set(race_df.nsmallest(3, "finish_position")["driver"])
            top3_pred_r = set(race_df.head(3)["driver"])
            top3_hit_r = len(top3_real_r & top3_pred_r)
            winner_pred = race_df.iloc[0]["driver"]
            winner_real = race_df.loc[race_df["finish_position"].idxmin(), "driver"]
            winner_ok = "✅" if winner_pred == winner_real else "❌"
            ax_r.text(5, 0.2,
                      f"MAE: {mae_r:.2f} pos  |  Top 3: {top3_hit_r}/3  |  Vincitore: {winner_ok}",
                      fontsize=10, color=ACCENT, ha="center", va="center",
                      fontfamily="monospace",
                      bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG,
                                edgecolor=ACCENT, alpha=0.8))

        plt.tight_layout(pad=0.5)
        pdf.savefig(fig_r, facecolor=DARK_BG)
        plt.close(fig_r)

        # ── PAGINA 3: RIEPILOGO AFFIDABILITÀ ──
        fig_s, ax_s = plt.subplots(figsize=(12, 6))
        fig_s.patch.set_facecolor(DARK_BG)
        ax_s.set_facecolor(DARK_BG)
        ax_s.axis("off")
        ax_s.set_xlim(0, 10)
        ax_s.set_ylim(0, 10)

        ax_s.text(5, 9.2, f"📈 AFFIDABILITÀ MODELLO — {race_name} {year}",
                  fontsize=18, fontweight="bold", color=ACCENT, ha="center",
                  fontfamily="monospace")

        # Box con metriche
        metrics = [
            ("Errore medio (CV)", f"{cv_mae:.2f} ± {cv_std:.2f} pos"),
            ("Accuratezza stimata", f"{accuracy_pct:.0f}%"),
        ]
        if recent_errors:
            avg_recent = np.mean(recent_errors)
            metrics.append(("MAE ultime gare", f"{avg_recent:.2f} pos"))

        fp_label = "Dati FP inclusi ✅" if has_fp else "Solo storico + Elo"
        metrics.append(("Fonte dati", fp_label))

        for i, (label, value) in enumerate(metrics):
            y = 7.0 - i * 1.2
            ax_s.text(2.5, y, label, fontsize=13, color=TEXT_CLR,
                      ha="right", va="center", fontfamily="monospace")
            ax_s.text(3.0, y, value, fontsize=13, fontweight="bold",
                      color=GOLD, ha="left", va="center", fontfamily="monospace")

        # Barra accuratezza visuale grande
        bar_y = 2.5
        ax_s.text(5, bar_y + 1.0, "ACCURATEZZA", fontsize=11, color=TEXT_CLR,
                  ha="center", fontfamily="monospace")

        # Background bar
        ax_s.barh(bar_y, 6, left=2, height=0.6,
                  color="#2a2a4a", edgecolor=TEXT_CLR, linewidth=0.5)
        # Filled bar
        fill_w = accuracy_pct / 100 * 6
        bar_color = GREEN if accuracy_pct >= 80 else (ORANGE if accuracy_pct >= 65 else RED)
        ax_s.barh(bar_y, fill_w, left=2, height=0.6,
                  color=bar_color, alpha=0.85, edgecolor="none")
        ax_s.text(2 + fill_w + 0.15, bar_y, f"{accuracy_pct:.0f}%",
                  fontsize=14, fontweight="bold", color=bar_color,
                  va="center", fontfamily="monospace")

        # Giudizio
        if accuracy_pct >= 80:
            verdict = "🟢 MODELLO AFFIDABILE"
        elif accuracy_pct >= 65:
            verdict = "🟡 MODELLO BUONO"
        else:
            verdict = "🔴 MODELLO INCERTO"
        ax_s.text(5, 1.0, verdict, fontsize=15, fontweight="bold",
                  color=bar_color, ha="center", va="center",
                  fontfamily="monospace")

        plt.tight_layout(pad=0.5)
        pdf.savefig(fig_s, facecolor=DARK_BG)
        plt.close(fig_s)

    return pdf_path


# ============================================================
# 4. WHAT-IF SCENARIOS
# ============================================================

def what_if_grid_change(year: int, round_num: int, driver: str, new_grid: int):
    """Cosa succede se un pilota parte da un'altra posizione?
    Uso: what_if_grid_change(2023, 22, "VER", 20)"""
    features = _load_features()
    predictor = _train_silent(features)

    race_data = features[
        (features["year"] == year) & (features["round"] == round_num)
    ].copy()
    if race_data.empty:
        print(f"❌ Gara non trovata")
        return

    race_name = race_data["race_name"].iloc[0]
    original = predictor.predict_race(race_data.copy())

    race_modified = race_data.copy()
    mask = race_modified["driver"] == driver
    old_grid = race_modified.loc[mask, "grid_position"].values[0]
    race_modified.loc[mask, "grid_position"] = new_grid
    race_modified.loc[mask, "is_front_row"] = int(new_grid <= 2)
    race_modified.loc[mask, "is_top5"] = int(new_grid <= 5)
    race_modified.loc[mask, "is_top10"] = int(new_grid <= 10)

    modified = predictor.predict_race(race_modified)

    orig_pred = original[original["driver"] == driver]["predicted_position"].values[0]
    mod_pred = modified[modified["driver"] == driver]["predicted_position"].values[0]

    print(f"\n🔮 WHAT-IF: {race_name} {year}")
    print(f"   {driver}: P{int(old_grid)} → P{int(new_grid)}")
    print("=" * 60)
    print(f"  Previsione originale:  P{orig_pred:.1f}")
    print(f"  Previsione modificata: P{mod_pred:.1f}")
    print(f"  Differenza: {mod_pred - orig_pred:+.1f} posizioni")

    print(f"\n  Top 10 con {driver} a P{int(new_grid)}:")
    for _, row in modified.head(10).iterrows():
        marker = " ←" if row["driver"] == driver else ""
        print(f"    P{row['predicted_rank']:.0f} {row['driver']} ({row['team']}){marker}")


# ============================================================
# 5. EVALUATION
# ============================================================

def full_evaluation():
    """Valutazione completa del modello."""
    features = _load_features()
    predictor = F1Predictor()
    predictor.train(features)
    predictions = predictor.predict(features)

    results = evaluate_predictions(predictions)
    print_evaluation_report(results)
    return results


def feature_importance_analysis():
    """Analisi dettagliata feature importance."""
    features = _load_features()
    predictor = F1Predictor()
    results = predictor.train(features)

    importance = results["feature_importance"]
    print(f"\n🔬 FEATURE IMPORTANCE ({results['num_features']} feature)")
    print("=" * 60)
    for _, row in importance.iterrows():
        pct = row["importance"] * 100
        bar = "█" * int(pct * 2)
        print(f"  {row['feature']:35s} {bar} {pct:.1f}%")
    return importance


def compare_base_vs_advanced():
    """Confronto diretto: modello base vs avanzato."""
    features = _load_features()
    predictor = F1Predictor()
    predictor.compare_base_vs_advanced(features)


# ============================================================
# 6. ADVANCED: WEATHER ANALYSIS
# ============================================================

def weather_performance():
    """Come performa ogni team in diverse condizioni meteo?"""
    features = _load_features()

    if "air_temp" not in features.columns:
        print("❌ Dati meteo non disponibili. Esegui: python -m src.advanced_data_loader")
        return

    print("\n🌤️ PERFORMANCE PER CONDIZIONI METEO")
    print("=" * 60)

    # Temperatura
    features["temp_band"] = pd.cut(
        features["air_temp"],
        bins=[0, 20, 28, 50],
        labels=["Freddo (<20°C)", "Medio (20-28°C)", "Caldo (>28°C)"]
    )

    print("\n🌡️ Performance media per team e temperatura:")
    pivot = features.pivot_table(
        values="finish_position", index="team", columns="temp_band", aggfunc="mean"
    ).round(1)
    print(pivot.to_string())

    # Chi migliora di più col freddo vs caldo?
    if "Freddo (<20°C)" in pivot.columns and "Caldo (>28°C)" in pivot.columns:
        pivot["cold_advantage"] = pivot["Caldo (>28°C)"] - pivot["Freddo (<20°C)"]
        print("\n❄️ Team che preferiscono il freddo (valore positivo = meglio col freddo):")
        for team, row in pivot.sort_values("cold_advantage", ascending=False).iterrows():
            if pd.notna(row["cold_advantage"]):
                emoji = "❄️" if row["cold_advantage"] > 0 else "☀️"
                print(f"  {emoji} {team:<20s}: {row['cold_advantage']:+.1f} posizioni")

    # Pioggia
    if "is_wet" in features.columns:
        wet_races = features[features["is_wet"] == 1]
        dry_races = features[features["is_wet"] == 0]

        if len(wet_races) > 20:
            print(f"\n🌧️ Performance sotto la pioggia ({wet_races.groupby(['year','round']).ngroups} gare):")
            wet_perf = wet_races.groupby("driver")["finish_position"].mean().round(1)
            dry_perf = dry_races.groupby("driver")["finish_position"].mean().round(1)

            rain_df = pd.DataFrame({"wet": wet_perf, "dry": dry_perf}).dropna()
            rain_df["rain_advantage"] = rain_df["dry"] - rain_df["wet"]
            rain_df = rain_df.sort_values("rain_advantage", ascending=False)

            print("  Piloti che migliorano sotto la pioggia:")
            for driver, row in rain_df.head(10).iterrows():
                emoji = "🌧️" if row["rain_advantage"] > 0 else "☀️"
                print(f"    {emoji} {driver:4s}: asciutto P{row['dry']:.1f} → "
                      f"pioggia P{row['wet']:.1f} ({row['rain_advantage']:+.1f})")


# ============================================================
# 7. ADVANCED: TYRE ANALYSIS
# ============================================================

def tyre_degradation_analysis():
    """Analisi degradazione gomme per pilota e team."""
    deg_file = PROCESSED_DATA_DIR / "tyre_degradation.csv"

    if not deg_file.exists():
        print("❌ Dati degradazione non disponibili. Esegui: python -m src.advanced_data_loader")
        return

    deg = pd.read_csv(deg_file)

    print("\n🔴 ANALISI DEGRADAZIONE GOMME")
    print("=" * 60)

    # Degradazione media per compound
    print("\n📊 Degradazione media per tipo di gomma (sec/giro):")
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        comp_data = deg[deg["compound"] == compound]
        if not comp_data.empty:
            avg_deg = comp_data["deg_slope"].mean()
            avg_stint = comp_data["stint_length"].mean()
            bar = "█" * int(avg_deg * 50) if avg_deg > 0 else ""
            print(f"  {compound:8s}: +{avg_deg:.3f} sec/giro  "
                  f"(stint medio: {avg_stint:.0f} giri)  {bar}")

    # Migliori gestori di gomme (piloti)
    print("\n🏆 Migliori gestori di gomme (bassa degradazione = meglio):")
    driver_deg = deg.groupby("driver").agg(
        avg_deg=("deg_slope", "mean"),
        consistency=("consistency", "mean"),
        stints=("stint", "count"),
    ).round(4)
    driver_deg = driver_deg[driver_deg["stints"] >= 20]
    driver_deg = driver_deg.sort_values("avg_deg")

    for driver, row in driver_deg.head(10).iterrows():
        bar = "🟢" * max(1, int((0.1 - row["avg_deg"]) * 30))
        print(f"  {driver:4s}: {row['avg_deg']:.4f} sec/giro  "
              f"consistenza: {row['consistency']:.3f}  {bar}")

    print("\n⚠️ Peggiori gestori (alta degradazione):")
    for driver, row in driver_deg.tail(5).iterrows():
        print(f"  {driver:4s}: {row['avg_deg']:.4f} sec/giro")

    # Per team
    print("\n🏎️ Degradazione per team:")
    team_deg = deg.groupby(
        deg["driver"].map(lambda d: _driver_to_team(d, deg))
    )["deg_slope"].mean().sort_values()

    # Approssimazione: usiamo i dati raw
    driver_teams = pd.read_csv(PROCESSED_DATA_DIR / "all_races.csv")[["driver", "team"]].drop_duplicates()
    driver_team_map = dict(zip(driver_teams["driver"], driver_teams["team"]))
    deg["team"] = deg["driver"].map(driver_team_map)

    team_deg = deg.groupby("team")["deg_slope"].mean().sort_values()
    for team, val in team_deg.items():
        bar = "█" * int(val * 50) if val > 0 else ""
        print(f"  {team:<20s}: {val:.4f} sec/giro  {bar}")


def _driver_to_team(driver, deg_df):
    """Helper per mappare pilota → team."""
    return driver  # Placeholder


# ============================================================
# 8. ADVANCED: CIRCUIT ANALYSIS
# ============================================================

def circuit_analysis():
    """Analisi team-circuito fit."""
    if not ADVANCED:
        print("❌ Module avanzati non disponibili")
        return

    print("\n🏁 ANALISI CIRCUITI")
    print("=" * 60)

    # Mostra tutti i circuiti con le loro caratteristiche
    print("\n📋 Caratteristiche circuiti:")
    print(f"  {'Circuito':<30s} {'Tipo':<10s} {'DF':>4s} {'Power':>6s} "
          f"{'Sorpasso':>8s} {'Gomme':>6s}")
    print("-" * 70)

    for name, data in sorted(CIRCUIT_DATA.items()):
        print(
            f"  {name:<30s} {data['type']:<10s} {data['downforce']:>4s} "
            f"{data['power_sensitivity']:>6.1f} "
            f"{data['overtaking_difficulty']:>8.1f} "
            f"{data['tyre_stress']:>6.1f}"
        )


def team_circuit_fit(team_name: str):
    """
    Analisi dettagliata di come un team performa su diversi tipi di circuito.
    Uso: team_circuit_fit("Ferrari")
    """
    df = load_all_data()

    if not ADVANCED:
        print("❌ Module avanzati non disponibili")
        return

    from src.circuit_data import enrich_with_circuit_data
    df = enrich_with_circuit_data(df)

    team_data = df[df["team"].str.contains(team_name, case=False)]
    if team_data.empty:
        print(f"❌ Team '{team_name}' non trovato")
        print(f"   Team disponibili: {sorted(df['team'].unique())}")
        return

    actual_name = team_data["team"].iloc[0]
    print(f"\n🏎️ ANALISI CIRCUITO: {actual_name}")
    print("=" * 60)

    # Per tipo di downforce
    print("\n📊 Performance per livello di downforce:")
    for level in ["low", "medium", "high"]:
        level_data = team_data[team_data["circuit_downforce"] == DOWNFORCE_MAP[level]]
        if not level_data.empty:
            avg = level_data["finish_position"].mean()
            count = level_data.groupby(["year", "round"]).ngroups
            bar = "█" * int(20 - avg) if avg < 20 else ""
            print(f"  {level:8s} downforce: P{avg:.1f} media ({count} gare)  {bar}")

    # Per tipo di circuito
    print("\n📊 Performance per tipo di circuito:")
    for ctype in ["permanent", "hybrid", "street"]:
        from src.circuit_data import CIRCUIT_TYPE_MAP
        type_data = team_data[team_data["circuit_type"] == CIRCUIT_TYPE_MAP[ctype]]
        if not type_data.empty:
            avg = type_data["finish_position"].mean()
            count = type_data.groupby(["year", "round"]).ngroups
            print(f"  {ctype:10s}: P{avg:.1f} media ({count} gare)")

    # Migliori e peggiori circuiti
    print("\n✅ Migliori circuiti:")
    circuit_perf = team_data.groupby("race_name")["finish_position"].mean().sort_values()
    for name, avg in circuit_perf.head(5).items():
        print(f"  P{avg:.1f}  {name}")

    print("\n❌ Peggiori circuiti:")
    for name, avg in circuit_perf.tail(5).items():
        print(f"  P{avg:.1f}  {name}")


def driver_circuit_specialist(driver_code: str):
    """
    Su quali circuiti un pilota è più forte?
    Uso: driver_circuit_specialist("LEC")
    """
    df = load_all_data()
    driver_data = df[df["driver"] == driver_code]

    if driver_data.empty:
        print(f"❌ '{driver_code}' non trovato")
        return

    print(f"\n🎯 CIRCUITI SPECIALITÀ: {driver_code}")
    print("=" * 60)

    circuit_perf = driver_data.groupby("race_name").agg(
        avg_pos=("finish_position", "mean"),
        best=("finish_position", "min"),
        races=("finish_position", "count"),
    ).round(1)
    circuit_perf = circuit_perf[circuit_perf["races"] >= 2]

    print("\n✅ Circuiti migliori:")
    for name, row in circuit_perf.sort_values("avg_pos").head(7).iterrows():
        print(f"  P{row['avg_pos']:.1f} media (best: P{row['best']:.0f}, "
              f"{int(row['races'])} gare)  {name}")

    print("\n❌ Circuiti peggiori:")
    for name, row in circuit_perf.sort_values("avg_pos").tail(5).iterrows():
        print(f"  P{row['avg_pos']:.1f} media (best: P{row['best']:.0f}, "
              f"{int(row['races'])} gare)  {name}")


# ============================================================
# 9. ADVANCED: STRATEGY ANALYSIS
# ============================================================

def strategy_analysis():
    """Analisi strategia pit stop per team."""
    pits_file = PROCESSED_DATA_DIR / "pit_stops.csv"

    if not pits_file.exists():
        print("❌ Dati pit stop non disponibili")
        return

    pits = pd.read_csv(pits_file)
    df = load_all_data()

    # Merge team info
    team_map = dict(zip(
        df.groupby("driver")["team"].last().index,
        df.groupby("driver")["team"].last().values,
    ))
    pits["team"] = pits["driver"].map(team_map)

    print("\n🔧 ANALISI STRATEGIA PIT STOP")
    print("=" * 60)

    # Media pit stop per team
    print("\n📊 Media pit stop per team:")
    team_pits = pits.groupby("team")["num_pit_stops"].mean().sort_values()
    for team, avg in team_pits.items():
        bar = "█" * int(avg * 5)
        print(f"  {team:<20s}: {avg:.2f} soste/gara  {bar}")

    # Strategia aggressiva vs conservativa
    print("\n🎲 Stile strategico (1 sosta = conservativo, 2+ = aggressivo):")
    for team in pits["team"].dropna().unique():
        team_data = pits[pits["team"] == team]
        one_stop = (team_data["num_pit_stops"] == 1).mean() * 100
        two_plus = (team_data["num_pit_stops"] >= 2).mean() * 100
        style = "Conservativo" if one_stop > 50 else "Aggressivo"
        print(f"  {team:<20s}: 1-stop {one_stop:.0f}% | 2+ stop {two_plus:.0f}% → {style}")


# ============================================================
# 10. AGGIORNA STAGIONE — Scarica tutto e ricostruisce le feature
# ============================================================

def aggiorna_stagione():
    """
    Aggiorna tutti i dati della stagione corrente in un solo comando.

    Esegue in ordine:
    1. Download risultati gare (force)
    2. Download dati avanzati (giri, meteo, pit stop)
    3. Download dati prove libere (FP1/FP2/FP3)
    4. Ricalcolo Elo piloti e team (con reset regolamento se necessario)
    5. Ricostruzione feature matrix avanzata

    Tutto viene salvato nei file CSV in data/processed/.
    """
    import time
    from config import SEASONS, REGULATION_RESET_YEARS

    print("\n" + "=" * 60)
    print("🔄 AGGIORNAMENTO STAGIONE COMPLETO")
    print("=" * 60)
    print(f"  Stagioni configurate: {SEASONS}")
    print(f"  Reset regolamento: {REGULATION_RESET_YEARS}")
    start_time = time.time()

    # Step 1: Download risultati gare
    print(f"\n📥 STEP 1/5: Download risultati gare...")
    print("-" * 40)
    try:
        df = load_all_data(force_download=True)
        print(f"  ✅ {len(df)} righe caricate ({df.groupby(['year','round']).ngroups} gare)")
    except Exception as e:
        print(f"  ❌ Errore download gare: {e}")
        return

    # Step 2: Download dati avanzati (meteo, gomme, pit stop)
    print(f"\n📥 STEP 2/5: Download dati avanzati (meteo, gomme, pit stop)...")
    print("-" * 40)
    try:
        from src.advanced_data_loader import download_advanced_data
        download_advanced_data(force=True)
        print(f"  ✅ Dati avanzati scaricati")
    except ImportError:
        print(f"  ⚠️ Modulo advanced_data_loader non trovato. Saltato.")
    except Exception as e:
        print(f"  ⚠️ Errore dati avanzati: {e}")
        print(f"     Continuo senza dati avanzati...")

    # Step 3: Download dati FP
    print(f"\n📥 STEP 3/5: Download dati prove libere...")
    print("-" * 40)
    try:
        from src.fp_data_loader import download_fp_data
        download_fp_data(force=True)
        print(f"  ✅ Dati FP scaricati")
    except ImportError:
        print(f"  ⚠️ Modulo fp_data_loader non trovato. Saltato.")
    except Exception as e:
        print(f"  ⚠️ Errore dati FP: {e}")
        print(f"     Continuo senza dati FP...")

    # Step 4: Ricalcolo Elo piloti e team
    print(f"\n🎯 STEP 4/5: Ricalcolo Elo piloti e team...")
    print("-" * 40)
    elo_system = EloSystem()

    # Elo piloti (usa TUTTO lo storico, nessun reset)
    elo_history = []
    for season_year in sorted(df["year"].unique()):
        season_data = df[df["year"] == season_year]
        season_elo = elo_system.process_season(season_data)
        elo_history.append(season_elo)
    elo_history = pd.concat(elo_history, ignore_index=True)

    # Elo team (con reset regolamentare automatico)
    team_elo_history = compute_team_elo(df)

    elo_history.to_csv(PROCESSED_DATA_DIR / "elo_history.csv", index=False)
    team_elo_history.to_csv(PROCESSED_DATA_DIR / "team_elo_history.csv", index=False)
    print(f"  ✅ Elo piloti: {len(elo_history)} record")
    print(f"  ✅ Elo team: {len(team_elo_history)} record")

    # Mostra classifica Elo attuale
    print(f"\n  🏆 Top 10 Elo piloti:")
    rankings = elo_system.get_current_ratings()
    for i, (_, row) in enumerate(rankings.head(10).iterrows()):
        print(f"    {i+1:2d}. {row['driver']:4s}  {row['elo_rating']:7.1f}")

    # Step 5: Ricostruzione feature matrix
    print(f"\n🔧 STEP 5/5: Ricostruzione feature matrix avanzata...")
    print("-" * 40)
    if ADVANCED:
        features = build_advanced_features(elo_history, team_elo_history)
        output_file = PROCESSED_DATA_DIR / "advanced_features.csv"
        features.to_csv(output_file, index=False)
        print(f"  ✅ Feature avanzate: {features.shape[0]} righe × {features.shape[1]} colonne")
    else:
        from src.feature_engineering import build_feature_matrix
        features = build_feature_matrix(elo_history, team_elo_history)
        output_file = PROCESSED_DATA_DIR / "features.csv"
        features.to_csv(output_file, index=False)
        print(f"  ✅ Feature base: {features.shape[0]} righe × {features.shape[1]} colonne")

    # Riepilogo
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'=' * 60}")
    print(f"✅ AGGIORNAMENTO COMPLETATO in {minutes}m {seconds}s")
    print(f"{'=' * 60}")
    print(f"  Stagioni: {sorted(df['year'].unique().tolist())}")
    print(f"  Gare totali: {df.groupby(['year', 'round']).ngroups}")
    print(f"  Feature: {features.shape[1]} colonne")
    print(f"  File output: {output_file}")

    # Nota regolamento
    current_year = df["year"].max()
    if current_year in REGULATION_RESET_YEARS:
        n_races_era = df[df["year"] == current_year].groupby("round").ngroups
        print(f"\n  ℹ️  Anno {current_year}: nuovi regolamenti — {n_races_era} gare nell'era corrente")
        print(f"     Elo team resettato, storico team filtrato a solo {current_year}+")

    return features


# ============================================================
# 11. SCARICA DATI FP STORICI — Per avere feature FP per il training
# ============================================================

def scarica_fp_storici():
    """
    Scarica i dati delle prove libere (FP1/FP2/FP3) per tutte le stagioni.

    PERCHÉ SERVE?
    Senza dati FP storici il modello non può imparare che le FP sono
    predittive della posizione finale. Il download include ~92 gare × 3
    sessioni, ci vuole un po' ma va fatto una volta sola.

    Dopo il download, usa l'opzione 30 (Aggiorna stagione) per ricostruire
    le feature con i dati FP inclusi.
    """
    print(f"\n{'=' * 60}")
    print(f"📥 DOWNLOAD DATI PROVE LIBERE STORICI")
    print(f"{'=' * 60}")
    print(f"  Questo scarica FP1/FP2/FP3 per tutte le gare disponibili.")
    print(f"  Ci vuole tempo (~10-15 minuti) ma va fatto una volta sola.")
    print(f"  Dopo il download, usa opzione 30 per ricostruire le feature.\n")

    try:
        from src.fp_data_loader import download_fp_data
        download_fp_data(force=True)
        print(f"\n  ✅ Download completato!")
        print(f"  ℹ️  Ora usa opzione 30 per ricostruire le feature con i dati FP.")
    except Exception as e:
        print(f"\n  ❌ Errore: {e}")
        print(f"  Riprova più tardi (potrebbe essere un limite API).")


# ============================================================
# MENU INTERATTIVO
# ============================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          🏎️  F1 PREDICTOR — COMMAND CENTER  🏎️          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  ESPLORA I DATI:                                         ║
║    1.  Panoramica dati                                   ║
║    2.  Griglia vs risultato finale                       ║
║    3.  Analisi DNF                                       ║
║    4.  Migliori rimontatori                              ║
║    5.  Performance team nel tempo                        ║
║                                                          ║
║  ELO RATINGS:                                            ║
║    6.  Classifica Elo attuale                            ║
║    7.  Storico Elo pilota                                ║
║    8.  Confronta piloti                                  ║
║    9.  Testa a testa                                     ║
║                                                          ║
║  PREVISIONI:                                             ║
║   10.  Prevedi una gara specifica                        ║
║   11.  Simula un'intera stagione                         ║
║   22.  Prevedi con dati FP (pre-gara)                    ║
║   23.  Prevedi prossima gara (auto-detect FP)            ║
║   24.  Analisi weekend FP                                ║
║   29.  🏁 PREVISIONI WEEKEND (quali + gara + confidence) ║
║                                                          ║
║  WHAT-IF:                                                ║
║   12.  Cambia griglia di partenza                        ║
║                                                          ║
║  VALUTAZIONE:                                            ║
║   13.  Valutazione completa modello                      ║
║   14.  Feature importance                                ║
║   15.  Confronto base vs avanzato                        ║
║                                                          ║
║  ANALISI AVANZATE:                                       ║
║   16.  Performance per condizioni meteo                  ║
║   17.  Degradazione gomme                                ║
║   18.  Caratteristiche circuiti                          ║
║   19.  Team-circuito fit                                 ║
║   20.  Circuiti specialità pilota                        ║
║   21.  Analisi strategia pit stop                        ║
║                                                          ║
║  GRAFICI FP:                                             ║
║   25.  Long run box plots (singola FP)                   ║
║   26.  Long run traces / degradazione                    ║
║   27.  Telemetria top 3 best laps                        ║
║   28.  Tutti i grafici del weekend                       ║
║                                                          ║
║  AGGIORNAMENTO:                                          ║
║   30.  🔄 Aggiorna stagione (scarica + ricostruisci)     ║
║   31.  📥 Scarica dati FP storici (2022-2026)            ║
║                                                          ║
║  0. Esci                                                 ║
╚══════════════════════════════════════════════════════════╝
""")

    simple_commands = {
        1: explore_data,
        2: grid_vs_finish,
        3: dnf_analysis,
        4: best_overtakers,
        5: team_performance_over_time,
        6: show_elo_rankings,
        13: full_evaluation,
        14: feature_importance_analysis,
        15: compare_base_vs_advanced,
        16: weather_performance,
        17: tyre_degradation_analysis,
        18: circuit_analysis,
        21: strategy_analysis,
    }

    try:
        choice = int(input("→ Scelta: "))

        if choice == 0:
            print("Ciao! 👋")
        elif choice == 7:
            d = input("  Codice pilota (es. VER): ").strip().upper()
            show_driver_elo_history(d)
        elif choice == 8:
            drivers = input("  Piloti separati da virgola (es. VER,HAM,NOR): ")
            codes = [d.strip().upper() for d in drivers.split(",")]
            compare_drivers(*codes)
        elif choice == 9:
            a = input("  Pilota A: ").strip().upper()
            b = input("  Pilota B: ").strip().upper()
            head_to_head(a, b)
        elif choice == 10:
            y = int(input("  Anno: "))
            r = int(input("  Round: "))
            predict_specific_race(y, r)
        elif choice == 11:
            y = int(input("  Anno: "))
            simulate_season(y)
        elif choice == 12:
            y = int(input("  Anno: "))
            r = int(input("  Round: "))
            d = input("  Pilota: ").strip().upper()
            g = int(input("  Nuova griglia: "))
            what_if_grid_change(y, r, d, g)
        elif choice == 19:
            t = input("  Nome team (es. Ferrari): ").strip()
            team_circuit_fit(t)
        elif choice == 20:
            d = input("  Codice pilota (es. LEC): ").strip().upper()
            driver_circuit_specialist(d)
        elif choice == 22:
            y = int(input("  Anno: "))
            r = int(input("  Round: "))
            predict_with_fp(y, r)
        elif choice == 23:
            predict_next_race()
        elif choice == 24:
            y = int(input("  Anno: "))
            r = int(input("  Round: "))
            fp_weekend_analysis(y, r)
        elif choice == 29:
            y = int(input("  Anno: "))
            r = int(input("  Round: "))
            predict_weekend(y, r)
        elif choice in [25, 26, 27, 28]:
            if not FP_PLOTS:
                print("❌ Modulo grafici non disponibile. Controlla le dipendenze.")
            else:
                y = int(input("  Anno: "))
                r = int(input("  Round: "))
                if choice == 28:
                    plot_fp_weekend(y, r, save=True, show=True)
                else:
                    s = input("  Sessione (FP1/FP2/FP3, default FP2): ").strip().upper() or "FP2"
                    if choice == 25:
                        plot_long_runs(y, r, s, save=True, show=True)
                    elif choice == 26:
                        plot_long_run_traces(y, r, s, save=True, show=True)
                    elif choice == 27:
                        plot_telemetry_top3(y, r, s, save=True, show=True)
        elif choice == 30:
            aggiorna_stagione()
        elif choice == 31:
            scarica_fp_storici()
        elif choice in simple_commands:
            simple_commands[choice]()
        else:
            print("Scelta non valida")
    except (ValueError, EOFError):
        print("Input non valido")