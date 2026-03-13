"""
evaluation.py — Valutazione del Modello
========================================

Come facciamo a sapere se il nostro modello è "buono"?
Non basta dire "ha indovinato il vincitore 3 volte su 5".
Servono metriche precise e confronti con baseline sensate.

METRICHE CHE USIAMO:

1. MAE (Mean Absolute Error):
   Media dell'errore assoluto. Es: se prevedi P3 e il risultato è P5,
   l'errore è |3-5| = 2 posizioni. Facile da interpretare.

2. RMSE (Root Mean Squared Error):
   Come MAE ma penalizza di più gli errori grandi.
   Se sbagli una previsione di 10 posizioni, RMSE lo "punisce" molto.

3. Top-N Accuracy:
   Quante volte il modello indovina i top 3? I top 5? Il podio?

4. Confronto con Baseline:
   Il modello batte la strategia "stupida" di dire che tutti finiscono
   nella posizione di partenza? Se no, il modello è inutile.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(df: pd.DataFrame) -> dict:
    """
    Calcola tutte le metriche di valutazione.

    Il DataFrame deve avere le colonne:
    - finish_position: posizione reale
    - predicted_position: posizione prevista dal modello
    - grid_position: posizione in griglia (per la baseline)
    """
    actual = df["finish_position"]
    predicted = df["predicted_position"]
    grid = df["grid_position"]

    # --- Metriche principali ---
    model_mae = mean_absolute_error(actual, predicted)
    model_rmse = np.sqrt(mean_squared_error(actual, predicted))

    # --- Baseline: "tutti finiscono dove partono" ---
    # Se il nostro modello non batte questa baseline, è inutile!
    valid_grid = df.dropna(subset=["grid_position"])
    baseline_mae = mean_absolute_error(
        valid_grid["finish_position"], valid_grid["grid_position"]
    )
    baseline_rmse = np.sqrt(
        mean_squared_error(valid_grid["finish_position"], valid_grid["grid_position"])
    )

    # --- Top-N Accuracy ---
    # Per ogni gara: quanti dei veri top N erano nei nostri top N previsti?
    top_n_results = {}
    for n in [1, 3, 5, 10]:
        correct = 0
        total = 0

        for (year, round_num), race_df in df.groupby(["year", "round"]):
            if len(race_df) < n:
                continue

            actual_top_n = set(
                race_df.nsmallest(n, "finish_position")["driver"]
            )
            predicted_top_n = set(
                race_df.nsmallest(n, "predicted_position")["driver"]
            )

            # Quanti piloti compaiono in ENTRAMBI i set?
            overlap = len(actual_top_n & predicted_top_n)
            correct += overlap
            total += n

        accuracy = correct / total if total > 0 else 0
        top_n_results[f"top{n}_accuracy"] = accuracy

    # --- Risultati ---
    results = {
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "improvement_mae": (baseline_mae - model_mae) / baseline_mae * 100,
        "improvement_rmse": (baseline_rmse - model_rmse) / baseline_rmse * 100,
        **top_n_results,
    }

    return results


def print_evaluation_report(results: dict):
    """Stampa un report leggibile delle metriche."""

    print("\n" + "=" * 60)
    print("📊 REPORT DI VALUTAZIONE DEL MODELLO")
    print("=" * 60)

    print("\n--- Errore Medio ---")
    print(f"  Modello MAE:   {results['model_mae']:.2f} posizioni")
    print(f"  Baseline MAE:  {results['baseline_mae']:.2f} posizioni")
    improvement = results["improvement_mae"]
    emoji = "✅" if improvement > 0 else "❌"
    print(f"  Miglioramento: {emoji} {improvement:+.1f}%")

    print("\n--- Errore Quadratico ---")
    print(f"  Modello RMSE:  {results['model_rmse']:.2f} posizioni")
    print(f"  Baseline RMSE: {results['baseline_rmse']:.2f} posizioni")
    improvement_rmse = results["improvement_rmse"]
    emoji = "✅" if improvement_rmse > 0 else "❌"
    print(f"  Miglioramento: {emoji} {improvement_rmse:+.1f}%")

    print("\n--- Top-N Accuracy ---")
    print(f"  Vincitore (Top 1): {results['top1_accuracy']:.1%}")
    print(f"  Podio (Top 3):     {results['top3_accuracy']:.1%}")
    print(f"  Top 5:             {results['top5_accuracy']:.1%}")
    print(f"  Top 10:            {results['top10_accuracy']:.1%}")

    print("\n--- Interpretazione ---")
    if results["improvement_mae"] > 10:
        print("  🎉 Il modello batte significativamente la baseline!")
        print("  Le feature (Elo, forma, circuito) aggiungono valore reale.")
    elif results["improvement_mae"] > 0:
        print("  👍 Il modello batte la baseline, ma di poco.")
        print("  C'è margine di miglioramento nelle feature o nel modello.")
    else:
        print("  ⚠️ Il modello NON batte la baseline.")
        print("  La griglia di partenza da sola è più predittiva.")
        print("  Rivedi le feature o prova un modello diverso.")

    print("=" * 60)


def per_race_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analisi gara per gara: mostra dove il modello funziona e dove no.
    Utile per capire i pattern (es: il modello sbaglia di più sotto la pioggia?).
    """
    race_results = []

    for (year, round_num), race_df in df.groupby(["year", "round"]):
        race_name = race_df["race_name"].iloc[0] if "race_name" in race_df else f"R{round_num}"

        actual = race_df["finish_position"]
        predicted = race_df["predicted_position"]

        mae = mean_absolute_error(actual, predicted)

        # Il modello ha indovinato il vincitore?
        actual_winner = race_df.loc[actual.idxmin(), "driver"]
        predicted_winner = race_df.loc[predicted.idxmin(), "driver"]
        correct_winner = actual_winner == predicted_winner

        race_results.append(
            {
                "year": year,
                "round": round_num,
                "race_name": race_name,
                "mae": mae,
                "correct_winner": correct_winner,
                "actual_winner": actual_winner,
                "predicted_winner": predicted_winner,
            }
        )

    return pd.DataFrame(race_results)


if __name__ == "__main__":
    from src.model import F1Predictor

    # Carica dati e modello
    df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    predictor = F1Predictor()
    predictor.train(df)

    # Previsioni su tutto il dataset
    predictions = predictor.predict(df)

    # Valutazione
    results = evaluate_predictions(predictions)
    print_evaluation_report(results)

    # Analisi per gara
    print("\n🏁 Analisi per gara (ultime 10):")
    race_analysis = per_race_analysis(predictions)
    print(race_analysis.tail(10).to_string(index=False))