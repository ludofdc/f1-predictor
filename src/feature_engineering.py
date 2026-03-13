"""
feature_engineering.py — Creazione delle Feature per il Modello
===============================================================

COS'È UNA FEATURE?
Nel machine learning, una "feature" è un'informazione che dai al modello
per aiutarlo a fare previsioni. È come i fattori che TU considereresti
se dovessi prevedere chi vince una gara:

- "Verstappen parte dalla pole" → feature: grid_position = 1
- "Ha un Elo alto" → feature: elo_pre_race = 1650
- "Su questa pista va sempre forte" → feature: circuit_avg_position = 2.3
- "Nelle ultime 5 gare è andato benissimo" → feature: recent_form = 1.8

Questo modulo prende i dati grezzi e li trasforma in feature utili.

CONCETTO CHIAVE: "Feature Engineering"
È spesso la parte PIÙ importante di un progetto ML.
Un modello semplice con feature intelligenti batte quasi sempre
un modello complesso con feature scadenti.
"""

import pandas as pd
import numpy as np
import sys

sys.path.append("..")
from config import ROLLING_WINDOW, PROCESSED_DATA_DIR


def add_recent_form(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Calcola la "forma recente" di ogni pilota: media delle ultime N posizioni.

    Esempio con window=5:
    Ultime 5 gare di Verstappen: [1, 2, 1, 3, 1] → media = 1.6 (ottima forma)
    Ultime 5 gare di Tsunoda:    [12, 8, 15, 11, 9] → media = 11.0

    COME FUNZIONA:
    - groupby("driver"): raggruppa per pilota
    - rolling(window): prende una "finestra mobile" di N righe
    - shift(1): FONDAMENTALE! Shifta di 1 per usare solo dati PASSATI
      (non puoi usare il risultato di QUESTA gara per predire QUESTA gara!)
    """
    df = df.sort_values(["driver", "year", "round"]).copy()

    # Per ogni pilota, calcoliamo la media mobile delle ultime N posizioni
    df["recent_form"] = (
        df.groupby("driver")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    # Stessa cosa per il tasso di DNF recente
    df["recent_dnf_rate"] = (
        df.groupby("driver")["dnf"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    return df


def add_circuit_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola come va ogni pilota su ogni circuito specifico.

    Alcuni piloti sono "specialisti" di certi circuiti:
    - Hamilton a Silverstone
    - Verstappen a Spa
    - Leclerc a Monaco (meno fortunato, ma veloce!)

    Per ogni combinazione (pilota, circuito), calcoliamo la media storica
    delle posizioni finali PRECEDENTI (mai includere la gara corrente!).
    """
    df = df.sort_values(["year", "round"]).copy()

    circuit_stats = []

    for idx, row in df.iterrows():
        driver = row["driver"]
        race_name = row["race_name"]
        year = row["year"]
        round_num = row["round"]

        # Prendiamo solo le gare PRECEDENTI su questo circuito per questo pilota
        past_at_circuit = df[
            (df["driver"] == driver)
            & (df["race_name"] == race_name)
            & ((df["year"] < year) | ((df["year"] == year) & (df["round"] < round_num)))
        ]

        if len(past_at_circuit) > 0:
            circuit_avg = past_at_circuit["finish_position"].mean()
            circuit_races = len(past_at_circuit)
        else:
            circuit_avg = np.nan  # NaN = "non abbiamo dati"
            circuit_races = 0

        circuit_stats.append(
            {
                "idx": idx,
                "circuit_avg_position": circuit_avg,
                "circuit_experience": circuit_races,
            }
        )

    circuit_df = pd.DataFrame(circuit_stats).set_index("idx")
    df = df.join(circuit_df)

    return df


def add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature derivate dalla posizione in griglia (qualifica).

    In F1 la qualifica è molto predittiva: chi parte davanti, spesso finisce davanti.
    Ma non è tutto: creiamo feature che catturano sfumature:

    - grid_position: posizione pura
    - is_front_row: binaria, 1 se parti 1°-2° (vantaggio enorme alla partenza)
    - is_top10: binaria, 1 se parti in top 10
    - grid_group: categorica, raggruppa le posizioni in fasce
    """
    df = df.copy()

    # Feature binarie
    df["is_front_row"] = (df["grid_position"] <= 2).astype(int)
    df["is_top5"] = (df["grid_position"] <= 5).astype(int)
    df["is_top10"] = (df["grid_position"] <= 10).astype(int)

    return df


def build_feature_matrix(
    elo_history: pd.DataFrame,
    team_elo_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Funzione principale: combina TUTTE le feature in un unico DataFrame
    pronto per il modello di machine learning.

    È come preparare gli ingredienti prima di cucinare:
    ogni feature è un ingrediente, questa funzione li mette tutti insieme.
    """
    print("🔧 Costruzione feature matrix...")

    # Partiamo dall'Elo history (che ha già elo_pre_race)
    df = elo_history.copy()

    # Aggiungiamo l'Elo del team
    df = df.merge(
        team_elo_history[["year", "round", "team", "team_elo_pre_race"]],
        on=["year", "round", "team"],
        how="left",
    )

    # Aggiungiamo le feature
    print("  📊 Calcolo forma recente...")
    df = add_recent_form(df)

    print("  🏁 Calcolo storico circuito...")
    df = add_circuit_history(df)

    print("  📍 Calcolo feature griglia...")
    df = add_grid_features(df)

    # Riempiamo i NaN con valori sensati
    # (le prime gare di un pilota non hanno storico)
    df["recent_form"] = df["recent_form"].fillna(df["finish_position"].median())
    df["circuit_avg_position"] = df["circuit_avg_position"].fillna(
        df["finish_position"].median()
    )
    df["recent_dnf_rate"] = df["recent_dnf_rate"].fillna(0)
    df["circuit_experience"] = df["circuit_experience"].fillna(0)

    print(f"  ✅ Feature matrix: {df.shape[0]} righe × {df.shape[1]} colonne")

    return df


# Lista delle feature che il modello userà per le previsioni
FEATURE_COLUMNS = [
    "grid_position",
    "elo_pre_race",
    "team_elo_pre_race",
    "recent_form",
    "recent_dnf_rate",
    "circuit_avg_position",
    "circuit_experience",
    "is_front_row",
    "is_top5",
    "is_top10",
]

# Colonna target: quello che vogliamo prevedere
TARGET_COLUMN = "finish_position"


if __name__ == "__main__":
    # Test: carichiamo i dati e costruiamo le feature
    elo_hist = pd.read_csv(PROCESSED_DATA_DIR / "elo_history.csv")
    team_elo_hist = pd.read_csv(PROCESSED_DATA_DIR / "team_elo_history.csv")

    features_df = build_feature_matrix(elo_hist, team_elo_hist)

    print("\n📊 Anteprima feature matrix:")
    print(features_df[FEATURE_COLUMNS + [TARGET_COLUMN]].head(10))

    print("\n📈 Correlazioni con la posizione finale:")
    correlations = features_df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()[
        TARGET_COLUMN
    ]
    print(correlations.sort_values())

    # Salviamo
    features_df.to_csv(PROCESSED_DATA_DIR / "features.csv", index=False)
    print(f"\n💾 Feature salvate in {PROCESSED_DATA_DIR / 'features.csv'}")