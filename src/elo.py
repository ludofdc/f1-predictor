"""
elo.py — Sistema di Rating Elo adattato alla Formula 1
=======================================================

COS'È IL SISTEMA ELO?
Il sistema Elo è stato inventato da Arpad Elo negli anni '60 per classificare
i giocatori di scacchi. L'idea geniale è semplice:

1. Ogni giocatore ha un RATING numerico (partenza: 1500)
2. Prima di una partita, calcoli la PROBABILITÀ ATTESA di vittoria
   basandoti sulla differenza di rating
3. Dopo la partita, aggiorni i rating:
   - Se vinci contro uno più forte → guadagni MOLTI punti
   - Se vinci contro uno più debole → guadagni POCHI punti
   - Se perdi contro uno più debole → perdi MOLTI punti

LA FORMULA MATEMATICA:
    Probabilità attesa di A vs B:
    E(A) = 1 / (1 + 10^((R_B - R_A) / 400))

    Aggiornamento rating:
    R_new = R_old + K * (S - E)

    Dove:
    - R = rating
    - K = "velocità di aggiornamento" (il nostro K-factor)
    - S = risultato reale (1 = vittoria, 0 = sconfitta)
    - E = risultato atteso (la probabilità calcolata sopra)

ADATTAMENTO ALLA F1:
In F1 non c'è un singolo "vincitore vs perdente", ci sono 20 piloti.
La soluzione: trattiamo ogni gara come una serie di CONFRONTI A COPPIE.
Se finisci 3° e un altro finisce 7°, hai "vinto" contro di lui.
Sommiamo tutti i confronti e aggiorniamo il rating di conseguenza.
"""

import pandas as pd
import numpy as np
import sys

sys.path.append("..")
from config import (
    ELO_INITIAL_RATING,
    ELO_K_FACTOR,
    ELO_SCALE_FACTOR,
    ELO_FP_BOOTSTRAP_SCALE,
    PROCESSED_DATA_DIR,
    REGULATION_RESET_YEARS,
)


class EloSystem:
    """
    Classe che gestisce il sistema Elo per la F1.

    COS'È UNA CLASSE?
    Pensa a una classe come a uno "stampo" per creare oggetti.
    EloSystem è lo stampo: definisce COSA sa fare il sistema Elo.
    Quando scrivi `elo = EloSystem()`, crei un OGGETTO concreto da quello stampo.

    L'oggetto tiene in memoria:
    - I rating di tutti i piloti (self.ratings)
    - La storia dei rating nel tempo (self.history)
    """

    def __init__(self, k_factor: float = ELO_K_FACTOR):
        """
        __init__ è il "costruttore": viene chiamato quando crei l'oggetto.

        self = l'oggetto stesso (come "io" in italiano)
        self.ratings = un dizionario {nome_pilota: rating}
        """
        self.k_factor = k_factor
        self.ratings = {}  # Dizionario vuoto: lo riempiremo gara dopo gara
        self.history = []  # Lista vuota: salveremo lo storico qui

    def get_rating(self, driver: str) -> float:
        """
        Restituisce il rating di un pilota.
        Se il pilota è nuovo (non l'abbiamo mai visto), gli assegniamo 1500.

        Il metodo .setdefault() è un trucco elegante di Python:
        "Se la chiave esiste, restituisci il valore. Altrimenti, inseriscila
        con il valore di default e restituisci quello."
        """
        return self.ratings.setdefault(driver, ELO_INITIAL_RATING)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calcola la probabilità attesa che A batta B.

        Esempio:
        - A ha rating 1600, B ha 1400 (differenza = 200)
        - E(A) = 1 / (1 + 10^(-200/400)) = 1 / (1 + 10^(-0.5)) ≈ 0.76
        - Cioè: A ha il 76% di probabilità di battere B

        Se i rating sono uguali:
        - E = 1 / (1 + 10^0) = 1 / 2 = 0.5 (50-50, logico!)
        """
        exponent = (rating_b - rating_a) / ELO_SCALE_FACTOR
        return 1.0 / (1.0 + 10.0 ** exponent)

    def update_after_race(self, race_result: list[tuple[str, int]]):
        """
        Aggiorna i rating dopo una gara.

        Parametri:
            race_result: lista di tuple (pilota, posizione_finale)
                         Es: [("VER", 1), ("NOR", 2), ("LEC", 3), ...]

        ALGORITMO:
        Per ogni coppia di piloti (A, B):
        1. Calcoliamo il risultato atteso E(A vs B)
        2. Il risultato reale S è:
           - 1.0 se A ha finito davanti a B
           - 0.0 se A ha finito dietro a B
        3. Accumuliamo (S - E) per ogni confronto
        4. Alla fine, aggiorniamo: R_new = R_old + K * somma(S - E) / num_confronti

        Dividiamo per num_confronti per normalizzare: altrimenti il rating
        cambierebbe troppo perché ci sono ~190 confronti a coppia per gara!
        """
        # Filtriamo i piloti che non hanno una posizione valida (DNF senza classificazione)
        valid_results = [
            (driver, pos) for driver, pos in race_result
            if pd.notna(pos) and pos > 0
        ]

        if len(valid_results) < 2:
            return

        # Calcoliamo gli aggiornamenti per ogni pilota
        rating_changes = {}

        for i, (driver_a, pos_a) in enumerate(valid_results):
            rating_a = self.get_rating(driver_a)
            total_delta = 0.0
            num_comparisons = 0

            for j, (driver_b, pos_b) in enumerate(valid_results):
                if i == j:
                    continue  # Non confrontare un pilota con se stesso!

                rating_b = self.get_rating(driver_b)

                # Risultato atteso (probabilità)
                expected = self.expected_score(rating_a, rating_b)

                # Risultato reale: 1 se A davanti a B, 0 altrimenti
                # (posizione più BASSA = più davanti, es. 1° > 3°)
                actual = 1.0 if pos_a < pos_b else 0.0

                # Accumuliamo la differenza
                total_delta += actual - expected
                num_comparisons += 1

            if num_comparisons > 0:
                # Normalizziamo e applichiamo il K-factor
                rating_changes[driver_a] = (
                    self.k_factor * total_delta / num_comparisons
                )

        # Applichiamo TUTTI i cambiamenti contemporaneamente
        # (importante: non aggiornare uno alla volta, altrimenti i confronti
        #  successivi userebbero rating già modificati!)
        for driver, change in rating_changes.items():
            self.ratings[driver] = self.get_rating(driver) + change

    def process_season(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa un intero DataFrame di risultati gara dopo gara.

        Per ogni gara:
        1. Salviamo i rating PRIMA della gara (utili come feature per la predizione)
        2. Aggiorniamo i rating con il risultato
        3. Salviamo tutto nello storico

        Ritorna un DataFrame con i rating pre-gara per ogni pilota per ogni gara.
        """
        history_records = []

        # Raggruppiamo per (anno, round) = una gara
        for (year, round_num), race_df in df.groupby(["year", "round"]):
            race_name = race_df["race_name"].iloc[0]

            # 1. Salviamo i rating PRE-GARA (prima dell'aggiornamento)
            for _, row in race_df.iterrows():
                driver = row["driver"]
                history_records.append(
                    {
                        "year": year,
                        "round": round_num,
                        "race_name": race_name,
                        "driver": driver,
                        "team": row["team"],
                        "elo_pre_race": self.get_rating(driver),
                        "grid_position": row["grid_position"],
                        "finish_position": row["finish_position"],
                        "dnf": row["dnf"],
                    }
                )

            # 2. Aggiorniamo i rating con il risultato della gara
            race_result = list(
                zip(race_df["driver"], race_df["finish_position"])
            )
            self.update_after_race(race_result)

            # 3. Aggiungiamo i rating POST-GARA allo storico
            for record in history_records[-len(race_df):]:
                record["elo_post_race"] = self.get_rating(record["driver"])

        return pd.DataFrame(history_records)

    def get_current_ratings(self) -> pd.DataFrame:
        """
        Restituisce i rating attuali come DataFrame ordinato.
        Utile per vedere la "classifica Elo" attuale.
        """
        data = [
            {"driver": driver, "elo_rating": rating}
            for driver, rating in self.ratings.items()
        ]
        return (
            pd.DataFrame(data)
            .sort_values("elo_rating", ascending=False)
            .reset_index(drop=True)
        )

    def reset_all_ratings(self):
        """
        Resetta TUTTI i rating al valore iniziale (1500).

        Usato quando cambiano i regolamenti tecnici (es. 2026):
        le vetture sono completamente diverse, quindi la storia
        competitiva dei team non è più rappresentativa.

        NOTA: questo metodo NON viene mai chiamato per l'Elo piloti,
        solo per l'Elo team (la bravura del pilota non cambia con le regole).
        """
        for entity in list(self.ratings.keys()):
            self.ratings[entity] = ELO_INITIAL_RATING


def compute_team_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola anche l'Elo per i TEAM (costruttori).
    Stessa logica, ma raggruppiamo per team invece che per pilota.

    Per il team, il "risultato" è la posizione del loro miglior pilota.
    """
    team_elo = EloSystem(k_factor=ELO_K_FACTOR * 0.7)
    # K-factor più basso per i team: cambiano meno rapidamente dei piloti

    history_records = []
    last_year_processed = None

    for (year, round_num), race_df in df.groupby(["year", "round"]):
        # ── RESET REGOLAMENTO ──
        # Quando entriamo in un anno di cambio regolamento (es. 2026),
        # resettiamo l'Elo di tutti i team perché le vetture sono
        # completamente nuove e la storia non è più rappresentativa.
        if (
            year in REGULATION_RESET_YEARS
            and last_year_processed is not None
            and last_year_processed < year
        ):
            team_elo.reset_all_ratings()

            # ── BOOTSTRAP FP ──
            # A inizio nuova era, usiamo i dati FP del primo round
            # per inizializzare l'Elo team con la gerarchia reale
            # invece di lasciare tutti a 1500.
            fp_summary_file = PROCESSED_DATA_DIR / "fp_summary.csv"
            if fp_summary_file.exists():
                fp_summary = pd.read_csv(fp_summary_file)
                first_round_fp = fp_summary[fp_summary["year"] == year]
                if not first_round_fp.empty:
                    first_round_num = first_round_fp["round"].min()
                    first_round_fp = first_round_fp[
                        first_round_fp["round"] == first_round_num
                    ]
                    # Mapping driver→team dalla gara corrente
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

        last_year_processed = year

        # Per ogni team, prendiamo la miglior posizione finale
        team_results = (
            race_df.groupby("team")["finish_position"]
            .min()  # min = posizione migliore (1° > 2°)
            .reset_index()
        )

        # Salviamo rating pre-gara
        for _, row in team_results.iterrows():
            team = row["team"]
            history_records.append(
                {
                    "year": year,
                    "round": round_num,
                    "team": team,
                    "team_elo_pre_race": team_elo.get_rating(team),
                }
            )

        # Aggiorniamo
        result_list = list(
            zip(team_results["team"], team_results["finish_position"])
        )
        team_elo.update_after_race(result_list)

    return pd.DataFrame(history_records)


# ============================================================
# ESECUZIONE DIRETTA: python -m src.elo
# ============================================================
if __name__ == "__main__":
    from src.data_loader import load_all_data

    # 1. Carichiamo i dati
    print("📂 Caricamento dati...")
    df = load_all_data()

    # 2. Calcoliamo Elo piloti
    print("\n🎯 Calcolo Elo piloti...")
    elo_system = EloSystem()
    elo_history = elo_system.process_season(df)

    # 3. Calcoliamo Elo team
    print("🏗️ Calcolo Elo team...")
    team_elo_history = compute_team_elo(df)

    # 4. Salviamo
    elo_history.to_csv(PROCESSED_DATA_DIR / "elo_history.csv", index=False)
    team_elo_history.to_csv(
        PROCESSED_DATA_DIR / "team_elo_history.csv", index=False
    )

    # 5. Mostriamo la classifica attuale
    print("\n🏆 Classifica Elo attuale (piloti):")
    print(elo_system.get_current_ratings().head(15).to_string(index=False))