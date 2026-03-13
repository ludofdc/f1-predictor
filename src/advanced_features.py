"""
advanced_features.py — Feature Engineering Avanzato
====================================================
Questo modulo crea feature sofisticate usando TUTTI i dati disponibili:

1. METEO → Come performa ogni team/pilota con caldo/freddo/pioggia
2. GOMME → Degradazione media, gestione gomme, strategia compound
3. STRATEGIA → Numero pit stop, timing, undercut/overcut
4. CIRCUITO → Tipo di pista, downforce, power sensitivity
5. TEAM-CIRCUITO FIT → Quanto un team performa su un tipo di circuito
6. PROVE LIBERE → Best lap, long run pace, degradazione FP, passo per compound
7. MOMENTUM STAGIONALE → Trend recente, velocità Elo, forma pesata

Queste feature catturano pattern che il modello base non può vedere.
Per esempio: "Ferrari va forte sui circuiti ad alto carico aerodinamico
ma soffre a Monza" oppure "Mercedes migliora quando fa freddo".
Le feature FP catturano il "form del weekend" specifico: i dati delle
prove libere sono il miglior indicatore del potenziale reale in gara.
"""

import pandas as pd
import numpy as np
import sys

sys.path.append("..")
from config import PROCESSED_DATA_DIR, ROLLING_WINDOW, get_regulation_era_start
from src.circuit_data import enrich_with_circuit_data, DOWNFORCE_MAP, CIRCUIT_TYPE_MAP


# ============================================================
# HELPER: FILTRO ERA REGOLAMENTARE (per feature team)
# ============================================================

def _filter_regulation_era(
    df: pd.DataFrame, current_year: int, current_round: int,
    entity_col: str, entity_value,
) -> pd.DataFrame:
    """
    Filtra i dati storici di un TEAM tenendo conto dei cambiamenti regolamentari.

    Quando siamo in un anno post-reset regolamento (es. 2026), la storia
    dei team dalle ere precedenti è irrilevante perché le vetture sono
    completamente diverse. Quindi teniamo solo i dati dell'era corrente.

    Per i PILOTI questo filtro NON viene usato: la bravura resta.

    Parametri:
        df: DataFrame completo
        current_year: anno della gara corrente
        current_round: round della gara corrente
        entity_col: colonna dell'entità (es. "team")
        entity_value: valore dell'entità (es. "Ferrari")

    Ritorna:
        DataFrame filtrato con solo i dati dell'era regolamentare corrente,
        e solo le gare PRIMA di quella corrente (no data leakage).
    """
    era_start = get_regulation_era_start(current_year)

    past = df[
        (df[entity_col] == entity_value)
        & (df["year"] >= era_start)
        & (
            (df["year"] < current_year)
            | ((df["year"] == current_year) & (df["round"] < current_round))
        )
    ]
    return past


# ============================================================
# 1. FEATURE METEO
# ============================================================

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge dati meteo e calcola come ogni team/pilota performa
    in diverse condizioni meteorologiche.

    PERCHÉ CONTA?
    - Alcune macchine funzionano meglio col freddo (finestra gomme)
    - La pioggia cambia completamente la gerarchia
    - L'umidità alta influenza le power unit
    """
    weather_file = PROCESSED_DATA_DIR / "weather.csv"

    if not weather_file.exists():
        print("  ⚠️ Dati meteo non disponibili. Esegui: python -m src.advanced_data_loader")
        return df

    weather = pd.read_csv(weather_file)

    # Merge: aggiungiamo il meteo della gara al DataFrame
    df = df.merge(
        weather[["year", "round", "air_temp", "track_temp",
                 "humidity", "rainfall", "wind_speed"]],
        on=["year", "round"],
        how="left",
    )

    # Categorizzazione temperatura
    # Freddo < 20°C, Medio 20-30°C, Caldo > 30°C
    df["temp_category"] = pd.cut(
        df["air_temp"],
        bins=[-np.inf, 20, 30, np.inf],
        labels=[0, 1, 2],  # 0=freddo, 1=medio, 2=caldo
    ).astype(float)

    # Pioggia come feature binaria
    df["is_wet"] = df["rainfall"].astype(float).fillna(0)

    # Performance storica del TEAM per categoria di temperatura
    # "Come va Ferrari quando fa freddo vs quando fa caldo?"
    df = df.sort_values(["year", "round"]).copy()

    for temp_cat in [0, 1, 2]:
        col_name = f"team_perf_temp_{int(temp_cat)}"
        df[col_name] = np.nan

        for idx, row in df.iterrows():
            # Filtro regulation-aware: solo dati dell'era corrente per i team
            past = _filter_regulation_era(df, row["year"], row["round"], "team", row["team"])
            past = past[past["temp_category"] == temp_cat]
            if len(past) >= 3:
                df.at[idx, col_name] = past["finish_position"].mean()

    # Performance storica sotto la pioggia
    df["team_wet_performance"] = np.nan
    df["driver_wet_performance"] = np.nan

    for idx, row in df.iterrows():
        # Team: filtrato per era regolamentare
        past_wet_team = _filter_regulation_era(df, row["year"], row["round"], "team", row["team"])
        past_wet_team = past_wet_team[past_wet_team["is_wet"] == 1]
        if len(past_wet_team) >= 2:
            df.at[idx, "team_wet_performance"] = past_wet_team["finish_position"].mean()

        # Driver: usa TUTTO lo storico (la bravura non cambia col regolamento)
        past_wet_driver = df[
            (df["driver"] == row["driver"])
            & (df["is_wet"] == 1)
            & ((df["year"] < row["year"])
               | ((df["year"] == row["year"]) & (df["round"] < row["round"])))
        ]
        if len(past_wet_driver) >= 2:
            df.at[idx, "driver_wet_performance"] = past_wet_driver["finish_position"].mean()

    return df


# ============================================================
# 2. FEATURE GOMME E DEGRADAZIONE
# ============================================================

def add_tyre_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature sulla gestione delle gomme.

    PERCHÉ CONTA?
    - Un team che degrada meno le gomme può fare stint più lunghi
    - Stint più lunghi = meno pit stop = meno tempo perso
    - La degradazione varia per compound: SOFT degrada molto, HARD poco
    - Alcuni piloti sono "tyre whisperers" (gentili con le gomme)
    """
    deg_file = PROCESSED_DATA_DIR / "tyre_degradation.csv"

    if not deg_file.exists():
        print("  ⚠️ Dati degradazione non disponibili. Esegui: python -m src.advanced_data_loader")
        return df

    deg = pd.read_csv(deg_file)

    df = df.sort_values(["year", "round"]).copy()

    # Per ogni pilota: media della degradazione nelle ultime N gare
    df["avg_tyre_deg"] = np.nan
    df["avg_consistency"] = np.nan
    df["avg_stint_length"] = np.nan

    # Per ogni team: media della degradazione (indica bontà della macchina)
    df["team_avg_tyre_deg"] = np.nan

    for idx, row in df.iterrows():
        year = row["year"]
        rnd = row["round"]

        # Degradazione passata del PILOTA
        past_driver_deg = deg[
            (deg["driver"] == row["driver"])
            & ((deg["year"] < year)
               | ((deg["year"] == year) & (deg["round"] < rnd)))
        ].tail(ROLLING_WINDOW * 3)  # Ultimi N gare * ~3 stint per gara

        if len(past_driver_deg) >= 3:
            df.at[idx, "avg_tyre_deg"] = past_driver_deg["deg_slope"].mean()
            df.at[idx, "avg_consistency"] = past_driver_deg["consistency"].mean()
            df.at[idx, "avg_stint_length"] = past_driver_deg["stint_length"].mean()

        # Degradazione passata del TEAM (filtrato per era regolamentare)
        era_start = get_regulation_era_start(year)
        past_team_deg = deg[
            (deg["driver"].isin(
                df[df["team"] == row["team"]]["driver"].unique()
            ))
            & (deg["year"] >= era_start)
            & ((deg["year"] < year)
               | ((deg["year"] == year) & (deg["round"] < rnd)))
        ].tail(ROLLING_WINDOW * 6)  # 2 piloti * 3 stint * N gare

        if len(past_team_deg) >= 5:
            df.at[idx, "team_avg_tyre_deg"] = past_team_deg["deg_slope"].mean()

    return df


# ============================================================
# 3. FEATURE STRATEGIA PIT STOP
# ============================================================

def add_strategy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature sulla strategia di gara.

    PERCHÉ CONTA?
    - Alcuni team fanno costantemente pit stop più veloci
    - Il numero di soste influenza il risultato
    - La scelta di compound (aggressiva vs conservativa) è una feature
    - Errori strategici (pit stop al momento sbagliato) si vedono nei dati
    """
    pits_file = PROCESSED_DATA_DIR / "pit_stops.csv"

    if not pits_file.exists():
        print("  ⚠️ Dati pit stop non disponibili. Esegui: python -m src.advanced_data_loader")
        return df

    pits = pd.read_csv(pits_file)

    df = df.sort_values(["year", "round"]).copy()

    # Merge: num pit stop di questa gara (lo usiamo come feature solo per analisi post-gara)
    df = df.merge(
        pits[["year", "round", "driver", "num_pit_stops", "num_compounds"]],
        on=["year", "round", "driver"],
        how="left",
    )

    # Media storica pit stop del team (indica qualità del team operativamente)
    df["team_avg_pit_stops"] = np.nan

    for idx, row in df.iterrows():
        year = row["year"]
        rnd = row["round"]

        # Filtrato per era regolamentare (team history)
        era_start = get_regulation_era_start(year)
        past_pits = pits[
            (pits["driver"].isin(
                df[df["team"] == row["team"]]["driver"].unique()
            ))
            & (pits["year"] >= era_start)
            & ((pits["year"] < year)
               | ((pits["year"] == year) & (pits["round"] < rnd)))
        ]

        if len(past_pits) >= 5:
            df.at[idx, "team_avg_pit_stops"] = past_pits["num_pit_stops"].mean()

    return df


# ============================================================
# 4. FEATURE CIRCUITO
# ============================================================

def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge le caratteristiche fisiche del circuito e calcola
    il "fit" tra team/pilota e tipo di circuito.

    PERCHÉ CONTA?
    - Red Bull potrebbe andare forte sui circuiti con basso carico
    - Ferrari potrebbe eccellere ad alto carico
    - Alcuni piloti sono specialisti dei circuiti cittadini
    """
    # Step 1: Aggiungiamo le caratteristiche fisiche del circuito
    df = enrich_with_circuit_data(df)

    df = df.sort_values(["year", "round"]).copy()

    # Step 2: Performance storica del team per TIPO di circuito
    # Filtrato per era regolamentare: in 2026 usa solo dati 2026+
    for downforce_level, downforce_val in DOWNFORCE_MAP.items():
        col_name = f"team_perf_df_{downforce_level}"
        df[col_name] = np.nan

        for idx, row in df.iterrows():
            past = _filter_regulation_era(df, row["year"], row["round"], "team", row["team"])
            past = past[past["circuit_downforce"] == downforce_val]
            if len(past) >= 3:
                df.at[idx, col_name] = past["finish_position"].mean()

    for circuit_type, type_val in CIRCUIT_TYPE_MAP.items():
        col_name = f"team_perf_type_{circuit_type}"
        df[col_name] = np.nan

        for idx, row in df.iterrows():
            past = _filter_regulation_era(df, row["year"], row["round"], "team", row["team"])
            past = past[past["circuit_type"] == type_val]
            if len(past) >= 3:
                df.at[idx, col_name] = past["finish_position"].mean()

    # Step 3: Performance del PILOTA per tipo di circuito
    for downforce_level, downforce_val in DOWNFORCE_MAP.items():
        col_name = f"driver_perf_df_{downforce_level}"
        df[col_name] = np.nan

        for idx, row in df.iterrows():
            past = df[
                (df["driver"] == row["driver"])
                & (df["circuit_downforce"] == downforce_val)
                & ((df["year"] < row["year"])
                   | ((df["year"] == row["year"]) & (df["round"] < row["round"])))
            ]
            if len(past) >= 2:
                df.at[idx, col_name] = past["finish_position"].mean()

    # Step 4: "Power team fit"
    # Se un team va bene dove il motore conta (high power_sensitivity)
    # e questa gara ha high power_sensitivity, è un buon segno
    df["team_power_fit"] = np.nan

    for idx, row in df.iterrows():
        # Performance del team su circuiti con simile power_sensitivity
        # Filtrato per era regolamentare
        ps = row.get("power_sensitivity", 0.5)
        past = _filter_regulation_era(df, row["year"], row["round"], "team", row["team"])
        past = past[past["power_sensitivity"].between(ps - 0.2, ps + 0.2)]
        if len(past) >= 3:
            df.at[idx, "team_power_fit"] = past["finish_position"].mean()

    return df


# ============================================================
# 5. FEATURE PROVE LIBERE (FP1, FP2, FP3)
# ============================================================

def add_fp_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature dalle sessioni di prove libere.

    PERCHÉ CONTA?
    - Le FP sono il primo dato REALE del weekend
    - FP2 long run = simulazione gara (passo, degradazione, strategia)
    - Il gap dal leader in FP indica il potenziale reale
    - La degradazione in FP anticipa problemi in gara
    - Un pilota veloce in FP ma lento in gara = problemi di setup/strategia

    NOTA: Non c'è data leakage — le FP avvengono PRIMA della gara.
    """
    summary_file = PROCESSED_DATA_DIR / "fp_summary.csv"

    if not summary_file.exists():
        print("  ⚠️ Dati FP non disponibili. Esegui: python -m src.fp_data_loader")
        return df

    fp_summary = pd.read_csv(summary_file)

    # === MERGE DIRETTO: dati FP di QUESTO weekend ===
    # Queste feature catturano la forma specifica del weekend
    fp_cols_to_merge = [
        "year", "round", "driver",
        "fp_best_lap_delta",      # Gap dal leader nelle FP
        "fp_best_lap_pct",        # Percentuale dal leader
        "fp2_delta",              # Gap specifico FP2 (la più importante)
        "fp3_delta",              # Gap specifico FP3 (pre-qualifica!)
        "fp2_long_run_pace",      # Passo long run FP2
        "fp2_long_run_deg",       # Degradazione long run FP2
        "fp2_long_run_consistency",  # Costanza long run FP2
        "fp_long_run_pace",       # Miglior long run pace tra le FP
        "fp_long_run_deg",        # Miglior degradazione long run
        "fp_long_run_consistency", # Miglior costanza long run
        "fp_total_laps",          # Giri totali (indica preparazione)
    ]

    available_cols = [c for c in fp_cols_to_merge if c in fp_summary.columns]
    df = df.merge(
        fp_summary[available_cols],
        on=["year", "round", "driver"],
        how="left",
    )

    # === FEATURE TEAM FP: forza della macchina dal weekend ===
    # Queste feature aggregano i dati FP dei piloti dello stesso team
    # per stimare la competitività della VETTURA (non solo del pilota).
    # Fondamentale a inizio era regolamentare quando l'Elo team è azzerato.
    df["fp_team_best_delta"] = np.nan
    df["fp_team_median_delta"] = np.nan
    df["fp_team_long_run_rank"] = np.nan

    for (year, rnd), race_group in df.groupby(["year", "round"]):
        # Per ogni team in questa gara, calcola la forza FP
        for team, team_group in race_group.groupby("team"):
            team_deltas = team_group["fp_best_lap_delta"].dropna()
            team_indices = team_group.index

            if len(team_deltas) >= 1:
                # Miglior delta FP del team (potenziale massimo macchina)
                df.loc[team_indices, "fp_team_best_delta"] = team_deltas.min()
                # Mediana delta FP del team (forza macchina consistente)
                df.loc[team_indices, "fp_team_median_delta"] = team_deltas.median()

        # Ranking team per long run pace (media dei piloti del team)
        team_lr = race_group.groupby("team")["fp_long_run_pace"].mean().dropna()
        if len(team_lr) >= 2:
            # Rank: 1 = miglior passo, N = peggior passo
            team_lr_rank = team_lr.rank(method="min")
            for team, rank in team_lr_rank.items():
                mask = (df["year"] == year) & (df["round"] == rnd) & (df["team"] == team)
                df.loc[mask, "fp_team_long_run_rank"] = rank

    # === FEATURE DERIVATE: confronto FP con storico ===
    # "Questo weekend sei più veloce o più lento del tuo solito?"
    df = df.sort_values(["year", "round"]).copy()

    # Media storica del delta FP del pilota (rolling)
    df["fp_delta_vs_history"] = np.nan
    df["fp_deg_vs_history"] = np.nan

    for idx, row in df.iterrows():
        year = row["year"]
        rnd = row["round"]
        driver = row["driver"]

        # Delta FP storico del pilota
        past_fp = df[
            (df["driver"] == driver)
            & ((df["year"] < year)
               | ((df["year"] == year) & (df["round"] < rnd)))
        ]

        if len(past_fp) >= 3:
            past_delta = past_fp["fp_best_lap_delta"].dropna()
            if len(past_delta) >= 3:
                avg_delta = past_delta.tail(ROLLING_WINDOW).mean()
                current_delta = row.get("fp_best_lap_delta")
                if pd.notna(current_delta) and pd.notna(avg_delta):
                    # Positivo = meglio del solito, Negativo = peggio del solito
                    df.at[idx, "fp_delta_vs_history"] = avg_delta - current_delta

            past_deg = past_fp["fp_long_run_deg"].dropna()
            if len(past_deg) >= 3:
                avg_deg = past_deg.tail(ROLLING_WINDOW).mean()
                current_deg = row.get("fp_long_run_deg")
                if pd.notna(current_deg) and pd.notna(avg_deg):
                    # Positivo = degrada meno del solito
                    df.at[idx, "fp_deg_vs_history"] = avg_deg - current_deg

    return df


# ============================================================
# 6. FEATURE MOMENTUM STAGIONALE
# ============================================================

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature sul momentum stagionale.

    PERCHÉ CONTA?
    - I team migliorano/peggiorano durante la stagione (upgrade, affidabilità)
    - Un pilota in crescita ha più probabilità di fare bene
    - La velocità di cambiamento dell'Elo cattura trend a medio termine
    - I punti recenti indicano la competitività attuale

    FEATURE:
    - season_points_pct: percentuale punti rispetto al massimo in stagione
    - elo_velocity: variazione Elo nelle ultime N gare (trend)
    - recent_form_weighted: media pesata delle ultime posizioni (recenti pesano di più)
    - position_trend: slope delle posizioni nelle ultime N gare (migliora o peggiora?)
    - team_elo_velocity: variazione Elo del team (upgrade/downgrade)
    """
    df = df.sort_values(["year", "round"]).copy()

    df["elo_velocity"] = np.nan
    df["team_elo_velocity"] = np.nan
    df["recent_form_weighted"] = np.nan
    df["position_trend"] = np.nan
    df["season_points_pct"] = np.nan

    for idx, row in df.iterrows():
        year = row["year"]
        rnd = row["round"]
        driver = row["driver"]
        team = row["team"]

        # Solo gare della stessa stagione, prima di questa
        past_season = df[
            (df["year"] == year)
            & (df["round"] < rnd)
        ]

        past_driver = past_season[past_season["driver"] == driver]

        if len(past_driver) >= 3:
            # === ELO VELOCITY ===
            # Differenza Elo tra ora e N gare fa
            if "elo_pre_race" in past_driver.columns:
                elo_values = past_driver["elo_pre_race"].dropna()
                if len(elo_values) >= 3:
                    recent = elo_values.tail(ROLLING_WINDOW)
                    # Slope dell'Elo: positivo = in crescita
                    x = np.arange(len(recent))
                    slope, _ = np.polyfit(x, recent.values, 1)
                    df.at[idx, "elo_velocity"] = slope

            # === RECENT FORM WEIGHTED ===
            # Media pesata: gare più recenti contano di più
            positions = past_driver["finish_position"].dropna().tail(ROLLING_WINDOW)
            if len(positions) >= 3:
                weights = np.arange(1, len(positions) + 1, dtype=float)
                weights = weights / weights.sum()
                df.at[idx, "recent_form_weighted"] = np.average(positions.values, weights=weights)

            # === POSITION TREND ===
            # Slope delle posizioni: negativo = sta migliorando
            positions_all = past_driver["finish_position"].dropna().tail(ROLLING_WINDOW)
            if len(positions_all) >= 3:
                x = np.arange(len(positions_all))
                slope, _ = np.polyfit(x, positions_all.values, 1)
                df.at[idx, "position_trend"] = slope

        # === TEAM ELO VELOCITY ===
        past_team = past_season[past_season["team"] == team]
        if len(past_team) >= 3 and "team_elo_pre_race" in past_team.columns:
            team_elo_values = past_team.groupby("round")["team_elo_pre_race"].first()
            if len(team_elo_values) >= 3:
                recent = team_elo_values.tail(ROLLING_WINDOW)
                x = np.arange(len(recent))
                slope, _ = np.polyfit(x, recent.values, 1)
                df.at[idx, "team_elo_velocity"] = slope

        # === SEASON POINTS PCT ===
        # Che percentuale dei punti massimi possibili ha questo pilota?
        if len(past_driver) >= 1 and "points" in past_driver.columns:
            total_points = past_driver["points"].sum()
            # Max punti possibili = 25 (vittoria) * gare disputate
            max_possible = len(past_driver) * 25
            if max_possible > 0:
                df.at[idx, "season_points_pct"] = total_points / max_possible

    return df


# ============================================================
# 7. BUILD COMPLETO
# ============================================================

def build_advanced_features(
    elo_history: pd.DataFrame,
    team_elo_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Costruisce la feature matrix completa con TUTTE le feature avanzate.

    Questa funzione:
    1. Parte dalla feature matrix base (Elo, griglia, forma recente)
    2. Aggiunge feature meteo
    3. Aggiunge feature gomme
    4. Aggiunge feature strategia
    5. Aggiunge feature circuito
    6. Aggiunge feature prove libere (FP1/FP2/FP3)
    7. Aggiunge feature momentum stagionale
    """
    # Importiamo le feature base
    from src.feature_engineering import (
        add_recent_form,
        add_circuit_history,
        add_grid_features,
    )

    print("🔧 Costruzione FEATURE MATRIX AVANZATA...")
    print("=" * 50)

    # Step 0: Base (Elo + merge team Elo)
    df = elo_history.copy()
    df = df.merge(
        team_elo_history[["year", "round", "team", "team_elo_pre_race"]],
        on=["year", "round", "team"],
        how="left",
    )

    # Step 1: Feature base
    print("  📊 Feature base (forma recente, griglia)...")
    df = add_recent_form(df)
    df = add_circuit_history(df)
    df = add_grid_features(df)

    # Step 2: Meteo
    print("  🌤️ Feature meteo...")
    df = add_weather_features(df)

    # Step 3: Gomme
    print("  🔴 Feature gomme e degradazione...")
    df = add_tyre_features(df)

    # Step 4: Strategia
    print("  🔧 Feature strategia pit stop...")
    df = add_strategy_features(df)

    # Step 5: Circuito
    print("  🏁 Feature circuito e team-circuit fit...")
    df = add_circuit_features(df)

    # Step 6: Prove Libere
    print("  🏎️ Feature prove libere (FP1/FP2/FP3)...")
    df = add_fp_features(df)

    # Step 7: Momentum stagionale
    print("  📈 Feature momentum stagionale...")
    df = add_momentum_features(df)

    # Step 8: Riempimento NaN con mediane
    # NOTA: NON riempiamo finish_position — serve NaN per gare future
    # (non ancora avvenute) per poter fare previsioni pre-gara
    print("  🧹 Pulizia valori mancanti...")
    cols_to_preserve_nan = {"finish_position", "classified_position", "points"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in cols_to_preserve_nan:
            continue
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"\n  ✅ Feature matrix avanzata: {df.shape[0]} righe × {df.shape[1]} colonne")

    # Elenco feature disponibili
    feature_cols = [col for col in df.columns if col not in [
        "year", "round", "race_name", "driver", "team",
        "finish_position", "dnf", "elo_post_race",
        "classified_position", "status", "points",
        "driver_full_name", "quali_position",
    ]]

    print(f"  📋 Feature disponibili ({len(feature_cols)}):")
    for col in sorted(feature_cols):
        non_null = df[col].notna().sum()
        print(f"     {col:<35s} ({non_null}/{len(df)} valori)")

    return df


# Lista aggiornata delle feature per il modello
ADVANCED_FEATURE_COLUMNS = [
    # === Base ===
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
    # === Meteo ===
    "air_temp",
    "track_temp",
    "humidity",
    "is_wet",
    "temp_category",
    "team_perf_temp_0",         # Performance team col freddo
    "team_perf_temp_1",         # Performance team temperatura media
    "team_perf_temp_2",         # Performance team col caldo
    "team_wet_performance",
    "driver_wet_performance",
    # === Gomme ===
    "avg_tyre_deg",             # Degradazione media pilota
    "avg_consistency",          # Costanza del pilota
    "avg_stint_length",         # Lunghezza media stint
    "team_avg_tyre_deg",        # Degradazione media del team
    # === Strategia ===
    "team_avg_pit_stops",       # Media pit stop del team
    # === Circuito ===
    "circuit_length_km",
    "circuit_turns",
    "circuit_downforce",
    "circuit_type",
    "power_sensitivity",
    "overtaking_difficulty",
    "tyre_stress",
    "altitude_m",
    # === Team-Circuit Fit ===
    "team_perf_df_low",         # Team su circuiti low downforce
    "team_perf_df_medium",      # Team su circuiti medium downforce
    "team_perf_df_high",        # Team su circuiti high downforce
    "team_perf_type_permanent", # Team su circuiti permanenti
    "team_perf_type_hybrid",    # Team su circuiti ibridi
    "team_perf_type_street",    # Team su circuiti cittadini
    "driver_perf_df_low",       # Pilota su circuiti low downforce
    "driver_perf_df_medium",    # Pilota su circuiti medium downforce
    "driver_perf_df_high",      # Pilota su circuiti high downforce
    "team_power_fit",           # Fit team-power sensitivity
    # === Prove Libere ===
    "fp_best_lap_delta",        # Gap dal leader nelle FP (secondi)
    "fp_best_lap_pct",          # Percentuale dal leader FP
    "fp2_delta",                # Gap specifico FP2
    "fp3_delta",                # Gap specifico FP3 (pre-qualifica!)
    "fp2_long_run_pace",        # Passo long run FP2 (secondi)
    "fp2_long_run_deg",         # Degradazione long run FP2 (sec/giro)
    "fp2_long_run_consistency", # Costanza long run FP2 (std)
    "fp_long_run_pace",         # Miglior long run pace tra le FP
    "fp_long_run_deg",          # Miglior degradazione long run
    "fp_long_run_consistency",  # Miglior costanza long run
    "fp_total_laps",            # Giri totali nelle FP
    "fp_team_best_delta",       # Miglior delta FP del team (forza macchina)
    "fp_team_median_delta",     # Mediana delta FP del team (consistenza macchina)
    "fp_team_long_run_rank",    # Ranking team per long run pace
    "fp_delta_vs_history",      # Delta FP vs media storica pilota
    "fp_deg_vs_history",        # Degradazione FP vs media storica
    # === Momentum Stagionale ===
    "elo_velocity",             # Trend Elo nelle ultime gare (slope)
    "team_elo_velocity",        # Trend Elo team (upgrade/downgrade)
    "recent_form_weighted",     # Media posizioni pesata (recenti contano di più)
    "position_trend",           # Slope posizioni (negativo = migliora)
    "season_points_pct",        # % punti rispetto al massimo possibile
]


if __name__ == "__main__":
    from src.elo import EloSystem, compute_team_elo
    from src.data_loader import load_all_data

    # 1. Carica dati base
    print("📂 Caricamento dati...")
    df = load_all_data()

    # 2. Calcola Elo
    print("🎯 Calcolo Elo...")
    elo = EloSystem()
    elo_history = elo.process_season(df)
    team_elo = compute_team_elo(df)

    # 3. Costruisci feature avanzate
    features = build_advanced_features(elo_history, team_elo)

    # 4. Salva
    output_file = PROCESSED_DATA_DIR / "advanced_features.csv"
    features.to_csv(output_file, index=False)
    print(f"\n💾 Salvato in {output_file}")

    # 5. Mostra correlazioni con la posizione finale
    print("\n📈 Top 20 correlazioni con finish_position:")
    available = [c for c in ADVANCED_FEATURE_COLUMNS if c in features.columns]
    corr = features[available + ["finish_position"]].corr()["finish_position"]
    corr = corr.drop("finish_position").sort_values()
    for feat, val in corr.head(10).items():
        print(f"  {feat:<35s} {val:+.3f}  {'(più basso = meglio)' if val < 0 else ''}")
    print("  ...")
    for feat, val in corr.tail(10).items():
        print(f"  {feat:<35s} {val:+.3f}")