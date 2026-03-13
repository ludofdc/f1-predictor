"""
fp_data_loader.py — Scaricamento Dati Prove Libere (FP1, FP2, FP3)
===================================================================
Questo modulo scarica i dati delle sessioni di prove libere:

- FP1: Primo contatto con la pista, setup iniziale
- FP2: Simulazione gara (long run) e qualifica (short run)
- FP3: Ultimo setup prima della qualifica

PERCHÉ LE FP SONO IMPORTANTI PER LE PREVISIONI?
- FP2 è la sessione più rappresentativa del passo gara
- I long run (stint di 10+ giri) simulano le condizioni di gara
- Il gap dai migliori in FP indica il potenziale reale del weekend
- La degradazione in FP anticipa i problemi di gestione gomme

I dati estratti:
1. Best lap per pilota per sessione
2. Long run pace (media tempi su stint lunghi)
3. Long run degradation (pendenza tempi su stint lunghi)
4. Consistency (deviazione standard sui long run)
5. Pace per compound (SOFT, MEDIUM, HARD)
"""

import pandas as pd
import numpy as np
import fastf1
import warnings
import time
import sys

sys.path.append("..")
from config import SEASONS, RAW_DATA_DIR, PROCESSED_DATA_DIR

warnings.filterwarnings("ignore", category=FutureWarning)


def setup_cache():
    """Configura la cache di fastf1."""
    cache_dir = RAW_DATA_DIR / "fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def _extract_session_data(year: int, round_num: int, session_name: str) -> pd.DataFrame:
    """
    Estrae i dati chiave da una singola sessione FP.

    Per ogni pilota calcola:
    - best_lap: miglior tempo in secondi
    - median_lap: tempo mediano (rimuovendo outlier)
    - long_run_pace: media tempi su stint di 10+ giri (esclusi out/in lap)
    - long_run_deg: degradazione (slope) su stint lunghi
    - long_run_consistency: STD dei tempi su stint lunghi
    - pace per compound (soft_pace, medium_pace, hard_pace)

    Parametri:
        session_name: "FP1", "FP2", o "FP3"
    """
    try:
        session = fastf1.get_session(year, round_num, session_name)
        session.load()
        laps = session.laps

        if laps is None or laps.empty:
            return pd.DataFrame()

        records = []

        for driver in laps["Driver"].unique():
            driver_laps = laps[laps["Driver"] == driver].copy()

            # Converti tempi in secondi
            if hasattr(driver_laps["LapTime"], "dt"):
                driver_laps["lap_seconds"] = driver_laps["LapTime"].dt.total_seconds()
            else:
                continue

            # Rimuovi giri invalidi (pit in/out, installazione, ecc.)
            valid = driver_laps["lap_seconds"].dropna()
            if len(valid) < 3:
                continue

            # Rimuovi outlier con IQR (come in advanced_data_loader)
            median_time = valid.median()
            iqr = valid.quantile(0.75) - valid.quantile(0.25)
            if iqr > 0:
                clean = valid[(valid > median_time - 2 * iqr) & (valid < median_time + 2 * iqr)]
            else:
                clean = valid

            if len(clean) < 3:
                continue

            record = {
                "year": year,
                "round": round_num,
                "session": session_name,
                "driver": driver,
                "best_lap": clean.min(),
                "median_lap": clean.median(),
                "total_laps": len(valid),
            }

            # === LONG RUN ANALYSIS ===
            # Un long run è uno stint di 8+ giri consecutivi sullo stesso compound
            # È la cosa più vicina a una simulazione di gara
            for stint_num, stint_laps in driver_laps.groupby("Stint"):
                stint_times = stint_laps["lap_seconds"].dropna()
                compound = stint_laps["Compound"].dropna()

                if len(stint_times) < 8 or compound.empty:
                    continue

                # Pulizia stint (rimuovi primo e ultimo giro = out lap / in lap)
                if len(stint_times) > 2:
                    stint_clean = stint_times.iloc[1:-1]
                else:
                    stint_clean = stint_times

                # Rimuovi outlier dallo stint
                stint_median = stint_clean.median()
                stint_iqr = stint_clean.quantile(0.75) - stint_clean.quantile(0.25)
                if stint_iqr > 0:
                    stint_clean = stint_clean[
                        (stint_clean > stint_median - 2 * stint_iqr)
                        & (stint_clean < stint_median + 2 * stint_iqr)
                    ]

                if len(stint_clean) >= 5:
                    # Degradazione: slope dei tempi
                    x = np.arange(len(stint_clean))
                    slope, _ = np.polyfit(x, stint_clean.values, 1)

                    record["long_run_pace"] = stint_clean.mean()
                    record["long_run_deg"] = slope
                    record["long_run_consistency"] = stint_clean.std()
                    record["long_run_compound"] = compound.iloc[0]
                    record["long_run_laps"] = len(stint_clean)
                    break  # Prendi il primo long run significativo

            # === PACE PER COMPOUND ===
            for comp in ["SOFT", "MEDIUM", "HARD"]:
                comp_laps = driver_laps[driver_laps["Compound"] == comp]["lap_seconds"].dropna()
                if len(comp_laps) >= 3:
                    # Rimuovi outlier
                    comp_median = comp_laps.median()
                    comp_iqr = comp_laps.quantile(0.75) - comp_laps.quantile(0.25)
                    if comp_iqr > 0:
                        comp_clean = comp_laps[
                            (comp_laps > comp_median - 2 * comp_iqr)
                            & (comp_laps < comp_median + 2 * comp_iqr)
                        ]
                    else:
                        comp_clean = comp_laps

                    if len(comp_clean) >= 2:
                        record[f"{comp.lower()}_pace"] = comp_clean.median()

            records.append(record)

        return pd.DataFrame(records)

    except Exception as e:
        print(f"    ⚠️ Errore {session_name} {year} R{round_num}: {e}")
        return pd.DataFrame()


def _compute_relative_metrics(fp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola metriche RELATIVE per ogni sessione.

    Perché relative?
    - Un tempo di 1:20.5 non significa nulla in assoluto
    - Ma essere 0.3s più veloce del leader in FP2 è molto informativo
    - Le metriche relative sono comparabili tra circuiti diversi

    Calcola:
    - best_lap_delta: gap dal miglior tempo della sessione
    - long_run_delta: gap dal miglior long run pace della sessione
    - best_lap_pct: percentuale rispetto al leader (1.0 = leader)
    """
    if fp_data.empty:
        return fp_data

    result = fp_data.copy()

    for (year, rnd, session), group in result.groupby(["year", "round", "session"]):
        # Delta dal miglior tempo
        if "best_lap" in group.columns:
            best = group["best_lap"].min()
            if pd.notna(best) and best > 0:
                mask = (result["year"] == year) & (result["round"] == rnd) & (result["session"] == session)
                result.loc[mask, "best_lap_delta"] = group["best_lap"] - best
                result.loc[mask, "best_lap_pct"] = group["best_lap"] / best

        # Delta dal miglior long run
        if "long_run_pace" in group.columns:
            lr_valid = group["long_run_pace"].dropna()
            if len(lr_valid) > 0:
                best_lr = lr_valid.min()
                if best_lr > 0:
                    mask = (result["year"] == year) & (result["round"] == rnd) & (result["session"] == session)
                    result.loc[mask, "long_run_delta"] = group["long_run_pace"] - best_lr

    return result


def _build_race_calendar() -> pd.DataFrame:
    """
    Costruisce la lista di tutte le gare per cui scaricare dati FP.

    Usa due fonti:
    1. all_races.csv — gare già corse (risultati disponibili)
    2. Calendario fastf1 — per trovare gare future con FP già svolte
       (es. il weekend è iniziato ma la gara non è ancora avvenuta)

    In questo modo scarichiamo le FP anche di gare non ancora corse.
    """
    from datetime import datetime

    race_entries = []

    # Fonte 1: gare già registrate in all_races.csv
    races_file = PROCESSED_DATA_DIR / "all_races.csv"
    if races_file.exists():
        races_df = pd.read_csv(races_file)
        existing = (
            races_df.groupby(["year", "round"])["race_name"]
            .first()
            .reset_index()
        )
        for _, row in existing.iterrows():
            race_entries.append((int(row["year"]), int(row["round"]), row["race_name"]))

    existing_keys = {(y, r) for y, r, _ in race_entries}

    # Fonte 2: calendario fastf1 per le stagioni correnti
    # Questo cattura gare dove le FP sono avvenute ma la gara no
    today = datetime.now()
    for year in SEASONS:
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule["RoundNumber"] > 0]

            for _, race_info in races.iterrows():
                rnd = int(race_info["RoundNumber"])
                name = race_info["EventName"]

                if (year, rnd) in existing_keys:
                    continue  # Già nel dataset

                # Includi solo se la data dell'evento è passata (o oggi)
                event_date = pd.to_datetime(race_info.get("EventDate"))
                if event_date is not None and event_date.date() <= today.date():
                    race_entries.append((year, rnd, name))

        except Exception:
            continue

    # Converti in DataFrame e ordina
    result = pd.DataFrame(race_entries, columns=["year", "round", "race_name"])
    result = result.drop_duplicates(subset=["year", "round"]).sort_values(["year", "round"])
    return result.reset_index(drop=True)


def download_fp_data(force: bool = False):
    """
    Funzione principale: scarica i dati FP1, FP2, FP3 per tutte le gare.

    Salva i file:
    - fp_data.csv: tutti i dati FP per sessione/pilota/gara
    - fp_summary.csv: riepilogo aggregato per pilota/gara (media delle 3 sessioni)
    """
    setup_cache()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fp_file = PROCESSED_DATA_DIR / "fp_data.csv"
    summary_file = PROCESSED_DATA_DIR / "fp_summary.csv"

    if not force and fp_file.exists():
        print("📂 Dati FP già presenti. Usa force=True per riscaricare.")
        return

    print("🔄 Download dati Prove Libere (FP1, FP2, FP3)...")
    print("   ⏱️ Questo richiede parecchio tempo — 3 sessioni per gara.\n")

    # Carichiamo il calendario
    # Usiamo due fonti: all_races.csv (gare già avvenute) + calendario fastf1 (gare future)
    race_list = _build_race_calendar()

    all_fp = []

    for _, race in race_list.iterrows():
        year = int(race["year"])
        rnd = int(race["round"])
        name = race["race_name"]

        print(f"  🏁 {year} R{rnd:02d}: {name}")

        for fp_session in ["FP1", "FP2", "FP3"]:
            print(f"    {fp_session}...", end=" ", flush=True)
            try:
                data = _extract_session_data(year, rnd, fp_session)
                if not data.empty:
                    all_fp.append(data)
                    print(f"✅ ({len(data)} piloti)")
                else:
                    print("⚠️ nessun dato")
            except Exception as e:
                print(f"❌ {e}")

            time.sleep(0.3)

    if not all_fp:
        print("❌ Nessun dato FP scaricato!")
        return

    # Concatena e calcola metriche relative
    fp_full = pd.concat(all_fp, ignore_index=True)
    fp_full = _compute_relative_metrics(fp_full)

    # Salva dati dettagliati
    fp_full.to_csv(fp_file, index=False)
    print(f"\n💾 Dati FP dettagliati: {len(fp_full)} righe → {fp_file}")

    # === CREA SUMMARY AGGREGATO ===
    # Per ogni gara + pilota: combina FP1, FP2, FP3 in metriche uniche
    summary = _create_fp_summary(fp_full)
    summary.to_csv(summary_file, index=False)
    print(f"💾 Summary FP: {len(summary)} righe → {summary_file}")

    print("\n✅ Download dati FP completato!")


def _create_fp_summary(fp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Crea un riepilogo per gara+pilota combinando le 3 sessioni FP.

    La logica:
    - best_lap: il miglior tempo tra FP1/FP2/FP3 (di solito FP3 o FP2)
    - fp2_best_lap: specifico FP2 (la sessione più rappresentativa)
    - long_run_pace: preferisce FP2, poi FP1 (dove si fanno i long run)
    - pace per compound: migliore tra le sessioni

    FP2 è la sessione chiave perché:
    - Si svolgono simulazioni di gara complete
    - Le condizioni (orario, temperatura) sono simili alla gara
    - I team testano la strategia
    """
    records = []

    for (year, rnd), race_group in fp_data.groupby(["year", "round"]):
        for driver, driver_group in race_group.groupby("driver"):
            record = {
                "year": year,
                "round": rnd,
                "driver": driver,
            }

            # Best lap assoluto (tra tutte le FP)
            if "best_lap" in driver_group.columns:
                record["fp_best_lap"] = driver_group["best_lap"].min()

            # Delta dal leader (media delle sessioni)
            if "best_lap_delta" in driver_group.columns:
                record["fp_best_lap_delta"] = driver_group["best_lap_delta"].min()

            # Percentuale dal leader
            if "best_lap_pct" in driver_group.columns:
                record["fp_best_lap_pct"] = driver_group["best_lap_pct"].min()

            # === FP2 specifico (la sessione più importante per passo gara) ===
            fp2 = driver_group[driver_group["session"] == "FP2"]
            if not fp2.empty:
                fp2_row = fp2.iloc[0]
                if pd.notna(fp2_row.get("best_lap")):
                    record["fp2_best_lap"] = fp2_row["best_lap"]
                if pd.notna(fp2_row.get("best_lap_delta")):
                    record["fp2_delta"] = fp2_row["best_lap_delta"]
                if pd.notna(fp2_row.get("long_run_pace")):
                    record["fp2_long_run_pace"] = fp2_row["long_run_pace"]
                if pd.notna(fp2_row.get("long_run_deg")):
                    record["fp2_long_run_deg"] = fp2_row["long_run_deg"]
                if pd.notna(fp2_row.get("long_run_consistency")):
                    record["fp2_long_run_consistency"] = fp2_row["long_run_consistency"]

            # === FP3 specifico (ultima sessione prima della qualifica) ===
            # Il delta FP3 è il miglior indicatore del passo qualifica
            # perché i team fanno i "qualifying simulation run" in FP3
            fp3 = driver_group[driver_group["session"] == "FP3"]
            if not fp3.empty:
                fp3_row = fp3.iloc[0]
                if pd.notna(fp3_row.get("best_lap_delta")):
                    record["fp3_delta"] = fp3_row["best_lap_delta"]

            # === Long run: preferisci FP2, poi FP1, poi FP3 ===
            for pref_session in ["FP2", "FP1", "FP3"]:
                sess = driver_group[driver_group["session"] == pref_session]
                if not sess.empty and pd.notna(sess.iloc[0].get("long_run_pace")):
                    row = sess.iloc[0]
                    record["fp_long_run_pace"] = row["long_run_pace"]
                    record["fp_long_run_deg"] = row.get("long_run_deg")
                    record["fp_long_run_consistency"] = row.get("long_run_consistency")
                    record["fp_long_run_laps"] = row.get("long_run_laps")
                    if pd.notna(row.get("long_run_delta")):
                        record["fp_long_run_delta"] = row["long_run_delta"]
                    break

            # === Pace per compound (miglior tempo tra le sessioni) ===
            for comp in ["soft", "medium", "hard"]:
                col = f"{comp}_pace"
                if col in driver_group.columns:
                    comp_times = driver_group[col].dropna()
                    if len(comp_times) > 0:
                        record[f"fp_{comp}_pace"] = comp_times.min()

            # Numero totale di giri nelle FP
            if "total_laps" in driver_group.columns:
                record["fp_total_laps"] = driver_group["total_laps"].sum()

            records.append(record)

    return pd.DataFrame(records)


if __name__ == "__main__":
    download_fp_data(force=True)
