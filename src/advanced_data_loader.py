"""
advanced_data_loader.py — Scaricamento Dati Avanzati
=====================================================
Questo modulo scarica dati DETTAGLIATI per ogni gara:
- Lap times (tempi giro per giro)
- Compound e stint (gestione gomme)
- Pit stops
- Meteo (temperatura, pioggia, umidità)

Questi dati sono MOLTO più pesanti dei semplici risultati.
Per questo li teniamo separati dal data_loader base.

NOTA: Il download richiede parecchio tempo la prima volta
(fastf1 scarica molti MB per gara). Usa la cache.
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


def get_race_laps(year: int, round_num: int) -> pd.DataFrame:
    """
    Scarica i dati giro per giro di una gara.

    Per ogni giro di ogni pilota abbiamo:
    - Tempo del giro
    - Tipo di gomma (SOFT/MEDIUM/HARD/INTERMEDIATE/WET)
    - Vita della gomma (quanti giri ci ha fatto)
    - Numero dello stint (1° stint, 2° stint, ecc.)
    - Velocità nei vari punti del circuito
    """
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load()
        laps = session.laps

        if laps is None or laps.empty:
            return pd.DataFrame()

        lap_data = pd.DataFrame({
            "year": year,
            "round": round_num,
            "driver": laps["Driver"],
            "lap_number": laps["LapNumber"],
            # .dt.total_seconds() converte timedelta in secondi
            "lap_time_seconds": laps["LapTime"].dt.total_seconds()
                if hasattr(laps["LapTime"], "dt") else None,
            "compound": laps["Compound"],       # SOFT, MEDIUM, HARD, etc.
            "tyre_life": laps["TyreLife"],       # Giri fatti con questa gomma
            "stint": laps["Stint"],              # Numero dello stint
            "fresh_tyre": laps["FreshTyre"],     # Gomma nuova o usata?
            "speed_trap": laps["SpeedST"] if "SpeedST" in laps.columns else None,
        })

        return lap_data

    except Exception as e:
        print(f"    ⚠️ Errore laps R{round_num}: {e}")
        return pd.DataFrame()


def get_race_weather(year: int, round_num: int) -> dict:
    """
    Scarica i dati meteo di una gara.

    Ritorna un DIZIONARIO con le medie della gara:
    - Temperatura aria e pista
    - Umidità
    - Se ha piovuto
    - Velocità del vento

    Perché le medie? Perché il meteo cambia durante la gara,
    ma per le nostre feature ci basta sapere le condizioni generali.
    """
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load()
        weather = session.weather_data

        if weather is None or weather.empty:
            return {}

        return {
            "air_temp": weather["AirTemp"].mean(),
            "track_temp": weather["TrackTemp"].mean(),
            "humidity": weather["Humidity"].mean(),
            "rainfall": weather["Rainfall"].any(),  # True se ha piovuto
            "wind_speed": weather["WindSpeed"].mean(),
            "air_temp_min": weather["AirTemp"].min(),
            "air_temp_max": weather["AirTemp"].max(),
            "track_temp_min": weather["TrackTemp"].min(),
            "track_temp_max": weather["TrackTemp"].max(),
        }

    except Exception as e:
        print(f"    ⚠️ Errore meteo R{round_num}: {e}")
        return {}


def get_pit_stops(year: int, round_num: int) -> pd.DataFrame:
    """
    Estrae i pit stop dai dati dei giri.

    Un pit stop si identifica quando cambia lo stint di un pilota.
    Calcoliamo:
    - Quanti pit stop ha fatto ogni pilota
    - A quale giro ha fatto ogni pit stop
    - Quanto tempo ha perso (approssimato)
    """
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load()
        laps = session.laps

        if laps is None or laps.empty:
            return pd.DataFrame()

        pit_data = []

        for driver in laps["Driver"].unique():
            driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")

            # Un pit stop avviene quando cambia lo stint
            stints = driver_laps["Stint"].dropna()
            if len(stints) < 2:
                pit_data.append({
                    "year": year,
                    "round": round_num,
                    "driver": driver,
                    "num_pit_stops": 0,
                    "pit_laps": "",
                    "compounds_used": driver_laps["Compound"].dropna().unique().tolist(),
                    "num_compounds": driver_laps["Compound"].dropna().nunique(),
                })
                continue

            # Trova i giri in cui è avvenuto il pit stop
            stint_changes = stints.diff().fillna(0)
            pit_laps = driver_laps.loc[stint_changes > 0, "LapNumber"].tolist()

            # Compound usati in ordine
            compounds_per_stint = (
                driver_laps.groupby("Stint")["Compound"]
                .first()
                .dropna()
                .tolist()
            )

            pit_data.append({
                "year": year,
                "round": round_num,
                "driver": driver,
                "num_pit_stops": len(pit_laps),
                "pit_laps": str(pit_laps),
                "compounds_used": compounds_per_stint,
                "num_compounds": len(set(compounds_per_stint)),
            })

        return pd.DataFrame(pit_data)

    except Exception as e:
        print(f"    ⚠️ Errore pit stops R{round_num}: {e}")
        return pd.DataFrame()


def compute_tyre_degradation(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la degradazione delle gomme per ogni stint di ogni pilota.

    DEGRADAZIONE = quanto il tempo sul giro peggiora man mano che
    la gomma invecchia. È un concetto chiave in F1:
    - Le SOFT degradano velocemente ma sono più veloci all'inizio
    - Le HARD degradano lentamente ma sono più lente all'inizio
    - Un team bravo gestisce le gomme meglio (stint più lunghi)

    Calcoliamo il "slope" (pendenza) dei tempi per stint:
    - Slope alto = gomme che degradano velocemente
    - Slope basso = buona gestione gomme
    """
    if laps_df.empty:
        return pd.DataFrame()

    degradation_records = []

    for (year, rnd, driver, stint), stint_laps in laps_df.groupby(
        ["year", "round", "driver", "stint"]
    ):
        times = stint_laps["lap_time_seconds"].dropna()

        # Servono almeno 5 giri per calcolare una degradazione significativa
        if len(times) < 5:
            continue

        # Rimuoviamo outlier (in-lap, out-lap, safety car)
        # Usiamo il metodo IQR: rimuoviamo tempi troppo lontani dalla mediana
        median_time = times.median()
        iqr = times.quantile(0.75) - times.quantile(0.25)
        mask = (times > median_time - 2 * iqr) & (times < median_time + 2 * iqr)
        clean_times = times[mask]

        if len(clean_times) < 5:
            continue

        # Calcoliamo la pendenza con un fit lineare
        # np.polyfit(x, y, 1) restituisce [slope, intercept]
        x = np.arange(len(clean_times))
        slope, intercept = np.polyfit(x, clean_times.values, 1)

        compound = stint_laps["compound"].iloc[0]
        stint_length = len(times)

        degradation_records.append({
            "year": year,
            "round": rnd,
            "driver": driver,
            "stint": stint,
            "compound": compound,
            "stint_length": stint_length,
            "deg_slope": slope,            # Secondi persi per giro
            "avg_lap_time": clean_times.mean(),
            "best_lap_time": clean_times.min(),
            "consistency": clean_times.std(),  # Bassa std = pilota costante
        })

    return pd.DataFrame(degradation_records)


def download_advanced_data(force: bool = False):
    """
    Funzione principale: scarica tutti i dati avanzati.

    Salva 3 file CSV:
    1. laps.csv — tutti i giri di tutte le gare
    2. weather.csv — meteo di ogni gara
    3. pit_stops.csv — pit stop di ogni pilota per ogni gara
    4. tyre_degradation.csv — degradazione gomme per stint
    """
    setup_cache()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    laps_file = PROCESSED_DATA_DIR / "laps.csv"
    weather_file = PROCESSED_DATA_DIR / "weather.csv"
    pits_file = PROCESSED_DATA_DIR / "pit_stops.csv"
    deg_file = PROCESSED_DATA_DIR / "tyre_degradation.csv"

    # Controlla se esiste già (a meno che non forziamo)
    if not force and all(f.exists() for f in [laps_file, weather_file, pits_file]):
        print("📂 Dati avanzati già presenti. Usa force=True per riscaricare.")
        return

    print("🔄 Download dati avanzati (giri, meteo, pit stop)...")
    print("   ⏱️ Questo richiede tempo — ci sono molti dati per gara.\n")

    # Carichiamo il calendario da all_races.csv per sapere quali gare scaricare
    races_file = PROCESSED_DATA_DIR / "all_races.csv"
    if not races_file.exists():
        print("❌ Prima esegui: python -m src.data_loader")
        return

    races_df = pd.read_csv(races_file)
    race_list = (
        races_df.groupby(["year", "round"])["race_name"]
        .first()
        .reset_index()
    )

    all_laps = []
    all_weather = []
    all_pits = []

    for _, race in race_list.iterrows():
        year = int(race["year"])
        rnd = int(race["round"])
        name = race["race_name"]

        print(f"  🏁 {year} R{rnd:02d}: {name}...", end=" ", flush=True)

        try:
            # Laps
            laps = get_race_laps(year, rnd)
            if not laps.empty:
                all_laps.append(laps)

            # Weather
            weather = get_race_weather(year, rnd)
            if weather:
                weather["year"] = year
                weather["round"] = rnd
                weather["race_name"] = name
                all_weather.append(weather)

            # Pit stops
            pits = get_pit_stops(year, rnd)
            if not pits.empty:
                all_pits.append(pits)

            print("✅")

        except Exception as e:
            print(f"❌ {e}")
            continue

        # Pausa breve per non sovraccaricare l'API
        time.sleep(0.5)

    # Salviamo tutto
    if all_laps:
        laps_full = pd.concat(all_laps, ignore_index=True)
        laps_full.to_csv(laps_file, index=False)
        print(f"\n💾 Laps: {len(laps_full)} righe → {laps_file}")

        # Calcoliamo la degradazione
        print("📊 Calcolo degradazione gomme...")
        deg_df = compute_tyre_degradation(laps_full)
        deg_df.to_csv(deg_file, index=False)
        print(f"💾 Degradazione: {len(deg_df)} stint → {deg_file}")

    if all_weather:
        weather_full = pd.DataFrame(all_weather)
        weather_full.to_csv(weather_file, index=False)
        print(f"💾 Meteo: {len(weather_full)} gare → {weather_file}")

    if all_pits:
        pits_full = pd.concat(all_pits, ignore_index=True)
        pits_full.to_csv(pits_file, index=False)
        print(f"💾 Pit stops: {len(pits_full)} righe → {pits_file}")

    print("\n✅ Download dati avanzati completato!")


if __name__ == "__main__":
    download_advanced_data(force=True)