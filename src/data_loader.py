"""
data_loader.py — Scarica e prepara i dati F1
=============================================
Questo modulo si occupa di:
1. Scaricare i risultati delle gare (da fastf1/Ergast API)
2. Pulirli e organizzarli in un formato utile
3. Salvarli come file CSV per non doverli riscaricare ogni volta

CONCETTO CHIAVE: "Separazione delle responsabilità"
Questo file fa UNA cosa: gestire i dati. Non calcola Elo, non fa previsioni.
Ogni modulo ha il suo compito. Questo rende il codice facile da mantenere.
"""

import pandas as pd
import fastf1
import warnings
import sys
import re

# Importiamo le configurazioni dal nostro file config.py
# Il ".." dice "vai nella cartella padre" (da src/ a f1-predictor/)
sys.path.append("..")
from config import SEASONS, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Disattiviamo i warning fastididi di fastf1 (non sono errori, solo avvisi)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# MAPPING gare → race_id e slug per il sito ufficiale F1
# ============================================================
# Il sito formula1.com usa URL tipo:
#   /en/results/{anno}/races/{race_id}/{slug}/starting-grid
#
# Questi ID cambiano ogni anno. Qui mappiamo le gare 2026.
# Per aggiungere nuovi anni, basta aggiungere le entry.
# Gli ID si trovano su: https://www.formula1.com/en/results/{anno}/races
#
# FORMATO: (anno, round) → (race_id, slug_url)

F1_RACE_IDS = {
    # ─── 2026 ───
    (2026, 1): (1279, "australia"),
    (2026, 2): (1280, "china"),
    (2026, 3): (1281, "japan"),
    (2026, 4): (1282, "bahrain"),
    (2026, 5): (1283, "saudi-arabia"),
    (2026, 6): (1284, "miami"),
    (2026, 7): (1285, "canada"),
    (2026, 8): (1286, "monaco"),
    (2026, 9): (1287, "barcelona-catalunya"),
    (2026, 10): (1288, "austria"),
    (2026, 11): (1289, "great-britain"),
    (2026, 12): (1290, "belgium"),
    (2026, 13): (1291, "hungary"),
    (2026, 14): (1292, "netherlands"),
    (2026, 15): (1293, "italy"),
    (2026, 16): (1294, "spain"),
    (2026, 17): (1295, "azerbaijan"),
    (2026, 18): (1296, "singapore"),
    (2026, 19): (1297, "united-states"),
    (2026, 20): (1298, "mexico"),
    (2026, 21): (1299, "brazil"),
    (2026, 22): (1300, "las-vegas"),
    (2026, 23): (1301, "qatar"),
    (2026, 24): (1302, "abu-dhabi"),
}


def _fetch_starting_grid_from_f1(year: int, round_num: int) -> dict | None:
    """
    Scarica la starting grid ufficiale dal sito formula1.com.

    Ritorna:
        Un dizionario {abbreviazione_pilota: posizione_griglia}
        oppure None se non disponibile.

    COME FUNZIONA:
    1. Costruiamo l'URL della pagina starting-grid sul sito F1
    2. Scarichiamo l'HTML con urllib (nessuna dipendenza extra)
    3. Estraiamo posizione + abbreviazione pilota con regex
       Pattern: la tabella HTML ha:
         - Posizione in: flush-left...">POS</td>
         - Abbreviazione in: md:hidden">ABC</span>
    4. Ritorniamo il dizionario {abbr: posizione}

    NOTA: se il sito F1 cambia struttura HTML, il regex potrebbe rompersi.
    In quel caso la funzione ritorna None e si usa il fallback fastf1/qualifica.
    """
    key = (year, round_num)
    if key not in F1_RACE_IDS:
        return None

    race_id, slug = F1_RACE_IDS[key]
    url = f"https://www.formula1.com/en/results/{year}/races/{race_id}/{slug}/starting-grid"

    try:
        import urllib.request
        import ssl

        # Disabilitiamo la verifica SSL (alcuni Mac hanno problemi con i certificati)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html",
            },
        )
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # REGEX: la tabella del sito F1 ha questa struttura per ogni riga:
        #   <td ...flush-left...">POS</td>  ...  <span ...md:hidden">ABC</span>
        # dove POS = posizione (1-22) e ABC = abbreviazione pilota (3 lettere)
        pattern = r'flush-left[^"]*">(\d+)</td>.*?md:hidden">([A-Z]{3})</span>'
        matches = re.findall(pattern, html, re.DOTALL)

        if len(matches) >= 15:
            grid = {}
            for pos_str, abbr in matches:
                grid[abbr] = int(pos_str)
            return grid

        return None

    except Exception:
        # Qualsiasi errore (rete, parsing, timeout) → ritorniamo None
        return None


def setup_directories():
    """
    Crea le cartelle per i dati se non esistono ancora.
    mkdir(parents=True) crea anche le cartelle intermedie.
    exist_ok=True evita errori se la cartella esiste già.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Configura la cache di fastf1 (evita di riscaricare dati già ottenuti)
    cache_dir = RAW_DATA_DIR / "fastf1_cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def get_season_results(year: int) -> pd.DataFrame:
    """
    Scarica i risultati di TUTTE le gare di una stagione.

    Parametri:
        year: l'anno della stagione (es. 2024)

    Ritorna:
        Un DataFrame pandas con una riga per ogni pilota per ogni gara.

    COME FUNZIONA:
    1. Chiediamo a fastf1 il calendario della stagione
    2. Per ogni gara, scarichiamo la sessione "Race" (la gara vera)
    3. Se la gara non è ancora avvenuta, proviamo a creare una riga
       "provvisoria" con i dati della qualifica (griglia + piloti/team)
    4. Estraiamo i risultati e li mettiamo tutti insieme
    """
    print(f"\n{'='*50}")
    print(f"📥 Scaricamento dati stagione {year}...")
    print(f"{'='*50}")

    all_races = []

    # Otteniamo il calendario: lista di tutti i GP della stagione
    schedule = fastf1.get_event_schedule(year)

    # Filtriamo solo le gare "convenzionali" (escludiamo testing, ecc.)
    # I round > 0 sono le gare vere del campionato
    races = schedule[schedule["RoundNumber"] > 0]

    for _, race_info in races.iterrows():
        round_num = race_info["RoundNumber"]
        race_name = race_info["EventName"]

        try:
            print(f"  🏁 Round {round_num}: {race_name}...", end=" ")

            # Carichiamo la sessione gara
            # 'R' = Race (altri valori: 'Q' = Qualifica, 'FP1' = Prove Libere 1)
            session = fastf1.get_session(year, round_num, "R")
            session.load()

            # session.results è un DataFrame con tutti i risultati
            results = session.results

            if results is None or results.empty:
                # ── GARA NON ANCORA AVVENUTA ──
                # Proviamo a creare dati provvisori dalla qualifica
                # Questo permette di fare previsioni PRE-GARA
                race_data = _try_create_from_qualifying(year, round_num, race_name)
                if race_data is not None:
                    all_races.append(race_data)
                    print(f"📋 {len(race_data)} piloti (da qualifica, gara non ancora avvenuta)")
                else:
                    print("⚠️ Nessun dato disponibile, skip")
                continue

            # Creiamo un DataFrame pulito con solo le colonne che ci servono
            race_data = pd.DataFrame(
                {
                    "year": year,
                    "round": round_num,
                    "race_name": race_name,
                    "driver": results["Abbreviation"],
                    "driver_full_name": results["FullName"],
                    "team": results["TeamName"],
                    "grid_position": results["GridPosition"],
                    "finish_position": results["Position"],
                    "points": results["Points"],
                    "status": results["Status"],
                    # ClassifiedPosition è la posizione ufficiale
                    # (può essere diversa se un pilota è squalificato)
                    "classified_position": results["ClassifiedPosition"],
                }
            )

            # Pulizia: convertiamo le posizioni in numeri
            # Alcune posizioni possono essere 'R' (retired) o NaN
            race_data["grid_position"] = pd.to_numeric(
                race_data["grid_position"], errors="coerce"
            )
            race_data["finish_position"] = pd.to_numeric(
                race_data["finish_position"], errors="coerce"
            )

            # Aggiungiamo una colonna "dnf" (Did Not Finish)
            # Un pilota è DNF se il suo status non è "Finished" e non è "+X Lap(s)"
            race_data["dnf"] = ~race_data["status"].str.contains(
                r"Finished|\+\d+ Lap", case=False, na=True
            )

            all_races.append(race_data)
            print(f"✅ {len(race_data)} piloti")

        except Exception as e:
            print(f"❌ Errore: {e}")
            continue

    if not all_races:
        print(f"⚠️ Nessun dato trovato per {year}")
        return pd.DataFrame()

    # Concateniamo tutti i DataFrame in uno solo
    # ignore_index=True resetta l'indice (0, 1, 2, ... invece di indici duplicati)
    season_df = pd.concat(all_races, ignore_index=True)
    print(f"\n✅ Stagione {year}: {len(season_df)} righe totali")

    return season_df


def _try_create_from_qualifying(year: int, round_num: int, race_name: str):
    """
    Crea dati provvisori per una gara non ancora avvenuta, usando la qualifica.

    Quando la gara non è ancora corsa ma la qualifica sì, creiamo
    righe "provvisorie" con:
    - grid_position costruita dalla qualifica (Position + fondo griglia per DNS/no-time)
    - finish_position = NaN (gara non corsa)
    - dnf = False
    - points = 0

    GESTIONE GRIGLIA (priorità):
    1. Proviamo a caricare la sessione Race per GridPosition ufficiale
       (include penalità). Disponibile solo se pubblicata dalla FIA.
    2. Fallback 1: proviamo a scaricare la starting grid dal sito
       ufficiale formula1.com (include penalità, ordine definitivo).
    3. Fallback 2: usiamo Position dalla qualifica.
    4. Piloti senza tempo (Position = NaN): vengono messi in fondo
       alla griglia in ordine di apparizione (tipicamente DNS/penalità pesanti).

    Questo permette al modello di fare previsioni PRE-GARA.
    """
    try:
        session = fastf1.get_session(year, round_num, "Q")
        session.load()
        results = session.results

        if results is None or results.empty:
            return None

        # === COSTRUZIONE GRIGLIA ===
        # Step 1: Proviamo a prendere GridPosition dalla sessione Race
        #         (contiene penalità applicate = griglia effettiva)
        grid_from_race = None
        try:
            race_session = fastf1.get_session(year, round_num, "R")
            race_session.load()
            if (
                race_session.results is not None
                and not race_session.results.empty
                and "GridPosition" in race_session.results.columns
            ):
                gp = race_session.results[["Abbreviation", "GridPosition"]].copy()
                gp["GridPosition"] = pd.to_numeric(gp["GridPosition"], errors="coerce")
                if gp["GridPosition"].notna().sum() > 0:
                    grid_from_race = dict(
                        zip(gp["Abbreviation"], gp["GridPosition"])
                    )
        except Exception:
            pass  # Sessione Race non disponibile — proviamo sito F1

        # Step 1b: Se fastf1 non ha la griglia, proviamo il sito ufficiale F1
        #          Questo è utile per gare "in corso" dove la griglia è pubblicata
        #          sul sito ma non ancora nella sessione Race di fastf1.
        grid_from_f1 = None
        if grid_from_race is None:
            grid_from_f1 = _fetch_starting_grid_from_f1(year, round_num)
            if grid_from_f1:
                print(f"🌐 Griglia ufficiale dal sito F1...", end=" ")

        # Step 2: Costruisci la griglia (con priorità)
        quali_positions = pd.to_numeric(results["Position"], errors="coerce")

        if grid_from_race:
            # Priorità 1: Griglia ufficiale dalla Race session (con penalità)
            grid_positions = results["Abbreviation"].map(grid_from_race)
        elif grid_from_f1:
            # Priorità 2: Griglia dal sito ufficiale formula1.com
            grid_positions = results["Abbreviation"].map(grid_from_f1)
        else:
            # Priorità 3: Fallback = risultato qualifica come griglia
            grid_positions = quali_positions.copy()

        # Step 3: Piloti senza posizione (NaN) = non hanno segnato tempo
        #         Li mettiamo in fondo alla griglia
        max_valid = grid_positions.dropna().max()
        if pd.isna(max_valid):
            max_valid = 0
        next_pos = int(max_valid) + 1
        for idx in grid_positions[grid_positions.isna()].index:
            grid_positions.at[idx] = next_pos
            next_pos += 1

        race_data = pd.DataFrame(
            {
                "year": year,
                "round": round_num,
                "race_name": race_name,
                "driver": results["Abbreviation"],
                "driver_full_name": results["FullName"],
                "team": results["TeamName"],
                "grid_position": grid_positions,
                "finish_position": float("nan"),        # Gara non avvenuta
                "points": 0.0,
                "status": "Not Yet Raced",
                "classified_position": float("nan"),
            }
        )

        race_data["grid_position"] = pd.to_numeric(
            race_data["grid_position"], errors="coerce"
        )
        race_data["dnf"] = False

        return race_data

    except Exception:
        return None


def get_qualifying_results(year: int) -> pd.DataFrame:
    """
    Scarica i risultati delle qualifiche per una stagione.
    Li usiamo come feature aggiuntiva (la qualifica è molto predittiva in F1).
    """
    print(f"\n📥 Scaricamento qualifiche {year}...")

    all_quali = []
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule["RoundNumber"] > 0]

    for _, race_info in races.iterrows():
        round_num = race_info["RoundNumber"]

        try:
            session = fastf1.get_session(year, round_num, "Q")
            session.load()
            results = session.results

            if results is None or results.empty:
                continue

            quali_data = pd.DataFrame(
                {
                    "year": year,
                    "round": round_num,
                    "driver": results["Abbreviation"],
                    "quali_position": results["Position"],
                }
            )

            quali_data["quali_position"] = pd.to_numeric(
                quali_data["quali_position"], errors="coerce"
            )

            all_quali.append(quali_data)

        except Exception:
            continue

    if not all_quali:
        return pd.DataFrame()

    return pd.concat(all_quali, ignore_index=True)


def load_all_data(force_download: bool = False) -> pd.DataFrame:
    """
    Funzione principale: carica tutti i dati.

    Se i dati sono già stati scaricati (file CSV esiste), li carica dal file.
    Altrimenti li scarica da zero.

    Parametri:
        force_download: se True, riscarica tutto anche se il file esiste

    Ritorna:
        DataFrame completo con tutti i risultati di tutte le stagioni.
    """
    setup_directories()

    output_file = PROCESSED_DATA_DIR / "all_races.csv"

    # Se il file esiste già e non vogliamo forzare il download, carichiamo quello
    if output_file.exists() and not force_download:
        print(f"📂 Caricamento dati da {output_file}...")
        return pd.read_csv(output_file)

    # Altrimenti, scarichiamo tutto
    print("🔄 Download completo dei dati F1...")
    print(f"   Stagioni: {SEASONS}")

    all_data = []
    all_quali = []

    for year in SEASONS:
        # Risultati gara
        race_df = get_season_results(year)
        if not race_df.empty:
            all_data.append(race_df)

        # Risultati qualifica
        quali_df = get_qualifying_results(year)
        if not quali_df.empty:
            all_quali.append(quali_df)

    if not all_data:
        raise ValueError("Nessun dato scaricato! Controlla la connessione.")

    # Uniamo tutto
    full_df = pd.concat(all_data, ignore_index=True)

    # Se abbiamo dati qualifica, li uniamo ai dati gara
    if all_quali:
        quali_full = pd.concat(all_quali, ignore_index=True)
        # merge = come un JOIN in SQL
        # Uniamo su year + round + driver (la chiave univoca)
        full_df = full_df.merge(
            quali_full, on=["year", "round", "driver"], how="left"
        )

    # Ordiniamo per anno, round e posizione finale
    full_df = full_df.sort_values(
        ["year", "round", "finish_position"]
    ).reset_index(drop=True)

    # Salviamo il CSV
    full_df.to_csv(output_file, index=False)
    print(f"\n💾 Dati salvati in {output_file}")
    print(f"   Totale: {len(full_df)} righe")

    return full_df


# ============================================================
# Questo blocco si esegue SOLO quando lanci direttamente questo file:
#   python -m src.data_loader
# NON si esegue quando importi il modulo da un altro file.
# ============================================================
if __name__ == "__main__":
    df = load_all_data(force_download=True)
    print("\n📊 Anteprima dati:")
    print(df.head(10))
    print(f"\nColonne: {list(df.columns)}")
    print(f"Stagioni: {df['year'].unique()}")
    print(f"Piloti unici: {df['driver'].nunique()}")