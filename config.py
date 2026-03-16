"""
config.py — Configurazione centrale del progetto
=================================================
Tutti i parametri e le costanti stanno qui.
Quando vuoi cambiare qualcosa (anni, K-factor, ecc.) modifichi SOLO questo file.
"""

from pathlib import Path

# ============================================================
# PERCORSI FILE
# ============================================================
# Path() crea percorsi che funzionano su qualsiasi sistema operativo
# __file__ è il percorso di QUESTO file (config.py)
# .parent è la cartella che lo contiene (f1-predictor/)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ============================================================
# PARAMETRI DATI
# ============================================================
# Anni da scaricare per il training del modello
# 2022 = inizio era effetto suolo (regolamento attuale)
SEASONS = list(range(2022, 2027))  # [2022, 2023, 2024, 2025]

# Quando inizia la stagione 2026, aggiungi semplicemente:
SEASONS_2026 = [2026]

# ============================================================
# PARAMETRI REGOLAMENTO
# ============================================================
# Anni in cui cambiano i regolamenti tecnici in modo significativo.
# In questi anni, l'Elo dei TEAM viene resettato a 1500 perché
# la competitività delle vetture cambia completamente.
# L'Elo dei PILOTI NON viene resettato (l'abilità resta).
#
# Storico:
#   2022 = inizio era effetto suolo
#   2026 = nuovi regolamenti tecnici (motore, aerodinamica, peso)
REGULATION_RESET_YEARS = [2026]


def get_regulation_era_start(year: int) -> int:
    """
    Restituisce il primo anno dell'era regolamentare corrente.

    Per 2022-2025: ritorna 2022 (era effetto suolo).
    Per 2026+: ritorna 2026 (nuovi regolamenti).

    Serve per filtrare i dati storici dei team: quando ci sono
    nuovi regolamenti, la storia dei team delle ere precedenti
    non è più rappresentativa (le vetture sono completamente diverse).
    """
    starts = sorted(REGULATION_RESET_YEARS)
    era_start = SEASONS[0]  # default: primo anno nel dataset
    for reset_year in starts:
        if year >= reset_year:
            era_start = reset_year
    return era_start


# ============================================================
# PARAMETRI ELO
# ============================================================
# Il sistema Elo è nato per gli scacchi. L'idea è semplice:
# - Ogni giocatore ha un "rating" (punteggio di forza)
# - Se batti qualcuno più forte di te, guadagni molti punti
# - Se batti qualcuno più debole, guadagni pochi punti
# - Il rating converge verso il tuo "vero" livello di abilità

# Rating iniziale per un nuovo pilota
ELO_INITIAL_RATING = 1500

# K-factor: quanto velocemente il rating reagisce ai nuovi risultati
# - K alto (40-60) = reagisce molto, volatile
# - K basso (10-20) = reagisce poco, stabile
# Per la F1 usiamo un K medio-alto perché ci sono poche gare (20-24/anno)
ELO_K_FACTOR = 32

# Fattore di scala: controlla la "sensibilità" della formula Elo
# 400 è il valore standard (dagli scacchi)
# Significato: con 400 punti di differenza, il più forte ha ~91% di probabilità di vincere
ELO_SCALE_FACTOR = 400

# Scala per il bootstrap FP dell'Elo team a inizio era regolamentare.
# A inizio nuova era (es. 2026), l'Elo team viene resettato a 1500.
# Se ci sono dati FP del primo round, li usiamo per inizializzare
# l'Elo team con una gerarchia realistica basata sul passo in pista.
# Il valore converte il gap FP (secondi) in punti Elo:
#   80 → 1 secondo di gap ≈ 80 punti Elo di differenza
ELO_FP_BOOTSTRAP_SCALE = 80

# ============================================================
# PARAMETRI MODELLO
# ============================================================
# Quante gare recenti considerare per la "forma" attuale del pilota
ROLLING_WINDOW = 5

# Percentuale dati per il test set (20% = ultime ~4-5 gare della stagione)
TEST_SIZE = 0.2

# Random seed per riproducibilità
# (garantisce che i risultati siano identici ogni volta che esegui il codice)
RANDOM_SEED = 42