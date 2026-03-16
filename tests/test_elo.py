"""
test_elo.py — Test per il sistema Elo
======================================

PERCHÉ SCRIVERE TEST?
I test verificano automaticamente che il codice funzioni correttamente.
Senza test, ogni volta che modifichi qualcosa potresti rompere
qualcosa che prima funzionava (si chiama "regressione").

Per eseguire i test: pytest tests/test_elo.py -v
(-v = verbose, mostra i dettagli di ogni test)

COME FUNZIONA PYTEST:
- Ogni funzione che inizia con "test_" viene eseguita automaticamente
- assert verifica che una condizione sia vera
- Se l'assert fallisce → il test fallisce → sai che c'è un bug
"""

import sys
sys.path.insert(0, ".")

from src.elo import EloSystem


def test_initial_rating():
    """Un nuovo pilota deve avere rating 1500."""
    elo = EloSystem()
    rating = elo.get_rating("VER")
    assert rating == 1500, f"Rating iniziale dovrebbe essere 1500, non {rating}"


def test_expected_score_equal_ratings():
    """Due piloti con lo stesso rating devono avere 50% di probabilità."""
    elo = EloSystem()
    expected = elo.expected_score(1500, 1500)
    assert abs(expected - 0.5) < 0.001, f"Expected score dovrebbe essere 0.5, non {expected}"


def test_expected_score_higher_rating_favored():
    """Il pilota con rating più alto deve avere >50% di probabilità."""
    elo = EloSystem()
    # Pilota A (1600) vs Pilota B (1400)
    expected_a = elo.expected_score(1600, 1400)
    assert expected_a > 0.5, f"Pilota più forte dovrebbe avere >50%, non {expected_a}"

    # Simmetria: la probabilità di B deve essere 1 - probabilità di A
    expected_b = elo.expected_score(1400, 1600)
    assert abs(expected_a + expected_b - 1.0) < 0.001, "Le probabilità devono sommare a 1"


def test_rating_update_winner_gains():
    """Il vincitore deve guadagnare punti."""
    elo = EloSystem()

    # Simuliamo una gara con 3 piloti
    race_result = [("VER", 1), ("HAM", 2), ("LEC", 3)]
    elo.update_after_race(race_result)

    # Il vincitore deve avere rating > 1500 (iniziale)
    assert elo.get_rating("VER") > 1500, "Il vincitore deve guadagnare punti"
    # L'ultimo deve avere rating < 1500
    assert elo.get_rating("LEC") < 1500, "L'ultimo deve perdere punti"


def test_rating_update_zero_sum():
    """La somma dei cambiamenti di rating deve essere circa zero.
    (Il sistema Elo è a somma zero: i punti guadagnati da uno sono persi da un altro)"""
    elo = EloSystem()

    race_result = [("VER", 1), ("HAM", 2), ("LEC", 3), ("NOR", 4)]
    elo.update_after_race(race_result)

    total_change = sum(
        elo.get_rating(driver) - 1500
        for driver in ["VER", "HAM", "LEC", "NOR"]
    )
    assert abs(total_change) < 1.0, f"Somma cambiamenti dovrebbe essere ~0, non {total_change}"


def test_consistent_winner_rises():
    """Un pilota che vince sempre deve avere un rating molto alto."""
    elo = EloSystem()

    # Simuliamo 10 gare dove VER vince sempre
    for _ in range(10):
        race_result = [("VER", 1), ("HAM", 2), ("LEC", 3)]
        elo.update_after_race(race_result)

    assert elo.get_rating("VER") > 1600, "Un vincitore seriale deve avere rating alto"
    assert elo.get_rating("VER") > elo.get_rating("HAM"), "VER deve essere sopra HAM"
    assert elo.get_rating("HAM") > elo.get_rating("LEC"), "HAM deve essere sopra LEC"


def test_empty_race_no_crash():
    """Una gara vuota o con 1 solo pilota non deve causare errori."""
    elo = EloSystem()

    # Gara vuota
    elo.update_after_race([])

    # Un solo pilota
    elo.update_after_race([("VER", 1)])

    # Nessun crash = test passato
    assert True


def test_get_current_ratings_sorted():
    """La classifica deve essere ordinata dal rating più alto al più basso."""
    elo = EloSystem()

    # Creiamo una situazione con rating diversi
    for _ in range(5):
        elo.update_after_race([("VER", 1), ("HAM", 2), ("LEC", 3)])

    ratings_df = elo.get_current_ratings()

    # Verifica che sia ordinato in modo decrescente
    ratings_list = ratings_df["elo_rating"].tolist()
    assert ratings_list == sorted(ratings_list, reverse=True), "La classifica deve essere ordinata"


def test_team_elo_regulation_reset():
    """L'Elo dei team deve resettarsi negli anni di cambio regolamento.
    L'Elo dei piloti NON deve resettarsi (la bravura resta)."""
    import pandas as pd
    from src.elo import compute_team_elo
    from config import REGULATION_RESET_YEARS, ELO_INITIAL_RATING

    # Creiamo dati finti che attraversano il confine 2025→2026
    rows = []
    for year in [2025, 2026]:
        for rnd in [1, 2]:
            for driver, team, pos in [
                ("VER", "Red Bull Racing", 1),
                ("HAM", "Mercedes", 2),
                ("LEC", "Ferrari", 3),
            ]:
                rows.append({
                    "year": year, "round": rnd, "race_name": f"Test GP R{rnd}",
                    "driver": driver, "team": team,
                    "grid_position": pos, "finish_position": pos, "dnf": False,
                })
    df = pd.DataFrame(rows)

    # 2026 deve essere un anno di reset
    assert 2026 in REGULATION_RESET_YEARS, "2026 deve essere in REGULATION_RESET_YEARS"

    # ── TEAM ELO: deve resettarsi nel 2026 ──
    # Dopo il reset, l'Elo potrebbe essere 1500 (senza FP) oppure diverso
    # se c'è un bootstrap FP. In ogni caso, l'Elo pre-2026 NON deve
    # sopravvivere: verifichiamo che i valori di fine 2025 vengano cancellati.
    team_elo_history = compute_team_elo(df)

    # Prendiamo l'Elo di fine 2025 (pre-race R2)
    last_2025 = team_elo_history[
        (team_elo_history["year"] == 2025) & (team_elo_history["round"] == 2)
    ]
    first_2026_team = team_elo_history[
        (team_elo_history["year"] == 2026) & (team_elo_history["round"] == 1)
    ]

    # Red Bull aveva vinto tutte le gare nel 2025, quindi il suo Elo 2025
    # deve essere alto. Nel 2026 deve essere resettato (vicino a 1500 o
    # inizializzato da FP, ma NON il valore alto di fine 2025).
    rb_elo_2025 = last_2025[last_2025["team"] == "Red Bull Racing"][
        "team_elo_pre_race"
    ].iloc[0]
    rb_elo_2026 = first_2026_team[first_2026_team["team"] == "Red Bull Racing"][
        "team_elo_pre_race"
    ].iloc[0]
    # L'Elo 2025 di Red Bull è > 1500 (vincitore). Dopo il reset 2026
    # deve essere sceso significativamente (reset + eventuale bootstrap FP).
    assert rb_elo_2025 > ELO_INITIAL_RATING, (
        "Red Bull nel 2025 dovrebbe avere Elo > 1500 (vincitore)"
    )
    assert rb_elo_2026 < rb_elo_2025, (
        f"Red Bull Elo 2026 ({rb_elo_2026:.1f}) deve essere < 2025 ({rb_elo_2025:.1f}) dopo reset"
    )

    # ── Verifica che nel 2025 i team NON siano tutti a 1500 ──
    # (dopo 2 gare nel 2025, i rating devono essere diversi da 1500)
    last_2025_team = team_elo_history[
        (team_elo_history["year"] == 2025) & (team_elo_history["round"] == 2)
    ]
    any_different = any(
        abs(row["team_elo_pre_race"] - ELO_INITIAL_RATING) > 0.01
        for _, row in last_2025_team.iterrows()
    )
    assert any_different, (
        "Nel round 2 del 2025, almeno un team dovrebbe avere Elo diverso da 1500"
    )

    # ── DRIVER ELO: NON deve resettarsi ──
    elo = EloSystem()
    # Processiamo le gare 2025
    for rnd in [1, 2]:
        race = df[(df["year"] == 2025) & (df["round"] == rnd)]
        result_list = list(zip(race["driver"], race["finish_position"]))
        elo.update_after_race(result_list)

    # Salviamo i rating post-2025
    ver_elo_after_2025 = elo.get_rating("VER")
    assert ver_elo_after_2025 != ELO_INITIAL_RATING, (
        "Dopo 2 gare nel 2025, VER dovrebbe avere Elo diverso da 1500"
    )

    # Processiamo le gare 2026 (senza reset — l'Elo piloti NON si resetta)
    for rnd in [1, 2]:
        race = df[(df["year"] == 2026) & (df["round"] == rnd)]
        result_list = list(zip(race["driver"], race["finish_position"]))
        elo.update_after_race(result_list)

    # VER Elo nel 2026 deve essere > 1500 (non resettato)
    ver_elo_in_2026 = elo.get_rating("VER")
    assert ver_elo_in_2026 > ELO_INITIAL_RATING, (
        f"VER Elo NON deve resettarsi nel 2026: atteso >{ELO_INITIAL_RATING}, "
        f"ottenuto {ver_elo_in_2026}"
    )


def test_team_elo_fp_bootstrap():
    """A inizio era regolamentare, l'Elo team deve essere inizializzato
    usando i dati FP del primo round (bootstrap), non restare a 1500."""
    import pandas as pd
    import os
    from src.elo import compute_team_elo
    from config import (
        REGULATION_RESET_YEARS,
        ELO_INITIAL_RATING,
        ELO_FP_BOOTSTRAP_SCALE,
        PROCESSED_DATA_DIR,
    )

    # Creiamo un fp_summary.csv temporaneo con delta FP diversi per team
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp_file = PROCESSED_DATA_DIR / "fp_summary.csv"
    fp_backup = None
    if fp_file.exists():
        fp_backup = pd.read_csv(fp_file)

    # FP finti: Mercedes veloce (delta=0.1), Ferrari media (0.5), Red Bull lenta (1.0)
    fp_data = pd.DataFrame([
        {"year": 2026, "round": 1, "driver": "HAM", "fp_best_lap_delta": 0.1},
        {"year": 2026, "round": 1, "driver": "RUS", "fp_best_lap_delta": 0.1},
        {"year": 2026, "round": 1, "driver": "LEC", "fp_best_lap_delta": 0.5},
        {"year": 2026, "round": 1, "driver": "SAI", "fp_best_lap_delta": 0.5},
        {"year": 2026, "round": 1, "driver": "VER", "fp_best_lap_delta": 1.0},
        {"year": 2026, "round": 1, "driver": "PER", "fp_best_lap_delta": 1.0},
    ])
    fp_data.to_csv(fp_file, index=False)

    try:
        # Dati gara finti: 2025 + 2026
        rows = []
        for year in [2025, 2026]:
            for rnd in [1, 2]:
                for driver, team, pos in [
                    ("HAM", "Mercedes", 1),
                    ("RUS", "Mercedes", 2),
                    ("LEC", "Ferrari", 3),
                    ("SAI", "Ferrari", 4),
                    ("VER", "Red Bull Racing", 5),
                    ("PER", "Red Bull Racing", 6),
                ]:
                    rows.append({
                        "year": year, "round": rnd,
                        "race_name": f"Test GP R{rnd}",
                        "driver": driver, "team": team,
                        "grid_position": pos, "finish_position": pos,
                        "dnf": False,
                    })
        df = pd.DataFrame(rows)

        assert 2026 in REGULATION_RESET_YEARS

        team_elo_history = compute_team_elo(df)
        first_2026 = team_elo_history[
            (team_elo_history["year"] == 2026)
            & (team_elo_history["round"] == 1)
        ]

        # Mercedes (delta=0.1) deve avere Elo > Ferrari (delta=0.5) > Red Bull (delta=1.0)
        merc_elo = first_2026[first_2026["team"] == "Mercedes"][
            "team_elo_pre_race"
        ].iloc[0]
        ferrari_elo = first_2026[first_2026["team"] == "Ferrari"][
            "team_elo_pre_race"
        ].iloc[0]
        rb_elo = first_2026[first_2026["team"] == "Red Bull Racing"][
            "team_elo_pre_race"
        ].iloc[0]

        # Verifica gerarchia
        assert merc_elo > ferrari_elo, (
            f"Mercedes ({merc_elo:.1f}) dovrebbe avere Elo > Ferrari ({ferrari_elo:.1f})"
        )
        assert ferrari_elo > rb_elo, (
            f"Ferrari ({ferrari_elo:.1f}) dovrebbe avere Elo > Red Bull ({rb_elo:.1f})"
        )

        # Verifica che NON siano tutti 1500
        assert merc_elo != ELO_INITIAL_RATING, (
            "Con bootstrap FP, Mercedes non dovrebbe essere a 1500"
        )

        # Verifica valori attesi
        expected_merc = ELO_INITIAL_RATING + (-0.1 * ELO_FP_BOOTSTRAP_SCALE)
        expected_rb = ELO_INITIAL_RATING + (-1.0 * ELO_FP_BOOTSTRAP_SCALE)
        assert abs(merc_elo - expected_merc) < 0.01, (
            f"Mercedes Elo atteso {expected_merc}, ottenuto {merc_elo}"
        )
        assert abs(rb_elo - expected_rb) < 0.01, (
            f"Red Bull Elo atteso {expected_rb}, ottenuto {rb_elo}"
        )

    finally:
        # Ripristina fp_summary.csv originale
        if fp_backup is not None:
            fp_backup.to_csv(fp_file, index=False)
        elif fp_file.exists():
            os.remove(fp_file)


if __name__ == "__main__":
    # Puoi anche eseguire i test direttamente con: python tests/test_elo.py
    import pytest
    pytest.main([__file__, "-v"])