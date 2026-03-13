"""
circuit_data.py — Database Caratteristiche Circuiti
====================================================
Questo file contiene informazioni STATICHE su ogni circuito.
Non vengono da un'API ma sono inserite manualmente perché
sono dati stabili che non cambiano (o cambiano molto raramente).

Queste info sono fondamentali per capire PERCHÉ certi team
vanno meglio su certi circuiti:

- Un circuito con molti rettilinei favorisce chi ha il motore migliore
- Un circuito con molte curve lente favorisce chi ha il carico aerodinamico
- Un circuito cittadino favorisce chi è bravo in qualifica (difficile sorpassare)

TIPI DI CIRCUITO:
- "street": cittadino (Monaco, Singapore, Baku) → stretto, muri vicini
- "permanent": circuito permanente (Monza, Spa, Silverstone) → pista dedicata
- "hybrid": mix (Melbourne, Montreal) → strade pubbliche ma con vie di fuga

DOWNFORCE LEVEL:
- "high": molto carico aerodinamico (Monaco, Budapest) → curve lente
- "medium": equilibrato (Silverstone, Barcellona) → mix curve e rettilinei
- "low": poco carico (Monza, Spa) → lunghi rettilinei, velocità massima

POWER SENSITIVITY:
Quanto conta la potenza del motore su questa pista (0-1).
Monza = 0.9 (motore fondamentale), Monaco = 0.2 (quasi irrilevante).
"""
import pandas as pd

# Dizionario: nome_gara → caratteristiche del circuito
CIRCUIT_DATA = {
    # ===== CIRCUITI TRADIZIONALI =====
    "Australian Grand Prix": {
        "circuit": "Albert Park",
        "city": "Melbourne",
        "country": "Australia",
        "type": "hybrid",                # Strade pubbliche ma veloci
        "length_km": 5.278,
        "turns": 14,
        "downforce": "medium",
        "power_sensitivity": 0.6,
        "overtaking_difficulty": 0.5,     # 0=facile, 1=impossibile
        "tyre_stress": 0.5,              # Quanto stressa le gomme
        "altitude_m": 0,
    },
    "Bahrain Grand Prix": {
        "circuit": "Sakhir",
        "city": "Sakhir",
        "country": "Bahrain",
        "type": "permanent",
        "length_km": 5.412,
        "turns": 15,
        "downforce": "medium",
        "power_sensitivity": 0.6,
        "overtaking_difficulty": 0.3,     # Buone zone DRS
        "tyre_stress": 0.7,              # Posteriori soffrono molto
        "altitude_m": 0,
    },
    "Saudi Arabian Grand Prix": {
        "circuit": "Jeddah Corniche",
        "city": "Jeddah",
        "country": "Arabia Saudita",
        "type": "street",
        "length_km": 6.174,
        "turns": 27,
        "downforce": "low",              # Pista velocissima
        "power_sensitivity": 0.8,
        "overtaking_difficulty": 0.6,
        "tyre_stress": 0.4,
        "altitude_m": 0,
    },
    "Japanese Grand Prix": {
        "circuit": "Suzuka",
        "city": "Suzuka",
        "country": "Giappone",
        "type": "permanent",
        "length_km": 5.807,
        "turns": 18,
        "downforce": "high",
        "power_sensitivity": 0.5,
        "overtaking_difficulty": 0.6,
        "tyre_stress": 0.7,
        "altitude_m": 50,
    },
    "Chinese Grand Prix": {
        "circuit": "Shanghai International",
        "city": "Shanghai",
        "country": "Cina",
        "type": "permanent",
        "length_km": 5.451,
        "turns": 16,
        "downforce": "medium",
        "power_sensitivity": 0.7,        # Lungo rettilineo
        "overtaking_difficulty": 0.4,
        "tyre_stress": 0.6,
        "altitude_m": 5,
    },
    "Miami Grand Prix": {
        "circuit": "Miami International Autodrome",
        "city": "Miami",
        "country": "USA",
        "type": "hybrid",
        "length_km": 5.412,
        "turns": 19,
        "downforce": "medium",
        "power_sensitivity": 0.6,
        "overtaking_difficulty": 0.4,
        "tyre_stress": 0.6,
        "altitude_m": 0,
    },
    "Emilia Romagna Grand Prix": {
        "circuit": "Imola",
        "city": "Imola",
        "country": "Italia",
        "type": "permanent",
        "length_km": 4.909,
        "turns": 19,
        "downforce": "high",
        "power_sensitivity": 0.4,
        "overtaking_difficulty": 0.7,     # Pista stretta
        "tyre_stress": 0.5,
        "altitude_m": 47,
    },
    "Monaco Grand Prix": {
        "circuit": "Monte Carlo",
        "city": "Monaco",
        "country": "Monaco",
        "type": "street",
        "length_km": 3.337,
        "turns": 19,
        "downforce": "high",             # Massimo carico
        "power_sensitivity": 0.1,        # Motore quasi irrilevante
        "overtaking_difficulty": 0.95,    # Quasi impossibile sorpassare
        "tyre_stress": 0.3,
        "altitude_m": 0,
    },
    "Spanish Grand Prix": {
        "circuit": "Barcellona-Catalunya",
        "city": "Barcellona",
        "country": "Spagna",
        "type": "permanent",
        "length_km": 4.657,
        "turns": 16,
        "downforce": "high",
        "power_sensitivity": 0.5,
        "overtaking_difficulty": 0.6,
        "tyre_stress": 0.7,              # Curva 3 distrugge le gomme
        "altitude_m": 100,
    },
    "Canadian Grand Prix": {
        "circuit": "Gilles Villeneuve",
        "city": "Montreal",
        "country": "Canada",
        "type": "hybrid",
        "length_km": 4.361,
        "turns": 14,
        "downforce": "low",
        "power_sensitivity": 0.8,        # Stop-and-go, trazione importante
        "overtaking_difficulty": 0.3,
        "tyre_stress": 0.5,
        "altitude_m": 13,
    },
    "Austrian Grand Prix": {
        "circuit": "Red Bull Ring",
        "city": "Spielberg",
        "country": "Austria",
        "type": "permanent",
        "length_km": 4.318,
        "turns": 10,
        "downforce": "low",
        "power_sensitivity": 0.8,
        "overtaking_difficulty": 0.3,     # Buone zone sorpasso
        "tyre_stress": 0.5,
        "altitude_m": 677,               # Altitudine conta!
    },
    "British Grand Prix": {
        "circuit": "Silverstone",
        "city": "Silverstone",
        "country": "UK",
        "type": "permanent",
        "length_km": 5.891,
        "turns": 18,
        "downforce": "high",
        "power_sensitivity": 0.5,
        "overtaking_difficulty": 0.4,
        "tyre_stress": 0.8,              # Curve veloci = stress gomme
        "altitude_m": 150,
    },
    "Hungarian Grand Prix": {
        "circuit": "Hungaroring",
        "city": "Budapest",
        "country": "Ungheria",
        "type": "permanent",
        "length_km": 4.381,
        "turns": 14,
        "downforce": "high",             # Quasi come Monaco
        "power_sensitivity": 0.3,
        "overtaking_difficulty": 0.8,     # Molto difficile sorpassare
        "tyre_stress": 0.6,
        "altitude_m": 264,
    },
    "Belgian Grand Prix": {
        "circuit": "Spa-Francorchamps",
        "city": "Spa",
        "country": "Belgio",
        "type": "permanent",
        "length_km": 7.004,
        "turns": 19,
        "downforce": "low",
        "power_sensitivity": 0.8,
        "overtaking_difficulty": 0.3,
        "tyre_stress": 0.6,
        "altitude_m": 400,
    },
    "Dutch Grand Prix": {
        "circuit": "Zandvoort",
        "city": "Zandvoort",
        "country": "Olanda",
        "type": "permanent",
        "length_km": 4.259,
        "turns": 14,
        "downforce": "high",
        "power_sensitivity": 0.3,
        "overtaking_difficulty": 0.8,
        "tyre_stress": 0.6,
        "altitude_m": 0,
    },
    "Italian Grand Prix": {
        "circuit": "Monza",
        "city": "Monza",
        "country": "Italia",
        "type": "permanent",
        "length_km": 5.793,
        "turns": 11,
        "downforce": "low",              # Minimo carico
        "power_sensitivity": 0.9,        # Il circuito più sensibile al motore
        "overtaking_difficulty": 0.3,
        "tyre_stress": 0.4,
        "altitude_m": 162,
    },
    "Azerbaijan Grand Prix": {
        "circuit": "Baku City Circuit",
        "city": "Baku",
        "country": "Azerbaijan",
        "type": "street",
        "length_km": 6.003,
        "turns": 20,
        "downforce": "low",
        "power_sensitivity": 0.8,        # Lunghissimo rettilineo
        "overtaking_difficulty": 0.4,
        "tyre_stress": 0.5,
        "altitude_m": -28,
    },
    "Singapore Grand Prix": {
        "circuit": "Marina Bay",
        "city": "Singapore",
        "country": "Singapore",
        "type": "street",
        "length_km": 4.940,
        "turns": 19,
        "downforce": "high",
        "power_sensitivity": 0.3,
        "overtaking_difficulty": 0.7,
        "tyre_stress": 0.7,
        "altitude_m": 0,
    },
    "United States Grand Prix": {
        "circuit": "COTA",
        "city": "Austin",
        "country": "USA",
        "type": "permanent",
        "length_km": 5.513,
        "turns": 20,
        "downforce": "medium",
        "power_sensitivity": 0.6,
        "overtaking_difficulty": 0.4,
        "tyre_stress": 0.6,
        "altitude_m": 163,
    },
    "Mexico City Grand Prix": {
        "circuit": "Hermanos Rodríguez",
        "city": "Città del Messico",
        "country": "Messico",
        "type": "permanent",
        "length_km": 4.304,
        "turns": 17,
        "downforce": "high",
        "power_sensitivity": 0.7,
        "overtaking_difficulty": 0.4,
        "tyre_stress": 0.6,
        "altitude_m": 2285,              # Altissimo! L'aria rarefatta cambia tutto
    },
    "São Paulo Grand Prix": {
        "circuit": "Interlagos",
        "city": "San Paolo",
        "country": "Brasile",
        "type": "permanent",
        "length_km": 4.309,
        "turns": 15,
        "downforce": "medium",
        "power_sensitivity": 0.7,
        "overtaking_difficulty": 0.3,
        "tyre_stress": 0.5,
        "altitude_m": 750,
    },
    "Las Vegas Grand Prix": {
        "circuit": "Las Vegas Strip",
        "city": "Las Vegas",
        "country": "USA",
        "type": "street",
        "length_km": 6.201,
        "turns": 17,
        "downforce": "low",
        "power_sensitivity": 0.8,
        "overtaking_difficulty": 0.4,
        "tyre_stress": 0.4,
        "altitude_m": 610,
    },
    "Qatar Grand Prix": {
        "circuit": "Lusail",
        "city": "Lusail",
        "country": "Qatar",
        "type": "permanent",
        "length_km": 5.380,
        "turns": 16,
        "downforce": "high",
        "power_sensitivity": 0.5,
        "overtaking_difficulty": 0.5,
        "tyre_stress": 0.8,              # Molto aggressivo sulle gomme
        "altitude_m": 0,
    },
    "Abu Dhabi Grand Prix": {
        "circuit": "Yas Marina",
        "city": "Abu Dhabi",
        "country": "Emirati",
        "type": "hybrid",
        "length_km": 5.281,
        "turns": 16,
        "downforce": "medium",
        "power_sensitivity": 0.6,
        "overtaking_difficulty": 0.4,
        "tyre_stress": 0.5,
        "altitude_m": 0,
    },
}


# Mappatura numerica per il tipo di downforce
DOWNFORCE_MAP = {"low": 0, "medium": 1, "high": 2}

# Mappatura per il tipo di circuito
CIRCUIT_TYPE_MAP = {"permanent": 0, "hybrid": 1, "street": 2}


def get_circuit_features(race_name: str) -> dict:
    """
    Restituisce le feature del circuito come dizionario numerico,
    pronto per essere usato nel modello.

    Se il circuito non è nel database, restituisce valori medi.
    """
    if race_name in CIRCUIT_DATA:
        data = CIRCUIT_DATA[race_name]
        return {
            "circuit_length_km": data["length_km"],
            "circuit_turns": data["turns"],
            "circuit_downforce": DOWNFORCE_MAP.get(data["downforce"], 1),
            "circuit_type": CIRCUIT_TYPE_MAP.get(data["type"], 0),
            "power_sensitivity": data["power_sensitivity"],
            "overtaking_difficulty": data["overtaking_difficulty"],
            "tyre_stress": data["tyre_stress"],
            "altitude_m": data["altitude_m"],
        }
    else:
        # Valori medi come fallback
        return {
            "circuit_length_km": 5.0,
            "circuit_turns": 16,
            "circuit_downforce": 1,
            "circuit_type": 0,
            "power_sensitivity": 0.5,
            "overtaking_difficulty": 0.5,
            "tyre_stress": 0.5,
            "altitude_m": 100,
        }


def enrich_with_circuit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge le caratteristiche del circuito al DataFrame.
    """
    circuit_features = df["race_name"].apply(get_circuit_features)
    circuit_df = pd.DataFrame(circuit_features.tolist(), index=df.index)
    return pd.concat([df, circuit_df], axis=1)