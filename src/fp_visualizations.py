"""
fp_visualizations.py — Grafici Prove Libere (Long Run + Telemetria)
====================================================================
Questo modulo genera grafici professionali per l'analisi delle FP:

1. LONG RUN BOX PLOTS
   - Box plot stile Project F1 per ogni stint lungo
   - Separati in due pannelli: Top 4 team (Ferrari, Mercedes, McLaren, Red Bull)
     e resto della griglia
   - Colori coerenti con le scuderie ufficiali F1
   - Mostra: media, quartili, outlier, compound, numero giri

2. TELEMETRIA BEST LAP (TOP 3)
   - Speed trace, throttle, brake, gear per i 3 migliori tempi
   - Gap cumulativo tra i piloti
   - Colori scuderia ufficiali
   - Separato per settore con annotazioni

Usa fastf1.plotting per i colori ufficiali delle scuderie.
Matplotlib per i grafici statici (esportati in PDF per massima qualità).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interattivo per salvataggio PDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import fastf1
import fastf1.plotting
import warnings
import sys
from pathlib import Path

sys.path.append("..")
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURAZIONE VISUALE
# ============================================================

# Sfondo scuro stile broadcast F1
DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COLOR = "#2a3a5c"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLOR = "#00d4ff"

# I 4 top team (separati nei grafici long run)
TOP_TEAMS = ["Red Bull Racing", "Ferrari", "McLaren", "Mercedes"]

# Alias team (fastf1 usa nomi lunghi, ma possono variare tra stagioni)
TEAM_ALIASES = {
    "Red Bull Racing": ["Red Bull Racing", "Red Bull", "RBR"],
    "Ferrari": ["Ferrari", "Scuderia Ferrari"],
    "McLaren": ["McLaren", "McLaren F1 Team"],
    "Mercedes": ["Mercedes", "Mercedes-AMG Petronas F1 Team", "Mercedes-AMG"],
}

# Colori compound gomme (stile F1 ufficiale)
COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0072C6",
    "UNKNOWN": "#888888",
}

# Colori team fallback (se fastf1 non disponibile per quella sessione)
TEAM_COLORS_FALLBACK = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Mercedes": "#27F4D2",
    "Aston Martin": "#229971",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    # Nomi storici / varianti
    "Alfa Romeo": "#C92D4B",
    "AlphaTauri": "#5E8FAA",
}


def _setup_cache():
    """Configura la cache di fastf1."""
    cache_dir = RAW_DATA_DIR / "fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def _get_team_color(team_name: str, session=None) -> str:
    """
    Ottieni il colore della scuderia.
    Prova prima con fastf1.plotting, poi fallback ai colori hardcoded.
    """
    if session is not None:
        try:
            color = fastf1.plotting.get_team_color(team_name, session)
            if color:
                return color
        except Exception:
            pass

    # Fallback: cerca nei colori hardcoded
    for team_key, color in TEAM_COLORS_FALLBACK.items():
        if team_key.lower() in team_name.lower() or team_name.lower() in team_key.lower():
            return color

    return "#888888"  # Grigio per team sconosciuti


def _get_driver_team_map(session) -> dict:
    """Mappa pilota → team dalla sessione fastf1."""
    try:
        results = session.results
        if results is not None and not results.empty:
            return dict(zip(results["Abbreviation"], results["TeamName"]))
    except Exception:
        pass
    return {}


def _is_top_team(team_name: str) -> bool:
    """Controlla se il team è tra i top 4."""
    for top in TOP_TEAMS:
        if top.lower() in team_name.lower() or team_name.lower() in top.lower():
            return True
    return False


def _format_laptime(seconds: float) -> str:
    """Formatta secondi in M:SS.mmm"""
    if pd.isna(seconds):
        return "N/A"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


def _apply_dark_theme(fig, axes):
    """Applica il tema scuro a figura e assi."""
    fig.patch.set_facecolor(DARK_BG)

    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, which="both")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.3, linestyle="-")


# ============================================================
# 1. LONG RUN BOX PLOTS
# ============================================================

def plot_long_runs(year: int, round_num: int, session_name: str = "FP2",
                   save: bool = True, show: bool = True) -> plt.Figure:
    """
    Genera box plot dei long run stile Project F1.

    Separato in due pannelli:
    - SOPRA: Top 4 team (Ferrari, Mercedes, McLaren, Red Bull)
    - SOTTO: Altre scuderie

    Ogni box mostra:
    - Il compound usato (colore del bordo)
    - Il numero di giri dello stint
    - Media (rombo rosso), mediana (linea), quartili, whisker

    Parametri:
        year: anno (es. 2024)
        round_num: round del GP (es. 5)
        session_name: "FP1", "FP2", o "FP3" (default FP2)
        save: se True, salva il grafico come PDF
        show: se True, mostra il grafico
    """
    _setup_cache()

    print(f"📊 Generazione long run box plots: {year} R{round_num} {session_name}...")

    # Carica la sessione
    try:
        session = fastf1.get_session(year, round_num, session_name)
        session.load()
    except Exception as e:
        print(f"❌ Errore caricamento sessione: {e}")
        return None

    laps = session.laps
    if laps is None or laps.empty:
        print("❌ Nessun dato giri disponibile")
        return None

    event_name = session.event["EventName"]
    driver_team_map = _get_driver_team_map(session)

    # === IDENTIFICA I LONG RUN ===
    # Un long run è uno stint di almeno 8 giri sullo stesso compound
    long_runs = []

    for driver in laps["Driver"].unique():
        driver_laps = laps[laps["Driver"] == driver].copy()

        if hasattr(driver_laps["LapTime"], "dt"):
            driver_laps["lap_seconds"] = driver_laps["LapTime"].dt.total_seconds()
        else:
            continue

        for stint_num, stint_laps in driver_laps.groupby("Stint"):
            times = stint_laps["lap_seconds"].dropna()
            compound = stint_laps["Compound"].dropna()

            if len(times) < 8 or compound.empty:
                continue

            # Rimuovi primo e ultimo giro (out-lap / in-lap)
            if len(times) > 2:
                clean_times = times.iloc[1:-1]
            else:
                clean_times = times

            # Rimuovi outlier con IQR
            median_t = clean_times.median()
            iqr = clean_times.quantile(0.75) - clean_times.quantile(0.25)
            if iqr > 0:
                clean_times = clean_times[
                    (clean_times > median_t - 2.5 * iqr)
                    & (clean_times < median_t + 2.5 * iqr)
                ]

            if len(clean_times) < 5:
                continue

            team = driver_team_map.get(driver, "Unknown")
            comp_name = compound.iloc[0] if not compound.empty else "UNKNOWN"

            long_runs.append({
                "driver": driver,
                "team": team,
                "stint": int(stint_num),
                "compound": comp_name,
                "num_laps": len(clean_times),
                "times": clean_times.values,
                "avg": clean_times.mean(),
                "is_top_team": _is_top_team(team),
            })

    if not long_runs:
        print("⚠️ Nessun long run trovato (stint < 8 giri)")
        return None

    # Ordina per tempo medio
    long_runs.sort(key=lambda x: x["avg"])

    # Separa top team e altri
    top_runs = [r for r in long_runs if r["is_top_team"]]
    other_runs = [r for r in long_runs if not r["is_top_team"]]

    # === CREA FIGURA ===
    has_top = len(top_runs) > 0
    has_other = len(other_runs) > 0
    n_panels = has_top + has_other

    if n_panels == 0:
        print("⚠️ Nessun long run da visualizzare")
        return None

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(max(14, max(len(top_runs), len(other_runs)) * 1.2 + 2), 6 * n_panels),
        squeeze=False,
    )
    _apply_dark_theme(fig, axes)

    panel_idx = 0

    for runs, title_suffix in [(top_runs, "Top 4 Teams"), (other_runs, "Altre Scuderie")]:
        if not runs:
            continue

        ax = axes[panel_idx, 0]
        panel_idx += 1

        positions = list(range(1, len(runs) + 1))
        labels = []

        for i, run in enumerate(runs):
            pos = positions[i]
            team_color = _get_team_color(run["team"], session)
            comp_color = COMPOUND_COLORS.get(run["compound"], "#888888")

            # Box plot manuale per controllo completo dei colori
            bp = ax.boxplot(
                [run["times"]],
                positions=[pos],
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="#FF4444",
                               markeredgecolor="#FF4444", markersize=6),
                medianprops=dict(color=TEXT_COLOR, linewidth=1.5),
                whiskerprops=dict(color=team_color, linewidth=1.2),
                capprops=dict(color=team_color, linewidth=1.2),
                flierprops=dict(marker="o", markerfacecolor=team_color,
                                markeredgecolor=team_color, markersize=3, alpha=0.5),
            )

            # Colora il box: facecolor = team, bordo = compound
            for patch in bp["boxes"]:
                patch.set_facecolor(team_color)
                patch.set_edgecolor(comp_color)
                patch.set_linewidth(2.5)
                patch.set_alpha(0.7)

            # Label: PILOTA Stint N / COMPOUND (giri)
            label = f"{run['driver']} Stint {run['stint']}\n{run['compound']}\n({run['num_laps']})"
            labels.append(label)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8, color=TEXT_COLOR, ha="center")
        ax.set_ylabel("Lap Time (seconds)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{event_name} {year} — {session_name} Long Run Programs: {title_suffix}",
            fontsize=14, fontweight="bold", pad=15,
        )

        # Legenda compound
        comp_patches = []
        compounds_seen = set(r["compound"] for r in runs)
        for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
            if comp in compounds_seen:
                comp_patches.append(
                    mpatches.Patch(
                        facecolor="none",
                        edgecolor=COMPOUND_COLORS[comp],
                        linewidth=2.5,
                        label=comp,
                    )
                )

        # Legenda media
        comp_patches.append(
            plt.Line2D([0], [0], marker="D", color="w", label="Average",
                       markerfacecolor="#FF4444", markersize=8, linestyle="None")
        )

        ax.legend(
            handles=comp_patches, loc="upper right",
            fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
            labelcolor=TEXT_COLOR,
        )

        # Nota (X) = Number of Laps
        ax.annotate(
            "(X) = Number of Laps",
            xy=(0.5, -0.12), xycoords="axes fraction",
            ha="center", fontsize=9, color=TEXT_COLOR, alpha=0.7,
        )

    fig.tight_layout(pad=2.0)

    if save:
        output_dir = PROCESSED_DATA_DIR / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"long_run_{year}_R{round_num:02d}_{session_name}.pdf"
        fig.savefig(filepath, format="pdf", bbox_inches="tight", facecolor=DARK_BG)
        print(f"💾 Salvato: {filepath}")

    if show:
        plt.show()

    return fig


# ============================================================
# 2. TELEMETRIA BEST LAP — TOP 3
# ============================================================

def plot_telemetry_top3(year: int, round_num: int, session_name: str = "FP2",
                        save: bool = True, show: bool = True) -> plt.Figure:
    """
    Genera confronto telemetrico dei 3 migliori tempi della sessione.

    Pannelli (dall'alto verso il basso):
    1. Speed trace (km/h) con annotazioni settori
    2. Gap cumulativo dal leader (secondi)
    3. Throttle (%)
    4. Gear
    5. RPM (se disponibile)

    Colori: scuderia ufficiale di ogni pilota.

    Parametri:
        year: anno
        round_num: round del GP
        session_name: "FP1", "FP2", o "FP3"
    """
    _setup_cache()

    print(f"📊 Generazione telemetria top 3: {year} R{round_num} {session_name}...")

    try:
        session = fastf1.get_session(year, round_num, session_name)
        session.load()
    except Exception as e:
        print(f"❌ Errore caricamento sessione: {e}")
        return None

    laps = session.laps
    if laps is None or laps.empty:
        print("❌ Nessun dato disponibile")
        return None

    event_name = session.event["EventName"]

    # Trova i 3 migliori tempi (giri più veloci, un pilota = un tempo)
    quicklaps = laps.pick_quicklaps(threshold=1.07)
    if quicklaps.empty:
        quicklaps = laps.copy()

    # Un giro per pilota (il migliore)
    best_per_driver = (
        quicklaps.groupby("Driver")
        .apply(lambda x: x.nsmallest(1, "LapTime"), include_groups=False)
        .reset_index(drop=True)
    )
    top3_laps = best_per_driver.nsmallest(3, "LapTime")

    if len(top3_laps) < 2:
        print("⚠️ Meno di 2 piloti con giri validi")
        return None

    driver_team_map = _get_driver_team_map(session)

    # === CARICA TELEMETRIA ===
    telemetry_data = []
    driver_info = []

    for _, lap in top3_laps.iterrows():
        try:
            tel = lap.get_telemetry().add_distance()
            if tel.empty:
                continue

            driver = lap["Driver"]
            team = driver_team_map.get(driver, "Unknown")
            color = _get_team_color(team, session)
            lap_time = lap["LapTime"].total_seconds()

            telemetry_data.append(tel)
            driver_info.append({
                "driver": driver,
                "team": team,
                "color": color,
                "lap_time": lap_time,
                "lap_time_str": _format_laptime(lap_time),
            })
        except Exception as e:
            print(f"  ⚠️ Telemetria non disponibile per {lap['Driver']}: {e}")
            continue

    if len(telemetry_data) < 2:
        print("⚠️ Telemetria insufficiente")
        return None

    # === COSTRUISCI PANNELLI ===
    n_panels = 4  # Speed, Gap, Throttle, Gear
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(16, 3.5 * n_panels),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1]},
    )
    _apply_dark_theme(fig, axes)

    # Titolo principale
    title_parts = []
    for info in driver_info:
        title_parts.append(f"{info['driver']} {info['lap_time_str']}")
    title = f"{event_name} {year} — {session_name} Best Lap: " + " vs ".join(
        [info["driver"] for info in driver_info]
    )

    fig.suptitle(title, fontsize=15, fontweight="bold", color=TEXT_COLOR, y=0.98)

    # Sotto-titolo con tempi
    times_str = "   ".join(
        [f"{info['driver']}: {info['lap_time_str']}" for info in driver_info]
    )
    fig.text(0.5, 0.955, times_str, ha="center", fontsize=11, color=ACCENT_COLOR)

    # Riferimento: il leader (primo pilota)
    ref_tel = telemetry_data[0]
    ref_distance = ref_tel["Distance"].values

    # --- PANNELLO 1: SPEED TRACE ---
    ax_speed = axes[0]
    for tel, info in zip(telemetry_data, driver_info):
        ax_speed.plot(
            tel["Distance"], tel["Speed"],
            color=info["color"], linewidth=1.5, label=info["driver"], alpha=0.9,
        )

    ax_speed.set_ylabel("Speed (km/h)", fontsize=11, fontweight="bold")
    ax_speed.legend(
        loc="upper right", fontsize=10,
        facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
    )

    # Aggiungi marcatori settori (se disponibili)
    try:
        circuit_info = session.get_circuit_info()
        if hasattr(circuit_info, "marshal_sectors") or hasattr(circuit_info, "corners"):
            corners = circuit_info.corners
            for _, corner in corners.iterrows():
                dist = corner["Distance"]
                number = corner["Number"]
                if dist <= ref_distance.max():
                    ax_speed.axvline(x=dist, color=GRID_COLOR, alpha=0.3, linewidth=0.5)
                    ax_speed.text(
                        dist, ax_speed.get_ylim()[1] * 0.98,
                        f"T{number}", fontsize=6, color=TEXT_COLOR, alpha=0.5,
                        ha="center", va="top",
                    )
    except Exception:
        pass  # Marcatori settori opzionali

    # --- PANNELLO 2: GAP DAL LEADER ---
    ax_gap = axes[1]

    for i, (tel, info) in enumerate(zip(telemetry_data, driver_info)):
        if i == 0:
            # Leader: linea a zero
            ax_gap.axhline(y=0, color=info["color"], linewidth=1.5, alpha=0.9)
            continue

        # Calcola gap cumulativo interpolando sulla distanza del leader
        try:
            # Tempo cumulativo per distanza
            ref_time = ref_tel["Time"].dt.total_seconds().values if hasattr(ref_tel["Time"], "dt") else ref_tel["Time"].values
            comp_time = tel["Time"].dt.total_seconds().values if hasattr(tel["Time"], "dt") else tel["Time"].values

            # Interpola per avere la stessa distanza
            ref_dist = ref_tel["Distance"].values
            comp_dist = tel["Distance"].values

            common_dist = np.linspace(
                max(ref_dist.min(), comp_dist.min()),
                min(ref_dist.max(), comp_dist.max()),
                500,
            )

            ref_interp = np.interp(common_dist, ref_dist, ref_time)
            comp_interp = np.interp(common_dist, comp_dist, comp_time)
            gap = comp_interp - ref_interp

            ax_gap.plot(common_dist, gap, color=info["color"], linewidth=1.5, alpha=0.9)
        except Exception:
            pass

    ax_gap.set_ylabel("Gap (s)", fontsize=11, fontweight="bold")
    ax_gap.axhline(y=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)

    # --- PANNELLO 3: THROTTLE ---
    ax_throttle = axes[2]
    for tel, info in zip(telemetry_data, driver_info):
        if "Throttle" in tel.columns:
            ax_throttle.plot(
                tel["Distance"], tel["Throttle"],
                color=info["color"], linewidth=1, alpha=0.8,
            )
    ax_throttle.set_ylabel("Throttle (%)", fontsize=11, fontweight="bold")
    ax_throttle.set_ylim(-5, 105)

    # --- PANNELLO 4: GEAR ---
    ax_gear = axes[3]
    for tel, info in zip(telemetry_data, driver_info):
        if "nGear" in tel.columns:
            ax_gear.plot(
                tel["Distance"], tel["nGear"],
                color=info["color"], linewidth=1, alpha=0.8,
            )
    ax_gear.set_ylabel("Gear", fontsize=11, fontweight="bold")
    ax_gear.set_xlabel("Distance (m)", fontsize=11, fontweight="bold")
    ax_gear.set_ylim(0, 9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        output_dir = PROCESSED_DATA_DIR / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"telemetry_top3_{year}_R{round_num:02d}_{session_name}.pdf"
        fig.savefig(filepath, format="pdf", bbox_inches="tight", facecolor=DARK_BG)
        print(f"💾 Salvato: {filepath}")

    if show:
        plt.show()

    return fig


# ============================================================
# 3. LONG RUN DEGRADATION TRACE
# ============================================================

def plot_long_run_traces(year: int, round_num: int, session_name: str = "FP2",
                         save: bool = True, show: bool = True) -> plt.Figure:
    """
    Mostra l'evoluzione giro per giro dei long run.

    A differenza del box plot (che mostra la distribuzione),
    questo grafico mostra COME il tempo evolve nel corso dello stint.
    Utile per vedere:
    - Chi degrada di più / meno
    - Chi ha un "cliff" improvviso
    - La differenza di pace tra compound

    Due pannelli: Top 4 team e Altre Scuderie.
    """
    _setup_cache()

    print(f"📊 Generazione long run traces: {year} R{round_num} {session_name}...")

    try:
        session = fastf1.get_session(year, round_num, session_name)
        session.load()
    except Exception as e:
        print(f"❌ Errore caricamento sessione: {e}")
        return None

    laps = session.laps
    if laps is None or laps.empty:
        print("❌ Nessun dato disponibile")
        return None

    event_name = session.event["EventName"]
    driver_team_map = _get_driver_team_map(session)

    # Identifica long run
    long_runs_top = []
    long_runs_other = []

    for driver in laps["Driver"].unique():
        driver_laps = laps[laps["Driver"] == driver].copy()

        if hasattr(driver_laps["LapTime"], "dt"):
            driver_laps["lap_seconds"] = driver_laps["LapTime"].dt.total_seconds()
        else:
            continue

        team = driver_team_map.get(driver, "Unknown")
        color = _get_team_color(team, session)
        is_top = _is_top_team(team)

        for stint_num, stint_laps in driver_laps.groupby("Stint"):
            times = stint_laps["lap_seconds"].dropna()
            compound = stint_laps["Compound"].dropna()

            if len(times) < 8 or compound.empty:
                continue

            # Rimuovi out-lap e in-lap
            if len(times) > 2:
                clean_times = times.iloc[1:-1]
            else:
                clean_times = times

            # Rimuovi outlier
            median_t = clean_times.median()
            iqr = clean_times.quantile(0.75) - clean_times.quantile(0.25)
            if iqr > 0:
                mask = (clean_times > median_t - 2.5 * iqr) & (clean_times < median_t + 2.5 * iqr)
                clean_times = clean_times[mask]

            if len(clean_times) < 5:
                continue

            comp_name = compound.iloc[0]

            run_data = {
                "driver": driver,
                "team": team,
                "color": color,
                "stint": int(stint_num),
                "compound": comp_name,
                "times": clean_times.values,
                "lap_numbers": list(range(1, len(clean_times) + 1)),
            }

            if is_top:
                long_runs_top.append(run_data)
            else:
                long_runs_other.append(run_data)

    has_top = len(long_runs_top) > 0
    has_other = len(long_runs_other) > 0
    n_panels = has_top + has_other

    if n_panels == 0:
        print("⚠️ Nessun long run trovato")
        return None

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 5 * n_panels), squeeze=False,
    )
    _apply_dark_theme(fig, axes)

    panel_idx = 0

    for runs, title_suffix in [(long_runs_top, "Top 4 Teams"), (long_runs_other, "Altre Scuderie")]:
        if not runs:
            continue

        ax = axes[panel_idx, 0]
        panel_idx += 1

        for run in runs:
            # Linea stile compound (tratteggio per HARD, continuo per SOFT)
            linestyle = "-" if run["compound"] in ["SOFT", "MEDIUM"] else "--"

            ax.plot(
                run["lap_numbers"], run["times"],
                color=run["color"], linewidth=1.8, alpha=0.85,
                linestyle=linestyle,
                label=f"{run['driver']} S{run['stint']} {run['compound']} ({len(run['times'])})",
            )

            # Annotazione con nome pilota all'inizio dello stint
            ax.annotate(
                run["driver"],
                xy=(run["lap_numbers"][0], run["times"][0]),
                fontsize=7, color=run["color"], fontweight="bold",
                xytext=(-15, 5), textcoords="offset points",
            )

        ax.set_xlabel("Stint Lap Number", fontsize=11, fontweight="bold")
        ax.set_ylabel("Lap Time (seconds)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{event_name} {year} — {session_name} Long Run Traces: {title_suffix}",
            fontsize=14, fontweight="bold", pad=15,
        )

        ax.legend(
            loc="upper left", fontsize=8, ncol=2,
            facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
        )

    fig.tight_layout(pad=2.0)

    if save:
        output_dir = PROCESSED_DATA_DIR / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"long_run_traces_{year}_R{round_num:02d}_{session_name}.pdf"
        fig.savefig(filepath, format="pdf", bbox_inches="tight", facecolor=DARK_BG)
        print(f"💾 Salvato: {filepath}")

    if show:
        plt.show()

    return fig


# ============================================================
# 4. FUNZIONE COMPLETA: TUTTI I GRAFICI DI UN WEEKEND
# ============================================================

def plot_fp_weekend(year: int, round_num: int, save: bool = True, show: bool = True):
    """
    Genera TUTTI i grafici di analisi FP per un weekend.

    Per ogni sessione (FP1, FP2, FP3):
    - Long run box plot (top 4 + altri)
    - Long run traces (degradazione)
    - Telemetria top 3 best laps

    Parametri:
        year: anno
        round_num: round del GP
    """
    print(f"\n{'=' * 60}")
    print(f"🏎️ GRAFICI FP COMPLETI: {year} Round {round_num}")
    print(f"{'=' * 60}\n")

    for fp_session in ["FP1", "FP2", "FP3"]:
        print(f"\n--- {fp_session} ---")

        try:
            # Long run box plots
            plot_long_runs(year, round_num, fp_session, save=save, show=show)
        except Exception as e:
            print(f"  ⚠️ Long run box plot: {e}")

        try:
            # Long run traces
            plot_long_run_traces(year, round_num, fp_session, save=save, show=show)
        except Exception as e:
            print(f"  ⚠️ Long run traces: {e}")

        try:
            # Telemetria top 3
            plot_telemetry_top3(year, round_num, fp_session, save=save, show=show)
        except Exception as e:
            print(f"  ⚠️ Telemetria: {e}")

    print(f"\n✅ Grafici completati!")
    output_dir = PROCESSED_DATA_DIR / "plots"
    print(f"📂 Tutti i file in: {output_dir}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grafici FP - F1 Predictor")
    parser.add_argument("year", type=int, help="Anno (es. 2024)")
    parser.add_argument("round", type=int, help="Round del GP (es. 5)")
    parser.add_argument("--session", default="FP2", choices=["FP1", "FP2", "FP3"],
                        help="Sessione (default: FP2)")
    parser.add_argument("--all", action="store_true",
                        help="Genera tutti i grafici per tutte le FP")
    parser.add_argument("--no-show", action="store_true",
                        help="Non mostrare i grafici (solo salva)")

    args = parser.parse_args()

    show = not args.no_show

    if args.all:
        plot_fp_weekend(args.year, args.round, save=True, show=show)
    else:
        plot_long_runs(args.year, args.round, args.session, save=True, show=show)
        plot_long_run_traces(args.year, args.round, args.session, save=True, show=show)
        plot_telemetry_top3(args.year, args.round, args.session, save=True, show=show)
