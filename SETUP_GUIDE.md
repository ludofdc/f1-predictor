# 🚀 GUIDA SETUP — Passo dopo Passo

Questa guida ti porta da zero a progetto funzionante.
Segui ogni passo nell'ordine. Se qualcosa non funziona, leggi l'errore
con calma — spesso la soluzione è nel messaggio di errore stesso.

---

## Passo 1: Prepara l'ambiente

Apri il terminale in VS Code (Ctrl + `) e scrivi:

```bash
# Vai dove vuoi creare il progetto (es. Desktop)
cd ~/Desktop

# Clona il tuo repo (prima crealo su github.com, vuoto, con nome "f1-predictor")
git clone https://github.com/TUO_USERNAME/f1-predictor.git
cd f1-predictor

# Crea un ambiente virtuale (isola le librerie del progetto)
python3 -m venv venv

# Attivalo:
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
```

Se vedi `(venv)` prima del prompt, l'ambiente è attivo. ✅

---

## Passo 2: Scarica i dati

```bash
python -m src.data_loader
```

Questo scarica i risultati di TUTTE le gare 2022-2025.
La prima volta ci vogliono 5-15 minuti (fastf1 scarica parecchi dati).
Le volte successive è istantaneo perché usa la cache.

---

## Passo 3: Calcola i rating Elo

```bash
python -m src.elo
```

Vedrai la classifica Elo attuale dei piloti. Controlla se ha senso:
Verstappen dovrebbe essere in cima, i rookie più in basso.

---

## Passo 4: Costruisci le feature

```bash
python -m src.feature_engineering
```

Questo crea il file `data/processed/features.csv` con tutte le
feature pronte per il modello.

---

## Passo 5: Allena e valuta il modello

```bash
python -m src.model
```

Vedrai:
- I risultati della cross-validation (MAE per ogni fold)
- La feature importance (quali fattori contano di più)
- Una previsione di esempio sull'ultima gara

---

## Passo 6: Esegui i test

```bash
pytest tests/ -v
```

Tutti i test devono passare. Se no, c'è un bug da fixare.

---

## Passo 7: Pusha su GitHub

```bash
git add .
git commit -m "feat: initial project setup with Elo system and prediction model"
git push origin main
```

---

## Passo 8 (Opzionale): Esplora con i notebook

Apri VS Code, installa l'estensione "Jupyter", e crea notebook nella
cartella `notebooks/` per esplorare i dati visivamente.

---

## Struttura dei comandi da ricordare

| Cosa vuoi fare                    | Comando                              |
|-----------------------------------|--------------------------------------|
| Scaricare dati                    | `python -m src.data_loader`          |
| Calcolare Elo                     | `python -m src.elo`                  |
| Creare feature                    | `python -m src.feature_engineering`  |
| Allenare il modello               | `python -m src.model`                |
| Eseguire i test                   | `pytest tests/ -v`                   |
| Aggiornare con nuovi dati 2026    | Aggiungi 2026 a SEASONS in config.py |

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'fastf1'"**
→ L'ambiente virtuale non è attivo. Fai `source venv/bin/activate`.

**"No data available for round X"**
→ Quella gara non si è ancora corsa. Normale per le ultime gare della stagione.

**I test falliscono**
→ Assicurati di essere nella cartella `f1-predictor/` quando esegui pytest.