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

## Passo 2: Setup iniziale (solo la prima volta)

Un solo comando scarica tutti i dati, calcola Elo, e costruisce le feature:

```bash
python commands.py
# → Scegli 30 (Aggiorna stagione)
```

Questo fa automaticamente:
1. Download risultati gare (2022-2026)
2. Download dati avanzati (meteo, gomme, pit stop)
3. Download dati prove libere (FP1/FP2/FP3)
4. Calcolo Elo piloti e team
5. Costruzione feature matrix completa

La prima volta ci vogliono 15-20 minuti (fastf1 scarica parecchi dati).
Le volte successive è molto più veloce perché usa la cache.

---

## Passo 3: Esegui i test

```bash
pytest tests/ -v
```

Tutti i test devono passare. Se no, c'è un bug da fixare.

---

## Passo 4: Pusha su GitHub

```bash
git add .
git commit -m "feat: improvements"
git push origin main
```

---

## Uso quotidiano

Dopo il setup iniziale, hai solo 2 comandi da ricordare:

| Cosa vuoi fare                    | Comando                              |
|-----------------------------------|--------------------------------------|
| **Aggiornare dati** (dopo ogni GP)  | `python commands.py` → opzione 26    |
| **Prevedere un weekend**            | `python commands.py` → opzione 10    |

### Previsioni weekend (comando 10)

Ti chiede anno, round, e se usare i dati FP:
- **Con FP** (default): usa i dati delle prove libere per previsioni migliori
- **Senza FP**: usa solo storico + Elo (utile prima del weekend)

Produce: qualifica prevista + gara prevista + confidence + PDF.

### Aggiorna stagione (comando 30)

Da usare dopo ogni weekend di gara per aggiornare tutto.
Un solo comando, fa tutto automaticamente.

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'fastf1'"**
→ L'ambiente virtuale non è attivo. Fai `source venv/bin/activate`.

**"No data available for round X"**
→ Quella gara non si è ancora corsa. Normale per le ultime gare della stagione.

**I test falliscono**
→ Assicurati di essere nella cartella `f1-predictor/` quando esegui pytest.