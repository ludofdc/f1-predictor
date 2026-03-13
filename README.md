# 🏎️ F1 Predictor — Formula 1 Race Prediction & Analytics

A data-driven approach to predicting Formula 1 race results using historical data, Elo ratings, and machine learning.

## 🎯 Project Goal

Build a scalable prediction framework trained on 2022–2025 data (ground effect era), designed to:
- Predict race finishing positions from qualifying and historical performance
- Track driver and constructor strength over time using an adapted Elo rating system
- Monitor model performance across the 2026 regulation change as a real-world **regime change** case study

## 📊 Key Features

- **Elo Rating System** — Adapted from chess to F1: tracks driver and team strength, updates after each race
- **Race Prediction Model** — Combines grid position, Elo ratings, circuit history, and recent form
- **Model Monitoring** — Tracks prediction accuracy over time, detects when the model degrades
- **Interactive Dashboard** — Visualize ratings, predictions, and model performance

## 🏗️ Project Structure

```
f1-predictor/
├── README.md
├── requirements.txt
├── SETUP_GUIDE.py
├── config.py                  # Configuration and constants
├── data/
│   ├── raw/                   # Raw data from API (auto-generated)
│   └── processed/             # Cleaned datasets (auto-generated)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Download and cache F1 data
│   ├── feature_engineering.py # Create features for the model
│   ├── elo.py                 # Elo rating system
│   ├── model.py               # Prediction model
│   └── evaluation.py          # Model evaluation metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_elo_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_2026_monitoring.ipynb
├── tests/
│   └── test_elo.py
└── .gitignore
```

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/f1-predictor.git
cd f1-predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data and compute Elo ratings
python3 -m src.data_loader
python3 -m src.elo
```

## 📈 Methodology

### Elo Rating System
Each driver starts with a base rating of 1500. After each race, ratings update based on:
- Finishing position relative to expected performance
- Head-to-head results against other drivers
- K-factor that weights recent results more heavily

### Prediction Model
Features include:
- Grid position (qualifying result)
- Current Elo rating (driver + constructor)
- Circuit-specific historical performance
- Recent form (rolling average of last 5 races)
- DNF probability estimation

### Regime Change Analysis (2026)
The 2026 F1 regulations introduce major aerodynamic and power unit changes. This project tracks:
- Model accuracy degradation in early 2026
- Elo rating volatility during the transition
- Adaptation speed with progressive retraining

## 🛠️ Tech Stack

- **Python 3.10+**
- **fastf1** — Official F1 telemetry and session data
- **pandas / numpy** — Data manipulation
- **scikit-learn** — ML models
- **matplotlib / seaborn / plotly** — Visualization
- **streamlit** — Interactive dashboard (coming soon)

## 📝 License

MIT License — feel free to use, modify, and share.

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.