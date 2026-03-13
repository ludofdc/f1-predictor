"""
model.py — Modello Predittivo per i Risultati F1
=================================================

AGGIORNATO: Ora il modello rileva automaticamente se le feature avanzate
sono disponibili e le usa. Se non ci sono, usa le feature base.

Questo approccio si chiama "graceful degradation":
il sistema funziona sempre, ma migliora quando ha più dati.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import sys

sys.path.append("..")
from config import RANDOM_SEED, TEST_SIZE, PROCESSED_DATA_DIR
from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN

# Proviamo a importare le feature avanzate
try:
    from src.advanced_features import ADVANCED_FEATURE_COLUMNS
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False


class F1Predictor:
    """
    Il modello predittivo per la F1.

    Rileva automaticamente se le feature avanzate sono disponibili:
    - Se sì: usa ~40 feature (meteo, gomme, circuito, strategia)
    - Se no: usa ~10 feature base (Elo, griglia, forma recente)
    """

    def __init__(self, model_type: str = "random_forest", use_advanced: bool = True):
        """
        Parametri:
            model_type: "random_forest" o "gradient_boosting"
            use_advanced: se True, usa feature avanzate quando disponibili
        """
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                random_state=RANDOM_SEED,
            )
        else:
            raise ValueError(f"Modello '{model_type}' non supportato")

        self.model_type = model_type
        self.use_advanced = use_advanced
        self.feature_columns = None
        self.is_trained = False

    def _detect_features(self, df: pd.DataFrame) -> list:
        """
        Rileva automaticamente quali feature sono disponibili nel DataFrame.

        1. Se use_advanced=True E le colonne avanzate esistono → usa quelle
        2. Altrimenti → usa le feature base
        3. Rimuove feature con troppi NaN (>50%)
        """
        if self.use_advanced and ADVANCED_AVAILABLE:
            available = [c for c in ADVANCED_FEATURE_COLUMNS if c in df.columns]

            if len(available) > len(FEATURE_COLUMNS):
                good_features = []
                for col in available:
                    non_null_ratio = df[col].notna().mean()
                    if non_null_ratio >= 0.5:
                        good_features.append(col)

                if len(good_features) > len(FEATURE_COLUMNS):
                    print(f"   🚀 Feature avanzate rilevate: {len(good_features)} feature")
                    return good_features

        available_base = [c for c in FEATURE_COLUMNS if c in df.columns]
        print(f"   📊 Feature base: {len(available_base)} feature")
        return available_base

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepara i dati. Rileva automaticamente le feature disponibili."""
        if self.feature_columns is None:
            self.feature_columns = self._detect_features(df)

        clean_df = df.dropna(subset=[TARGET_COLUMN])

        X = clean_df[self.feature_columns].copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        y = clean_df[TARGET_COLUMN]

        return X, y, clean_df

    def train(self, df: pd.DataFrame) -> dict:
        """Addestra il modello con cross-validation temporale."""
        print("🏋️ Training del modello...")

        X, y, clean_df = self.prepare_data(df)
        print(f"   Dati: {len(X)} righe, {len(self.feature_columns)} feature")

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            temp_model = type(self.model)(**self.model.get_params())
            temp_model.fit(X_train, y_train)

            predictions = temp_model.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)
            cv_scores.append(mae)
            print(f"   Fold {fold+1}: MAE = {mae:.2f} posizioni")

        print(f"   📊 Media CV: MAE = {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

        self.model.fit(X, y)
        self.is_trained = True

        importance = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        print(f"\n   🔍 Feature Importance (top 15):")
        for i, (_, row) in enumerate(importance.head(15).iterrows()):
            bar = "█" * int(row["importance"] * 50)
            print(f"   {row['feature']:35s} {bar} {row['importance']:.3f}")

        remaining = len(importance) - 15
        if remaining > 0:
            remaining_imp = importance.iloc[15:]["importance"].sum()
            print(f"   {'... altre ' + str(remaining) + ' feature':35s} {'░' * int(remaining_imp * 50)} {remaining_imp:.3f}")

        return {
            "cv_mae_mean": np.mean(cv_scores),
            "cv_mae_std": np.std(cv_scores),
            "cv_scores": cv_scores,
            "feature_importance": importance,
            "num_features": len(self.feature_columns),
            "features_used": self.feature_columns,
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fa previsioni su nuovi dati."""
        if not self.is_trained:
            raise RuntimeError("Il modello non è ancora stato addestrato! Chiama train() prima.")

        X, _, clean_df = self.prepare_data(df)
        predictions = self.model.predict(X)

        result = clean_df.copy()
        result["predicted_position"] = predictions

        return result

    def predict_race(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prevede e ordina i risultati di una singola gara."""
        predictions = self.predict(df)
        predictions = predictions.sort_values("predicted_position").reset_index(drop=True)
        predictions["predicted_rank"] = range(1, len(predictions) + 1)
        return predictions

    def compare_base_vs_advanced(self, df: pd.DataFrame) -> dict:
        """
        Confronta il modello base con quello avanzato.
        Utile per capire quanto valore aggiungono le feature extra.
        """
        print("\n" + "=" * 60)
        print("⚔️ CONFRONTO: Feature Base vs Feature Avanzate")
        print("=" * 60)

        print("\n--- Modello BASE (10 feature) ---")
        base_model = F1Predictor(model_type=self.model_type, use_advanced=False)
        base_results = base_model.train(df)

        print("\n--- Modello AVANZATO (40+ feature) ---")
        adv_model = F1Predictor(model_type=self.model_type, use_advanced=True)
        adv_results = adv_model.train(df)

        base_mae = base_results["cv_mae_mean"]
        adv_mae = adv_results["cv_mae_mean"]
        improvement = (base_mae - adv_mae) / base_mae * 100

        print(f"\n📊 RISULTATO:")
        print(f"   Base MAE:     {base_mae:.3f} posizioni ({base_results['num_features']} feature)")
        print(f"   Avanzato MAE: {adv_mae:.3f} posizioni ({adv_results['num_features']} feature)")
        if improvement > 0:
            print(f"   ✅ Miglioramento: {improvement:.1f}%")
        else:
            print(f"   ⚠️ Nessun miglioramento ({improvement:.1f}%)")
            print(f"   Le feature extra potrebbero aggiungere rumore.")

        return {
            "base_mae": base_mae,
            "advanced_mae": adv_mae,
            "improvement_pct": improvement,
        }

    def save(self, filepath: str = None):
        """Salva il modello e le feature usate."""
        if filepath is None:
            filepath = str(PROCESSED_DATA_DIR / "f1_model.joblib")

        save_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type,
        }
        joblib.dump(save_data, filepath)
        print(f"💾 Modello salvato in {filepath}")

    def load(self, filepath: str = None):
        """Carica un modello salvato."""
        if filepath is None:
            filepath = str(PROCESSED_DATA_DIR / "f1_model.joblib")

        save_data = joblib.load(filepath)
        self.model = save_data["model"]
        self.feature_columns = save_data["feature_columns"]
        self.model_type = save_data.get("model_type", "random_forest")
        self.is_trained = True
        print(f"📂 Modello caricato da {filepath}")
        print(f"   Feature: {len(self.feature_columns)}")


if __name__ == "__main__":
    advanced_file = PROCESSED_DATA_DIR / "advanced_features.csv"
    base_file = PROCESSED_DATA_DIR / "features.csv"

    if advanced_file.exists():
        print("📂 Caricamento feature AVANZATE...")
        df = pd.read_csv(advanced_file)
    elif base_file.exists():
        print("📂 Caricamento feature base...")
        df = pd.read_csv(base_file)
    else:
        print("❌ Nessun file feature trovato!")
        print("   Esegui prima: python -m src.feature_engineering")
        print("   Oppure:       python -m src.advanced_features")
        sys.exit(1)

    # Crea e allena
    predictor = F1Predictor(model_type="random_forest")
    results = predictor.train(df)

    # Se ci sono feature avanzate, confronta base vs avanzato
    if advanced_file.exists() and len(df.columns) > 20:
        predictor.compare_base_vs_advanced(df)

    # Previsione sull'ultima gara
    last_year = df["year"].max()
    last_round = df[df["year"] == last_year]["round"].max()
    last_race = df[(df["year"] == last_year) & (df["round"] == last_round)]

    print(f"\n🏁 Previsione per l'ultima gara ({last_race['race_name'].iloc[0]}):")
    prediction = predictor.predict_race(last_race)

    display_cols = [
        "predicted_rank", "driver", "team",
        "grid_position", "predicted_position", "finish_position"
    ]
    available_cols = [c for c in display_cols if c in prediction.columns]
    print(prediction[available_cols].to_string(index=False))

    predictor.save()