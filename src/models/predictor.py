# src/models/predictor.py
"""Loads the trained model and answers single predictions for the dashboard."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

from src.config import MODELS_DIR
from src.data.clean import clean_data
from src.data.sample import load_sample
from src.models.train_sampled import CLASS_NAMES, META_PATH, MODEL_PATH, TEST_PATH

WEIGHTS_PATH = MODELS_DIR / "sampled_class_weights.npy"


class Predictor:
    def __init__(self):
        for p in (MODEL_PATH, META_PATH):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found. Run: python src/models/train_sampled.py --geo")
        self.meta = json.loads(META_PATH.read_text())
        self.model = xgb.XGBClassifier()
        self.model.load_model(MODEL_PATH)
        self.features = list(self.model.get_booster().feature_names)
        self.uses_geo = "Start_Lat" in self.features
        self.weights = np.load(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else np.ones(4)
        self.weather_options, self.sun_options = self._category_levels()

    def _category_levels(self):
        """build_features encodes Weather_Condition / Sunrise_Sunset as category codes
        (alphabetical). Rebuild the same mapping from the cached training sample."""
        df = clean_data(load_sample(self.meta["rows"]))
        weather = list(df["Weather_Condition"].astype("category").cat.categories)
        sun = list(df["Sunrise_Sunset"].astype("category").cat.categories)
        return weather, sun

    def predict(self, x: dict) -> dict:
        dow = int(x["day_of_week"])
        row = {
            "Temperature(F)": x["temperature"], "Humidity(%)": x["humidity"],
            "Visibility(mi)": x["visibility"], "Wind_Speed(mph)": x["wind"],
            "Precipitation(in)": np.nan if x.get("precipitation") is None else x["precipitation"],
            "Weather_Condition": self.weather_options.index(x["weather"]) if x["weather"] in self.weather_options else -1,
            "Sunrise_Sunset": self.sun_options.index(x["sunrise_sunset"]) if x["sunrise_sunset"] in self.sun_options else -1,
            "Crossing": int(x["crossing"]), "Junction": int(x["junction"]), "Traffic_Signal": int(x["traffic_signal"]),
            "Hour": int(x["hour"]), "Day_of_Week": dow, "Month": int(x["month"]), "Is_Weekend": int(dow in (5, 6)),
            "Start_Lat": x.get("lat", np.nan), "Start_Lng": x.get("lng", np.nan),
        }
        X = pd.DataFrame([row])[self.features].astype(float)
        proba = self.model.predict_proba(X)[0]
        scored = proba * self.weights
        decision = int(np.argmax(scored))
        return {"proba": [float(p) for p in proba], "decision": decision, "label": CLASS_NAMES[decision],
                "share": float(scored[decision] / scored.sum())}

    def performance(self) -> dict:
        """Per-class metrics and confusion matrices on the saved held-out test set."""
        test = pd.read_parquet(TEST_PATH)
        y = test.pop("target").to_numpy()
        proba = self.model.predict_proba(test[self.features])
        out = {}
        for name, pred in (("plain", np.argmax(proba, axis=1)), ("tuned", np.argmax(proba * self.weights, axis=1))):
            rep = classification_report(y, pred, labels=[0, 1, 2, 3], target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0)
            out[name] = {"report": rep, "confusion": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist()}
        out["n_test"] = int(len(y))
        return out
