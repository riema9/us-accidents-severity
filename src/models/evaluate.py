# src/models/evaluate.py
"""Evaluate the model saved by train_sampled.py on the held-out test set, and make
a SHAP feature-importance chart.   Run from the project root:
    python src/models/evaluate.py
    python src/models/evaluate.py --no-shap
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import ROOT_DIR
from src.models.train_sampled import CLASS_NAMES, MODEL_PATH, TEST_PATH
from src.config import MODELS_DIR

WEIGHTS_PATH = MODELS_DIR / "sampled_class_weights.npy"

FIG_DIR = ROOT_DIR / "reports" / "figures"


def shap_importance(model, X, n=5000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    Xs = X.sample(min(n, len(X)), random_state=42)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    if isinstance(sv, list):                      # older shap: list of (n, features) per class
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)                           # -> (n, features, classes)
    if sv.shape[1] == len(CLASS_NAMES) and sv.shape[2] == Xs.shape[1]:
        sv = np.transpose(sv, (0, 2, 1))
    per_class = np.abs(sv).mean(axis=0)           # (features, classes)
    order = np.argsort(per_class.sum(axis=1))     # ascending for a horizontal bar chart

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    import json
    feats = [str(f) for f in np.array(Xs.columns)[order]]
    (FIG_DIR / "shap_importance.json").write_text(json.dumps(
        {"features": feats, "classes": CLASS_NAMES, "values": [[float(v) for v in per_class[order, c]] for c in range(len(CLASS_NAMES))]}))
    sev_colors = ["#7BC043", "#F2B705", "#F07B2C", "#D7263D"]
    fig, ax = plt.subplots(figsize=(8, 5))
    left = np.zeros(len(order))
    for c, name in enumerate(CLASS_NAMES):
        ax.barh(feats, per_class[order, c], left=left, label=name, color=sev_colors[c])
        left += per_class[order, c]
    ax.set_xlabel("mean |SHAP value| (summed over classes)")
    ax.set_title("Feature importance (SHAP)", color="#0B1530", loc="left", fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "shap_feature_importance.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved SHAP chart -> {out}")
    top = pd.Series(per_class.sum(axis=1), index=Xs.columns).sort_values(ascending=False)
    print("Top features (mean |SHAP|):\n" + top.head(10).round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    if not MODEL_PATH.exists() or not TEST_PATH.exists():
        raise SystemExit("Run `python src/models/train_sampled.py` first.")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    test = pd.read_parquet(TEST_PATH)
    y = test.pop("target")
    X = test[model.get_booster().feature_names]

    labels = [0, 1, 2, 3]
    proba = model.predict_proba(X)
    runs = [("plain argmax (balanced class weights)", np.argmax(proba, axis=1))]
    if WEIGHTS_PATH.exists():
        runs.append(("tuned class weights " + str(np.round(np.load(WEIGHTS_PATH), 3).tolist()),
                     np.argmax(proba * np.load(WEIGHTS_PATH), axis=1)))

    print(f"Test rows: {len(y):,}")
    print(f"Majority-class baseline accuracy: {(y == y.mode()[0]).mean():.4f}  "
          "(accuracy alone is misleading on imbalanced data)")
    for title, pred in runs:
        print("\n" + "=" * 60 + f"\n {title}\n" + "=" * 60)
        print(f"Accuracy: {accuracy_score(y, pred):.4f}   Macro-F1: {f1_score(y, pred, average='macro'):.4f}"
              f"   Weighted-F1: {f1_score(y, pred, average='weighted'):.4f}\n")
        print(classification_report(y, pred, labels=labels, target_names=CLASS_NAMES, zero_division=0))
        cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels),
                          index=[f"true {n}" for n in CLASS_NAMES], columns=[f"pred {n[-1]}" for n in CLASS_NAMES])
        print("Confusion matrix:\n" + cm.to_string())

    if not args.no_shap:
        shap_importance(model, X)


if __name__ == "__main__":
    main()
