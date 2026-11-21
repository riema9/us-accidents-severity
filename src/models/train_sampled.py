# src/models/train_sampled.py
"""Train on a random sample of the full dataset with a proper train/val/test split.

Optuna tunes on the validation set only. The test set is touched once, at the end,
and saved to data/processed/ so evaluate.py can reproduce the numbers.

Examples (run from the project root):
    python src/models/train_sampled.py                      # 1M rows, random split
    python src/models/train_sampled.py --split time         # train on early years, test on latest
    python src/models/train_sampled.py --rows 200000 --trials 5   # quick run
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, TARGET_COL
from src.data.clean import clean_data
from src.data.sample import load_sample
from src.features.build import build_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ["Severity 1", "Severity 2", "Severity 3", "Severity 4"]
MODEL_PATH = MODELS_DIR / "sampled_xgb.json"
META_PATH = MODELS_DIR / "sampled_meta.json"
TEST_PATH = PROCESSED_DATA_DIR / "test_sampled.parquet"


def split_data(X, y, start_time, mode):
    """70/15/15 split. 'random' is stratified; 'time' trains on the earliest crashes
    and tests on the most recent ones (checks robustness to drift over the years)."""
    if mode == "time":
        order = start_time.loc[X.index].sort_values().index
        n = len(order)
        tr, va, te = order[: int(0.70 * n)], order[int(0.70 * n): int(0.85 * n)], order[int(0.85 * n):]
        return X.loc[tr], X.loc[va], X.loc[te], y.loc[tr], y.loc[va], y.loc[te]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.15 / 0.85, random_state=RANDOM_SEED, stratify=y_tr
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def _acc_macro_f1(y, pred):
    cm = np.bincount(y * 4 + pred, minlength=16).reshape(4, 4)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return tp.sum() / cm.sum(), f1.mean()


def tune_class_weights(proba, y, min_acc, seed=RANDOM_SEED):
    """Find per-class multipliers for the predicted probabilities (decision-threshold
    tuning). Picks the weights with the best macro-F1 among those whose accuracy is at
    least `min_acc`; if none reach it, falls back to the most accurate. Severity 2 is
    fixed at 1. Fitted on the validation set only."""
    y = np.asarray(y, dtype=int)
    rng = np.random.RandomState(seed)

    def key(w):
        acc, f1 = _acc_macro_f1(y, np.argmax(proba * w, axis=1))
        return (1, f1) if acc >= min_acc else (0, acc)

    best_w = np.ones(4)
    best_k = key(best_w)
    for lo, hi, n in ((np.log(0.05), np.log(20.0), 4000), (-0.4, 0.4, 2000)):
        for _ in range(n):
            if hi < 0.5 and lo < 0:  # refinement: jitter around the current best
                w = best_w * np.exp(rng.uniform(lo, hi, 4))
            else:
                w = np.exp(rng.uniform(lo, hi, 4))
            w = w / w[1]
            k = key(w)
            if k > best_k:
                best_w, best_k = w, k
    return best_w, best_k[0] == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000, help="sample size drawn from the full CSV")
    ap.add_argument("--trials", type=int, default=10, help="Optuna trials")
    ap.add_argument("--tune-rows", type=int, default=200_000, help="train rows used during tuning (speed)")
    ap.add_argument("--split", choices=["random", "time"], default="random")
    ap.add_argument("--refresh", action="store_true", help="rebuild the cached sample")
    ap.add_argument("--geo", action="store_true", help="also use Start_Lat/Start_Lng as features")
    ap.add_argument("--min-acc", type=float, default=0.70, help="accuracy floor for class-weight tuning")
    args = ap.parse_args()

    n_jobs = max(1, (os.cpu_count() or 2) // 2)  # leave some CPU free so the laptop stays usable

    df = load_sample(args.rows, refresh=args.refresh)
    df = clean_data(df)
    start_time = df["Start_Time"].copy()  # build_features drops it; needed for the time split
    geo = df[["Start_Lat", "Start_Lng"]].copy()  # build_features drops these too
    df = build_features(df)
    if args.geo:
        df = df.join(geo)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int) - 1
    logger.info("Class distribution (full sample):\n" + y.map(dict(enumerate(CLASS_NAMES))).value_counts().to_string())

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, start_time, args.split)
    logger.info(f"Split ({args.split}): train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}")
    if set(y_tr.unique()) != {0, 1, 2, 3}:
        raise RuntimeError(f"Training set is missing a severity class: {sorted(y_tr.unique())}")

    classes = np.arange(4)
    cw = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y_tr)))
    w_tr = y_tr.map(cw).to_numpy()

    # tune on a subsample of train for speed; final model uses all of train
    tune_idx = np.random.RandomState(RANDOM_SEED).permutation(len(X_tr))[: args.tune_rows]
    X_tune, y_tune, w_tune = X_tr.iloc[tune_idx], y_tr.iloc[tune_idx], w_tr[tune_idx]

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": "multi:softprob", "num_class": 4, "tree_method": "hist",
            "random_state": RANDOM_SEED, "n_jobs": n_jobs,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_tune, y_tune, sample_weight=w_tune)
        return f1_score(y_va, m.predict(X_va), average="macro")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    logger.info(f"Tuning: {args.trials} trials on {len(X_tune):,} rows (metric: validation macro-F1)")
    study.optimize(objective, n_trials=args.trials)
    logger.info(f"Best validation macro-F1: {study.best_value:.4f}  params: {study.best_params}")

    best = dict(study.best_params, objective="multi:softprob", num_class=4, tree_method="hist",
                random_state=RANDOM_SEED, n_jobs=n_jobs)
    model = xgb.XGBClassifier(**best)
    model.fit(X_tr, y_tr, sample_weight=w_tr)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    test_df = X_te.copy()
    test_df["target"] = y_te.to_numpy()
    test_df.to_parquet(TEST_PATH, index=False)

    results = {}
    for name, Xs, ys in (("val", X_va, y_va), ("test", X_te, y_te)):
        p = model.predict(Xs)
        results[name] = {"accuracy": float(accuracy_score(ys, p)), "macro_f1": float(f1_score(ys, p, average="macro"))}

    logger.info(f"Tuning class weights on validation set (accuracy floor {args.min_acc:.2f})...")
    w, feasible = tune_class_weights(model.predict_proba(X_va), y_va, args.min_acc)
    np.save(MODELS_DIR / "sampled_class_weights.npy", w)
    tuned = np.argmax(model.predict_proba(X_te) * w, axis=1)
    results["test_tuned"] = {"accuracy": float(accuracy_score(y_te, tuned)),
                             "macro_f1": float(f1_score(y_te, tuned, average="macro")),
                             "class_weights": [float(x) for x in w], "met_accuracy_floor_on_val": bool(feasible)}
    baseline = float((y_te == y_te.mode()[0]).mean())
    META_PATH.write_text(json.dumps({
        "rows": args.rows, "split": args.split, "trials": args.trials, "best_params": best,
        "results": results, "majority_baseline_test_accuracy": baseline,
    }, indent=2, default=float))

    print("\n" + "=" * 50)
    print(f" RESULTS  (split: {args.split}, sample: {args.rows:,} rows)")
    print("=" * 50)
    for name in ("val", "test", "test_tuned"):
        r = results[name]
        print(f"{name:>10}: accuracy={r['accuracy']:.4f}  macro-F1={r['macro_f1']:.4f}")
    print(f"Majority-class baseline accuracy on test: {baseline:.4f}")
    print(f"Tuned class weights {np.round(w, 3).tolist()}  (met {args.min_acc:.2f} accuracy floor on val: {feasible})")
    print("\nTest report, plain argmax (balanced class weights):\n")
    print(classification_report(y_te, model.predict(X_te), target_names=CLASS_NAMES, zero_division=0))
    print("Test report, tuned class weights:\n")
    print(classification_report(y_te, tuned, target_names=CLASS_NAMES, zero_division=0))
    print(f"Saved model -> {MODEL_PATH}\nSaved test set -> {TEST_PATH}\nNext: python src/models/evaluate.py")


if __name__ == "__main__":
    main()
