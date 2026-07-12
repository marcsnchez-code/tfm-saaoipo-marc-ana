"""
SAOR-Poker | Comparación de modelos de propensión: Random Forest vs XGBoost
============================================================================
Entrena ambos modelos sobre el MISMO dataset, con las MISMAS variables de
comportamiento (sin variables de gasto, para evitar fuga de datos) y la MISMA
partición train/test. Compara su capacidad predictiva mediante AUC y otras
métricas, con validación cruzada para robustez.

Variable objetivo: converted (1 = el jugador compró, 0 = no).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report,
                             precision_score, recall_score, f1_score, roc_curve)
from xgboost import XGBClassifier
from pathlib import Path

# Rutas robustas relativas a la ubicación del script (independientes del CWD)
try:
    # Ruta robusta cuando se ejecuta como script (.py)
    BASE = Path(__file__).resolve().parent
except NameError:
    # __file__ no existe en Jupyter: se asume que el notebook se ejecuta
    # desde su propia carpeta (convención del paquete de entregables)
    BASE = Path.cwd()
DATASETS = BASE.parent / "05_datasets"

SEED = 42


def load_features():
    players = pd.read_parquet(DATASETS / "player_master.parquet")
    # Variables de COMPORTAMIENTO únicamente (sin gasto -> sin fuga de datos)
    X = players[["age", "sessions_per_week", "avg_session_min"]].copy()
    X["engagement_num"] = players["engagement_level"].map({"Low": 0, "Medium": 1, "High": 2})
    y = players["converted"]
    return X, y


def main():
    X, y = load_features()
    print(f"Dataset: {len(X):,} jugadores | tasa de conversión: {y.mean():.1%}")
    print(f"Variables predictoras: {list(X.columns)}\n")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)

    # --- Modelo 1: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, random_state=SEED,
        class_weight="balanced", n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_proba = rf.predict_proba(Xte)[:, 1]
    rf_pred = rf.predict(Xte)

    # --- Modelo 2: XGBoost ---
    # scale_pos_weight maneja el desbalanceo (ratio negativos/positivos)
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, random_state=SEED,
        eval_metric="logloss", n_jobs=-1)
    xgb.fit(Xtr, ytr)
    xgb_proba = xgb.predict_proba(Xte)[:, 1]
    xgb_pred = xgb.predict(Xte)

    # --- Métricas comparadas ---
    def metrics(name, y_true, y_pred, y_proba):
        return {
            "Modelo": name,
            "AUC": round(roc_auc_score(y_true, y_proba), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "F1": round(f1_score(y_true, y_pred), 4),
        }

    results = pd.DataFrame([
        metrics("Random Forest", yte, rf_pred, rf_proba),
        metrics("XGBoost", yte, xgb_pred, xgb_proba),
    ])
    print("=== COMPARACIÓN DE MODELOS (conjunto de test) ===")
    print(results.to_string(index=False))

    # --- Validación cruzada (AUC) para robustez ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rf_cv = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    xgb_cv = cross_val_score(xgb, X, y, cv=cv, scoring="roc_auc")
    print("\n=== AUC EN VALIDACIÓN CRUZADA (5 folds) ===")
    print(f"  Random Forest: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
    print(f"  XGBoost:       {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

    # --- Importancia de variables (ambos modelos) ---
    print("\n=== IMPORTANCIA DE VARIABLES ===")
    imp = pd.DataFrame({
        "variable": X.columns,
        "Random Forest": rf.feature_importances_,
        "XGBoost": xgb.feature_importances_,
    }).round(3)
    print(imp.to_string(index=False))

    # Guardar curvas ROC para el gráfico
    rf_fpr, rf_tpr, _ = roc_curve(yte, rf_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(yte, xgb_proba)
    np.savez("roc_data.npz",
             rf_fpr=rf_fpr, rf_tpr=rf_tpr, rf_auc=roc_auc_score(yte, rf_proba),
             xgb_fpr=xgb_fpr, xgb_tpr=xgb_tpr, xgb_auc=roc_auc_score(yte, xgb_proba))
    results.to_csv(DATASETS / "model_comparison.csv", index=False)
    return results


if __name__ == "__main__":
    # Rastro de ejecución: fecha, máquina y versión de Python (evidencia de reproducción)
    from datetime import datetime
    import platform
    print(f"[Ejecutado {datetime.now():%Y-%m-%d %H:%M:%S} | {platform.node()} | Python {platform.python_version()}]\n")
    main()
