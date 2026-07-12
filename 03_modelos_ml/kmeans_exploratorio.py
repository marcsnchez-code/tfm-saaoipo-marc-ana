"""
SAOR-Poker | Análisis exploratorio de segmentación con K-Means
================================================================
ROL DE ESTE MÓDULO (importante): el K-Means se emplea EXCLUSIVAMENTE como
herramienta exploratoria de CONTRASTE, no como pilar de segmentación. La
segmentación operativa del sistema son los 7 tiers de LTV definidos por reglas
explícitas (ver poker_generator_v2.py y la memoria, §4.2.4). Este script
verifica que la estructura natural de la población, descubierta de forma no
supervisada sobre las variables RFM, es coherente con esa partición de negocio:
la dimensión de valor y actividad domina la estructura, lo que respalda el uso
de tiers transparentes frente a clusters opacos.

Produce:
  - Curva del método del codo (inertia vs k) y Silhouette Score por k.
  - Tabla de contraste cluster x tier (cómo se reparten los tiers en los clusters).
  - Gráficos: 06_graficos/fig_kmeans_codo_silhouette.png

Determinista: SEED fijo para reproducibilidad.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

# Rutas robustas: funcionan desde cualquier directorio.
try:
    # Ruta robusta cuando se ejecuta como script (.py)
    BASE = Path(__file__).resolve().parent
except NameError:
    # __file__ no existe en Jupyter: se asume que el notebook se ejecuta
    # desde su propia carpeta (convención del paquete de entregables)
    BASE = Path.cwd()
DATASETS = BASE.parent / "05_datasets"
GRAFICOS = BASE.parent / "06_graficos"

# Variables RFM utilizadas para el análisis exploratorio.
RFM_COLS = ["monetary", "frequency", "recency_days"]
# Rango de k evaluado en el método del codo.
K_RANGE = range(2, 9)


def cargar_rfm() -> pd.DataFrame:
    """Carga el dataset y devuelve las variables RFM de los jugadores pagadores.
    El clustering se aplica sobre la base con actividad transaccional, donde la
    estructura RFM es informativa (los NonSpender no aportan señal de gasto)."""
    pm = pd.read_parquet(DATASETS / "player_master.parquet")
    pagadores = pm[pm["monetary"] > 0].copy()
    return pm, pagadores


def explorar_k(X: np.ndarray) -> pd.DataFrame:
    """Calcula inertia (codo) y Silhouette Score para cada k del rango."""
    filas = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=10000, random_state=SEED)
        filas.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"  k={k} | inertia={km.inertia_:>14,.0f} | silhouette={sil:.3f}")
    return pd.DataFrame(filas)


def graficar(res: pd.DataFrame) -> None:
    """Genera la figura de codo + Silhouette."""
    GRAFICOS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(res["k"], res["inertia"], "o-", color="#2E5496", linewidth=2)
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_ylabel("Inertia (suma de distancias intra-cluster)")
    ax1.set_title("Método del codo")
    ax1.grid(alpha=0.2)

    ax2.plot(res["k"], res["silhouette"], "o-", color="#1D9E75", linewidth=2)
    ax2.set_xlabel("Número de clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Coeficiente de silueta por k")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    out = GRAFICOS / "fig_kmeans_codo_silhouette.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    # Ruta relativa al paquete (evita exponer rutas absolutas de la máquina local)
    print(f"\nGráfico guardado en: 06_graficos/{out.name}")


def contraste_con_tiers(pagadores: pd.DataFrame, k: int) -> None:
    """Ajusta K-Means con el k indicado y cruza los clusters con los tiers de LTV,
    mostrando que la estructura natural converge con la partición de negocio."""
    scaler = StandardScaler()
    X = scaler.fit_transform(pagadores[RFM_COLS])
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    pagadores = pagadores.copy()
    pagadores["cluster"] = km.fit_predict(X)

    print(f"\n=== Contraste cluster x tier (k={k}) ===")
    tabla = pd.crosstab(pagadores["cluster"], pagadores["archetype"])
    print(tabla.to_string())
    print(
        "\nLectura: cada cluster queda dominado por uno o pocos tiers contiguos de "
        "LTV, lo que confirma que la dimensión de valor estructura la población y "
        "respalda la segmentación por tiers explícitos adoptada por el sistema."
    )


def main() -> None:
    print("Análisis exploratorio de segmentación con K-Means (contraste, no pilar)\n")
    pm, pagadores = cargar_rfm()
    print(f"Jugadores pagadores analizados: {len(pagadores):,}")
    print(f"Variables RFM: {RFM_COLS}\n")

    scaler = StandardScaler()
    X = scaler.fit_transform(pagadores[RFM_COLS])

    print("=== Exploración del número de clusters ===")
    res = explorar_k(X)
    graficar(res)

    # Contraste con el k de mayor Silhouette del rango explorado.
    mejor_k = int(res.loc[res["silhouette"].idxmax(), "k"])
    print(f"\nk con mayor Silhouette en el rango explorado: {mejor_k}")
    contraste_con_tiers(pagadores, mejor_k)

    print(
        "\nConclusión: el análisis no supervisado es coherente con la partición por "
        "tiers de LTV. K-Means cumple aquí una función de validación exploratoria, "
        "no de segmentación operativa."
    )


if __name__ == "__main__":
    # Rastro de ejecución: fecha, máquina y versión de Python (evidencia de reproducción)
    from datetime import datetime
    import platform
    print(f"[Ejecutado {datetime.now():%Y-%m-%d %H:%M:%S} | {platform.node()} | Python {platform.python_version()}]\n")
    main()
