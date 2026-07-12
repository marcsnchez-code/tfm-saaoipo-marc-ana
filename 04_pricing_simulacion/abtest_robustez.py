"""
SAOR-Poker | A/B test — Pruebas de robustez del resultado principal
====================================================================
Regenera las DOS cifras de robustez que la memoria reporta en la sección 4.6,
reutilizando exactamente la lógica del A/B canónico (`abtest_v4.py`) por import,
de modo que ambas sean trazables y consistentes con el resultado principal
(+70,3 % de uplift, ARPU 64,8 -> 110,2 $, error de estimación de la CZ ±25 %).

Las dos pruebas son:

  1. RESTRICCIÓN AL CATÁLOGO COMÚN  ->  uplift +32,4 % (ARPU ~85,7 $)
     El sistema dinámico NO construye precios libres alrededor de la Comfort
     Zone, sino que se restringe a seleccionar del MISMO catálogo de 14 puntos
     que usa el estático: localiza el punto del catálogo más próximo a la
     tolerancia estimada (CZ_est) y muestra la ventana de 5 precios CONSECUTIVOS
     del catálogo CENTRADA en ese punto. Iguala por completo el conjunto de
     precios disponible para ambos sistemas, neutralizando la objeción de que la
     ventaja del dinámico proceda de usar precios no disponibles para el estático.
     La diferencia entre este +32,4 % y el +70,3 % del dinámico libre cuantifica
     el valor de la personalización FINA del precio frente a la mera selección
     dentro de un catálogo discreto.

  2. CHECK CON cz_raw (CONSERVADURISMO DE cz_final)  ->  uplift +74,3 %
     Repite el A/B principal (±25 %) tomando como disposición de pago real (DPR)
     la `cz_raw` (SIN decaimiento temporal) en lugar de la `cz_final` (CON
     decaimiento). Al no penalizar la inactividad, las tolerancias son mayores y
     el uplift sube. Confirma que el uso de `cz_final` en el resultado principal
     es la elección CONSERVADORA: modera el uplift en lugar de inflarlo.

Reutiliza de abtest_v4.py: CATALOG, WINDOW, DYN_MULT, K, WEEKS, prob_matrix,
static_arpu, dynamic_windows y la mecánica de asignación estratificada por tier.

Uso:
    python abtest_robustez.py
Determinista: 40 semillas (0..39), idénticas al A/B canónico.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# Reutilizamos la lógica canónica del A/B v4 (misma ubicación de datos y modelo)
import abtest_v4 as ab

# Rutas robustas relativas a la ubicación del script (independientes del CWD)
try:
    # Ruta robusta cuando se ejecuta como script (.py)
    BASE = Path(__file__).resolve().parent
except NameError:
    # __file__ no existe en Jupyter: se asume que el notebook se ejecuta
    # desde su propia carpeta (convención del paquete de entregables)
    BASE = Path.cwd()
DATASETS = BASE.parent / "05_datasets"

N_SEEDS = ab.N_SEEDS          # 40, igual que el A/B canónico
CATALOG = ab.CATALOG
WINDOW = ab.WINDOW
HALF = WINDOW // 2
NOISE_MAIN = 0.25             # supuesto principal de error de estimación (±25 %)


# ---------------------------------------------------------------------------
# PRUEBA 1 — Sistema dinámico RESTRINGIDO al catálogo común
# ---------------------------------------------------------------------------

def catalog_windows(cz_est):
    """
    Ventana de 5 precios CONSECUTIVOS del catálogo, centrada en el punto del
    catálogo más próximo a la CZ estimada. Replica la variante descrita en la
    memoria (sección 4.6): el dinámico elige del mismo catálogo que el estático.

    - Se localiza el índice del punto de catálogo más cercano a cz_est.
    - Se toma la ventana de 5 puntos centrada en él (2 por debajo, 2 por encima),
      recortada a los extremos del catálogo para no salirse del rango.
    - Para cz_est <= 0 (jugador sin tolerancia estimada) se ofrece la ventana
      base del catálogo, igual que en dynamic_windows de abtest_v4.
    """
    idx_near = np.abs(CATALOG[None, :] - cz_est[:, None]).argmin(axis=1)
    start = np.clip(idx_near - HALF, 0, len(CATALOG) - WINDOW)
    cols = start[:, None] + np.arange(WINDOW)[None, :]
    w = CATALOG[cols]
    nocz = cz_est <= 0
    if nocz.any():
        w[nocz, :] = CATALOG[:WINDOW]
    return w


def dynamic_arpu_catalog(dpr, cz_est, opp, rng):
    """Dinámico restringido al catálogo; la compra se decide según la DPR real."""
    windows = catalog_windows(cz_est)
    p = ab.prob_matrix(windows, dpr)                 # decisión según DPR real
    p_buy = 1.0 - np.prod(1.0 - p, axis=1)
    wsum = p.sum(axis=1); wsum = np.where(wsum == 0, 1, wsum)
    weights = p / wsum[:, None]
    exp_price = (weights * windows).sum(axis=1)
    buys = rng.binomial(opp, p_buy)
    return buys * exp_price


def run_seed_catalog(dpr, opp, seg, noise, seed):
    """Una réplica del A/B con el dinámico restringido al catálogo común.
    Reproduce la asignación estratificada por tier de abtest_v4.run_seed."""
    rng = np.random.default_rng(seed)
    n = len(dpr)
    err = rng.uniform(-noise, noise, size=n)
    cz_est = np.maximum(dpr * (1 + err), 0)

    arm = np.empty(n, dtype="<U1")
    for tier in np.unique(seg):
        ix = np.where(seg == tier)[0].copy(); rng.shuffle(ix)
        arm[ix] = np.where(np.arange(len(ix)) % 2 == 0, "A", "B")

    rev = np.zeros(n)
    a = arm == "A"; b = arm == "B"
    rev[a] = ab.static_arpu(dpr[a], opp[a], rng)
    rev[b] = dynamic_arpu_catalog(dpr[b], cz_est[b], opp[b], rng)

    ba, da = rev[a].mean(), rev[b].mean()
    _, p = stats.mannwhitneyu(rev[b], rev[a], alternative="greater")
    return ba, da, 100 * (da - ba) / ba if ba > 0 else 0, p < 0.05


# ---------------------------------------------------------------------------
# ORQUESTACIÓN
# ---------------------------------------------------------------------------

def _aggregate(res):
    """Promedia base/dyn/uplift/%sig e IC95% sobre las réplicas."""
    ba = np.array([r[0] for r in res]); da = np.array([r[1] for r in res])
    up = np.array([r[2] for r in res]); sg = np.array([r[3] for r in res])
    ci = 1.96 * up.std() / np.sqrt(len(res))
    return (round(ba.mean(), 2), round(da.mean(), 2), round(up.mean(), 2),
            round(up.mean() - ci, 2), round(up.mean() + ci, 2), round(100 * sg.mean(), 1))


def main():
    pm = pd.read_parquet(DATASETS / "player_master.parquet")
    seg = pm.archetype.to_numpy()
    opp = (pm.sessions_per_week * ab.WEEKS).round().clip(lower=1).astype(int).to_numpy()

    dpr_final = pm.cz_final.to_numpy()   # DPR canónica (con decaimiento) -> conservadora
    dpr_raw = pm.cz_raw.to_numpy()       # DPR sin decaimiento -> robustez al alza

    print(f"A/B test — Pruebas de robustez | {len(pm):,} jugadores | {N_SEEDS} semillas\n")
    rows = []

    # --- PRUEBA 1: catálogo común (DPR = cz_final, ruido ±25 %) ---
    res1 = [run_seed_catalog(dpr_final, opp, seg, NOISE_MAIN, s) for s in range(N_SEEDS)]
    ba, da, up, lo, hi, sig = _aggregate(res1)
    print("[1] Dinámico RESTRINGIDO al catálogo común (igualdad total de precios)")
    print(f"    ARPU {ba} -> {da} $ | uplift {up:.1f}% | IC95% [{lo:.1f},{hi:.1f}] | sig {sig:.0f}%")
    print("    (memoria 4.6: ARPU ~85,7 $, +32,4 %)\n")
    rows.append({"prueba": "catalogo_comun", "dpr": "cz_final", "ruido": "±25%",
                 "arpu_base": ba, "arpu_dyn": da, "uplift": up,
                 "ic_low": lo, "ic_high": hi, "pct_sig": sig})

    # --- PRUEBA 2: check cz_raw (DPR = cz_raw, dinámico LIBRE, ruido ±25 %) ---
    res2 = [ab.run_seed(dpr_raw, opp, seg, NOISE_MAIN, s) for s in range(N_SEEDS)]
    ba, da, up, lo, hi, sig = _aggregate(res2)
    print("[2] Check con cz_raw (sin decaimiento) — confirma que cz_final es conservador")
    print(f"    ARPU {ba} -> {da} $ | uplift {up:.1f}% | IC95% [{lo:.1f},{hi:.1f}] | sig {sig:.0f}%")
    print("    (referencia: +74,3 % reproducible, frente al +70,3 % con cz_final)\n")
    rows.append({"prueba": "check_cz_raw", "dpr": "cz_raw", "ruido": "±25%",
                 "arpu_base": ba, "arpu_dyn": da, "uplift": up,
                 "ic_low": lo, "ic_high": hi, "pct_sig": sig})

    out = DATASETS / "abtest_robustez.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Guardado: {out.name}")


if __name__ == "__main__":
    # Rastro de ejecución: fecha, máquina y versión de Python (evidencia de reproducción)
    from datetime import datetime
    import platform
    print(f"[Ejecutado {datetime.now():%Y-%m-%d %H:%M:%S} | {platform.node()} | Python {platform.python_version()}]\n")
    main()
