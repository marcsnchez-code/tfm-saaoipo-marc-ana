"""
SAOR-Poker | A/B test v4 — Disposición de pago REAL vs CZ ESTIMADA con ruido
=============================================================================
Modelo realista (Opción A): se separa la disposición de pago real del jugador
(DPR, su "verdad" interna) de la estimación que el sistema calcula (CZ_est).

  CZ_est = DPR * (1 + e),   e ~ Uniforme(-ruido, +ruido)

- El sistema DINÁMICO construye su ventana de 5 precios sobre CZ_est (no ve la DPR).
- El sistema ESTÁTICO muestra su ventana fija del catálogo (no usa nada del jugador).
- En AMBOS, el jugador DECIDE comprar según su DPR real (su tolerancia verdadera).

Así, cuando la estimación se desvía, el motor dinámico falla igual que en
producción: a veces ofrece precios demasiado altos (pierde la venta) y a veces
demasiado bajos (deja ingreso sobre la mesa).

La DPR se toma como la CZ del dataset (la tolerancia "verdadera" del jugador).
Precios dinámicos redondeados al alza al entero más cercano.
"""

import numpy as np
import pandas as pd
from scipy import stats
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

N_SEEDS = 40
K = 6.0
WEEKS = 30 / 7
CATALOG = np.array([0.99,1.99,4.99,9.99,19.99,29.99,49.99,79.99,
                    99.99,129.99,149.99,199.99,249.99,299.99])
WINDOW = 5
DYN_MULT = np.array([0.80, 0.90, 1.00, 1.10, 1.20])  # centrada en la CZ
TIER_ORDER = ["NonSpender","Tier1","Tier2","Tier3","Mobys","Whales","Megalodons"]


def prob_matrix(prices, dpr):
    """Probabilidad de compra de cada precio según la DISPOSICIÓN REAL del jugador."""
    d = np.where(dpr <= 0, 1e9, dpr)[:, None]
    p = 1.0 / (1.0 + np.exp(np.clip(K * (prices / d - 1.0), -50, 50)))
    nodpr = dpr <= 0
    if nodpr.any():
        p[nodpr, :] = np.where(prices[nodpr, :] <= CATALOG[0], 0.02, 0.005)
    return p


def dynamic_windows(cz_est):
    """Ventana dinámica centrada en la CZ ESTIMADA, redondeada al alza."""
    w = np.ceil(cz_est[:, None] * DYN_MULT[None, :])
    nocz = cz_est <= 0
    if nocz.any():
        w[nocz, :] = CATALOG[:WINDOW]
    return w


def static_arpu(dpr, opp, rng):
    """Estático: ventana del catálogo que avanza según el precio comprado.
    La compra se decide según la DPR real."""
    n = len(dpr)
    start = np.zeros(n, dtype=int)
    revenue = np.zeros(n)
    max_opp = int(opp.max())
    active = np.arange(max_opp)[None, :] < opp[:, None]
    for t in range(max_opp):
        act = active[:, t]
        if not act.any():
            continue
        idx = start[:, None] + np.arange(WINDOW)[None, :]
        windows = CATALOG[idx]
        p = prob_matrix(windows, dpr)
        p_buy = 1.0 - np.prod(1.0 - p, axis=1)
        buy = (rng.random(n) < p_buy) & act
        if buy.any():
            wsum = p.sum(axis=1); wsum = np.where(wsum == 0, 1, wsum)
            weights = p / wsum[:, None]
            for i in np.where(buy)[0]:
                pos = rng.choice(WINDOW, p=weights[i])
                revenue[i] += windows[i, pos]
                start[i] = min(start[i] + pos, len(CATALOG) - WINDOW)
    return revenue


def dynamic_arpu(dpr, cz_est, opp, rng):
    """Dinámico: ventana centrada en la CZ ESTIMADA; compra según DPR real."""
    windows = dynamic_windows(cz_est)
    p = prob_matrix(windows, dpr)                    # decisión según DPR real
    p_buy = 1.0 - np.prod(1.0 - p, axis=1)
    wsum = p.sum(axis=1); wsum = np.where(wsum == 0, 1, wsum)
    weights = p / wsum[:, None]
    exp_price = (weights * windows).sum(axis=1)
    buys = rng.binomial(opp, p_buy)
    return buys * exp_price


def run_seed(dpr, opp, seg, noise, seed, by_tier=False):
    rng = np.random.default_rng(seed)
    n = len(dpr)
    # CZ estimada = DPR * (1 + error uniforme)
    err = rng.uniform(-noise, noise, size=n)
    cz_est = np.maximum(dpr * (1 + err), 0)

    arm = np.empty(n, dtype="<U1")
    for tier in np.unique(seg):
        ix = np.where(seg == tier)[0].copy(); rng.shuffle(ix)
        arm[ix] = np.where(np.arange(len(ix)) % 2 == 0, "A", "B")

    rev = np.zeros(n)
    a = arm == "A"; b = arm == "B"
    rev[a] = static_arpu(dpr[a], opp[a], rng)
    rev[b] = dynamic_arpu(dpr[b], cz_est[b], opp[b], rng)

    if by_tier:
        out = {}
        for tier in TIER_ORDER:
            m = seg == tier
            out[tier] = (rev[m & a].mean() if (m & a).sum() else 0,
                         rev[m & b].mean() if (m & b).sum() else 0)
        return out
    ba, da = rev[a].mean(), rev[b].mean()
    _, p = stats.mannwhitneyu(rev[b], rev[a], alternative="greater")
    return ba, da, 100*(da-ba)/ba if ba > 0 else 0, p < 0.05


def main():
    pm = pd.read_parquet(DATASETS / "player_master.parquet")
    dpr = pm.cz_final.to_numpy()   # la CZ del dataset es la disposición REAL
    seg = pm.archetype.to_numpy()
    opp = (pm.sessions_per_week * WEEKS).round().clip(lower=1).astype(int).to_numpy()

    print(f"A/B test v4 | DPR real vs CZ estimada con ruido | {len(pm):,} jugadores\n")
    print(f"{'Ruido':>8}{'ARPU base':>11}{'ARPU dyn':>11}{'Uplift':>9}{'IC95%':>16}{'% sig':>7}")
    summary = []
    for noise in [0.0, 0.15, 0.25, 0.40]:
        res = [run_seed(dpr, opp, seg, noise, s) for s in range(N_SEEDS)]
        ba = np.array([r[0] for r in res]); da = np.array([r[1] for r in res])
        up = np.array([r[2] for r in res]); sg = np.array([r[3] for r in res])
        ci = 1.96*up.std()/np.sqrt(N_SEEDS)
        tag = "sin ruido" if noise == 0 else f"±{int(noise*100)}%"
        print(f"{tag:>8}{ba.mean():>11.2f}{da.mean():>11.2f}{up.mean():>8.1f}%{f'[{up.mean()-ci:.1f},{up.mean()+ci:.1f}]':>16}{100*sg.mean():>6.0f}%")
        summary.append({"ruido":tag,"arpu_base":round(ba.mean(),2),"arpu_dyn":round(da.mean(),2),
                        "uplift":round(up.mean(),2),"ic_low":round(up.mean()-ci,2),
                        "ic_high":round(up.mean()+ci,2),"pct_sig":round(100*sg.mean(),1)})
    pd.DataFrame(summary).to_csv(DATASETS / "abtest_v4_sensibilidad_ruido.csv", index=False)
    print("\nGuardado: abtest_v4_sensibilidad_ruido.csv")


if __name__ == "__main__":
    # Rastro de ejecución: fecha, máquina y versión de Python (evidencia de reproducción)
    from datetime import datetime
    import platform
    print(f"[Ejecutado {datetime.now():%Y-%m-%d %H:%M:%S} | {platform.node()} | Python {platform.python_version()}]\n")
    main()
