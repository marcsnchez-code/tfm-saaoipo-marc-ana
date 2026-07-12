"""
SAOR-Poker | A/B test multi-semilla v3 — VECTORIZADO
=====================================================
Versión vectorizada con NumPy del A/B test del motor v3, para ejecutar las
40 semillas en tiempo razonable. Lógica idéntica a la descrita en pricing_engine_v3.

NOTA: su resultado (+75,4 %, guardado en `05_datasets/abtest_v3_resultado.csv`)
es HISTÓRICO: corresponde a la iteración v3, SIN modelo de error de estimación
de la CZ, y queda superado por el resultado canónico de `abtest_v4.py` (+70,3 %
con error ±25 %). Se conserva en el paquete como trazabilidad del desarrollo.
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
DYN_MULT = np.array([0.80, 0.90, 1.00, 1.10, 1.20])  # ventana centrada en la CZ (-20/-10/CZ/+10/+20)
TIER_ORDER = ["NonSpender","Tier1","Tier2","Tier3","Mobys","Whales","Megalodons"]


def prob_matrix(prices, cz):
    """prices: (n,5) ventana por jugador. cz: (n,). Devuelve prob (n,5)."""
    czc = np.where(cz <= 0, 1e9, cz)[:, None]
    ratio = prices / czc
    p = 1.0 / (1.0 + np.exp(np.clip(K * (ratio - 1.0), -50, 50)))
    # Jugadores con CZ<=0: prob casi nula salvo precio mínimo
    nocz = cz <= 0
    if nocz.any():
        p[nocz, :] = np.where(prices[nocz, :] <= CATALOG[0], 0.02, 0.005)
    return p


def expected_revenue(prices, cz):
    """Ingreso esperado por oportunidad. prices (n,5), cz (n,). Devuelve (n,)."""
    p = prob_matrix(prices, cz)              # (n,5)
    p_buy = 1.0 - np.prod(1.0 - p, axis=1)   # (n,)
    wsum = p.sum(axis=1)
    wsum = np.where(wsum == 0, 1, wsum)
    weights = p / wsum[:, None]
    exp_price = (weights * prices).sum(axis=1)
    return p_buy * exp_price, p_buy


def dynamic_windows(cz):
    """(n,5) ventana dinámica centrada en la CZ, con precios redondeados al alza
    al entero más cercano (1.20 -> 2, 4.87 -> 5)."""
    w = cz[:, None] * DYN_MULT[None, :]
    w = np.ceil(w)                       # redondeo HACIA ARRIBA al entero
    nocz = cz <= 0
    if nocz.any():
        w[nocz, :] = CATALOG[:WINDOW]
    return w


def static_arpu_progressive(cz, opp, rng):
    """
    ARPU del sistema estático con ventana que avanza SEGÚN EL PRECIO COMPRADO:
    tras una compra, la ventana se recoloca para que el precio comprado sea el
    más bajo visible. Comprar alto hace saltar la ventana varios peldaños.
    """
    n = len(cz)
    start = np.zeros(n, dtype=int)        # índice de inicio de la ventana
    revenue = np.zeros(n)
    max_opp = int(opp.max())
    active_mask = np.arange(max_opp)[None, :] < opp[:, None]

    for t in range(max_opp):
        active = active_mask[:, t]
        if not active.any():
            continue
        idx = start[:, None] + np.arange(WINDOW)[None, :]
        windows = CATALOG[idx]                       # (n,5)
        p = prob_matrix(windows, cz)
        p_buy = 1.0 - np.prod(1.0 - p, axis=1)
        buy = (rng.random(n) < p_buy) & active
        if buy.any():
            wsum = p.sum(axis=1); wsum = np.where(wsum == 0, 1, wsum)
            weights = p / wsum[:, None]
            for i in np.where(buy)[0]:
                # posición elegida dentro de la ventana (0..4)
                pos = rng.choice(WINDOW, p=weights[i])
                revenue[i] += windows[i, pos]
                # la ventana se recoloca: el precio comprado pasa a ser el más bajo
                new_start = start[i] + pos
                start[i] = min(new_start, len(CATALOG) - WINDOW)
    return revenue


def dynamic_arpu(cz, opp, rng):
    """ARPU del sistema dinámico: ventana fija alrededor de la CZ."""
    n = len(cz)
    windows = dynamic_windows(cz)
    er_per_opp, _ = expected_revenue(windows, cz)
    # ingreso esperado total = er por oportunidad * nº oportunidades
    # (la ventana no cambia, así que es determinista en esperanza; añadimos ruido binomial)
    p = prob_matrix(windows, cz)
    p_buy = 1.0 - np.prod(1.0 - p, axis=1)
    wsum = p.sum(axis=1); wsum = np.where(wsum == 0, 1, wsum)
    weights = p / wsum[:, None]
    exp_price = (weights * windows).sum(axis=1)
    # simular nº de compras ~ Binomial(opp, p_buy)
    buys = rng.binomial(opp, p_buy)
    return buys * exp_price


def run_seed(cz, opp, seg, seed, by_tier=False):
    rng = np.random.default_rng(seed)
    n = len(cz)
    arm = np.empty(n, dtype="<U1")
    for tier in np.unique(seg):
        ix = np.where(seg == tier)[0].copy()
        rng.shuffle(ix)
        arm[ix] = np.where(np.arange(len(ix)) % 2 == 0, "A", "B")

    rev = np.zeros(n)
    a_mask = arm == "A"; b_mask = arm == "B"
    rev[a_mask] = static_arpu_progressive(cz[a_mask], opp[a_mask], rng)
    rev[b_mask] = dynamic_arpu(cz[b_mask], opp[b_mask], rng)

    if by_tier:
        out = {}
        for tier in TIER_ORDER:
            m = seg == tier
            out[tier] = (rev[m & a_mask].mean() if (m & a_mask).sum() else 0,
                         rev[m & b_mask].mean() if (m & b_mask).sum() else 0)
        return out

    base = rev[a_mask]; dyn = rev[b_mask]
    ba, da = base.mean(), dyn.mean()
    _, p = stats.mannwhitneyu(dyn, base, alternative="greater")
    return ba, da, 100*(da-ba)/ba if ba > 0 else 0, p < 0.05


def main():
    pm = pd.read_parquet(DATASETS / "player_master.parquet")
    cz = pm.cz_final.to_numpy()
    seg = pm.archetype.to_numpy()
    opp = (pm.sessions_per_week * WEEKS).round().clip(lower=1).astype(int).to_numpy()

    print(f"A/B test v3 vectorizado | {len(pm):,} jugadores | {N_SEEDS} semillas\n")
    res = [run_seed(cz, opp, seg, s) for s in range(N_SEEDS)]
    ba = np.array([r[0] for r in res]); da = np.array([r[1] for r in res])
    up = np.array([r[2] for r in res]); sg = np.array([r[3] for r in res])
    ci = 1.96*up.std()/np.sqrt(N_SEEDS)

    print("=== RESULTADO GLOBAL ===")
    print(f"  ARPU baseline estático : {ba.mean():.2f} ± {1.96*ba.std()/np.sqrt(N_SEEDS):.2f}")
    print(f"  ARPU dinámico (CZ)     : {da.mean():.2f} ± {1.96*da.std()/np.sqrt(N_SEEDS):.2f}")
    print(f"  UPLIFT                 : {up.mean():.1f}%  IC95% [{up.mean()-ci:.1f}, {up.mean()+ci:.1f}]")
    print(f"  Réplicas significativas: {100*sg.mean():.0f}%")

    print("\n=== ARPU POR TIER (semilla 0) ===")
    bt = run_seed(cz, opp, seg, 0, by_tier=True)
    print(f"  {'Tier':<12}{'Baseline':>10}{'Dinámico':>10}{'Uplift':>9}")
    for tier in TIER_ORDER:
        b, d = bt[tier]
        upv = f"{100*(d-b)/b:+.0f}%" if b > 0 else "n/a"
        print(f"  {tier:<12}{b:>10.2f}{d:>10.2f}{upv:>9}")

    pd.DataFrame([{"uplift_medio":round(up.mean(),2),"ic95_low":round(up.mean()-ci,2),
        "ic95_high":round(up.mean()+ci,2),"arpu_baseline":round(ba.mean(),2),
        "arpu_dinamico":round(da.mean(),2),"pct_significativo":round(100*sg.mean(),1)}]
    ).to_csv(DATASETS / "abtest_v3_resultado.csv", index=False)
    print("\nGuardado: abtest_v3_resultado.csv")


if __name__ == "__main__":
    main()
