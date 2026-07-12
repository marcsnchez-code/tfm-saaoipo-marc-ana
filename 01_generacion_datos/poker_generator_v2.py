"""
SAOR-Poker | Generador de dataset v2 — Tiers de LTV unificados
===============================================================
Reescritura del generador para que la clasificación de jugadores se base en
una única métrica unificada: el Lifetime Value (LTV = gasto acumulado en USD).

Tiers (sobre LTV en USD):
  NonSpender : LTV = 0 (nunca gasta)
  Tier1      : 0  < LTV < 20
  Tier2      : 20 <= LTV < 80
  Tier3      : 80 <= LTV < 250
  Mobys      : 250 <= LTV < 1000
  Whales     : 1000 <= LTV < 3000
  Megalodons : LTV >= 3000

Enfoque: a cada jugador pagador se le asigna un LTV objetivo dentro de su tier
(muestreado de forma que la distribución dentro del tier sea realista) y se
generan sus transacciones —del catálogo de precios— de modo que su suma se
aproxime a ese LTV. Así, cada jugador cae en su tier POR CONSTRUCCIÓN.

Catálogo de precios (USD): 0.99 .. 299.99 (catálogo de 14 puntos).
Histórico: 18 meses. Reproducible (semilla fija).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

SEED = 42
N_TOTAL = 71_429          # población total (se mantiene el tamaño previo)
PAYER_RATE = 0.28          # 28% paga -> ~20.000 pagadores
HISTORY_MONTHS = 18
REFERENCE_DATE = pd.Timestamp("2026-05-28")
HISTORY_START = REFERENCE_DATE - pd.DateOffset(months=HISTORY_MONTHS)

PRICE_CATALOG = [0.99, 1.99, 4.99, 9.99, 19.99, 29.99, 49.99, 79.99, 99.99, 129.99, 149.99, 199.99, 249.99, 299.99]

# Proporciones de los PAGADORES por tier (pirámide decreciente realista)
TIER_MIX = {
    "Tier1":      0.42,
    "Tier2":      0.28,
    "Tier3":      0.18,
    "Mobys":      0.085,
    "Whales":     0.030,
    "Megalodons": 0.005,
}
# Rango de LTV (USD) por tier. El LTV objetivo se muestrea dentro del rango.
TIER_LTV_RANGE = {
    "Tier1":      (4, 20),
    "Tier2":      (20, 80),
    "Tier3":      (80, 250),
    "Mobys":      (250, 1000),
    "Whales":     (1000, 3000),
    "Megalodons": (3000, 8000),
}
# Comportamiento de juego por tier (sesiones/semana, duración media min)
TIER_BEHAVIOR = {
    "Tier1":      {"sw": (3, 1.5),  "sd": (25, 10), "eng": [0.55, 0.35, 0.10]},
    "Tier2":      {"sw": (6, 2.0),  "sd": (40, 14), "eng": [0.30, 0.50, 0.20]},
    "Tier3":      {"sw": (10, 3.0), "sd": (60, 18), "eng": [0.15, 0.50, 0.35]},
    "Mobys":      {"sw": (16, 4.0), "sd": (85, 22), "eng": [0.05, 0.40, 0.55]},
    "Whales":     {"sw": (22, 5.0), "sd": (110, 28),"eng": [0.02, 0.28, 0.70]},
    "Megalodons": {"sw": (28, 6.0), "sd": (140, 32),"eng": [0.00, 0.15, 0.85]},
}
ENG_LABELS = ["Low", "Medium", "High"]
NONPAYER_ENG = [0.65, 0.28, 0.07]


def ltv_to_transactions(ltv_target, rng):
    """
    Genera una lista de importes del catálogo cuya suma se aproxima al LTV
    objetivo. Estrategia: ir añadiendo price points hasta acercarse al objetivo,
    eligiendo precios proporcionales a la escala del LTV (LTV alto -> tickets
    más grandes), con variabilidad realista.
    """
    catalog = np.array(PRICE_CATALOG)
    amounts = []
    remaining = ltv_target
    # Ticket típico según escala del LTV (más alto si el jugador es de mayor valor)
    while remaining >= PRICE_CATALOG[0]:
        # Candidatos que no exceden demasiado lo que queda
        feasible = catalog[catalog <= max(remaining * 1.2, PRICE_CATALOG[0])]
        if len(feasible) == 0:
            feasible = catalog[:1]
        # Sesgo hacia tickets coherentes con la escala restante
        weights = 1.0 / (1.0 + np.abs(feasible - remaining * 0.4))
        weights = weights / weights.sum()
        pick = float(rng.choice(feasible, p=weights))
        amounts.append(pick)
        remaining -= pick
        if len(amounts) > 2000:  # salvaguarda
            break
    if not amounts:
        amounts = [PRICE_CATALOG[0]]
    return amounts


def main():
    rng = np.random.default_rng(SEED)
    n_payers = int(round(N_TOTAL * PAYER_RATE))
    n_nonpayers = N_TOTAL - n_payers

    # --- Asignar tiers a los pagadores ---
    tiers = list(TIER_MIX.keys())
    probs = np.array(list(TIER_MIX.values()))
    probs = probs / probs.sum()
    assigned = rng.choice(tiers, size=n_payers, p=probs)

    player_rows = []
    tx_rows = []

    # --- Pagadores ---
    for i, tier in enumerate(assigned):
        uid = f"POKER_{i:06d}"
        lo, hi = TIER_LTV_RANGE[tier]
        # LTV objetivo: muestreo log-uniforme dentro del rango (realista)
        ltv_target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        amounts = ltv_to_transactions(ltv_target, rng)

        beh = TIER_BEHAVIOR[tier]
        sw = max(0, round(rng.normal(*beh["sw"]), 1))
        sd = max(1, int(rng.normal(*beh["sd"])))
        eng = ENG_LABELS[rng.choice(3, p=beh["eng"])]

        # Distribución temporal: los tiers altos, activos hasta hoy;
        # los bajos tienden a enfriarse antes.
        n_tx = len(amounts)
        if tier in ("Tier1", "Tier2"):
            offsets = rng.beta(1.6, 2.6, size=n_tx) * (REFERENCE_DATE - HISTORY_START).days
        elif tier in ("Megalodons", "Whales"):
            offsets = rng.uniform(0, (REFERENCE_DATE - HISTORY_START).days, size=n_tx)
        else:
            offsets = rng.beta(2.0, 2.0, size=n_tx) * (REFERENCE_DATE - HISTORY_START).days

        for amt, off in zip(amounts, offsets):
            tx_rows.append({
                "user_id": uid,
                "timestamp": HISTORY_START + pd.Timedelta(days=float(off)),
                "amount": amt,
            })

        player_rows.append({
            "user_id": uid, "tier": tier, "converted": 1,
            "age": int(np.clip(rng.normal(34, 11), 18, 80)),
            "sessions_per_week": sw, "avg_session_min": sd,
            "engagement_level": eng,
        })

    # --- No pagadores ---
    for j in range(n_nonpayers):
        eng = ENG_LABELS[rng.choice(3, p=NONPAYER_ENG)]
        sw_map = {"Low": (1.5, 0.8), "Medium": (4, 1.5), "High": (9, 2.5)}[eng]
        sd_map = {"Low": (15, 6), "Medium": (35, 12), "High": (60, 18)}[eng]
        player_rows.append({
            "user_id": f"POKER_NP_{j:06d}", "tier": "NonSpender", "converted": 0,
            "age": int(np.clip(rng.normal(31, 12), 18, 80)),
            "sessions_per_week": max(0, round(rng.normal(*sw_map), 1)),
            "avg_session_min": max(1, int(rng.normal(*sd_map))),
            "engagement_level": eng,
        })

    players = pd.DataFrame(player_rows)
    tx = pd.DataFrame(tx_rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Producto por transacción (igual que antes)
    def ptype(a):
        if a <= 4.99:   p = [0.80, 0.15, 0.05]
        elif a <= 49.99:p = [0.45, 0.40, 0.15]
        else:           p = [0.20, 0.55, 0.25]
        return rng.choice(["chips", "offer", "tournament"], p=p)
    tx["product_type"] = [ptype(a) for a in tx["amount"].to_numpy()]

    players.to_parquet("players_v2.parquet")
    tx.to_parquet("transactions_v2.parquet")

    # --- Verificación de tiers reales ---
    ltv = tx.groupby("user_id").amount.sum()
    players = players.set_index("user_id")
    players["ltv_real"] = ltv
    players["ltv_real"] = players["ltv_real"].fillna(0.0)

    def tier_of(v):
        if v <= 0: return "NonSpender"
        if v < 20: return "Tier1"
        if v < 80: return "Tier2"
        if v < 250: return "Tier3"
        if v < 1000: return "Mobys"
        if v < 3000: return "Whales"
        return "Megalodons"
    players["tier_check"] = players["ltv_real"].apply(tier_of)
    players = players.reset_index()
    players.to_parquet("players_v2.parquet")

    print(f"Población: {len(players):,} | pagadores: {n_payers:,} | no pagadores: {n_nonpayers:,}")
    print(f"Transacciones: {len(tx):,}\n")
    order = ["NonSpender", "Tier1", "Tier2", "Tier3", "Mobys", "Whales", "Megalodons"]
    print("Reparto por tier (verificado sobre LTV real):")
    vc = players.tier_check.value_counts()
    rev = players.groupby("tier_check").ltv_real.sum()
    for t in order:
        n = vc.get(t, 0)
        pct = 100 * n / len(players)
        rpct = 100 * rev.get(t, 0) / rev.sum()
        print(f"  {t:<11}: {n:>6,} ({pct:>5.1f}%)  | aporta {rpct:>5.1f}% de los ingresos")
    # Coherencia tier asignado vs tier real
    match = (players[players.converted==1].tier == players[players.converted==1].tier_check).mean()
    print(f"\nCoherencia tier asignado vs LTV real (pagadores): {100*match:.1f}%")


if __name__ == "__main__":
    main()
