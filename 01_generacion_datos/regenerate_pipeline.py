"""
SAOR-Poker | Pipeline maestro de regeneración (catálogo de 14 puntos)
======================================================================
Reproduce el dataset completo de forma canónica y coherente:
  1. Genera jugadores y transacciones con el catálogo de 14 price points
     (0,99 .. 299,99 $) mediante poker_generator_v2.
  2. Calcula la Comfort Zone (cz_raw y cz_final con decaimiento) con el
     módulo comfort_zone (7 tramos: 100/90/70/50/30/10/0).
  3. Consolida player_master.parquet y transactions_final.parquet.

Determinista: SEED fijo para reproducibilidad.
"""

import numpy as np
import pandas as pd
import importlib.util, sys
from pathlib import Path

SEED = 42

# Rutas robustas relativas a la ubicación del script (independientes del CWD)
try:
    # Ruta robusta cuando se ejecuta como script (.py)
    BASE = Path(__file__).resolve().parent
except NameError:
    # __file__ no existe en Jupyter: se asume que el notebook se ejecuta
    # desde su propia carpeta (convención del paquete de entregables)
    BASE = Path.cwd()
DATASETS = BASE.parent / "05_datasets"
CZ_PATH = BASE.parent / "02_comfort_zone" / "comfort_zone.py"
GEN_PATH = BASE / "poker_generator_v2.py"

# --- Cargar el generador v2 (catálogo ya actualizado a 14 puntos) ---
spec = importlib.util.spec_from_file_location("gen", GEN_PATH)
gen = importlib.util.module_from_spec(spec); sys.modules["gen"] = gen
spec.loader.exec_module(gen)

# --- Cargar el módulo de Comfort Zone ---
spec2 = importlib.util.spec_from_file_location("cz", CZ_PATH)
czmod = importlib.util.module_from_spec(spec2); sys.modules["cz"] = czmod
spec2.loader.exec_module(czmod)

REFERENCE_DATE = gen.REFERENCE_DATE


def build():
    rng = np.random.default_rng(SEED)
    n_payers = int(round(gen.N_TOTAL * gen.PAYER_RATE))
    n_nonpayers = gen.N_TOTAL - n_payers

    tiers = list(gen.TIER_MIX.keys())
    probs = np.array(list(gen.TIER_MIX.values())); probs = probs / probs.sum()
    assigned = rng.choice(tiers, size=n_payers, p=probs)

    player_rows, tx_rows = [], []
    for i, tier in enumerate(assigned):
        uid = f"POKER_{i:06d}"
        lo, hi = gen.TIER_LTV_RANGE[tier]
        ltv_target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        amounts = gen.ltv_to_transactions(ltv_target, rng)
        beh = gen.TIER_BEHAVIOR[tier]
        sw = max(0, round(rng.normal(*beh["sw"]), 1))
        sd = max(1, int(rng.normal(*beh["sd"])))
        eng = gen.ENG_LABELS[rng.choice(3, p=beh["eng"])]
        n_tx = len(amounts)
        span = (REFERENCE_DATE - gen.HISTORY_START).days
        if tier in ("Tier1", "Tier2"):
            offsets = rng.beta(1.6, 2.6, size=n_tx) * span
        elif tier in ("Megalodons", "Whales"):
            offsets = rng.uniform(0, span, size=n_tx)
        else:
            offsets = rng.beta(2.0, 2.0, size=n_tx) * span
        for amt, off in zip(amounts, offsets):
            tx_rows.append({"user_id": uid,
                            "timestamp": gen.HISTORY_START + pd.Timedelta(days=float(off)),
                            "amount": amt})
        player_rows.append({"user_id": uid, "tier": tier, "converted": 1,
            "age": int(np.clip(rng.normal(34, 11), 18, 80)),
            "sessions_per_week": sw, "avg_session_min": sd, "engagement_level": eng})

    for j in range(n_nonpayers):
        eng = gen.ENG_LABELS[rng.choice(3, p=gen.NONPAYER_ENG)]
        sw_map = {"Low": (1.5, 0.8), "Medium": (4, 1.5), "High": (9, 2.5)}[eng]
        sd_map = {"Low": (15, 6), "Medium": (35, 12), "High": (60, 18)}[eng]
        player_rows.append({"user_id": f"POKER_NP_{j:06d}", "tier": "NonSpender", "converted": 0,
            "age": int(np.clip(rng.normal(31, 12), 18, 80)),
            "sessions_per_week": max(0, round(rng.normal(*sw_map), 1)),
            "avg_session_min": max(1, int(rng.normal(*sd_map))),
            "engagement_level": eng})

    players = pd.DataFrame(player_rows)
    tx = pd.DataFrame(tx_rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    def ptype(a):
        if a <= 4.99:    p = [0.80, 0.15, 0.05]
        elif a <= 49.99: p = [0.45, 0.40, 0.15]
        else:            p = [0.20, 0.55, 0.25]
        return rng.choice(["chips", "offer", "tournament"], p=p)
    tx["product_type"] = [ptype(a) for a in tx["amount"].to_numpy()]

    # --- LTV real y verificación de tier ---
    ltv = tx.groupby("user_id").amount.sum()
    players = players.set_index("user_id")
    players["monetary"] = ltv.reindex(players.index).fillna(0.0)

    def tier_of(v):
        if v <= 0: return "NonSpender"
        if v < 20: return "Tier1"
        if v < 80: return "Tier2"
        if v < 250: return "Tier3"
        if v < 1000: return "Mobys"
        if v < 3000: return "Whales"
        return "Megalodons"
    players["archetype"] = players["monetary"].apply(tier_of)
    players = players.reset_index()

    # --- RFM: recency y frequency ---
    last_tx = tx.groupby("user_id").timestamp.max()
    freq = tx.groupby("user_id").size()
    players = players.set_index("user_id")
    players["recency_days"] = (REFERENCE_DATE - last_tx.reindex(players.index)).dt.days
    players["recency_days"] = players["recency_days"].fillna(9999).astype(int)
    players["frequency"] = freq.reindex(players.index).fillna(0).astype(int)
    players = players.reset_index()

    # --- Comfort Zone (cz_raw y cz_final) ---
    cz = czmod.compute_comfort_zone(tx, reference_date=REFERENCE_DATE)
    players = players.merge(cz[["user_id", "cz_raw", "cz_final"]], on="user_id", how="left")
    players["cz_raw"] = players["cz_raw"].fillna(0.0)
    players["cz_final"] = players["cz_final"].fillna(0.0)

    players.to_parquet(DATASETS / "player_master.parquet")
    tx.to_parquet(DATASETS / "transactions_final.parquet")

    # --- Resumen ---
    order = ["NonSpender","Tier1","Tier2","Tier3","Mobys","Whales","Megalodons"]
    print(f"Población: {len(players):,} | pagadores: {n_payers:,} | no pagadores: {n_nonpayers:,}")
    print(f"Transacciones: {len(tx):,}")
    print(f"Precios distintos en transacciones: {sorted(tx.amount.unique())}")
    print(f"Precio máximo: {tx.amount.max()}")
    coh = (players.archetype == players.archetype).mean()
    print(f"\nReparto por tier:")
    vc = players.archetype.value_counts()
    for t in order:
        print(f"  {t:<12}{vc.get(t,0):>8,}")
    print(f"\nCZ media (activos con cz_final>0): {players[players.cz_final>0].cz_final.mean():.2f}")
    print(f"Pagadores con cz_final=0: {((players.converted==1)&(players.cz_final==0)).sum():,}")


if __name__ == "__main__":
    build()
