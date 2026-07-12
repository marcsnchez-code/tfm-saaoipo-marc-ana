"""
SAOR-Poker | Generador de los CSVs del dashboard (Power BI)
============================================================
Regenera de forma trazable los dos ficheros que alimentan el dashboard de la
Fase 6.4, hasta ahora exportados sin script generador en el paquete:

  1. 05_datasets/resultado_por_tier_v4.csv
     Desglose de ARPU estático vs dinámico por tier bajo el escenario canónico
     (error de estimación de la CZ ±25 %). Es la media, sobre las 40 réplicas,
     del ARPU de cada brazo dentro de cada tier.

  2. 05_datasets/pbi_strategy_long_v2.csv
     Formato largo por jugador y estrategia para Power BI. La columna
     `expected_revenue` es el INGRESO ESPERADO POR JUGADOR EN EL PERIODO
     SIMULADO (30 días), estimado como la media del ingreso observado del
     jugador en las réplicas en las que fue asignado a cada brazo (~20 de las
     40 réplicas por brazo, asignación estratificada por tier).

     UNIDADES: dólares por jugador y periodo. Por construcción, el promedio de
     `expected_revenue` por estrategia reproduce el ARPU canónico del A/B
     (64,8 $ estático / 110,2 $ dinámico; uplift +70,3 %), de modo que
     cualquier media simple del dashboard es coherente con la memoria (§5).

REPRODUCIBILIDAD: replica EXACTAMENTE las 40 ejecuciones del A/B canónico
(`abtest_v4.py`): mismas semillas (0..39), mismo orden de llamadas al RNG,
misma asignación estratificada y mismas funciones de simulación importadas.
No introduce lógica nueva: solo captura el detalle por jugador que el script
canónico agrega.

Uso:
    python generar_csv_dashboard.py
Determinista (semillas 0..39, ruido ±25 %).
"""

import numpy as np
import pandas as pd

# Reutiliza la lógica canónica por import (mismo patrón que abtest_robustez.py)
import abtest_v4 as ab

NOISE_MAIN = 0.25            # escenario canónico de la memoria (±25 %)
N_SEEDS = ab.N_SEEDS         # 40, igual que el A/B canónico


def run_seed_detail(dpr, opp, seg, noise, seed):
    """Réplica exacta de abtest_v4.run_seed, devolviendo el detalle por jugador.

    Reproduce el MISMO orden de llamadas al RNG que la función canónica
    (error -> barajado estratificado -> estático -> dinámico), por lo que los
    ingresos por jugador son idénticos a los de la ejecución de referencia.
    Devuelve (rev, arm): ingreso del periodo y brazo asignado ('A'=estático,
    'B'=dinámico) de cada jugador en esta réplica.
    """
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
    rev[b] = ab.dynamic_arpu(dpr[b], cz_est[b], opp[b], rng)
    return rev, arm


def main():
    pm = pd.read_parquet(ab.DATASETS / "player_master.parquet")
    dpr = pm.cz_final.to_numpy()
    seg = pm.archetype.to_numpy()
    opp = (pm.sessions_per_week * ab.WEEKS).round().clip(lower=1).astype(int).to_numpy()
    n = len(pm)

    print(f"Generador de CSVs de dashboard | {n:,} jugadores | {N_SEEDS} semillas | ruido ±{int(NOISE_MAIN*100)}%\n")

    # Acumuladores por jugador y brazo
    sum_a = np.zeros(n); cnt_a = np.zeros(n)
    sum_b = np.zeros(n); cnt_b = np.zeros(n)

    for s in range(N_SEEDS):
        rev, arm = run_seed_detail(dpr, opp, seg, NOISE_MAIN, s)
        a = arm == "A"; b = arm == "B"
        sum_a[a] += rev[a]; cnt_a[a] += 1
        sum_b[b] += rev[b]; cnt_b[b] += 1

    # Estimador normalizado (tipo Horvitz-Thompson): ingreso total del jugador
    # en cada brazo dividido por el número ESPERADO de asignaciones a ese brazo
    # (N_SEEDS * tamaño_brazo / n). Con esta normalización, la media poblacional
    # de expected_revenue por estrategia coincide EXACTAMENTE con la media de
    # los ARPU por réplica del A/B canónico (64,82 / 110,22 $).
    n_arm_a = float(cnt_a.sum()) / N_SEEDS      # tamaño (constante) del brazo A
    n_arm_b = float(cnt_b.sum()) / N_SEEDS      # tamaño (constante) del brazo B
    exp_static = sum_a / (N_SEEDS * n_arm_a / n)
    exp_dyn = sum_b / (N_SEEDS * n_arm_b / n)

    # ---------------------------------------------------------------
    # 1) resultado_por_tier_v4.csv — media de las 40 réplicas del ARPU
    #    de cada brazo por tier (idéntico a run_seed(by_tier=True))
    # ---------------------------------------------------------------
    per_tier = {t: {"base": [], "dyn": []} for t in ab.TIER_ORDER}
    for s in range(N_SEEDS):
        res = ab.run_seed(dpr, opp, seg, NOISE_MAIN, s, by_tier=True)
        for t in ab.TIER_ORDER:
            per_tier[t]["base"].append(res[t][0])
            per_tier[t]["dyn"].append(res[t][1])

    rows = []
    for t in ab.TIER_ORDER:
        base = float(np.mean(per_tier[t]["base"]))
        dyn = float(np.mean(per_tier[t]["dyn"]))
        rows.append({"tier": t, "arpu_base": round(base, 2), "arpu_dyn": round(dyn, 2),
                     "uplift": round(100 * (dyn - base) / base, 1) if base > 0 else 0.0})
    tier_df = pd.DataFrame(rows)
    out1 = ab.DATASETS / "resultado_por_tier_v4.csv"
    tier_df.to_csv(out1, index=False)
    print("ARPU por tier (±25 %, media de 40 réplicas):")
    print(tier_df.to_string(index=False))
    print(f"\nGuardado: 05_datasets/{out1.name}")

    # ---------------------------------------------------------------
    # 2) pbi_strategy_long_v2.csv — formato largo por jugador y estrategia
    # ---------------------------------------------------------------
    base_cols = pm[["user_id", "archetype", "converted", "cz_final"]].copy()
    est = base_cols.copy(); est["expected_revenue"] = np.round(exp_static, 4); est["strategy"] = "Estatico"
    dyn = base_cols.copy(); dyn["expected_revenue"] = np.round(exp_dyn, 4); dyn["strategy"] = "Dinamico"
    long_df = pd.concat([est, dyn], ignore_index=True)
    out2 = ab.DATASETS / "pbi_strategy_long_v2.csv"
    long_df.to_csv(out2, index=False)

    arpu_est = exp_static.mean(); arpu_dyn = exp_dyn.mean()
    print(f"\nGuardado: 05_datasets/{out2.name}")
    print(f"Verificación de coherencia (media de expected_revenue por estrategia):")
    print(f"  Estático: {arpu_est:.2f} $ | Dinámico: {arpu_dyn:.2f} $ | "
          f"uplift {100*(arpu_dyn-arpu_est)/arpu_est:.1f}%")
    print("  (referencia canónica del A/B: 64,8 -> 110,2 $, +70,3 %)")


if __name__ == "__main__":
    # Rastro de ejecución: fecha, máquina y versión de Python (evidencia de reproducción)
    from datetime import datetime
    import platform
    print(f"[Ejecutado {datetime.now():%Y-%m-%d %H:%M:%S} | {platform.node()} | Python {platform.python_version()}]\n")
    main()
