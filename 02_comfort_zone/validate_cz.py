"""
Validación del módulo Comfort Zone.

Genera un dataframe transaccional con la MISMA estructura que Online Retail II
(user_id, timestamp, amount) e incluye usuarios-caso diseñados para verificar:
  - los siete tramos del Motor de Decaimiento (100/90/70/50/30/10/0),
    con un usuario-caso por tramo
  - la selección de ventana del Motor de Cálculo
  - el efecto del filtro de ruido (<2,99) activado vs desactivado

Cuando se disponga del fichero real, basta con:
    df = pd.read_excel("online_retail_II.xlsx")  # o read_parquet
    tx = adapt_online_retail_ii(df)
    cz = compute_comfort_zone(tx, apply_noise_filter=True)
y el resto del pipeline es idéntico.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Rastro de ejecución: fecha, máquina y versión de Python (evidencia de reproducción)
from datetime import datetime
import platform
print(f"[Ejecutado {datetime.now():%Y-%m-%d %H:%M:%S} | {platform.node()} | Python {platform.python_version()}]\n")

# Permite ejecutar el script desde cualquier CWD localizando comfort_zone.py
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:
    sys.path.insert(0, str(Path.cwd()))

from comfort_zone import (
    compute_comfort_zone,
    deterioration_weight,
    DETERIORATION_TIERS,
)

REF = pd.Timestamp("2026-05-28")  # fecha de referencia ("hoy" del análisis)


def make_tx(user_id, days_ago_list, amounts):
    """Crea transacciones para un usuario: cada compra a 'd' días de REF."""
    return pd.DataFrame({
        "user_id": user_id,
        "timestamp": [REF - pd.Timedelta(days=d) for d in days_ago_list],
        "amount": amounts,
    })


# --- Usuarios-caso, uno por comportamiento que queremos verificar ----------
# Pesos canónicos: 100/90/70/50/30/10/0 % en cortes 14/30/45/60/90/180 días.
cases = [
    # U1: última compra hace 5 días -> tramo 0-14 -> peso 100%.
    #     ventana 30D capta (5,12), CZ bruta=media(20,30)=25 -> final 25.
    make_tx("U1_reciente", [5, 12], [20.0, 30.0]),
    # U1b: última compra hace 20 días -> tramo 14-30 -> peso 90%.
    #      ventana 30D capta la compra, CZ bruta=100 -> final 90.
    make_tx("U1b_20dias", [20], [100.0]),
    # U2: última compra hace 45 días -> tramo 30-45 -> peso 70%.
    #     sin tx en 30D; ventana de respaldo 60D capta (45,50), media=50 -> final 35.
    make_tx("U2_45dias", [45, 50], [40.0, 60.0]),
    # U2b: última compra hace 50 días -> tramo 45-60 -> peso 50%.
    #      sin tx en 30D; ventana de respaldo 60D capta la compra, CZ bruta=80 -> final 40.
    make_tx("U2b_50dias", [50], [80.0]),
    # U3: última compra hace 80 días -> tramo 60-90 -> peso 30%.
    #     ventana de respaldo 90D, media=100 -> final 30.
    make_tx("U3_80dias", [80], [100.0]),
    # U4: última compra hace 150 días -> tramo 90-180 -> peso 10%.
    #     ventana de respaldo 180D capta (150,170), media=200 -> final 20.
    make_tx("U4_150dias", [150, 170], [150.0, 250.0]),
    # U5: última compra hace 300 días -> tramo >180 -> peso 0%.
    #     ventana de respaldo 365D capta la compra (CZ bruta=500), pero el
    #     decaimiento la anula -> final 0 (resultado válido, no ausencia de cálculo).
    make_tx("U5_300dias", [300], [500.0]),
    # U6: última compra hace 400 días -> peso 0% y, además, fuera de toda
    #     ventana de cálculo (>365D) -> CZ bruta NaN -> final 0.
    make_tx("U6_400dias", [400], [80.0]),
    # U7: ruido. Compras válidas (10.0) + ruido (<2,99). Con filtro CZ=10,
    #     sin filtro CZ baja por incluir los 2.0 y 1.5.
    make_tx("U7_ruido", [10, 10, 10], [10.0, 2.0, 1.5]),
]

tx_all = pd.concat(cases, ignore_index=True)

print("=" * 70)
print("TRANSACCIONES DE PRUEBA (estructura Online Retail II canónica)")
print("=" * 70)
print(tx_all.to_string(index=False))
print(f"\nFecha de referencia: {REF.date()}  |  Total transacciones: {len(tx_all)}")

# --- Tabla de tramos del decaimiento (para la memoria) ---------------------
print("\n" + "=" * 70)
print("FUNCIÓN DE DECAIMIENTO (escalonada)")
print("=" * 70)
prev = -1
for upper, w in DETERIORATION_TIERS:
    lo = prev + 1
    hi = "∞" if np.isinf(upper) else int(upper)
    print(f"  {lo:>4}–{hi:<4} días sin compra  ->  peso {int(w*100):>3} %")
    prev = upper if not np.isinf(upper) else prev

# --- CZ CON filtro de ruido ------------------------------------------------
print("\n" + "=" * 70)
print("COMFORT ZONE  ·  filtro de ruido ACTIVADO (apply_noise_filter=True)")
print("=" * 70)
cz_on = compute_comfort_zone(tx_all, reference_date=REF, apply_noise_filter=True)
print(cz_on.to_string(index=False))

# --- CZ SIN filtro de ruido (validación del efecto del filtro) -------------
print("\n" + "=" * 70)
print("COMFORT ZONE  ·  filtro de ruido DESACTIVADO (apply_noise_filter=False)")
print("=" * 70)
cz_off = compute_comfort_zone(tx_all, reference_date=REF, apply_noise_filter=False)
print(cz_off.to_string(index=False))

# --- Comparación del efecto del filtro sobre el usuario con ruido ----------
print("\n" + "=" * 70)
print("EFECTO DEL FILTRO sobre U7_ruido (CZ bruta)")
print("=" * 70)
r_on = cz_on.loc[cz_on.user_id == "U7_ruido", "cz_raw"].iloc[0]
r_off = cz_off.loc[cz_off.user_id == "U7_ruido", "cz_raw"].iloc[0]
print(f"  Con filtro  : CZ bruta = {r_on}  (solo cuenta la compra de 10.0)")
print(f"  Sin filtro  : CZ bruta = {r_off}  (incluye ruido 2.0 y 1.5)")

# --- Aserciones automáticas de los tramos esperados ------------------------
print("\n" + "=" * 70)
print("VERIFICACIÓN AUTOMÁTICA DE PESOS ESPERADOS")
print("=" * 70)
expected = {
    "U1_reciente": 1.00, "U1b_20dias": 0.90, "U2_45dias": 0.70,
    "U2b_50dias": 0.50, "U3_80dias": 0.30, "U4_150dias": 0.10,
    "U5_300dias": 0.00, "U6_400dias": 0.00,
}
ok = True
for uid, w_exp in expected.items():
    w_got = cz_on.loc[cz_on.user_id == uid, "deterioration_weight"].iloc[0]
    status = "OK" if abs(w_got - w_exp) < 1e-9 else "FALLO"
    if status == "FALLO":
        ok = False
    print(f"  {uid:<14} peso esperado {w_exp:>4}  obtenido {w_got:>4}  [{status}]")

# --- Aserciones de la CZ final (pipeline completo: cálculo x decaimiento) ---
print("\n" + "=" * 70)
print("VERIFICACIÓN AUTOMÁTICA DE CZ FINAL ESPERADA")
print("=" * 70)
# CZ final = CZ bruta (media en la ventana que capta la actividad) x peso.
expected_final = {
    "U1_reciente": 25.0,   # media(20,30)=25 x 1.00
    "U1b_20dias":  90.0,   # 100 x 0.90
    "U2_45dias":   35.0,   # media(40,60)=50 x 0.70
    "U2b_50dias":  40.0,   # 80 x 0.50
    "U3_80dias":   30.0,   # 100 x 0.30
    "U4_150dias":  20.0,   # media(150,250)=200 x 0.10
    "U5_300dias":   0.0,   # 500 x 0.00
    "U6_400dias":   0.0,   # fuera de toda ventana -> CZ bruta NaN -> 0
}
for uid, f_exp in expected_final.items():
    row = cz_on.loc[cz_on.user_id == uid, "cz_final"].iloc[0]
    f_got = 0.0 if pd.isna(row) else float(row)
    status = "OK" if abs(f_got - f_exp) < 1e-6 else "FALLO"
    if status == "FALLO":
        ok = False
    print(f"  {uid:<14} CZ final esperada {f_exp:>6.2f}  obtenida {f_got:>6.2f}  [{status}]")

# --- Asercion del efecto del filtro de ruido sobre U7 ----------------------
# Con filtro: solo cuenta la compra de 10.0 -> CZ bruta = 10.
# Sin filtro: media(10, 2, 1.5) = 4.5.
assert abs(r_on - 10.0) < 1e-6, f"U7 con filtro deberia dar 10, dio {r_on}"
assert abs(r_off - 4.5) < 1e-6, f"U7 sin filtro deberia dar 4.5, dio {r_off}"
print("\n  Filtro de ruido sobre U7: OK (con filtro=10.0, sin filtro=4.5)")

print("\n>>> RESULTADO:", "TODOS LOS TRAMOS CORRECTOS" if ok else "HAY FALLOS")
