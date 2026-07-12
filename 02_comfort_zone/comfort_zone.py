"""
SAOR-Poker | Módulo de Comfort Zone (CZ)
=========================================
Implementa los dos motores de la lógica de pricing dinámico:

  1. Motor de Cálculo (CZ Calculation):
     Calcula la CZ bruta = promedio de las transacciones VÁLIDAS dentro de
     la ventana temporal principal de 30 días hacia atrás desde la fecha de
     referencia. Se mantienen ventanas de respaldo más amplias
     (60/90/180/365 días) solo para jugadores sin transacciones recientes,
     de modo que la CZ siga siendo calculable; la de 30 días es la preferente.
     La ventana define qué transacciones se promedian.

  2. Motor de Decaimiento (CZ Deterioration):
     Ajusta el peso de la CZ según la RECENCIA (días desde la última compra)
     mediante una función ESCALONADA de 7 tramos: 100/90/70/50/30/10/0 %
     en los cortes 14/30/45/60/90/180 días.

  CZ_final = CZ_bruta * peso_decaimiento

Diseñado para validarse primero sobre Online Retail II (datos reales de
e-commerce) y reutilizarse después sobre el dataset sintético de póker.

Estructura de entrada esperada (esquema canónico):
  - user_id   : identificador único de usuario/cliente
  - timestamp : fecha-hora de la transacción (datetime)
  - amount    : importe monetario de la transacción (float)

Nota de validación: el filtro de ruido (<2,99 y mecánicas rígidas) es un
paso SEPARADO y DESACTIVABLE, para poder medir su efecto sobre datos reales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. PARÁMETROS (centralizados para facilitar el ajuste y la defensa del TFM)
# ---------------------------------------------------------------------------

# Ventanas del Motor de Cálculo, en días. Define qué transacciones se promedian.
# Ventana principal del motor de cálculo: 30 días. Se mantienen ventanas de
# respaldo más amplias para jugadores sin transacciones recientes, de modo que
# la CZ siga siendo calculable; la de 30 días es la preferente.
CALCULATION_WINDOWS = [30, 60, 90, 180, 365]

# Tramos del Motor de Decaimiento. Cada tupla es (límite_superior_días, peso).
# Se lee como: "si la recencia es <= límite, aplica este peso".
# El último tramo (inf) garantiza peso 0 más allá de 365 días.
DETERIORATION_TIERS = [
    (14,   1.00),   # 0-14 días   -> 100 %
    (30,   0.90),   # 14-30 días  ->  90 %
    (45,   0.70),   # 30-45 días  ->  70 %
    (60,   0.50),   # 45-60 días  ->  50 %
    (90,   0.30),   # 60-90 días  ->  30 %
    (180,  0.10),   # 90-180 días ->  10 %
    (np.inf, 0.00),  # > 180 días  ->   0 %
]

# Filtro de ruido (Motor de Cálculo). Desactivable mediante `apply_noise_filter`.
# Umbral definitivo: excluye importes < 2.99 $ y mecánicas estáticas (piggy bank).
NOISE_MIN_AMOUNT = 2.99            # se excluyen importes < 2,99
RIGID_MECHANIC_FLAG = "is_rigid"   # columna opcional booleana para "piggy bank" etc.


# ---------------------------------------------------------------------------
# 2. MOTOR DE DECAIMIENTO (CZ Deterioration)
# ---------------------------------------------------------------------------

def deterioration_weight(days_since_last_purchase: float,
                         tiers: list[tuple[float, float]] = DETERIORATION_TIERS
                         ) -> float:
    """
    Devuelve el peso de decaimiento (0.0 - 1.0) según la recencia, usando la
    función escalonada definida en `tiers`.

    Parámetros
    ----------
    days_since_last_purchase : float
        Días transcurridos desde la última transacción válida.
    tiers : list[(limite_dias, peso)]
        Tramos ordenados de menor a mayor límite.

    Ejemplos
    --------
    >>> deterioration_weight(0)     # compra hoy
    1.0
    >>> deterioration_weight(45)    # 45 días sin comprar
    0.5
    >>> deterioration_weight(400)   # más de 180 días
    0.0
    """
    if days_since_last_purchase is None or np.isnan(days_since_last_purchase):
        return 0.0
    # Recencia negativa (datos sucios) se trata como recencia 0.
    d = max(0.0, float(days_since_last_purchase))
    for upper_limit, weight in tiers:
        if d <= upper_limit:
            return weight
    return 0.0  # salvaguarda; no debería alcanzarse por el tramo np.inf


# ---------------------------------------------------------------------------
# 3. MOTOR DE CÁLCULO (CZ Calculation)
# ---------------------------------------------------------------------------

def _filter_noise(tx: pd.DataFrame,
                  apply_noise_filter: bool,
                  amount_col: str = "amount") -> pd.DataFrame:
    """
    Aplica (o no) el filtro de ruido del Motor de Cálculo:
      - excluye importes < NOISE_MIN_AMOUNT
      - excluye mecánicas rígidas si existe la columna RIGID_MECHANIC_FLAG
    """
    if not apply_noise_filter:
        return tx
    mask = tx[amount_col] >= NOISE_MIN_AMOUNT
    if RIGID_MECHANIC_FLAG in tx.columns:
        mask &= ~tx[RIGID_MECHANIC_FLAG].astype(bool)
    return tx[mask]


def calculate_cz_raw(user_tx: pd.DataFrame,
                     reference_date: pd.Timestamp,
                     window_days: int,
                     apply_noise_filter: bool = True,
                     timestamp_col: str = "timestamp",
                     amount_col: str = "amount") -> float:
    """
    CZ bruta para UN usuario: promedio de los importes de las transacciones
    válidas ocurridas dentro de [reference_date - window_days, reference_date].

    Devuelve np.nan si no hay transacciones válidas en la ventana
    (el usuario no tiene CZ calculable en ese bloque temporal).
    """
    window_start = reference_date - pd.Timedelta(days=window_days)
    in_window = user_tx[
        (user_tx[timestamp_col] > window_start) &
        (user_tx[timestamp_col] <= reference_date)
    ]
    in_window = _filter_noise(in_window, apply_noise_filter, amount_col)
    if len(in_window) == 0:
        return np.nan
    return float(in_window[amount_col].mean())


def select_calculation_window(user_tx: pd.DataFrame,
                              reference_date: pd.Timestamp,
                              windows: list[int] = CALCULATION_WINDOWS,
                              apply_noise_filter: bool = True,
                              timestamp_col: str = "timestamp",
                              amount_col: str = "amount") -> tuple[float, int]:
    """
    Selecciona la ventana de cálculo más estrecha que contenga transacciones
    válidas (de 30 hacia 365). Refleja la imagen: el sistema intenta primero
    el bloque de 30D y, si no hay datos suficientes, amplía la ventana.

    Devuelve (cz_bruta, ventana_usada). Si ninguna ventana tiene datos,
    devuelve (np.nan, -1).
    """
    for w in sorted(windows):
        cz = calculate_cz_raw(user_tx, reference_date, w,
                              apply_noise_filter, timestamp_col, amount_col)
        if not np.isnan(cz):
            return cz, w
    return np.nan, -1


# ---------------------------------------------------------------------------
# 4. ORQUESTADOR: CZ FINAL POR USUARIO
# ---------------------------------------------------------------------------

def compute_comfort_zone(transactions: pd.DataFrame,
                         reference_date: pd.Timestamp | None = None,
                         apply_noise_filter: bool = True,
                         user_col: str = "user_id",
                         timestamp_col: str = "timestamp",
                         amount_col: str = "amount") -> pd.DataFrame:
    """
    Calcula la Comfort Zone final para TODOS los usuarios de un dataframe
    transaccional con esquema canónico (user_id, timestamp, amount).

    CZ_final = CZ_bruta * peso_decaimiento(recencia)

    Devuelve un dataframe por usuario con las columnas:
      user_id, last_purchase, recency_days, cz_window, cz_raw,
      deterioration_weight, cz_final, n_valid_tx

    Parámetros
    ----------
    reference_date : fecha "hoy" del análisis. Si es None, se usa la fecha
        máxima del dataset (práctica estándar en RFM).
    apply_noise_filter : activa/desactiva el filtro de ruido (<2,99 y rígidas).
    """
    tx = transactions.copy()
    tx[timestamp_col] = pd.to_datetime(tx[timestamp_col])

    if reference_date is None:
        reference_date = tx[timestamp_col].max()
    reference_date = pd.Timestamp(reference_date)

    rows = []
    for user_id, g in tx.groupby(user_col):
        # --- recencia: días desde la última transacción VÁLIDA ---
        g_valid = _filter_noise(g, apply_noise_filter, amount_col)
        if len(g_valid) == 0:
            last_purchase = pd.NaT
            recency = np.nan
        else:
            last_purchase = g_valid[timestamp_col].max()
            recency = (reference_date - last_purchase).days

        # --- Motor de Cálculo: CZ bruta y ventana usada ---
        cz_raw, window = select_calculation_window(
            g, reference_date, CALCULATION_WINDOWS,
            apply_noise_filter, timestamp_col, amount_col)

        # --- Motor de Decaimiento: peso por recencia ---
        weight = deterioration_weight(recency)

        # --- CZ final ---
        cz_final = (cz_raw * weight) if not np.isnan(cz_raw) else np.nan

        rows.append({
            user_col: user_id,
            "last_purchase": last_purchase,
            "recency_days": recency,
            "cz_window": window,
            "cz_raw": round(cz_raw, 4) if not np.isnan(cz_raw) else np.nan,
            "deterioration_weight": weight,
            "cz_final": round(cz_final, 4) if not np.isnan(cz_final) else np.nan,
            "n_valid_tx": len(g_valid),
        })

    return pd.DataFrame(rows).sort_values("cz_final",
                                          ascending=False,
                                          na_position="last").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. ADAPTADOR PARA ONLINE RETAIL II
# ---------------------------------------------------------------------------

def adapt_online_retail_ii(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte Online Retail II al esquema canónico (user_id, timestamp, amount).

    Online Retail II tiene una fila por LÍNEA de factura, así que:
      - importe de línea = Quantity * Price
      - se agregan las líneas de una misma factura en una sola transacción
        (una compra = una factura), que es lo correcto para la CZ.
      - se descartan cancelaciones (Invoice que empieza por 'C') y filas sin
        CustomerID (clientes anónimos no agregables).

    Acepta las variantes de nombres de columna habituales del dataset.
    """
    df = df.copy()
    # Normalizar nombres (el dataset aparece con distintas mayúsculas/espacios)
    colmap = {c.lower().strip().replace(" ", ""): c for c in df.columns}

    def col(*candidates):
        for cand in candidates:
            if cand in colmap:
                return colmap[cand]
        raise KeyError(f"No encuentro ninguna de {candidates} en {list(df.columns)}")

    invoice = col("invoice", "invoiceno")
    stock_qty = col("quantity")
    price = col("price", "unitprice")
    cust = col("customerid", "customer id")
    date = col("invoicedate")

    # Descartar cancelaciones (Invoice con prefijo 'C') y clientes anónimos
    df = df[~df[invoice].astype(str).str.startswith("C")]
    df = df.dropna(subset=[cust])

    df["line_amount"] = df[stock_qty] * df[price]
    df = df[df["line_amount"] > 0]  # descartar importes no positivos

    # Una transacción = una factura: agregamos las líneas
    agg = (df.groupby([cust, invoice])
             .agg(amount=("line_amount", "sum"),
                  timestamp=(date, "first"))
             .reset_index()
             .rename(columns={cust: "user_id"}))

    return agg[["user_id", "timestamp", "amount"]]


if __name__ == "__main__":
    # Rastro de ejecución: fecha, máquina y versión de Python (evidencia de reproducción)
    from datetime import datetime
    import platform
    print(f"[Ejecutado {datetime.now():%Y-%m-%d %H:%M:%S} | {platform.node()} | Python {platform.python_version()}]\n")
    # Autotest del Motor de Decaimiento, alineado con DETERIORATION_TIERS
    # (100/90/70/50/30/10/0 en cortes 14/30/45/60/90/180 días).
    assert deterioration_weight(0) == 1.00
    assert deterioration_weight(14) == 1.00
    assert deterioration_weight(15) == 0.90
    assert deterioration_weight(30) == 0.90
    assert deterioration_weight(31) == 0.70
    assert deterioration_weight(45) == 0.70
    assert deterioration_weight(46) == 0.50
    assert deterioration_weight(60) == 0.50
    assert deterioration_weight(61) == 0.30
    assert deterioration_weight(90) == 0.30
    assert deterioration_weight(91) == 0.10
    assert deterioration_weight(180) == 0.10
    assert deterioration_weight(181) == 0.00
    assert deterioration_weight(1000) == 0.00
    print("Motor de Decaimiento: todos los tramos OK")
