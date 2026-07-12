"""
SAOR-Poker | Motor de pricing v3 — Ventanas de 5 precios
=========================================================
NOTA DE AUTORÍA DEL RESULTADO:
  Este módulo documenta de forma legible la mecánica de ambos sistemas de
  pricing. El resultado canónico del TFM (+70,3 % de uplift) lo produce
  `abtest_v4.py`, que implementa una versión VECTORIZADA y equivalente de
  esta misma lógica (mismo catálogo de 14 puntos, misma ventana de 5 precios,
  mismos multiplicadores dinámicos -20/-10/CZ/+10/+20 %). Ante cualquier duda,
  abtest_v4.py es la fuente de verdad de las cifras reportadas.

Catálogo estático ampliado (14 price points, USD):
    0.99, 1.99, 4.99, 9.99, 19.99, 29.99, 49.99, 79.99,
    99.99, 129.99, 149.99, 199.99, 249.99, 299.99

SISTEMA ESTÁTICO:
  - Ventana de 5 precios consecutivos del catálogo.
  - Parte de los 5 más bajos (0.99 .. 19.99).
  - Cada compra sube la ventana un peldaño (elimina el más bajo, añade el
    siguiente por arriba). Nunca retrocede.

SISTEMA DINÁMICO:
  - 5 precios LIBRES construidos alrededor de la Comfort Zone (CZ),
    ventana centrada en la CZ y redondeada al alza al entero:
      P1 = CZ * 0.80   (-20%)
      P2 = CZ * 0.90   (-10%)
      P3 = CZ * 1.00   (CZ, referencia)
      P4 = CZ * 1.10   (+10%)
      P5 = CZ * 1.20   (+20%)

MODELO DE ELECCIÓN (común a ambos sistemas):
  El jugador ve 5 precios. Para cada uno se calcula su probabilidad logística
  de compra individual sobre el cociente precio/CZ. El jugador elige entre
  ellos (o no comprar) de forma proporcional a esa probabilidad. Así, los
  precios bajos se compran más pero rinden menos, y los altos al revés;
  el ingreso esperado captura ese equilibrio.
"""

import numpy as np

K = 6.0
CATALOG = [0.99, 1.99, 4.99, 9.99, 19.99, 29.99, 49.99, 79.99,
           99.99, 129.99, 149.99, 199.99, 249.99, 299.99]
WINDOW_SIZE = 5

# Multiplicadores del sistema dinámico respecto a la CZ
DYNAMIC_MULTIPLIERS = [0.80, 0.90, 1.00, 1.10, 1.20]  # ventana centrada en la CZ (-20/-10/CZ/+10/+20)


def purchase_prob(price, cz):
    """Probabilidad logística de compra de UN precio dado, frente a la CZ."""
    if cz <= 0:
        return 0.02 if price <= CATALOG[0] else 0.005
    return 1.0 / (1.0 + np.exp(np.clip(K * (price / cz - 1.0), -50, 50)))


def static_window(n_purchases):
    """Ventana de 5 precios del catálogo según número de compras (sin retroceso)."""
    start = min(n_purchases, len(CATALOG) - WINDOW_SIZE)
    return CATALOG[start:start + WINDOW_SIZE]


def dynamic_window(cz):
    """5 precios construidos alrededor de la CZ, redondeados al alza al entero."""
    if cz <= 0:
        # Sin CZ, se ofrece la ventana base del catálogo (jugador sin historial)
        return CATALOG[:WINDOW_SIZE]
    import math
    return [float(math.ceil(cz * m)) for m in DYNAMIC_MULTIPLIERS]


def expected_revenue_from_window(window, cz):
    """
    Ingreso esperado de mostrar una ventana de precios a un jugador con CZ dada.
    El jugador elige entre los precios (o no comprar) proporcionalmente a la
    probabilidad de compra de cada uno. Devuelve (ingreso_esperado, prob_compra).
    """
    probs = np.array([purchase_prob(p, cz) for p in window])
    prices = np.array(window)
    # Peso de cada opción = su probabilidad individual. Normalizamos para repartir
    # la "intención de compra" entre las opciones visibles.
    total_intent = probs.sum()
    if total_intent == 0:
        return 0.0, 0.0
    # Probabilidad de comprar ALGO: saturación suave de la intención agregada
    p_buy = 1.0 - np.prod(1.0 - probs)
    # Reparto entre precios proporcional a la probabilidad individual
    weights = probs / total_intent
    expected_price = (weights * prices).sum()
    return p_buy * expected_price, p_buy


def simulate_static(cz, n_opportunities, rng):
    """Simula a un jugador en el sistema estático durante sus oportunidades."""
    revenue = 0.0
    purchases = 0
    nc = 0
    for _ in range(n_opportunities):
        window = static_window(nc)
        probs = np.array([purchase_prob(p, cz) for p in window])
        p_buy = 1.0 - np.prod(1.0 - probs)
        if rng.random() < p_buy:
            # Elige un precio proporcional a su probabilidad
            weights = probs / probs.sum() if probs.sum() > 0 else np.ones(len(window)) / len(window)
            chosen = rng.choice(window, p=weights)
            revenue += chosen
            purchases += 1
            nc += 1
    return revenue, purchases


def simulate_dynamic(cz, n_opportunities, rng):
    """Simula a un jugador en el sistema dinámico (ventana fija alrededor de la CZ)."""
    window = dynamic_window(cz)
    probs = np.array([purchase_prob(p, cz) for p in window])
    revenue = 0.0
    purchases = 0
    for _ in range(n_opportunities):
        p_buy = 1.0 - np.prod(1.0 - probs)
        if rng.random() < p_buy:
            weights = probs / probs.sum() if probs.sum() > 0 else np.ones(len(window)) / len(window)
            chosen = rng.choice(window, p=weights)
            revenue += chosen
            purchases += 1
    return revenue, purchases
