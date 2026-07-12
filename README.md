# SAAOIPO — Paquete de entregables (versión final)

**Sistema Analítico Aplicado a la Optimización de Ingresos en Póker Online**
TFM — Máster en Business Analytics, Universidad Europea de Madrid
Autores: Marc Sánchez Marchador y Ana Isabel Rodríguez García · Director: Jesús Gil

Este paquete contiene la versión final y coherente del proyecto, tras la
auditoría de reproducibilidad. Todos los artefactos (código, datos y
logs de ejecución) corresponden a la misma iteración del desarrollo que la
memoria entregada por separado, cuyas cifras reproducen exactamente.

---

## Resultado principal

El sistema de pricing dinámico incrementa el ARPU un **70,3%** sobre el sistema
estático de referencia (ARPU 64,8 → 110,2 \$), bajo un error de estimación de la
Comfort Zone del ±25%, significativo en el 100 % de las 40 réplicas del test A/B.
Análisis de sensibilidad: el uplift se mantiene entre **+57,7%** (error ±40%) y
**+78,6%** (estimación perfecta), siempre significativo.

Dos pruebas de robustez adicionales (`04_pricing_simulacion/abtest_robustez.py`)
respaldan el resultado: con el sistema dinámico **restringido al mismo catálogo**
que el estático (igualdad total de precios) el uplift sigue siendo **+32,4%**; y
usando la tolerancia **sin decaimiento** (cz_raw) asciende a **+74,3%**, lo que
confirma que emplear cz_final en el resultado principal es la opción conservadora.

---

## Estructura del paquete

### 01_generacion_datos/
- `poker_generator_v2.py` — Generador del dataset sintético por 7 tiers de LTV.
  Catálogo de 14 price points (0,99–299,99 \$).
- `regenerate_pipeline.py` — **Script maestro**. Reproduce el dataset completo de
  forma canónica: genera jugadores y transacciones, calcula la Comfort Zone
  (cz_raw y cz_final) y consolida los dos parquet finales. Determinista (SEED=42).
  Uso: `python regenerate_pipeline.py`

### 02_comfort_zone/
- `comfort_zone.py` — Módulo de la Comfort Zone. Motor de cálculo (ventana de
  30 días, umbral de ruido 2,99 \$) y motor de decaimiento (7 tramos:
  100/90/70/50/30/10/0 % en cortes 14/30/45/60/90/180 días). Incluye autotest
  (`python comfort_zone.py` → "todos los tramos OK").
- `validate_cz.py` — Validación de la CZ sobre el dataset real Online Retail II.

### 03_modelos_ml/
- `model_comparison.py` — Entrenamiento y comparación de los modelos de propensión
  (Random Forest AUC 0,885 vs XGBoost AUC 0,913).
- `kmeans_exploratorio.py` — Análisis no supervisado de contraste (K-Means sobre RFM).
  Método del codo y Silhouette Score.
  Al ejecutarlo se crea localmente la carpeta no versionada `06_graficos/` con la figura generada. Verifica que la estructura natural de la
  población converge con la segmentación por tiers de LTV. NO es el pilar de
  segmentación, solo herramienta exploratoria. Uso: `python kmeans_exploratorio.py`

### 04_pricing_simulacion/
- `pricing_engine_v3.py` — Motor de pricing: catálogo de 14 puntos, ventanas de 5
  precios. Sistema estático (ventana del catálogo que avanza según el precio
  comprado) y dinámico (ventana centrada en la CZ: −20/−10/CZ/+10/+20%,
  redondeada al alza).
- `abtest_v3_vec.py` — A/B test vectorizado (versión sin modelo de error de estimación).
  Su resultado (+75,4%, en `abtest_v3_resultado.csv`) es HISTÓRICO y queda superado
  por `abtest_v4.py`; se conserva como trazabilidad de las iteraciones del desarrollo.
- `abtest_v4.py` — **A/B test canónico**. Modelo realista: disposición de pago real
  vs CZ estimada con ruido, con análisis de sensibilidad. Produce el resultado
  principal (+70,3%). Uso: `python abtest_v4.py`
- `generar_csv_dashboard.py` — **Generador de los CSVs del dashboard**. Replica
  exactamente las 40 réplicas del A/B canónico (importa `abtest_v4.py`, mismas
  semillas y orden de RNG) y produce `resultado_por_tier_v4.csv` y
  `pbi_strategy_long_v2.csv`. Uso: `python generar_csv_dashboard.py`
- `abtest_robustez.py` — **Pruebas de robustez**. Reutiliza la lógica de `abtest_v4.py`
  por import (mismo modelo de elección y estratificación, 40 semillas) y regenera
  las dos cifras de robustez de la sección 4.6: (1) dinámico restringido al catálogo
  común → **+32,4 %** (ARPU 64,8 → 85,7 \$), y (2) check con cz_raw (sin decaimiento)
  → **+74,3 %**, que confirma que el uso de cz_final es conservador. Genera
  `05_datasets/abtest_robustez.csv`. Uso: `python abtest_robustez.py`

### 05_datasets/
- `player_master.parquet` — Tabla de jugadores (71.429 filas): columnas user_id,
  `tier` (tier objetivo asignado en la generación), `archetype` (tier real según el
  LTV final; **columna autoritativa usada para estratificar el A/B**, coincide con
  `tier` en el 99,7%), RFM (frequency, recency_days), engagement, monetary (LTV),
  cz_raw, cz_final.
- `transactions_final.parquet` — Transacciones (143.930 filas): user_id, timestamp,
  amount, product_type. Precios del catálogo de 14 puntos.
- `abtest_v4_sensibilidad_ruido.csv` — Resultados del A/B por nivel de error.
- `abtest_robustez.csv` — Resultados de las dos pruebas de robustez (catálogo común
  +32,4%; check cz_raw +74,3%), generado por `abtest_robustez.py`.
- `resultado_por_tier_v4.csv` — Desglose de ARPU por tier (±25 %, media de las 40
  réplicas). Generado por `04_pricing_simulacion/generar_csv_dashboard.py`.
- `pbi_strategy_long_v2.csv` — Datos para Power BI: ingreso esperado por jugador y
  estrategia (estático vs dinámico), con los 7 tiers de LTV. Generado por
  `04_pricing_simulacion/generar_csv_dashboard.py`. UNIDADES: dólares por jugador
  en el periodo simulado (30 días); la media de `expected_revenue` por estrategia
  reproduce el ARPU canónico (64,82 / 110,22 \$). El uplift como ratio de medias es
  +70,1%, frente al +70,3% de la memoria (media de los uplifts por réplica):
  ambas son correctas y la diferencia es puramente de agregación.

### 06_logs/
Salidas de consola reales de la ejecución de referencia de los scripts que
producen las cifras citadas en la memoria. Cada log incluye una cabecera con
fecha, máquina y versión de Python (`[Ejecutado AAAA-MM-DD HH:MM:SS | host |
Python X.Y.Z]`), impresa por los propios scripts, como rastro de ejecución.
- `log_abtest_v4.txt` — Resultado principal (+70,3%) y sensibilidad al error.
- `log_abtest_robustez.txt` — Robustez: catálogo común (+32,4%) y cz_raw (+74,3%).
- `log_model_comparison.txt` — AUC 0,913 (XGBoost) vs 0,885 (Random Forest).
- `log_kmeans_exploratorio.txt` — Codo y Silhouette (k=2, S=0,772).
- `log_generar_csv_dashboard.txt` — Regeneración de los CSVs del dashboard
  (verificación: media 64,82 / 110,22 \$ por estrategia).
- `log_comfort_zone.txt`, `log_validate_cz.txt` — Autotest y validación de la CZ
  (7 tramos de decaimiento, filtro de ruido 2,99 \$).

### 07_notebooks/
Entorno Jupyter de reproducción. Cada notebook ejecuta el script canónico del
paquete mediante `%run` (código fuente único, sin duplicación) e incluye los
outputs de la ejecución de referencia con las cifras canónicas guardadas:
- `00_reproduccion_completa.ipynb` — Lanzadera: ejecuta los cinco bloques en
  orden (~15–20 min). Pensado para la reproducción interactiva completa.
- `01_comfort_zone_y_validacion.ipynb` — Autotest del decaimiento y validación CZ.
- `02_model_comparison.ipynb` — RF vs XGBoost (AUC 0,885 / 0,913).
- `03_kmeans_exploratorio.ipynb` — Codo y Silhouette (k=2, S=0,772).
- `04_abtest_v4.ipynb` — Resultado principal (+70,3%) y sensibilidad.
- `05_abtest_robustez.ipynb` — Robustez (+32,4% y +74,3%).

Requisito adicional para los notebooks: `jupyter` (`pip install jupyter`).

---

## Reproducir el pipeline completo

```
cd 01_generacion_datos && python regenerate_pipeline.py   # genera los parquet
cd ../04_pricing_simulacion && python abtest_v4.py          # reproduce el +70,3 %
cd ../04_pricing_simulacion && python abtest_robustez.py    # reproduce +32,4 % y +74,3 %
```

Dependencias (versiones exactas del entorno de referencia donde se reprodujeron
todas las cifras de la memoria):

```
Python 3.12.3
pandas==3.0.2
numpy==2.4.4
scipy==1.17.1
scikit-learn==1.8.0
xgboost==3.3.0
pyarrow==24.0.0
matplotlib==3.10.8
```

Nota de reproducibilidad: las cifras principales (+70,3%, +32,4%, +74,3%,
sensibilidad 57,7–78,6%, AUC XGBoost 0,913, Silhouette 0,772) son deterministas
y se reproducen exactamente con las semillas fijadas. Las métricas secundarias de
clasificación (precisión, recall, F1 y tercer decimal del AUC del Random Forest)
pueden variar ligeramente (±0,01) con otras versiones de scikit-learn/xgboost.

---

## Notas de coherencia

- **Catálogo unificado**: generación, simulación y memoria usan el mismo catálogo
  de 14 puntos (0,99–299,99 \$).
- **Disposición de pago**: en la simulación se aproxima con la Comfort Zone (con su
  decaimiento aplicado). Es una elección conservadora: sin decaimiento (cz_raw), el
  uplift ascendería a +74,3% frente al +70,3% de referencia (ver `abtest_robustez.py`).
  Documentado en la sección 4.2.8 de la memoria.
- **Dos definiciones de ARPU**: el ARPU histórico del dataset (48,07 dólares) describe los
  datos de partida; el ARPU simulado (64,8 / 110,2 dólares) cuantifica el resultado del
  experimento A/B. Son magnitudes distintas, no contradictorias.
