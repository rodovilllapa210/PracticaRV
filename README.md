# Práctica de Arbitraje en Mercados Financieros

### Rodolfo Villena Lapaz

Detección de oportunidades de arbitraje cross-venue con simulación de latencia en mercados españoles.

## Características Principales

- ✅ **Análisis multi-venue**: BME, CBOE, AQUIS, TURQUOISE
- ✅ **Simulación de latencia**: 14 niveles desde 0μs hasta 100ms
- ✅ **Filtros de calidad**: Eliminación de oportunidades sospechosas o no ejecutables
- ✅ **Análisis de pares de venues**: Identificación de direccionalidad del arbitraje
- ✅ **Procesamiento escalable**: 5 ISINs (rápido) o 64 ISINs (completo)
- ✅ **Visualizaciones avanzadas**: Heatmaps, distribuciones, decay charts
- ✅ **Resultados realistas**: Diferenciación entre oportunidades teóricas vs. ejecutables

## Requisitos del Sistema

- Python 3.8 o superior
- ~4 GB de RAM (para procesar todos los ISINs)
- ~500 MB de espacio en disco para los datos

## Instalación

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv .venv
```

### 2. Activar entorno virtual

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
.
├── PracticaRV.py              # Script principal (ejecutable)
├── utilities.py               # Funciones auxiliares
├── requirements.txt           # Dependencias Python
├── run_all_isins.py          # Script para ejecutar con todos los ISINs
├── README.md                  # Este archivo
├── INSTRUCCIONES.txt          # Guía rápida en español
├── DATA_BIG/                 # Carpeta con datos de mercado (no incluida)
│   ├── BME_2025-11-07/
│   ├── CBOE_2025-11-07/
│   ├── AQUIS_2025-11-07/
│   └── TURQUOISE_2025-11-07/
└── [Archivos generados]      # Ver sección "Salidas Generadas"
```

## Uso

### Ejecución básica (5 ISINs seleccionados - RÁPIDO)

```bash
python PracticaRV.py
```

Esto procesará 5 ISINs principales españoles en ~40 segundos.

### Ejecución con todos los ISINs disponibles (COMPLETO)

**Opción A - Usando script auxiliar (recomendado):**
```bash
python run_all_isins.py
```

**Opción B - Editando manualmente:**

Editar `PracticaRV.py` línea 31 y cambiar:
```python
USE_ALL_ISINS = True  # En lugar de False
```

Luego ejecutar:
```bash
python PracticaRV.py
```

Esto procesará los 64 ISINs disponibles en ~8-10 minutos.

## Salidas Generadas

El script genera los siguientes archivos:

### Tablas CSV

1. **money_table.csv**
   - Profit realizado por ISIN y nivel de latencia
   - Formato: Filas = ISINs, Columnas = Latencias

2. **venue_pairs_analysis.csv**
   - Análisis detallado de pares de venues
   - Columnas: sell_venue, buy_venue, direction, opportunities, total_profit, avg_profit, avg_spread, avg_quantity
   - Permite identificar qué pares son más rentables y frecuentes

### Gráficos PNG (Alta Resolución - 300 DPI)

3. **decay_chart_total.png**
   - Profit total vs latencia (gráfico principal)
   - Incluye estadísticas de decay a 1ms y 10ms
   - Líneas verticales en latencias clave

4. **decay_chart_by_isin.png**
   - Profit por ISIN vs latencia
   - Muestra solo los Top 10 ISINs más rentables
   - Evita saturación visual con muchos ISINs

5. **profit_distribution.png**
   - Histogramas de distribución de profits
   - Tres paneles: 0μs, 1ms, 10ms
   - Incluye media y mediana

6. **venue_pairs_heatmap.png**
   - Dos heatmaps lado a lado:
     * Izquierda: Total Profit por par de venues (€)
     * Derecha: Número de Oportunidades por par
   - Filas = Venue de venta (max bid)
   - Columnas = Venue de compra (min ask)

7. **venue_pairs_top10.png**
   - Dos gráficos de barras:
     * Superior: Top 10 pares más rentables
     * Inferior: Top 10 pares con más oportunidades
   - Formato direccional: A→B indica vender en A, comprar en B

## Configuración

Puedes modificar los siguientes parámetros en `PracticaRV.py`:

```python
SESSION = '2025-11-07'              # Fecha de la sesión
DATA_FOLDER = 'DATA_BIG'            # Carpeta con datos
USE_ALL_ISINS = False               # True para procesar todos los ISINs
MAX_ISINS_TO_PLOT = 10              # Número máximo de ISINs en gráficos

LATENCY_LEVELS = [0, 100, 500, 1000, 2000, 3000, 4000, 5000, 
                  10000, 15000, 20000, 30000, 50000, 100000]

# Filtros de calidad para oportunidades
ENABLE_QUALITY_FILTERS = True       # Activar/desactivar filtros

FILTER_CONFIG = {
    "min_duration_ms": 5.0,          # Duración mínima razonable (5ms)
    "max_duration_ms": 60_000.0,     # Límite para stale quotes (60s)
    "max_spread_eur": 50.0,          # Spread máximo razonable (€50)
    "min_tradable_qty": 5,           # Mínimo aceptable (5 shares)
    "depth_100_qty_max": 3,          # Depth=100% pero qty<3 → sospechoso
}
```

## Filtros de Calidad para Oportunidades

El sistema incluye **filtros de calidad** para eliminar oportunidades de arbitraje sospechosas o no realistas, mejorando la robustez de los resultados.

### ¿Por qué filtrar?

En datos de mercado reales, pueden aparecer oportunidades que técnicamente cumplen la condición `max_bid > min_ask` pero que:
- Son **demasiado breves** para ejecutar (< 5ms)
- Son **stale quotes** (datos desactualizados > 60s)
- Tienen **spreads irreales** (> €50)
- Tienen **cantidades no ejecutables** (< 5 shares)
- Presentan **anomalías de profundidad** (depth 100% pero qty muy baja)

### Filtros Implementados

1. **Duración Mínima** (`min_duration_ms: 5.0`)
   - Elimina oportunidades que duran menos de 5ms
   - Razón: Imposibles de ejecutar con latencia de red real

2. **Duración Máxima** (`max_duration_ms: 60000.0`)
   - Elimina oportunidades que duran más de 60 segundos
   - Razón: Probablemente stale quotes (datos no actualizados)

3. **Spread Máximo** (`max_spread_eur: 50.0`)
   - Elimina oportunidades con spread > €50
   - Razón: Spreads irreales, probablemente errores de datos

4. **Cantidad Mínima** (`min_tradable_qty: 5`)
   - Elimina oportunidades con menos de 5 shares disponibles
   - Razón: Volúmenes demasiado pequeños para ser rentables

5. **Anomalía de Profundidad** (`depth_100_qty_max: 3`)
   - Elimina oportunidades con depth=100% pero qty ≤ 3
   - Razón: Indicador de datos sospechosos

### Impacto de los Filtros (5 ISINs)

| Métrica | Sin Filtros | Con Filtros | Reducción |
|---------|-------------|-------------|-----------|
| **Oportunidades** | 412 | 177 | **-57%** |
| **Profit (0 latency)** | €661.07 | €281.53 | **-57%** |
| **Profit (1ms)** | €628.54 | €260.99 | **-58%** |
| **Profit (10ms)** | €381.57 | €169.17 | **-56%** |

### Impacto por ISIN

| ISIN | Nombre | Oportunidades | Reducción |
|------|--------|---------------|-----------|
| ES0113211835 | Inditex | 65 → 21 | **-68%** |
| ES0113900J37 | Santander | 295 → 146 | **-51%** |
| ES0144580Y14 | Telefónica | 16 → 4 | **-75%** |
| ES0178430E18 | Iberdrola | 33 → 5 | **-85%** |
| ES0113679I37 | BBVA | 3 → 1 | **-67%** |

### Interpretación

- ✅ **Más de la mitad** de las oportunidades detectadas eran cuestionables
- ✅ Los resultados **filtrados** son más **realistas** y **ejecutables**
- ✅ **Iberdrola** tuvo la mayor reducción (85%) - sugiere más ruido en datos
- ✅ **Santander** tuvo la menor reducción (51%) - datos más limpios
- ✅ El **decay pattern** se mantiene similar (~40% a 10ms)

### Desactivar Filtros

Para comparar resultados o análisis académico, puedes desactivar los filtros:

```python
ENABLE_QUALITY_FILTERS = False  # En PracticaRV.py línea 73
```

### Ajustar Filtros

Puedes ajustar los parámetros según tus necesidades:

```python
# Más conservador (para trading institucional)
FILTER_CONFIG = {
    "min_duration_ms": 10.0,         # Requiere 10ms mínimo
    "max_duration_ms": 30_000.0,     # Máximo 30s
    "max_spread_eur": 10.0,          # Spread máximo €10
    "min_tradable_qty": 100,         # Mínimo 100 shares
    "depth_100_qty_max": 10,         # Más estricto
}

# Más permisivo (para análisis exploratorio)
FILTER_CONFIG = {
    "min_duration_ms": 1.0,          # Acepta 1ms
    "max_duration_ms": 120_000.0,    # Hasta 2 minutos
    "max_spread_eur": 100.0,         # Spread hasta €100
    "min_tradable_qty": 1,           # Acepta 1 share
    "depth_100_qty_max": 1,          # Menos estricto
}
```

## Tiempo de Ejecución Estimado

Hardware de referencia: CPU moderna, 8GB RAM

- **5 ISINs**: ~40 segundos
- **10 ISINs**: ~1.5 minutos
- **30 ISINs**: ~4 minutos
- **64 ISINs (todos)**: ~8-10 minutos

## Análisis de Pares de Venues

### ¿Qué es el análisis venue-to-venue?

El análisis identifica oportunidades de arbitraje **direccionales** entre cada par de venues:

- **A→B**: Vender en venue A (max bid) y comprar en venue B (min ask)
- **B→A**: Vender en venue B (max bid) y comprar en venue A (min ask)

### Métricas calculadas por par:

1. **Opportunities**: Número de veces que aparece ese par
2. **Total Profit**: Suma de todos los profits de ese par
3. **Avg Profit**: Profit promedio por oportunidad
4. **Avg Spread**: Diferencia de precio media
5. **Avg Quantity**: Volumen disponible medio

### Interpretación de resultados:

- Si **A→B** tiene más profit que **B→A**, el venue A tiende a tener mejores precios de venta
- Los pares con más oportunidades indican mayor desincronización entre venues
- El análisis ayuda a identificar estrategias de routing óptimas

### Ejemplo de resultados (5 ISINs):

```
Top 3 Pares Más Rentables:
1. CBOE→BME: €250.03 (111 oportunidades)
2. BME→CBOE: €223.00 (59 oportunidades)
3. AQUIS→BME: €82.86 (92 oportunidades)
```

## Troubleshooting

### Error: "No module named 'pandas'"
```bash
pip install -r requirements.txt
```

### Error: "DATA_BIG folder not found"
Asegúrate de que la carpeta `DATA_BIG` está en el mismo directorio que `PracticaRV.py`

### Memoria insuficiente
Reduce el número de ISINs procesados editando la lista `ISINS_TO_PROCESS` en `PracticaRV.py`

### El script se queda colgado en Step 4
Esto es normal, Step 4 (Latency Simulation) puede tardar varios minutos con muchos ISINs. La versión optimizada usa operaciones vectorizadas para máxima velocidad.

## Pasos del Análisis

1. **Step 1: Data Loading & Cleaning**
   - Carga archivos QTE (quotes) y STS (trading status)
   - Filtra magic numbers y precios inválidos
   - Aplica addressability filter (solo continuous trading)
   - Limpia timestamps con nanosecond trick

2. **Step 2 & 3: Consolidated Tape & Arbitrage Detection**
   - Crea consolidated tape combinando todas las venues
   - Detecta oportunidades de arbitraje (max_bid > min_ask)
   - Identifica rising edges (primera aparición)
   - Calcula profit teórico
   - **Aplica filtros de calidad** para eliminar oportunidades sospechosas ⭐ NUEVO

3. **Step 3.5: Venue-to-Venue Pair Analysis** ⭐ NUEVO
   - Analiza pares direccionales de venues
   - Identifica qué pares son más rentables
   - Genera matrices de profit y oportunidades

4. **Step 4: Latency Simulation**
   - Simula ejecución con delays realistas
   - Calcula profit realizado a cada nivel de latencia
   - Usa merge_asof optimizado para máxima velocidad

5. **Results & Visualizations**
   - Genera Money Table
   - Crea gráficos de decay
   - Produce análisis de pares de venues
   - Exporta todos los resultados

## Para la Entrega Final

### Archivos a incluir:

- `PracticaRV.py` (código principal)
- `utilities.py` (funciones auxiliares)
- `money_table.csv` (resultados de latencia)
- `venue_pairs_analysis.csv` (análisis de pares)
- Todos los gráficos PNG generados
- Este README.md

### En el informe, incluir:

1. **Money Table completa** con todos los ISINs y latencias
2. **Impacto de los filtros de calidad**:
   - Comparación con/sin filtros
   - Justificación de los filtros aplicados
   - Interpretación de la reducción de oportunidades
3. **Análisis de pares de venues**:
   - Top 10 pares más rentables
   - Interpretación de la direccionalidad
   - Implicaciones para estrategias de trading
4. **Gráficos generados** con análisis
5. **Análisis del decay de profit** con latencia
6. **Top 5 ISINs más rentables**
7. **Conclusiones sobre**:
   - Importancia de la baja latencia en HFT
   - Impacto de la calidad de datos en resultados
   - Qué pares de venues ofrecen mejores oportunidades
   - Direccionalidad del arbitraje entre venues
   - Estrategias de routing óptimas
   - Diferencia entre oportunidades teóricas vs. ejecutables

## Autor

Rodolfo Villena Lapaz
Diciembre 2025
