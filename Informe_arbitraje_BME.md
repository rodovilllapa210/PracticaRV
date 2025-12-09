# Informe de Arbitraje BME

### Rodolfo Villena Lapaz

## Resumen Ejecutivo

Análisis de oportunidades de arbitraje en acciones españolas cotizadas en múltiples centros de negociación (BME, CBOE, AQUIS, TURQUOISE) durante una sesión de trading.

## 1. Metodología

### 1.1. Datos y Preprocesamiento
- **Fuentes**: BME (principal), CBOE, AQUIS, TURQUOISE
- **Período**: Una sesión completa de trading
- **Filtros aplicados**:
  - Eliminación de cantidades nulas o NaN
  - Filtro de addressability: conservar snapshots con código de negociación continua
  - Construcción de marca temporal única mediante "truco del nanosegundo"

### 1.2. Filtros de Calidad
Aplicados sobre señales de arbitraje (rising edges):
- Duración mínima: ≥ 5 ms
- Duración máxima: ≤ 60 s (evitar cotizaciones obsoletas)
- Spread máximo: ≤ €50
- Cantidad negociable mínima: ≥ 5 acciones
- Detección de anomalías de profundidad (spreads del 100% con cantidades ínfimas)

### 1.3. Construcción de Cinta Consolidada
Para cada ISIN:
- Índice: timestamp
- Columnas: bid/ask y cantidades de cada venue
- Cálculos en cada instante:
  - Max Bid global y venue vendedor
  - Min Ask global y venue comprador
  - Spread = Max Bid – Min Ask
  - Cantidad ejecutable = min(qty_bid_max, qty_ask_min)
  - Beneficio teórico = spread × cantidad

### 1.4. Simulación de Latencia
Para cada rising edge en T:
- Simulación de ejecución en T + Δ
- Δ ∈ {0, 100, 500, 1.000, 2.000, 3.000, 4.000, 5.000, 10.000, 15.000, 20.000, 30.000, 50.000, 100.000} μs
- Recálculo del beneficio en T+Δ usando merge_asof vectorizado

## 2. Resultados

### 2.1. Existencia de Oportunidades de Arbitraje
**Sí, existen oportunidades significativas:**
- 1.034 oportunidades de trading únicas (rising edges)
- Beneficio total teórico a latencia 0: €1.184,11
- Distribución con cola gruesa: mayoría de ISINs generan beneficios modestos, pocos concentran beneficios elevados

**Concentración por ISIN:**
- ISIN más rentable: ~€600 (más del 50% del beneficio total)
- Top 10 ISINs concentran prácticamente todo el P&L
- Resto de ISINs aportan cantidades marginales

### 2.2. Análisis por Pares de Venues
**Dominancia BME-CBOE:**
- BME→CBOE: ~€393 de beneficio, 396 oportunidades
- CBOE→BME: ~€384 de beneficio, 319 oportunidades
- Otros pares: beneficios significativamente menores (€3-€130)

**Conclusión operativa:** Un engine focalizado en el par BME-CBOE captura la mayoría del valor.

### 2.3. Curva de Latency Decay

| Latencia | Beneficio Total | Pérdida vs 0µs |
|----------|----------------|----------------|
| 0 µs     | €1.184,11      | 0%             |
| 1 ms     | €1.176,18      | ~0,7%          |
| 10 ms    | €613,13        | ~48,2%         |
| 100 ms   | ~€150          | ~87%           |

**Interpretación:**
- Tramo 0-1 ms: casi plano (la mayoría de oportunidades sobreviven)
- Tramo 1-10 ms: fuerte caída del beneficio
- Tramo 10-100 ms: caída progresiva hacia nivel asintótico bajo

## 3. Conclusiones para Prop Trading

### 3.1. Viabilidad de la Estrategia
- **Sí existen oportunidades** de arbitraje en acciones españolas
- Beneficio máximo teórico: ~€1,18k por sesión
- Altamente concentrado en pocos ISINs y principalmente en par BME-CBOE

### 3.2. Requisitos de Infraestructura
- **Prioridad absoluta**: conectividad de baja latencia entre BME y CBOE
- **Latencia óptima**: sub-milisegundo para capturar ~99% del beneficio teórico
- **Umbral crítico**: a partir de 10 ms se pierde más del 50% del P&L

### 3.3. Recomendaciones Estratégicas
1. **Focalización**: Concentrarse en ISINs recurrentes en Top-10 por beneficio
2. **Simplificación**: Excluir ISINs con P&L casi nulo para reducir complejidad
3. **Infraestructura**: Invertir prioritariamente en conectividad BME-CBOE

## 4. Limitaciones y Extensiones

### 4.1. Limitaciones Actuales
- Análisis sólo en mejor nivel de libro (niveles 1-10 no explorados)
- No se consideran costes de transacción, fees, rebates ni slippage
- Supuesto de ejecución completa sin impacto de mercado

### 4.2. Extensiones Propuestas
1. Incorporar costes explícitos para estimar P&L neto
2. Analizar estabilidad temporal (varias sesiones, distintos regímenes)
3. Estimar beneficio incremental de reducción de latencia
4. Explorar niveles más profundos del libro de órdenes
5. Modelar impacto de mercado y slippage realista

---

*Informe generado a partir del análisis de datos de trading y simulación de estrategias de arbitraje estadístico.*
