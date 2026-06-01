# Spec: 07_balance_analysis.ipynb — Análisis de Balance y Recomendaciones

## Design

**Objetivo:**
Sintetizar todos los análisis anteriores (EDA, estadística, ML, NLP) en conclusiones
accionables de balance del juego. Produce el Balance Index compuesto por elemento
y la tabla de recomendaciones priorizada para el diseñador del juego.

**Entradas:**
- `data/exports/eda_hallazgos.csv` — 10 hallazgos clave del EDA
- `data/exports/statistical_results.csv` — p-values y effect sizes (Fase 3)
- `data/exports/ml_predictions.csv` — predicciones y feature importance (Fase 4a)
- `data/exports/player_clusters.csv` — perfiles de jugador (Fase 4a)
- `data/exports/nlp_results.csv` — sentiment scores (Fase 4b)
- `data/processed/sessions.parquet`, `levels.parquet`, `rooms.parquet`
- `src/analysis.py` — a crear en este notebook
- `src/visualization.py` — plots base (Fase 3)

**Salidas:**
- `data/exports/balance_index.csv` — Balance Index por elemento + desglose de componentes
- `data/exports/balance_recommendations.csv` — tabla de recomendaciones con prioridad
- `reports/balance_report.html` — informe HTML ejecutivo (generado con nbconvert o jinja2)
- `reports/figures/fig7_1_heatmap_elemento_nivel.png`
- `reports/figures/fig7_2_salas_abandono.png`
- `reports/figures/fig7_3_balance_index.png`
- `reports/figures/fig7_4_perfiles_elemento.png`

**Conexión con proyectos anteriores:**
- **P7 (Análisis de Producción):** mismo patrón de tabla de recomendaciones accionables con prioridad (Alta/Media/Baja)
- **P3 (R estadístico):** citar p-values de Fase 3 como evidencia estadística en cada recomendación

**Visualizaciones previstas:**
1. Heatmap elemento × nivel: tasa de muerte media por celda (seaborn heatmap)
2. Barplot horizontal: top-10 salas con mayor abandono (damageTaken + deaths combinados)
3. Barplot: Balance Index por elemento, coloreado con paleta de elementos
4. Radar chart / spider plot: perfil de cada elemento (win_rate, sentiment, kills_efficiency, cast_accuracy)

---

## Requirements

**Funciones a crear en `src/analysis.py`:**

```python
def compute_balance_index(df_sessions: pd.DataFrame, nlp_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Balance Index = win_rate × sentiment_score × kills_efficiency por elemento.
    win_rate: proporción de isVictory por playerElement
    sentiment_score: media de sentiment_score de nlp_results por playerElement (NaN → 0.5)
    kills_efficiency: totalKills / totalTimeSecs (normalizado 0-1 min-max por elemento)
    Devuelve DataFrame con columnas: playerElement, win_rate, sentiment_score,
                                     kills_efficiency, balance_index
    """

def generate_recommendations(balance_index: pd.DataFrame,
                              statistical_results: pd.DataFrame,
                              spell_efficiency: pd.DataFrame,
                              eda_hallazgos: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tabla de recomendaciones de balance.
    Columnas: element, area, finding, recommendation, priority (Alta/Media/Baja), evidence
    """

def element_level_matrix(df_sessions: pd.DataFrame, df_levels: pd.DataFrame,
                          df_rooms: pd.DataFrame, metric: str = 'death_rate') -> pd.DataFrame:
    """
    Construye matriz elemento × nivel con la métrica indicada.
    metric: 'death_rate' | 'completion_rate' | 'avg_damage'
    """

def top_abandoned_rooms(df_rooms: pd.DataFrame, df_sessions: pd.DataFrame,
                        n: int = 10) -> pd.DataFrame:
    """
    Identifica las n salas con mayor abandono.
    Score = deaths + (damageTaken / damageTaken.max()).
    """
```

**Fórmula Balance Index:**
```
kills_efficiency_norm(e) = (kills_per_min(e) - min(kills_per_min)) / (max - min)
balance_index(e) = win_rate(e) × sentiment_score(e) × (1 + kills_efficiency_norm(e))
```
Nota: si sentiment_score no disponible (sin comentarios) → usar 0.5 (neutral)

**Tabla de recomendaciones — áreas:**
- `spell_balance` — rebalanceo de hechizos (AOE=0 kills, Beam demasiado eficiente)
- `difficulty` — dificultad por nivel/sala (salas de mayor abandono)
- `element_balance` — diferencias entre elementos (si estadísticamente significativas)
- `telemetry` — bugs de telemetría detectados (Unknown damage, AOE sin kills)
- `sample_size` — recomendación de recolección de más datos

**Librerías requeridas:**
- `pandas`, `numpy` — cálculos
- `matplotlib`, `seaborn` — heatmaps y barplots
- `plotly.express` — gráficos interactivos para el HTML (opcional)

**Outputs mínimos:**
- `balance_index.csv` con Balance Index para cada elemento
- `balance_recommendations.csv` con al menos 5 recomendaciones priorizadas
- `balance_report.html` exportado
- 4 figuras en `reports/figures/`

**Restricciones:**
- Toda recomendación debe citar evidencia de al menos una fase anterior
- El Balance Index debe incluir nota de limitación por n=11 sesiones
- Prioridad Alta: solo si hay evidencia estadística o bug confirmado; Media: tendencia EDA; Baja: observación cualitativa

---

## Tasks

### Bloque 1 — Carga y merge de todos los exports
- [x] Cargar todos los CSV de `data/exports/` y los Parquet procesados
- [x] nlp_results ya incluye playerElement (no requiere merge adicional)
- [x] Merge `player_clusters` con `sessions` por `sessionId`
- [x] Verificar cobertura: 24 sesiones, 5 con nlp_results
- [x] **Jira: TFM-16**

### Bloque 2 — Crear `src/analysis.py`
- [x] Implementar `compute_balance_index()`
- [x] Implementar `element_level_matrix()`
- [x] Implementar `top_abandoned_rooms()`
- [x] Implementar `generate_recommendations()`
- [x] **Jira: TFM-43**

### Bloque 3 — Heatmap elemento × nivel
- [x] Llamar `element_level_matrix()` con metric='death_rate'
- [x] Visualizar con `seaborn.heatmap()` → `fig7_1_heatmap_elemento_nivel.png`
- [x] Repetir para metric='completion_rate'
- [x] **Jira: TFM-29**

### Bloque 4 — Salas con mayor abandono
- [x] Llamar `top_abandoned_rooms(df_rooms, df_sessions, n=10)`
- [x] Barplot horizontal de top salas → `fig7_2_salas_abandono.png`
- [x] **Jira: TFM-17**

### Bloque 5 — Perfil por elemento
- [x] Calcular win_rate, cast_accuracy, kd_ratio, kills_per_min por elemento
- [x] Radar chart → `fig7_4_perfiles_elemento.png`
- [x] Identificar fortalezas y debilidades por elemento
- [x] **Jira: TFM-30**

### Bloque 6 — Balance Index
- [x] Llamar `compute_balance_index(df_sessions, nlp_results)`
- [x] Fire=0.3125, Water=0.1250, Earth=0.0, Wind=0.0
- [x] Barplot → `fig7_3_balance_index.png`
- [x] Exportar `balance_index.csv`
- [x] **Jira: TFM-44**

### Bloque 7 — Tabla de recomendaciones
- [x] Llamar `generate_recommendations()` con todos los inputs
- [x] 10 recomendaciones: 6 Alta, 3 Media, 1 Baja
- [x] Exportar `balance_recommendations.csv`
- [x] **Jira: TFM-31, TFM-32**

### Bloque 8 — Exportar informe HTML
- [x] Informe HTML generado en Fase 6 (`08_report.ipynb`) — **TFM-43**

## Tasks pendientes (del roadmap, completadas)

- [x] **Balance de enemigos — `blocked` alto, `damage` alto, raramente matados** — `fig8b_balance_enemigos.png` (3 subplots) — **TFM-17**
- [x] **Efectos de estado (`status_*`) vs kills** — Spearman por efecto, `fig8c_status_vs_kills.png` — **TFM-32**
