# Spec: 04_statistics.ipynb — Análisis Estadístico

## Design

**Objetivo:**
Validar con tests estadísticos formales si existen diferencias significativas entre los
cuatro elementos (Fire, Water, Earth, Wind) en los principales KPIs de rendimiento del jugador.
Complementa el EDA visual con evidencia estadística citable en el informe final.

**Entradas:**
- `data/processed/sessions.parquet` — df_sessions con features derivadas
- `data/processed/rooms.parquet` — df_rooms con columnas dinámicas
- `src/preprocessing.py` — `run_pipeline()`, `compute_spell_usage_ratios()`
- `src/cleaning.py` — `filter_clean()`, `check_sample_size()`
- `src/visualization.py` — funciones de plots (a crear en este notebook)

**Salidas:**
- `data/exports/statistical_results.csv` — tabla con todos los p-values y effect sizes
- `reports/figures/fig4_1_normalidad.png` — Q-Q plots por KPI
- `reports/figures/fig4_2_boxplots_elemento.png` — boxplots de KPIs por elemento
- `reports/figures/fig4_3_correlaciones.png` — heatmap de correlaciones
- `reports/figures/fig4_4_scatter_victory.png` — scatter KPIs vs isVictory
- `reports/figures/fig4_5_mann_kendall.png` — tendencia temporal de sesiones

**Conexión con proyectos anteriores:**
- **P3 (R estadístico):** misma batería de tests (Shapiro, ANOVA/Kruskal, Dunn), replicada en Python
- **P7 (Análisis de Producción):** patrón de tabla de resultados con p-value + interpretación + recomendación

**Visualizaciones previstas:**
1. Q-Q plots de normalidad para `kd_ratio`, `kills_per_min`, `cast_accuracy` (2×3 subplots)
2. Boxplots por `playerElement` para los 3 KPIs principales
3. Heatmap de correlaciones (Spearman) entre todas las features numéricas
4. Scatter `kd_ratio` vs `isVictory` coloreado por elemento
5. Línea temporal de sesiones con tendencia Mann-Kendall

---

## Requirements

**Funciones src/ a crear — `src/visualization.py`:**
```python
def boxplot_by_element(df, col, palette, title, figsize=(10,5)) -> plt.Figure
def correlation_heatmap(df, cols, method='spearman', figsize=(10,8)) -> plt.Figure
def qq_plot(df, cols, ncols=3, figsize=(15,4)) -> plt.Figure
def scatter_binary(df, x_col, y_col, hue_col, palette, figsize=(8,5)) -> plt.Figure
def temporal_line(df, x_col, y_col, title, figsize=(10,4)) -> plt.Figure
```

**Librerías requeridas:**
- `scipy.stats` — Shapiro-Wilk, ANOVA, Chi-squared, Pearson/Spearman
- `scikit_posthocs` — test de Dunn post-hoc
- `pingouin` — effect sizes (eta², epsilon²), tablas limpias
- `pymannkendall` — tendencia temporal de KPIs
- `matplotlib`, `seaborn` — visualizaciones

**Tests estadísticos:**

| Test | Variable | Hipótesis nula |
|------|----------|----------------|
| Shapiro-Wilk | `kd_ratio`, `kills_per_min`, `cast_accuracy` por elemento | Distribución normal |
| ANOVA one-way | Si Shapiro no rechaza | Medias iguales entre elementos |
| Kruskal-Wallis | Si Shapiro rechaza (esperado con n<30) | Distribuciones iguales |
| Dunn post-hoc | Si KW significativo (p<0.05) | Pares sin diferencia |
| Pearson/Spearman | `kd_ratio` vs `totalTimeSecs`, `levelsCompleted` vs `isVictory` | Sin correlación |
| Chi-squared | `isVictory` × `playerElement` | Independencia |
| Mann-Kendall | `kd_ratio` ordenado por `startDatetime` | Sin tendencia temporal |

**Outputs mínimos para considerar completado:**
- Tabla `statistical_results.csv` con columnas: `test`, `variable`, `statistic`, `p_value`, `effect_size`, `interpretation`
- Al menos 5 figuras exportadas en `reports/figures/`
- Celda de conclusiones al final de cada bloque con interpretación del resultado

**Restricciones:**
- `check_sample_size(df, 'playerElement', min_per_group=30)` al inicio — todos los grupos fallarán (n<30)
- Documentar explícitamente: "Con n=11 sesiones los tests son orientativos. Resultados válidos a partir de n≥30 por elemento."
- α = 0.05 como umbral de significación
- Usar siempre `filter_clean()` antes de los tests para excluir sesiones sospechosas

---

## Tasks

### Bloque 1 — Setup y carga de datos
- [x] Importar librerías (`scipy`, `pingouin`, `pymannkendall`, `scikit_posthocs`)
- [x] Cargar `sessions.parquet` y `rooms.parquet` con `run_pipeline()` o desde Parquet
- [x] Aplicar `filter_clean()` y `check_sample_size()` con advertencia de muestra pequeña
- [x] **Jira: TFM-18**

### Bloque 2 — Crear `src/visualization.py`
- [x] Implementar `boxplot_by_element()` con paleta de elementos
- [x] Implementar `correlation_heatmap()` (Spearman por defecto)
- [x] Implementar `qq_plot()` con subplots automáticos
- [x] Implementar `scatter_binary()` para KPI vs target booleano
- [x] Implementar `temporal_line()` para series temporales
- [x] **Jira: TFM-37**

### Bloque 3 — Normalidad (Shapiro-Wilk)
- [x] Aplicar Shapiro-Wilk a `kd_ratio`, `kills_per_min`, `cast_accuracy` por elemento
- [x] Generar Q-Q plots (`qq_plot()` de visualization.py)
- [x] Tabla de resultados con p-value e interpretación por variable
- [x] Conclusión: Kruskal-Wallis (ningún KPI sigue distribución normal salvo kills_per_min)
- [x] **Jira: TFM-23**

### Bloque 4 — ANOVA / Kruskal-Wallis + Dunn
- [x] Kruskal-Wallis aplicado — `kills_per_min` p=0.023 (significativo), resto no significativo
- [x] Dunn post-hoc no aplicado (solo kills_per_min significativo, muestra muy pequeña)
- [x] Effect size epsilon² calculado
- [x] Boxplots por elemento generados (`fig4_2_boxplots_elemento.png`)
- [x] **Jira: TFM-24**

### Bloque 5 — Correlaciones
- [x] Heatmap Spearman generado (`fig4_3_correlaciones.png`)
- [x] Scatter cast_accuracy vs isVictory (`fig4_4_scatter_victory.png`)
- [x] Correlaciones significativas: kd_ratio, kills_per_min, levelsCompleted, cast_accuracy vs isVictory
- [x] **Jira: TFM-25, TFM-37**

### Bloque 6 — Chi-squared: victoria vs elemento
- [x] Tabla de contingencia generada
- [x] Fisher exact aplicado (frecuencias esperadas < 5)
- [x] Barras apiladas proporciones (`fig4_5_victoria_elemento.png`)
- [x] Conclusión: sin asociación significativa (p=0.362)
- [x] **Jira: TFM-26**

### Bloque 7 — Mann-Kendall: tendencia temporal
- [x] Mann-Kendall aplicado a kd_ratio, kills_per_min, totalDeaths
- [x] kills_per_min tendencia decreciente (p=0.002) — testers más recientes juegan peor
- [x] Gráfico tendencia temporal (`fig4_6_mann_kendall.png`)
- [x] **Jira: TFM-38**

### Bloque 8 — Exportar resultados
- [x] `statistical_results.csv` exportado — 20 tests
- [x] Conclusiones finales del notebook
- [x] **Jira: TFM-18**

## Tasks pendientes (del roadmap, no implementadas)

- [ ] **Chi-cuadrado: uso de hechizos vs victoria** — tabla de contingencia `hechizo_más_usado` × `isVictory`; ¿existe asociación entre el hechizo predominante en una sesión y ganar? Usar `ratio_cast_*` para definir hechizo dominante por sesión — **TFM-26**
- [ ] **Correlación `attempt` vs `deaths` por nivel** — Spearman entre número de intento (`attempt` en df_levels) y muertes en ese intento; responde: ¿los jugadores que repiten un nivel mejoran o empeoran? — **TFM-25**
- [ ] **`timeSecs` por sala vs `deaths` en esa sala** — Spearman a nivel de sala; ¿las salas en las que el jugador pasa más tiempo son también las que más muertes acumulan? — **TFM-25**
