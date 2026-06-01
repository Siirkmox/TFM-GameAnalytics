# Spec: 03_eda.ipynb — Análisis Exploratorio de Datos (EDA)

> **Estado: PARCIALMENTE COMPLETADO**  
> Tareas del roadmap original añadidas como pendientes al final del documento.

## Design

**Objetivo:**
Explorar visualmente los datos de sesiones, niveles y salas para identificar
patrones de comportamiento del jugador, dificultad por nivel/sala, uso y
eficiencia de hechizos, y hallazgos clave que guíen los análisis posteriores.

**Entradas:**
- `data/processed/sessions.parquet`, `levels.parquet`, `rooms.parquet`
- `src/preprocessing.py` — `run_pipeline()`, `compute_spell_efficiency()`, `compute_spell_usage_ratios()`, `rooms_to_long()`
- `src/cleaning.py` — `filter_clean()`, `check_sample_size()`

**Salidas:**
- `data/exports/eda_hallazgos.csv` — 10 hallazgos clave del EDA
- 15 figuras en `reports/figures/fig2_*.png` a `fig7_*.png`

**Paleta de colores por elemento:**
```python
PALETTE_ELEM = {'Fire': '#e74c3c', 'Water': '#3498db', 'Earth': '#27ae60', 'Wind': '#f39c12'}
```

**Conexión con proyectos anteriores:**
- **P2 (EDA avanzado):** mismo patrón de bloques temáticos con conclusión al final de cada uno
- **P3 (R estadístico):** los hallazgos del EDA son las hipótesis a validar en Fase 3

**Secciones del notebook (8 bloques, 15 figuras):**
1. Configuración
2. Sesiones globales — distribuciones, win rate, popularidad por elemento
3. Análisis por nivel — funnel, dificultad, grinding (attempt)
4. Análisis por sala — heatmap de muertes, firstSpell táctico, salas peligrosas
5. Análisis de hechizos — uso vs eficiencia, noMana, uso relativo
6. Análisis de enemigos — kills, blocked, damage
7. Comentarios de jugadores — overview cualitativo
8. Resumen ejecutivo — exportar `eda_hallazgos.csv`

---

## Requirements

**Funciones src/ utilizadas:**
- `run_pipeline(raw_path)` — cargar los tres DataFrames
- `filter_clean(df_sessions)` — excluir sesiones sospechosas para análisis estadístico
- `check_sample_size(df, 'playerElement', 30)` — advertir n<30
- `compute_spell_efficiency(df_rooms)` — tabla spell_type, efficiency
- `compute_spell_usage_ratios(df_rooms, df_sessions)` — ratio_cast_* por sesión
- `rooms_to_long(df_rooms, 'cast')` — formato tidy para comparar hechizos
- `rooms_to_long(df_rooms, 'kills')` — formato tidy para comparar enemigos

**Figuras generadas:**

| Figura | Contenido |
|--------|-----------|
| `fig2_1_distribuciones_sesion.png` | Histogramas totalTimeSecs, totalKills, totalDeaths |
| `fig2_2_winrate_elemento.png` | Barplot win rate por playerElement |
| `fig2_3_popularidad_elemento.png` | Countplot sesiones por elemento |
| `fig3_1_funnel_niveles.png` | Barplot sesiones que llegan a cada nivel |
| `fig3_2_dificultad_nivel.png` | Heatmap deaths + damageTaken por nivel |
| `fig3_3_grinding_attempt.png` | Barplot attempt medio por nivel |
| `fig4_1_heatmap_muertes_sala.png` | Heatmap deaths por (levelId, roomId) |
| `fig4_2_firstspell_tactica.png` | Countplot firstSpell por levelId |
| `fig5_1_uso_vs_eficiencia.png` | Scatter total_cast vs spell_efficiency |
| `fig5_2_nomana_demanda.png` | Barplot noMana_* (demanda insatisfecha de maná) |
| `fig5_3_uso_relativo.png` | Stacked bar ratio_cast_* por sesión |
| `fig6_1_kills_enemigo.png` | Barplot kills totales por tipo de enemigo |
| `fig6_2_damage_enemigo.png` | Barplot damageTaken por tipo de enemigo |
| `fig7_1_longitud_comentarios.png` | Histograma longitud de comentarios |
| `fig7_2_comentarios_elemento.png` | Countplot comentarios por elemento |

**Hallazgos clave exportados en `eda_hallazgos.csv`:**
1. Win rate global: 9.1% (1/11 victorias) — dificultad muy alta
2. Fire es el elemento más jugado (7/11 sesiones)
3. Beam: eficiencia 2.04 kills/cast — el hechizo más letal por uso
4. AOE: 0 kills a pesar de ser usado — posible bug de telemetría
5. damage_Unknown: valores altos — bug de telemetría en Unity
6. Nivel 1 es el cuello de botella del funnel (mayor abandono)
7. Grinding detectado: attempt>1 en niveles medios
8. firstSpell varía por nivel → comportamiento táctico del jugador
9. noMana frecuente en Blast → candidato a reducir coste de maná
10. Projectile: hechizo más usado pero eficiencia baja (0.017)

**Librerías requeridas:**
- `matplotlib`, `seaborn` — visualizaciones estáticas
- `pandas`, `numpy` — manipulación

**Restricciones:**
- Todas las figuras guardadas en `reports/figures/` con nombres estandarizados `fig{bloque}_{n}_*.png`
- Cada bloque termina con celda Markdown de conclusiones
- `check_sample_size()` ejecutado al inicio con advertencia visible
- `data/exports/eda_hallazgos.csv` exportado al final con columnas: `id`, `hallazgo`, `valor`, `implicacion`

---

## Tasks (completadas)

- [x] Setup y carga de datos con `run_pipeline()` — **TFM-13**
- [x] Bloque 2: distribuciones globales y win rate por elemento — **TFM-34**
- [x] Bloque 3: funnel de niveles, dificultad, análisis de grinding (attempt) — **TFM-34**
- [x] Bloque 4: heatmap de muertes por sala, análisis firstSpell táctico — **TFM-35**
- [x] Bloque 5: uso vs eficiencia de hechizos, noMana, uso relativo — **TFM-35**
- [x] Bloque 6: análisis de enemigos (kills, damage) — **TFM-36**
- [x] Bloque 7: overview de comentarios de jugadores — **TFM-36**
- [x] Bloque 8: resumen ejecutivo + exportar `eda_hallazgos.csv` — **TFM-13**

## Tasks pendientes (del roadmap, completadas)

- [x] **Curva de dificultad esperada vs real** — `fig6j_curva_dificultad.png` — **TFM-34**
- [x] **Matriz de efectividad hechizo × elemento** — `fig6k_matriz_hechizo_elemento.png` — **TFM-35**
- [x] **Correlación tipo de enemigo vs muertes de sala** — `fig6l_enemigo_vs_muertes.png` — **TFM-36**
- [x] **`playerCommentContext` cruzado con elemento y nivel completado** — `fig6m_comentarios_contexto_elemento.png` — **TFM-36**
