# Spec: 02_preprocessing.ipynb — Preprocesamiento y Aplanado

> **Estado: COMPLETADO** ✓  
> Este spec documenta lo que se implementó. No hay tareas pendientes.

## Design

**Objetivo:**
Transformar la jerarquía anidada `sessions → levels → rooms` del JSON raw en tres
DataFrames planos (`df_sessions`, `df_levels`, `df_rooms`), aplicar limpieza con flags
de sesiones sospechosas, generar variables derivadas para EDA y ML, y serializar
los resultados en `data/processed/` como Parquet y CSV.

**Entradas:**
- `data/raw/sessions_raw.json` — salida de `01_extraction.ipynb`
- `src/preprocessing.py` — pipeline de aplanado y feature engineering
- `src/cleaning.py` — validación, outliers, flags de calidad

**Salidas:**
- `data/processed/sessions.parquet` + `sessions.csv`
- `data/processed/levels.parquet` + `levels.csv`
- `data/processed/rooms.parquet` + `rooms.csv`

**Conexión con proyectos anteriores:**
- **P1 (Data Wrangling):** mismo patrón flatten + feature engineering + validación
- **P2 (Limpieza avanzada):** criterios de flag `is_suspicious` (no descartar con muestra pequeña)

**Secciones del notebook:**
1. Configuración — paths, imports
2. Limpieza (TFM-11) — detección de sesiones sospechosas, niveles incompletos
3. Variables calculadas (TFM-12) — features derivadas de sesión y sala
4. Estadísticas descriptivas rápidas — por elemento y por nivel
5. Guardar en data/processed/ — Parquet + CSV
6. Conclusiones

---

## Requirements

**Módulos src/ utilizados:**

**`src/preprocessing.py`** — funciones de aplanado y feature engineering:
```python
run_pipeline(raw_path)          # → (df_sessions, df_levels, df_rooms) completos
rooms_to_long(df_rooms, prefix) # → formato tidy por tipo de hechizo/enemigo
compute_spell_efficiency(df_rooms)        # → spell_type, total_cast, total_kills, spell_efficiency
compute_spell_usage_ratios(df_rooms, df_sessions)  # → ratio_cast_* por sesión (features ML)
```

**`src/cleaning.py`** — validación y limpieza:
```python
flag_suspicious_sessions(df)    # → añade columna is_suspicious
get_cleaning_report(df_s, df_l, df_r)  # → dict con métricas de calidad
print_cleaning_report(report)   # → imprime reporte formateado
validate_sessions(df)           # → lista de errores de integridad
filter_clean(df, verbose=True)  # → solo sesiones no sospechosas
check_sample_size(df, group_col='playerElement', min_per_group=30)  # → advertencia n<30
```

**Columnas derivadas generadas:**

*df_sessions:*
| Columna | Fórmula |
|---------|---------|
| `kd_ratio` | `totalKills / max(totalDeaths, 1)` |
| `kills_per_min` | `totalKills / (totalTimeSecs / 60)` |
| `time_per_level` | `totalTimeSecs / max(levelsCompleted, 1)` |
| `completion_rate` | `levelsCompleted / totalLevels` |
| `totalRooms` | count de salas de la sesión |
| `totalCast` | suma de todos los `cast_X` de la sesión |
| `is_suspicious` | flag de sesión problemática |
| `hasComment` | `playerComment.strip().len() > 0` |
| `startDatetime` / `endDatetime` | Unix timestamp → datetime UTC |

*df_rooms:*
| Columna | Fórmula |
|---------|---------|
| `total_kills_room` | suma de `kills_*` |
| `total_cast_room` | suma de `cast_*` |
| `total_miss_room` | suma de `miss_*` |
| `cast_accuracy` | `(total_cast - total_miss) / max(total_cast, 1)` |
| `kills_per_sec` | `total_kills_room / max(timeSecs, 1)` |

**Criterios de limpieza (umbrales en `cleaning.py`):**
- `MIN_TIME_SECS = 30` — sesiones más cortas: probable prueba o crash
- `MAX_TIME_SECS = 3600` — sesiones más largas: probable AFK
- Sin actividad: `totalKills == 0 AND levelsCompleted == 0`

**Decisión de limpieza:**
Con n=11 sesiones no se descarta ninguna. Solo se marca `is_suspicious=True`.
`filter_clean()` permite excluirlas en análisis estadísticos que requieran datos limpios.

**Librerías requeridas:**
- `pandas`, `numpy` — manipulación de datos
- `pathlib` — rutas multiplataforma
- `pyarrow` o `fastparquet` — serialización Parquet

**Outputs mínimos:**
- Los 6 archivos en `data/processed/` (3 Parquet + 3 CSV)
- Print de shapes: `df_sessions: N filas × M columnas`
- Inventario de hechizos, enemigos y efectos de estado detectados
- Resumen por elemento: sesiones, victorias, kills medias, tiempo medio
- Resumen por nivel: muertes medias, daño medio, tiempo medio, intentos medios

**Restricciones:**
- `data/processed/` no se sube a git
- Los NaN en columnas dinámicas de sala se rellenan con 0 (ausencia = no ocurrió)
- El pipeline es idempotente: re-ejecutar produce los mismos resultados con los mismos datos

---

## Tasks (completadas)

- [x] `flatten_sessions()`, `flatten_levels()`, `flatten_rooms()` en `preprocessing.py` — **TFM-11**
- [x] `add_session_features()` y `add_room_features()` con columnas derivadas — **TFM-12**
- [x] `flag_suspicious_sessions()` en `cleaning.py` con criterios documentados — **TFM-11**
- [x] `get_cleaning_report()`, `validate_sessions()`, `filter_clean()` — **TFM-11**
- [x] `compute_spell_efficiency()` y `compute_spell_usage_ratios()` — **TFM-12**
- [x] `rooms_to_long()` para formato tidy por prefijo — **TFM-12**
- [x] Estadísticas descriptivas por elemento y por nivel — **TFM-12**
- [x] Serialización Parquet + CSV en `data/processed/` — **TFM-12**

**Hallazgos del preprocesamiento:**
- 4 sesiones marcadas como `is_suspicious` (3 muy cortas, 1 outlier de tiempo 3647s)
- Hechizos: Projectile y Blast son los más usados; AOE y Beam puntuales
- Enemigos: 15+ tipos distintos; bosses con kills bajas (esperado)
- Bug detectado: columna `damage_Unknown` con valores altos — posible error de telemetría Unity
