# TFM — Análisis de Balance de Juego: Hack & Slash con Firestore

## Contexto del proyecto

Juego hack & slash donde los jugadores eligen un elemento (Fuego, Agua, etc.) y atraviesan niveles compuestos por salas. Los datos de cada sesión se almacenan en Firestore en una jerarquía de tres niveles:

```
sessions/{sessionId}
  └── levels/{levelId}
        └── rooms/{roomId}
```

**Objetivo final:** usar los datos recogidos para tomar decisiones de balanceo del juego (hechizos, enemigos, niveles, elementos).

---

## Qué técnicas del master se aplican aquí

| Proyecto del master | Qué se reutiliza en el TFM |
|---|---|
| **P1 — EDA** | Estructura `src/cleaning.py` + `visualization.py`, función `configurar_estilo()`, secciones de exploración multidimensional |
| **P2 — SQL** | Modelo estrella conceptual para entender la jerarquía session→level→room; consultas de aggregación como referencia |
| **P3 — Estadística** | Regresión logística para predecir `isVictory`, selección de features, CLI scripts |
| **P4 — Fabric/ETL** | Pipeline medallion `raw → processed → exports`, notebooks secuenciales por etapa |
| **P5 — ML** | Feature engineering (estilo RFM adaptado a gaming), XGBoost + Optuna para modelo predictivo, validación cruzada, explicabilidad |
| **P6 — NLP** | `DistilBERT` para sentiment de `playerComment`, `TF-IDF + NMF` para topic modeling de comentarios, `wordcloud` |
| **P7 — IA Generativa** | Arquitectura por capas (`core/`, `services/`), Pydantic schemas, posible Streamlit dashboard final |

---

## Estructura de carpetas del proyecto

Sigue el estilo modular progresivo de los proyectos anteriores (educativo en notebooks, limpio en `src/`):

```
Proyecto-TFM-HackAndSlash/
├── TFM_ROADMAP.md
├── data/
│   ├── raw/                ← exportaciones brutas de Firestore (JSON)
│   ├── processed/          ← DataFrames aplanados y limpios (CSV/Parquet)
│   └── exports/            ← tablas finales para visualización e informe
├── notebooks/
│   ├── 01_extraction.ipynb         ← Fase 0: Firestore → raw
│   ├── 02_preprocessing.ipynb      ← Fase 1: aplanado, limpieza, features
│   ├── 03_eda.ipynb                ← Fase 2: exploración visual
│   ├── 04_statistics.ipynb         ← Fase 3: tests estadísticos
│   ├── 05_ml_predictor.ipynb       ← Fase 4a: modelo predictivo de victoria
│   ├── 06_nlp_comments.ipynb       ← Fase 4b: análisis de comentarios
│   ├── 07_balance_analysis.ipynb   ← Fase 5: conclusiones de balance
│   └── 08_report.ipynb             ← Fase 6: informe final
├── src/
│   ├── firestore_client.py   ← extracción de Firestore (firebase-admin)
│   ├── preprocessing.py      ← aplanado jerarquía + columnas derivadas
│   ├── cleaning.py           ← validación, nulos, outliers
│   ├── visualization.py      ← plots estandarizados (estilo P1)
│   ├── analysis.py           ← funciones de balance y métricas de juego
│   └── nlp_utils.py          ← sentiment + topic modeling (estilo P6)
├── reports/
│   ├── balance_report.html
│   └── figures/              ← imágenes exportadas para el informe
├── .env                      ← credenciales Firebase (NO subir a git)
└── requirements.txt
```

---

## Roadmap completo

### Fase 0 — Extracción de datos de Firestore ✅
*(Reutiliza patrón de `firestore_client.py` + pipeline medallion de P4)*

- [x] Configurar credenciales Firebase (service account JSON en `credentials/`)
- [x] Conectar via JWT (sin firebase-admin, solo `cryptography`)
- [x] Exportar colección `sessions` completa con subcolecciones `levels` y `rooms`
- [x] Guardar en `data/raw/sessions_raw.json`
- [x] Documentar versiones de juego disponibles (`gameVersion`) y plataformas (`platform`)
- [x] Crear script reutilizable (`firestore_client.py`) con función `export_all_sessions()`

### Fase 1 — Preprocesamiento y aplanado ✅
*(Patrón bronze→silver de P4 + limpieza estilo P1)*

- [x] Aplanar la jerarquía `session → level → room` en tres DataFrames (`preprocessing.py`)
- [x] Convertir `startUnixTime` / `endUnixTime` a datetime
- [x] Normalizar columnas dinámicas → formato largo con `rooms_to_long()`
- [x] Detectar y marcar sesiones incompletas / outliers (flag `is_suspicious`)
- [x] Crear columnas derivadas:
  - `kd_ratio`, `kills_per_min`, `time_per_level`, `completion_rate`
  - `cast_accuracy`, `kills_per_sec`, `totalRooms`, `totalCast`
- [ ] `spell_efficiency` = kills / cast por hechizo (se calcula en Fase 2 EDA)
- [ ] Separar por `gameVersion` cuando haya múltiples versiones

### Fase 2 — EDA (Análisis Exploratorio)
*(Estructura secciones numeradas de P5, modularidad de P1, visualizaciones de P6)*

- [ ] **Sesiones globales**
  - Distribución de `totalTimeSecs`, `totalDeaths`, `totalKills`
  - Tasa de victoria global (`isVictory`)
  - Popularidad de cada `playerElement` (bar chart)
  - Win rate por elemento (barh comparativo, estilo P6)
- [ ] **Niveles**
  - Dificultad por nivel: muertes, tiempo, intentos medios
  - Funnel de progresión: % de jugadores que llegan a cada nivel (estilo P5 con curva)
  - Curva de dificultad esperada vs real (line chart con anotaciones como P1)
  - Distribución de `attempt` por nivel → detectar grinding (nº de reintentos antes de pasar)
- [ ] **Salas**
  - Heatmap de muertes por sala/nivel (hotspots de dificultad)
  - `firstSpell` más común por sala → comportamiento táctico al entrar a cada sala
  - Ratio uso vs efectividad por hechizo (scatter plot)
- [ ] **Hechizos y elementos**
  - Ranking: hechizos más usados, más efectivos, más fallados
  - `noMana_{hechizo}` como indicador de demanda latente (bar chart)
  - Matriz de efectividad por elemento (heatmap)
- [ ] **Enemigos**
  - Enemigos con mayor `damageTaken` / `blocked` (barh)
  - Correlación tipo de enemigo vs muertes de sala
- [ ] **Comentarios de jugadores** (`playerComment`)
  - WordCloud por contexto (Victory vs GameOver) — librería `wordcloud` de P6
  - Distribución de longitud de comentarios
  - Análisis de `playerCommentContext` cruzado con `playerElement` y nivel completado

### Fase 3 — Análisis estadístico
*(Tests de P3, regresión logística, ANOVA — misma metodología)*

> **Criterio de suficiencia muestral:** Para que ANOVA y Chi-cuadrado sean válidos se necesitan
> al menos ~30 sesiones por elemento. Con el volumen actual (11 sesiones) los tests son orientativos.
> Esta limitación debe quedar documentada como limitación metodológica en el informe final.

- [ ] **Normalidad:** Shapiro-Wilk en métricas clave antes de elegir tests
- [ ] **Comparación entre elementos**
  - Si normal: ANOVA one-way → post-hoc Tukey
  - Si no normal: Kruskal-Wallis → post-hoc Dunn
  - Variable: win rate, totalDeaths, totalKills por `playerElement`
- [ ] **Correlaciones** (Pearson o Spearman según normalidad)
  - `totalDeaths` vs `isVictory`
  - `cast_accuracy` vs `isVictory` → ¿la precisión con hechizos predice la victoria?
  - `accuracy_hechizo` vs kills conseguidos
  - `timeSecs` por sala vs `deaths` en esa sala
- [ ] **Análisis de hechizos**
  - Chi-cuadrado: ¿el uso de ciertos hechizos está asociado a la victoria?
- [ ] **Curva de dificultad**
  - Test de tendencia (Mann-Kendall): ¿aumenta la dificultad nivel a nivel?
- [ ] **Análisis de aprendizaje**
  - Correlación `attempt` vs `deaths` por nivel → ¿los jugadores que repiten mejoran?

### Fase 4a — Modelo predictivo de victoria
*(Estilo completo de P5: feature engineering + 3 modelos + Optuna + explicabilidad)*

- [ ] **Definir target:** `isVictory` (clasificación binaria)
- [ ] **Features candidatos:**
  - `playerElement`, `platform`, `gameVersion`
  - `totalDeaths`, `totalKills`, `totalTimeSecs`, `levelsCompleted`
  - `kd_ratio`, `completion_rate`
  - Métricas agregadas de hechizos y salas
  - Uso relativo de hechizos por sesión (% de cast por tipo: Projectile, AOE, Blast, Beam)
- [ ] **Preprocesamiento ML:**
  - OneHotEncoding para `playerElement`, `platform`
  - StandardScaler para features numéricas
  - Manejo de clase desbalanceada si aplica
- [ ] **Modelos (igual que P5):**
  1. Logistic Regression (baseline)
  2. Random Forest
  3. XGBoost
- [ ] **Optimización:** Optuna + validación cruzada 5-fold
- [ ] **Métricas:** ROC-AUC, F1, matriz de confusión
- [ ] **Explicabilidad:** Feature Importance (XGBoost) + coeficientes (LR)
  → ¿Qué predice más la victoria? → directo a recomendaciones de balance
- [ ] **Análisis de error por elemento:** ¿el modelo falla más con algún elemento concreto?
  → sesgo del modelo = posible desequilibrio real en el juego

### Fase 4b — NLP en comentarios de jugadores
*(Pipeline completo de P6: limpieza → sentiment → topic modeling)*

- [ ] **Limpieza de texto:** eliminar URLs, caracteres especiales, stopwords en español/inglés
- [ ] **Análisis de sentimiento:** modelo multilingüe `nlptown/bert-base-multilingual-uncased-sentiment`
  - Usar multilingüe en lugar de DistilBERT inglés — los comentarios pueden estar en español
  - Comparar sentimiento Victory vs GameOver
- [ ] **Topic modeling:** `TF-IDF + NMF` (6-8 topics)
  - Identificar temas: diversión, dificultad, frustración, bugs, mecánicas, etc.
- [ ] **Cruce de datos:** sentimiento por `playerElement`, por nivel completado
  - ¿Qué elementos generan comentarios más positivos/negativos?
  - Cruce de `levelsCompleted` vs sentimiento → ¿los que llegan más lejos valoran mejor el juego?

### Fase 5 — Análisis de balance
*(Integra todo lo anterior en recomendaciones concretas)*

- [ ] **Balance de elementos:** win rate + sentiment + kills → ¿algún elemento dominante o inviable?
  - **Índice de balance compuesto por elemento** = `win_rate × sentiment_score × kills_efficiency`
    → un score único que resume rendimiento + experiencia subjetiva del jugador
- [ ] **Matriz elemento × nivel**
  - Heatmap: qué elemento sufre más muertes en qué nivel concreto
  - Detecta si hay niveles diseñados para un elemento pero hostiles para otros
- [ ] **Balance de hechizos:**
  - Ranking efectividad: `spell_efficiency` = kills/cast
  - Popularidad vs efectividad → hechizos sobreusados ineficientes o infravalorados
  - Alta tasa `noMana` → reducir coste de maná
- [ ] **Balance de enemigos:**
  - `blocked` alto → escudos excesivos (nerf resistencia)
  - `damage` muy alto → nerf daño
  - Raramente matados → demasiado difícil o mecánica poco intuitiva
- [ ] **Balance de niveles:**
  - Spike de dificultad: nivel con muertes >> media global
  - Salas que concentran la mayoría de muertes en su nivel
- [ ] **Efectos de estado (`status_{efecto}`):**
  - ¿Correlacionan con kills? → efectos de estado útiles vs irrelevantes
- [ ] **Tabla resumen de recomendaciones:**
  - Formato: Elemento afectado | Problema detectado | Dato que lo soporta | Propuesta concreta | Prioridad (Alta/Media/Baja)

### Fase 6 — Informe y presentación final
*(Estilo P6: informe generado desde notebooks + posible Streamlit de P7)*

- [ ] Generar visualizaciones finales exportadas a `reports/figures/`
- [ ] Notebook `08_report.ipynb` con resumen ejecutivo
- [ ] HTML/PDF con nbconvert o similar
- [ ] **Opcional (extensión P7):** Dashboard Streamlit interactivo `09_dashboard.py`
  - Reutiliza arquitectura por capas de P7 (`core/`, `services/`)
  - Filtros por `gameVersion`, `playerElement`, `platform`
  - Heatmap de dificultad de salas interactivo (Plotly)
  - Tabla de recomendaciones de balance con prioridades
  - Cierra el círculo de todos los proyectos del máster

---

## Prompt reutilizable para Claude / IA

Pega este prompt al inicio de cualquier sesión de análisis con IA:

```
Eres un analista de datos especializado en game analytics y balance de videojuegos.

Estoy desarrollando mi TFM de un Master en Data Science. El proyecto es un juego hack & slash donde
los jugadores eligen un elemento (Fuego, Agua, etc.) y recorren niveles compuestos por salas.
Los datos se recogen en Firestore con esta jerarquía:

- sessions/{sessionId}: sessionId, playerElement, startUnixTime, endUnixTime, isVictory,
  totalTimeSecs, totalDeaths, totalKills, levelsCompleted, playerComment, playerCommentContext,
  platform, gameVersion
- /levels/{levelId}: attempt, timeSecs, kills, deaths, damageTaken
- /levels/{levelId}/rooms/{roomId}: timeSecs, deaths, damageTaken, firstSpell,
  kills_{enemigo}, killedWith_{hechizo}, cast_{hechizo}, miss_{hechizo}, blocked_{enemigo},
  damage_{enemigo}, status_{efecto}, noMana_{hechizo}

OBJETIVO: usar los datos para tomar decisiones de balanceo del juego.

STACK TÉCNICO (Python):
- Datos: pandas, numpy
- Visualización: matplotlib, seaborn, wordcloud
- ML: scikit-learn, xgboost, optuna
- NLP: transformers (DistilBERT), sklearn TF-IDF + NMF
- Stats: scipy, statsmodels

Cuando te pida análisis, sigue este criterio:
1. Identifica el indicador clave relevante para la pregunta
2. Sugiere la visualización más apropiada
3. Indica qué test estadístico aplica si hay comparación entre grupos
4. Concluye con una recomendación concreta de balance (buff / nerf / rediseño)
5. Señala si el tamaño de muestra actual es suficiente o si necesito más datos
6. Sugiere el código Python específico para implementarlo

Dataset actual: [INDICA AQUÍ: nº sesiones, rango de fechas, versiones de juego, plataforma]
```

---

## Checklist de entrega final

- [ ] `01_extraction.ipynb` ejecutable de principio a fin (reproduce extracción completa)
- [ ] `02_preprocessing.ipynb` con aplanado documentado y columnas derivadas explicadas
- [ ] `03_eda.ipynb` con mínimo 15 visualizaciones comentadas con conclusiones
- [ ] `04_statistics.ipynb` con tests justificados y p-valores interpretados en contexto de juego
- [ ] `05_ml_predictor.ipynb` con 3 modelos comparados, Optuna y tabla de feature importance
- [ ] `06_nlp_comments.ipynb` con sentiment + topics + cruce con datos de juego
- [ ] `07_balance_analysis.ipynb` con tabla de recomendaciones priorizada
- [ ] `08_report.ipynb` con resumen ejecutivo
- [ ] Informe HTML/PDF exportado en `reports/`
- [ ] `requirements.txt` con dependencias fijadas
- [ ] `.env.example` con variables de entorno necesarias (sin valores reales)

---

## Dependencias recomendadas (`requirements.txt`)

```
# Extracción
firebase-admin

# Datos
pandas
numpy

# Visualización
matplotlib
seaborn
plotly
wordcloud

# Estadística
scipy
statsmodels

# Machine Learning
scikit-learn
xgboost
optuna

# NLP
transformers
torch

# Notebooks e informe
jupyter
nbconvert

# Utilidades
python-dotenv
openpyxl
```
