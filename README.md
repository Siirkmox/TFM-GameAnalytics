# TFM — Análisis de Balance de Juego: Hack & Slash

**Trabajo de Fin de Máster · Data Science**  
**Autor:** Samuel Alcaraz Rodriguez  
**Presentación:** junio 2026  

Pipeline completo de Data Science sobre datos reales de sesiones de un videojuego Hack & Slash
almacenados en Firebase Firestore. El objetivo es detectar desequilibrios entre los 4 elementos
jugables (Fire, Water, Earth, Wind) y generar recomendaciones accionables para el diseñador.

---

## Resultados principales

| Elemento | Balance Index | Win Rate |
|----------|--------------|---------|
| Fire     | 0.5818       | 36%     |
| Earth    | 0.5104       | 29%     |
| Water    | 0.2111       | 22%     |
| Wind     | 0.1289       | 14%     |

- **Win rate global:** 29.4% (n=34 sesiones limpias WebGL)
- **Hechizo dominante:** Beam (1.12 kills/cast, ×8 sobre Projectile)
- **42 recomendaciones** de balance identificadas, 22 de alta prioridad

---

## Estructura del proyecto

```
├── notebooks/
│   ├── 00_data_quality.ipynb     # Auditoría de calidad de datos Firestore
│   ├── 01_extraction.ipynb       # Extracción Firestore → raw (JWT, sin firebase-admin)
│   ├── 02_preprocessing.ipynb    # Aplanado jerarquía + feature engineering
│   ├── 03_eda.ipynb              # Análisis exploratorio visual
│   ├── 04_statistics.ipynb       # Tests estadísticos (Kruskal-Wallis, Chi², Dunn)
│   ├── 05_ml_predictor.ipynb     # ML: LR + RF + XGBoost + Optuna + K-Means
│   ├── 06_nlp_comments.ipynb     # NLP: BERT multilingüe + TF-IDF/NMF + Gemini
│   ├── 07_balance_analysis.ipynb # Balance Index + recomendaciones
│   └── 08_report.ipynb           # Informe ejecutivo + exportación HTML
├── src/
│   ├── firestore_client.py       # Cliente Firestore con JWT
│   ├── preprocessing.py          # Aplanado + normalización + features
│   ├── cleaning.py               # Validación, outliers, filtros
│   ├── visualization.py          # Plots estandarizados (paleta por elemento)
│   ├── analysis.py               # Balance Index + recomendaciones IA (Gemini)
│   ├── nlp_utils.py              # Sentiment BERT + topic modeling + análisis Gemini
│   ├── game_agent.py             # Agente LangGraph con 9 herramientas de consulta
│   ├── game_tools.py             # Tools LangChain para el agente
│   └── gemini_client.py          # Cliente Gemini con retry y fallback de modelos
├── dashboard/                    # App Streamlit (deploy en Streamlit Community Cloud)
│   ├── 09_dashboard.py           # Dashboard (8 pestañas + agente IA)
│   ├── requirements.txt          # Dependencias mínimas para el deploy (~150 MB)
│   └── .streamlit/config.toml    # Tema oscuro corporativo
├── data/
│   ├── raw/                      # JSON exportados de Firestore (no en git)
│   ├── processed/                # Parquets limpios (parquets sí en git, CSVs no)
│   └── exports/                  # CSVs de resultados finales
├── models/                       # Modelos .pkl entrenados
├── reports/
│   ├── balance_report.html       # Informe ejecutivo HTML
│   └── figures/                  # Todas las figuras generadas
├── requirements-dev.txt          # Dependencias completas para notebooks (torch, transformers, optuna...)
└── README.md
```

---

## Instalación

```bash
git clone https://github.com/Siirkmox/TFM-GameAnalytics.git
cd TFM-GameAnalytics

# Para ejecutar los notebooks (pipeline completo)
pip install -r requirements-dev.txt

# Para ejecutar solo el dashboard (mucho más rápido)
pip install -r dashboard/requirements.txt
```

Copia `.env.example` a `.env` y rellena las credenciales:

```bash
cp .env.example .env
```

Variables necesarias en `.env`:

```
GOOGLE_API_KEY=tu_clave_gemini
```

Para la extracción de Firestore también necesitas `credentials/firebase-service-account.json`
(no incluido en el repositorio por seguridad).

---

## Ejecución

### Dashboard interactivo

```bash
streamlit run dashboard/09_dashboard.py
```

El dashboard carga directamente desde `data/exports/` y `models/` — no requiere
reejecutar los notebooks si ya están en el repositorio.

### Notebooks (orden de ejecución)

```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

Cada notebook exporta sus resultados a `data/exports/` y `reports/figures/`
para que el siguiente los consuma.

### Informe HTML

El notebook `08_report.ipynb` genera `reports/balance_report.html` automáticamente
al ejecutarse. También está disponible en el repositorio como artefacto precompilado.

---

## Pipeline de análisis

| Fase | Técnica | Output |
|------|---------|--------|
| Extracción | JWT Service Account, jerarquía sessions→levels→rooms | `data/raw/` |
| Preprocesamiento | Aplanado, normalización killedWith, features derivadas | `sessions.parquet` |
| EDA | Distribuciones, correlaciones, funnel de progresión | Figuras fig2-fig4 |
| Estadística | Kruskal-Wallis, Chi², Dunn post-hoc, Mann-Kendall | `statistical_results.csv` |
| ML | LR + RF + XGBoost + Optuna (50 trials) + K-Means (LOO) | Modelos `.pkl` |
| NLP | BERT multilingüe (nlptown) + TF-IDF/NMF + Gemini | `nlp_results.csv` |
| Balance | Balance Index ponderado (win_rate×0.5 + kills_eff×0.3 + sentiment×0.2) | `balance_index.csv` |
| IA Generativa | Agente LangGraph + Gemini + 9 herramientas de consulta | Dashboard interactivo |

---

## Tecnologías

Python · Pandas · scikit-learn · XGBoost · Optuna · HuggingFace Transformers  
LangChain · LangGraph · Gemini API · Streamlit · Plotly · Firebase Firestore
