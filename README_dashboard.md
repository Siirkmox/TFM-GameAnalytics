# Dashboard TFM — Hack & Slash Balance Analysis

Dashboard interactivo Streamlit para explorar los resultados del análisis de balance del TFM.

## Requisitos

```bash
pip install -r requirements.txt
```

Dependencias principales: `streamlit`, `plotly`, `pandas`, `numpy`, `scikit-learn`

## Ejecución

Desde la raíz del proyecto:

```bash
streamlit run 09_dashboard.py
```

El dashboard abre automáticamente en `http://localhost:8501`

## Estructura del dashboard

| Tab | Contenido |
|-----|-----------|
| 📊 Resumen | KPIs globales, Balance Index por elemento, hallazgos EDA |
| 🔥 Elementos | Win rate, K/D ratio, kills/min, cast accuracy y radar de KPIs por elemento |
| ✨ Hechizos | Eficiencia de hechizos, matchup matrix, uso relativo |
| 🏰 Salas | Heatmap dificultad elemento×nivel, top salas abandono, funnel Level1, churn rate |
| 🤖 ML | Curvas ROC, feature importance, clusters K-Means, tabla de predicciones |
| 📋 Recomendaciones | Tabla filtrable por prioridad y elemento |

## Filtros disponibles

- **Sidebar:** selección de elementos (Fire / Water / Earth / Wind)
- **Tab Recomendaciones:** filtro por prioridad (Alta / Media / Baja) y por elemento

## Datos que consume

El dashboard carga los archivos de `data/exports/` (solo lectura, sin recalcular):

```
data/exports/
├── balance_index.csv
├── balance_recommendations.csv
├── statistical_results.csv
├── ml_predictions.csv
├── player_clusters.csv
├── nlp_results.csv
├── eda_hallazgos.csv
└── churn_analysis.csv
```

Las figuras estáticas se cargan desde `reports/figures/`.

## Informe HTML

El informe ejecutivo está disponible en:

```
reports/balance_report.html
```

Se puede abrir directamente en el navegador (todas las imágenes están embebidas).

## Regenerar informe HTML

```bash
cd notebooks
jupyter nbconvert --to html --no-input --embed-images 08_report.ipynb --output ../reports/balance_report.html
```
