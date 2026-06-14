# Arcane Descent — Game Balance Analytics

> Análisis end-to-end de balance de un videojuego Hack & Slash usando telemetría real generada por jugadores. **No es un dataset descargado: el juego se construyó desde cero en Unity para tener datos propios.**

[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://tfm-gameanalytics.streamlit.app)
[![Juego](https://img.shields.io/badge/Juega-itch.io-FA5C5C?logo=itch.io)](https://sirkmox.itch.io/arcanedescent)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph%20%2B%20Gemini-1A8FFF)](https://langchain-ai.github.io/langgraph/)

**Autor:** Samuel Alcaraz Rodriguez · [LinkedIn](https://www.linkedin.com/in/samuelalcarazrodriguez/)
**Trabajo de Fin de Máster — Data Science** · 2026

---

## 🎯 Enlaces rápidos

| | |
|---|---|
| 📊 **Dashboard interactivo** | [tfm-gameanalytics.streamlit.app](https://tfm-gameanalytics.streamlit.app) |
| 🎮 **Juega al juego** | [sirkmox.itch.io/arcanedescent](https://sirkmox.itch.io/arcanedescent) |
| 📑 **Informe HTML** | [reports/balance_report.html](reports/balance_report.html) |

---

## 💡 Lo que descubrió el análisis

Sobre **n=39 sesiones limpias** de jugadores reales:

- **⚖️ Desbalance entre magos:** Fire (Puntuación Global 0.58) está **4× mejor balanceado** que Wind (0.14)
- **🎯 Hechizo desperdiciado:** Beam es **×8.6 más eficiente** que Projectile (1.17 vs 0.14 kills/cast) pero se usa 14 veces menos
- **🚪 Cuello de botella:** Solo el **57% del juego se completa** de media — la mitad del contenido no llega a verse
- **🤖 ML predictivo:** XGBoost predice victoria/derrota con **89.7% accuracy** — la habilidad pesa más que el elemento
- **🔧 Ciclo cerrado:** **12 bugs de telemetría** detectados durante el análisis y corregidos en Unity (27% de las 44 recomendaciones implementadas)

### Puntuación Global por elemento

| Elemento | Puntuación Global | Win Rate | n sesiones |
|----------|:-----------------:|:--------:|:----------:|
| 🔥 **Fire**  | **0.58** | 36% | 11 |
| 🪨 Earth | 0.48 | 38% | 8 |
| 💧 Water | 0.26 | 30% | 10 |
| 💨 **Wind**  | **0.14** | 20% | 10 |

> **Fórmula:** Puntuación Global = 0.5 × Win Rate + 0.3 × Eficiencia de combate + 0.2 × Sentimiento del jugador

---

## 🔬 Pipeline de análisis

```
Firestore → Limpieza → Estadística → ML → NLP → Puntuación Global → Dashboard + Agente IA
   JWT      n=58→39   Kruskal-Chi² XGB  BERT      0.5/0.3/0.2         LangGraph + 9 tools
```

| Fase | Técnica | Output |
|------|---------|--------|
| **Extracción** | Cliente JWT propio sobre Firestore REST API (sin `firebase-admin`) | `data/raw/` |
| **Preprocesamiento** | Aplanado de jerarquía `sessions → levels → rooms` + feature engineering | `sessions.parquet` |
| **Limpieza** | Detección automática de sesiones anómalas (Editor, <2min, >1h, sin actividad) | n=58 → n=39 limpias |
| **EDA + Estadística** | Kruskal-Wallis, Chi², Dunn post-hoc, Mann-Kendall | `statistical_results.csv` |
| **ML** | Logistic Regression + Random Forest + **XGBoost** con tuning Optuna (50 trials) + K-Means | Modelos `.pkl` |
| **NLP** | Análisis de sentimiento con BERT multilingüe (`nlptown`) + topic modeling TF-IDF/NMF | `nlp_results.csv` |
| **Puntuación Global** | Métrica compuesta ponderada por elemento | `balance_index.csv` |
| **Agente IA** | LangGraph + Gemini 2.5 con **9 herramientas** que consultan los CSVs en tiempo real | Dashboard interactivo |

---

## 🛠️ Stack tecnológico

**Lenguajes:** Python 3.11
**Datos:** Pandas · NumPy · PyArrow
**ML/Estadística:** scikit-learn · XGBoost · Optuna · scipy · statsmodels · pingouin
**NLP:** HuggingFace Transformers · PyTorch
**IA Generativa:** LangChain · LangGraph · Gemini API
**Visualización:** Plotly · Matplotlib · Seaborn
**Dashboard:** Streamlit
**Backend de datos:** Firebase Firestore (con cliente JWT propio)

---

## 🚀 Ejecución local

### Solo dashboard (recomendado — arranque rápido)

```bash
git clone https://github.com/Siirkmox/TFM-GameAnalytics.git
cd TFM-GameAnalytics
pip install -r dashboard/requirements.txt
streamlit run dashboard/09_dashboard.py
```

El dashboard carga directamente desde `data/exports/` y `models/` versionados en el repo.
No necesita reejecutar los notebooks.

### Pipeline completo de notebooks

```bash
pip install -r requirements-dev.txt
```

Orden de ejecución: `00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`

Cada notebook exporta sus resultados a `data/exports/` y `reports/figures/` para que el siguiente los consuma.

### Variables de entorno

Copia `.env.example` a `.env` y rellena:

```bash
GOOGLE_API_KEY=tu_clave_gemini_aqui     # Para el agente IA y NLP
ADMIN_PASSWORD=password_dashboard       # Para acciones destructivas del dashboard
```

Para la extracción de Firestore necesitas también `credentials/firebase-service-account.json` (no incluido por seguridad).

---

## 📁 Estructura del proyecto

```
├── dashboard/                ← App Streamlit (deploy en Streamlit Cloud)
│   ├── 09_dashboard.py       # 8 pestañas + agente IA conversacional
│   ├── requirements.txt      # Deps mínimas (~150 MB)
│   └── .streamlit/
├── notebooks/                ← Pipeline reproducible 00→08
├── src/                      ← Código modular reutilizable
│   ├── firestore_client.py   # Cliente JWT propio (sin firebase-admin)
│   ├── preprocessing.py      # Aplanado + feature engineering
│   ├── cleaning.py           # Detección de anomalías
│   ├── analysis.py           # Puntuación Global + recomendaciones IA
│   ├── nlp_utils.py          # BERT + topic modeling
│   ├── game_agent.py         # Agente LangGraph
│   ├── game_tools.py         # 9 herramientas que consultan los CSVs
│   └── gemini_client.py      # Cliente Gemini con retry y fallback
├── data/
│   ├── raw/                  # JSON Firestore (no en git)
│   ├── processed/            # Parquets versionados (sessions/levels/rooms)
│   └── exports/              # CSVs de resultados finales
├── models/                   # Modelos .pkl (LR, RF, XGBoost, K-Means)
├── reports/
│   ├── balance_report.html   # Informe ejecutivo
│   └── figures/              # Todas las figuras
├── requirements-dev.txt      # Deps completas (torch, transformers, optuna...)
└── README.md
```

---

## 📌 Limitaciones

- **n=39 sesiones** sigue siendo muestra orientativa para un análisis estadístico robusto (ideal: n≥30 por elemento, total ≥120)
- 3 de 4 elementos ya cumplen el umbral mínimo de n≥10 (Fire, Water, Wind). Solo Earth queda en n=8
- Los resultados son **descriptivos y orientativos**, no inferencias definitivas
- El propio dashboard etiqueta cada métrica con un indicador de fiabilidad (✅ / ⚠️) según el tamaño de muestra disponible

---

## 📬 Contacto

**Samuel Alcaraz Rodriguez**

- LinkedIn: [linkedin.com/in/samuelalcarazrodriguez](https://www.linkedin.com/in/samuelalcarazrodriguez/)
- GitHub: [@Siirkmox](https://github.com/Siirkmox)
- Email: samuelalcarazrodriguez@gmail.com

---

<sub>🤖 Si tienes feedback o ideas para mejorarlo, ¡no dudes en abrir un issue!</sub>
