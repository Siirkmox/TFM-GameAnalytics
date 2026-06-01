# TFM — Análisis de Balance de Juego: Hack & Slash con Firestore

Proyecto de Data Science desarrollado en Python para el TFM del Master.
Analiza datos de sesiones de un juego Hack & Slash almacenados en Firestore
para generar recomendaciones de balanceo del juego.

## Git — Commits y subidas a GitHub

- **No añadir Co-Authored-By de Claude** en los commits de este proyecto.
  Los commits deben ir firmados únicamente con el autor humano (`Siirkmox`).
- Rama principal: `master`
- Repositorio: https://github.com/Siirkmox/TFM-GameAnalytics
- Nunca subir a git: credenciales, `.env`, `data/raw/`, `data/processed/`

## Estructura del proyecto

```
Proyecto-TFM-HackAndSlash/
├── data/
│   ├── raw/                ← exportaciones brutas de Firestore (JSON) — no subir
│   ├── processed/          ← DataFrames aplanados y limpios (CSV/Parquet) — no subir
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
│   ├── firestore_client.py   ← extracción autenticada (JWT, sin firebase-admin)
│   ├── preprocessing.py      ← aplanado jerarquía + columnas derivadas
│   ├── cleaning.py           ← validación, nulos, outliers
│   ├── visualization.py      ← plots estandarizados
│   ├── analysis.py           ← métricas de balance + recomendaciones IA
│   ├── nlp_utils.py          ← sentiment + topic modeling + análisis Gemini
│   ├── game_agent.py         ← agente LangGraph con herramientas de consulta
│   ├── game_tools.py         ← tools LangChain para el agente (9 herramientas)
│   └── gemini_client.py      ← cliente Gemini con retry y fallback de modelos
├── reports/
│   ├── balance_report.html
│   └── figures/
├── credentials/              ← service account Firebase — NO subir a git
├── .env                      ← credenciales — NO subir a git
└── requirements.txt
```

## Roadmap de fases

| Fase | Notebook | Estado |
|------|----------|--------|
| 0a — Calidad de datos | `00_data_quality.ipynb` | ✅ Completo |
| 0b — Extracción Firestore | `01_extraction.ipynb` | ✅ Completo |
| 1 — Preprocesamiento y limpieza | `02_preprocessing.ipynb` | ✅ Completo |
| 2 — EDA (Análisis Exploratorio) | `03_eda.ipynb` | ✅ Completo |
| 3 — Análisis estadístico | `04_statistics.ipynb` | ✅ Completo |
| 4a — ML predictor de victoria | `05_ml_predictor.ipynb` | ✅ Completo |
| 4b — NLP comentarios | `06_nlp_comments.ipynb` | ✅ Completo |
| 5 — Análisis de balance | `07_balance_analysis.ipynb` | ✅ Completo |
| 6 — Informe y presentación | `08_report.ipynb` + `09_dashboard.py` | ✅ Completo |

Specs detallados (Design + Requirements + Tasks + Jira) en `.claude/specs/`.
**Fecha límite:** presentación 11-12 de junio de 2026.

## Datos en Firestore

Jerarquía:
```
sessions/{sessionId}
  └── levels/{levelId}
        └── rooms/{roomId}
```

Autenticación: Service Account JWT (sin firebase-admin). Ver `src/firestore_client.py`.
Credenciales: `credentials/firebase-service-account.json` (no subir a git).

Estado actual de datos (actualizar con cada extracción):
- Sesiones: 52 | Niveles: 104 | Salas: 519
- Elementos (WebGL limpias, n=34): Fire×11, Water×9, Wind×7, Earth×7
- Victorias: 10/34 WebGL limpias (29.4%)
- Plataformas: WebGL×47, Editor×5
- Versión de juego: 0.1

## Jira — Gestión de tareas

- Proyecto: **TFM** en Atlassian Cloud (cloudId: `d96716e8-a8f1-460e-86d2-131f2e41b5d0`)
- Épicas: TFM-1 (Datos), TFM-2 (Preparación), TFM-3 (Modelado), TFM-4 (Validación), TFM-5 (Exposición)
- **IDs de transición:** `21` = En curso, `31` = Listo (Finalizada)
- Credenciales: `.claude/jira_credentials.json` (no subir a git) — usar `urllib.request` + Basic auth para escritura (el MCP es solo lectura en Windows)

## Convenciones de código

- Comentarios en todo el código nuevo (clases, métodos, bloques lógicos)
- Responder siempre en español
- Notebooks: secciones numeradas con markdown, conclusiones al final de cada bloque
- `src/`: módulos Python puros, sin lógica de notebook
- Extraer funciones reutilizables a `src/` cuando aparecen en más de un notebook
