# Spec: 05_ml_predictor.ipynb — Modelo Predictivo de Victoria + Clustering

## Design

**Objetivo:**
Construir un pipeline ML para predecir `isVictory` (clasificación binaria) a partir de
las métricas de rendimiento del jugador. Complementar con K-Means clustering para
identificar perfiles de jugador según estilo de juego.

**Entradas:**
- `data/processed/sessions.parquet` — df_sessions con features derivadas
- `data/processed/rooms.parquet` — df_rooms para `compute_spell_usage_ratios()`
- `src/preprocessing.py` — `compute_spell_usage_ratios()`
- `src/cleaning.py` — `filter_clean()`
- `src/visualization.py` — plots base (creado en Fase 3)

**Salidas:**
- `data/exports/ml_predictions.csv` — predicciones y probabilidades por sesión
- `data/exports/player_clusters.csv` — cluster asignado por sesión + centroide
- `models/logistic_regression.pkl` — modelo LR serializado
- `models/random_forest.pkl` — modelo RF serializado
- `models/xgboost_model.pkl` — modelo XGB serializado
- `reports/figures/fig5_1_feature_importance.png`
- `reports/figures/fig5_2_roc_curves.png`
- `reports/figures/fig5_3_confusion_matrices.png`
- `reports/figures/fig5_4_silhouette.png`
- `reports/figures/fig5_5_cluster_scatter.png`

**Conexión con proyectos anteriores:**
- **P5 (ML pipeline):** mismo stack LogReg + RF + XGBoost + Optuna, mismos patrones de evaluación
- **P7:** feature importance como entrada a las recomendaciones de balance

**Visualizaciones previstas:**
1. Importancia de features (barplot horizontal, top 10) para RF y XGB
2. Curvas ROC de los 3 modelos superpuestas
3. Matrices de confusión (3 subplots)
4. Silhouette score vs k (línea con punto óptimo marcado)
5. Scatter PCA 2D de clusters coloreados por `playerElement`

---

## Requirements

**Feature matrix:**
```python
FEATURES = [
    'kd_ratio', 'kills_per_min', 'time_per_level', 'cast_accuracy',
    'totalRooms', 'levelsCompleted', 'totalCast',
    'ratio_cast_Beam', 'ratio_cast_Blast', 'ratio_cast_Projectile', 'ratio_cast_AOE'
    # ratio_cast_* generadas por compute_spell_usage_ratios()
]
TARGET = 'isVictory'
```

**Modelos:**
- Logistic Regression (`sklearn.linear_model.LogisticRegression`, `C=1.0`, `max_iter=1000`)
- Random Forest (`sklearn.ensemble.RandomForestClassifier`, `n_estimators=100`)
- XGBoost (`xgboost.XGBClassifier`, `eval_metric='logloss'`, `use_label_encoder=False`)

**Tuning con Optuna:**
- 50 trials (con solo 11 sesiones: orientativo)
- Espacio de búsqueda: `n_estimators` [50,300], `max_depth` [2,8], `learning_rate` [0.01,0.3]
- Cross-validation: `StratifiedKFold(n_splits=3)` (n_splits=3 máximo con n=11)
- Métrica objetivo: ROC-AUC

**Clustering:**
- K-Means con k en rango [2, 6]
- Silhouette score para cada k → elegir k óptimo
- Features de clustering: mismas que el modelo (escaladas con StandardScaler)
- PCA 2D para visualización

**Métricas de evaluación:**
- ROC-AUC (principal)
- F1-score (macro)
- Confusion matrix
- Classification report

**Librerías requeridas:**
- `sklearn` — LogReg, RF, KMeans, Silhouette, PCA, StandardScaler, metrics
- `xgboost` — XGBClassifier
- `optuna` — hyperparameter tuning
- `joblib` — serialización de modelos
- `matplotlib`, `seaborn`

**Outputs mínimos:**
- Tabla comparativa de los 3 modelos con AUC, F1, accuracy
- Mejor modelo identificado y justificado
- K óptimo para clustering identificado
- Todos los modelos serializados en `models/`

**Restricciones:**
- Con n=11: no hay split train/test real, usar `cross_val_score` o LOO (Leave-One-Out)
- Documentar explícitamente la limitación de muestra en cada celda de resultados
- `LeaveOneOut` de sklearn como estrategia de validación principal
- No usar `train_test_split` con n<30 — resultados no representativos
- Crear directorio `models/` si no existe

---

## Tasks

### Bloque 1 — Setup y feature matrix
- [x] Importar librerías (`sklearn`, `xgboost`, `optuna`, `joblib`)
- [x] Cargar datos y aplicar `filter_clean()`
- [x] Generar `ratio_cast_*` con `compute_spell_usage_ratios()`
- [x] Merge features en X, definir y como `isVictory`
- [x] Escalar con `StandardScaler` y guardar scaler
- [x] Advertencia explícita: n=24 sesiones, 5 victorias/19 derrotas
- [x] **Jira: TFM-14**

### Bloque 2 — Logistic Regression baseline
- [x] Entrenar LogReg con `LeaveOneOut` + `class_weight='balanced'`
- [x] Calcular AUC, F1 macro, accuracy
- [x] Coeficientes visualizados como barplot (`fig5_0_lr_coeficientes.png`)
- [x] **Jira: TFM-39**

### Bloque 3 — Random Forest y XGBoost
- [x] RF y XGB entrenados con LOO + `scale_pos_weight` para desequilibrio
- [x] Feature importance RF y XGB (`fig5_1_feature_importance.png`)
- [x] Curvas ROC superpuestas (`fig5_2_roc_curves.png`)
- [x] Matrices de confusión (`fig5_3_confusion_matrices.png`)
- [x] **Jira: TFM-14**

### Bloque 4 — Optuna hyperparameter tuning
- [x] 50 trials con StratifiedKFold(n_splits=3)
- [x] XGBoost reentrenado con mejores parámetros
- [x] Comparación AUC base vs tuned
- [x] **Jira: TFM-40**

### Bloque 5 — Comparación y mejor modelo
- [x] Tabla comparativa 4 modelos: LR, RF, XGB, XGB tuned
- [x] Mejor modelo seleccionado por AUC
- [x] 3 modelos + scaler serializados en `models/`
- [x] `ml_predictions.csv` exportado (20 sesiones)
- [x] **Jira: TFM-15**

### Bloque 6 — K-Means clustering
- [x] Silhouette para k=2..6 (`fig5_4_silhouette.png`)
- [x] K-Means con k óptimo — 5 clusters identificados
- [x] PCA 2D con victorias marcadas con estrella (`fig5_5_cluster_scatter.png`)
- [x] Perfiles: clusters separan por intensidad de combate más que por elemento
- [x] **Jira: TFM-15**

### Bloque 7 — Exportar resultados
- [x] `player_clusters.csv` exportado (20 sesiones, 5 clusters)
- [x] Centroides en espacio original impresos
- [x] Conclusiones: `levelsCompleted` y `kd_ratio` top predictores de victoria
- [x] **Jira: TFM-40**

## Tasks pendientes (del roadmap, no implementadas)

- [x] **Análisis de error del modelo por elemento** — `fig5_6_error_por_elemento.png`; tasa de error por elemento calculada — **TFM-15**
