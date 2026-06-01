# Spec: 06_nlp_comments.ipynb — Análisis NLP de Comentarios

## Design

**Objetivo:**
Extraer señales cualitativas de los comentarios de los jugadores (`playerComment` y
`playerCommentContext`) mediante análisis de sentimiento y modelado de tópicos.
El score de sentimiento normalizado alimenta el Balance Index de la Fase 5.

**Entradas:**
- `data/processed/sessions.parquet` — columnas `playerComment`, `playerCommentContext`, `playerElement`, `isVictory`, `sessionId`
- `src/nlp_utils.py` — a crear en este notebook

**Salidas:**
- `data/exports/nlp_results.csv` — sessionId, sentiment_score (0-1), sentiment_stars, dominant_topic, comment_length
- `reports/figures/fig6_1_wordcloud.png` — nube de palabras de todos los comentarios
- `reports/figures/fig6_2_sentimiento_elemento.png` — barras de sentimiento medio por elemento
- `reports/figures/fig6_3_sentimiento_victoria.png` — distribución de sentimiento según isVictory
- `reports/figures/fig6_4_topics.png` — barras de pesos de tópicos NMF

**Conexión con proyectos anteriores:**
- **P6 (NLP con HuggingFace):** mismo patrón pipeline Transformers + TF-IDF + NMF
- Diferencia clave: modelo multilingüe `nlptown/bert-base-multilingual-uncased-sentiment` en lugar de DistilBERT en inglés

**Visualizaciones previstas:**
1. WordCloud de todos los comentarios (unión de `playerComment` + `playerCommentContext`)
2. Barplot: sentimiento medio (0-1) por `playerElement` con error bars
3. Boxplot o violinplot: distribución de `sentiment_score` según `isVictory` (True/False)
4. Barplot horizontal: pesos de las top-5 palabras por tópico NMF

---

## Requirements

**Funciones a crear en `src/nlp_utils.py`:**

```python
def get_sentiment_scores(texts: list[str], model_name: str = 'nlptown/bert-base-multilingual-uncased-sentiment') -> list[dict]:
    """
    Ejecuta análisis de sentimiento con HuggingFace pipeline.
    Devuelve lista de {'stars': int (1-5), 'score': float (0-1), 'raw_label': str}.
    Normaliza estrellas a [0,1]: (stars - 1) / 4.
    Maneja textos vacíos devolviendo {'stars': 3, 'score': 0.5, 'raw_label': 'neutral'}.
    """

def extract_topics_tfidf(texts: list[str], max_features: int = 100, ngram_range=(1,2)) -> tuple[np.ndarray, list[str]]:
    """
    Vectoriza textos con TF-IDF.
    Devuelve (matrix, feature_names).
    """

def extract_topics_nmf(tfidf_matrix, feature_names: list[str], n_topics: int = 3, n_top_words: int = 5) -> pd.DataFrame:
    """
    Aplica NMF sobre la matriz TF-IDF.
    Devuelve DataFrame con columnas: topic_id, top_words, weights.
    """

def assign_dominant_topic(tfidf_matrix, nmf_model) -> list[int]:
    """Devuelve el índice del tópico dominante para cada documento."""
```

**Modelo de sentimiento:**
- `nlptown/bert-base-multilingual-uncased-sentiment`
- Salida: etiquetas `"1 star"` a `"5 stars"` → extraer número → normalizar a [0,1]
- Usar `transformers.pipeline('sentiment-analysis', model=..., tokenizer=...)`
- Truncar textos a 512 tokens (límite BERT)

**Topic modeling:**
- TF-IDF: `max_features=100`, `ngram_range=(1,2)`, `stop_words` español manual (no nltk)
- NMF: `n_components=3` (con ~10 textos, más de 3 tópicos es ruido)
- Interpretar tópicos manualmente y nombrarlos (ej. "dificultad", "hechizos", "nivel")

**Librerías requeridas:**
- `transformers` — pipeline de sentimiento
- `torch` — backend para Transformers
- `sklearn.feature_extraction.text` — TfidfVectorizer
- `sklearn.decomposition` — NMF
- `wordcloud` — nube de palabras
- `matplotlib`, `seaborn`

**Outputs mínimos:**
- `nlp_results.csv` con `sentiment_score` (0-1) para cada sesión con comentario
- Al menos 3 figuras exportadas
- Interpretación cualitativa de los tópicos encontrados
- `sentiment_score` medio por elemento para uso en Balance Index (Fase 5)

**Restricciones:**
- Solo ~10 sesiones tienen comentario (`hasComment == True`) — análisis cualitativo, no cuantitativo
- Documentar: "Con n≈10 comentarios los resultados son ilustrativos. El modelo identifica tendencias pero no patrones estadísticamente sólidos."
- Para sesiones sin comentario: `sentiment_score = NaN` (no imputar a 0.5 en este notebook — la Fase 5 decidirá)
- Descargar modelo en primera ejecución: puede tardar ~2 min (usar `cache_dir='models/nlp_cache'`)

---

## Tasks

### Bloque 1 — Carga y preparación de comentarios
- [x] 5 comentarios disponibles (de 29 sesiones): Fire×2, Wind×2, Water×1
- [x] Texto combinado playerComment + playerCommentContext
- [x] **Jira: TFM-27**

### Bloque 2 — Crear `src/nlp_utils.py`
- [x] `get_sentiment_scores()` con BERT multilingüe, normalización [0,1]
- [x] `extract_topics_tfidf()` con stop words español manual
- [x] `extract_topics_nmf()` y `assign_dominant_topic()`
- [x] **Jira: TFM-28**

### Bloque 3 — Análisis de sentimiento
- [x] Sentimiento calculado con `nlptown/bert-base-multilingual-uncased-sentiment`
- [x] WordCloud generado (`fig6_1_wordcloud.png`)
- [x] Scores: Fire(victoria)=0.50, Fire(derrota)=0.75, Water=0.75, Wind×2=0.00/0.50
- [x] **Jira: TFM-28**

### Bloque 4 — Topic modeling TF-IDF + NMF
- [x] NMF con n_topics=2 (n=5 docs, 3 tópicos sería ruido)
- [x] Tópico 0: "Bugs / Problemas técnicos" | Tópico 1: "Experiencia / Dificultad"
- [x] Visualización top palabras (`fig6_4_topics.png`)
- [x] **Jira: TFM-41**

### Bloque 5 — Cross-analysis sentimiento × elemento × victoria
- [x] Barplot sentimiento por elemento + victoria (`fig6_2_sentimiento_elemento.png`)
- [x] Wind=sentimiento más negativo (bug arqueros), Fire/Water=positivo-neutral
- [x] Earth y Water sin comentarios → imputar 0.5 en Fase 5
- [x] **Jira: TFM-42**

### Bloque 6 — Exportar resultados
- [x] `nlp_results.csv` exportado — 5 sesiones
- [x] Tabla sentiment_score medio por elemento lista para Balance Index
- [x] **Jira: TFM-42**
