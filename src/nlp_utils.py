"""
nlp_utils.py — Análisis de sentimiento y topic modeling para comentarios de jugadores.

Funciones principales:
  - get_sentiment_scores   : sentimiento con BERT multilingüe (nlptown), normalizado 0-1
  - extract_topics_tfidf   : vectorización TF-IDF
  - extract_topics_nmf     : topic modeling con NMF
  - assign_dominant_topic  : tópico dominante por documento
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Stop words en español para TF-IDF (sin dependencia de nltk)
STOP_WORDS_ES = {
    'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con',
    'contra', 'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante',
    'e', 'el', 'ella', 'ellas', 'ellos', 'en', 'entre', 'era', 'es', 'esa',
    'esas', 'ese', 'eso', 'esos', 'esta', 'estaba', 'estado', 'estar', 'este',
    'esto', 'estos', 'fue', 'han', 'hasta', 'hay', 'he', 'la', 'las', 'le',
    'les', 'lo', 'los', 'me', 'mi', 'mis', 'muy', 'más', 'ni', 'no', 'nos',
    'o', 'para', 'pero', 'por', 'que', 'se', 'si', 'sin', 'sobre', 'su',
    'sus', 'también', 'te', 'tengo', 'ti', 'tiene', 'todo', 'todos', 'un',
    'una', 'unas', 'unos', 'ya', 'yo',
}


def get_sentiment_scores(
    texts: list[str],
    model_name: str = 'nlptown/bert-base-multilingual-uncased-sentiment',
    cache_dir: str = 'models/nlp_cache'
) -> list[dict]:
    """
    Análisis de sentimiento con BERT multilingüe.
    Devuelve lista de dicts: {'stars': int (1-5), 'score': float (0-1), 'raw_label': str}.
    Textos vacíos o None devuelven score neutral 0.5 (3 estrellas).
    """
    from transformers import pipeline

    # Cargar pipeline una sola vez
    sentiment_pipe = pipeline(
        'sentiment-analysis',
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=512,
        cache_dir=cache_dir,
    )

    results = []
    for text in texts:
        # Manejar textos vacíos
        if not text or not str(text).strip():
            results.append({'stars': 3, 'score': 0.5, 'raw_label': 'neutral'})
            continue

        try:
            out = sentiment_pipe(str(text)[:512])[0]
            # Etiqueta formato: "1 star", "2 stars", ... "5 stars"
            stars = int(out['label'].split()[0])
            score = (stars - 1) / 4  # normalizar a [0, 1]
            results.append({'stars': stars, 'score': round(score, 4), 'raw_label': out['label']})
        except Exception:
            results.append({'stars': 3, 'score': 0.5, 'raw_label': 'error'})

    return results


def extract_topics_tfidf(
    texts: list[str],
    max_features: int = 100,
    ngram_range: tuple = (1, 2)
) -> tuple[np.ndarray, list[str]]:
    """
    Vectoriza textos con TF-IDF.
    Devuelve (matrix, feature_names).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=list(STOP_WORDS_ES),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()
    return matrix, feature_names


def extract_topics_nmf(
    tfidf_matrix,
    feature_names: list[str],
    n_topics: int = 2,
    n_top_words: int = 5
) -> tuple[pd.DataFrame, object]:
    """
    Aplica NMF sobre la matriz TF-IDF.
    Devuelve (df_topics, nmf_model).
    df_topics tiene columnas: topic_id, top_words, weights.
    """
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=500)
    nmf.fit(tfidf_matrix)

    rows = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_words   = [feature_names[i] for i in top_indices]
        top_weights = [round(topic[i], 4) for i in top_indices]
        rows.append({
            'topic_id'  : topic_idx,
            'top_words' : ', '.join(top_words),
            'weights'   : top_weights,
        })

    return pd.DataFrame(rows), nmf


def assign_dominant_topic(tfidf_matrix, nmf_model) -> list[int]:
    """Devuelve el índice del tópico dominante para cada documento."""
    doc_topics = nmf_model.transform(tfidf_matrix)
    return doc_topics.argmax(axis=1).tolist()


def analyze_comments_with_gemini(df_comments: pd.DataFrame) -> str:
    """
    Análisis cualitativo profundo de comentarios de jugadores usando Gemini.
    df_comments debe tener columnas: playerComment, playerCommentContext,
    playerElement, isVictory, sentiment_score (opcional).
    Devuelve el análisis como string en markdown.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import gemini_client

    comentarios = []
    for _, row in df_comments.iterrows():
        comentario = str(row.get('playerComment', '') or '').strip()
        contexto   = str(row.get('playerCommentContext', '') or '').strip()
        if not comentario:
            continue
        elemento   = row.get('playerElement', 'Desconocido')
        victoria   = '✓ Victoria' if row.get('isVictory') else '✗ Derrota'
        sentiment  = row.get('sentiment_score')
        sent_str   = f' | Sentimiento: {sentiment:.2f}' if pd.notna(sentiment) else ''
        comentarios.append(
            f'- [{elemento} | {victoria}{sent_str}] "{comentario}"'
            + (f'\n  Contexto: {contexto}' if contexto else '')
        )

    if not comentarios:
        return 'No hay comentarios de jugadores disponibles para analizar.'

    bloque = '\n'.join(comentarios)
    n = len(comentarios)

    prompt = f"""Eres un experto en UX de videojuegos y análisis de feedback de jugadores.
Tienes {n} comentarios de jugadores de Arcane Descent, un Hack & Slash con 4 elementos jugables (Fire, Water, Earth, Wind).

COMENTARIOS:
{bloque}

Realiza un análisis cualitativo profundo con las siguientes secciones:

## 1. Temas recurrentes
Identifica los 3-5 temas más frecuentes. Para cada uno: nombre del tema, frecuencia/evidencia y citas relevantes.

## 2. Análisis por elemento
¿Hay diferencias notables en el feedback según el elemento usado? Señala patrones específicos por elemento.

## 3. Relación victoria/derrota con el feedback
¿Cambia el tono y los temas del feedback según si el jugador ganó o perdió?

## 4. Bugs y problemas técnicos reportados
Lista todos los bugs o comportamientos anómalos mencionados explícitamente.

## 5. Sugerencias de mejora identificadas
Lista todas las mejoras que los jugadores sugieren (explícita o implícitamente).

## 6. Conclusión y prioridades para el equipo de desarrollo
Resume los 3 puntos más urgentes a atender según el feedback.

Responde en español. Sé concreto y cita los comentarios cuando sea relevante."""

    return gemini_client.invoke(prompt)
