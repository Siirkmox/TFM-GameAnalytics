"""
analysis.py — Métricas de balance y recomendaciones para el TFM.

Funciones principales:
  - compute_killedwith_efficiency : eficiencia real de hechizos via killedWith_*
  - compute_balance_index         : índice compuesto por elemento (win_rate × sentiment × kills_eff)
  - element_level_matrix          : matriz elemento × nivel con métrica configurable
  - top_abandoned_rooms           : salas con mayor score de abandono
  - generate_recommendations      : tabla de recomendaciones priorizadas
"""

import pandas as pd
import numpy as np

# Mapeo de hechizo → columnas killedWith_* que le corresponden
KILLEDWITH_SPELL_MAP = {
    'Beam':       ['killedWith_Beam'],
    'Projectile': ['killedWith_Projectile'],
    'Blast':      ['killedWith_Blast'],
    'AOE':        ['killedWith_AOE'],
}


def compute_killedwith_efficiency(df_rooms: pd.DataFrame,
                                   df_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la eficiencia real de cada hechizo por elemento usando killedWith_*.
    Más fiable que kills_per_min porque usa kills atribuidos directamente al hechizo.
    Devuelve DataFrame con columnas: element, spell, kills, casts, efficiency (kills/cast).
    """
    rooms_elem = df_rooms.merge(
        df_sessions[['sessionId', 'playerElement']], on='sessionId', how='left'
    )

    rows = []
    for spell, kw_cols in KILLEDWITH_SPELL_MAP.items():
        valid_cols = [c for c in kw_cols if c in rooms_elem.columns]
        if not valid_cols:
            continue
        kills_by_elem = rooms_elem.groupby('playerElement')[valid_cols].sum().sum(axis=1)

        cast_col = f'cast_{spell}'
        casts_by_elem = (
            rooms_elem.groupby('playerElement')[cast_col].sum()
            if cast_col in rooms_elem.columns
            else pd.Series(0, index=kills_by_elem.index)
        )

        for elem in kills_by_elem.index:
            k = int(kills_by_elem[elem])
            c = int(casts_by_elem.get(elem, 0))
            eff = round(k / c, 4) if c > 0 else 0.0
            rows.append({'element': elem, 'spell': spell, 'kills': k, 'casts': c, 'efficiency': eff})

    return pd.DataFrame(rows)


def compute_balance_index(df_sessions: pd.DataFrame,
                          nlp_results: pd.DataFrame,
                          df_rooms: pd.DataFrame = None) -> pd.DataFrame:
    """
    Balance Index = 0.5 × win_rate + 0.3 × kills_efficiency_norm + 0.2 × sentiment_norm por elemento.

    - win_rate           : proporción de isVictory por playerElement
    - kills_efficiency   : si df_rooms disponible, usa killedWith_* (eficiencia real);
                           si no, usa kills_per_min normalizada como fallback
    - sentiment_norm     : media de nlp_results por elemento normalizada a [0,1] (NaN → 0.5 neutral)
    """
    # Win rate por elemento
    wr = df_sessions.groupby('playerElement')['isVictory'].mean().rename('win_rate')

    # Kills efficiency: killedWith real si disponible, kills_per_min como fallback
    if df_rooms is not None:
        kw_eff = compute_killedwith_efficiency(df_rooms, df_sessions)
        # Eficiencia media ponderada por hechizo para cada elemento
        kpm = kw_eff.groupby('element').apply(
            lambda g: (g['kills'].sum() / g['casts'].sum()) if g['casts'].sum() > 0 else 0.0
        ).rename('kills_efficiency_raw')
        kpm.index.name = 'playerElement'
    else:
        kpm = df_sessions.groupby('playerElement')['kills_per_min'].mean().rename('kills_efficiency_raw')

    kpm_min, kpm_max = kpm.min(), kpm.max()
    if kpm_max > kpm_min:
        keff = ((kpm - kpm_min) / (kpm_max - kpm_min)).rename('kills_efficiency')
    else:
        keff = pd.Series(0.5, index=kpm.index, name='kills_efficiency')

    # Sentiment score medio por elemento (NaN → 0.5)
    sent = nlp_results.groupby('playerElement')['sentiment_score'].mean().rename('sentiment_score')

    # Merge todo
    df_bi = pd.concat([wr, keff, sent], axis=1).reset_index()
    df_bi['sentiment_score'] = df_bi['sentiment_score'].fillna(0.5)

    # Fórmula
    df_bi['balance_index'] = (
        df_bi['win_rate'] *
        df_bi['sentiment_score'] *
        (1 + df_bi['kills_efficiency'])
    ).round(4)

    return df_bi.sort_values('balance_index', ascending=False).reset_index(drop=True)


def element_level_matrix(df_sessions: pd.DataFrame,
                         df_levels: pd.DataFrame,
                         df_rooms: pd.DataFrame,
                         metric: str = 'death_rate') -> pd.DataFrame:
    """
    Construye matriz elemento × nivel con la métrica indicada.
    metric: 'death_rate' | 'completion_rate' | 'avg_damage'
    Devuelve DataFrame pivotado (elementos en filas, niveles en columnas).
    """
    # Unir sesiones con niveles para tener playerElement
    df = df_levels.merge(df_sessions[['sessionId', 'playerElement']], on='sessionId', how='left')

    if metric == 'death_rate':
        # Muertes medias por sala en ese nivel (deaths / timeSecs como proxy de tasa)
        agg = df.groupby(['playerElement', 'levelId'])['deaths'].mean()
        title = 'Muertes medias por nivel'

    elif metric == 'completion_rate':
        # Proporción de intentos completados (incomplete=False)
        df['completed'] = (~df['incomplete']).astype(int)
        agg = df.groupby(['playerElement', 'levelId'])['completed'].mean()
        title = 'Tasa de completado por nivel'

    elif metric == 'avg_damage':
        agg = df.groupby(['playerElement', 'levelId'])['damageTaken'].mean()
        title = 'Daño recibido medio por nivel'

    else:
        raise ValueError(f'metric desconocida: {metric}')

    # Ordenar niveles numéricamente
    pivot = agg.unstack(level='levelId').fillna(0)
    level_order = sorted(pivot.columns, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    pivot = pivot[level_order]

    return pivot


def top_abandoned_rooms(df_rooms: pd.DataFrame,
                        df_sessions: pd.DataFrame,
                        n: int = 10) -> pd.DataFrame:
    """
    Identifica las n salas con mayor score de abandono.
    Score = deaths_norm + damageTaken_norm (ambos normalizados 0-1).
    """
    df = df_rooms.merge(df_sessions[['sessionId', 'playerElement']], on='sessionId', how='left')

    agg = df.groupby(['levelId', 'roomId']).agg(
        deaths_mean=('deaths', 'mean'),
        damage_mean=('damageTaken', 'mean'),
        n_visits=('sessionId', 'count'),
    ).reset_index()

    # Normalizar y combinar
    for col in ['deaths_mean', 'damage_mean']:
        col_max = agg[col].max()
        agg[f'{col}_norm'] = agg[col] / col_max if col_max > 0 else 0

    agg['abandonment_score'] = (agg['deaths_mean_norm'] + agg['damage_mean_norm']) / 2

    return agg.nlargest(n, 'abandonment_score').reset_index(drop=True)


def generate_recommendations(balance_index: pd.DataFrame,
                              statistical_results: pd.DataFrame,
                              spell_efficiency: pd.DataFrame,
                              eda_hallazgos: pd.DataFrame) -> pd.DataFrame:
    """
    Genera tabla de recomendaciones de balance priorizadas.
    Columnas: element, area, finding, recommendation, priority, evidence
    """
    recs = []

    # Calcular eficiencias de hechizos dinámicamente desde spell_efficiency si está disponible
    def _eff(spell: str) -> float:
        if spell_efficiency is not None and 'spell_type' in spell_efficiency.columns:
            row = spell_efficiency[spell_efficiency['spell_type'] == spell]
            if len(row):
                return float(row.iloc[0].get('spell_efficiency', row.iloc[0].get('efficiency', 0)))
        return 0.0

    eff_beam = _eff('Beam')
    eff_proj = _eff('Projectile')
    eff_aoe  = _eff('AOE')
    eff_blast = _eff('Blast')
    ratio_beam_proj = eff_beam / eff_proj if eff_proj > 0 else 0

    # --- BUGS DE TELEMETRÍA (prioridad Alta — evidencia directa) ---
    recs.append({
        'element': 'Todos',
        'area': 'spell_balance',
        'finding': f'AOE eficiente ({eff_aoe:.2f} kills/cast) pero muy infrautilizado vs Projectile',
        'recommendation': 'Mejorar feedback visual y comunicación del hechizo AOE para que los jugadores '
                          'lo perciban como opción viable. Estadísticamente es eficiente.',
        'priority': 'Media',
        'evidence': f'EDA (datos WebGL limpios): AOE eff={eff_aoe:.3f} via killedWith_*Aoe',
    })
    recs.append({
        'element': 'Todos',
        'area': 'telemetry',
        'finding': 'damage_Unknown con valores altos en salas',
        'recommendation': 'Identificar la fuente de daño "Unknown" en Unity y asignarle '
                          'un tipo correcto para que el análisis de balance de enemigos sea válido.',
        'priority': 'Alta',
        'evidence': 'EDA: columna damage_Unknown presente con valores significativos',
    })

    # --- BALANCE DE HECHIZOS (prioridad Alta) ---
    recs.append({
        'element': 'Todos',
        'area': 'spell_balance',
        'finding': f'Beam ({eff_beam:.2f} kills/cast) vs Projectile ({eff_proj:.3f}) — desbalance x{ratio_beam_proj:.0f}',
        'recommendation': 'Rebalancear: reducir daño de Beam o aumentar velocidad/daño de Projectile '
                          'para igualar su valor competitivo. Beam domina demasiado.',
        'priority': 'Alta',
        'evidence': f'EDA: Beam={eff_beam:.3f}, Blast={eff_blast:.3f}, Projectile={eff_proj:.3f}, AOE={eff_aoe:.3f}',
    })
    recs.append({
        'element': 'Todos',
        'area': 'spell_balance',
        'finding': 'noMana frecuente en Blast indica demanda insatisfecha de maná',
        'recommendation': 'Reducir el coste de maná de Blast o aumentar la regeneración de maná '
                          'para que los jugadores puedan usar su hechizo preferido con más fluidez.',
        'priority': 'Media',
        'evidence': 'EDA: noMana_Blast es el valor más alto en análisis de demanda latente',
    })

    # --- DIFICULTAD (prioridad Media) ---
    recs.append({
        'element': 'Todos',
        'area': 'difficulty',
        'finding': 'Level1 es el mayor cuello de botella del funnel (mayor tasa de abandono)',
        'recommendation': 'Revisar la dificultad del primer nivel. Considerar una sala tutorial '
                          'o reducir el daño de los primeros enemigos para mejorar la retención inicial.',
        'priority': 'Media',
        'evidence': 'EDA: funnel de niveles — Level1 tiene mayor tasa de sesiones que no progresan',
    })
    recs.append({
        'element': 'Todos',
        'area': 'difficulty',
        'finding': 'attempt > 1 frecuente (grinding) en niveles medios',
        'recommendation': 'Evaluar si el grinding es intencionado o indica picos de dificultad '
                          'no diseñados. Añadir checkpoints intermedios si el diseño no lo contempla.',
        'priority': 'Media',
        'evidence': 'EDA: attempt medio > 1 en Level1-Level4; deaths corregidos desde attempt',
    })

    # --- BALANCE DE ELEMENTOS ---
    # Recuperar p-value de Kruskal kills_per_min
    kw_row = statistical_results[
        (statistical_results['test'] == 'Kruskal-Wallis') &
        (statistical_results['variable'] == 'kills_per_min')
    ]
    kw_p = kw_row['p_value'].values[0] if len(kw_row) else 1.0

    # Elemento con menor balance index
    worst = balance_index.iloc[-1]['playerElement'] if len(balance_index) else 'Wind'
    best  = balance_index.iloc[0]['playerElement']  if len(balance_index) else 'Fire'

    recs.append({
        'element': worst,
        'area': 'element_balance',
        'finding': f'{worst} tiene el Balance Index más bajo del dataset',
        'recommendation': f'Investigar si {worst} tiene mecánicas únicas que dificultan su uso. '
                          f'Comparar kit de hechizos con {best} y ajustar si hay asimetría injustificada.',
        'priority': 'Media' if kw_p >= 0.05 else 'Alta',
        'evidence': f'Balance Index: {worst} = {balance_index.iloc[-1]["balance_index"]:.4f}; '
                    f'Kruskal-Wallis kills_per_min p={kw_p:.3f}',
    })

    # --- FEEDBACK DE JUGADORES ---
    recs.append({
        'element': 'Todos',
        'area': 'difficulty',
        'finding': 'Jugador pide que enemigos no se activen al entrar para planificar estrategia',
        'recommendation': 'Considerar añadir un breve período de gracia (1-2 segundos) al entrar '
                          'en una sala antes de que los enemigos comiencen a atacar.',
        'priority': 'Baja',
        'evidence': 'NLP: comentario Wind — "me gustaria que los enemigos no se activaran nada mas entrar"',
    })
    recs.append({
        'element': 'Wind',
        'area': 'telemetry',
        'finding': 'Bug reportado: arqueros persiguen pero no atacan',
        'recommendation': 'Revisar el comportamiento de IA de enemigos arqueros. '
                          'Posible problema con la máquina de estados de ataque a distancia.',
        'priority': 'Alta',
        'evidence': 'NLP: comentario Wind — "arqueros no me estaban atacando, me perseguian pero no atacaban"',
    })

    # --- MUESTRA — conteos dinámicos desde balance_index ---
    n_clean  = int(balance_index['n_sessions'].sum()) if 'n_sessions' in balance_index.columns else None
    elem_counts = balance_index.set_index('playerElement')['n_sessions'].to_dict() \
        if 'n_sessions' in balance_index.columns else {}
    sample_detail = ', '.join(f'{e}={int(n)}' for e, n in elem_counts.items()) if elem_counts \
        else 'ver check_sample_size()'
    n_str = f'n={n_clean} sesiones limpias' if n_clean else 'muestra insuficiente'
    recs.append({
        'element': 'Todos',
        'area': 'sample_size',
        'finding': f'Dataset insuficiente: {n_str}, {sample_detail}',
        'recommendation': 'Recolectar mínimo 30 sesiones por elemento antes de tomar decisiones '
                          'de balance basadas en estadística. Actualizar resultados con datos frescos.',
        'priority': 'Alta',
        'evidence': 'Estadística Fase 3: check_sample_size() — todos los grupos < 30',
    })

    df_recs = pd.DataFrame(recs)
    # Ordenar por prioridad
    priority_order = {'Alta': 0, 'Media': 1, 'Baja': 2}
    df_recs['_order'] = df_recs['priority'].map(priority_order)
    df_recs = df_recs.sort_values('_order').drop(columns='_order').reset_index(drop=True)

    return df_recs


def generate_ai_recommendations(balance_index: pd.DataFrame,
                                 statistical_results: pd.DataFrame,
                                 spell_efficiency: pd.DataFrame,
                                 top_rooms: pd.DataFrame,
                                 existing_recs: pd.DataFrame = None) -> list[dict]:
    """
    Genera recomendaciones adicionales de balance usando Gemini.
    Devuelve lista de dicts con claves: element, area, priority, finding, recommendation, evidence.
    """
    import sys
    import json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import gemini_client

    bi_str = balance_index.to_string(index=False) if balance_index is not None else 'No disponible'

    stat_relevantes = statistical_results[statistical_results['p_value'] < 0.1] \
        if statistical_results is not None and 'p_value' in statistical_results.columns \
        else statistical_results
    stat_str = stat_relevantes.to_string(index=False) if stat_relevantes is not None else 'No disponible'

    spell_str = spell_efficiency.to_string(index=False) if spell_efficiency is not None else 'No disponible'
    rooms_str = top_rooms.head(10).to_string(index=False) if top_rooms is not None else 'No disponible'

    existing_str = ''
    if existing_recs is not None and not existing_recs.empty:
        lines = []
        for _, r in existing_recs.iterrows():
            arr = ' [ARREGLADO]' if r.get('arreglado') else ''
            lines.append(
                f"- [{r.get('priority','?')}] [{r.get('element','?')}] [{r.get('area','?')}]{arr} "
                f"{str(r.get('finding', ''))[:120]}"
            )
        existing_str = '\n'.join(lines)

    prompt = f"""Eres un Game Designer experto en balance de videojuegos de tipo Hack & Slash.
Tienes datos analíticos del juego "Arcane Descent" con 4 elementos jugables (Fire, Water, Earth, Wind).
El dataset tiene ~34 sesiones limpias — los resultados son orientativos, no estadísticamente definitivos.

## BALANCE INDEX POR ELEMENTO:
{bi_str}

## TESTS ESTADÍSTICOS SIGNIFICATIVOS (p < 0.1):
{stat_str}

## EFICIENCIA DE HECHIZOS (kills/cast):
{spell_str}

## TOP SALAS MÁS DIFÍCILES:
{rooms_str}

## RECOMENDACIONES YA IDENTIFICADAS (NO repetir):
{existing_str if existing_str else 'Ninguna aún'}

---

Genera 5-8 recomendaciones de balance ADICIONALES que no estén ya cubiertas.
Devuelve ÚNICAMENTE un array JSON válido, sin texto adicional, sin bloques de código markdown.
Cada elemento del array debe tener exactamente estas claves:
- "element": elemento afectado (Fire / Water / Earth / Wind / Todos / nombre compuesto)
- "area": una de spell_balance, element_balance, difficulty, ux, telemetry, enemy_balance
- "priority": Alta, Media o Baja
- "finding": descripción concisa del problema detectado en los datos (máx 150 caracteres)
- "recommendation": acción específica para el equipo de desarrollo Unity (máx 200 caracteres)
- "evidence": dato o métrica que respalda esto (máx 150 caracteres)

Instrucciones:
1. Cada recomendación debe tener evidencia en los datos proporcionados.
2. Propón al menos una recomendación sobre enemigos y una sobre UX.
3. No repitas hallazgos ya listados en RECOMENDACIONES YA IDENTIFICADAS.
4. Responde únicamente en español."""

    respuesta = gemini_client.invoke(prompt)

    # Limpiar posibles bloques de código que Gemini añada igualmente
    texto = respuesta.strip()
    if texto.startswith('```'):
        texto = '\n'.join(texto.split('\n')[1:])
    if texto.endswith('```'):
        texto = '\n'.join(texto.split('\n')[:-1])
    texto = texto.strip()

    return json.loads(texto)


def generate_executive_summary(balance_index: pd.DataFrame,
                                statistical_results: pd.DataFrame,
                                recommendations: pd.DataFrame,
                                n_sessions: int = 34) -> str:
    """
    Genera un resumen ejecutivo del análisis de balance usando Gemini.
    Devuelve el resumen como string en markdown.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import gemini_client

    bi_str = balance_index.to_string(index=False) if balance_index is not None else 'No disponible'

    high_recs = recommendations[recommendations['priority'] == 'Alta'] \
        if recommendations is not None and 'priority' in recommendations.columns \
        else pd.DataFrame()
    high_recs_str = high_recs[['element', 'area', 'finding']].to_string(index=False) \
        if not high_recs.empty else 'Ninguna identificada'

    # KPIs globales desde balance_index
    total_sessions = n_sessions
    best_elem = balance_index.iloc[0]['playerElement'] if len(balance_index) else 'N/A'
    worst_elem = balance_index.iloc[-1]['playerElement'] if len(balance_index) else 'N/A'
    win_rate_global = balance_index['win_rate'].mean() if 'win_rate' in balance_index.columns else 0

    prompt = f"""Eres el Data Scientist a cargo del análisis de balance del juego "Arcane Descent".
Debes escribir un resumen ejecutivo profesional para el equipo de desarrollo (Game Designers y programadores Unity).

## CONTEXTO DEL ANÁLISIS
- Sesiones analizadas: {total_sessions} sesiones WebGL limpias (filtradas de anomalías)
- Elementos jugables: Fire, Water, Earth, Wind
- Win rate global: {win_rate_global:.1%}
- Elemento más equilibrado: {best_elem}
- Elemento con mayor margen de mejora: {worst_elem}

## BALANCE INDEX FINAL (métrica compuesta):
{bi_str}

## RECOMENDACIONES DE ALTA PRIORIDAD:
{high_recs_str}

---

Escribe un resumen ejecutivo en español con las siguientes secciones:

## Resumen Ejecutivo — Análisis de Balance Arcane Descent

### Estado General del Balance
Evaluación general en 2-3 párrafos. ¿El juego está equilibrado? ¿Qué destacar?

### Hallazgos Clave
Bullet points con los 5 hallazgos más importantes del análisis.

### Elementos Jugables
Tabla o resumen comparativo de los 4 elementos con fortalezas y debilidades.

### Prioridades Inmediatas
Los 3 cambios más urgentes a implementar antes del próximo playtest.

### Limitaciones del Análisis
Mención breve de las limitaciones metodológicas (tamaño de muestra, etc.).

Tono: profesional, directo, orientado a decisiones. Máximo 600 palabras."""

    return gemini_client.invoke(prompt)
