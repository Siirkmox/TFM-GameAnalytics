"""
preprocessing.py — Aplanado de la jerarquía Firestore y feature engineering.

Transforma la estructura anidada sessions → levels → rooms en tres DataFrames
planos listos para análisis:
  - df_sessions : una fila por sesión
  - df_levels   : una fila por nivel (con sessionId como clave foránea)
  - df_rooms    : una fila por sala  (con sessionId y levelId como claves)

Las columnas dinámicas de sala (kills_X, cast_X, etc.) se mantienen en formato
wide para el EDA y se ofrecen también en formato long (tidy) para análisis
por tipo de enemigo/hechizo.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


# ─── Constantes ──────────────────────────────────────────────────────────────

# Prefijos de columnas dinámicas en salas
DYNAMIC_PREFIXES = ['kills', 'killedWith', 'cast', 'miss', 'blocked', 'damage', 'status', 'noMana']

# Columnas estáticas de sala (no dinámicas)
ROOM_STATIC_COLS = ['sessionId', 'levelId', 'roomId', 'timeSecs', 'deaths', 'damageTaken', 'firstSpell']

# Normalización de nombres históricos killedWith_* → nombres canónicos post-fix
# Permite coexistencia de sesiones pre-fix y post-fix en el mismo parquet.
# El elemento se mantiene via playerElement de la sesión (join rooms ↔ sessions).
KILLEDWITH_RENAME = {
    # Projectile (nombres por elemento → canónico)
    'killedWith_Fireball':             'killedWith_Projectile',
    'killedWith_StoneBullet':          'killedWith_Projectile',
    'killedWith_WaterballProjectile':  'killedWith_Projectile',
    'killedWith_WindBulletProjectile': 'killedWith_Projectile',
    # Blast
    'killedWith_FireBlast':            'killedWith_Blast',
    'killedWith_FireBlast_(1)':        'killedWith_Blast',
    'killedWith_WaterBlast':           'killedWith_Blast',
    'killedWith_WindBlast':            'killedWith_Blast',
    'killedWith_Particle_System':      'killedWith_Blast',   # Earth Blast pre-fix (bug collider)
    # AOE
    'killedWith_EarthSlamSpikesAoe':   'killedWith_AOE',
    'killedWith_WaterAoeBody':         'killedWith_AOE',
    'killedWith_WindAoe':              'killedWith_AOE',
    # Beam
    'killedWith_BeamBody':             'killedWith_Beam',
}


# ─── Carga ───────────────────────────────────────────────────────────────────

def load_raw(path: str) -> list[dict]:
    """Carga el JSON raw exportado por firestore_client."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ─── Aplanado ────────────────────────────────────────────────────────────────

def flatten_sessions(sessions: list[dict]) -> pd.DataFrame:
    """
    Construye el DataFrame de sesiones a partir de la lista raw.
    Convierte timestamps Unix a datetime y normaliza tipos.
    """
    rows = []
    for s in sessions:
        rows.append({
            'sessionId'           : s.get('sessionId'),
            'playerElement'       : s.get('playerElement'),
            'startUnixTime'       : s.get('startUnixTime'),
            'endUnixTime'         : s.get('endUnixTime'),
            'isVictory'           : bool(s.get('isVictory', False)),
            'totalTimeSecs'       : float(s.get('totalTimeSecs', 0)),
            'totalDeaths'         : int(s.get('totalDeaths', 0)),
            'totalKills'          : int(s.get('totalKills', 0)),
            'levelsCompleted'     : int(s.get('levelsCompleted', 0)),
            'playerComment'       : s.get('playerComment', '') or '',
            'playerCommentContext': s.get('playerCommentContext', '') or '',
            'platform'            : s.get('platform'),
            'gameVersion'         : s.get('gameVersion'),
            'eventCount'          : int(s.get('eventCount', 0)),
        })

    df = pd.DataFrame(rows)

    # Convertir timestamps a datetime UTC
    df['startDatetime'] = pd.to_datetime(df['startUnixTime'], unit='s', utc=True)
    df['endDatetime']   = pd.to_datetime(df['endUnixTime'],   unit='s', utc=True)

    # Booleano para sesiones con comentario
    df['hasComment'] = df['playerComment'].str.strip().str.len() > 0

    return df


def flatten_levels(sessions: list[dict]) -> pd.DataFrame:
    """
    Construye el DataFrame de niveles, uno por fila.
    Incluye sessionId como clave foránea.
    """
    rows = []
    for s in sessions:
        sid = s.get('sessionId')
        for lv in s.get('levels', []):
            rows.append({
                'sessionId'  : sid,
                'levelId'    : lv.get('levelId'),
                'attempt'    : _to_int(lv.get('attempt', 1)),
                'timeSecs'   : _to_float(lv.get('timeSecs', 0)),
                'kills'      : _to_int(lv.get('kills', 0)),
                'deaths'     : _to_int(lv.get('deaths', 0)),
                'damageTaken': _to_float(lv.get('damageTaken', 0)),
                'incomplete' : bool(lv.get('incomplete', False)),
            })

    return pd.DataFrame(rows)


def flatten_rooms(sessions: list[dict]) -> pd.DataFrame:
    """
    Construye el DataFrame de salas, uno por fila.
    Las columnas dinámicas (kills_X, cast_X, etc.) se incluyen como columnas wide.
    Los NaN en columnas dinámicas se rellenan con 0 (ausencia = no ocurrió).
    """
    rows = []
    for s in sessions:
        sid = s.get('sessionId')
        for lv in s.get('levels', []):
            lid = lv.get('levelId')
            for rm in lv.get('rooms', []):
                row = {
                    'sessionId'  : sid,
                    'levelId'    : lid,
                    'roomId'     : rm.get('roomId'),
                    'timeSecs'   : _to_float(rm.get('timeSecs', 0)),
                    'deaths'     : _to_int(rm.get('deaths', 0)),
                    'damageTaken': _to_float(rm.get('damageTaken', 0)),
                    'firstSpell' : rm.get('firstSpell', ''),
                }
                # Añadir todas las columnas dinámicas del documento
                for k, v in rm.items():
                    if k not in ROOM_STATIC_COLS and _is_dynamic(k):
                        row[k] = _to_float(v)
                rows.append(row)

    df = pd.DataFrame(rows)

    # Rellenar NaN en columnas dinámicas con 0
    dynamic_cols = [c for c in df.columns if _is_dynamic(c)]
    df[dynamic_cols] = df[dynamic_cols].fillna(0)

    # Normalizar nombres killedWith_* históricos → canónicos
    df = normalize_killedwith(df)

    return df


def normalize_killedwith(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unifica nombres killedWith_* históricos (pre-fix Unity) con los canónicos (post-fix).
    Columnas con el mismo nombre canónico se suman (para coexistencia pre/post fix).
    El elemento del jugador se mantiene via playerElement de la sesión — no se pierde
    información porque cada sesión pertenece a un único playerElement.
    """
    cols_a_renombrar = {old: new for old, new in KILLEDWITH_RENAME.items() if old in df.columns}

    for old_col, new_col in cols_a_renombrar.items():
        if new_col in df.columns:
            # Ya existe la columna canónica (sesiones post-fix) → sumar
            df[new_col] = df[new_col] + df[old_col]
        else:
            df[new_col] = df[old_col]
        df = df.drop(columns=[old_col])

    return df


# ─── Feature engineering ─────────────────────────────────────────────────────

def add_session_features(df_sessions: pd.DataFrame, df_levels: pd.DataFrame,
                         df_rooms: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas derivadas al DataFrame de sesiones (estilo RFM adaptado a gaming).

    Nuevas columnas:
      - kd_ratio           : kills / max(deaths, 1)
      - time_per_level     : totalTimeSecs / max(levelsCompleted, 1)
      - kills_per_min      : totalKills / (totalTimeSecs / 60)
      - level_finish_rate  : levelsCompleted / niveles intentados en la sesión (mide abandono dentro de un nivel)
      - game_completion    : levelsCompleted / 4 (% del juego completado — el juego tiene 4 niveles totales)
      - total_cast         : suma de todos los cast_X en salas de la sesión
      - total_rooms        : número de salas visitadas en la sesión
    """
    df = df_sessions.copy()

    # Ratios básicos de combate
    df['kd_ratio']       = df['totalKills'] / df['totalDeaths'].clip(lower=1)
    df['kills_per_min']  = df['totalKills'] / (df['totalTimeSecs'] / 60).clip(lower=0.01)
    df['time_per_level'] = df['totalTimeSecs'] / df['levelsCompleted'].clip(lower=1)

    # Número total de niveles en la sesión (completados + incompletos)
    levels_per_session = df_levels.groupby('sessionId').size().rename('totalLevels')
    df = df.merge(levels_per_session, on='sessionId', how='left')
    df['totalLevels']    = df['totalLevels'].fillna(0).astype(int)

    # level_finish_rate: de los niveles que intento, cuantos termino (antes se llamaba completion_rate)
    # No mide '% del juego completado' — solo mide abandono dentro de un nivel iniciado.
    df['level_finish_rate'] = df['levelsCompleted'] / df['totalLevels'].clip(lower=1)

    # game_completion: '% del juego completado' sobre los 4 niveles totales del juego.
    # Esta es la metrica que normalmente se entiende por 'porcentaje del juego completado'.
    GAME_TOTAL_LEVELS = 4
    df['game_completion'] = df['levelsCompleted'] / GAME_TOTAL_LEVELS

    # Alias mantenido por compatibilidad con notebooks existentes — DEPRECATED.
    df['completion_rate'] = df['level_finish_rate']

    # Total de salas visitadas
    rooms_per_session = df_rooms.groupby('sessionId').size().rename('totalRooms')
    df = df.merge(rooms_per_session, on='sessionId', how='left')
    df['totalRooms'] = df['totalRooms'].fillna(0).astype(int)

    # Total de hechizos lanzados (suma de todos los cast_X)
    cast_cols = [c for c in df_rooms.columns if c.startswith('cast_')]
    if cast_cols:
        total_cast = df_rooms.groupby('sessionId')[cast_cols].sum().sum(axis=1).rename('totalCast')
        df = df.merge(total_cast, on='sessionId', how='left')
        df['totalCast'] = df['totalCast'].fillna(0).astype(int)
    else:
        df['totalCast'] = 0

    return df


def add_room_features(df_rooms: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas derivadas al DataFrame de salas.

    Nuevas columnas:
      - total_kills_room  : suma de todos los kills_X de la sala
      - total_cast_room   : suma de todos los cast_X de la sala
      - total_miss_room   : suma de todos los miss_X de la sala
      - cast_accuracy     : (total_cast - total_miss) / max(total_cast, 1)
      - kills_per_sec     : total_kills_room / max(timeSecs, 1)
    """
    df = df_rooms.copy()

    kills_cols = [c for c in df.columns if c.startswith('kills_')]
    cast_cols  = [c for c in df.columns if c.startswith('cast_')]
    miss_cols  = [c for c in df.columns if c.startswith('miss_')]

    df['total_kills_room'] = df[kills_cols].sum(axis=1) if kills_cols else 0
    df['total_cast_room']  = df[cast_cols].sum(axis=1)  if cast_cols  else 0
    df['total_miss_room']  = df[miss_cols].sum(axis=1)  if miss_cols  else 0

    df['cast_accuracy'] = (df['total_cast_room'] - df['total_miss_room']) / df['total_cast_room'].clip(lower=1)
    df['kills_per_sec'] = df['total_kills_room'] / df['timeSecs'].clip(lower=1)

    return df


def compute_spell_efficiency(df_rooms: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula spell_efficiency = kills / cast por tipo de hechizo, agregado globalmente.

    Para cada tipo de hechizo detectado en las columnas cast_X, cruza con los kills
    atribuidos a ese hechizo via killedWith_X para obtener una métrica de eficiencia.

    Returns:
        DataFrame con columnas: spell_type, total_cast, total_kills, spell_efficiency
        ordenado por spell_efficiency descendente.
    """
    # Excluir columnas derivadas que empiezan por cast_ pero no son hechizos
    DERIVED = {'cast_accuracy'}
    cast_cols = [c for c in df_rooms.columns if c.startswith('cast_') and c not in DERIVED]
    rows = []

    for col in cast_cols:
        spell = col[len('cast_'):]
        total_cast = df_rooms[col].sum()
        if total_cast == 0:
            continue

        # Buscar kills atribuidos a este hechizo via killedWith_X
        killed_col = f'killedWith_{spell}'
        if killed_col in df_rooms.columns:
            total_kills = df_rooms[killed_col].sum()
        else:
            # Si no hay killedWith_ exacto, buscar coincidencia parcial por nombre
            matches = [c for c in df_rooms.columns
                       if c.startswith('killedWith_') and spell.lower() in c.lower()]
            total_kills = df_rooms[matches].sum().sum() if matches else 0.0

        rows.append({
            'spell_type'       : spell,
            'total_cast'       : total_cast,
            'total_kills'      : total_kills,
            'spell_efficiency' : total_kills / total_cast,
        })

    df_eff = pd.DataFrame(rows)
    if not df_eff.empty:
        df_eff = df_eff.sort_values('spell_efficiency', ascending=False).reset_index(drop=True)

    return df_eff


def compute_spell_usage_ratios(df_rooms: pd.DataFrame, df_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el uso relativo de cada hechizo por sesión (% de cast sobre total).
    Útil como features para el modelo ML de predicción de victoria.

    Returns:
        DataFrame con una fila por sesión y columnas ratio_cast_<spell> (0–1).
    """
    DERIVED = {'cast_accuracy'}
    cast_cols = [c for c in df_rooms.columns if c.startswith('cast_') and c not in DERIVED]
    if not cast_cols:
        return df_sessions[['sessionId']].copy()

    # Suma de cada hechizo por sesión
    df_agg = df_rooms.groupby('sessionId')[cast_cols].sum()
    df_agg['total'] = df_agg.sum(axis=1).clip(lower=1)

    for col in cast_cols:
        spell = col[len('cast_'):]
        df_agg[f'ratio_cast_{spell}'] = df_agg[col] / df_agg['total']

    ratio_cols = [c for c in df_agg.columns if c.startswith('ratio_cast_')]
    return df_agg[ratio_cols].reset_index()


def rooms_to_long(df_rooms: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Convierte las columnas dinámicas de un prefijo dado (ej. 'cast') a formato
    long (tidy): una fila por (sesión, nivel, sala, tipo).

    Útil para comparar hechizos o enemigos entre sí.

    Args:
        prefix: uno de 'kills', 'killedWith', 'cast', 'miss', 'blocked',
                'damage', 'status', 'noMana'
    Returns:
        DataFrame con columnas: sessionId, levelId, roomId, <prefix>_type, value
    """
    id_cols   = ['sessionId', 'levelId', 'roomId']
    val_cols  = [c for c in df_rooms.columns if c.startswith(f'{prefix}_')]

    if not val_cols:
        return pd.DataFrame(columns=id_cols + [f'{prefix}_type', 'value'])

    df_long = df_rooms[id_cols + val_cols].melt(
        id_vars    = id_cols,
        value_vars = val_cols,
        var_name   = f'{prefix}_type',
        value_name = 'value'
    )
    # Limpiar el prefijo del nombre de tipo
    df_long[f'{prefix}_type'] = df_long[f'{prefix}_type'].str.replace(f'{prefix}_', '', n=1)
    # Eliminar filas con valor 0 (no ocurrió)
    df_long = df_long[df_long['value'] > 0].reset_index(drop=True)

    return df_long


# ─── Detección de anomalías ──────────────────────────────────────────────────

# Número máximo de niveles que puede completar un jugador en una sola sesión
MAX_LEVELS_PER_SESSION = 4

def flag_anomalies(df_sessions: pd.DataFrame, df_levels: pd.DataFrame) -> pd.DataFrame:
    """
    Marca sesiones anómalas con una columna `anomaly_flag` (bool) y
    una columna `anomaly_reasons` (str) con la lista de causas detectadas.

    Anomalías conocidas y sus causas raíz:
    ─────────────────────────────────────────────────────────────────────────
    1. levelsCompleted > MAX_LEVELS_PER_SESSION
       Causa raíz (bug HANDS): ElementSelectionUI no llamaba a
       GameStatsTracker.StartNewGame() al seleccionar elemento desde el
       Main Menu directo (sin pasar por PlayAgain / GameOver).
       El acumulador _levelsCompleted heredaba el valor de la sesión anterior,
       resultando en múltiplos de 4 (8, 12, 16…).
       Fix aplicado en ElementSelectionUI.ConfirmAndLoad() el 2026-05-13.

    2. levelsCompleted > levels_en_Firestore
       La sesión declara más niveles completados de los que tiene subcolecciones
       en Firestore. Misma causa raíz que (1), o fallo parcial de upload.

    3. Sesión muy corta con niveles completados (< 30 s con lvlsCompleted > 0)
       Sesión fantasma: se creó un documento en Firestore pero el jugador no
       llegó a jugar (carga de página, cierre inmediato, etc.).

    4. Pocos eventos con muchos niveles completados (< 20 events, lvlsCompleted >= 4)
       Los datos de sala provienen de sesiones anteriores (datos fantasma).
       El eventCount real debería ser proporcional al tiempo de juego.

    5. totalTimeSecs extremadamente largo (> 3600 s)
       Sesión dejada en pausa durante horas. El tiempo no refleja tiempo real
       de juego activo.

    Returns:
        df_sessions con columnas añadidas: `anomaly_flag`, `anomaly_reasons`,
        `levels_in_db` (número de subcolecciones en Firestore para validación).
    """
    df = df_sessions.copy()

    # Número de niveles subidos a Firestore por sesión
    levels_in_db = df_levels.groupby('sessionId').size().rename('levels_in_db')
    df = df.merge(levels_in_db, on='sessionId', how='left')
    df['levels_in_db'] = df['levels_in_db'].fillna(0).astype(int)

    reasons_list = []
    for _, row in df.iterrows():
        reasons = []

        if row['levelsCompleted'] > MAX_LEVELS_PER_SESSION:
            reasons.append(
                f"levelsCompleted={row['levelsCompleted']} > MAX ({MAX_LEVELS_PER_SESSION})"
                " — bug acumulación GameStatsTracker"
            )

        if row['levelsCompleted'] > 0 and row['levelsCompleted'] != row['levels_in_db']:
            reasons.append(
                f"levelsCompleted={row['levelsCompleted']} != levels_Firestore={row['levels_in_db']}"
            )

        if row['levelsCompleted'] > 0 and row['totalTimeSecs'] < 30:
            reasons.append(
                f"tiempo={row['totalTimeSecs']:.1f}s con levelsCompleted={row['levelsCompleted']}"
                " — sesión fantasma"
            )

        if row['levelsCompleted'] >= MAX_LEVELS_PER_SESSION and row['eventCount'] < 20:
            reasons.append(
                f"eventCount={row['eventCount']} para levelsCompleted={row['levelsCompleted']}"
                " — datos de sala fantasma"
            )

        if row['totalTimeSecs'] > 3600:
            reasons.append(
                f"totalTimeSecs={row['totalTimeSecs']:.0f}s > 3600 — sesión pausada/abandonada"
            )

        reasons_list.append('; '.join(reasons))

    df['anomaly_reasons'] = reasons_list
    df['anomaly_flag']    = df['anomaly_reasons'].str.len() > 0

    n_anomalies = df['anomaly_flag'].sum()
    if n_anomalies > 0:
        import warnings
        warnings.warn(
            f"[flag_anomalies] {n_anomalies} sesiones anómalas detectadas. "
            f"Revisar columna 'anomaly_reasons'.",
            UserWarning, stacklevel=2
        )

    return df


# ─── Pipeline completo ───────────────────────────────────────────────────────

def run_pipeline(raw_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el pipeline completo de aplanado, feature engineering y detección de anomalías.

    Returns:
        (df_sessions, df_levels, df_rooms) con todas las columnas derivadas.
        df_sessions incluye `anomaly_flag` y `anomaly_reasons`.
    """
    sessions    = load_raw(raw_path)
    df_sessions = flatten_sessions(sessions)
    df_levels   = flatten_levels(sessions)
    df_rooms    = flatten_rooms(sessions)

    df_rooms    = add_room_features(df_rooms)
    df_sessions = add_session_features(df_sessions, df_levels, df_rooms)
    df_sessions = flag_anomalies(df_sessions, df_levels)

    return df_sessions, df_levels, df_rooms


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_int(v) -> int:
    try: return int(float(v))
    except (TypeError, ValueError): return 0

def _to_float(v) -> float:
    try: return float(v)
    except (TypeError, ValueError): return 0.0

def _is_dynamic(col: str) -> bool:
    """Devuelve True si la columna tiene un prefijo dinámico conocido."""
    return any(col.startswith(f'{p}_') for p in DYNAMIC_PREFIXES)
