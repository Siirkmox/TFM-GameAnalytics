"""
cleaning.py — Validación, detección de outliers y limpieza de DataFrames del TFM.

Extrae la lógica de limpieza de 02_preprocessing.ipynb en funciones reutilizables,
siguiendo el patrón modular de P1 del máster (src/cleaning.py).

Funciones principales:
  - flag_suspicious_sessions  : marca sesiones problemáticas con is_suspicious
  - get_cleaning_report       : resumen de calidad del dataset
  - validate_sessions         : comprueba integridad mínima del DataFrame
  - filter_clean              : filtra sesiones no sospechosas para análisis robustos
"""

import pandas as pd
import numpy as np


# ─── Umbrales de limpieza ────────────────────────────────────────────────────

# Sesiones más cortas que esto probablemente son pruebas o errores
MIN_TIME_SECS = 30

# Sesiones más largas que esto son outliers claros (AFK, juego pausado, etc.)
MAX_TIME_SECS = 3600

# Sesiones sin kills ni niveles completados — posiblemente test o crash inmediato
MIN_KILLS_FOR_valid = 0  # se usa en combinación con levelsCompleted


# ─── Funciones principales ───────────────────────────────────────────────────

def flag_suspicious_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade (o recalcula) la columna `is_suspicious` en df_sessions.

    Criterios:
      - totalTimeSecs < MIN_TIME_SECS  → sesión muy corta (probable prueba)
      - totalTimeSecs > MAX_TIME_SECS  → sesión muy larga (probable AFK/outlier)
      - totalKills == 0 AND levelsCompleted == 0  → sin actividad real

    Con pocos datos no se descartan sesiones — solo se marcan para
    poder filtrarlas en análisis específicos que requieran datos limpios.

    Returns:
        DataFrame con columna `is_suspicious` actualizada.
    """
    df = df.copy()

    mask_short   = df['totalTimeSecs'] < MIN_TIME_SECS
    mask_long    = df['totalTimeSecs'] > MAX_TIME_SECS
    mask_no_activity = (df['totalKills'] == 0) & (df['levelsCompleted'] == 0)

    df['is_suspicious'] = mask_short | mask_long | mask_no_activity

    return df


def get_cleaning_report(df_sessions: pd.DataFrame,
                        df_levels: pd.DataFrame,
                        df_rooms: pd.DataFrame) -> dict:
    """
    Genera un resumen de calidad del dataset para documentar en el informe.

    Returns:
        Dict con métricas de calidad: totales, sospechosas, nulos, duplicados, etc.
    """
    n_sessions = len(df_sessions)
    n_suspicious = df_sessions['is_suspicious'].sum() if 'is_suspicious' in df_sessions.columns else 0

    # Nulos por DataFrame
    nulls_sessions = df_sessions.isnull().sum()
    nulls_levels   = df_levels.isnull().sum()
    nulls_rooms    = df_rooms[['timeSecs', 'deaths', 'damageTaken', 'firstSpell']].isnull().sum()

    # Duplicados
    dup_sessions = df_sessions.duplicated(subset='sessionId').sum()
    dup_rooms    = df_rooms.duplicated(subset=['sessionId', 'levelId', 'roomId']).sum()

    # Niveles incompletos
    n_incomplete = df_levels['incomplete'].sum() if 'incomplete' in df_levels.columns else 0

    # Sesiones sin comentario
    n_no_comment = (~df_sessions['hasComment']).sum() if 'hasComment' in df_sessions.columns else None

    report = {
        'total_sessions'      : n_sessions,
        'total_levels'        : len(df_levels),
        'total_rooms'         : len(df_rooms),
        'suspicious_sessions' : int(n_suspicious),
        'clean_sessions'      : int(n_sessions - n_suspicious),
        'duplicate_sessions'  : int(dup_sessions),
        'duplicate_rooms'     : int(dup_rooms),
        'incomplete_levels'   : int(n_incomplete),
        'sessions_no_comment' : int(n_no_comment) if n_no_comment is not None else None,
        'nulls_sessions'      : nulls_sessions[nulls_sessions > 0].to_dict(),
        'nulls_levels'        : nulls_levels[nulls_levels > 0].to_dict(),
        'nulls_rooms'         : nulls_rooms[nulls_rooms > 0].to_dict(),
        'time_min_secs'       : float(df_sessions['totalTimeSecs'].min()),
        'time_max_secs'       : float(df_sessions['totalTimeSecs'].max()),
        'time_mean_secs'      : float(df_sessions['totalTimeSecs'].mean()),
    }

    return report


def print_cleaning_report(report: dict) -> None:
    """Imprime el informe de calidad de forma legible."""
    print('=== Informe de calidad del dataset ===')
    print(f'  Sesiones totales      : {report["total_sessions"]}')
    print(f'  Niveles totales       : {report["total_levels"]}')
    print(f'  Salas totales         : {report["total_rooms"]}')
    print(f'  Sesiones sospechosas  : {report["suspicious_sessions"]}')
    print(f'  Sesiones limpias      : {report["clean_sessions"]}')
    print(f'  Niveles incompletos   : {report["incomplete_levels"]}')
    print(f'  Duplicados sesiones   : {report["duplicate_sessions"]}')
    print(f'  Duplicados salas      : {report["duplicate_rooms"]}')
    print(f'  Sin comentario        : {report["sessions_no_comment"]}')
    print(f'  Tiempo min/max/media  : {report["time_min_secs"]:.0f}s / '
          f'{report["time_max_secs"]:.0f}s / {report["time_mean_secs"]:.0f}s')

    if report['nulls_sessions']:
        print(f'  Nulos sesiones        : {report["nulls_sessions"]}')
    if report['nulls_levels']:
        print(f'  Nulos niveles         : {report["nulls_levels"]}')
    if report['nulls_rooms']:
        print(f'  Nulos salas           : {report["nulls_rooms"]}')


def validate_sessions(df: pd.DataFrame) -> list[str]:
    """
    Comprueba que df_sessions tiene las columnas mínimas necesarias para el análisis.

    Returns:
        Lista de errores encontrados (vacía si todo está bien).
    """
    required = [
        'sessionId', 'playerElement', 'isVictory', 'totalTimeSecs',
        'totalDeaths', 'totalKills', 'levelsCompleted', 'platform', 'gameVersion',
    ]
    errors = []

    for col in required:
        if col not in df.columns:
            errors.append(f'Columna requerida ausente: {col}')

    if 'sessionId' in df.columns and df['sessionId'].duplicated().any():
        errors.append('sessionId tiene duplicados')

    if 'isVictory' in df.columns and not df['isVictory'].isin([True, False]).all():
        errors.append('isVictory contiene valores no booleanos')

    if 'totalTimeSecs' in df.columns and (df['totalTimeSecs'] < 0).any():
        errors.append('totalTimeSecs tiene valores negativos')

    return errors


def filter_clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Devuelve solo las sesiones no sospechosas.
    Útil para análisis estadísticos y ML que requieren datos de calidad.

    Args:
        verbose: si True, imprime cuántas sesiones se filtran.
    """
    if 'is_suspicious' not in df.columns:
        df = flag_suspicious_sessions(df)

    mask = ~df['is_suspicious']
    if verbose:
        print(f'filter_clean: {mask.sum()} sesiones limpias de {len(df)} totales '
              f'({df["is_suspicious"].sum()} sospechosas eliminadas)')
    return df[mask].reset_index(drop=True)


def check_sample_size(df: pd.DataFrame,
                      group_col: str = 'playerElement',
                      min_per_group: int = 30) -> pd.DataFrame:
    """
    Comprueba si hay suficientes muestras por grupo para tests estadísticos válidos.
    Imprime una advertencia si algún grupo está por debajo del mínimo recomendado.

    Args:
        group_col   : columna por la que agrupar (por defecto playerElement)
        min_per_group: mínimo recomendado para ANOVA/Kruskal-Wallis (por defecto 30)

    Returns:
        DataFrame con conteo por grupo y flag `sufficient`.
    """
    counts = df.groupby(group_col).size().reset_index(name='n_sessions')
    counts['sufficient'] = counts['n_sessions'] >= min_per_group

    if not counts['sufficient'].all():
        insuficientes = counts[~counts['sufficient']][group_col].tolist()
        print(f'ADVERTENCIA: grupos con menos de {min_per_group} sesiones '
              f'(tests estadisticos orientativos): {insuficientes}')
    else:
        print(f'OK: todos los grupos tienen >= {min_per_group} sesiones.')

    return counts
