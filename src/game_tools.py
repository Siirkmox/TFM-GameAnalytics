"""
game_tools.py — Herramientas de consulta de datos para el agente de game analytics.
Cada tool lee los parquets/CSVs del proyecto y devuelve un resumen textual
que el agente puede incluir en su respuesta.
"""
import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

BASE    = Path(__file__).parent.parent
PROC    = BASE / "data" / "processed"
EXPORTS = BASE / "data" / "exports"


def _sessions() -> pd.DataFrame:
    s = pd.read_parquet(PROC / "sessions.parquet")
    return s[~s["is_suspicious"]]


def _rooms() -> pd.DataFrame:
    clean_ids = _sessions()["sessionId"]
    r = pd.read_parquet(PROC / "rooms.parquet")
    return r[r["sessionId"].isin(clean_ids)]


def _levels() -> pd.DataFrame:
    clean_ids = _sessions()["sessionId"]
    l = pd.read_parquet(PROC / "levels.parquet")
    return l[l["sessionId"].isin(clean_ids)]


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def consultar_resumen_global() -> str:
    """
    Devuelve un resumen global del dataset: número de sesiones, victorias,
    win rate, sesiones por elemento y niveles completados.
    Úsala cuando necesites contexto general del juego o del dataset.
    """
    s = _sessions()
    resumen = {
        "sesiones_totales": len(s),
        "victorias": int(s["isVictory"].sum()),
        "win_rate": f"{s['isVictory'].mean():.1%}",
        "sesiones_por_elemento": s["playerElement"].value_counts().to_dict(),
        "levels_completados_media": round(s["levelsCompleted"].mean(), 2),
        "tiempo_medio_sesion_min": round(s["totalTimeSecs"].mean() / 60, 1),
    }
    return json.dumps(resumen, ensure_ascii=False, indent=2)


@tool
def consultar_kpis_por_elemento(elemento: str = "todos") -> str:
    """
    Devuelve KPIs de rendimiento por elemento del jugador: win rate, kills/min,
    kd_ratio, cast_accuracy, muertes medias y tasa de abandono.
    elemento: 'Fire', 'Water', 'Earth', 'Wind' o 'todos'.
    """
    s = _sessions()
    if elemento.lower() != "todos":
        s = s[s["playerElement"].str.lower() == elemento.lower()]
        if s.empty:
            return f"No hay sesiones limpias para el elemento '{elemento}'."

    grp = s.groupby("playerElement")
    resultado = {}
    for elem, df in grp:
        resultado[elem] = {
            "n_sesiones": len(df),
            "win_rate": f"{df['isVictory'].mean():.1%}",
            "kills_per_min_media": round(df["kills_per_min"].mean(), 3) if "kills_per_min" in df.columns else None,
            "kd_ratio_media": round(df["kd_ratio"].mean(), 2) if "kd_ratio" in df.columns else None,
            "cast_accuracy_media": f"{df['cast_accuracy'].mean():.1%}" if "cast_accuracy" in df.columns else None,
            "muertes_media": round(df["totalDeaths"].mean(), 2) if "totalDeaths" in df.columns else None,
            "abandono_nivel0": f"{(df['levelsCompleted'] == 0).mean():.1%}" if "levelsCompleted" in df.columns else None,
        }
    return json.dumps(resultado, ensure_ascii=False, indent=2)


@tool
def consultar_dificultad_salas(top_n: int = 10) -> str:
    """
    Devuelve las salas más difíciles del juego ordenadas por score de dificultad.
    El score combina muertes, daño recibido, tiempo por kill y tiempo en sala (máx 4.0).
    top_n: número de salas a devolver (por defecto 10).
    """
    r = _rooms()
    agg = r.groupby(["levelId", "roomId"]).agg(
        muertes=("deaths", "mean"),
        daño=("damageTaken", "mean"),
        tiempo=("timeSecs", "mean"),
        kills=("total_kills_room", "mean"),
        sesiones=("sessionId", "count"),
    ).reset_index()
    agg["tiempo_por_kill"] = agg.apply(
        lambda x: x["tiempo"] / x["kills"] if x["kills"] > 0 else x["tiempo"], axis=1
    )
    norm = lambda s: s / (s.max() + 1e-9)
    agg["score"] = (norm(agg["muertes"]) + norm(agg["daño"]) +
                    norm(agg["tiempo_por_kill"]) + norm(agg["tiempo"]))
    top = agg.sort_values("score", ascending=False).head(top_n)
    top["sala"] = top["levelId"] + "/" + top["roomId"]
    resultado = top[["sala", "score", "muertes", "daño", "tiempo_por_kill", "tiempo", "sesiones"]].round(2).to_dict(orient="records")
    return json.dumps(resultado, ensure_ascii=False, indent=2)


@tool
def consultar_balance_index() -> str:
    """
    Devuelve el Balance Index (puntuacion_global) de cada elemento.
    Combina win_rate (50%), kills_efficiency (30%) y sentiment_score (20%).
    Incluye si los datos son suficientes (n>=10 sesiones).
    """
    bi = pd.read_csv(EXPORTS / "balance_index.csv")
    return bi.to_json(orient="records", force_ascii=False, indent=2)


@tool
def consultar_recomendaciones(solo_pendientes: bool = True) -> str:
    """
    Devuelve las recomendaciones de balance del juego.
    solo_pendientes=True devuelve solo las no arregladas.
    solo_pendientes=False devuelve todas incluyendo las ya arregladas.
    """
    recs = pd.read_csv(EXPORTS / "balance_recommendations.csv")
    if solo_pendientes and "arreglado" in recs.columns:
        recs = recs[~recs["arreglado"].astype(bool)]
    cols = ["element", "area", "priority", "finding", "recommendation", "evidence"]
    cols = [c for c in cols if c in recs.columns]
    return recs[cols].to_json(orient="records", force_ascii=False, indent=2)


@tool
def consultar_hechizos() -> str:
    """
    Devuelve estadísticas de uso y eficiencia de cada hechizo (Beam, Projectile, Blast, AOE):
    total lanzamientos, kills, eficiencia kills/cast y eventos noMana.
    """
    r = _rooms()
    kmap = {
        "Projectile": ["killedWith_Projectile"],
        "Blast":      ["killedWith_Blast"],
        "Beam":       ["killedWith_Beam"],
        "AOE":        ["killedWith_AOE"],
    }
    resultado = []
    for spell, kcols in kmap.items():
        valid = [c for c in kcols if c in r.columns]
        kills = int(r[valid].sum().sum()) if valid else 0
        casts = int(r[f"cast_{spell}"].sum()) if f"cast_{spell}" in r.columns else 0
        nm_col = f"noMana_{spell}"
        nomana = int(r[nm_col].sum()) if nm_col in r.columns else 0
        resultado.append({
            "hechizo": spell,
            "lanzamientos": casts,
            "kills": kills,
            "eficiencia_kills_per_cast": round(kills / casts, 3) if casts > 0 else 0,
            "noMana": nomana,
        })
    return json.dumps(resultado, ensure_ascii=False, indent=2)


@tool
def consultar_enemigos() -> str:
    """
    Devuelve estadísticas de cada tipo de enemigo: kills totales, daño infligido
    al jugador y ratio blocked/kills (resistencia).
    """
    r = _rooms()
    enemy_names = [
        "Barbarian1Hand", "RangerBow", "Knight1H", "Knight2H", "Rogue",
        "Barbarian2Hand", "RangerCrossbow", "Barbarian2HBoss", "Barbarian1HBoss",
        "KnightBossBlack", "KnightBossGold", "Rogue_Hooded", "RangerBowBoss",
    ]
    resultado = []
    for en in enemy_names:
        kills   = int(r[f"kills_{en}"].sum())   if f"kills_{en}"   in r.columns else 0
        damage  = int(r[f"damage_{en}"].sum())  if f"damage_{en}"  in r.columns else 0
        blocked = int(r[f"blocked_{en}"].sum()) if f"blocked_{en}" in r.columns else 0
        if kills == 0 and damage == 0:
            continue
        resultado.append({
            "enemigo": en,
            "kills_totales": kills,
            "daño_infligido_al_jugador": damage,
            "blocked": blocked,
            "ratio_blocked_kills": round(blocked / kills, 2) if kills > 0 else None,
        })
    resultado.sort(key=lambda x: x["daño_infligido_al_jugador"], reverse=True)
    return json.dumps(resultado, ensure_ascii=False, indent=2)


@tool
def consultar_funnel_niveles() -> str:
    """
    Devuelve el funnel de progresión: cuántas sesiones llegan a cada nivel
    y qué porcentaje no pasa al siguiente (cuello de botella).
    """
    s = _sessions()
    l = _levels()
    funnel = l.groupby("levelId")["sessionId"].nunique().reset_index()
    funnel.columns = ["nivel", "sesiones"]
    funnel = funnel.sort_values("nivel").reset_index(drop=True)
    funnel["pct_llegada"] = (funnel["sesiones"] / len(s) * 100).round(1)
    funnel["caida_pct"] = (funnel["pct_llegada"] - funnel["pct_llegada"].shift(-1).fillna(funnel["pct_llegada"])).round(1)
    return funnel.to_json(orient="records", force_ascii=False, indent=2)


@tool
def consultar_estadisticas_tests() -> str:
    """
    Devuelve los resultados de los tests estadísticos (Kruskal-Wallis, Spearman,
    Chi-squared) realizados en la Fase 3. Incluye p-values y effect sizes.
    Útil para responder preguntas sobre significancia estadística.
    """
    stat = pd.read_csv(EXPORTS / "statistical_results.csv")
    # Devolver solo los más relevantes (p<0.1) para no saturar el contexto
    relevantes = stat[stat["p_value"] < 0.1].sort_values("p_value")
    if relevantes.empty:
        return stat.to_json(orient="records", force_ascii=False, indent=2)
    return relevantes.to_json(orient="records", force_ascii=False, indent=2)


# Lista de todas las tools para importar en el agente
ALL_TOOLS = [
    consultar_resumen_global,
    consultar_kpis_por_elemento,
    consultar_dificultad_salas,
    consultar_balance_index,
    consultar_recomendaciones,
    consultar_hechizos,
    consultar_enemigos,
    consultar_funnel_niveles,
    consultar_estadisticas_tests,
]
