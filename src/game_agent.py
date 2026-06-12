"""
game_agent.py — Agente LangGraph para análisis de balance del juego Arcane Descent.
Patrón basado en agents/graph.py del Proyecto 7 (IAGenerativa).
"""
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from gemini_client import get_llm_for_agent
from game_tools import ALL_TOOLS


def _stats_actuales() -> dict:
    """
    Lee el tamano de muestra y la distribucion por elemento desde el parquet
    para que el system prompt nunca quede desfasado al anadir mas sesiones.
    Devuelve un dict con: n_total, elementos_suficientes (n>=10) y elementos_insuficientes (n<10).
    """
    try:
        repo_root = Path(__file__).resolve().parent.parent
        parquet_path = repo_root / "data" / "processed" / "sessions.parquet"
        df = pd.read_parquet(parquet_path)
        if "is_suspicious" in df.columns:
            df = df[~df["is_suspicious"]]
        n_total = len(df)
        if "playerElement" in df.columns:
            counts = df.groupby("playerElement")["sessionId"].nunique().to_dict()
        else:
            counts = {}
        suficientes = [e for e, n in counts.items() if n >= 10]
        insuficientes = [(e, n) for e, n in counts.items() if n < 10]
        return {
            "n_total": n_total,
            "suficientes": suficientes,
            "insuficientes": insuficientes,
        }
    except Exception:
        # Si algo falla, no rompemos el agente — devolvemos None y el prompt usa fallback genérico.
        return {"n_total": None, "suficientes": [], "insuficientes": []}


def _construir_system_prompt() -> str:
    """Construye el system prompt con el tamano de muestra dinamico."""
    stats = _stats_actuales()
    n_total = stats["n_total"]
    suf = stats["suficientes"]
    insuf = stats["insuficientes"]

    # Linea de limitaciones dinamica
    if n_total is not None and n_total > 0:
        linea_n = f"- Dataset actual: n={n_total} sesiones limpias — los resultados son orientativos."
        if suf and insuf:
            insuf_str = ", ".join(f"{e} (n={n})" for e, n in insuf)
            suf_str = ", ".join(suf)
            linea_suf = f"- Elementos con muestra suficiente (n>=10): {suf_str}. Elementos con muestra insuficiente: {insuf_str}."
        elif suf and not insuf:
            linea_suf = f"- Todos los elementos tienen muestra suficiente (n>=10): {', '.join(suf)}."
        elif insuf and not suf:
            insuf_str = ", ".join(f"{e} (n={n})" for e, n in insuf)
            linea_suf = f"- Ningun elemento alcanza n>=10. Detalle: {insuf_str}."
        else:
            linea_suf = "- No se ha podido determinar la distribucion por elemento."
    else:
        linea_n = "- Dataset pequeno — los resultados son orientativos."
        linea_suf = "- Consulta consultar_kpis_por_elemento para conocer el tamano de muestra por elemento."

    return f"""Eres un experto en análisis de balance de videojuegos especializado en el juego Arcane Descent, un Hack & Slash con 4 elementos jugables: Fire, Water, Earth y Wind.

## TU ROL
Analizas datos reales de sesiones de juego para responder preguntas sobre balance, dificultad, rendimiento de elementos y hechizos, y proponer recomendaciones accionables para el equipo de desarrollo Unity.

## DATOS DISPONIBLES
Tienes acceso a herramientas que consultan:
- Sesiones de juego (win rate, kills, muertes, tiempo, KPIs por elemento)
- Salas y dificultad (score de dificultad, daño, tiempo por kill)
- Puntuación Global por elemento (métrica compuesta de rendimiento: 50% victoria + 30% eficiencia + 20% sentimiento)
- Recomendaciones de balance pendientes
- Estadísticas de hechizos (uso, eficiencia, noMana)
- Estadísticas de enemigos (kills, daño infligido, resistencia)
- Funnel de progresión por nivel
- Tests estadísticos (p-values, effect sizes)

## CÓMO RESPONDER
1. Usa SIEMPRE las herramientas para obtener datos actuales antes de responder — nunca inventes cifras.
2. Cita los datos concretos en tu respuesta (win rates, scores, p-values).
3. Distingue entre hallazgos estadísticamente significativos y tendencias orientativas (n<10 sesiones).
4. Cuando identifiques un problema, propón una solución concreta para Unity.
5. Si la pregunta es muy amplia, estructura la respuesta con secciones claras.

## LIMITACIONES A MENCIONAR
{linea_n}
{linea_suf}
- Los bugs de telemetría ya corregidos en Unity afectan a sesiones históricas.

## TONO
Técnico pero claro. Directo. Usa bullets y negritas para facilitar la lectura.
Responde siempre en español.
"""


SYSTEM_PROMPT = _construir_system_prompt()

# ── Estado del grafo ──────────────────────────────────────────────────────────
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# ── Nodos del grafo ───────────────────────────────────────────────────────────
def _nodo_agente(state: AgentState) -> dict:
    """Invoca el LLM con el historial de mensajes y las tools disponibles."""
    llm = get_llm_for_agent().bind_tools(ALL_TOOLS)
    # Inyectar system prompt al inicio si no está ya
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    respuesta = llm.invoke(messages)
    return {"messages": [respuesta]}


def _decidir_siguiente(state: AgentState) -> str:
    """Decide si continuar con tools o terminar."""
    ultimo = state["messages"][-1]
    if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
        return "tools"
    return END


# ── Construcción del grafo ────────────────────────────────────────────────────
def _construir_grafo():
    tool_node = ToolNode(ALL_TOOLS)
    grafo = StateGraph(AgentState)
    grafo.add_node("agente", _nodo_agente)
    grafo.add_node("tools", tool_node)
    grafo.set_entry_point("agente")
    grafo.add_conditional_edges("agente", _decidir_siguiente, {"tools": "tools", END: END})
    grafo.add_edge("tools", "agente")
    return grafo.compile()


# Compilar el grafo una sola vez al importar el módulo
_grafo = None


def get_grafo():
    global _grafo
    if _grafo is None:
        _grafo = _construir_grafo()
    return _grafo


def invocar_agente(pregunta: str, historial: list[BaseMessage] | None = None) -> tuple[str, list[BaseMessage]]:
    """
    Invoca el agente con una pregunta y un historial opcional de mensajes previos.
    Devuelve (respuesta_texto, historial_actualizado).
    """
    grafo = get_grafo()
    mensajes_entrada = (historial or []) + [HumanMessage(content=pregunta)]
    resultado = grafo.invoke({"messages": mensajes_entrada})
    historial_nuevo = resultado["messages"]
    # Extraer texto de la última respuesta del agente
    ultimo = historial_nuevo[-1]
    if hasattr(ultimo, "content"):
        c = ultimo.content
        if isinstance(c, str):
            respuesta = c
        elif isinstance(c, list):
            respuesta = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
        else:
            respuesta = str(c)
    else:
        respuesta = str(ultimo)
    return respuesta, historial_nuevo
