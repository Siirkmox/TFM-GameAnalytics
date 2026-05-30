"""
gemini_client.py — Cliente Gemini con retry RPM y fallback modelo+clave.
Patrón idéntico a llm_service.py del Proyecto 7 (IAGenerativa).
"""
import os
import re
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ── API Keys (carga automática igual que Proyecto 7) ─────────────────────────
_base = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEYS = [_base] if _base else []
_i = 2
while True:
    _k = os.getenv(f"GOOGLE_API_KEY_{_i}")
    if not _k:
        break
    GOOGLE_API_KEYS.append(_k)
    _i += 1

# ── Modelos en orden de preferencia (mismos que Proyecto 7) ──────────────────
MODELOS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
]

MAX_RETRIES_RPM = 5

# ── Estado de fallback ────────────────────────────────────────────────────────
_estado = {"modelo_idx": 0, "clave_idx": 0}


def _crear_llm(modelo: str, clave_idx: int) -> ChatGoogleGenerativeAI:
    if not GOOGLE_API_KEYS:
        raise RuntimeError("No hay ninguna GOOGLE_API_KEY configurada en .env")
    return ChatGoogleGenerativeAI(
        model=modelo,
        google_api_key=GOOGLE_API_KEYS[clave_idx],
        temperature=0.3,
    )


def _extraer_texto(msg) -> str:
    """Extrae texto de un mensaje cuyo .content puede ser str o list (Gemini 2.5+)."""
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        partes = [p.get("text", "") if isinstance(p, dict) else str(p) for p in c]
        return " ".join(partes)
    return str(c)


def _extraer_retry_delay(error_str: str, default: float = 15.0) -> float:
    match = re.search(r"retryDelay.*?(\d+(?:\.\d+)?)\s*s", error_str)
    return float(match.group(1)) + 1 if match else default


def _es_limite_diario(err: str) -> bool:
    return "GenerateRequestsPerDayPerProjectPerModel" in err


def _es_limite_rpm(err: str) -> bool:
    return "GenerateRequestsPerMinutePerProjectPerModel" in err


def _siguiente_combinacion() -> bool:
    """Avanza a la siguiente combinación modelo+clave. Devuelve False si se agotaron todas."""
    # Primero rotar clave dentro del mismo modelo
    if _estado["clave_idx"] + 1 < len(GOOGLE_API_KEYS):
        _estado["clave_idx"] += 1
        return True
    # Si se agotaron las claves, pasar al siguiente modelo y resetear claves
    if _estado["modelo_idx"] + 1 < len(MODELOS):
        _estado["modelo_idx"] += 1
        _estado["clave_idx"] = 0
        return True
    return False


def _llm_actual() -> ChatGoogleGenerativeAI:
    return _crear_llm(MODELOS[_estado["modelo_idx"]], _estado["clave_idx"])


def invoke(prompt: str, system: str | None = None) -> str:
    """
    Invoca Gemini con el prompt dado.
    Gestiona automáticamente límites RPM y fallback modelo+clave.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    llm = _llm_actual()
    for _ in range(MAX_RETRIES_RPM + len(GOOGLE_API_KEYS) * len(MODELOS)):
        try:
            resp = llm.invoke(messages)
            return _extraer_texto(resp)
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" not in err and "429" not in err:
                raise
            if _es_limite_rpm(err):
                delay = _extraer_retry_delay(err)
                time.sleep(delay)
                continue
            if _es_limite_diario(err):
                if not _siguiente_combinacion():
                    raise RuntimeError("Todas las combinaciones modelo+clave están agotadas.") from e
                llm = _llm_actual()
                continue
            raise

    raise RuntimeError("Se agotaron los reintentos de la API Gemini.")


def get_llm_for_agent() -> ChatGoogleGenerativeAI:
    """Devuelve el LLM configurado para usar como base del agente LangGraph."""
    return _crear_llm(MODELOS[_estado["modelo_idx"]], _estado["clave_idx"])
