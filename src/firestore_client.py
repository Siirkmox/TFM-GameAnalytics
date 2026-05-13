"""
firestore_client.py — Extracción de datos de Firestore via REST + autenticación JWT.

No usa firebase-admin: genera un JWT con el service account y lo intercambia
por un access token de Google OAuth 2.0 para llamar a la API REST de Firestore.
Esto evita instalar dependencias pesadas y funciona en cualquier entorno Python 3.8+
con cryptography instalado.

Jerarquía en Firestore:
  sessions/{sessionId}                              → resumen de sesión
  sessions/{sessionId}/levels/{levelId}             → stats por nivel
  sessions/{sessionId}/levels/{levelId}/rooms/{roomId} → agregados por sala
"""

import json
import time
import base64
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


# ─── Constantes ──────────────────────────────────────────────────────────────

FIRESTORE_SCOPE = "https://www.googleapis.com/auth/datastore"
TOKEN_URL       = "https://oauth2.googleapis.com/token"
GRANT_TYPE      = "urn:ietf:params:oauth:grant-type:jwt-bearer"


# ─── Cliente ─────────────────────────────────────────────────────────────────

class FirestoreClient:
    """
    Cliente ligero para leer documentos de Firestore con autenticación de service account.
    El access token se cachea internamente y se renueva cuando caduca.
    """

    def __init__(self, service_account_path: str):
        """
        Carga el service account y prepara el cliente.

        Args:
            service_account_path: ruta al JSON del service account de Firebase.
        """
        path = Path(service_account_path)
        if not path.exists():
            raise FileNotFoundError(f"Service account no encontrado: {path}")

        with open(path) as f:
            self._sa = json.load(f)

        self._project_id  = self._sa["project_id"]
        self._base_url    = (
            f"https://firestore.googleapis.com/v1/projects/{self._project_id}"
            f"/databases/(default)/documents"
        )
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0.0

    # ─── API pública ─────────────────────────────────────────────────────────

    def export_all_sessions(self) -> list[dict]:
        """
        Exporta la jerarquía completa de Firestore:
        sesiones → niveles → salas.

        Returns:
            Lista de dicts planos, uno por sesión, con las subcolecciones
            anidadas bajo las claves 'levels' (lista de dicts con clave 'rooms').
        """
        sessions = self._list_all(f"{self._base_url}/sessions")
        result   = []

        for doc in sessions:
            session = self._parse_document(doc)
            sid     = session["sessionId"]

            levels_raw = self._list_all(f"{self._base_url}/sessions/{sid}/levels")
            levels     = []

            for level_doc in levels_raw:
                level    = self._parse_document(level_doc)
                lid      = level_doc["name"].split("/")[-1]
                level["levelId"] = lid

                rooms_raw = self._list_all(
                    f"{self._base_url}/sessions/{sid}/levels/{lid}/rooms"
                )
                rooms = []
                for room_doc in rooms_raw:
                    room = self._parse_document(room_doc)
                    room["roomId"] = room_doc["name"].split("/")[-1]
                    rooms.append(room)

                level["rooms"] = rooms
                levels.append(level)

            session["levels"] = levels
            result.append(session)
            print(f"  [{sid}] {len(levels)} niveles, "
                  f"{sum(len(l['rooms']) for l in levels)} salas")

        return result

    def get_project_id(self) -> str:
        return self._project_id

    # ─── Autenticación ───────────────────────────────────────────────────────

    def _get_access_token(self) -> str:
        """Devuelve un access token válido, renovándolo si ha caducado."""
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        jwt    = self._make_jwt()
        body   = f"grant_type={urllib.parse.quote(GRANT_TYPE)}&assertion={jwt}".encode()
        req    = urllib.request.Request(
            TOKEN_URL, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())

        self._access_token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600)
        return self._access_token

    def _make_jwt(self) -> str:
        """Genera un JWT firmado con la clave privada del service account."""
        now = int(time.time())
        header  = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
        payload = _b64(json.dumps({
            "iss": self._sa["client_email"],
            "sub": self._sa["client_email"],
            "aud": TOKEN_URL,
            "iat": now,
            "exp": now + 3600,
            "scope": FIRESTORE_SCOPE,
        }).encode())

        msg = header + b"." + payload
        key = serialization.load_pem_private_key(
            self._sa["private_key"].encode(),
            password=None,
            backend=default_backend(),
        )
        sig = _b64(key.sign(msg, padding.PKCS1v15(), hashes.SHA256()))
        return (msg + b"." + sig).decode()

    # ─── HTTP ────────────────────────────────────────────────────────────────

    def _get(self, url: str) -> dict:
        """GET autenticado a Firestore REST."""
        token = self._get_access_token()
        req   = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def _list_all(self, collection_url: str, page_size: int = 300) -> list[dict]:
        """
        Pagina automáticamente una colección de Firestore hasta obtener todos
        los documentos, respetando el límite de pageSize por petición.
        """
        documents  = []
        next_token = None

        while True:
            url  = f"{collection_url}?pageSize={page_size}"
            if next_token:
                url += f"&pageToken={next_token}"
            data = self._get(url)
            documents.extend(data.get("documents", []))
            next_token = data.get("nextPageToken")
            if not next_token:
                break

        return documents

    # ─── Parsing ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_document(doc: dict) -> dict:
        """
        Convierte un documento Firestore (con tipos envueltos como {"stringValue": "x"})
        en un dict Python plano con valores nativos.
        """
        result = {}
        for key, typed_val in doc.get("fields", {}).items():
            result[key] = _unwrap(typed_val)
        return result


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _b64(data: bytes) -> bytes:
    """Base64 URL-safe sin padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def _unwrap(typed_val: dict):
    """
    Convierte el valor tipado de Firestore a Python nativo.
    Tipos soportados: stringValue, integerValue, doubleValue, booleanValue, nullValue.
    """
    if "stringValue"  in typed_val: return typed_val["stringValue"]
    if "integerValue" in typed_val: return int(typed_val["integerValue"])
    if "doubleValue"  in typed_val: return float(typed_val["doubleValue"])
    if "booleanValue" in typed_val: return typed_val["booleanValue"]
    if "nullValue"    in typed_val: return None
    # Tipo desconocido — devolver el primer valor sin convertir
    return list(typed_val.values())[0]


# Importar urllib.parse que se usa en _get_access_token
import urllib.parse
