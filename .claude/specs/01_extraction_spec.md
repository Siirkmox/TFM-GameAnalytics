# Spec: 01_extraction.ipynb — Extracción de Firestore

> **Estado: COMPLETADO** ✓  
> Este spec documenta lo que se implementó. No hay tareas pendientes.

## Design

**Objetivo:**
Extraer la jerarquía completa de Firestore (`sessions → levels → rooms`) mediante
autenticación JWT con Service Account y guardarla en `data/raw/` como JSON estructurado,
sin dependencias de `firebase-admin`.

**Entradas:**
- `credentials/firebase-service-account.json` — Service Account de Firebase (no subir a git)
- `src/firestore_client.py` — cliente HTTP autenticado con JWT propio

**Salidas:**
- `data/raw/sessions_raw.json` — jerarquía completa sin modificar
- `data/raw/extraction_metadata.json` — metadatos: fecha, conteos, versiones, plataformas

**Conexión con proyectos anteriores:**
- **P1 (ETL Pipeline):** mismo patrón de extracción → raw → procesado (medallion)
- **P4 (Fabric/ETL):** patrón de metadatos de extracción para trazabilidad

**Secciones del notebook:**
1. Configuración — paths, imports, `FirestoreClient`
2. Conexión y extracción — `client.export_all_sessions()`
3. Resumen de lo extraído — conteos, versiones, plataformas, tasa de victoria
4. Inspección de una sesión — vista detallada de estructura anidada
5. Guardar en data/raw/ — JSON + metadatos
6. Inventario de campos dinámicos — catálogo de prefijos `kills_*`, `cast_*`, etc.

---

## Requirements

**Módulo central: `src/firestore_client.py`**
- Autenticación: JWT firmado con clave RSA del Service Account (sin `firebase-admin`)
- Dependencia extra: `cryptography` (única librería no estándar)
- Métodos: `FirestoreClient(credentials_path)`, `get_project_id()`, `export_all_sessions()`
- `export_all_sessions()` recorre recursivamente `sessions → levels → rooms`

**Estructura del JSON de salida:**
```json
[
  {
    "sessionId": "...",
    "playerElement": "Fire",
    "isVictory": false,
    "totalKills": 42,
    ...
    "levels": [
      {
        "levelId": "Level1",
        "kills": 10,
        ...
        "rooms": [
          {
            "roomId": "Room1",
            "kills_Skeleton": 3,
            "cast_Beam": 5,
            ...
          }
        ]
      }
    ]
  }
]
```

**Campos dinámicos detectados en salas (prefijos):**
- `kills_*` — kills por tipo de enemigo
- `killedWith_*` — kills atribuidos a cada hechizo
- `cast_*` — lanzamientos por hechizo
- `miss_*` — fallos por hechizo
- `blocked_*` — bloqueos por enemigo
- `damage_*` — daño por tipo
- `status_*` — efectos de estado aplicados
- `noMana_*` — intentos de lanzamiento sin maná

**Librerías requeridas:**
- `cryptography` — firma JWT RSA
- `json`, `datetime`, `pathlib` — stdlib

**Outputs mínimos:**
- `sessions_raw.json` guardado en `data/raw/`
- `extraction_metadata.json` con conteos y fecha UTC
- Print de resumen: sesiones, niveles, salas, elementos, plataformas, victorias

**Restricciones:**
- No subir a git: `credentials/`, `data/raw/`
- El notebook debe ser re-ejecutable: cada ejecución sobreescribe el JSON con datos frescos
- Si `credentials/` no existe, el notebook falla con mensaje claro (no silenciosamente)

---

## Tasks (completadas)

- [x] Implementar `src/firestore_client.py` con autenticación JWT — **TFM-6, TFM-7**
- [x] Método `export_all_sessions()` con recursión `sessions → levels → rooms` — **TFM-8**
- [x] Bloque de resumen (conteos, versiones, plataformas, tasa victoria) — **TFM-9**
- [x] Inventario automático de campos dinámicos por prefijo — **TFM-9**
- [x] Guardar `sessions_raw.json` y `extraction_metadata.json` — **TFM-10**
- [x] Inspección manual de una sesión completa (debug/documentación) — **TFM-10**

**Estado de datos en última extracción:**
- Sesiones: 11 | Niveles: 18 | Salas: 82
- Elementos: Fire, Earth, Water, Wind
- Plataformas: Editor, WebGL
- Versión: 0.1 | Victorias: 1/11 (9.1%)
