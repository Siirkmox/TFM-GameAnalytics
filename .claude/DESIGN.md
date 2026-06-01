# DESIGN.md — Hack And Slash
> Arquitectura técnica y decisiones de diseño
> Actualizado el 2026-05-26 para reflejar el estado real del juego (v0.9)

---

## 1. Visión General

Hack & Slash 3D isométrico desarrollado en Unity. El jugador elige un elemento mágico (Fuego, Agua, Tierra, Viento) y atraviesa salas de combate lanzando hechizos contra enemigos melee y a distancia. El juego incluye 4 niveles con progresión lineal, guardado automático por sala y telemetría enviada a Firebase Firestore para análisis del TFM.

---

## 2. Arquitectura de Alto Nivel

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ESCENAS Unity                                   │
│  ┌──────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ MainMenu │──▶│ Level1  │──▶│ Level2  │──▶│ Level3  │──▶│ Level4  │     │
│  └──────────┘   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘     │
│                       │             │              │              │           │
│                  VictoryScreen (si no hay siguiente nivel)                   │
└──────────────────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        ▼                    ▼                      ▼
  ┌──────────┐        ┌───────────┐          ┌──────────┐
  │ Gameplay │        │    UI     │          │  Audio   │
  │ Systems  │        │  Systems  │          │  System  │
  └────┬─────┘        └─────┬─────┘          └──────────┘
       │                    │
  ┌────┴─────────────────────────────────┐
  │            Singletons (DontDestroy)  │
  │  GameManager | InputController       │
  │  ObjectPoolManager | AudioManager    │
  │  TelemetryManager | SaveSystem       │
  │  GameStatsTracker | FirestoreUploader│
  │  SceneLoader                         │
  └──────────────────────────────────────┘
```

---

## 3. Sistemas y Componentes Principales

### 3.1 Singletons persistentes entre escenas

| Singleton | Responsabilidad |
|---|---|
| `GameManager` | Estado global del juego, flujo de muerte (YouDied/GameOver), suscripción a eventos del jugador |
| `InputController` | Activar/desactivar grupos de input por contexto |
| `ObjectPoolManager` | Pool centralizado de proyectiles, VFX, enemigos |
| `AudioManager` | Reproducción de SFX y música, gestión de loops |
| `TelemetryManager` | Captura eventos de telemetría durante la sesión |
| `FirestoreUploader` | Sube agregados de sesión a Firebase Firestore via REST al finalizar |
| `SaveSystem` | Serialización y deserialización del progreso en JSON local |
| `GameStatsTracker` | Acumula stats de nivel y globales (kills, daño, muertes, tiempo) |
| `SceneLoader` | Transiciones de escena seguras, único punto de entrada para cambiar escenas |

**Decisión**: Todos usan el patrón `DontDestroyOnLoad` con comprobación de instancia previa. `SceneLoader` es el único punto de entrada para cambiar de escena — restaura `Time.timeScale` y notifica sistemas antes de cada transición.

### 3.2 Sistema de Salud y Mana

```
HealthSystem (Component)
├── float maxHealth, currentHealth, shield
├── float damageReduction, absorbAmount
├── bool isInvulnerable
├── Events: OnHealthChanged, OnDamageTaken, OnDamageTakenWithAttacker, OnDeath
└── Métodos: TakeDamage(), Heal(), ApplyShield()

ManaSystem (Component)
├── float maxMana, currentMana, regenRate
├── Events: OnManaChanged, OnManaUsed, OnManaInsufficient, OnManaFull
└── Métodos: UseMana(), RegenerateMana()

PlayerCombatRegen (Component)
├── Detecta si hay enemigos en rango para definir "en combate"
├── Ajusta regenRate del ManaSystem (3/s combate, 10/s fuera)
└── Pausa regeneración de vida en combate
```

### 3.3 Sistema de Magia

**Controles de hechizos:**
- Clic izquierdo → Proyectil
- Clic derecho → Blast
- E → Escudo
- Q → AOE
- R → Aura
- Espacio → Beam

```
BaseMagicCaster (Abstract MonoBehaviour)
├── 6 tipos de hechizo: Projectile, Blast, Shield, AOE, Aura, Beam
├── Gestión de cooldowns y coste de mana
├── Events: OnSpellCast, OnSpellCastWithContext (element, spellType)
└── Métodos abstractos por elemento

FireMagicCaster / WaterMagicCaster
EarthMagicCaster / WindMagicCaster
└── Heredan BaseMagicCaster, configuran VFX y stats elementales

SpellDamage (Component en prefabs de hechizo)
├── Tipos: SingleHit, Continuous, AreaPeriodic, OnEnter, Cone
├── Knockback radial / forward / pushOutOfCollider
└── Aplica StatusEffect: BurnEffect, SlowEffect, StunEffect

AuraEffect (Component)
└── Modifica stats del portador mientras está activo
    Fire: +25% daño | Water: +5 HP/s | Wind: -30% CD | Earth: -50% daño recibido

ShieldEffect (Component)
└── Comportamiento defensivo por elemento:
    Fire: quema a los atacantes
    Water: ralentiza atacantes + cura 10% daño recibido
    Earth: absorbe 50 puntos de daño (vida temporal)
    Wind: recupera 15% del daño recibido como maná
```

**Comportamiento por elemento:**

| Elemento | Mecánica principal | Hechizos | Escudo | Aura |
|---|---|---|---|---|
| Fuego | Burn (DOT 30% daño/tick, 3s) | Todos aplican Burn | Quema atacantes | +25% daño |
| Agua | Slow (50% velocidad, 2s) | Todos aplican Slow | Slow + cura 10% | +5 HP/s |
| Tierra | Stun (0.05s/punto daño, máx 3s) | Todos aplican Stun | Absorbe 50 daño | -50% daño recibido |
| Viento | Knockback (empuje radial/forward) | Projectile/Beam/Blast/AOE pushean | Regen 15% maná | -30% cooldowns |

### 3.4 Sistema de Enemigos

```
EnemyType (ScriptableObject)
├── Stats: health, speed, detectionRange, attackRange, prepareRange, cooldown
├── blockChance, xpReward, lootTable
├── EnemyClass: Barbarian1H | Barbarian2H | Knight1H | Knight2H | Rogue
│              RangerBow | RangerCrossbow | RangerCrossbow2H | Boss variants
├── MeleeAttackType: Slice | Stab | Chop | Spinning | 2Hand | Dual | Unarmed | Boss
└── RangedAttackData: Normal | Spread | Burst | Rain | Throw (TurretAttackData)

BasicEnemyAI (MonoBehaviour - Máquina de estados, enemigos melee)
├── Estados: Idle → Chase → Attack → Returning → Death
├── Detección: SphereOverlap con detectionRange
├── Separación entre enemigos: steering behavior
└── Reacciona a daño; soporta bloqueo según blockChance

RangedEnemyAI (MonoBehaviour - Máquina de estados, enemigos a distancia)
├── Estados: Idle → Chase → Prepare → Attack → Returning → Death
├── RangedAttackType enum: Normal | Spread | Burst | Rain | Throw
├── Normal: disparo único al jugador
├── Spread: varios proyectiles en abanico simultáneos
├── Burst: ráfaga de N proyectiles con cadencia rápida
├── Rain: lluvia de flechas con zonas de daño por tick (RainZone prefab)
├── Throw: deploy de torretas (solo RangerCrossbow2H)
│   ├── Lanza TurretController via animación Throw
│   ├── HideWeapon / ShowWeapon via Animation Events
│   └── Máximo de torretas simultáneas: TurretAttackData.MaxTurrets
└── Al morir destruye todas las torretas activas (DestroyOnBossDeath)

TurretController (MonoBehaviour)
├── Rota hacia el jugador en plano horizontal
├── Dispara alternando entre shootOrigin1 / shootOrigin2
├── Tiene HealthSystem propio con EnemyHealthBar
├── Rigidbody con FreezePositionY (sin NavMesh)
└── Se destruye cuando muere o cuando el boss muere

IEnemyAI (interfaz común)
└── Garantiza: TakeDamage(), Die(), IsAlive
```

**Enemigos implementados (100 instancias en 4 niveles):**

| Tipo | Clase | Notas | L1 | L2 | L3 | L4 | Total |
|---|---|---|---|---|---|---|---|
| Barbarian1Hand | BasicEnemyAI | Slash | 4 | - | - | - | 4 |
| Barbarian2Hand | BasicEnemyAI | 2Hand | 1 | 2 | 3 | 3 | 9 |
| Knight1H | BasicEnemyAI | Alto blockChance | 2 | 3 | 2 | - | 7 |
| Knight2H | BasicEnemyAI | Alto blockChance | 3 | 4 | 6 | 8 | 21 |
| Rogue | BasicEnemyAI | DualWield, alta velocidad | 1 | 8 | 7 | 8 | 24 |
| RangerBow | RangedEnemyAI | Normal + Rain | 2 | 4 | 4 | - | 10 |
| RangerCrossbow | RangedEnemyAI | Normal + Burst | 1 | 2 | 6 | 6 | 15 |
| **Bosses** | | | **2** | **2** | **3** | **3** | **10** |

**Bosses (10 instancias):**

| Boss | Clase | L1 | L2 | L3 | L4 | Total |
|---|---|---|---|---|---|---|
| Barbarian1HBoss | BasicEnemyAI | 1 | - | - | - | 1 |
| Barbarian2HBoss | BasicEnemyAI | 1 | - | 1 | 1 | 3 |
| KnightBossBlack | BasicEnemyAI | - | 1 | - | 1 | 2 |
| KnightBossGold | BasicEnemyAI | - | 1 | 1 | - | 2 |
| Rogue_Hooded | BasicEnemyAI | - | - | 1 | - | 1 |
| RangerBowBoss | RangedEnemyAI | - | - | - | 1 | 1 |

### 3.5 Level Design

```
CombatRoom (MonoBehaviour)
├── BoxCollider trigger de entrada del jugador
├── Cierra puertas → spawnea enemigos (EnemySpawnEntry[])
├── EntrySpawnPoint: posición donde aparece el jugador al continuar
├── Events: OnRoomEntered, OnRoomCompleted
├── Cuando enemigos = 0 → abre puertas → SaveSystem.SaveOnRoomCompleted()
├── MarkAsCompleted() / OpenDoorsFromSave() — restauración desde save
└── ForceAllEnemiesReturn() — si el jugador sale de la sala durante combate

Door (MonoBehaviour)
├── Estados: Open / Closed
└── Anima apertura/cierre

VictoryManager (MonoBehaviour, por escena — no DontDestroyOnLoad)
├── Detecta automáticamente todas las CombatRooms
├── Cuando todas completadas → LevelCompletedUI (stats del nivel: tiempo, kills, daño, muertes)
├── Al pulsar "Siguiente Nivel" → SceneLoader.LoadNextLevel()
└── Si no hay siguiente nivel → VictoryScreenUI (stats globales de la partida)
```

**Estructura de niveles:**

| Nivel | Total salas | Bosses |
|-------|-------------|--------|
| Level1 | 5 | Barbarian1HBoss + Barbarian2HBoss |
| Level2 | 6 | KnightBossBlack + KnightBossGold |
| Level3 | 7 | Barbarian2HBoss + KnightBossGold + Rogue_Hooded |
| Level4 | 6 | Barbarian2HBoss + KnightBossBlack + RangerBowBoss |

**Flujo de guardado por sala:**
- Al completar cada sala → auto-guardado (completedRooms + TotalStats)
- Al continuar → jugador aparece en la entrada de la primera sala incompleta
- Al morir y Retry → `ConsolidateLevelStats()` + `SaveManual()` antes de recargar
- Guardado manual desde pausa → solo TotalStats (no LevelStats parciales)

### 3.6 Sistema de Audio

```
AudioManager (Singleton)
├── AudioSource[] sfxSources (pool)
├── AudioSource musicSource (loop)
├── AudioSource ambientSource (loop)
└── Métodos: PlaySFX(clip), PlayMusic(clip), StopMusic(), FadeMusic()

Integración:
- Sonidos de pasos: Animation Events → AudioManager.PlaySFX()
- Hechizos: BaseMagicCaster.OnSpellCast → AudioManager.PlaySFX()
- Enemigos: suscripción a HealthSystem.OnDamageTaken / OnDeath
- Entorno: suscripción a Door.OnDoorOpened / CombatRoom.OnRoomCompleted
- Música: CombatRoom.OnRoomEntered / OnRoomCompleted
```

### 3.7 Progresión y Guardado

```
SaveData (clase serializable)
├── int levelNumber               // Nivel actual del jugador
├── string selectedElement        // Elemento mágico elegido
├── List<string> completedRooms   // Nombres de salas completadas
├── float totalTime               // Tiempo acumulado
├── int totalKills                // Kills acumulados
├── float totalDamageTaken        // Daño recibido acumulado
├── int totalDeaths               // Muertes totales (todos los intentos)
├── long savedAtUnixTime          // Timestamp del último guardado
└── string telemetrySessionId     // ID sesión Firestore (para Continue)

SaveSystem (Singleton, DontDestroyOnLoad)
├── Serializa SaveData a JSON en Application.persistentDataPath/savegame.json
├── SaveOnRoomCompleted(roomName) — auto-guardado al completar sala
├── SaveManual()                  — guardado manual (TotalStats + Deaths)
├── MarkCompletedRooms()          — paso 1 al cargar escena: marca salas completadas
├── OpenCompletedRoomDoors()      — paso 2 (siguiente frame): abre puertas
└── SpawnPlayerAtFirstIncompleteRoom() — teletransporta al jugador

GameStatsTracker (Singleton, DontDestroyOnLoad)
├── Stats de nivel (intento actual): LevelTime, LevelKills, LevelDamageTaken
├── Stats globales (toda la partida): TotalTime, TotalKills, TotalDamageTaken, TotalDeaths
├── ConsolidateLevelStats() — añade stats del nivel a los globales
├── ResetLevelStats()       — reinicia stats del nivel (para Retry)
└── CompleteLevel()         — consolida + marca nivel como completado
```

### 3.8 Object Pool

```
ObjectPoolManager (Singleton)
├── Dictionary<GameObject, Queue<GameObject>> pools (clave: referencia al prefab, no nombre)
├── Métodos: Get(prefab), Return(go, prefab), Prewarm(prefab, count)
└── Prewarm configurable via PoolConfig[] en Inspector

Usos:
- Proyectiles de hechizos
- Partículas VFX (impactos, explosiones)
- FloatingDamageNumber
- EnemyHealthBar
```

### 3.9 Telemetría

```
TelemetryManager (Singleton, DontDestroyOnLoad)
├── Registra eventos durante la sesión en memoria (List<TelemetryEvent>)
├── Al finalizar sesión → escribe JSON local + llama FirestoreUploader.UploadSession()
└── Contexto automático: _currentLevel y _currentRoom se actualizan por eventos de escena/sala

FirestoreUploader (Singleton, DontDestroyOnLoad)
├── Recibe TelemetrySession al final de la sesión
├── BuildAggregates(): itera eventos y construye contadores por sala/nivel
├── Sube a Firebase Firestore via REST (no SDK — compatible con WebGL)
└── Estructura en Firestore:
    sessions/{sessionId}                          → resumen de sesión
    sessions/{sessionId}/levels/{levelId}         → stats por nivel
    sessions/{sessionId}/levels/{levelId}/rooms/{roomId} → agregados por sala
```

**Eventos registrados:**

| Categoría | Evento | Datos clave |
|---|---|---|
| Sesión | SessionStart / SessionEnd | element, isVictory, totalTime |
| Nivel | LevelStart / LevelComplete / RetryLevel | level, attempt, timeSecs, kills |
| Sala | RoomEnter / RoomComplete | room, timeSecs, damageTaken |
| Jugador | PlayerDeath / DamageTaken / CheckpointActivated | room, killerType, amount, position |
| Hechizos | SpellCast / SpellHit / SpellMiss / SpellBlocked / NoMana | element, spellType, enemy |
| Efectos | StatusApplied | effect (Burn/Slow/Stun/Knockback), enemy, duration |
| Enemigos | EnemyKilled | enemyType, killedWith, position |
| Comportamiento | MovementStats / FirstSpellInRoom / ElementSelected | distancia, spellType, element |

**Agregados por sala en Firestore:**

| Campo | Descripción |
|---|---|
| `kills_{enemyType}` | Kills por tipo de enemigo |
| `killedWith_{spellType}` | Kills por tipo de hechizo |
| `damage_{enemyType}` | Daño recibido por fuente |
| `cast_{spellType}` | Hechizos lanzados por tipo |
| `blocked_{enemyType}` | Bloqueos por tipo de enemigo |
| `status_{effect}` | Efectos aplicados (Burn/Slow/Stun/Knockback) |
| `miss_{spellType}` | Hechizos fallados por tipo |
| `noMana_{spellType}` | Intentos sin maná por tipo |
| `deaths` | Muertes en la sala |
| `timeSecs` | Tiempo en la sala |
| `firstSpell` | Primer hechizo lanzado al entrar |

---

## 4. Decisiones Técnicas Clave

| Decisión | Alternativa descartada | Motivo |
|---|---|---|
| `CharacterController` para movimiento del jugador | `Rigidbody` | Mayor control sobre colisiones, sin física acumulada |
| `ScriptableObject` para tipos de enemigo | Prefabs con datos inline | Reutilización, edición en Inspector sin instanciar |
| Object Pool centralizado | `Instantiate/Destroy` en tiempo real | Evitar GC spikes en combate intenso |
| `Animation Events` para sonidos y hitboxes | Coroutines o polling | Precisión de frame exacta, sin overhead en Update |
| Firestore REST sin SDK | Firebase SDK Unity | El SDK de Firebase no es compatible con WebGL (builds para itch.io) |
| Agregados calculados en cliente antes de subir | Subir eventos individuales | Reduce escrituras a Firestore; dataset más limpio para análisis en Python |
| JSON local como buffer + upload al cerrar sesión | Upload en tiempo real por evento | Evita latencia de red en gameplay; un solo batch al finalizar |
| `DontDestroyOnLoad` + check de instancia previa | Recargar singletons por escena | Evitar duplicados al transicionar entre escenas |
| `ConsolidateLevelStats()` antes de `ResetLevelStats()` en Retry | Solo reset | Preserva kills/daño de intentos fallidos en totales de sesión |
| `SaveManual()` antes de recargar escena en Retry | Solo guardar en RoomComplete | Evita que `RestoreFromSave` revierta deaths al recargar |

---

## 5. Diagrama de Capas de Dependencias

```
┌─────────────────────┐
│   UI / Presentación  │  (HUD, GameOverUI, LevelCompletedUI, VictoryScreenUI, YouDiedUI)
└──────────┬──────────┘
           │ lee de
┌──────────▼──────────┐
│   Gameplay Systems  │  (HealthSystem, ManaSystem, BaseMagicCaster,
│                     │   BasicEnemyAI, RangedEnemyAI, CombatRoom)
└──────────┬──────────┘
           │ usa
┌──────────▼──────────┐
│   Core / Infra      │  (ObjectPoolManager, AudioManager, SaveSystem,
│                     │   TelemetryManager, FirestoreUploader,
│                     │   GameStatsTracker, InputController, SceneLoader)
└─────────────────────┘
```

**Regla**: Las capas superiores pueden depender de las inferiores. Las inferiores no deben referenciar capas superiores — solo emiten eventos.

---

## 6. Estructura de Carpetas

```
Assets/
├── Scripts/
│   ├── Core/           # Singletons: GameManager, SaveSystem, GameStatsTracker, SceneLoader, ObjectPool, InputController
│   ├── Player/         # PlayerMovement, HealthSystem, ManaSystem, PlayerCombatRegen
│   ├── Magic/          # BaseMagicCaster, *MagicCaster, SpellDamage, StatusEffects/, AuraEffect, ShieldEffect
│   ├── Enemy/          # BasicEnemyAI, RangedEnemyAI, EnemyType (SO), TurretController, WeaponHitbox
│   ├── Level/          # CombatRoom, Door, Checkpoint, PlayerSpawnPoint
│   ├── UI/             # PlayerHUD, SpellSlotsUI, GameOverUI, LevelCompletedUI, VictoryScreenUI, YouDiedUI
│   ├── Audio/          # AudioManager, UIAudioManager
│   ├── Telemetry/      # TelemetryManager, TelemetryEvent, FirestoreUploader
│   └── Editor/         # BuildVersionIncrementer (auto-incrementa bundleVersion en cada build)
├── ScriptableObjects/
│   └── Enemies/        # EnemyType assets
├── Prefabs/
│   ├── Spells/
│   ├── Enemies/
│   ├── VFX/
│   └── UI/
└── Scenes/
    ├── MainMenu.unity
    └── Levels/
        ├── Level1.unity
        ├── Level2.unity
        ├── Level3.unity
        └── Level4.unity
```
