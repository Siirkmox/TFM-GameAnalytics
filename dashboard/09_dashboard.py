"""
09_dashboard.py — Dashboard TFM: Análisis de Balance Hack & Slash
Ejecución local: python -m streamlit run dashboard/09_dashboard.py
Deploy: Streamlit Community Cloud apunta a dashboard/09_dashboard.py
"""

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# El dashboard vive en dashboard/ pero importa de ../src y lee de ../data y ../models.
# REPO_ROOT apunta a la raíz del repo independientemente de dónde se lance Streamlit.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

# ── Configuración ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arcane Descent | Game Balance Analytics",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _inject_css():
    st.markdown("""
<style>
/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 24px; }
.stTabs [data-baseweb="tab"]      { padding-left: 8px; padding-right: 8px; }

/* ── Card base ── */
.card {
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border-left: 4px solid #1a8fff;
    background: #f8f9fa;
}
.card-sm {
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 4px;
    background: #f8f9fa;
}


/* ── Card typography ── */
.card-title {
    font-size: 0.85em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #1a1a1a;
}
.card-value {
    font-size: 1.15em;
    font-weight: 700;
    color: #2c3e50;
}
.card-obs {
    font-size: 0.82em;
    color: #666;
}
.card-label {
    font-size: 0.95em;
    font-weight: 700;
    color: #1a1a1a;
}

/* ── Section spacing ── */
.section-gap { margin-top: 24px; }

/* ── Checkboxes sidebar ── */
[data-testid="stSidebar"] input[type="checkbox"]:checked + div {
    background-color: #1a8fff !important;
    border-color: #1a8fff !important;
}
[data-testid="stSidebar"] input[type="checkbox"]:checked + div svg {
    fill: white !important;
}

/* ── Sidebar logo sin borde ── */
[data-testid="stSidebar"] img {
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    display: block;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #000000;
}
</style>
""", unsafe_allow_html=True)

_inject_css()

PALETTE = {"Fire": "#e74c3c", "Water": "#1a8fff", "Earth": "#8B5E3C", "Wind": "#B0C4DE"}
PRIO_COLOR = {"Alta": "#f5c6cb", "Media": "#ffeeba", "Baja": "#d4edda"}

BASE    = REPO_ROOT  # raíz del repo: data/, models/, reports/, src/
FIGS    = BASE / "reports" / "figures"
EXPORTS = BASE / "data" / "exports"

# ── Carga de datos ───────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    sessions = pd.read_parquet(BASE / "data" / "processed" / "sessions.parquet")
    rooms    = pd.read_parquet(BASE / "data" / "processed" / "rooms.parquet")
    levels   = pd.read_parquet(BASE / "data" / "processed" / "levels.parquet")
    clean    = sessions[~sessions["is_suspicious"]]
    rooms_c  = rooms[rooms["sessionId"].isin(clean["sessionId"])]
    levels_c = levels[levels["sessionId"].isin(clean["sessionId"])]

    bi   = pd.read_csv(EXPORTS / "balance_index.csv")
    recs = pd.read_csv(EXPORTS / "balance_recommendations.csv")
    ml   = pd.read_csv(EXPORTS / "ml_predictions.csv")
    cls  = pd.read_csv(EXPORTS / "player_clusters.csv")
    fi   = pd.read_csv(EXPORTS / "feature_importances.csv")
    nlp  = pd.read_csv(EXPORTS / "nlp_results.csv")
    stat = pd.read_csv(EXPORTS / "statistical_results.csv")

    # Calcular eficiencias de hechizos dinámicamente
    kmap = {
        "Projectile": ["killedWith_Projectile"],
        "Blast":      ["killedWith_Blast"],
        "Beam":       ["killedWith_Beam"],
        "AOE":        ["killedWith_AOE"],
    }
    spell_stats = []
    for spell, kcols in kmap.items():
        valid = [c for c in kcols if c in rooms_c.columns]
        kills = rooms_c[valid].sum().sum()
        casts = rooms_c[f"cast_{spell}"].sum()
        nomana_col = f"noMana_{spell}"
        nomana = rooms_c[nomana_col].sum() if nomana_col in rooms_c.columns else 0
        spell_stats.append({
            "Hechizo": spell, "Lanzamientos": int(casts),
            "Kills": int(kills), "noMana": int(nomana),
            "Eficiencia (kills/cast)": round(kills / casts, 3) if casts > 0 else 0,
        })
    df_spells = pd.DataFrame(spell_stats).sort_values("Lanzamientos", ascending=False)

    return sessions, clean, rooms_c, levels_c, bi, recs, ml, cls, fi, nlp, stat, df_spells

(sessions, clean, rooms_c, levels_c,
 bi, recs, ml, cls, fi, nlp, stat, df_spells) = cargar_datos()

# Métricas globales
n_total   = len(sessions)
n_clean   = len(clean)
n_susp    = n_total - n_clean
wr_global = clean["isVictory"].mean()
n_wins    = int(clean["isVictory"].sum())

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    _logo = BASE / "reports" / "logo.png"
    if _logo.exists():
        st.image(str(_logo), use_container_width=True)
    st.markdown("**Arcane Descent**")
    st.caption("Game Balance Analytics · 2026")
    st.divider()

    st.markdown("### Filtros")
    st.markdown("**Elemento del jugador**")
    _checks = {}
    for _e, _c in PALETTE.items():
        _col_card, _col_chk = st.columns([4, 1])
        _col_card.markdown(f"""
<div class='card-sm' style='border-left:4px solid {_c}'>
<span class='card-label'>{_e}</span>
</div>""", unsafe_allow_html=True)
        _checks[_e] = _col_chk.checkbox("", value=True, key=f"chk_{_e}")
    elem_sel = [e for e, v in _checks.items() if v]
    if not elem_sel:
        elem_sel = list(PALETTE.keys())

    st.divider()

    # ── Modo administrador ────────────────────────────────────────────────────
    # Protege acciones destructivas (escritura en CSV compartido) frente a visitantes anónimos.
    # La password se lee de Streamlit Secrets en cloud o de .env en local.
    def _admin_password() -> str:
        import os
        # En Streamlit Cloud
        try:
            if "ADMIN_PASSWORD" in st.secrets:
                return st.secrets["ADMIN_PASSWORD"]
        except Exception:
            pass
        # En local
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
        return os.getenv("ADMIN_PASSWORD", "")

    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    with st.expander("🔒 Modo administrador"):
        if st.session_state.is_admin:
            st.success("Admin activo")
            if st.button("Cerrar sesión admin", key="btn_admin_logout"):
                st.session_state.is_admin = False
                st.rerun()
        else:
            _pwd = st.text_input("Contraseña", type="password", key="admin_pwd_input")
            if st.button("Acceder", key="btn_admin_login"):
                _expected = _admin_password()
                if _expected and _pwd == _expected:
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("Contraseña incorrecta")

    st.divider()
    st.caption("Samuel Alcaraz Rodriguez · Game Balance Analytics")


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_resumen, tab_elem, tab_spells, tab_salas, tab_ml, tab_feedback, tab_recs, tab_ia = st.tabs([
    "Resumen", "Elementos", "Hechizos",
    "Salas y Dificultad", "Modelo Predictivo", "Feedback de Jugadores", "Recomendaciones", "🤖 Asistente IA",
])


# ══════════════════════════════════════════════════════════════════════════════
# 1 — RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
with tab_resumen:
    st.header("Resumen Ejecutivo")
    st.caption(
        f"{n_clean} sesiones WebGL válidas de {n_total} registradas · "
        "Resultados orientativos — se requieren ≥30 sesiones por elemento para validación estadística robusta."
    )

    # KPIs principales
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sesiones válidas (WebGL)", n_clean, delta=f"de {n_total} registradas", delta_color="off")
    c2.metric("Tasa de victoria global", f"{wr_global:.0%}",
              help="Victorias / sesiones válidas. Referencia saludable para prototipo: 20-40%.")
    # Progreso medio del juego = media de game_completion (salas visitadas / 24 salas totales)
    _game_compl_avg = clean["game_completion"].mean() if "game_completion" in clean.columns else 0
    c3.metric("Progreso medio del juego", f"{_game_compl_avg:.0%}",
              help="Media de salas visitadas por sesión sobre las 24 salas totales del juego completo (Level1=5, Level2=6, Level3=7, Level4=6).")
    c4.metric("Recomendaciones de alta prioridad",
              int((recs["priority"] == "Alta").sum()),
              help="Acciones inmediatas identificadas en el análisis.")
    best_elem = bi.sort_values("puntuacion_global", ascending=False).iloc[0]
    c5.metric("Elemento mejor balanceado",
              best_elem["playerElement"],
              delta=f"Puntuación = {best_elem['puntuacion_global']:.3f}",
              help="Puntuación global: combina tasa de victoria, eficiencia de kills y sentimiento del jugador.")

    st.divider()

    # Puntuación global
    st.subheader("Puntuación global por elemento")
    bi_f = bi[bi["playerElement"].isin(elem_sel)].sort_values("puntuacion_global", ascending=False)
    _pocos = bi_f[~bi_f["datos_suficientes"]]["playerElement"].tolist()
    _aviso = f" ⚠️ {', '.join(_pocos)} tienen n<10 — valores menos fiables." if _pocos else ""
    st.caption(f"Puntuación global = 0.5 × tasa_victoria + 0.3 × kills_efficiency (norm.) + 0.2 × sentimiento. Cada elemento tiene un status de rol propio: Fire=Burn, Water=Slow, Earth=Stun, Wind=Knockback.{_aviso}")

    fig_bi = px.bar(
        bi_f, x="playerElement", y="puntuacion_global",
        color="playerElement", color_discrete_map=PALETTE,
        text=bi_f["puntuacion_global"].round(3).values,
        labels={"playerElement": "Elemento", "puntuacion_global": "Puntuación global"},
    )
    fig_bi.update_traces(
        textposition="outside", textfont_size=13,
        hovertemplate="<b>%{x}</b><br>Puntuación global: %{y:.4f}<extra></extra>",
    )
    fig_bi.update_yaxes(range=[0, bi_f["puntuacion_global"].max() * 1.3])
    fiable = bi_f[bi_f["datos_suficientes"]]["puntuacion_global"]
    if not fiable.empty:
        fig_bi.add_hline(y=fiable.mean(), line_dash="dash", line_color="gray",
                         annotation_text=f"Media elementos fiables ({fiable.mean():.3f})",
                         annotation_position="top right")
    fig_bi.update_layout(showlegend=False, height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_bi, use_container_width=True)

    st.subheader("Detalle por elemento")
    _clean_f = clean[clean["playerElement"].isin(elem_sel)]
    _grp = _clean_f.groupby("playerElement")
    _bi_live = pd.DataFrame({
        "Elemento": _grp["sessionId"].nunique().index,
        "Sesiones": _grp["sessionId"].nunique().values,
        "Victoria": _grp["isVictory"].mean().values,
    }).reset_index(drop=True)

    # kills_per_min normalizado
    _kpm = _grp["kills_per_min"].mean()
    _kpm_norm = (_kpm - _kpm.min()) / (_kpm.max() - _kpm.min()) if _kpm.max() > _kpm.min() else _kpm * 0 + 0.5
    _bi_live["Kills/min (norm.)"] = _bi_live["Elemento"].map(_kpm_norm).fillna(0)

    # Control de rol: status característico / total kills de ese elemento
    _role_status = {"Fire": "status_Burn", "Water": "status_Slow", "Earth": "status_Stun", "Wind": "status_Knockback"}
    _rooms_f = rooms_c[rooms_c["sessionId"].isin(_clean_f["sessionId"])]
    _control = {}
    for _elem, _scol in _role_status.items():
        if _elem not in elem_sel:
            continue
        _r = _rooms_f[_rooms_f["sessionId"].isin(_clean_f[_clean_f["playerElement"] == _elem]["sessionId"])]
        _total_kills = _r["total_kills_room"].sum()
        _control[_elem] = (_r[_scol].sum() / _total_kills) if _total_kills > 0 else 0
    _ctrl_s = pd.Series(_control)
    _ctrl_norm = (_ctrl_s - _ctrl_s.min()) / (_ctrl_s.max() - _ctrl_s.min()) if _ctrl_s.max() > _ctrl_s.min() else _ctrl_s * 0 + 0.5
    _bi_live["Control rol (norm.)"] = _bi_live["Elemento"].map(_ctrl_norm).fillna(0)

    # Sentimiento desde NLP
    _nlp_f = nlp[nlp["sessionId"].isin(_clean_f["sessionId"])]
    _sent = _nlp_f.groupby("playerElement")["sentiment_score"].mean()
    _sent_norm = (_sent - _sent.min()) / (_sent.max() - _sent.min()) if _sent.max() > _sent.min() else _sent * 0 + 0.5
    _bi_live["Sentimiento"] = _bi_live["Elemento"].map(_sent).fillna(0.5)
    _bi_live["_sent_norm"]  = _bi_live["Elemento"].map(_sent_norm).fillna(0.5)
    _bi_live["_vic_norm"]   = (_bi_live["Victoria"] - _bi_live["Victoria"].min()) / (_bi_live["Victoria"].max() - _bi_live["Victoria"].min() + 1e-9)

    # Puntuación global recalculada — misma fórmula que balance_index.csv: 0.5·wr + 0.3·keff + 0.2·sent
    _bi_live["Puntuación global"] = (
        0.50 * _bi_live["_vic_norm"] +
        0.30 * _bi_live["Kills/min (norm.)"] +
        0.20 * _bi_live["_sent_norm"]
    )
    _bi_live["Fiable"] = _bi_live["Sesiones"].apply(lambda n: "✅" if n >= 10 else "⚠️")
    _bi_live = _bi_live.sort_values("Puntuación global", ascending=False)
    _bi_live["Victoria"]             = _bi_live["Victoria"].map("{:.0%}".format)
    _bi_live["Kills/min (norm.)"]    = _bi_live["Kills/min (norm.)"].map("{:.3f}".format)
    _bi_live["Control rol (norm.)"]  = _bi_live["Control rol (norm.)"].map("{:.3f}".format)
    _bi_live["Sentimiento"]          = _bi_live["Sentimiento"].map("{:.2f}".format)
    _bi_live["Puntuación global"]    = _bi_live["Puntuación global"].map("{:.3f}".format)
    _bi_live = _bi_live[["Elemento","Sesiones","Victoria","Kills/min (norm.)","Sentimiento","Puntuación global","Fiable"]]
    st.dataframe(_bi_live, use_container_width=True, hide_index=True, height=230)
    st.caption("⚠️ = n<10 sesiones · Control rol: Fire=Burn, Water=Slow, Earth=Stun, Wind=Knockback — aplicaciones de status / kills totales (norm.)")

    st.divider()

    # Hallazgos clave como tarjetas
    st.subheader("Hallazgos clave del análisis")
    hallazgos = [
        ("", "Tasa de victoria", f"{wr_global:.0%} ({n_wins}/{n_clean} sesiones válidas)",
         "Win rate muy bajo — posible problema de dificultad o retención. Revisar churn." if wr_global < 0.20 else
         "Rango saludable para un prototipo (20-40%). El juego tiene dificultad adecuada." if wr_global <= 0.40 else
         "Win rate alto — el juego puede ser demasiado fácil. Considerar aumentar la dificultad."),
        ("", "Hechizo más eficiente",
         f"{df_spells.loc[df_spells['Eficiencia (kills/cast)'].idxmax(), 'Hechizo']} — "
         f"{df_spells['Eficiencia (kills/cast)'].max():.2f} kills/cast",
         (lambda best, worst: (
             f"{best['Hechizo']} es {best['Eficiencia (kills/cast)'] / worst['Eficiencia (kills/cast)']:.0f}× más eficiente que "
             f"{worst['Hechizo']} pero muy poco usado "
             f"({int(best['Lanzamientos'])} vs {int(worst['Lanzamientos'])} lanzamientos)."
         ))(
             df_spells.loc[df_spells['Eficiencia (kills/cast)'].idxmax()],
             df_spells.loc[df_spells['Lanzamientos'].idxmax()],
         )),
        ("", "Hechizo más usado",
         f"{df_spells.loc[df_spells['Lanzamientos'].idxmax(), 'Hechizo']} — "
         f"{int(df_spells['Lanzamientos'].max())} lanzamientos",
         (lambda r: (
             f"Eficiencia {'baja' if r['Eficiencia (kills/cast)'] < 0.3 else 'media' if r['Eficiencia (kills/cast)'] < 0.7 else 'alta'} "
             f"({r['Eficiencia (kills/cast)']:.3f} kills/cast). "
             f"{'Jugadores lo sobreestiman como opción de daño.' if r['Eficiencia (kills/cast)'] < 0.3 else 'Uso y eficiencia equilibrados.'}"
         ))(df_spells.loc[df_spells['Lanzamientos'].idxmax()]),),
        ("", "Mejor predictor de victoria",
         (lambda r: f"{r['variable'].replace(' vs isVictory','').strip()} — los jugadores que ganan matan más de lo que mueren"
          if r is not None and "kd" in r['variable'] else
          (f"{r['variable'].replace(' vs isVictory','').strip()} — factor más asociado a ganar" if r is not None else "N/D"))(
             stat[(stat['test'] == 'Spearman') & (stat['tipo'] == 'causal') & (stat['p_value'] < 0.05)]
             .sort_values('effect_size', ascending=False)
             .iloc[0] if len(stat[(stat['test'] == 'Spearman') & (stat['tipo'] == 'causal') & (stat['p_value'] < 0.05)]) > 0 else None
         ),
         (lambda r: (
             f"El ratio kills/muertes (kd_ratio) es el indicador que mejor distingue a los jugadores que ganan de los que pierden "
             f"(correlación {r['effect_size']:.2f} sobre 1.0). "
             f"El elemento elegido no tiene impacto significativo — ganar depende de la habilidad, no del elemento."
         ) if r is not None and "kd" in r['variable'] else
         (f"'{r['variable'].replace(' vs isVictory','').strip()}' es el factor más asociado a ganar (correlación {r['effect_size']:.2f})."
          if r is not None else "Sin predictores significativos."))(
             stat[(stat['test'] == 'Spearman') & (stat['tipo'] == 'causal') & (stat['p_value'] < 0.05)]
             .sort_values('effect_size', ascending=False)
             .iloc[0] if len(stat[(stat['test'] == 'Spearman') & (stat['tipo'] == 'causal') & (stat['p_value'] < 0.05)]) > 0 else None
         )),
        ("", "Cuello de botella",
         (lambda funnel: (
             f"{funnel.iloc[funnel['drop'].idxmax()]['levelId']} — "
             f"{funnel['drop'].max():.0%} de sesiones no pasan"
         ))(
             (lambda f: f.assign(drop=f['pct'] - f['pct'].shift(-1).fillna(0)))(
                 levels_c.groupby('levelId')['sessionId'].count().reset_index()
                 .rename(columns={'sessionId':'n'})
                 .assign(pct=lambda x: x['n'] / len(clean))
                 .sort_values('levelId')
                 .reset_index(drop=True)
             )
         ),
         (lambda funnel: (
             f"Mayor caída entre niveles en {funnel.iloc[funnel['drop'].idxmax()]['levelId']}. "
             f"{'Barrera de entrada demasiado alta.' if funnel.iloc[0]['levelId'] == funnel.iloc[funnel['drop'].idxmax()]['levelId'] else 'Pico de dificultad no esperado.'}"
         ))(
             (lambda f: f.assign(drop=f['pct'] - f['pct'].shift(-1).fillna(0)))(
                 levels_c.groupby('levelId')['sessionId'].count().reset_index()
                 .rename(columns={'sessionId':'n'})
                 .assign(pct=lambda x: x['n'] / len(clean))
                 .sort_values('levelId')
                 .reset_index(drop=True)
             )
         )),
        ("", "Bug de telemetría",
         (lambda v, n, fixed: f"damage_Unknown — {int(v)} pts en {n} salas {'✅ corregido en v. actual' if fixed else '⚠️ activo'}" if v > 0 else "damage_Unknown — sin datos afectados")(
             rooms_c["damage_Unknown"].sum() if "damage_Unknown" in rooms_c.columns else 0,
             int((rooms_c["damage_Unknown"] > 0).sum()) if "damage_Unknown" in rooms_c.columns else 0,
             (rooms_c[rooms_c["damage_Unknown"] > 0]["sessionId"].max() or "") <
             (sessions["sessionId"].max() or "") if "damage_Unknown" in rooms_c.columns else True,
         ),
         (lambda v, fixed: (
             "Bug ya corregido en Unity — las sesiones históricas afectadas se mantienen en el análisis pero el dato está sesgado en esas salas." if v > 0 and fixed
             else "Fuente de daño no identificada en Unity. Pendiente de corregir." if v > 0
             else "Sin daño desconocido en el dataset actual."
         ))(
             rooms_c["damage_Unknown"].sum() if "damage_Unknown" in rooms_c.columns else 0,
             (rooms_c[rooms_c["damage_Unknown"] > 0]["sessionId"].max() or "") <
             (sessions["sessionId"].max() or "") if "damage_Unknown" in rooms_c.columns else True,
         )),
    ]
    cols = st.columns(3)
    for i, (icon, titulo, valor, obs) in enumerate(hallazgos):
        with cols[i % 3]:
            st.markdown(f"""
<div class='card'>
<div class='card-title'>{titulo}</div>
<div class='card-value'>{valor}</div>
<div class='card-obs'>{obs}</div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Resumen ejecutivo con IA ───────────────────────────────────────────────
    st.subheader("🤖 Resumen ejecutivo generado por IA")
    st.caption(
        "Gemini genera un resumen ejecutivo profesional basándose en la Puntuación Global, "
        "los tests estadísticos y las recomendaciones de alta prioridad."
    )
    if st.button("Generar resumen ejecutivo", key="btn_exec_summary", type="primary"):
        with st.spinner("Gemini redactando el resumen..."):
            try:
                from analysis import generate_executive_summary
                _bi_for_summary = bi.rename(columns={"puntuacion_global": "balance_index"}) \
                    if "puntuacion_global" in bi.columns else bi
                _exec_summary = generate_executive_summary(
                    _bi_for_summary, stat, recs, n_sessions=n_clean,
                )
                st.markdown(_exec_summary)
            except Exception as _e:
                st.error(f"⚠️ Error al generar el resumen: {_e}")


# ══════════════════════════════════════════════════════════════════════════════
# 2 — ELEMENTOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_elem:
    st.header("Comparativa de Elementos")
    st.caption(
        "Comparativa de KPIs de rendimiento por elemento del jugador. "
        "Cada elemento tiene un rol diferenciado — las diferencias en kills/minuto responden al diseño, no a desbalance."
    )

    cls_f = cls[cls["playerElement"].isin(elem_sel)]
    ml_f  = ml[ml["playerElement"].isin(elem_sel)]

    if cls_f.empty:
        st.warning("Selecciona al menos un elemento en el sidebar.")
    else:
        # Fila 1: Win rate + kd_ratio
        col1, _gap1, col2 = st.columns([10, 1, 10])

        wr_df = ml_f.groupby("playerElement")["y_real"].agg(
            n="count", victorias="sum"
        ).reset_index()
        wr_df["Tasa de victoria"] = wr_df["victorias"] / wr_df["n"]
        fig_wr = px.bar(
            wr_df.sort_values("Tasa de victoria", ascending=False),
            x="playerElement", y="Tasa de victoria",
            color="playerElement", color_discrete_map=PALETTE,
            text=wr_df.sort_values("Tasa de victoria", ascending=False)["Tasa de victoria"].map("{:.0%}".format),
            title="Tasa de victoria por elemento",
            labels={"playerElement": "Elemento"},
        )
        fig_wr.update_traces(textposition="outside")
        fig_wr.update_yaxes(tickformat=".0%", range=[0, 1])
        fig_wr.update_layout(showlegend=False, height=320)
        col1.plotly_chart(fig_wr, use_container_width=True)

        # Conclusión dinámica del win rate
        _wr_sorted = wr_df.sort_values("Tasa de victoria", ascending=False)
        _best  = _wr_sorted.iloc[0]
        _worst = _wr_sorted.iloc[-1]
        _rango = _best["Tasa de victoria"] - _worst["Tasa de victoria"]
        if _rango < 0.10:
            _concl = f"Los elementos están **bien equilibrados** en tasa de victoria (diferencia máxima {_rango:.0%}). No hay un elemento claramente dominante."
        elif _rango < 0.25:
            _concl = (f"**{_best['playerElement']}** lidera con {_best['Tasa de victoria']:.0%}, "
                      f"seguido de cerca. Diferencia moderada ({_rango:.0%}) — posible desbalance leve.")
        else:
            _concl = (f"**{_best['playerElement']}** domina con {_best['Tasa de victoria']:.0%} vs "
                      f"{_worst['playerElement']} con solo {_worst['Tasa de victoria']:.0%} "
                      f"(diferencia de {_rango:.0%}). Desbalance significativo entre elementos.")
        col1.caption(_concl)

        def _boxplot_es(df, col, title, ylabel, height=320, yrange=None):
            """go.Box con tooltip en español para caja y puntos individuales."""
            fig = go.Figure()
            for elem in [e for e in ["Fire","Water","Earth","Wind"] if e in df["playerElement"].values]:
                vals = df[df["playerElement"] == elem][col].dropna()
                q1, med, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.5)), float(vals.quantile(0.75))
                iqr = q3 - q1
                low_fence = max(float(vals.min()), q1 - 1.5 * iqr)
                up_fence  = min(float(vals.max()), q3 + 1.5 * iqr)
                color = PALETTE.get(elem, "#888")
                # Caja — tooltip estadístico en español
                fig.add_trace(go.Box(
                    y=vals, x=[elem] * len(vals),
                    name=elem,
                    marker_color=color,
                    line_color=color,
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    customdata=[[q1, med, q3, iqr, low_fence, up_fence, len(vals)]] * len(vals),
                    hovertemplate=(
                        "<b>" + elem + "</b><br>"
                        "Mediana: %{customdata[1]:.2f}<br>"
                        "Q1: %{customdata[0]:.2f} — Q3: %{customdata[2]:.2f}<br>"
                        "IQR: %{customdata[3]:.2f}<br>"
                        "Bigote inf.: %{customdata[4]:.2f} — Bigote sup.: %{customdata[5]:.2f}<br>"
                        "Valor: %{y:.2f}<br>"
                        "N sesiones: %{customdata[6]:.0f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ))
            fig.update_traces(width=0.5)
            fig.update_layout(
                title=title,
                xaxis_title="Elemento",
                yaxis_title=ylabel,
                yaxis=dict(range=yrange) if yrange else {},
                showlegend=False,
                height=height,
            )
            return fig

        _cls_kills = cls_f.merge(
            clean[["sessionId", "totalKills"]], on="sessionId", how="left"
        ) if "totalKills" in clean.columns else cls_f.assign(totalKills=cls_f["kd_ratio"] * cls_f["totalDeaths"].replace(0, 1))
        _kd_agg = _cls_kills.groupby("playerElement").agg(
            kills_mediana=("totalKills", "median"),
            deaths_mediana=("totalDeaths", "median"),
        ).reset_index()
        # Orden consistente con el resto del dashboard
        _elem_order = [e for e in ["Fire", "Water", "Earth", "Wind"] if e in _kd_agg["playerElement"].values]
        _kd_agg["playerElement"] = pd.Categorical(_kd_agg["playerElement"], categories=_elem_order, ordered=True)
        _kd_agg = _kd_agg.sort_values("playerElement")

        fig_kd = go.Figure()
        fig_kd.add_trace(go.Bar(
            x=_kd_agg["playerElement"], y=_kd_agg["kills_mediana"],
            name="Kills", marker_color="#e74c3c",
            text=_kd_agg["kills_mediana"].round(0).astype(int),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Kills (mediana): %{y:.0f}<extra></extra>",
        ))
        fig_kd.add_trace(go.Bar(
            x=_kd_agg["playerElement"], y=_kd_agg["deaths_mediana"],
            name="Muertes", marker_color="#888888",
            text=_kd_agg["deaths_mediana"].round(1),
            textposition="outside",
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Muertes (mediana): %{y:.1f}<extra></extra>",
        ))
        fig_kd.update_layout(
            title="Kills y muertes del jugador por elemento (mediana)",
            xaxis_title="Elemento",
            yaxis=dict(title="Kills (mediana)", showgrid=True, range=[0, 80]),
            yaxis2=dict(title="Muertes (mediana)", overlaying="y", side="right",
                        range=[0, _kd_agg["deaths_mediana"].max() * 4]),
            barmode="group",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.15),
            legend_itemclick=False, legend_itemdoubleclick=False,
            height=320,
            margin=dict(r=100),
        )
        col2.plotly_chart(fig_kd, use_container_width=True)
        _kills_best = _kd_agg.sort_values("kills_mediana", ascending=False).iloc[0]
        _deaths_max = _kd_agg.sort_values("deaths_mediana", ascending=False).iloc[0]
        col2.caption(
            f"La mayoría de jugadores muere muy pocas veces (mediana global {cls_f['totalDeaths'].median():.0f} muertes/sesión) "
            f"— el combate directo es fácil. **{_kills_best['playerElement']}** es el elemento más letal "
            f"({_kills_best['kills_mediana']:.0f} kills/sesión de mediana). "
            f"**{_deaths_max['playerElement']}** es el que más muere ({_deaths_max['deaths_mediana']:.0f} muertes/sesión)."
        )

        # Fila 2: kills/min + cast accuracy
        col3, _gap2, col4 = st.columns([10, 1, 10])

        _kpm_agg = cls_f.groupby("playerElement")["kills_per_min"].agg(
            mediana="median", std="std", n="count"
        ).reset_index()
        _kpm_agg["error"] = 1.96 * _kpm_agg["std"] / _kpm_agg["n"].pow(0.5)
        _kpm_agg = _kpm_agg.sort_values("mediana", ascending=False)
        _kpm_agg["playerElement"] = pd.Categorical(
            _kpm_agg["playerElement"],
            categories=[e for e in ["Fire","Water","Earth","Wind"] if e in _kpm_agg["playerElement"].values],
            ordered=True
        )
        _kpm_agg = _kpm_agg.sort_values("playerElement")

        fig_kpm = go.Figure()
        fig_kpm.add_trace(go.Bar(
            x=_kpm_agg["playerElement"],
            y=_kpm_agg["mediana"],
            marker_color=[PALETTE.get(e, "#888") for e in _kpm_agg["playerElement"]],
            text=_kpm_agg["mediana"].round(2),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Kills/min (mediana): %{y:.2f}<extra></extra>",
        ))
        fig_kpm.update_layout(
            title="Kills por minuto por elemento (mediana)",
            xaxis_title="Elemento", yaxis_title="Kills / minuto",
            showlegend=False, height=320,
            yaxis=dict(range=[0, _kpm_agg["mediana"].max() * 1.3]),
        )
        col3.plotly_chart(fig_kpm, use_container_width=True)
        _kpm_best = _kpm_agg.sort_values("mediana", ascending=False).iloc[0]
        _kpm_worst = _kpm_agg.sort_values("mediana").iloc[0]
        col3.caption(
            f"**Fire** lidera por diseño: Burn (DOT) acumula kills pasivamente ({_kpm_best['mediana']:.2f} k/min). "
            f"Water/Earth/Wind tienen kills/min más bajos porque su valor está en el control: "
            f"Water ralentiza (Slow 50%), Earth stunea (escala con daño) y Wind empuja sin matar directamente (Knockback). "
            f"Una kills/min baja en estos elementos no indica desbalance — indica que el jugador está usando el rol correctamente."
        )

        _ca_agg = cls_f.groupby("playerElement")["cast_accuracy"].agg(
            mediana="median", std="std", n="count"
        ).reset_index()
        _ca_agg["error"] = 1.96 * _ca_agg["std"] / _ca_agg["n"].pow(0.5)
        _ca_agg["playerElement"] = pd.Categorical(
            _ca_agg["playerElement"],
            categories=[e for e in ["Fire","Water","Earth","Wind"] if e in _ca_agg["playerElement"].values],
            ordered=True
        )
        _ca_agg = _ca_agg.sort_values("playerElement")

        fig_ca = go.Figure()
        fig_ca.add_trace(go.Bar(
            x=_ca_agg["playerElement"],
            y=_ca_agg["mediana"],
            marker_color=[PALETTE.get(e, "#888") for e in _ca_agg["playerElement"]],
            text=_ca_agg["mediana"].map("{:.0%}".format),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Precisión mediana: %{y:.0%}<extra></extra>",
        ))
        fig_ca.update_layout(
            title="Precisión de lanzamiento por elemento (mediana)",
            xaxis_title="Elemento", yaxis_title="Precisión (0-1)",
            showlegend=False, height=320,
            yaxis=dict(tickformat=".0%", range=[0, _ca_agg["mediana"].max() * 1.3]),
        )
        col4.plotly_chart(fig_ca, use_container_width=True)
        _ca_best = _ca_agg.sort_values("mediana", ascending=False).iloc[0]
        _ca_worst = _ca_agg.sort_values("mediana").iloc[0]
        col4.caption(
            f"Proporción de hechizos lanzados que impactan a un enemigo. "
            f"Todos los elementos tienen precisión alta y similar (88-94%) — sin diferencia significativa (p>0.05). "
            f"Earth lidera ({_ca_best['mediana']:.0%}) lo cual es coherente con su diseño: el Stun escala con daño, "
            f"por lo que los jugadores de Earth apuntan más cuidadosamente para maximizar la duración del stun."
        )

        st.divider()

        # Churn rate por elemento: sesiones que no completaron ningún nivel
        st.subheader("Tasa de abandono temprano por elemento")
        st.caption("Sesiones con 0 niveles completados — el jugador abandona antes de terminar el Level1.")

        _churn_df = clean[clean["playerElement"].isin(elem_sel)].copy()
        _churn_df["abandono"] = _churn_df["levelsCompleted"] == 0
        _churn_agg = _churn_df.groupby("playerElement").agg(
            n=("sessionId", "count"),
            abandonos=("abandono", "sum"),
        ).reset_index()
        _churn_agg["tasa"] = _churn_agg["abandonos"] / _churn_agg["n"]
        _churn_agg["playerElement"] = pd.Categorical(
            _churn_agg["playerElement"],
            categories=[e for e in ["Fire","Water","Earth","Wind"] if e in _churn_agg["playerElement"].values],
            ordered=True,
        )
        _churn_agg = _churn_agg.sort_values("playerElement")

        fig_churn = go.Figure(go.Bar(
            x=_churn_agg["playerElement"],
            y=_churn_agg["tasa"],
            marker_color=[PALETTE.get(e, "#888") for e in _churn_agg["playerElement"]],
            text=_churn_agg.apply(lambda r: f"{r['tasa']:.0%} ({int(r['abandonos'])}/{int(r['n'])})", axis=1),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Abandono: %{text}<extra></extra>",
        ))
        fig_churn.update_layout(
            xaxis_title="Elemento", yaxis_title="Tasa de abandono",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            showlegend=False, height=320,
        )

        _churn_col, _churn_gap, _churn_txt = st.columns([10, 1, 10])
        _churn_col.plotly_chart(fig_churn, use_container_width=True)

        _worst_churn = _churn_agg.sort_values("tasa", ascending=False).iloc[0]
        _best_churn  = _churn_agg.sort_values("tasa").iloc[0]
        _churn_txt.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        _churn_txt.markdown(f"""
<div class='card-obs' style='font-size:0.92em;line-height:1.8'>
<b>{_worst_churn['playerElement']}</b> tiene la mayor tasa de abandono temprano
({_worst_churn['tasa']:.0%} de sus sesiones no completan ningún nivel).<br><br>
<b>{_best_churn['playerElement']}</b> retiene mejor a sus jugadores
({_best_churn['tasa']:.0%} de abandono).<br><br>
Una tasa alta puede indicar que el elemento resulta frustrante o difícil de aprender
desde el inicio — candidato prioritario para revisión de onboarding.
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3 — HECHIZOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_spells:
    st.header("Análisis de Hechizos")
    st.caption(
        "Eficiencia = kills atribuidos al hechizo / lanzamientos totales. "
        "noMana = intentos de uso sin maná suficiente (indicador de demanda insatisfecha)."
    )

    # ── Datos base ────────────────────────────────────────────────────────────
    _kmap_sp = {
        "Projectile": ["killedWith_Projectile"],
        "Blast":      ["killedWith_Blast"],
        "Beam":       ["killedWith_Beam"],
        "AOE":        ["killedWith_AOE"],
    }
    _spell_colors = {"Beam": "#e74c3c", "Projectile": "#1a8fff", "Blast": "#f39c12", "AOE": "#27ae60"}
    _df_sp = rooms_c.merge(clean[["sessionId","playerElement"]], on="sessionId", how="left")
    _elem_sp = [e for e in ["Fire","Water","Earth","Wind"] if e in _df_sp["playerElement"].values]

    _sp_rows = []
    for spell, kcols in _kmap_sp.items():
        valid = [c for c in kcols if c in _df_sp.columns]
        kills_s = _df_sp[valid].sum(axis=1) if valid else pd.Series(0, index=_df_sp.index)
        casts_s = _df_sp[f"cast_{spell}"]
        for elem in _elem_sp:
            mask = _df_sp["playerElement"] == elem
            k = float(kills_s[mask].sum())
            c = float(casts_s[mask].sum())
            _sp_rows.append({"Hechizo": spell, "playerElement": elem,
                              "kills": k, "casts": c,
                              "eff": round(k / c, 3) if c > 0 else 0})
    _sp_df = pd.DataFrame(_sp_rows)

    _nomana_map = {"Projectile": "noMana_Projectile", "Blast": "noMana_Blast",
                   "Beam": "noMana_Beam", "AOE": "noMana_AOE"}
    _nm_rows = []
    for spell, col in _nomana_map.items():
        if col in _df_sp.columns:
            for elem in _elem_sp:
                _nm_rows.append({"Hechizo": spell, "playerElement": elem,
                                  "noMana": float(_df_sp[_df_sp["playerElement"] == elem][col].sum())})
    _nm_df = pd.DataFrame(_nm_rows)

    def _spell_bar(metric, title, ylabel, fmt=".1f", df=None):
        src = df if df is not None else _sp_df
        _elems = [e for e in ["Fire","Water","Earth","Wind"] if e in src["playerElement"].values]
        _spells_render = [s for s in ["Beam", "Projectile", "Blast", "AOE"] if s in src["Hechizo"].values]
        fig = go.Figure()
        for spell in _spells_render:
            d = src[src["Hechizo"] == spell].set_index("playerElement")
            y = [float(d.loc[e, metric]) if e in d.index else 0 for e in _elems]
            fig.add_trace(go.Bar(
                name=spell, x=_elems, y=y,
                marker_color=_spell_colors[spell],
                text=[f"{v:{fmt}}" for v in y], textposition="outside",
                hovertemplate=f"<b>%{{x}}</b><br>{spell} — {ylabel}: %{{y:{fmt}}}<extra></extra>",
            ))
        fig.update_layout(
            title=title, xaxis_title="Elemento", yaxis_title=ylabel,
            barmode="group", height=340,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            legend_itemclick=False, legend_itemdoubleclick=False,
        )
        return fig

    # ══ SECCIÓN 1: Vista global ═══════════════════════════════════════════════
    st.subheader("Vista global — todos los hechizos")
    st.caption("Comparativa agregada de todos los hechizos independientemente del elemento.")

    # Tabla resumen global
    df_sp_show = df_spells.copy()
    df_sp_show["Eficiencia (kills/cast)"] = df_sp_show["Eficiencia (kills/cast)"].map("{:.3f}".format)
    df_sp_show["% uso"] = (df_spells["Lanzamientos"] / df_spells["Lanzamientos"].sum()).map("{:.1%}".format)
    st.dataframe(df_sp_show[["Hechizo","Lanzamientos","% uso","Kills","noMana","Eficiencia (kills/cast)"]],
                 use_container_width=True, hide_index=True, height=185)
    _beam_eff  = df_spells.loc[df_spells["Hechizo"]=="Beam",  "Eficiencia (kills/cast)"].values[0]
    _blast_nm  = int(df_spells.loc[df_spells["Hechizo"]=="Blast", "noMana"].values[0])
    _most_used = df_spells.sort_values("Lanzamientos", ascending=False).iloc[0]["Hechizo"]
    st.caption(
        f"**{_most_used}** y **Blast** son los ataques básicos (clic izq/der) — su uso mayoritario es lo esperado. "
        f"**Beam** ({_beam_eff:.2f} kills/cast) y **AOE** son significativamente más eficientes pero los jugadores apenas los usan — oportunidad de mejora en comunicación del kit. "
        f"**Blast** acumula {_blast_nm} intentos sin maná, la mayor demanda insatisfecha del dataset."
    )

    # Scatter uso vs eficiencia (filtrado por elem_sel del sidebar)
    _rooms_sel = rooms_c[rooms_c["sessionId"].isin(
        clean[clean["playerElement"].isin(elem_sel)]["sessionId"]
    )]
    _spell_stats_sel = []
    for _sp, _kcols in _kmap_sp.items():
        _valid = [c for c in _kcols if c in _rooms_sel.columns]
        _kills = _rooms_sel[_valid].sum().sum() if _valid else 0
        _casts = _rooms_sel[f"cast_{_sp}"].sum()
        _nm_col = f"noMana_{_sp}"
        _nm = _rooms_sel[_nm_col].sum() if _nm_col in _rooms_sel.columns else 0
        _spell_stats_sel.append({
            "Hechizo": _sp, "Lanzamientos": int(_casts),
            "Kills": int(_kills), "noMana": int(_nm),
            "Eficiencia (kills/cast)": round(_kills / _casts, 3) if _casts > 0 else 0,
        })
    _df_spells_sel = pd.DataFrame(_spell_stats_sel).sort_values("Lanzamientos", ascending=False)

    _order_sel = _df_spells_sel.sort_values("Lanzamientos", ascending=False)["Hechizo"].tolist()
    fig_uso_eff = go.Figure()
    fig_uso_eff.add_trace(go.Bar(
        x=_order_sel,
        y=[int(_df_spells_sel.loc[_df_spells_sel["Hechizo"]==s, "Lanzamientos"].values[0]) for s in _order_sel],
        name="Lanzamientos",
        marker_color="#1a8fff",
        text=[int(_df_spells_sel.loc[_df_spells_sel["Hechizo"]==s, "Lanzamientos"].values[0]) for s in _order_sel],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Lanzamientos: %{y}<extra></extra>",
    ))
    fig_uso_eff.add_trace(go.Scatter(
        x=_order_sel,
        y=[float(_df_spells_sel.loc[_df_spells_sel["Hechizo"]==s, "Eficiencia (kills/cast)"].values[0]) for s in _order_sel],
        name="Eficiencia (kills/cast)",
        mode="markers+lines+text",
        marker=dict(size=12, color="white", line=dict(color="gray", width=1)),
        line=dict(color="white", dash="dot"),
        text=[f"{float(_df_spells_sel.loc[_df_spells_sel['Hechizo']==s, 'Eficiencia (kills/cast)'].values[0]):.2f}" for s in _order_sel],
        textposition="top center",
        textfont=dict(color="white", size=11),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Eficiencia: %{y:.3f} kills/cast<extra></extra>",
    ))
    fig_uso_eff.update_layout(
        title=f"Uso vs Eficiencia — {', '.join(elem_sel)}",
        xaxis_title="Hechizo",
        yaxis=dict(title="Total lanzamientos", showgrid=True),
        yaxis2=dict(title="Eficiencia (kills/cast)", overlaying="y", side="right",
                    range=[0, 3]),
        barmode="group",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.08),
        legend_itemclick=False, legend_itemdoubleclick=False,
        height=380,
        margin=dict(t=50, b=40, r=120),
    )
    st.plotly_chart(fig_uso_eff, use_container_width=True)
    st.caption(
        "Barras = total lanzamientos (eje izquierdo). Línea blanca = eficiencia kills/cast (eje derecho). "
        "Projectile y Blast son los ataques básicos — su uso mayoritario es normal. "
        "**Beam** y **AOE** son hechizos especiales con mucha mayor eficiencia pero muy pocos usos — los jugadores no los están aprovechando. "
        "⚠️ Earth/AOE destaca con eficiencia muy alta pero muy pocos casts — los jugadores de Earth no están usando su hechizo más poderoso."
    )

    st.divider()

    # ══ SECCIÓN 2: Comparativa por elemento ══════════════════════════════════
    st.subheader("Comparativa por elemento — todos los hechizos")
    st.caption("Desglose de uso, efectividad y demanda insatisfecha de cada hechizo según el elemento del jugador.")

    spells_sel = st.multiselect(
        "Hechizos mostrados:",
        options=["Beam", "Projectile", "Blast", "AOE"],
        default=["Beam", "Projectile", "Blast", "AOE"],
        key="spells_multisel",
    )
    if not spells_sel:
        spells_sel = ["Beam", "Projectile", "Blast", "AOE"]

    # Lanzamientos
    st.markdown("**¿Qué hechizos prefiere cada elemento?**")
    _sp_df_sel = _sp_df[_sp_df["playerElement"].isin(elem_sel) & _sp_df["Hechizo"].isin(spells_sel)]
    _elem_sp_sel = [e for e in ["Fire","Water","Earth","Wind"] if e in _sp_df_sel["playerElement"].values]
    st.plotly_chart(_spell_bar("casts", "Lanzamientos por hechizo y elemento",
                               "Total lanzamientos", fmt=".0f", df=_sp_df_sel), use_container_width=True)
    _casts_top = _sp_df_sel.groupby("Hechizo")["casts"].sum().idxmax()
    _casts_by_elem = {e: _sp_df_sel[_sp_df_sel["playerElement"]==e]["casts"].sum() for e in _elem_sp_sel}
    _most_cast_elem = max(_casts_by_elem, key=_casts_by_elem.get) if _casts_by_elem else "N/D"
    st.caption(
        f"**{_casts_top}** es el hechizo más lanzado en los elementos seleccionados. "
        f"**{_most_cast_elem}** es el elemento que más hechizos lanza en total. "
        "Diferencias entre elementos indican preferencias de estilo de juego o acceso desigual a maná."
    )

    # Kills + eficiencia en paralelo
    st.markdown("**¿Qué hechizos son más letales y para quién?**")
    st.plotly_chart(_spell_bar("kills", "Kills por hechizo y elemento",
                               "Total kills", fmt=".0f", df=_sp_df_sel), use_container_width=True)
    st.plotly_chart(_spell_bar("eff", "Eficiencia (kills/cast) por hechizo y elemento",
                               "Kills / cast", fmt=".3f", df=_sp_df_sel), use_container_width=True)
    _kills_top = _sp_df_sel.groupby("Hechizo")["kills"].sum().idxmax()
    _eff_top   = _sp_df_sel.loc[_sp_df_sel["eff"].idxmax()]
    # Detectar anomalías 0 kills con muchos casts
    _zero_kills = _sp_df_sel[(_sp_df_sel["eff"] == 0) & (_sp_df_sel["casts"] > 20)]
    _zero_note = ""
    if not _zero_kills.empty:
        _zk_list = ", ".join(f"**{r['playerElement']}/{r['Hechizo']}** ({int(r['casts'])} casts)" for _, r in _zero_kills.iterrows())
        _zero_note = f" ⚠️ {_zk_list} registran 0 kills pese a tener muchos lanzamientos — posible bug de telemetría o atribución incorrecta de kills."
    st.caption(
        f"**{_kills_top}** acumula más kills totales. "
        f"La combinación más eficiente es **{_eff_top['Hechizo']}** con **{_eff_top['playerElement']}** "
        f"({_eff_top['eff']:.2f} kills/cast). "
        f"Beam y AOE pueden superar 1.0 kills/cast al impactar varios enemigos simultáneamente.{_zero_note}"
    )

    # Demanda insatisfecha (noMana)
    st.markdown("**¿Qué hechizos quieren usar pero no pueden?**")
    st.caption("Intentos de uso de un hechizo sin maná suficiente — indica demanda real no satisfecha.")
    _nm_df_sel = _nm_df[_nm_df["playerElement"].isin(elem_sel) & _nm_df["Hechizo"].isin(spells_sel)]
    fig_nm = go.Figure()
    for spell in [s for s in ["Beam", "Projectile", "Blast", "AOE"] if s in spells_sel]:
        d = _nm_df_sel[_nm_df_sel["Hechizo"] == spell].set_index("playerElement")
        y = [float(d.loc[e, "noMana"]) if e in d.index else 0 for e in _elem_sp_sel]
        fig_nm.add_trace(go.Bar(
            name=spell, x=_elem_sp_sel, y=y,
            marker_color=_spell_colors[spell],
            text=[f"{int(v)}" for v in y], textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{spell} — noMana: %{{y:.0f}}<extra></extra>",
        ))
    fig_nm.update_layout(
        xaxis_title="Elemento", yaxis_title="Eventos noMana",
        barmode="group", height=340,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    st.plotly_chart(fig_nm, use_container_width=True)
    if not _nm_df_sel.empty:
        _nm_top_spell = _nm_df_sel.groupby("Hechizo")["noMana"].sum().idxmax()
        _nm_top_elem  = _nm_df_sel.groupby("playerElement")["noMana"].sum().idxmax()
        _nm_top_val   = int(_nm_df_sel.groupby("Hechizo")["noMana"].sum().max())
        st.caption(
            f"**{_nm_top_spell}** acumula la mayor demanda insatisfecha ({_nm_top_val} eventos noMana). "
            f"**{_nm_top_elem}** es el elemento que más veces se queda sin maná. "
            "Candidato a reducción de coste de maná o aumento de regeneración."
        )

    st.divider()

    # ══ SECCIÓN 3: Detalle de un hechizo concreto ════════════════════════════
    st.subheader("Detalle de un hechizo concreto")
    _spell_sel = st.selectbox(
        "Selecciona el hechizo a analizar:",
        options=["Beam", "Projectile", "Blast", "AOE"],
        index=0,
        key="spell_detail_sel",
    )

    # Tabla detalle por elemento para el hechizo seleccionado (sin filtros — datos completos)
    _detail_rows = []
    for elem in _elem_sp:
        r    = _sp_df[(_sp_df["Hechizo"]==_spell_sel) & (_sp_df["playerElement"]==elem)]
        nm_r = _nm_df[(_nm_df["Hechizo"]==_spell_sel) & (_nm_df["playerElement"]==elem)]
        casts = float(r["casts"].values[0]) if len(r) else 0
        kills = float(r["kills"].values[0]) if len(r) else 0
        eff   = float(r["eff"].values[0])   if len(r) else 0
        nm    = float(nm_r["noMana"].values[0]) if len(nm_r) else 0
        total_casts_elem = float(_sp_df[_sp_df["playerElement"]==elem]["casts"].sum())
        pct_uso = casts / total_casts_elem if total_casts_elem > 0 else 0
        _detail_rows.append({
            "Elemento": elem, "Lanzamientos": int(casts),
            "% del total del elemento": f"{pct_uso:.1%}",
            "Kills": int(kills), "noMana": int(nm),
            "Eficiencia (kills/cast)": f"{eff:.3f}",
        })
    _detail_df = pd.DataFrame(_detail_rows)
    st.dataframe(_detail_df, use_container_width=True, hide_index=True)

    # Barras comparativas del hechizo seleccionado vs el resto
    col_d1, _gd, col_d2 = st.columns([10, 1, 10])

    # Eficiencia del hechizo seleccionado por elemento (datos completos)
    _glob_eff = {sp: float(_sp_df[_sp_df["Hechizo"]==sp]["eff"].mean()) for sp in ["Beam","Projectile","Blast","AOE"]}
    _sel_eff_by_elem = {
        row["Elemento"]: float(_sp_df[(_sp_df["Hechizo"]==_spell_sel) & (_sp_df["playerElement"]==row["Elemento"])]["eff"].values[0])
        if len(_sp_df[(_sp_df["Hechizo"]==_spell_sel) & (_sp_df["playerElement"]==row["Elemento"])]) else 0
        for _, row in _detail_df.iterrows()
    }
    fig_d_eff = go.Figure()
    fig_d_eff.add_trace(go.Bar(
        x=list(_sel_eff_by_elem.keys()),
        y=list(_sel_eff_by_elem.values()),
        marker_color=[PALETTE.get(e, "#888") for e in _sel_eff_by_elem.keys()],
        text=[f"{v:.3f}" for v in _sel_eff_by_elem.values()],
        textposition="outside",
        name=_spell_sel,
        hovertemplate="<b>%{x}</b><br>Eficiencia: %{y:.3f} kills/cast<extra></extra>",
    ))
    _media_sel = _glob_eff.get(_spell_sel, 0)
    fig_d_eff.add_hline(
        y=_media_sel, line_dash="dash", line_color="gray",
        annotation_text=f"Media global {_spell_sel} ({_media_sel:.3f})",
        annotation_position="top right",
    )
    fig_d_eff.update_layout(
        title=f"Eficiencia de {_spell_sel} por elemento (todos los datos)",
        xaxis_title="Elemento", yaxis_title="Kills / cast",
        showlegend=False, height=320,
        yaxis=dict(range=[0, max(_sel_eff_by_elem.values(), default=1) * 1.3 + 0.1]),
    )
    col_d1.plotly_chart(fig_d_eff, use_container_width=True)

    # Comparativa de eficiencia de todos los hechizos (datos completos)
    _eff_global_all = _sp_df.groupby("Hechizo")["eff"].mean().reset_index()
    _eff_colors = [_spell_colors.get(sp, "#888") if sp == _spell_sel else "#cccccc"
                   for sp in _eff_global_all["Hechizo"]]
    fig_d_comp = go.Figure(go.Bar(
        x=_eff_global_all["Hechizo"], y=_eff_global_all["eff"],
        marker_color=_eff_colors,
        text=_eff_global_all["eff"].map("{:.3f}".format),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Eficiencia media: %{y:.3f}<extra></extra>",
    ))
    fig_d_comp.update_layout(
        title=f"Eficiencia media de todos los hechizos (destacado: {_spell_sel})",
        xaxis_title="Hechizo", yaxis_title="Kills / cast (media)",
        showlegend=False, height=320,
        yaxis=dict(range=[0, _eff_global_all["eff"].max() * 1.3]) if not _eff_global_all.empty else {},
    )
    col_d2.plotly_chart(fig_d_comp, use_container_width=True)

    # Caption dinámico con datos completos
    _best_elem_for_spell  = max(_sel_eff_by_elem, key=_sel_eff_by_elem.get) if _sel_eff_by_elem else "N/D"
    _worst_elem_for_spell = min(_sel_eff_by_elem, key=_sel_eff_by_elem.get) if _sel_eff_by_elem else "N/D"
    _rank = sorted(_glob_eff, key=_glob_eff.get, reverse=True).index(_spell_sel) + 1 if _spell_sel in _glob_eff else "?"
    st.caption(
        f"**{_spell_sel}** es el {_rank}º hechizo más eficiente en la selección actual ({_media_sel:.3f} kills/cast). "
        f"Funciona mejor con **{_best_elem_for_spell}** y peor con **{_worst_elem_for_spell}**."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4 — SALAS Y DIFICULTAD
# ══════════════════════════════════════════════════════════════════════════════
with tab_salas:
    st.header("Salas y Curva de Dificultad")
    st.caption(
        "Análisis de dificultad por sala y nivel. "
        "Se identifican cuellos de botella en el funnel y salas con mayor abandono."
    )

    # Filtro por elemento
    _salas_clean = clean[clean["playerElement"].isin(elem_sel)]
    _salas_rooms = rooms_c[rooms_c["sessionId"].isin(_salas_clean["sessionId"])]
    _salas_levels = levels_c[levels_c["sessionId"].isin(_salas_clean["sessionId"])]

    # ── Funnel dinámico ───────────────────────────────────────────────────────
    st.subheader("Funnel de progresión por nivel")
    funnel_df = _salas_levels.groupby("levelId")["sessionId"].nunique().reset_index()
    funnel_df.columns = ["Nivel", "Sesiones"]
    funnel_df = funnel_df.sort_values("Nivel").reset_index(drop=True)
    funnel_df["% llegada"] = funnel_df["Sesiones"] / len(_salas_clean)
    funnel_df["Caída"] = funnel_df["% llegada"] - funnel_df["% llegada"].shift(-1).fillna(funnel_df["% llegada"])

    fig_funnel = go.Figure()
    fig_funnel.add_trace(go.Bar(
        x=funnel_df["Nivel"], y=funnel_df["Sesiones"],
        marker_color=["#D4A017" if v == funnel_df["Caída"].max() else "#1a8fff"
                      for v in funnel_df["Caída"]],
        text=funnel_df.apply(lambda r: f"{r['Sesiones']} ({r['% llegada']:.0%})", axis=1),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Sesiones: %{y}<br>% llegada: %{text}<extra></extra>",
    ))
    fig_funnel.update_layout(
        xaxis_title="Nivel", yaxis_title="Sesiones que llegan",
        showlegend=False, height=340,
        yaxis=dict(range=[0, funnel_df["Sesiones"].max() * 1.3]),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)
    _cuello = funnel_df.loc[funnel_df["Caída"].idxmax()]
    _cuello_next = funnel_df[funnel_df["Nivel"] > _cuello["Nivel"]]
    _cuello_next_name = _cuello_next.iloc[0]["Nivel"] if not _cuello_next.empty else "el siguiente nivel"
    st.caption(
        f"Barra dorada = mayor caída entre niveles consecutivos. "
        f"**{_cuello['Nivel']}** es el cuello de botella principal — "
        f"{_cuello['Caída']:.0%} de las sesiones que llegan a {_cuello['Nivel']} no alcanzan {_cuello_next_name} "
        f"({int(_cuello['Sesiones'])} → {int(_cuello_next.iloc[0]['Sesiones']) if not _cuello_next.empty else '0'} sesiones)."
    )

    st.divider()

    # ── Curva de dificultad real vs esperada ──────────────────────────────────
    st.subheader("Curva de dificultad por nivel")
    st.caption("Muertes medias por nivel — picos indican niveles con dificultad elevada.")
    _diff_df = _salas_levels.groupby("levelId")["deaths"].mean().reset_index()
    _diff_df.columns = ["Nivel", "Muertes medias"]
    _diff_df = _diff_df.sort_values("Nivel").reset_index(drop=True)
    fig_diff = go.Figure()
    fig_diff.add_trace(go.Scatter(
        x=_diff_df["Nivel"], y=_diff_df["Muertes medias"],
        mode="lines+markers+text",
        line=dict(color="#1a8fff", width=2),
        marker=dict(size=8),
        text=_diff_df["Muertes medias"].round(1),
        textposition="top center",
        hovertemplate="<b>%{x}</b><br>Muertes medias: %{y:.2f}<extra></extra>",
    ))
    fig_diff.update_layout(
        xaxis_title="Nivel", yaxis_title="Muertes medias",
        showlegend=False, height=340,
        yaxis=dict(range=[0.1, 1]),
    )
    st.plotly_chart(fig_diff, use_container_width=True)
    _pico = _diff_df.loc[_diff_df["Muertes medias"].idxmax()]
    _pico_obs = (
        " Es el único nivel al que llegan todos los jugadores (incluyendo los menos experimentados) — "
        "las muertes altas aquí reflejan curva de aprendizaje, no necesariamente un spike de dificultad intencionado."
        if _pico["Nivel"] == "Level1" else
        " Spike de dificultad en un nivel avanzado — los jugadores que llegan aquí ya han superado los anteriores, "
        "lo que hace el dato más representativo."
    )
    st.caption(
        f"**{_pico['Nivel']}** acumula más muertes medias ({_pico['Muertes medias']:.2f} muertes/sesión).{_pico_obs}"
    )

    st.divider()

    # ── Salas con mayor dificultad ────────────────────────────────────────────
    st.subheader("Salas con mayor dificultad")
    st.caption(
        "Score de dificultad = muertes + daño recibido + tiempo por kill + tiempo en sala "
        "(cada componente normalizado 0-1, máximo 4.0). "
        "Tiempo por kill mide cuánto cuesta eliminar a cada enemigo; "
        "tiempo en sala mide la duración total del combate."
    )
    _room_agg = _salas_rooms.groupby(["levelId","roomId"]).agg(
        muertes=("deaths", "mean"),
        daño=("damageTaken", "mean"),
        tiempo=("timeSecs", "mean"),
        kills=("total_kills_room", "mean"),
        sesiones=("sessionId", "count"),
    ).reset_index()
    _room_agg["tiempo_por_kill"] = _room_agg.apply(
        lambda r: r["tiempo"] / r["kills"] if r["kills"] > 0 else r["tiempo"], axis=1
    )
    _norm = lambda s: s / (s.max() + 1e-9)
    _room_agg["score"] = (_norm(_room_agg["muertes"]) +
                          _norm(_room_agg["daño"]) +
                          _norm(_room_agg["tiempo_por_kill"]) +
                          _norm(_room_agg["tiempo"]))
    _room_top = _room_agg.sort_values("score", ascending=False).head(10)
    _room_top["Sala"] = _room_top["levelId"] + " / " + _room_top["roomId"]
    fig_rooms = go.Figure(go.Bar(
        x=_room_top["score"].round(3),
        y=_room_top["Sala"],
        orientation="h",
        marker_color="#1a8fff",
        text=_room_top["score"].round(2),
        textposition="outside",
        customdata=np.stack([
            _room_top["muertes"].round(2),
            _room_top["daño"].round(0),
            _room_top["tiempo_por_kill"].round(1),
            _room_top["tiempo"].round(0),
            _room_top["kills"].round(1),
        ], axis=1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Score: %{x:.3f}<br>"
            "Muertes medias: %{customdata[0]}<br>"
            "Daño recibido: %{customdata[1]} pts<br>"
            "Tiempo por kill: %{customdata[2]} s<br>"
            "Tiempo en sala: %{customdata[3]} s<br>"
            "Kills medias: %{customdata[4]}<extra></extra>"
        ),
    ))
    fig_rooms.update_layout(
        xaxis_title="Score de dificultad (0-4)", yaxis_title="",
        showlegend=False, height=380,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_rooms, use_container_width=True)
    _hardest = _room_top.iloc[0]
    _score_max = _room_agg["score"].max()
    st.caption(
        f"**{_hardest['Sala']}** es la sala más difícil (score {_hardest['score']:.2f} sobre {_score_max:.2f} máximo observado) — "
        f"media de {_hardest['muertes']:.1f} muertes, {_hardest['daño']:.0f} pts de daño, "
        f"{_hardest['tiempo_por_kill']:.1f} s por kill y {_hardest['tiempo']:.0f} s en sala. "
        "Cada componente se normaliza entre 0 y 1 respecto al resto de salas, por lo que el máximo real raramente alcanza 4.0."
    )

    st.divider()

    # ── Tiempo medio por sala ─────────────────────────────────────────────────
    st.subheader("Tiempo medio por sala")
    st.caption("Duración media de cada sala en segundos. Salas largas pueden indicar alta densidad de enemigos o dificultad elevada.")
    _room_time = _room_agg.sort_values("tiempo", ascending=False).head(10).copy()
    _room_time["Sala"] = _room_time["levelId"] + " / " + _room_time["roomId"]
    fig_time = go.Figure(go.Bar(
        x=_room_time["tiempo"].round(1),
        y=_room_time["Sala"],
        orientation="h",
        marker_color="#1a8fff",
        text=_room_time["tiempo"].round(0).astype(int).astype(str) + " s",
        textposition="outside",
        customdata=np.stack([
            _room_time["kills"].round(1),
            _room_time["tiempo_por_kill"].round(1),
            _room_time["muertes"].round(2),
        ], axis=1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Tiempo medio: %{x:.0f} s<br>"
            "Kills medias: %{customdata[0]}<br>"
            "Tiempo por kill: %{customdata[1]} s<br>"
            "Muertes medias: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig_time.update_layout(
        xaxis_title="Tiempo medio (s)", yaxis_title="",
        showlegend=False, height=380,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, _room_time["tiempo"].max() * 1.2]),
    )
    st.plotly_chart(fig_time, use_container_width=True)
    _longest = _room_time.iloc[0]
    _tpk_median = _room_agg["tiempo_por_kill"].median()
    _tpk_ratio = _longest["tiempo_por_kill"] / _tpk_median if _tpk_median > 0 else 1
    _tpk_obs = (
        f"Tiempo por kill {_longest['tiempo_por_kill']:.1f} s — {_tpk_ratio:.1f}× la mediana del resto de salas, "
        "sugiere alta resistencia de enemigos o patron de combate complejo."
        if _tpk_ratio > 2 else
        f"Tiempo por kill {_longest['tiempo_por_kill']:.1f} s — similar a la mediana ({_tpk_median:.1f} s), "
        "el tiempo elevado responde principalmente al volumen de enemigos."
    )
    st.caption(
        f"**{_longest['Sala']}** es la sala donde los jugadores pasan más tiempo ({_longest['tiempo']:.0f} s de media). "
        f"{_tpk_obs}"
    )

    st.divider()

    # ── Balance de enemigos (dinámico) ────────────────────────────────────────
    st.subheader("Balance de enemigos")
    st.caption("Kills, daño infligido y resistencia (blocked/kills) por tipo de enemigo.")

    _enemy_names = ["Barbarian1Hand","RangerBow","Knight1H","Knight2H","Rogue",
                    "Barbarian2Hand","RangerCrossbow","Barbarian2HBoss","Barbarian1HBoss",
                    "KnightBossGold","KnightBossBlack","Rogue_Hooded","RangerBowBoss"]
    # Spawns por nivel según diseño del juego — desglosado porque no todos los enemigos
    # aparecen en todos los niveles, y no todas las sesiones llegan al mismo nivel.
    # Bosses (10 instancias totales, 6 tipos): L1→Barbarian1HBoss+Barbarian2HBoss,
    # L2→KnightBossBlack+KnightBossGold, L3→Barbarian2HBoss+KnightBossGold+Rogue_Hooded,
    # L4→Barbarian2HBoss+KnightBossBlack+RangerBowBoss
    _SPAWNS_BY_LEVEL = {
        "Barbarian1Hand":  {"Level1": 4},
        "Barbarian2Hand":  {"Level1": 1, "Level2": 2, "Level3": 3, "Level4": 3},
        "Knight1H":        {"Level1": 2, "Level2": 3, "Level3": 2},
        "Knight2H":        {"Level1": 3, "Level2": 4, "Level3": 6, "Level4": 8},
        "Rogue":           {"Level1": 1, "Level2": 8, "Level3": 7, "Level4": 8},
        "RangerBow":       {"Level1": 2, "Level2": 4, "Level3": 4},
        "RangerCrossbow":  {"Level1": 1, "Level2": 2, "Level3": 6, "Level4": 6},
        "Barbarian1HBoss": {"Level1": 1},
        "Barbarian2HBoss": {"Level1": 1, "Level3": 1, "Level4": 1},
        "KnightBossBlack": {"Level2": 1, "Level4": 1},
        "KnightBossGold":  {"Level2": 1, "Level3": 1},
        "Rogue_Hooded":    {"Level3": 1},
        "RangerBowBoss":   {"Level4": 1},
    }

    # Sesiones filtradas que llegaron a cada nivel
    _sess_per_level = {
        lvl: _salas_levels[_salas_levels["levelId"] == lvl]["sessionId"].nunique()
        for lvl in ["Level1", "Level2", "Level3", "Level4"]
    }

    _enem_rows = []
    for en in _enemy_names:
        kills   = _salas_rooms[f"kills_{en}"].sum()   if f"kills_{en}"   in _salas_rooms.columns else 0
        damage  = _salas_rooms[f"damage_{en}"].sum()  if f"damage_{en}"  in _salas_rooms.columns else 0
        blocked = _salas_rooms[f"blocked_{en}"].sum() if f"blocked_{en}" in _salas_rooms.columns else 0
        # Spawns esperados = suma por nivel de (spawns_nivel × sesiones_que_llegaron_a_ese_nivel)
        lvl_spawns = _SPAWNS_BY_LEVEL.get(en, {})
        spawns_totales = sum(lvl_spawns.get(lvl, 0) * _sess_per_level.get(lvl, 0)
                             for lvl in ["Level1", "Level2", "Level3", "Level4"])
        _enem_rows.append({
            "Enemigo": en,
            "Kills": kills,
            "Spawns totales": spawns_totales,
            "Spawns/nivel (diseño)": sum(lvl_spawns.values()),
            "Tasa de kill (%)": round(kills / spawns_totales * 100, 1) if spawns_totales > 0 else 0,
            "Daño infligido": damage,
            "Blocked": blocked,
            "Ratio blocked/kills": round(blocked / kills, 2) if kills > 0 else 0,
        })
    _enem_df = pd.DataFrame(_enem_rows).sort_values("Kills", ascending=False)
    _enem_df_present = _enem_df[_enem_df["Spawns/nivel (diseño)"] > 0]

    # Kills totales vs tasa de kill normalizada por spawns
    fig_ekills = go.Figure()
    fig_ekills.add_trace(go.Bar(
        name="Kills totales",
        x=_enem_df_present["Enemigo"], y=_enem_df_present["Kills"],
        marker_color="#1a8fff",
        text=_enem_df_present["Kills"].astype(int),
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>Kills totales: %{y}"
            "<br>Spawns esperados: %{customdata[0]}"
            "<br>Tasa de kill: %{customdata[1]:.1f}%<extra></extra>"
        ),
        customdata=_enem_df_present[["Spawns totales", "Tasa de kill (%)"]].values,
        yaxis="y1",
    ))
    fig_ekills.add_trace(go.Scatter(
        name="Tasa de kill (%)",
        x=_enem_df_present["Enemigo"], y=_enem_df_present["Tasa de kill (%)"],
        mode="markers",
        marker=dict(color="#D4A017", size=10, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>Tasa de kill: %{y:.1f}%<extra></extra>",
        yaxis="y2",
    ))
    fig_ekills.update_layout(
        title="Kills por tipo de enemigo — totales y tasa respecto a spawns",
        xaxis_title="Enemigo", height=380,
        xaxis_tickangle=-30,
        yaxis=dict(title="Kills totales", range=[0, _enem_df_present["Kills"].max() * 1.35]),
        yaxis2=dict(title="Tasa de kill (%)", overlaying="y", side="right",
                    range=[0, 110], showgrid=False),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.08),
        legend_itemclick=False, legend_itemdoubleclick=False,
        margin=dict(r=120),
    )
    st.plotly_chart(fig_ekills, use_container_width=True)
    _most_killed  = _enem_df_present.sort_values("Kills", ascending=False).iloc[0]
    _most_rate    = _enem_df_present.sort_values("Tasa de kill (%)", ascending=False).iloc[0]
    _least_rate   = _enem_df_present[_enem_df_present["Tasa de kill (%)"] > 0].sort_values("Tasa de kill (%)").iloc[0]
    _enem_normal = _enem_df_present[~_enem_df_present["Enemigo"].str.contains("Boss|Hooded")]
    _enem_boss   = _enem_df_present[_enem_df_present["Enemigo"].str.contains("Boss|Hooded")]
    _hardest_normal = _enem_normal.sort_values("Tasa de kill (%)").iloc[0] if len(_enem_normal) else None
    _hardest_boss   = _enem_boss.sort_values("Tasa de kill (%)").iloc[0]   if len(_enem_boss)   else None
    _caption_parts = ["Barras = kills totales · Diamante dorado = tasa de kill ajustada por spawns esperados según sesiones que llegaron a cada nivel."]
    if _hardest_normal is not None:
        _caption_parts.append(
            f"Entre enemigos normales, **{_hardest_normal['Enemigo']}** tiene la tasa de kill más baja "
            f"({_hardest_normal['Tasa de kill (%)']:.1f}%)."
        )
    if _hardest_boss is not None:
        _b1h = _enem_boss[_enem_boss["Enemigo"] == "Barbarian1HBoss"]
        _b1h_rate = _b1h["Tasa de kill (%)"].iloc[0] if not _b1h.empty else None
        if _b1h_rate is not None and _b1h_rate < 70:
            _caption_parts.append(
                f"**Barbarian1HBoss** (boss Level1) tiene tasa {_b1h_rate:.0f}% — "
                "el boss introductorio debería ser el más fácil de eliminar; "
                "considerar reducir su HP o velocidad de ataque."
            )
        else:
            _caption_parts.append(
                f"Entre bosses, **{_hardest_boss['Enemigo']}** es el más difícil de eliminar "
                f"({_hardest_boss['Tasa de kill (%)']:.1f}%). "
                "Las tasas altas en niveles avanzados (≥90%) pueden reflejar sesgo de supervivencia — "
                "solo los jugadores más habilidosos llegan a Level4."
            )
    st.caption(" ".join(_caption_parts))

    # Daño infligido por enemigo — solo enemigos con spawns definidos (excluye damage_Unknown)
    _enem_dmg = _enem_df_present[_enem_df_present["Daño infligido"] > 0].sort_values("Daño infligido", ascending=False)
    fig_edmg = go.Figure(go.Bar(
        x=_enem_dmg["Enemigo"], y=_enem_dmg["Daño infligido"],
        marker_color="#1a8fff",
        text=_enem_dmg["Daño infligido"].astype(int),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Daño: %{y}<extra></extra>",
    ))
    fig_edmg.update_layout(
        title="Daño infligido al jugador por enemigo",
        xaxis_title="Enemigo", yaxis_title="Daño total",
        showlegend=False, height=320, xaxis_tickangle=-30,
        yaxis=dict(range=[0, _enem_dmg["Daño infligido"].max() * 1.3]),
    )
    st.plotly_chart(fig_edmg, use_container_width=True)
    _most_dmg = _enem_dmg.iloc[0]
    _second_dmg = _enem_dmg.iloc[1] if len(_enem_dmg) > 1 else None
    _dmg_ratio = (_most_dmg["Daño infligido"] / _second_dmg["Daño infligido"]) if _second_dmg is not None and _second_dmg["Daño infligido"] > 0 else None
    _dmg_note = (
        f" — inflige {_dmg_ratio:.1f}x más daño que el segundo enemigo más peligroso ({_second_dmg['Enemigo']}). "
        "Posible sobreescalado en DPS o frecuencia de ataque."
        if _dmg_ratio is not None and _dmg_ratio > 1.5 else " — candidato a revisión de escalado."
    )
    st.caption(f"**{_most_dmg['Enemigo']}** inflige más daño al jugador{_dmg_note}")

    # Ratio blocked/kills
    _enem_blk = _enem_df[_enem_df["Ratio blocked/kills"] > 0].sort_values("Ratio blocked/kills", ascending=False)
    fig_eblk = go.Figure(go.Bar(
        x=_enem_blk["Enemigo"], y=_enem_blk["Ratio blocked/kills"],
        marker_color="#1a8fff",
        text=_enem_blk["Ratio blocked/kills"].round(2),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Blocked/kills: %{y:.2f}<extra></extra>",
    ))
    fig_eblk.update_layout(
        title="Resistencia relativa (blocked / kills)",
        xaxis_title="Enemigo", yaxis_title="Ratio",
        showlegend=False, height=320, xaxis_tickangle=-30,
        yaxis=dict(range=[0, _enem_blk["Ratio blocked/kills"].max() * 1.4]),
    )
    st.plotly_chart(fig_eblk, use_container_width=True)
    _most_blk = _enem_blk.iloc[0]
    _blk_context = ""
    if "KnightBoss" in _most_blk["Enemigo"] or "Knight" in _most_blk["Enemigo"]:
        _blk_context = (
            " Los Knights tienen alto blockChance por diseño — bloquean muchos ataques pero "
            "el jugador los elimina igualmente con persistencia. Un ratio alto aquí no implica sobreescalado, "
            "sino que requiere más ataques para derrotarlo."
        )
    else:
        _blk_context = " — candidato a revisión si la tasa de kill es baja."
    st.caption(
        f"**{_most_blk['Enemigo']}** tiene la mayor resistencia relativa "
        f"({_most_blk['Ratio blocked/kills']:.2f} ataques bloqueados por kill).{_blk_context}"
    )



# ══════════════════════════════════════════════════════════════════════════════
# 5 — MODELO PREDICTIVO
# ══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    st.header("Modelo Predictivo de Victoria")
    st.caption(
        "Clasificación binaria: predecir si una sesión acaba en victoria. "
        f"Validación **Leave-One-Out** (LOO) con n={n_clean} sesiones — resultados orientativos."
    )

    # AUC métricas
    y_real = ml["y_real"].values
    auc_lr  = roc_auc_score(y_real, ml["proba_lr"])
    auc_rf  = roc_auc_score(y_real, ml["proba_rf"])
    auc_xgb = roc_auc_score(y_real, ml["proba_xgb"])
    acc     = (ml["y_real"] == ml["pred_best"]).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regresión Logística — AUC", f"{auc_lr:.3f}",
              help="AUC = 0.5 → aleatorio, 1.0 → perfecto.")
    c2.metric("Random Forest — AUC", f"{auc_rf:.3f}")
    c3.metric("XGBoost — AUC", f"{auc_xgb:.3f}")
    c4.metric("Accuracy (mejor modelo)", f"{acc:.0%}",
              help="Porcentaje de predicciones correctas del mejor modelo (XGB).")

    _n_ml      = len(ml)
    _n_vic     = int(ml["y_real"].sum())
    _n_der     = _n_ml - _n_vic
    st.info(
        f"⚠️ Con n={_n_ml} sesiones y fuerte desbalance ({_n_vic} victorias / {_n_der} derrotas), "
        "los AUC altos deben interpretarse con cautela. "
        "El modelo aprende a distinguir patrones reales pero no es generalizable sin más datos. "
        "Se necesitarían al menos 200-300 sesiones para resultados robustos."
    )

    st.divider()

    # Curvas ROC + feature importance (interactivos)
    col_roc, col_imp = st.columns(2)

    # Curvas ROC en Plotly
    from sklearn.metrics import roc_curve
    fig_roc_plt = go.Figure()
    fig_roc_plt.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color="gray", dash="dash", width=1))
    for name, proba, color in [
        ("Logistic Regression", ml["proba_lr"],  "#f39c12"),
        ("Random Forest",       ml["proba_rf"],  "#2ecc71"),
        ("XGBoost",             ml["proba_xgb"], "#1a8fff"),
    ]:
        fpr, tpr, _ = roc_curve(ml["y_real"], proba)
        auc_val = roc_auc_score(ml["y_real"], proba)
        fig_roc_plt.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_val:.3f})",
            line=dict(color=color, width=2),
            hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>",
        ))
    fig_roc_plt.update_layout(
        title="Curvas ROC — Comparativa de Modelos",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=380,
        legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        legend_itemclick=False, legend_itemdoubleclick=False,
    )
    col_roc.plotly_chart(fig_roc_plt, use_container_width=True)
    col_roc.caption("Curvas ROC — validación Leave-One-Out.")

    # Feature importance RF en Plotly (top 10)
    _fi_top = fi.nlargest(10, "RF").sort_values("RF")
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        y=_fi_top["feature"], x=_fi_top["RF"],
        orientation="h", name="Random Forest",
        marker_color="#1a8fff",
        hovertemplate="<b>%{y}</b><br>RF: %{x:.3f}<extra></extra>",
    ))
    fig_fi.add_trace(go.Bar(
        y=_fi_top["feature"], x=fi.set_index("feature").loc[_fi_top["feature"], "XGB"],
        orientation="h", name="XGBoost",
        marker_color="#D4A017",
        hovertemplate="<b>%{y}</b><br>XGB: %{x:.3f}<extra></extra>",
    ))
    fig_fi.update_layout(
        title="Top 10 Features más Predictivas",
        xaxis_title="Importancia", barmode="group", height=380,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        legend_itemclick=False, legend_itemdoubleclick=False,
        margin=dict(r=100),
    )
    col_imp.plotly_chart(fig_fi, use_container_width=True)
    _top1_rf  = fi.nlargest(1, "RF").iloc[0]["feature"]
    _top1_xgb = fi.nlargest(1, "XGB").iloc[0]["feature"]
    _top3_rf  = ", ".join(fi.nlargest(3, "RF")["feature"].tolist())
    _elem_feats = [f for f in fi["feature"] if "playerElement" in f or f in ["Fire","Water","Earth","Wind"]]
    _elem_note  = "El elemento elegido no aparece entre los top features." if not _elem_feats else ""
    col_imp.caption(
        f"**RF**: predictor dominante → `{_top1_rf}` · "
        f"**XGB**: predictor dominante → `{_top1_xgb}`. "
        f"Top 3 RF: {_top3_rf}. {_elem_note}"
    )

    st.divider()

    # Variables más importantes — tabla de correlaciones
    st.subheader("Variables estadísticamente asociadas a la victoria")
    st.caption("Correlación Spearman entre cada variable y ganar la partida. Solo variables con p<0.05.")
    _output_vars = ["levelsCompleted", "totalRooms", "completion_rate", "level_finish_rate", "game_completion"]
    spear = stat[
        (stat["test"] == "Spearman") &
        (stat["variable"].str.contains("isVictory")) &
        (stat["p_value"] < 0.05)
    ][["variable","effect_size","p_value","interpretation"]].copy()
    spear["variable"] = spear["variable"].str.replace(" vs isVictory", "", regex=False)
    spear = spear[~spear["variable"].isin(_output_vars)]
    spear = spear.sort_values("effect_size", ascending=False)
    spear.columns = ["Variable","Correlación (rho)","p-valor","Interpretación"]
    spear["Correlación (rho)"] = spear["Correlación (rho)"].map("{:.3f}".format)
    spear["p-valor"]           = spear["p-valor"].map("{:.4f}".format)
    st.dataframe(spear, use_container_width=True, hide_index=True)

    st.divider()

    # Clustering
    st.subheader("Perfiles de jugador — K-Means clustering")
    # Silhouette dinámico para no quedar desactualizado si cambian los datos o features
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        _feats_sil = [c for c in ["kd_ratio","kills_per_min","time_per_level","cast_accuracy","totalCast","totalDeaths"] if c in cls.columns]
        _X_sil = StandardScaler().fit_transform(cls[_feats_sil].fillna(0))
        _sil = silhouette_score(_X_sil, cls["cluster"]) if cls["cluster"].nunique() > 1 else 0.0
    except Exception:
        _sil = float("nan")
    _k_real = int(cls["cluster"].nunique())
    _n_cls = int(len(cls))
    st.caption(
        f"Agrupación exploratoria (K-Means, k={_k_real}) de sesiones por métricas de combate. "
        f"Con n={_n_cls} sesiones y Silhouette={_sil:.3f} los clusters son orientativos, no definitivos. "
        "El objetivo es identificar si existe un perfil de jugador ganador independiente del elemento elegido."
    )

    # Clusters ganadores (tasa victoria > 50%)
    _win_clusters = cls.groupby("cluster")["isVictory"].mean()
    _win_cluster_ids = _win_clusters[_win_clusters > 0.5].index.tolist()
    _win_str = ", ".join([f"Cluster {c}" for c in _win_cluster_ids]) if _win_cluster_ids else "ninguno"

    # Tabla de stats por cluster
    cluster_stats = cls.groupby("cluster").agg(
        n=("sessionId","count"),
        victoria=("isVictory","mean"),
        kd_ratio=("kd_ratio","mean"),
        kills_min=("kills_per_min","mean"),
        precision=("cast_accuracy","mean"),
    ).reset_index().round(3)
    cluster_stats["victoria"] = cluster_stats["victoria"].map("{:.0%}".format)
    cluster_stats["kd_ratio"] = cluster_stats["kd_ratio"].map("{:.1f}".format)
    cluster_stats["kills_min"] = cluster_stats["kills_min"].map("{:.2f}".format)
    cluster_stats["precision"] = cluster_stats["precision"].map("{:.1%}".format)
    cluster_stats.columns = ["Cluster","n sesiones","Tasa victoria","K/D","Kills/min","Precisión"]
    st.dataframe(cluster_stats, use_container_width=True, hide_index=True, height=260)
    st.caption(
        f"Los jugadores se agrupan por estilo de combate, no por elemento. "
        f"**{_win_str}** concentran las sesiones ganadoras con K/D notablemente superior al resto."
    )

    st.divider()

    # Error por elemento
    st.subheader("Tasa de error del modelo por elemento")
    st.caption(
        "Un error alto en un elemento puede indicar que sus patrones de juego no se capturan bien "
        "con las métricas actuales (kd_ratio, cast_accuracy). "
        "Earth usa Stun (escalado con daño) como mecánica principal — su valor no aparece en kills/min, "
        "lo que puede confundir al modelo."
    )
    _err_df = ml.copy()
    _err_df["error"] = (_err_df["y_real"] != _err_df["pred_best"]).astype(int)
    _err_by_elem = (
        _err_df.groupby("playerElement")["error"]
        .agg(n_sesiones="count", n_errores="sum")
        .assign(tasa_error=lambda d: d["n_errores"] / d["n_sesiones"])
        .reset_index()
        .sort_values("tasa_error", ascending=False)
    )
    fig_err_plt = go.Figure(go.Bar(
        x=_err_by_elem["playerElement"],
        y=_err_by_elem["tasa_error"],
        marker_color=[PALETTE.get(e, "#888") for e in _err_by_elem["playerElement"]],
        text=_err_by_elem.apply(lambda r: f"{r['tasa_error']:.0%} ({int(r['n_errores'])}/{int(r['n_sesiones'])})", axis=1),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Tasa error: %{y:.0%}<br>Errores: %{customdata[0]} / %{customdata[1]} sesiones<extra></extra>",
        customdata=_err_by_elem[["n_errores","n_sesiones"]].values,
    ))
    fig_err_plt.update_layout(
        xaxis_title="Elemento", yaxis_title="Tasa de error",
        yaxis=dict(tickformat=".0%", range=[0, _err_by_elem["tasa_error"].max() * 1.4]),
        showlegend=False, height=320,
    )
    st.plotly_chart(fig_err_plt, use_container_width=True)
    _worst_elem = _err_by_elem.iloc[0]
    _zero_error = _err_by_elem[_err_by_elem["tasa_error"] == 0]["playerElement"].tolist()
    _zero_str = ", ".join(f"**{e}**" for e in _zero_error)
    st.caption(
        f"**{_worst_elem['playerElement']}** tiene la mayor tasa de error "
        f"({_worst_elem['tasa_error']:.0%} — {int(_worst_elem['n_errores'])} de {int(_worst_elem['n_sesiones'])} sesiones mal predichas). "
        + (f"{_zero_str} son predichos con 0% de error." if _zero_str else "")
    )

    st.divider()

    # Tabla de predicciones filtrable
    st.subheader("Predicciones individuales por sesión")
    ml_show = ml[ml["playerElement"].isin(elem_sel)].copy()
    ml_show["correcto"]    = ml_show["y_real"] == ml_show["pred_best"]
    ml_show["proba_mejor"] = ml_show["proba_xgb"].map("{:.3f}".format)
    ml_show["real"]        = ml_show["y_real"].map({1: "✅ Victoria", 0: "❌ Derrota"})
    ml_show["prediccion"]  = ml_show["pred_best"].map({1: "✅ Victoria", 0: "❌ Derrota"})
    ml_show["resultado"]   = ml_show["correcto"].map({True: "✓ Correcto", False: "✗ Error"})
    ml_display = ml_show[["sessionId","playerElement","real","prediccion","proba_mejor","resultado"]].rename(columns={
        "sessionId":"Sesión","playerElement":"Elemento",
        "real":"Resultado real","prediccion":"Predicción",
        "proba_mejor":"Prob. victoria (XGB)","resultado":"Acierto",
    }).reset_index(drop=True)
    st.dataframe(
        ml_display.style.apply(
            lambda row: [
                "" if row["Acierto"] == "✓ Correcto" else "background-color: #3a1a1a"
            ] * len(row),
            axis=1,
        ),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # Conclusión del modelo
    _best_auc  = max(auc_lr, auc_rf, auc_xgb)
    _best_name = {auc_lr: "Regresión Logística", auc_rf: "Random Forest", auc_xgb: "XGBoost"}[_best_auc]
    _n_errors  = int((~ml_show["correcto"]).sum())
    _n_total   = len(ml_show)
    st.markdown(
        f"**Conclusión:** **{_best_name}** fue el mejor modelo (AUC = {_best_auc:.3f}, accuracy = {acc:.0%} "
        f"sobre {_n_total} sesiones, {_n_errors} error{'es' if _n_errors != 1 else ''}). "
        f"Los predictores más relevantes son `{_top3_rf}` — "
        "reflejan la habilidad global del jugador más que el elemento elegido."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6 — FEEDBACK DE JUGADORES
# ══════════════════════════════════════════════════════════════════════════════
with tab_feedback:
    st.header("Feedback de Jugadores")
    st.caption(
        "Análisis de sentimiento y temas sobre los comentarios recogidos en sesiones de juego. "
        "Muestra cualitativa reducida — los resultados son orientativos, no estadísticamente significativos."
    )

    nlp_all = nlp[~nlp["spam_excluido"] & nlp["sentiment_score"].notna()]
    if nlp_all.empty:
        st.info("No hay comentarios válidos disponibles.")
    else:
        col_nlp1, _gnlp, col_nlp2 = st.columns([10, 1, 10])

        # Sentimiento por elemento
        _sent_by_elem = nlp_all.groupby("playerElement")["sentiment_score"].mean().reset_index()
        fig_sent = px.bar(
            _sent_by_elem,
            x="playerElement", y="sentiment_score",
            color="playerElement", color_discrete_map=PALETTE,
            text=_sent_by_elem["sentiment_score"].round(2),
            title="Sentimiento medio por elemento (0=negativo, 1=positivo)",
            labels={"playerElement": "Elemento", "sentiment_score": "Puntuación de sentimiento"},
        )
        fig_sent.update_traces(textposition="outside")
        fig_sent.update_yaxes(range=[0, 1])
        fig_sent.update_layout(showlegend=False, height=320)
        col_nlp1.plotly_chart(fig_sent, use_container_width=True)
        _best_sent  = _sent_by_elem.loc[_sent_by_elem["sentiment_score"].idxmax()]
        _worst_sent = _sent_by_elem.loc[_sent_by_elem["sentiment_score"].idxmin()]
        col_nlp1.caption(
            f"**{_best_sent['playerElement']}** tiene el sentimiento más positivo ({_best_sent['sentiment_score']:.2f}). "
            f"**{_worst_sent['playerElement']}** el más negativo ({_worst_sent['sentiment_score']:.2f})."
        )

        # Temas por elemento
        fig_topic = px.histogram(
            nlp_all, x="topic_name", color="playerElement",
            color_discrete_map=PALETTE,
            title="Temas detectados en comentarios",
            labels={"topic_name": "Tema", "count": "nº comentarios"},
            barmode="stack",
        )
        fig_topic.update_layout(
            height=320, xaxis_tickangle=-20, legend_title_text="Elemento",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            legend_itemclick=False, legend_itemdoubleclick=False,
            margin=dict(r=100),
        )
        col_nlp2.plotly_chart(fig_topic, use_container_width=True)
        _top_topic = nlp_all["topic_name"].value_counts().idxmax()
        _top_topic_n = nlp_all["topic_name"].value_counts().max()
        col_nlp2.caption(f"Tema más frecuente: **{_top_topic}** ({_top_topic_n} comentarios).")

        st.divider()

        # Tabla de comentarios
        st.subheader("Comentarios completos")
        merged_nlp = nlp_all.merge(
            sessions[["sessionId", "playerComment"]].dropna(), on="sessionId", how="left"
        )
        st.dataframe(
            merged_nlp[["playerElement", "sentiment_stars", "topic_name", "playerComment"]].rename(columns={
                "playerElement": "Elemento", "sentiment_stars": "Estrellas",
                "topic_name": "Tema", "playerComment": "Comentario",
            }),
            use_container_width=True, hide_index=True,
        )

        st.divider()

        # Conclusión
        _n_comments = len(nlp_all)
        _n_pos = (nlp_all["sentiment_score"] >= 0.6).sum()
        _n_neg = (nlp_all["sentiment_score"] <= 0.4).sum()
        _wind_neg = nlp_all[(nlp_all["playerElement"] == "Wind") & (nlp_all["sentiment_score"] <= 0.4)]
        _wind_note = (
            " Los comentarios negativos de **Wind** reportan un bug: arqueros que persiguen al jugador "
            "pero no atacan — prioritario para investigar en la IA de enemigos a distancia."
            if not _wind_neg.empty else ""
        )
        st.markdown(
            f"**Conclusión:** Se analizaron **{_n_comments} comentarios** válidos. "
            f"{_n_pos} son positivos (score ≥ 0.6) y {_n_neg} negativos (score ≤ 0.4).{_wind_note} "
            f"El sentimiento es orientativo dado el tamaño de la muestra — "
            f"se recomienda recoger al menos 50 comentarios para análisis significativo."
        )

# ══════════════════════════════════════════════════════════════════════════════
# 7 — RECOMENDACIONES
# ══════════════════════════════════════════════════════════════════════════════
with tab_recs:
    st.header("Recomendaciones de Balance")
    st.caption(
        "Tabla de acciones priorizadas para el diseñador del juego. "
        "**Alta** = evidencia estadística o bug confirmado. "
        "**Media** = tendencia observada en EDA. "
        "**Baja** = feedback cualitativo del jugador."
    )

    # Resumen visual por prioridad
    prio_counts = recs["priority"].value_counts().reindex(["Alta","Media","Baja"], fill_value=0)
    _n_done    = int(recs["arreglado"].sum()) if "arreglado" in recs.columns else 0
    _n_pending = len(recs) - _n_done
    ca, cm, cb, cd = st.columns(4)
    ca.metric("🔴 Alta prioridad",   prio_counts.get("Alta",  0), help="Acción inmediata")
    cm.metric("🟡 Media prioridad",  prio_counts.get("Media", 0), help="Planificar a corto plazo")
    cb.metric("🟢 Baja prioridad",   prio_counts.get("Baja",  0), help="Mejora cualitativa")
    cd.metric("✅ Completadas",       _n_done, help=f"{_n_pending} pendientes de {len(recs)} totales")

    if _n_done > 0:
        st.info(
            f"✅ **{_n_done} de {len(recs)} recomendaciones completadas** ({_n_done/len(recs):.0%}). "
            f"Quedan **{_n_pending} pendientes**."
        )

    st.divider()

    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)
    prio_sel = col_f1.selectbox("Prioridad", ["Todas","Alta","Media","Baja"])
    area_sel = col_f2.selectbox("Área",
        ["Todas"] + sorted(recs["area"].dropna().unique().tolist()))
    elem_rec = col_f3.selectbox("Elemento",
        ["Todos"] + sorted(recs["element"].dropna().unique().tolist()))

    recs_f = recs.copy()
    if prio_sel  != "Todas": recs_f = recs_f[recs_f["priority"] == prio_sel]
    if area_sel  != "Todas": recs_f = recs_f[recs_f["area"]     == area_sel]
    if elem_rec  != "Todos": recs_f = recs_f[recs_f["element"]  == elem_rec]

    st.caption(f"Mostrando **{len(recs_f)}** de {len(recs)} recomendaciones")

    _PRIO_EMOJI = {"Alta": "🔴 Alta", "Media": "🟡 Media", "Baja": "🟢 Baja"}
    _AREA_LABEL = {
        "spell_balance": "Balance hechizos", "difficulty": "Dificultad",
        "element_balance": "Balance elementos", "telemetry": "Telemetría / Bugs",
        "sample_size": "Tamaño de muestra", "enemy_balance": "Balance enemigos",
        "ux": "UX / Controles",
    }

    cols_rec_full = ["priority","area","element","recommendation","evidence","arreglado"]
    cols_rec_full = [c for c in cols_rec_full if c in recs_f.columns]
    _recs_display = recs_f[cols_rec_full].copy()
    _recs_display["priority"] = _recs_display["priority"].map(_PRIO_EMOJI).fillna(_recs_display["priority"])
    _recs_display["area"]     = _recs_display["area"].map(_AREA_LABEL).fillna(_recs_display["area"])
    if "arreglado" not in _recs_display.columns:
        _recs_display["arreglado"] = False
    _recs_display["arreglado"] = _recs_display["arreglado"].fillna(False).astype(bool)

    rename_rec = {
        "priority": "Prioridad", "area": "Área", "element": "Elemento",
        "recommendation": "Recomendación", "evidence": "Evidencia", "arreglado": "Arreglado",
    }

    # En modo no-admin la columna "Arreglado" tambien queda deshabilitada para evitar persistencia anonima
    _is_admin = st.session_state.get("is_admin", False)
    _disabled_cols = [c for c in rename_rec.values() if c != "Arreglado"]
    if not _is_admin:
        _disabled_cols.append("Arreglado")

    _edited = st.data_editor(
        _recs_display.rename(columns=rename_rec),
        use_container_width=True, hide_index=True, height=600,
        column_config={
            "Prioridad":     st.column_config.TextColumn("Prioridad",    width=80),
            "Área":          st.column_config.TextColumn("Área",         width=130),
            "Elemento":      st.column_config.TextColumn("Elemento",     width=80),
            "Recomendación": st.column_config.TextColumn("Recomendación",width=420),
            "Evidencia":     st.column_config.TextColumn("Evidencia",    width=280),
            "Arreglado":     st.column_config.CheckboxColumn(
                "✅", help="Marca cuando esta recomendación haya sido implementada",
                default=False, width=50,
            ),
        },
        disabled=_disabled_cols,
        key="recs_editor",
    )

    # Guardar cambios en el CSV solo si el usuario es admin y modifico algun checkbox
    if _is_admin:
        _edited_back = _edited.rename(columns={v: k for k, v in rename_rec.items()})
        if not _edited_back["arreglado"].equals(_recs_display["arreglado"]):
            _recs_full = recs.copy()
            _recs_full.loc[_recs_display.index, "arreglado"] = _edited_back["arreglado"].values
            _recs_full.to_csv(EXPORTS / "balance_recommendations.csv", index=False)
            st.toast("Cambios guardados en balance_recommendations.csv", icon="✅")

    st.divider()

    # Gráfico distribución por área
    area_counts = recs.groupby(["area","priority"]).size().reset_index(name="n")
    area_map = {
        "spell_balance":"Balance hechizos","difficulty":"Dificultad",
        "element_balance":"Balance elementos","telemetry":"Telemetría / Bugs",
        "sample_size":"Tamaño de muestra",
    }
    area_counts["area"] = area_counts["area"].map(area_map).fillna(area_counts["area"])
    fig_area = px.bar(
        area_counts, x="area", y="n", color="priority",
        color_discrete_map={"Alta":"#e74c3c","Media":"#f39c12","Baja":"#27ae60"},
        title="Recomendaciones por área y prioridad",
        labels={"area":"Área de mejora","n":"Nº recomendaciones","priority":"Prioridad"},
        barmode="stack",
    )
    fig_area.update_layout(
        height=360, xaxis_tickangle=-15,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        legend_itemclick=False, legend_itemdoubleclick=False,
        margin=dict(r=100),
    )
    st.plotly_chart(fig_area, use_container_width=True)

    st.divider()

    # Conclusión
    _n_alta = prio_counts.get("Alta", 0)
    _top_area = recs["area"].value_counts().idxmax()
    _top_area_es = area_map.get(_top_area, _top_area)
    st.markdown(
        f"**Conclusión:** Se identificaron **{len(recs)} recomendaciones** de balance, "
        f"de las cuales **{_n_alta} son de alta prioridad** y requieren acción inmediata. "
        f"El área con más recomendaciones es **{_top_area_es}**. "
        "Las conclusiones se basan en evidencia estadística (Fase 3), modelos predictivos (Fase 4) "
        "y análisis cualitativo de comentarios (Fase 4b)."
    )

    st.divider()

    # ── Recomendaciones adicionales con IA ────────────────────────────────────
    st.subheader("Recomendaciones adicionales con IA")
    st.caption(
        "Gemini analiza los datos de Puntuación Global, hechizos y salas para identificar "
        "oportunidades de mejora adicionales no cubiertas por el análisis automático."
    )

    if "ai_recs_generadas" not in st.session_state:
        st.session_state.ai_recs_generadas = None

    if st.button("Generar recomendaciones IA", key="btn_ai_recs", type="primary"):
        with st.spinner("Gemini analizando los datos..."):
            try:
                from analysis import generate_ai_recommendations, compute_killedwith_efficiency
                try:
                    _spell_eff = compute_killedwith_efficiency(rooms_c, clean)
                except Exception:
                    _spell_eff = None
                _rooms_agg_ai = rooms_c.groupby(["levelId", "roomId"]).agg(
                    deaths_mean=("deaths", "mean"),
                    damage_mean=("damageTaken", "mean"),
                    n_visits=("sessionId", "count"),
                ).reset_index()
                _nuevas = generate_ai_recommendations(
                    bi, stat, _spell_eff, _rooms_agg_ai, recs
                )
                st.session_state.ai_recs_generadas = _nuevas
            except Exception as _e:
                st.error(f"Error al generar recomendaciones con IA: {_e}")
                st.session_state.ai_recs_generadas = None

    if st.session_state.ai_recs_generadas:
        _nuevas = st.session_state.ai_recs_generadas
        st.caption(f"Gemini ha generado {len(_nuevas)} recomendaciones. Revisa y edita antes de guardar.")

        _PRIO_EMOJI_IA = {"Alta": "Alta", "Media": "Media", "Baja": "Baja"}
        _df_nuevas = pd.DataFrame(_nuevas)
        # Asegurar columnas mínimas
        for _col in ["element", "area", "priority", "finding", "recommendation", "evidence"]:
            if _col not in _df_nuevas.columns:
                _df_nuevas[_col] = ""
        _df_nuevas["guardar"] = True

        _edited_nuevas = st.data_editor(
            _df_nuevas[["guardar", "priority", "element", "area", "finding", "recommendation", "evidence"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "guardar":        st.column_config.CheckboxColumn("Guardar", default=True, width=70),
                "priority":       st.column_config.SelectboxColumn("Prioridad", options=["Alta", "Media", "Baja"], width=90),
                "element":        st.column_config.TextColumn("Elemento", width=100),
                "area":           st.column_config.SelectboxColumn("Area", options=[
                    "spell_balance", "element_balance", "difficulty",
                    "ux", "telemetry", "enemy_balance", "sample_size", "nlp"
                ], width=130),
                "finding":        st.column_config.TextColumn("Hallazgo", width=260),
                "recommendation": st.column_config.TextColumn("Recomendacion", width=280),
                "evidence":       st.column_config.TextColumn("Evidencia", width=220),
            },
            key="editor_ai_recs",
        )

        _a_guardar = _edited_nuevas[_edited_nuevas["guardar"] == True].drop(columns=["guardar"])
        st.caption(f"{len(_a_guardar)} de {len(_edited_nuevas)} seleccionadas para guardar.")

        # El guardado modifica el CSV compartido — restringido a admin para evitar spam en la demo pública
        _can_save = st.session_state.get("is_admin", False)
        if not _can_save:
            st.info("🔒 Guardar requiere modo administrador (sidebar). En esta demo pública solo el autor puede modificar el CSV compartido.")

        if st.button("Guardar en CSV", key="btn_guardar_ai_recs", type="primary",
                     disabled=len(_a_guardar) == 0 or not _can_save):
            _a_guardar["arreglado"] = False
            _recs_actualizado = pd.concat(
                [recs, _a_guardar], ignore_index=True
            )
            _recs_actualizado.to_csv(EXPORTS / "balance_recommendations.csv", index=False)
            st.success(f"{len(_a_guardar)} recomendaciones guardadas en balance_recommendations.csv.")
            st.session_state.ai_recs_generadas = None
            st.cache_data.clear()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 8 — ASISTENTE IA
# ══════════════════════════════════════════════════════════════════════════════
with tab_ia:
    st.header("🤖 Asistente IA de Balance")
    st.caption(
        "Agente inteligente con acceso a todos los datos del juego. "
        "Puedes preguntarle sobre elementos, hechizos, enemigos, dificultad, estadísticas o pedir recomendaciones."
    )

    # Inicializar historial en session_state
    if "ia_historial" not in st.session_state:
        st.session_state.ia_historial = []
    if "ia_mensajes_ui" not in st.session_state:
        st.session_state.ia_mensajes_ui = []

    # Preguntas de ejemplo
    st.markdown("**Ejemplos de preguntas:**")
    _ejemplos = [
        "Dame un resumen del estado actual del balance del juego",
        "¿Por qué pierde Water? ¿Qué le cambiarías?",
        "¿Qué enemigo debería nerfear primero?",
        "¿Qué sala es la más difícil y por qué?",
        "Analiza el balance de hechizos y dime cuál está roto",
        "¿Qué elemento es el más equilibrado según los datos?",
    ]
    _cols_ej = st.columns(3)
    for _i, _ej in enumerate(_ejemplos):
        if _cols_ej[_i % 3].button(_ej, key=f"ej_{_i}", use_container_width=True):
            st.session_state.ia_input_trigger = _ej

    st.divider()

    # Mostrar historial de chat
    for _msg in st.session_state.ia_mensajes_ui:
        with st.chat_message(_msg["role"]):
            st.markdown(_msg["content"])

    # Input del usuario
    _pregunta = st.chat_input("Pregunta sobre el balance del juego...")

    # Detectar si se pulsó un ejemplo
    if "ia_input_trigger" in st.session_state and st.session_state.ia_input_trigger:
        _pregunta = st.session_state.ia_input_trigger
        st.session_state.ia_input_trigger = None

    if _pregunta:
        # Mostrar mensaje del usuario
        st.session_state.ia_mensajes_ui.append({"role": "user", "content": _pregunta})
        with st.chat_message("user"):
            st.markdown(_pregunta)

        # Invocar el agente
        with st.chat_message("assistant"):
            with st.spinner("Analizando datos..."):
                try:
                    from game_agent import invocar_agente
                    _respuesta, st.session_state.ia_historial = invocar_agente(
                        _pregunta, st.session_state.ia_historial
                    )
                    st.markdown(_respuesta)
                    st.session_state.ia_mensajes_ui.append({"role": "assistant", "content": _respuesta})
                except Exception as _e:
                    _err_msg = f"⚠️ Error al invocar el agente: {_e}"
                    st.error(_err_msg)
                    st.session_state.ia_mensajes_ui.append({"role": "assistant", "content": _err_msg})

    # Botón para limpiar el historial
    if st.session_state.ia_mensajes_ui:
        st.divider()
        if st.button("🗑️ Limpiar conversación", key="clear_ia"):
            st.session_state.ia_historial = []
            st.session_state.ia_mensajes_ui = []
            st.rerun()
