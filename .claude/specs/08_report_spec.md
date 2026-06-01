# Spec: 08_report.ipynb + 09_dashboard.py — Informe Final y Dashboard

## Design

**Objetivo:**
Producir el informe ejecutivo final del TFM y un dashboard interactivo Streamlit
que permita explorar los resultados del análisis de balance de forma visual.

**Entradas:**
- Todos los exports en `data/exports/`: `eda_hallazgos.csv`, `statistical_results.csv`,
  `ml_predictions.csv`, `player_clusters.csv`, `nlp_results.csv`,
  `balance_index.csv`, `balance_recommendations.csv`
- Todas las figuras en `reports/figures/` (fig2_* a fig7_*)
- `src/visualization.py`, `src/analysis.py` — funciones de soporte

**Salidas:**
- `reports/balance_report.html` — informe HTML ejecutivo completo
- `09_dashboard.py` — aplicación Streamlit standalone
- `README_dashboard.md` — instrucciones de ejecución del dashboard

**Conexión con proyectos anteriores:**
- **P4 (Power BI / Streamlit):** mismo patrón de tabs temáticos en el dashboard
- **P7 (Informe de Producción):** estructura de informe ejecutivo: contexto → metodología → resultados → conclusiones → recomendaciones

**Estructura del informe `08_report.ipynb`:**
```
1. Portada y resumen ejecutivo (3-5 bullets clave)
2. Contexto del proyecto y datos
3. Metodología (pipeline de fases)
4. Hallazgos principales (con figuras de cada fase)
5. Balance Index y recomendaciones
6. Limitaciones y trabajo futuro
7. Conclusiones finales
```

**Estructura del dashboard `09_dashboard.py`:**
```python
st.tabs([
    "Resumen",         # Balance Index + top recomendaciones
    "Elementos",       # Comparativa Fire/Water/Earth/Wind
    "Hechizos",        # Spell efficiency + noMana + uso relativo
    "Salas",           # Heatmap elemento×nivel + top salas abandono
    "ML",              # Feature importance + clusters
    "Recomendaciones"  # Tabla completa filtrable por prioridad
])
```

---

## Requirements

**Notebook `08_report.ipynb`:**
- Una celda Markdown por sección con narrativa redactada
- Toda afirmación estadística debe citar p-value y test de Fase 3
- Incluir las figuras clave (no todas — seleccionar 8-10 más relevantes)
- Exportar a HTML: `jupyter nbconvert --to html 08_report.ipynb --output reports/balance_report.html`
- Conclusión final debe responder: ¿Está el juego equilibrado? ¿Qué elemento necesita rebalanceo urgente?

**Dashboard `09_dashboard.py` — Requirements técnicos:**
```python
import streamlit as st
import pandas as pd
import plotly.express as px

# Cargar todos los exports al inicio (con @st.cache_data)
@st.cache_data
def load_data():
    ...

# Paleta de elementos consistente con notebooks
PALETTE_ELEM = {'Fire': '#e74c3c', 'Water': '#3498db', 'Earth': '#27ae60', 'Wind': '#f39c12'}
```

**Tab "Recomendaciones" — filtros:**
- Selectbox de prioridad (Todas / Alta / Media / Baja)
- Selectbox de elemento (Todos / Fire / Water / Earth / Wind)
- Tabla interactiva con `st.dataframe()` + highlighting por prioridad

**Librerías requeridas:**
- `streamlit` — dashboard
- `plotly.express` — gráficos interactivos (compatibles con Streamlit)
- `pandas` — datos
- `nbconvert` — exportar notebook a HTML (solo para el notebook, no el dashboard)

**Outputs mínimos:**
- `balance_report.html` generado y funcional
- `09_dashboard.py` que arranca con `streamlit run 09_dashboard.py` sin errores
- `README_dashboard.md` con instrucciones de instalación y ejecución
- Todos los tabs del dashboard muestran datos reales (no placeholders)

**Restricciones:**
- El dashboard debe funcionar offline (no llamadas a APIs externas)
- Usar `@st.cache_data` para todas las cargas de datos (performance)
- No duplicar lógica de cálculo en el dashboard — cargar desde exports CSV
- El informe HTML debe ser autocontenido (imágenes embebidas, no paths relativos)
  → Usar `--no-input` y `--embed-images` en nbconvert

---

## Tasks

### Bloque 1 — Estructura del informe
- [ ] Celda Markdown: portada con título, autor, fecha, TFM
- [ ] Celda Markdown: resumen ejecutivo (5 bullets clave con hallazgos cuantificados)
- [ ] Celda Markdown: contexto del proyecto y descripción del dataset
- [ ] **Jira: TFM-19**

### Bloque 2 — Metodología y pipeline
- [ ] Diagrama de fases (Mermaid o imagen) mostrando pipeline completo
- [ ] Descripción de cada fase: inputs, técnica, outputs
- [ ] Nota sobre limitaciones del dataset (n=11 sesiones)
- [ ] **Jira: TFM-20**

### Bloque 3 — Hallazgos principales
- [ ] Seleccionar y mostrar 8-10 figuras más relevantes (una por hallazgo)
- [ ] Cada figura acompañada de párrafo interpretativo
- [ ] Citar p-values de Fase 3 cuando corresponda
- [ ] Tabla de Balance Index como figura central
- [ ] **Jira: TFM-20**

### Bloque 4 — Conclusiones y recomendaciones
- [ ] Tabla de `balance_recommendations.csv` formateada con colores
- [ ] Sección "Limitaciones": tamaño de muestra, versión del juego, plataformas
- [ ] Sección "Trabajo futuro": n≥30 por elemento, nuevos niveles, más hechizos
- [ ] Conclusión final: respuesta directa a las hipótesis del TFM
- [ ] **Jira: TFM-21**

### Bloque 5 — Exportar HTML
- [ ] Comando `nbconvert` con `--embed-images` para HTML autocontenido
- [ ] Verificar que el HTML abre correctamente en navegador
- [ ] Guardar en `reports/balance_report.html`
- [ ] **Jira: TFM-22**

### Bloque 6 — Dashboard Streamlit: estructura y datos
- [ ] Crear `09_dashboard.py` con estructura de tabs
- [ ] Implementar `load_data()` con `@st.cache_data`
- [ ] Sidebar: filtros globales (elemento, fecha)
- [ ] Tab "Resumen": KPIs principales + Balance Index barplot
- [ ] **Jira: TFM-45**

### Bloque 7 — Dashboard Streamlit: tabs restantes
- [ ] Tab "Elementos": comparativa de KPIs por elemento (radar o barplot agrupado)
- [ ] Tab "Hechizos": scatter uso vs eficiencia + noMana barplot
- [ ] Tab "Salas": heatmap elemento×nivel + top salas abandono
- [ ] Tab "ML": feature importance + scatter clusters PCA
- [ ] Tab "Recomendaciones": tabla filtrable por prioridad y elemento
- [ ] **Jira: TFM-45**

### Bloque 8 — README y cierre
- [ ] Crear `README_dashboard.md` con `pip install -r requirements.txt` + `streamlit run 09_dashboard.py`
- [ ] Test final: ejecutar dashboard y verificar todos los tabs
- [ ] Commit de todos los archivos no sensibles (excluir `data/raw/`, `data/processed/`, `credentials/`)
- [ ] **Jira: TFM-22**
