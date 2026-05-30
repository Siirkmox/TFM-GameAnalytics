"""
visualization.py — Funciones de visualización estandarizadas para el TFM.

Centraliza los plots reutilizables entre notebooks para mantener
coherencia visual (paleta, tamaños, estilo) en todo el proyecto.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# Paleta oficial del proyecto por elemento
PALETTE_ELEM = {
    'Fire' : '#e74c3c',   # rojo
    'Water': '#3498db',   # azul
    'Earth': '#8B5E3C',   # marrón tierra
    'Wind' : '#B0C4DE',   # gris azulado blanquecino
}

# Estilo global
plt.rcParams.update({
    'figure.dpi'     : 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size'      : 11,
})


def boxplot_by_element(df: pd.DataFrame, col: str, palette: dict = None,
                       title: str = None, figsize: tuple = (10, 5)) -> plt.Figure:
    """
    Boxplot de una variable numérica agrupada por playerElement.
    Superpone los puntos individuales (stripplot) para visibilizar n pequeño.
    """
    palette = palette or PALETTE_ELEM
    title   = title or col

    fig, ax = plt.subplots(figsize=figsize)
    order = [e for e in ['Fire', 'Water', 'Earth', 'Wind'] if e in df['playerElement'].unique()]

    sns.boxplot(data=df, x='playerElement', y=col, order=order,
                palette=palette, width=0.5, fliersize=0, ax=ax)
    sns.stripplot(data=df, x='playerElement', y=col, order=order,
                  palette=palette, size=6, jitter=True, alpha=0.7, ax=ax)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Elemento')
    ax.set_ylabel(col)
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame, cols: list[str],
                        method: str = 'spearman',
                        figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Heatmap de correlaciones entre columnas numéricas.
    Muestra solo el triángulo inferior para evitar redundancia.
    """
    corr = df[cols].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # ocultar triángulo superior

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, ax=ax)
    ax.set_title(f'Correlaciones {method.capitalize()}', fontweight='bold')
    fig.tight_layout()
    return fig


def qq_plot(df: pd.DataFrame, cols: list[str],
            ncols: int = 3, figsize: tuple = (15, 4)) -> plt.Figure:
    """
    Q-Q plots de normalidad para una lista de columnas.
    Genera subplots automáticos en una cuadrícula de ncols columnas.
    """
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        data = df[col].dropna()
        stats.probplot(data, dist='norm', plot=axes[i])
        axes[i].set_title(f'Q-Q: {col}', fontweight='bold')
        axes[i].get_lines()[0].set(markersize=5, alpha=0.7)
        axes[i].get_lines()[1].set(color='red', linewidth=1.5)

    # Ocultar ejes sobrantes
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Q-Q Plots de Normalidad', fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


def scatter_binary(df: pd.DataFrame, x_col: str, y_col: str,
                   hue_col: str, palette: dict = None,
                   figsize: tuple = (8, 5)) -> plt.Figure:
    """
    Scatter de una variable numérica vs una variable binaria (0/1 o bool).
    Añade jitter vertical para separar los puntos del eje binario.
    """
    palette = palette or PALETTE_ELEM
    fig, ax = plt.subplots(figsize=figsize)

    df_plot = df[[x_col, y_col, hue_col]].dropna().copy()
    # Jitter vertical para el eje binario
    df_plot['_y_jitter'] = df_plot[y_col].astype(float) + np.random.uniform(-0.05, 0.05, len(df_plot))

    for elem, grp in df_plot.groupby(hue_col):
        color = palette.get(elem, '#888888')
        ax.scatter(grp[x_col], grp['_y_jitter'], label=elem,
                   color=color, alpha=0.75, s=60, edgecolors='white', linewidths=0.5)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Derrota', 'Victoria'])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{x_col} vs {y_col}', fontweight='bold')
    ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    return fig


def temporal_line(df: pd.DataFrame, x_col: str, y_col: str,
                  title: str = None, hue_col: str = None,
                  palette: dict = None, figsize: tuple = (10, 4)) -> plt.Figure:
    """
    Gráfico de línea temporal con puntos y línea de tendencia (regresión lineal).
    Colorea los puntos por hue_col si se especifica.
    """
    palette = palette or PALETTE_ELEM
    title   = title or f'{y_col} en el tiempo'

    df_plot = df[[x_col, y_col] + ([hue_col] if hue_col else [])].dropna().copy()
    df_plot = df_plot.sort_values(x_col).reset_index(drop=True)
    x_num = np.arange(len(df_plot))  # índice ordinal para regresión

    fig, ax = plt.subplots(figsize=figsize)

    # Línea de conexión gris
    ax.plot(x_num, df_plot[y_col].values, color='#cccccc', linewidth=1, zorder=1)

    # Puntos coloreados por elemento
    if hue_col:
        for elem, grp in df_plot.groupby(hue_col):
            idx = grp.index
            ax.scatter(x_num[idx], grp[y_col].values,
                       color=palette.get(elem, '#888888'), label=elem,
                       s=60, zorder=2, edgecolors='white', linewidths=0.5)
        ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        ax.scatter(x_num, df_plot[y_col].values, color='#3498db', s=60, zorder=2)

    # Línea de tendencia
    slope, intercept, r, p, _ = stats.linregress(x_num, df_plot[y_col].values)
    ax.plot(x_num, slope * x_num + intercept, color='red',
            linewidth=2, linestyle='--', label=f'Tendencia (p={p:.3f})')

    ax.set_xlabel('Sesión (orden cronológico)')
    ax.set_ylabel(y_col)
    ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    return fig
