"""Shared Plotly figure builders."""
from __future__ import annotations

import plotly.graph_objects as go

from utils.data_loader import COLORS, PLOTLY_LAYOUT


def apply_theme(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    """Apply the Gridbreaker dark theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(family="Playfair Display", size=20, color=COLORS["text_hi"]),
                x=0.01, xanchor="left",
            )
        )
    return fig


def annotate(fig: go.Figure, x, y, text: str, color: str | None = None,
             ax: int = 0, ay: int = -40, arrow: bool = True) -> go.Figure:
    """Add an annotation with brand styling."""
    c = color or COLORS["glow"]
    fig.add_annotation(
        x=x, y=y, text=text, showarrow=arrow, arrowhead=2,
        arrowcolor=c, arrowwidth=1.5,
        ax=ax, ay=ay,
        font=dict(family="Inter", size=12, color=c),
        bgcolor="rgba(10,31,26,0.85)",
        bordercolor=c, borderwidth=1, borderpad=4,
    )
    return fig
