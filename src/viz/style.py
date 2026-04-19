"""
Global visualization style (eda_plan.md §15).
Import this module to apply consistent palette & rcParams across all notebooks.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Palette ────────────────────────────────────────────────────────────────────
# ColorBrewer-safe qualitative (max 7 categories)
PALETTE = {
    "primary":    "#1b4332",
    "secondary":  "#40916c",
    "accent":     "#52b788",
    "light":      "#95d5b2",
    "highlight":  "#d62828",   # alerts / 2019 shock
    "neutral":    "#adb5bd",
    "background": "#f8f9fa",
}

CATEGORY_COLORS = [
    "#2d6a4f", "#40916c", "#52b788", "#74c69d",
    "#95d5b2", "#b7e4c7", "#d8f3dc",
]

SEQUENTIAL = "YlGn"
DIVERGING  = "RdYlGn"


# ── rcParams ───────────────────────────────────────────────────────────────────
def apply():
    """Call once per notebook to set global style."""
    plt.rcParams.update({
        "figure.facecolor":      PALETTE["background"],
        "axes.facecolor":        PALETTE["background"],
        "axes.edgecolor":        "#ced4da",
        "axes.grid":             True,
        "grid.color":            "#dee2e6",
        "grid.linewidth":        0.5,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "font.size":             11,
        "axes.titlesize":        13,
        "axes.titleweight":      "bold",
        "axes.labelsize":        11,
        "xtick.labelsize":       10,
        "ytick.labelsize":       10,
        "legend.fontsize":       10,
        "figure.dpi":            120,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "lines.linewidth":       1.8,
        "patch.edgecolor":       "white",
        "patch.linewidth":       0.5,
    })


# ── Formatter helpers ──────────────────────────────────────────────────────────
def fmt_million(ax, axis="y"):
    """Format axis ticks as 'XM'."""
    fmt = mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def fmt_billion(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def fmt_percent(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x:.0f}%")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def add_source(ax, source: str = "Source: Datathon 2026 dataset"):
    """Add data source footnote to axes."""
    ax.annotate(
        source,
        xy=(0, -0.12), xycoords="axes fraction",
        fontsize=8, color=PALETTE["neutral"], style="italic",
    )
