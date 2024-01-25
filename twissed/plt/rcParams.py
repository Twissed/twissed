from .matplotlib import plt

rcParams: dict = {
    "axes.axisbelow": False,
    "axes.grid": True,
    "axes.prop_cycle": (
        plt.cycler(
            color=[
                "#023EFF",
                "#E8000B",
                "#1AC938",
                "#8B2BE2",
                "#007C00",
                "#00D7FF",
                "#FFC400",
            ]
        )
    ),
    "axes.xmargin": 0,
    "lines.linewidth": 2,
    "grid.linewidth": 0.5,
    "grid.linestyle": ":",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.right": True,
    "figure.subplot.bottom": 0.09,
    "figure.subplot.hspace": 0.05,
    "figure.subplot.left": 0.09,
    "figure.subplot.right": 0.99,
    "figure.subplot.top": 0.99,
    "figure.subplot.wspace": 0.05,
    "figure.figsize": (6, 5),
    "figure.autolayout": True,
    "font.size": 14,
    "mathtext.fontset": "cm",
    "font.family": "STIXGeneral",
    "text.usetex": False,
}
