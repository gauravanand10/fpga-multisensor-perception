import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# chart style (light surface, fixed categorical order)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# class colours for label maps (background dark grey + 5 classes)
CLASS_RGB = [(40, 40, 40), (42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164)]


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "font.family": "DejaVu Sans"})
    return plt


def colorize(labels):
    import numpy as np
    lut = np.array(CLASS_RGB, np.uint8)
    return lut[np.asarray(labels, np.int64)]
