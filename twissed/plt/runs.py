"""runs.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np
import pandas as pd
import scipy.constants as const
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy import interpolate

# Plots
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# twissed
from .colormap import Cmap
from ..utils import units

# Select color maps
cm = Cmap()


# Linear
def linear_fit(x, a, b):
    return a + b * x


# quad
def quad_fit(x, a, b, c):
    return a + b * x + c * x**2


class Runs:
    def __init__(self, path="df_full.csv") -> None:
        self.df = pd.read_csv(path).set_index("id").sort_index()

    def pairplot(self, feature, target, markersize=3.5, linewidth=1):
        fig, axs = plt.subplots(
            figsize=(len(feature) * 1.5, len(target) * 1.5),
            dpi=100,
            nrows=len(target),
            ncols=len(feature),
        )

        i = -1
        idmax = 0
        for xname in feature:
            i = i + 1
            idmin = idmax
            idmax = idmin + self.Nsampleitem
            ids = list(range(idmin, idmax))

            x = self.df[xname].iloc[ids].to_numpy() * units.convert_units(
                feature[xname][0], feature[xname][1]
            )

            xrange = [None, None]
            if feature[xname][2] == None:
                xrange[0] = self.df[xname].min() * units.convert_units(
                    feature[xname][0], feature[xname][1]
                )
            else:
                xrange[0] = feature[xname][2]
            if feature[xname][3] == None:
                xrange[1] = self.df[xname].max() * units.convert_units(
                    feature[xname][0], feature[xname][1]
                )
            else:
                xrange[1] = feature[xname][3]

            xline = np.linspace(xrange[0], xrange[1], 30)

            j = -1
            for yname in target:
                j = j + 1
                ax = axs[j, i]

                yunits = eval(self.df["units"].iloc[ids[0]])[yname]
                y = self.df[yname].iloc[ids].to_numpy() * units.convert_units(
                    yunits, target[yname]
                )

                yrange = [None, None]
                if target[yname][1] == None:
                    yrange[0] = self.df[yname].min() * units.convert_units(
                        yunits, target[yname][0]
                    )
                else:
                    yrange[0] = target[yname][1]
                if target[yname][2] == None:
                    yrange[1] = self.df[yname].max() * units.convert_units(
                        yunits, target[yname][0]
                    )
                else:
                    yrange[1] = target[yname][2]

                popt, _ = curve_fit(quad_fit, x, y)
                a, b, c = popt
                yline = quad_fit(xline, a, b, c)
                ax.plot(xline, yline, "-", color="red", lw=linewidth)

                popt, _ = curve_fit(linear_fit, x, y)
                a, b = popt
                yline = linear_fit(xline, a, b)
                ax.plot(xline, yline, "--", color="blue", lw=linewidth)

                f = interpolate.interp1d(x, y, kind="cubic", fill_value="extrapolate")
                ax.plot(xline, f(xline), "-", color="black", lw=linewidth)

                ax.plot(
                    x,
                    y,
                    marker="o",
                    color="black",
                    markersize=markersize,
                    linewidth=0.0,
                )

                if j == len(target) - 1:
                    _ = ax.set_xlabel(
                        units.convert_symb(str(xname), short=True)
                        + ", "
                        + units.convert_symb(str(feature[xname][1]), short=True)
                    )
                else:
                    _ = ax.set_xticks([])

                if i == 0:
                    _ = ax.set_ylabel(
                        units.convert_symb(str(yname), short=True)
                        + ", "
                        + units.convert_symb(str(yunits), short=True)
                    )
                else:
                    _ = ax.set_yticks([])

                _ = ax.set_xlim(xrange)
                _ = ax.set_ylim(yrange)
                _ = ax.grid(False)

        # plt.subplot_tool()
        fig.subplots_adjust(
            #                     left=0.,
            #                     bottom=0.1,
            #                     right=0.1,
            #                     top=0.2,
            wspace=0.01,
            hspace=0.01,
        )
