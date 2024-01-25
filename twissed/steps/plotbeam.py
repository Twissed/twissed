"""plotbeam.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np

# twissed
from ..plt.matplotlib import plt, sns
from ..plt.rcParams import rcParams
from ..utils.units import convert_symb


class StepsPlotBeam:
    def plot1D(
        self,
        xname,
        yname,
        xconv=None,
        yconv=None,
        xrange=[None, None],
        yrange=[None, None],
        ax=None,
        **kwargs,
    ):
        x = self.convert(xname, xconv)
        y = self.convert(yname, yconv)

        xunits = self.units[xname]
        if not xconv == None:
            xunits = xconv

        yunits = self.units[yname]
        if not yconv == None:
            yunits = yconv

        if ax == None:
            with plt.rc_context(rcParams):
                fig, ax = plt.subplots(
                    figsize=kwargs.get("figsize", rcParams["figure.figsize"]),
                    dpi=kwargs.get("dpi", 100),
                    tight_layout=True,
                    frameon=kwargs.get("frameon", True),
                )

        _ = ax.plot(x, y, label=convert_symb(str(yname)))

        _ = ax.legend()
        _ = ax.set_xlim(xrange)
        _ = ax.set_ylim(yrange)
        _ = ax.set_xlabel(convert_symb(str(xname)) + ", " + convert_symb(str(xunits)))
        _ = ax.set_ylabel(convert_symb(str(yname)) + ", " + convert_symb(str(yunits)))

    def plot2D(
        self,
        xname,
        yname,
        xconv=None,
        yconv=None,
        ax=None,
        iscbar=True,
        cmap="CMRmap_r",
        **kwargs,
    ):
        if xname == "z":
            xname = "z_avg"
        if yname == "x":
            yname = "x_charge_pos"
            z = np.array(self.x_charge).T
        if yname == "y":
            yname = "y_charge_pos"
            z = np.array(self.y_charge).T
        if yname == "z":
            yname = "z_charge_pos"
            z = np.array(self.z_charge).T

        x = self.convert(xname, xconv)
        y = self.convert(yname, yconv)

        xunits = self.units[xname]
        if not xconv == None:
            xunits = xconv

        yunits = self.units[yname]
        if not yconv == None:
            yunits = yconv

        if ax == None:
            with plt.rc_context(rcParams):
                fig, ax = plt.subplots(
                    figsize=kwargs.get("figsize", rcParams["figure.figsize"]),
                    dpi=kwargs.get("dpi", 100),
                    tight_layout=True,
                    frameon=kwargs.get("frameon", True),
                )

        g = ax.pcolormesh(x, y, z, cmap=cmap)

        if iscbar:
            cbar = plt.colorbar(g, ax=ax, pad=-0)
            ax.annotate("pC", xy=(1.09, 1), xycoords="axes fraction", color="black")

        ax.set_axisbelow(False)
        ax.grid(True, color="grey", lw=0.5)

        _ = ax.set_xlabel(convert_symb(str(xname)) + ", " + convert_symb(str(xunits)))
        _ = ax.set_ylabel(convert_symb(str(yname)) + ", " + convert_symb(str(yunits)))
