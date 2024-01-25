"""plotfield.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np
from typing import Optional, List, Union

# twissed
from ..utils.data import STD_DATABASE
from ..utils.units import convert_symb
from ..plt.matplotlib import plt, ticker
from ..plt.rcParams import rcParams
from ..plt.colormap import Cmap


def fmt(x, pos):
    a, b = "{:.0e}".format(x).split("e")
    b = int(b)
    if b == 0:
        return r"0"
    # return r'${} \cdot 10^{{{}}}$'.format(a, b)
    return "{}e{}".format(a, b)


class StepPlotField:
    def plot_field1D(
        self,
        xname: str,
        yname: str,
        xconv: Optional[str] = None,
        yconv: Optional[str] = None,
        xrange: Optional[List[Union[float, None]]] = [None, None],
        yrange: Optional[List[Union[float, None]]] = [None, None],
        ax: Optional[plt.Axes] = None,
        **kwargs
    ) -> None:
        x = self.convert(xname, xconv)
        y = self.convert(yname, yconv)

        if "units" in STD_DATABASE[xname]:
            units_base = STD_DATABASE[xname]["units"]
        else:
            units_base = None

        xunits = units_base
        if xconv is not None:
            xunits = xconv

        if "units" in STD_DATABASE[yname]:
            units_base = STD_DATABASE[yname]["units"]
        else:
            units_base = None

        yunits = units_base
        if yconv is not None:
            yunits = yconv

        if ax is None:
            with plt.rc_context(rcParams):
                fig, ax = plt.subplots(
                    figsize=kwargs.get("figsize", rcParams["figure.figsize"]),
                    dpi=kwargs.get("dpi", 100),
                    tight_layout=True,
                    frameon=kwargs.get("frameon", True),
                )

        if not kwargs.get("yaxisleft", False):
            _ = ax.plot(x, y, color=kwargs.get("color", "blue"))

            _ = ax.set_ylim(yrange)

            if kwargs.get("ylabel", True):
                _ = ax.set_ylabel(
                    self.label_plot(yname, yunits, kwargs.get("short", False))
                )
            else:
                _ = ax.set_yticks([])

        else:
            ax2 = ax.twinx()
            _ = ax2.plot(x, y, color=kwargs.get("color", "blue"))

            _ = ax2.set_ylim(yrange)

            if kwargs.get("ylabel", True):
                _ = ax2.set_ylabel(
                    self.label_plot(yname, yunits, kwargs.get("short", False))
                )
            else:
                _ = ax2.set_yticks([])

            _ = ax2.grid(False)

        _ = ax.set_xlim(xrange)

        if kwargs.get("xlabel", True):
            _ = ax.set_xlabel(
                self.label_plot(xname, xunits, kwargs.get("short", False))
            )
        else:
            _ = ax.set_xticks([])

        if kwargs.get("fwhm", True):
            x_in_fwhm = x[np.where(y >= np.max(y) / 2)]
            fwhm = np.max(x_in_fwhm) - np.min(x_in_fwhm)

            _ = ax.vlines(
                np.min(x_in_fwhm),
                ax.get_ylim()[0],
                np.max(y),
                linestyles="dashdot",
                colors="firebrick",
            )
            _ = ax.vlines(
                np.max(x_in_fwhm),
                ax.get_ylim()[0],
                np.max(y),
                linestyles="dashdot",
                colors="firebrick",
            )

        if kwargs.get("savefig", False) and ax is None:
            fig.savefig(kwargs.get("savefig", "name.png"), dpi=kwargs.get("dpi", 100))

    def plot_field2D(
        self,
        xname: str,
        yname: str,
        zname: str,
        xconv: Optional[str] = None,
        yconv: Optional[str] = None,
        zconv: Optional[str] = None,
        xrange: Optional[List[Union[float, None]]] = [None, None],
        yrange: Optional[List[Union[float, None]]] = [None, None],
        vrange: Optional[List[Union[float, None]]] = [None, None],
        ax: Optional[plt.Axes] = None,
        **kwargs
    ):
        twissed_cmap = Cmap()

        x = self.convert(xname, xconv)
        y = self.convert(yname, yconv)
        z = self.convert(zname, zconv)

        if "units" in STD_DATABASE[xname]:
            units_base = STD_DATABASE[xname]["units"]
        else:
            units_base = None

        xunits = units_base
        if not xconv == None:
            xunits = xconv

        if "units" in STD_DATABASE[yname]:
            units_base = STD_DATABASE[yname]["units"]
        else:
            units_base = None

        yunits = units_base
        if not yconv == None:
            yunits = yconv

        if "units" in STD_DATABASE[zname]:
            units_base = STD_DATABASE[zname]["units"]
        else:
            units_base = None

        zunits = units_base
        if not zconv == None:
            zunits = zconv

        if ax == None:
            with plt.rc_context(rcParams):
                fig, ax = plt.subplots(
                    figsize=kwargs.get("figsize", rcParams["figure.figsize"]),
                    dpi=kwargs.get("dpi", 100),
                    tight_layout=True,
                    frameon=kwargs.get("frameon", True),
                )

        g = ax.pcolormesh(
            x,
            y,
            z,
            vmin=vrange[0],
            vmax=vrange[1],
            cmap=kwargs.get("cmap", twissed_cmap.sky_icefire),
            shading=kwargs.get("shading", "auto"),
        )

        iscbar = kwargs.get("iscbar", True)
        if iscbar == True:
            cbar = plt.colorbar(g, ax=ax, pad=-0)
            _ = cbar.set_label(
                self.label_plot(zname, zunits, kwargs.get("short", False))
            )

        if iscbar == "cbar_reduced_b":
            cbar = plt.colorbar(
                g,
                shrink=0.4,
                pad=-0.13,
                location="top",
                orientation="horizontal",
                format=ticker.FuncFormatter(fmt),
            )
            t = ax.annotate(
                self.label_plot(zname, zunits, kwargs.get("short", False)),
                xy=(0.27, 0.96),
                xycoords="axes fraction",
                color="black",
            )

        if iscbar == "cbar_reduced_w":
            cbar = plt.colorbar(
                g,
                shrink=0.4,
                pad=-0.13,
                location="top",
                orientation="horizontal",
                format=ticker.FuncFormatter(fmt),
            )
            # cbar.ax.yaxis.set_tick_params(colors='white')
            cbar.ax.xaxis.set_tick_params(colors="white")
            cbar.outline.set_edgecolor("white")
            t = ax.annotate(
                self.label_plot(zname, zunits, kwargs.get("short", False)),
                xy=(0.27, 0.96),
                xycoords="axes fraction",
                color="white",
            )

        if kwargs.get("Ez1D", False):
            self.plot_field1D(
                "zfield",
                "Ez1D",
                xconv=xconv,
                xrange=xrange,
                yconv=kwargs.get("Ez1D_conv", "GV/m"),
                yrange=kwargs.get("Ez1D_range", [None, None]),
                color=kwargs.get("Ez1D_color", "blue"),
                xlabel=kwargs.get("xlabel", True),
                ylabel=kwargs.get("Ez1D_ylabel", True),
                yaxisleft=True,
                fwhm=False,
                ax=ax,
            )

        if kwargs.get("Ey2D_env", False):
            Ey2D_env_mask = np.abs(
                np.ma.masked_array(
                    self.Ey2D_env,
                    np.abs(self.Ey2D_env) < np.abs(self.Ey2D_env).max() * 0.5,
                )
            )
            g2 = ax.pcolormesh(
                x, y, Ey2D_env_mask, cmap=kwargs.get("Ey2D_env_cmap", "hot")
            )

        _ = ax.set_xlim(xrange)
        _ = ax.set_ylim(yrange)

        # if kwargs.get("grid",True):
        #     _ = ax.set_axisbelow(False) # Put grid on the front of the pcolormesh
        #     _ = ax.grid(True, color="grey", lw=0.5)

        if kwargs.get("xlabel", True):
            _ = ax.set_xlabel(
                self.label_plot(xname, xunits, kwargs.get("short", False))
            )
        else:
            _ = ax.set_xticks([])
        if kwargs.get("ylabel", True):
            _ = ax.set_ylabel(
                self.label_plot(yname, yunits, kwargs.get("short", False))
            )
        else:
            _ = ax.set_yticks([])

        if kwargs.get("savefig", False) and ax is None:
            fig.savefig(kwargs.get("savefig", "name.png"), dpi=kwargs.get("dpi", 100))
