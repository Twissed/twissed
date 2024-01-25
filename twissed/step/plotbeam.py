"""plotbeam.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np
import pandas as pd
import scipy.constants as const
from typing import Optional, List, Tuple, Union, Any


# twissed
from ..utils.data import STD_DATABASE
from ..utils.units import convert_units, convert_symb, pico
from ..utils.stats import weighted_avg, weighted_std
from ..plt.colormap import Cmap
from ..plt.matplotlib import plt, sns
from ..plt.rcParams import rcParams


class StepPlotBeam:
    """Sub class to plot figure from beam data."""

    def hist1D(
        self,
        xname: str,
        xconv: Optional[str] = None,
        xrange: Optional[List[Union[float, None]]] = [None, None],
        yname: Optional[str] = "charge",
        yconv: Optional[str] = None,
        yrange: Optional[List[Union[float, None]]] = [None, None],
        bins: Optional[int] = 100,
        dx: Optional[int] = None,
        plot: Optional[str] = "plot",
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plot/get 1D histogram data from the weighted beam.

        Example
        -------
        .. code-block:: python
            :caption: Simple plot of the x distribution (dQ/dx).

            # Note that x axis is convert in um with xconv. step.sigma_x is the standard deviation (bunch length) in x.
            H, xpos = step.hist1D('x',
                                xconv='um',
                                xrange=[-3*step.sigma_x*1e6,3*step.sigma_x*1e6],
                                bins=150,
                                # dx = 1, # Instead of bins
                                plot='plot',
                                linestyle='--',
                                fwhm=True,
                                )


        .. code-block::
            :caption: Multiple plots.

            fig, axs = plt.subplots(2,3, figsize=(4*3,4*2), dpi=100, tight_layout=True)
            fig.suptitle(f"dQ/dE with various options")

            H, xpos = step.hist1D('Ek',dx=1,plot='plot',ax=axs[0,0])
            _ = axs[0,0].set_title(f"plot: dx = 1 ($\Delta$E = { xpos[1]-xpos[0] :.2f})")

            H, xpos = step.hist1D('Ek',bins=50,plot='plot',ax=axs[0,1])
            _ = axs[0,1].set_title(f"plot: bins=50 ($\Delta$E = { xpos[1]-xpos[0] :.2f})")

            H, xpos = step.hist1D('Ek',bins=20,plot='hist',ax=axs[0,2])
            _ = axs[0,2].set_title(f"hist: bins=20 ($\Delta$E = { xpos[1]-xpos[0] :.2f})")

            H, xpos = step.hist1D('Ek',bins=50,plot='lineplot',ax=axs[1,0])
            _ = axs[1,0].set_title(f"lineplot: bins=50")

            H, xpos = step.hist1D('Ek',bins=40,plot='histplot',ax=axs[1,1])
            _ = axs[1,1].set_title(f"histplot: bins=50")

            H, xpos = step.hist1D('Ek',dx=1,plot='histplot',ax=axs[1,2])
            _ = axs[1,2].set_title(f"histplot: dx=1")

        Args:
            xname (str): Name of the beam value for the x axis (e.g. "x","uy","Ek",...)
            xconv (str, optional): Unit wanted for the x axis. Defaults to None.
            xrange (list, optional): Range [min, max] of the data. Defaults to [None,None].
            yname (str, optional): Set the y axis. Defaults to 'charge'.
            yconv (str, optional): Unit wanted for the y axis. Defaults to None.
            yrange (list, optional): Range [min, max] of the data. Defaults to [None,None].
            bins (int, optional): Number of bins used. Defaults to 100.
            dx (float, optional): Step in x wanted. **Modify bins**. Defaults to None.
            plot (str, optional): Type of plot wanted. Defaults to step.
            ax (matplotlib.pyplot.axes, optional): matplotlib.pyplot.axes object for plots. Defaults to None.
                If None, but plot != None, will create a new figure with its own ax.

        Returns
        -------
        hist (float): array
            The values of the histogram. See density and weights for a description of the possible semantics.
        xpos (float): array of dtype float
            Return the bin positions (len(hist)) at the center of bins.

        Other Parameters
        ----------------
        **kwargs : List of properties
            * xlabel (bool): Plot xlabel. Defaults to True.
            * ylabel (bool): Plot xlabel. Defaults to True.
            * short (bool): Use short unit label. Defaults to False.
            * linestyle (str): Defaults to "-".
            * fwhm (bool): Plot the FWHM limits. Defaults to False.
        """

        if "units" in STD_DATABASE[xname]:
            units_base = STD_DATABASE[xname]["units"]
        else:
            units_base = None

        if xname[-4:] == "_avg":
            xname = xname[:-4]
            xavg = weighted_avg(self.__dict__[xname], self.w)
            xdata = (self.__dict__[xname] - xavg) * convert_units(units_base, xconv)
        else:
            xdata = self.convert(xname, xconv)

        xunits = units_base
        if not xconv == None:
            xunits = xconv

        if yname == "charge":
            ydata = self.w * const.e / pico * convert_units("pC", yconv)
            units_base = STD_DATABASE["charge"]["units"]

        elif yname == "current":
            ydata = self.vz * self.w * const.e
            units_base = "A"
        else:
            ydata = self.__dict__[yname]
            if "units" in STD_DATABASE[yname]:
                units_base = STD_DATABASE[yname]["units"]
            else:
                units_base = None

        yunits = units_base
        if not yconv == None:
            yunits = yconv

        ydata *= convert_units(units_base, yconv)

        if kwargs.get("range_auto", False):
            xrange = [
                weighted_avg(xdata, self.w) - 5 * weighted_std(xdata, self.w),
                weighted_avg(xdata, self.w) + 5 * weighted_std(xdata, self.w),
            ]

        df = pd.DataFrame({"x": xdata, "y": ydata})
        if not xrange[0] == None:
            df = df.drop(df[df["x"] < xrange[0]].index)
        if not xrange[1] == None:
            df = df.drop(df[df["x"] > xrange[1]].index)
        if not yrange[0] == None:
            df = df.drop(df[df["y"] < yrange[0]].index)
        if not yrange[1] == None:
            df = df.drop(df[df["y"] > yrange[1]].index)

        if dx is not None:
            Nbins = int((np.max(df["x"].to_numpy()) - np.min(df["x"].to_numpy())) / dx)
            bins = np.linspace(
                np.min(df["x"].to_numpy()), np.max(df["x"].to_numpy()), Nbins + 1
            )

        H, xedges = np.histogram(
            df["x"].to_numpy(), bins=bins, weights=df["y"].to_numpy()
        )

        H = H.T

        if yname == "current":
            H = np.abs(
                H * bins / (np.max(df["x"].to_numpy()) - np.min(df["x"].to_numpy()))
            )

        xpos = xedges[1:] - np.abs(xedges[1] - xedges[0]) / 2.0

        # Plot
        if plot is not None:
            if ax is None:
                with plt.rc_context(rcParams):
                    fig, ax = plt.subplots(
                        figsize=kwargs.get("figsize", rcParams["figure.figsize"]),
                        dpi=kwargs.get("dpi", 100),
                        tight_layout=True,
                        frameon=kwargs.get("frameon", True),
                    )

            if plot == "step":
                _ = ax.step(
                    xpos, H, where="mid", linestyle=kwargs.get("linestyle", "-")
                )
            if plot == "bar":
                _ = ax.bar(
                    xpos,
                    H,
                    width=(xpos[-1] - xpos[0]) / bins,
                    edgecolor=kwargs.get("color", None),
                )
            if plot == "plot":
                _ = ax.plot(
                    xpos,
                    H,
                    linestyle=kwargs.get("linestyle", "-"),
                    color=kwargs.get("color", None),
                )
            elif plot == "hist":
                _ = ax.hist(df["x"].to_numpy(), bins=bins, weights=df["y"].to_numpy())
            elif plot == "lineplot":
                _ = sns.lineplot(x=xpos, y=H, ax=ax)
            elif plot == "histplot":
                _ = sns.histplot(
                    data=df,
                    x="x",
                    weights="y",
                    binwidth=dx,
                    bins=bins,
                    kde=True,
                    ax=ax,
                )  # binwidth overwrite bin !

            if kwargs.get("xlabel", True):
                _ = ax.set_xlabel(
                    self.label_plot(xname, xunits, kwargs.get("short", False))
                )
            if kwargs.get("ylabel", True):
                if yname == "charge" or yname == "current":
                    _ = ax.set_ylabel(
                        self.label_plot(yname, yunits, kwargs.get("short", False))
                    )
                else:
                    _ = ax.set_ylabel(f"{STD_DATABASE[yname]['name_latex']} (count)")

            # * fwhm
            if kwargs.get("fwhm", True):
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                x_in_fwhm = xpos[np.where(H >= np.max(H) / 2)]
                fwhm = np.max(x_in_fwhm) - np.min(x_in_fwhm)

                _ = ax.vlines(
                    np.min(x_in_fwhm),
                    ax.get_ylim()[0],
                    np.max(H),
                    linestyles="dashdot",
                    colors="firebrick",
                )
                _ = ax.vlines(
                    np.max(x_in_fwhm),
                    ax.get_ylim()[0],
                    np.max(H),
                    linestyles="dashdot",
                    colors="firebrick",
                )

                # Sigma
                _ = ax.hlines(
                    np.max(H) * 0.01,
                    weighted_avg(xdata, self.w) - weighted_std(xdata, self.w) / 2,
                    weighted_avg(xdata, self.w) + weighted_std(xdata, self.w) / 2,
                    linestyles="-",
                    colors=kwargs.get("marginals_color", "firebrick"),
                )

                _ = ax.set_xlim(xlim)
                _ = ax.set_ylim(ylim)

            if kwargs.get("savefig", False) and ax is None:
                fig.savefig(
                    kwargs.get("savefig", "name.png"), dpi=kwargs.get("dpi", 100)
                )

        return H, xpos

    def hist2D(
        self,
        xname: str,
        yname: str,
        xconv: Optional[str] = None,
        yconv: Optional[str] = None,
        xrange: Optional[List[Union[float, None]]] = [None, None],
        yrange: Optional[List[Union[float, None]]] = [None, None],
        bins: Optional[List[int]] = [100, 100],
        plot: Optional[str] = "pcolormesh",
        ax: Optional[plt.Axes] = None,
        iscbar: Optional[bool] = True,
        emit: Optional[bool] = True,
        vrange: Optional[List[Union[float, None]]] = [None, None],
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plot/get 2D histogram data from the weighted beam.


        Example
        -------
        .. code-block:: python
            :caption: Simple plot of the x distribution (dQ/dx).

            twissed_cmap = twissed.Cmap()

            H, xpos, ypos = step.hist2D(
                'x',
                'xp',
                xconv='m',
                yconv='mrad',
                xrange=[- 3*step.sigma_x*1e0, 3*step.sigma_x*1e0],
                yrange=[- 3*step.sigma_xp*1e3, 3*step.sigma_xp*1e3],
                plot='pcolormesh_improved',
                cmap=twissed_cmap.cubehelix_r,
                emit=True,
            )

        .. todo::
            Remove pcolormesh_improved. Add option for 1D histogram.
            Add more kwargs.


        Args:
            xname (str): Name of the beam value for the x axis (e.g. "x","uy","Ek",...)
            yname (str): Name of the beam value for the y axis (e.g. "x","uy","Ek",...)
            xconv (str, optional): Unit wanted for the x axis. Defaults to None.
            yconv (str, optional): Unit wanted for the y axis. Defaults to None.
            xrange (list, optional): Range [min, max] of the x data.. Defaults to [np.nan,np.nan].
            yrange (list, optional): Range [min, max] of the y data.. Defaults to [np.nan,np.nan].
            bins (list, optional): Number of bins used. Defaults to [100,100].
            plot (str, optional): Type of plot wanted. Available: "pcolormesh", "scatter", "hist2D, "hexbin", "bivariate". Defaults to "pcolormesh".
            ax (matplotlib.pyplot.axes, optional): matplotlib.pyplot.axes object for plots. Defaults to None.
                If None, but plot != None, will create a new figure with its own ax.
            iscbar (bool, optional): Plot colorbar. Defaults to True.
            emit (bool, optional): Plot emittance. Defaults to True.

        Returns
        -------
        hist (float): 2D array
            The bi-dimensional histogram of samples x and y. Values in x are histogrammed along the first dimension and values in y are histogrammed along the second dimension.
        xpos (float):1D array
            The bin positions (middle) along the x-axis.
        ypos (float): 1D array
            The bin positions (middle) along the y-axis.


        Other Parameters
        ----------------
        **kwargs : List of properties
            * cmap (str): colormap of . Defaults to 'tracewin'.
            * grid (str): Defaults to True.
            * shading (str): Pcolormesh shading. Defaults to "auto". "gouraud" available.
            * vmin (float): Pcolormesh vmin. Defaults to None.
            * vmax (float): Pcolormesh vmax. Defaults to None.
            * xlabel (bool): Plot xlabel. Defaults to True.
            * ylabel (bool): Plot xlabel. Defaults to True.
            * short (bool): Use short unit label. Defaults to False.
            * set_xticks (list): Defaults to False.
            * set_yticks (list): Defaults to False.
            * marginals_bar (bool): Plot small 1d histogram (bar format). Defaults to False.
            * marginals_plot (bool): Plot small 1d histogram (plotline format). Defaults to False.
            * marginals_step (bool): Plot small 1d histogram (step format). Defaults to True.
            * marginals_color (str): Color of the small 1d histogram. Defaults to "firebrick".
            * emit_color (str): Color of the emittance. Default to "firebrick".
            * panel_text (str): Name of the panel. For instance: "(a)", "(b)", or "i)".
        """

        if "units" in STD_DATABASE[xname]:
            units_base = STD_DATABASE[xname]["units"]
        else:
            units_base = None

        # Average data
        if xname[-4:] == "_avg":
            xname = xname[:-4]
            xavg = weighted_avg(self.__dict__[xname], self.w)
            xdata = (self.__dict__[xname] - xavg) * convert_units(units_base, xconv)
        else:
            xdata = self.convert(xname, xconv)

        xunits = units_base
        if not xconv == None:
            xunits = xconv

        if "units" in STD_DATABASE[yname]:
            units_base = STD_DATABASE[yname]["units"]
        else:
            units_base = None

        if yname[-4:] == "_avg":
            yname = yname[:-4]
            yavg = weighted_avg(self.__dict__[yname], self.w)
            ydata = (self.__dict__[yname] - yavg) * convert_units(units_base, yconv)
        else:
            ydata = self.convert(yname, yconv)

        yunits = units_base
        if not yconv is None:
            yunits = yconv

        if kwargs.get("range_auto", False):
            xrange = [
                weighted_avg(xdata, self.w) - 5 * weighted_std(xdata, self.w),
                weighted_avg(xdata, self.w) + 5 * weighted_std(xdata, self.w),
            ]
            yrange = [
                weighted_avg(ydata, self.w) - 5 * weighted_std(ydata, self.w),
                weighted_avg(ydata, self.w) + 5 * weighted_std(ydata, self.w),
            ]

        df = pd.DataFrame({xname: xdata, yname: ydata, "w": self.w})

        if xrange[0] is not None:
            df = df.drop(df[df[xname] < xrange[0]].index)
        else:
            xrange[0] = np.min(xdata)
        if xrange[1] is not None:
            df = df.drop(df[df[xname] > xrange[1]].index)
        else:
            xrange[1] = np.max(xdata)
        if yrange[0] is not None:
            df = df.drop(df[df[yname] < yrange[0]].index)
        else:
            yrange[0] = np.min(ydata)
        if yrange[1] is not None:
            df = df.drop(df[df[yname] > yrange[1]].index)
        else:
            yrange[1] = np.max(ydata)

        H, xedges, yedges = np.histogram2d(
            df[xname].to_numpy(),
            df[yname].to_numpy(),
            bins=(bins[0], bins[1]),
            weights=df["w"].to_numpy(),
        )
        H = H.T * const.e / pico

        xpos = xedges[1:] - np.abs(xedges[1] - xedges[0]) / 2.0
        ypos = yedges[1:] - np.abs(yedges[1] - yedges[0]) / 2.0
        X, Y = np.meshgrid(xpos, ypos)

        # Plot
        if plot is not None:
            # Select color maps
            twissed_cmap = Cmap()

            if ax is None:
                with plt.rc_context(rcParams):
                    fig, ax = plt.subplots(
                        figsize=kwargs.get("figsize", rcParams["figure.figsize"]),
                        dpi=kwargs.get("dpi", 100),
                        tight_layout=True,
                        frameon=kwargs.get("frameon", True),
                    )

            if plot == "pcolormesh":
                g = ax.pcolormesh(
                    X,
                    Y,
                    H,
                    cmap=kwargs.get("cmap", twissed_cmap.sky_icefire),
                    shading=kwargs.get("shading", "auto"),
                    vmin=vrange[0],
                    vmax=vrange[1],
                )
                if iscbar:
                    cbar = plt.colorbar(g, ax=ax, pad=-0)
                    if kwargs.get("annotate_pC", True):
                        ax.annotate(
                            "pC", xy=(1.09, 1), xycoords="axes fraction", color="black"
                        )
            elif plot == "contourf":
                g = ax.contourf(
                    X,
                    Y,
                    H,
                    kwargs.get("contourf_levels", 10),
                    cmap=kwargs.get("cmap", twissed_cmap.sky_icefire),
                    vmin=vrange[0],
                    vmax=vrange[1],
                )
                if iscbar:
                    cbar = plt.colorbar(g, ax=ax, pad=-0)
                    if kwargs.get("annotate_pC", True):
                        ax.annotate(
                            "pC", xy=(1.09, 1), xycoords="axes fraction", color="black"
                        )

            elif plot == "scatter":
                x = (
                    np.reshape(X, (np.shape(H)[0] * np.shape(H)[1],))
                    + (xedges[1] - xedges[0]) / 2.0
                )
                y = (
                    np.reshape(Y, (np.shape(H)[0] * np.shape(H)[1],))
                    + (yedges[1] - yedges[0]) / 2.0
                )
                c = np.reshape(H, (np.shape(H)[0] * np.shape(H)[1],))
                clim = np.max(c) / 10
                ids = np.where(c >= clim)[0]

                g = ax.scatter(
                    x[ids],
                    y[ids],
                    s=kwargs.get("size", 15),
                    c=c[ids],
                    cmap=kwargs.get("cmap", twissed_cmap.sky_icefire),
                    marker=kwargs.get("marker", "."),
                    alpha=kwargs.get("alpha", 1),
                )
                if iscbar:
                    cbar = plt.colorbar(g, ax=ax, pad=-0)
                    if kwargs.get("annotate_pC", True):
                        ax.annotate(
                            "pC", xy=(1.09, 1), xycoords="axes fraction", color="black"
                        )

            elif plot == "hist2d":
                g = ax.hist2d(
                    df[xname].to_numpy(),
                    df[yname].to_numpy(),
                    bins=[bins[0], bins[1]],
                    range=None,
                    density=False,
                    weights=df["w"].to_numpy() * const.e * 1e12,
                )
                if iscbar:
                    cbar = plt.colorbar(g[3], ax=ax, pad=-0)
                    if kwargs.get("annotate_pC", True):
                        ax.annotate(
                            "pC", xy=(1.09, 1), xycoords="axes fraction", color="black"
                        )

            elif plot == "hexbin":
                g = ax.hexbin(
                    df[xname].to_numpy(),
                    df[yname].to_numpy(),
                    C=df["w"].to_numpy() * const.e * 1e12,
                    gridsize=bins,
                    marginals=True,
                    cmap="mako",
                )
                ax.grid(False)
                if iscbar:
                    cbar = plt.colorbar(g, ax=ax, pad=-0)
                    ax.annotate(
                        "pC", xy=(1.09, 1), xycoords="axes fraction", color="black"
                    )

            elif plot == "bivariate":
                # sns.scatterplot(x=df[xname].to_numpy(), y=df[yname].to_numpy(), s=5, color=".15",ax=ax)
                sns.histplot(
                    x=df[xname].to_numpy(),
                    y=df[yname].to_numpy(),
                    weights=df["w"].to_numpy() * const.e * 1e12,
                    bins=50,
                    pthresh=0.1,
                    cmap=kwargs.get("cmap", twissed_cmap.sky_icefire),
                    ax=ax,
                    cbar=iscbar,
                )
                sns.kdeplot(
                    x=df[xname].to_numpy(),
                    y=df[yname].to_numpy(),
                    weights=df["w"].to_numpy() * const.e * 1e12,
                    levels=5,
                    color="w",
                    linewidths=1,
                    ax=ax,
                )
                if iscbar:
                    ax.annotate(
                        "pC", xy=(1.09, 1), xycoords="axes fraction", color="black"
                    )

            # Marginals
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if kwargs.get("marginals_bar", False):
                xr = ax.get_xlim()[1] - ax.get_xlim()[0]
                yr = ax.get_ylim()[1] - ax.get_ylim()[0]
                ax.bar(
                    xpos,
                    H.sum(0) / H.sum(0).max() * 0.1 * yr,
                    width=xr / bins[0],
                    bottom=ax.get_ylim()[0],
                    edgecolor=kwargs.get("marginals_color", "firebrick"),
                    fill=False,
                )
                ax.barh(
                    ypos,
                    H.sum(1) / H.sum(1).max() * 0.1 * xr,
                    height=yr / bins[1],
                    left=ax.get_xlim()[0],
                    edgecolor=kwargs.get("marginals_color", "firebrick"),
                    fill=False,
                )
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            elif kwargs.get("marginals_plot", False):
                xr = ax.get_xlim()[1] - ax.get_xlim()[0]
                yr = ax.get_ylim()[1] - ax.get_ylim()[0]
                ax.plot(
                    xpos,
                    H.sum(0) / H.sum(0).max() * 0.1 * yr + ax.get_ylim()[0],
                    color=kwargs.get("marginals_color", "firebrick"),
                )
                ax.plot(
                    H.sum(1) / H.sum(1).max() * 0.1 * xr + ax.get_xlim()[0],
                    ypos,
                    color=kwargs.get("marginals_color", "firebrick"),
                )
            elif kwargs.get("marginals_step", True):
                xr = ax.get_xlim()[1] - ax.get_xlim()[0]
                yr = ax.get_ylim()[1] - ax.get_ylim()[0]
                ax.step(
                    xpos,
                    H.sum(0) / H.sum(0).max() * 0.1 * yr + ax.get_ylim()[0],
                    color=kwargs.get("marginals_color", "firebrick"),
                    where="mid",
                )
                ax.step(
                    H.sum(1) / H.sum(1).max() * 0.1 * xr + ax.get_xlim()[0],
                    ypos,
                    color=kwargs.get("marginals_color", "firebrick"),
                    where="mid",
                )

            if kwargs.get("marginals_fwhm", True) and (
                kwargs.get("marginals_bar", False)
                or kwargs.get("marginals_plot", False)
                or kwargs.get("marginals_step", True)
            ):
                xr = ax.get_xlim()[1] - ax.get_xlim()[0]
                yr = ax.get_ylim()[1] - ax.get_ylim()[0]

                x_in_fwhm = xpos[np.where(H.sum(0) >= np.max(H.sum(0)) / 2)]
                _ = ax.vlines(
                    np.min(x_in_fwhm),
                    np.min(H.sum(0) / H.sum(0).max() * 0.1 * yr + ax.get_ylim()[0]),
                    np.max(H.sum(0) / H.sum(0).max() * 0.1 * yr + ax.get_ylim()[0]),
                    linestyles="dashdot",
                    colors=kwargs.get("marginals_color", "firebrick"),
                )
                _ = ax.vlines(
                    np.max(x_in_fwhm),
                    np.min(H.sum(0) / H.sum(0).max() * 0.1 * yr + ax.get_ylim()[0]),
                    np.max(H.sum(0) / H.sum(0).max() * 0.1 * yr + ax.get_ylim()[0]),
                    linestyles="dashdot",
                    colors=kwargs.get("marginals_color", "firebrick"),
                )

                y_in_fwhm = ypos[np.where(H.sum(1) >= np.max(H.sum(1)) / 2)]
                _ = ax.hlines(
                    np.min(y_in_fwhm),
                    np.min(H.sum(1) / H.sum(1).max() * 0.1 * xr + ax.get_xlim()[0]),
                    np.max(H.sum(1) / H.sum(1).max() * 0.1 * xr + ax.get_xlim()[0]),
                    linestyles="dashdot",
                    colors=kwargs.get("marginals_color", "firebrick"),
                )
                _ = ax.hlines(
                    np.max(y_in_fwhm),
                    np.min(H.sum(1) / H.sum(1).max() * 0.1 * xr + ax.get_xlim()[0]),
                    np.max(H.sum(1) / H.sum(1).max() * 0.1 * xr + ax.get_xlim()[0]),
                    linestyles="dashdot",
                    colors=kwargs.get("marginals_color", "firebrick"),
                )

                # Sigma
                _ = ax.hlines(
                    np.min(H.sum(0) / H.sum(0).max() * 0.1 * yr + ax.get_ylim()[0])
                    + (
                        np.max(H.sum(0) / H.sum(0).max() * 0.1 * yr + ax.get_ylim()[0])
                        - np.min(
                            H.sum(0) / H.sum(0).max() * 0.1 * yr + ax.get_ylim()[0]
                        )
                    )
                    * 0.1,
                    weighted_avg(xdata, self.w) - weighted_std(xdata, self.w) / 2,
                    weighted_avg(xdata, self.w) + weighted_std(xdata, self.w) / 2,
                    linestyles="-",
                    colors=kwargs.get("marginals_color", "firebrick"),
                )

                _ = ax.vlines(
                    np.min(H.sum(1) / H.sum(1).max() * 0.1 * xr + ax.get_xlim()[0])
                    + (
                        np.max(H.sum(1) / H.sum(1).max() * 0.1 * xr + ax.get_xlim()[0])
                        - np.min(
                            H.sum(1) / H.sum(1).max() * 0.1 * xr + ax.get_xlim()[0]
                        )
                    )
                    * 0.1,
                    weighted_avg(ydata, self.w) - weighted_std(ydata, self.w) / 2,
                    weighted_avg(ydata, self.w) + weighted_std(ydata, self.w) / 2,
                    linestyles="-",
                    colors=kwargs.get("marginals_color", "firebrick"),
                )

            # ax.set_xlim(xlim)
            # ax.set_ylim(ylim)
            # End Marginals

            # TODO : Add range size
            # ax.set_xlim(xrange)
            # ax.set_ylim(yrange)

            if kwargs.get("grid", True):
                # _ = ax.set_axisbelow(False)  # Put grid on the front of the pcolormesh
                _ = ax.grid(True)  # , color="grey", lw=0.5

            if kwargs.get("xlabel", True):
                _ = ax.set_xlabel(
                    self.label_plot(xname, xunits, kwargs.get("short", False))
                )
            if kwargs.get("ylabel", True):
                _ = ax.set_ylabel(
                    self.label_plot(yname, yunits, kwargs.get("short", False))
                )

            if kwargs.get("set_xticks", False):
                _ = ax.set_xticks(kwargs.get("set_xticks", []))
            if kwargs.get("set_yticks", False):
                _ = ax.set_yticks(kwargs.get("set_yticks", []))

            if kwargs.get("panel_text", False):
                if iscbar:
                    xy = (0.86, 0.88)
                else:
                    xy = (0.9, 0.88)
                _ = ax.annotate(
                    kwargs.get("panel_text", False),
                    xy=xy,
                    xycoords="axes fraction",
                    color="black",
                    fontsize=kwargs.get("panel_font", rcParams["font.size"]),
                )

            if emit:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                x_avg = weighted_avg(xdata, self.w)
                xx_avg = weighted_avg((xdata - x_avg) ** 2, self.w)
                xp_avg = weighted_avg(ydata, self.w)
                xpxp_avg = weighted_avg((ydata - xp_avg) ** 2, self.w)
                xxp_avg = weighted_avg((xdata - x_avg) * (ydata - xp_avg), self.w)

                # Define Twiss parameters
                emittance = np.sqrt(np.abs(xx_avg * xpxp_avg - xxp_avg**2))
                beta = xx_avg / emittance
                alpha = -xxp_avg / emittance
                # Plot the ellipse
                theta = np.linspace(0, 2 * np.pi, 100)
                x = np.sqrt(beta * emittance) * np.cos(theta)
                y = -np.sqrt(emittance / beta) * (alpha * np.cos(theta) + np.sin(theta))
                _ = ax.plot(
                    (x + x_avg),
                    (y + xp_avg),
                    color=kwargs.get("emit_color", "firebrick"),
                )

                # ax.set_xlim(xlim)
                # ax.set_ylim(ylim)

            if kwargs.get("savefig", False) and ax is None:
                fig.savefig(
                    kwargs.get("savefig", "name.png"), dpi=kwargs.get("dpi", 100)
                )

        return H, xpos, ypos

    def plot_emit(
        self,
        xname: str,
        yname: str,
        xconv: Optional[str] = None,
        yconv: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            xname (str): _description_
            yname (str): _description_
            xconv (Optional[str], optional): _description_. Defaults to None.
            yconv (Optional[str], optional): _description_. Defaults to None.
            ax (Optional[plt.Axes], optional): _description_. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_


        Other Parameters
        ----------------
        **kwargs : List of properties
            * xlabel (bool): Plot xlabel. Defaults to True.
            * ylabel (bool): Plot xlabel. Defaults to True.
            * short (bool): Use short unit label. Defaults to False.
        """
        if "units" in STD_DATABASE[xname]:
            units_base = STD_DATABASE[xname]["units"]
        else:
            units_base = None

        # Average data
        if xname[-4:] == "_avg":
            xname = xname[:-4]
            xavg = weighted_avg(self.__dict__[xname], self.w)
            xdata = (self.__dict__[xname] - xavg) * convert_units(units_base, xconv)
        else:
            xdata = self.convert(xname, xconv)

        xunits = units_base
        if not xconv == None:
            xunits = xconv

        if "units" in STD_DATABASE[yname]:
            units_base = STD_DATABASE[yname]["units"]
        else:
            units_base = None

        if yname[-4:] == "_avg":
            yname = yname[:-4]
            yavg = weighted_avg(self.__dict__[yname], self.w)
            ydata = (self.__dict__[yname] - yavg) * convert_units(units_base, yconv)
        else:
            ydata = self.convert(yname, yconv)

        yunits = units_base
        if not yconv == None:
            yunits = yconv

        x_avg = weighted_avg(xdata, self.w)

        xx_avg = weighted_avg((xdata - x_avg) ** 2, self.w)

        xp_avg = weighted_avg(ydata, self.w)

        xpxp_avg = weighted_avg((ydata - xp_avg) ** 2, self.w)

        xxp_avg = weighted_avg((xdata - x_avg) * (ydata - xp_avg), self.w)

        # Define Twiss parameters
        emittance = np.sqrt(np.abs(xx_avg * xpxp_avg - xxp_avg**2))

        beta = xx_avg / emittance

        alpha = -xxp_avg / emittance

        # Plot the ellipse
        theta = np.linspace(0, 2 * np.pi, 100)

        x = np.sqrt(beta * emittance) * np.cos(theta)

        y = -np.sqrt(emittance / beta) * (alpha * np.cos(theta) + np.sin(theta))

        if ax is None:
            with plt.rc_context(rcParams):
                fig, ax = plt.subplots(
                    figsize=kwargs.get("figsize", rcParams["figure.figsize"]),
                    dpi=kwargs.get("dpi", 100),
                    tight_layout=True,
                    frameon=kwargs.get("frameon", True),
                )

        _ = ax.plot((x + x_avg), (y + xp_avg))

        if kwargs.get("xlabel", True):
            _ = ax.set_xlabel(
                self.label_plot(xname, xunits, kwargs.get("short", False))
            )
        if kwargs.get("ylabel", True):
            _ = ax.set_ylabel(
                self.label_plot(yname, yunits, kwargs.get("short", False))
            )

        if kwargs.get("savefig", False) and ax is None:
            fig.savefig(kwargs.get("savefig", "name.png"), dpi=kwargs.get("dpi", 100))

        return (x + x_avg), (y + xp_avg)

    def plot_beam(
        self,
        xrange: Optional[List[Union[float, None]]] = [None, None],
        xprange: Optional[List[Union[float, None]]] = [None, None],
        yrange: Optional[List[Union[float, None]]] = [None, None],
        yprange: Optional[List[Union[float, None]]] = [None, None],
        zrange: Optional[List[Union[float, None]]] = [None, None],
        Ekrange: Optional[List[Union[float, None]]] = [None, None],
        **kwargs: Any,
    ) -> Tuple[plt.figure, plt.Axes]:
        """Plot the 6D phase spaces of a beam

        Example
        -------
        .. code-block:: python
            :caption: Plot the 6D phase spaces of a beam

            step.plot_beam()

        Args:
            xrange (list, optional): Range [min, max] of the x data. Defaults to [np.nan,np.nan].
            xprange (list, optional): Range [min, max] of the xp data. Defaults to [np.nan, np.nan].
            yrange (list, optional): Range [min, max] of the y data. Defaults to [np.nan,np.nan].
            yprange (list, optional): Range [min, max] of the yp data. Defaults to [np.nan, np.nan].
            zrange (list, optional): Range [min, max] of the z data. Defaults to [np.nan, np.nan].
            Ekrange (list, optional): Range [min, max] of the y data. Defaults to [np.nan, np.nan].

        Returns:
            fig: matplotlib.figure.
            axs: matplotlib.axes.Axes or array of axes.


        Other Parameters
        ----------------
        **kwargs : List of properties

            * xconv (str): Unit wanted for the x data. Defaults to "um".
            * xpconv (str): Unit wanted for the xp data. Defaults to "mrad".
            * yconv (str): Unit wanted for the y data. Defaults to "um".
            * ypconv (str): Unit wanted for the yp data. Defaults to "mrad".
            * zconv (str): Unit wanted for the z data. Defaults to "um".
            * Ekconv (str): Unit wanted for the Ek data. Defaults to "MeV".
            * bins (list): Number of bins used. Defaults to [100,100].
            * text (str): Display info. Defaults to True.
            * figsize (2-tuple of floats). Figure dimension (width, height) in inches. Default to (8, 8).
        """

        # Select color maps
        twissed_cmap = Cmap()

        with plt.rc_context(rcParams):
            fig, axs = plt.subplots(
                2,
                2,
                figsize=kwargs.get("figsize", rcParams["figure.figsize"]),
                dpi=kwargs.get("dpi", 100),
                # tight_layout=True,
                frameon=kwargs.get("frameon", True),
            )

        # Fig 1
        ax = axs[0, 0]
        _ = self.hist2D(
            "x",
            "xp",
            xconv=kwargs.get("xconv", "um"),
            yconv=kwargs.get("xpconv", "mrad"),
            xrange=xrange,
            yrange=xprange,
            plot="pcolormesh",
            bins=kwargs.get("bins", [100, 100]),
            ax=ax,
            emit=True,
            panel_text="(a)",
            range_auto=kwargs.get("range_auto", True),
        )

        # Fig 2
        ax = axs[0, 1]
        _ = self.hist2D(
            "y",
            "yp",
            xconv=kwargs.get("yconv", "um"),
            yconv=kwargs.get("ypconv", "mrad"),
            xrange=yrange,
            yrange=yprange,
            plot="pcolormesh",
            bins=kwargs.get("bins", [100, 100]),
            ax=ax,
            emit=True,
            panel_text="(b)",
            range_auto=kwargs.get("range_auto", True),
        )

        # Fig 3
        ax = axs[1, 0]
        _ = self.hist2D(
            "z_avg",
            "Ek",
            xconv=kwargs.get("zconv", "um"),
            yconv=kwargs.get("Ekconv", "MeV"),
            xrange=zrange,
            yrange=Ekrange,
            plot="pcolormesh",
            bins=kwargs.get("bins", [100, 100]),
            ax=ax,
            emit=True,
            panel_text="(c)",
            range_auto=kwargs.get("range_auto", True),
        )

        # Fig 4
        ax = axs[1, 1]
        _ = self.hist2D(
            "x",
            "y",
            xconv=kwargs.get("xconv", "um"),
            yconv=kwargs.get("yconv", "um"),
            xrange=xrange,
            yrange=yrange,
            plot="pcolormesh",
            bins=kwargs.get("bins", [100, 100]),
            ax=ax,
            emit=True,
            panel_text="(d)",
            range_auto=kwargs.get("range_auto", True),
        )

        if kwargs.get("savefig", False):
            fig.savefig(kwargs.get("savefig", "name.png"), dpi=kwargs.get("dpi", 100))

        return fig, axs
