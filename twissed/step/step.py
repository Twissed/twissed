"""step.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

from __future__ import annotations

import numpy as np
import scipy.constants as const
from typing import Any

# twissed
from ..utils.metadata import MetaData
from ..utils.stats import weighted_avg, weighted_mad, weighted_std, weighted_med
from ..utils import units
from .plotbeam import StepPlotBeam
from .plotfield import StepPlotField

# Variables that can be used to sort the beam.
BEAM_SELECTION = ["x", "y", "z", "ux", "uy", "uz", "w", "g", "Ek"]


class Step(
    MetaData,
    StepPlotBeam,
    StepPlotField,
):
    """
    Main Class for simulation parameters. Include beam, laser and plasma parameters at a given timestep.


    Examples
    --------

    .. code-block:: python

        #Creation of the step class
        step = twissed.Step()

        # Read data
        step = steps.read_beam(step,timestep,species='electrons',g=[10.,None])

        # Examples
        print(f"N particle: {step.N}")
        print(f"Positions {step.x} [m]")
        step.print('emit_norm_rms_y')
        step.print('charge')
        step.print('dt')
        print(f"Convert dt: {step.convert('dt','ps')} [ps]")


    .. code-block:: python

        step = twissed.read_dst("treacewin.dst")


    Main keys
    ---------

        * **x** (*float*) - Array of position in x, in m
        * **y** (*float*) - Array of position in y, in m
        * **z** (*float*) - Array of position in z, in m
        * **ux** (*float*) - Normalised momenta x :math:`u_x = \gamma v_x /c = \gamma beta_x` of the macro-particle of the beam
        * **uy** (*float*) - Normalised momenta y :math:`u_y = \gamma v_y /c = \gamma beta_y` of the macro-particle of the beam
        * **uy** (*float*) - Normalised momenta z :math:`u_z = \gamma v_z /c = \gamma beta_z` of the macro-particle of the beam
        * **w** (*float*) - Weighs of the macro-particles in term of particles, in number of particles per macro-particles'
        * **g** (*float*) - Lorentz factor of every single macro-particles
        * **Ek** (*float*) - Relativistic kinetic energy of macro-particles

    .. note::

        Keys are only available if defined from read or set functions.

    """

    def __init__(self, *args, **kwargs):
        self.verbose = kwargs.get("verbose", True)

    def __call__(self, *args, **kwargs) -> str:
        return "Step class of twissed at a given timestep."

    def copy(self) -> Step:
        """Create copy of Step class

        Returns:
            Step: copy of Step class
        """

        step_new = Step()

        for key in list(self.__dict__.keys()):
            step_new.__dict__[key] = self.__dict__[key]

        return step_new

    def beam_keep_index(self, indexes_to_keep):
        """
        Reduce the particles arrays with given indexes. Keep intact the dicts units and info.
        Such as:     self.w = self.w[indexes_to_keep]

        Args:
            indexes_to_keep (int): indexes to keep
        """
        for item in BEAM_SELECTION:
            self.__dict__[item] = self.__dict__[item][indexes_to_keep]

    def set_new_6D_beam(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        w: np.ndarray,
    ) -> None:
        """Set position and momenta of the beam

        Args:
            x (np.ndarray): Positions x of the macro-particle of the beam in m
            y (np.ndarray): Positions y of the macro-particle of the beam in m
            z (np.ndarray): Positions z of the macro-particle of the beam in m
            ux (np.ndarray): Normalised momenta x :math:`u_x = \gamma v_x /c = \gamma beta_x` of the macro-particle of the beam
            uy (np.ndarray): Normalised momenta y :math:`u_y = \gamma v_y /c = \gamma beta_y` of the macro-particle of the beam
            uz (np.ndarray): Normalised momenta z :math:`u_z = \gamma v_z /c = \gamma beta_z` of the macro-particle of the beam
            w (np.ndarray): Weighs of the macro-particles in term of number of particles
        """

        self.x = np.array(x)

        self.y = np.array(y)

        self.z = np.array(z)

        self.ux = np.array(ux)

        self.uy = np.array(uy)

        self.uz = np.array(uz)

        self.w = np.array(w)

    def get_beam(self, **kwargs) -> None:
        """
        Generate all the beam data.

        .. code-block:: python

            step.get_beam(x=[-12*1e-6,12*1e-6], g=[200,None],Ek_avg=[-50,50])

        Range arguments accepted for selection
        --------------------------------------

            * **x** (*float*) - Array of position in x, in m
            * **y** (*float*) - Array of position in y, in m
            * **z** (*float*) - Array of position in z, in m
            * **ux** (*float*) - Normalised momenta x :math:`u_x = \gamma v_x /c = \gamma beta_x` of the macro-particle of the beam
            * **uy** (*float*) - Normalised momenta y :math:`u_y = \gamma v_y /c = \gamma beta_y` of the macro-particle of the beam
            * **uy** (*float*) - Normalised momenta z :math:`u_z = \gamma v_z /c = \gamma beta_z` of the macro-particle of the beam
            * **w** (*float*) - Weighs of the macro-particles in term of particles, in number of particles per macro-particles
            * **g** (*float*) - Lorentz factor of every single macro-particles
            * **Ek** (*float*) - Relativistic kinetic energy of macro-particles

        .. note::

            * All the above arguments can be added with the '_avg' suffix to force the selection to be performed around the mean value.
            * Variables self.w, self.ux, self.uy, self.uz, self.x, self.y and self.z must have be defined.
        """

        # TODO: Add security self.w, self.ux, self.uy, self.uz, self.x, self.y and self.z

        self.g = np.sqrt(1 + self.ux**2 + self.uy**2 + self.uz**2)  # gamma

        self.Ek = (
            (self.g - 1) * const.m_e * const.c**2 / const.e / units.mega
        )  # Kinetic energy

        # TODO: Add weight selection and reducer (if N too high)

        # Selection of particles
        for name in BEAM_SELECTION:
            if kwargs.get(name, False):
                self.__dict__[name + "_range"] = kwargs.get(name, False)
                if not kwargs.get(name, False)[0] == None:
                    indexes_to_keep = np.where(
                        self.__dict__[name] >= kwargs.get(name, False)[0]
                    )
                    self.beam_keep_index(indexes_to_keep)
                if not kwargs.get(name, False)[1] == None:
                    indexes_to_keep = np.where(
                        self.__dict__[name] <= kwargs.get(name, False)[1]
                    )
                    self.beam_keep_index(indexes_to_keep)

            name_avg = name + "_avg"
            if kwargs.get(name_avg, False):
                self.__dict__[name_avg + "_range"] = kwargs.get(name_avg, False)
                mean = weighted_avg(self.__dict__[name], self.w)
                if not kwargs.get(name_avg, False)[0] == None:
                    indexes_to_keep = np.where(
                        self.__dict__[name] - mean >= kwargs.get(name_avg, False)[0]
                    )
                    self.beam_keep_index(indexes_to_keep)
                if not kwargs.get(name_avg, False)[1] == None:
                    indexes_to_keep = np.where(
                        self.__dict__[name] - mean <= kwargs.get(name_avg, False)[1]
                    )
                    self.beam_keep_index(indexes_to_keep)

        self.vz = self.uz / self.g * const.c  # Speed

        # ! Security for other species than electrons !!!

        self.charge = np.sum(self.w * const.e) / units.pico  # Total charge

        self.N = np.size(self.w)  # Number of particles

        if self.N < 10:
            if self.verbose:
                print("No beam!")
        else:
            # * Energies
            self.Ek_avg = weighted_avg(self.Ek, self.w)  # Mean energy

            self.Ek_med = weighted_med(self.Ek, self.w)  # Median energy

            self.Ek_std = weighted_std(self.Ek, self.w)  # STD energy

            self.Ek_mad = weighted_mad(self.Ek, self.w)  # MAD energy

            self.Ek_std_perc = self.Ek_std / self.Ek_avg * 100  # STD energy %

            self.Ek_mad_perc = self.Ek_mad / self.Ek_med * 100  # MAD energy %

            # * Lorentz
            self.g_avg = weighted_avg(self.g, self.w)  # Mean Lorentz

            self.g_med = weighted_med(self.g, self.w)  # Median Lorentz

            self.g_std = weighted_std(self.g, self.w)  # STD Lorentz

            self.g_mad = weighted_mad(self.g, self.w)  # MAD Lorentz

            # * Size
            self.sigma_x = weighted_std(self.x, self.w)  # size in x

            self.sigma_y = weighted_std(self.y, self.w)  # size in y

            self.sigma_z = weighted_std(self.z, self.w)  # size in z

            self.sigma_ux = weighted_std(self.ux, self.w)  # size in ux

            self.sigma_uy = weighted_std(self.uy, self.w)  # size in uy

            self.sigma_uz = weighted_std(self.uz, self.w)  # size in uz

            self.sigma_Ek = weighted_std(self.Ek, self.w)  # STD energy

            # * Average mean velocity
            self.betaz_avg = weighted_avg(self.vz / const.c, self.w)

            # * dp/p
            self.p = self.g * self.betaz_avg * const.c  # momenta

            self.p_avg = weighted_avg(self.p, self.w)

            self.dp = (self.p - self.p_avg) / self.p_avg  # Momenta variation dp/p

            self.dp_avg = weighted_avg(self.dp, self.w)

            self.sigma_dp = weighted_std(self.dp, self.w)

            # * xp and yp
            self.xp = self.ux / self.uz

            self.yp = self.uy / self.uz

            self.sigma_xp = weighted_std(self.xp, self.w)

            self.sigma_yp = weighted_std(self.yp, self.w)

            self.x_divergence = weighted_std(np.arctan2(self.ux, self.uz), self.w)

            self.y_divergence = weighted_std(np.arctan2(self.uy, self.uz), self.w)

            # * Center of mass <x>
            self.x_avg = weighted_avg(self.x, self.w)

            self.y_avg = weighted_avg(self.y, self.w)

            self.z_avg = weighted_avg(self.z, self.w)

            # * <x**2>
            self.xx_avg = weighted_avg((self.x - self.x_avg) ** 2, self.w)

            self.yy_avg = weighted_avg((self.y - self.y_avg) ** 2, self.w)

            # * <ux>
            self.ux_avg = weighted_avg(self.ux, self.w)

            self.uy_avg = weighted_avg(self.uy, self.w)

            self.uz_avg = weighted_avg(self.uz, self.w)

            # * <x'>
            self.xp_avg = weighted_avg(self.xp, self.w)

            self.yp_avg = weighted_avg(self.yp, self.w)

            # * <x'*2>
            self.xpxp_avg = weighted_avg((self.xp - self.xp_avg) ** 2, self.w)

            self.ypyp_avg = weighted_avg((self.yp - self.yp_avg) ** 2, self.w)

            # * <x x'>
            self.xxp_avg = weighted_avg(
                (self.x - self.x_avg) * (self.xp - self.xp_avg), self.w
            )

            self.yyp_avg = weighted_avg(
                (self.y - self.y_avg) * (self.yp - self.yp_avg), self.w
            )

            # * trace emittances
            self.emit_rms_x = (
                np.sqrt(np.abs(self.xx_avg * self.xpxp_avg - self.xxp_avg**2))
                / units.micro
            )

            self.emit_rms_y = (
                np.sqrt(np.abs(self.yy_avg * self.ypyp_avg - self.yyp_avg**2))
                / units.micro
            )

            self.emit_norm_rms_x = self.emit_rms_x * self.g_avg * self.betaz_avg

            self.emit_norm_rms_y = self.emit_rms_y * self.g_avg * self.betaz_avg

            # * Twiss parameters x, y
            self.beta_x = self.xx_avg / self.emit_rms_x / units.micro

            self.beta_y = self.yy_avg / self.emit_rms_y / units.micro

            self.gamma_x = self.xpxp_avg / self.emit_rms_x / units.micro

            self.gamma_y = self.ypyp_avg / self.emit_rms_y / units.micro

            self.alpha_x = -self.xxp_avg / self.emit_rms_x / units.micro

            self.alpha_y = -self.yyp_avg / self.emit_rms_y / units.micro

            # * Beam sigma matrix

            self.xy_avg = weighted_avg(
                (self.x - self.x_avg) * (self.y - self.y_avg), self.w
            )

            self.xyp_avg = weighted_avg(
                (self.x - self.x_avg) * (self.yp - self.yp_avg), self.w
            )

            self.yxp_avg = weighted_avg(
                (self.y - self.y_avg) * (self.xp - self.xp_avg), self.w
            )

            self.xpyp_avg = weighted_avg(
                (self.xp - self.xp_avg) * (self.yp - self.yp_avg), self.w
            )

            self.xz_avg = weighted_avg(
                (self.x - self.x_avg) * (self.z - self.z_avg), self.w
            )

            self.xdp_avg = weighted_avg(
                (self.x - self.x_avg) * (self.dp - self.dp_avg), self.w
            )

            self.zxp_avg = weighted_avg(
                (self.xp - self.xp_avg) * (self.z - self.z_avg), self.w
            )

            self.xpdp_avg = weighted_avg(
                (self.xp - self.xp_avg) * (self.dp - self.dp_avg), self.w
            )

            self.yz_avg = weighted_avg(
                (self.y - self.y_avg) * (self.z - self.z_avg), self.w
            )

            self.ydp_avg = weighted_avg(
                (self.y - self.y_avg) * (self.dp - self.dp_avg), self.w
            )

            self.zyp_avg = weighted_avg(
                (self.yp - self.yp_avg) * (self.z - self.z_avg), self.w
            )

            self.ypdp_avg = weighted_avg(
                (self.yp - self.yp_avg) * (self.dp - self.dp_avg), self.w
            )

            self.zz_avg = weighted_avg(
                (self.z - self.z_avg) * (self.z - self.z_avg), self.w
            )

            self.zdp_avg = weighted_avg(
                (self.z - self.z_avg) * (self.dp - self.dp_avg), self.w
            )

            self.dpdp_avg = weighted_avg(
                (self.dp - self.dp_avg) * (self.dp - self.dp_avg), self.w
            )

            sigma_matrix = [
                [
                    self.xx_avg,
                    self.xxp_avg,
                    self.xy_avg,
                    self.xyp_avg,
                    self.xz_avg,
                    self.xdp_avg,
                ],
                [
                    self.xxp_avg,
                    self.xpxp_avg,
                    self.yxp_avg,
                    self.xpyp_avg,
                    self.zxp_avg,
                    self.xpdp_avg,
                ],
                [
                    self.xy_avg,
                    self.yxp_avg,
                    self.yy_avg,
                    self.yyp_avg,
                    self.yz_avg,
                    self.ydp_avg,
                ],
                [
                    self.xyp_avg,
                    self.xpyp_avg,
                    self.yyp_avg,
                    self.ypyp_avg,
                    self.zyp_avg,
                    self.ypdp_avg,
                ],
                [
                    self.xz_avg,
                    self.zxp_avg,
                    self.yz_avg,
                    self.zyp_avg,
                    self.zz_avg,
                    self.zdp_avg,
                ],
                [
                    self.xdp_avg,
                    self.xpdp_avg,
                    self.ydp_avg,
                    self.ypdp_avg,
                    self.zdp_avg,
                    self.dpdp_avg,
                ],
            ]
            self.sigma_matrix = np.array(sigma_matrix)

            # * z emittances
            self.emit_rms_z = (
                np.sqrt(np.abs(self.zz_avg * self.dpdp_avg - self.zdp_avg**2))
                / units.micro
            )

            self.emit_norm_rms_z = self.emit_rms_z * self.g_avg * self.betaz_avg / 10

            # * Twiss parameters z
            self.beta_z = self.zz_avg * 1e6 / self.emit_rms_z * 10

            self.gamma_z = self.dpdp_avg * 1e6 / self.emit_rms_z

            self.alpha_z = -self.zdp_avg * 1e6 / self.emit_rms_z

            # * 4D and 6D emittances
            self.emit_norm_rms_4D = self.emit_norm_rms_x * self.emit_norm_rms_y

            self.emit_norm_rms_6D = (
                self.emit_norm_rms_x * self.emit_norm_rms_y * self.emit_norm_rms_z * 10
            )

            # * Dispersions
            self.x_dispersion = weighted_avg(
                (self.dp * self.x) / (self.dp**2), self.w
            )

            self.y_dispersion = weighted_avg(
                (self.dp * self.y) / (self.dp**2), self.w
            )

            self.get_advanced_beam()

    def get_advanced_beam(self, **kwargs: Any) -> None:
        """Add arrays to beam data"""

        # * dQ/dE histogramm

        if self.N > 1000:
            H, xpos = self.hist1D("Ek", dx=1, plot=None)
            x_in_fwhm = xpos[np.where(H >= np.max(H) / 2)]
            fwhm = np.max(x_in_fwhm) - np.min(x_in_fwhm)
        else:
            H = [0.0]
            xpos = [0.0]
            x_in_fwhm = 0.0
            fwhm = 0.0

        self.Ek_hist_yaxis = H

        self.Ek_hist_xaxis = xpos

        self.Ek_hist_peak = xpos[np.argmax(H)]

        self.Ek_hist_fwhm = fwhm

    def rotationXY(self, angle: float) -> Step:
        """Counterclockwise rotate the x-y plan to a given angle


        Example
        -------
        .. code-block:: python

            # 90 degree rotation of the beam
            step_new = step.rotationXY(90*np.pi/180)

        Args:
            angle (float): Angle of the counterclockwise rotation, in rad

        Returns:
            step: New step class
        """

        step_new = self.copy()

        x_new = self.x * np.cos(angle) - self.y * np.sin(angle)

        y_new = self.x * np.sin(angle) + self.y * np.cos(angle)

        ux_new = self.ux * np.cos(angle) - self.uy * np.sin(angle)

        uy_new = self.ux * np.sin(angle) + self.uy * np.cos(angle)

        step_new.set_new_6D_beam(x_new, y_new, self.z, ux_new, uy_new, self.uz, self.w)

        step_new.get_beam()

        return step_new
