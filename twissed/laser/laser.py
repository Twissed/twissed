"""laser.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np
import scipy.constants as const
from scipy.special import binom

# twissed
from ..utils.metadata import MetaData
from ..utils import physics
from ..utils import units


class Laser(MetaData):
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return "Laser class of twissed."

    def set_laser(
        self, lambda0: float, a0: float, waist0: float, duration_FWHM: float, **kwargs
    ) -> None:
        """Set the main laser parameters

        Args:
            lambda0 (float): _description_
            a0 (float): _description_
            waist0 (float): _description_
            duration_FWHM (float): _description_
        """
        self.a0 = a0
        self.lambda0 = lambda0
        self.waist0 = waist0
        self.duration_FWHM = duration_FWHM

        self.omega0 = physics.omega_laser(lambda0)
        self.Efield_laser = physics.reference_electric_field(lambda0)
        self.n_crit = physics.critical_density(lambda0)

    def set_power(
        self,
        z_focalisation: float,
        z_laser_from_right_border: float,
        nz: int,
        nr: int,
        dz: float,
        dr: float,
        plasma_density: float,
    ):
        ### mesh

        Lz = nz * dz  # longitudinal size of the simulation window, m
        Lr = nr * dr  # transverse size of the simulation window (half plane), m

        z_mesh = np.linspace(0, Lz, num=nz)
        r_mesh = np.linspace(0, Lr, num=nr)

        self.z_laser_from_right_border = z_laser_from_right_border

        self.center_laser = (Lz - z_laser_from_right_border) / const.c

        self.z_focalisation = z_focalisation

        temporal_integral = 0.0
        radial_integral = 0.0
        time_envelope = self.tgaussian
        for i in range(0, nz):
            temporal_integral += (
                np.vectorize(time_envelope)(z_mesh[i] / const.c)
            ) ** 2 * dz
        for j in range(0, nr):
            radial_integral += (
                np.abs(self.FGB(self.center_laser, r_mesh[j], self.center_laser)) ** 2
                * r_mesh[j]
                * dr
                * 2.0
                * np.pi
            )

        total_energy_product = (
            0.5
            * temporal_integral
            * radial_integral
            * self.Efield_laser**2
            * const.epsilon_0
        )

        polarization_factor = 1.0  # 1 for linear, 2 for circular

        self.energy_laser = total_energy_product * polarization_factor

        self.power_critical = (
            17 * (self.n_crit / plasma_density) * units.giga / units.tera
        )

        Power_waist = 0.0
        for j in range(0, int(self.waist0 / dr)):
            Power_waist += (
                np.abs(self.FGB(self.center_laser, r_mesh[j], self.center_laser)) ** 2
                * r_mesh[j]
                * dr
                * 2.0
                * np.pi
            )

        self.power_laser = (
            0.5
            * Power_waist
            * self.Efield_laser**2
            * const.epsilon_0
            * const.c
            / units.tera
        )

        self.power_ratio_critical = self.power_laser / self.power_critical

        transverse_profile_field = np.abs(
            self.FGB(self.center_laser, r_mesh[:], self.center_laser)
        )

        max_intensity = np.max(transverse_profile_field**2)
        max_field = np.max(transverse_profile_field)

        self.r_FWHM_intensity = (
            2
            * r_mesh[
                np.argmin(np.abs(transverse_profile_field**2 - max_intensity / 2.0))
            ]
        )

        self.r_FWHM_field = (
            2
            * r_mesh[
                np.argmin(
                    np.abs(
                        np.abs(
                            self.FGB(self.center_laser, r_mesh[:], self.center_laser)
                        )
                        - max_field / 2.0
                    )
                )
            ]
        )

        # * Longitudinal FWHM

        field_longi = np.zeros(np.size(z_mesh))
        for i in range(0, nz):
            field_longi[i] = np.abs(self.FGB(z_mesh[i], 0.0, z_mesh[i] / const.c))

        indexes = np.where(field_longi >= max(field_longi) / 2)[0]
        self.z_FWHM_field = (z_mesh[indexes[-1]] - z_mesh[indexes[0]]) / units.micro

        indexes = np.where(field_longi**2 >= max(field_longi**2) / 2)[0]
        self.z_FWHM_intensity = (z_mesh[indexes[-1]] - z_mesh[indexes[0]]) / units.micro

        self.omega_pe = physics.plasma_frequency(plasma_density)

    def tgaussian(self, t):
        """_summary_

        #### Author: F. Massimo
        #### Last Modification: 05/07/2023
        #### Purpose: compute total energy, FWHM duration and transverse FWHM of laser pulse

        Args:
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        order = 2
        sigma = (0.5 * self.duration_FWHM) ** order / np.log(2.0)
        return np.exp(-((t - self.center_laser) ** order) / sigma)

    def FGB(self, z, r, t):
        """_summary_

        #### Author: F. Massimo
        #### Last Modification: 05/07/2023
        #### Purpose: compute total energy, FWHM duration and transverse FWHM of laser pulse

        Args:
            z (_type_): _description_
            r (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        """
        return FGB laser profile, inspired from G. Maynard and FBPIC routines of FGB
        need the following parameters to be defined in the namelist,
        All in SMILEI units:
        a_0        : laser strengh
        N          : FGB order
        waist_0    : laser beam waist at focus
        z_focalisation    : longitudinal position in focus lab ref .
        duration_FWHM : laser pulse duration
        """
        N = 6
        N = int(round(N))
        w_foc = self.waist0 * np.sqrt(N + 1)
        zr = np.pi * (w_foc) ** 2 / self.lambda0
        inv_zr = 1.0 / zr
        diffract_factor = 1.0 + 1j * (z - self.z_focalisation) * inv_zr
        w = w_foc * np.abs(diffract_factor)
        scaled_radius_squared = 2 * (r**2) / w**2
        psi = np.angle(diffract_factor)
        inv_ctau2 = 1.0 / (self.duration_FWHM) ** 2
        laguerre_sum = np.zeros_like(r, dtype=np.complex128)
        for n in range(0, N + 1):
            # Recursive calculation of the Laguerre polynomial
            # - `L` represents $L_n$
            # - `L1` represents $L_{n-1}$
            # - `L2` represents $L_{n-2}$
            if n == 0:
                L = 1.0
            elif n == 1:
                L1 = L
                L = 1.0 - scaled_radius_squared
            else:
                L2 = L1
                L1 = L
                L = (((2 * n - 1) - scaled_radius_squared) * L1 - (n - 1) * L2) / n
            # Add to the sum, including the term for the additional Gouy phase
            cn = np.empty(N + 1)
            m_values = np.arange(n, N + 1)
            cn[n] = np.sum((1.0 / 2) ** m_values * binom(m_values, n)) / (N + 1)
            laguerre_sum += cn[n] * np.exp(-(2j * n) * psi) * L

        # space envelope
        exp_argument = -(r**2) / (w_foc**2 * diffract_factor)
        spatial_envelope = laguerre_sum * np.exp(exp_argument) / diffract_factor

        # time envelope is assumed Gaussian might change to have only spatial profile
        time_envelope = self.tgaussian

        # full envelope profile
        profile = self.a0 * spatial_envelope * np.vectorize(time_envelope)(t)

        return profile
