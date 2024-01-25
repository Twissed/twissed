"""physics.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np
import scipy.constants as const
from typing import Union


def plasma_frequency(ne: float) -> float:
    r"""Return the theoretical waist over :math:`z`.

    .. math::

        \omega_{\rm pe} = \sqrt{\frac{n_{\rm e} e^2}{m_{\rm e} \epsilon_0}}

    Args:
        ne (float): Plasma density in m-3

    Returns:
        float: Plasma frequency in rad/s.
    """
    return np.sqrt(ne * const.e**2 / const.m_e / const.epsilon_0)


def omega_laser(lambda0: float) -> float:
    r"""Return the laser frequency :math:`\omega_0`.

    .. math::

        \omega_0 =

    Args:
        lambda0 (float): Laser wavelength

    Returns:
        float: laser frequency in rad/s.
    """
    return 2.0 * np.pi * const.c / lambda0


def laser_strength(I0: float, lambda0: float, normalised: bool = False) -> float:
    r"""Laser strength parameter :math:`a_0`.

    .. math::

        a_0 = \sqrt{\frac{e^2}{2 \pi^2 \epsilon_0 m_e^2 c^5} \lambda_0 I_0}

    or if :code:`normalised == True`:

    .. math::

        a_0 = 0.855 \lambda_0 [\mu \mathrm{m}] \sqrt{I_0 [10^{18} \mathrm{W/cm}^2]}

    Args:
        I0 (float): Maximum intensity
        lambda0 (float): Wavelength
        normalised (bool): Normalised equation. Default to False.

    Returns:
        float: Laser strength parameter
    """
    if normalised:
        return np.sqrt(
            (const.e**2)
            / (2 * np.pi**2 * const.epsilon_0 * const.m_e**2 * const.c**5)
            * lambda0
            * I0
        )
    else:
        return 0.855 * lambda0 * np.sqrt(I0)


def critical_density(lambda0: float) -> float:
    r"""Laser critical density :math:`n_{\mathrm{c}}`.

    .. math::

        n_{\mathrm{c}} = \frac{m_e \epsilon_0 \omega_0^2}{e^2}


    Args:
        lambda0 (float): Wavelength

    Returns:
        float: Laser critical density
    """
    return (
        const.m_e
        * const.epsilon_0
        / const.e**2
        * (2 * np.pi * const.c / lambda0) ** 2
    )


def accelerating_electric_field(ne: float) -> float:
    r"""Return the maximum accelerating electric field (a.k.a the wavebreaking field or space charge field) :math:`E_{\rm max}`.

    .. math::

        E_{\rm max} = \frac{m_e c \omega_{\rm pe}}{e}

    Args:
        ne (float): Plasma density in m-3

    Returns:
        float: Maximum accelerating electric field in V/m
    """
    return const.m_e * const.c * plasma_frequency(ne) / const.e


def reference_electric_field(lambda0: float) -> float:
    r"""Return the reference electric field :math:`E_{0}` (useful for normalisation).

    .. math::

        E_{0} = \frac{m_e c \omega_0}{e}

    Args:
        lambda0 (float): Wavelength

    Returns:
        float: Reference electric field in V/m
    """
    return const.m_e * const.c * omega_laser(lambda0) / const.e


def Rayleigh_length(w0: float, lambda0: float) -> float:
    r"""Return the Rayleigh length :math:`z_{\rm R}`.

    .. math::

        z_{\rm R} = \frac{\pi w_0^2}{\lambda_0}

    Args:
        w0 (float): Maximum waist
        lambda0 (float): Wavelength

    Returns:
        float: Rayleigh length
    """
    return np.pi * w0**2 / lambda0


def waist0_theory(
    z: Union[float, np.ndarray], w0: float, lambda0: float, zfoc: float = 0
) -> Union[float, np.ndarray]:
    r"""Return the theoretical waist over :math:`z`.

    .. math::

        w(z) = w_0 \sqrt{1 + \left( \frac{z - z_{\rm foc}}{z_{\rm R}} \right)^2}

    Args:
        z (Union[float, np.ndarray]): Positions in z
        w0 (float): Maximum waist
        zfoc (float): Z focal position. Default to 0.
        lambda0 (float): Laser wavelength

    Returns:
        Union[float, np.ndarray]: Theoretical waist
    """
    zr = Rayleigh_length(w0, lambda0)
    return w0 * np.sqrt(1.0 + ((z - zfoc) / zr) ** 2)


def convert_laser_duration_FWHM(t: float, convert: str = "fbpic_to_FWHM") -> float:
    """Convert laser FWHM duration.

    Args:
        t (float): time to convert.
        convert (str, optional): Type of conversion.
            Chose between "fbpic_to_FWHM", "FWHM_to_fbpic", "smilei_to_FWHM" or "FWHM_to_smilei". Defaults to "fbpic_to_FWHM".

    Returns:
        float: Converted time
    """
    if convert == "fbpic_to_FWHM":
        return t * np.sqrt(2 * np.log(2))

    elif convert == "FWHM_to_fbpic":
        return t / np.sqrt(2 * np.log(2))

    if convert == "smilei_to_FWHM":
        return t / np.sqrt(2)

    elif convert == "FWHM_to_smilei":
        return t * np.sqrt(2)
