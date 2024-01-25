"""tracewin.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""


import numpy as np
import h5py
import scipy.constants as const
from typing import Any, Optional, List

# twissed
from ..step.step import Step
from ..utils import units


def get_smilei_timesteps(filename, verbose=False):
    f = h5py.File(filename, "r")

    timesteps = []
    for item in list(f.keys()):
        if item.isdigit():
            timesteps.append(int(item))
        if item == "data":
            for jtem in list(f["data"].keys()):
                if jtem.isdigit():
                    timesteps.append(int(jtem))
        if item == "Times":
            timesteps = list(f["Times"])

    timesteps.sort()
    return timesteps


def smilei_field(
    step: Step,
    filename: str,
    timestep: int,
    omega0: float,
    n_crit: float,
) -> Step:
    step.omega0 = omega0

    f = h5py.File(filename, "r")

    dimension = f.attrs["dimension"]

    probe_list = f.attrs["fields"].decode().split(",")

    x_moved = f[str(timestep).zfill(10)].attrs["x_moved"] * const.c / omega0

    data = f[str(timestep).zfill(10)][()]

    if dimension == 1:
        step.zfield = np.array(f["positions"])[:, 0] * const.c / omega0 + x_moved
        step.Nzfield = len(step.zfield)

    if dimension == 2:
        # ! CORRIGER
        print("WARNING: Le reshape des champs 2D ne marche pas correctement!!")

        step.zfield = (
            np.sort(np.unique(np.array(f["positions"])[:, 0])) * const.c / omega0
            + x_moved
        )
        step.rfield = (
            np.sort(np.unique(np.array(f["positions"])[:, 1])) * const.c / omega0
            + x_moved
        )

        step.Nzfield = len(step.zfield)
        step.Nrfield = len(step.rfield)

    for i in range(len(probe_list)):
        item = probe_list[i]
        if item == "Ex" and dimension == 1:
            step.Ez1D = (
                data[i, :] * const.m_e * const.c * omega0 / const.e
            )  # change x to z

        elif item == "Ex" and dimension == 2:
            # ! a corriger
            step.Ez2D = (
                data[i, :] * const.m_e * const.c * omega0 / const.e
            )  # change x to z
            step.Ez2D = np.reshape(step.Ez2D, (step.Nzfield, step.Nrfield)).T

        elif item == "Env_E_abs" and dimension == 1:
            step.Ey1D_env = (
                data[i, :] * const.m_e * const.c * omega0 / const.e
            )  # change z to x
            step.a01D = step.Ey1D_env * const.e / (const.m_e * const.c * omega0)

        elif item == "Env_E_abs" and dimension == 2:
            # ! a corriger
            step.Ey2D_env = (
                data[i, :] * const.m_e * const.c * omega0 / const.e
            )  # change z to x
            step.Ey2D_env = np.reshape(step.Ey2D_env, (step.Nzfield, step.Nrfield)).T

            step.a02D = step.Ey2D_env * const.e / (const.m_e * const.c * omega0)

        elif item == "Ez" and dimension == 1:
            step.Ex1D = (
                data[i, :] * const.m_e * const.c * omega0 / const.e
            )  # change z to x

        elif item == "Ez" and dimension == 2:
            # ! a corriger
            step.Ex2D = (
                data[i, :] * const.m_e * const.c * omega0 / const.e
            )  # change z to x
            step.Ex2D = np.reshape(step.Ex2D, (step.Nzfield, step.Nrfield)).T

        elif item == "Rho" and dimension == 1:
            step.rho1D = data[i, :]
            step.density1D = -step.rho1D / const.e

        elif item == "Rho" and dimension == 2:
            # ! a corriger
            step.rho2D = data[i, :]
            step.rho2D = np.reshape(step.rho2D, (step.Nzfield, step.Nrfield)).T

            step.density2d = -step.rho2D / const.e
        else:
            if dimension == 1:
                setattr(step, item, data[i, :])
            elif dimension == 2:
                setattr(
                    step, item, np.reshape(data[i, :], (step.Nzfield, step.Nrfield)).T
                )

    return step


def smilei_particle(
    step: Step,
    filename: str,
    timestep: int,
    species: str,
    omega0: float,
    Disordered: Optional[bool] = True,
    **kwargs: Any
) -> Step:
    f = h5py.File(filename, "r")

    if Disordered:
        data = f["data"]

        if timestep == -1:
            timestep = list(data.keys())[-1]

        data = data[str(timestep).zfill(10)]

        step.dt = data.attrs["dt"]
        step.time = data.attrs["time"]
        timeUnitSI = data.attrs["timeUnitSI"]
        x_moved = data.attrs["x_moved"]

        particles = data["particles"]
        particles = particles[species]

        position = particles["position"]
        unitSI_pos = position["x"].attrs["unitSI"]
        x = np.asarray(position["x"]) * unitSI_pos  # z !!
        unitSI_pos = position["y"].attrs["unitSI"]
        y = np.asarray(position["y"]) * unitSI_pos  # y
        unitSI_pos = position["z"].attrs["unitSI"]
        z = np.asarray(position["z"]) * unitSI_pos  # x !!

        unitSI_wei = particles["weight"].attrs["unitSI"]
        w = (
            np.asarray(particles["weight"]) / const.e * units.pico
        )  # * unitSI_wei #* const.m_e

        momentum = particles["momentum"]
        unitSI_mom = momentum["x"].attrs["unitSI"]
        ux = np.asarray(momentum["x"])
        unitSI_mom = momentum["y"].attrs["unitSI"]
        uy = np.asarray(momentum["y"])
        unitSI_mom = momentum["z"].attrs["unitSI"]
        uz = np.asarray(momentum["z"])

        # Update Step class
        # WARNING: Inversion x and z for Smilei!
        step.set_new_6D_beam(z, y, x, uz, uy, ux, w)

        # Compute everything!
        step.get_beam()

        return step

    else:
        if timestep == -1:
            id_timestep = -1
        else:
            id_timestep = list(np.array(f["Times"])).index(timestep)

        x = np.array(f["x"])[id_timestep, :] * const.c / omega0  # z !!
        y = np.array(f["y"])[id_timestep, :] * const.c / omega0
        z = np.array(f["z"])[id_timestep, :] * const.c / omega0  # x !!

        w = np.array(f["w"])[id_timestep, :] / const.e * units.pico

        ux = np.array(f["px"])[id_timestep, :]
        uy = np.array(f["py"])[id_timestep, :]
        uz = np.array(f["pz"])[id_timestep, :]

        # Find and remove nan
        indexes = np.isnan(x)
        x = x[~indexes]
        y = y[~indexes]
        z = z[~indexes]
        w = w[~indexes]
        ux = ux[~indexes]
        uy = uy[~indexes]
        uz = uz[~indexes]

        # Update Step class
        # WARNING: Inversion x and z for Smilei!
        step.set_new_6D_beam(z, y, x, uz, uy, ux, w)

        # Compute everything!
        step.get_beam()

        return step
