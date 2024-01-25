"""fbpic.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import h5py
import numpy as np
import os
import scipy.constants as const
from scipy.signal import hilbert

# twissed
from ..step.step import Step


def get_fbpic_timesteps(directory, verbose=False):
    directory = directory + "/hdf5"
    if verbose:
        print(directory)

    filenames = next(os.walk(directory), (None, None, []))[2]  # [] if no file

    timesteps = []
    for file in filenames:
        timestep = file[4:-3]
        timesteps.append(int(timestep))

    timesteps.sort()
    if verbose:
        print(f"timesteps: {timesteps}")

    return timesteps


def get_data_attr(step, data, timestep):
    step.dt = data.attrs["dt"]
    step.time = data.attrs["time"]
    step.timestep = timestep
    step.timeUnitSI = data.attrs["timeUnitSI"]


def fbpic_field(step, directory, timestep, **kwargs):
    """Read hdf5 files from FBPIC.

    Args:
        step (Step class): Step class to be used
        directory (str): Path of the data
        timestep (int): Id of the timestep to be taken

    Returns:
        Step class: Step class to be used
    """

    directory = directory + "/hdf5"
    filename = directory + "/data" + str(timestep).zfill(8) + ".h5"

    if kwargs.get("verbose", True):
        print(f"Read file {filename}")

    # Open file
    f = h5py.File(filename, "r")

    data = f["data"]

    data = data[str(timestep)]

    get_data_attr(step, data, timestep)

    #### Fields
    fields = data["fields"]  # ['B', 'E', 'J', 'rho']
    # fields.attrs.keys()  # ['chargeCorrection', 'chargeCorrectionParameters', 'currentSmoothing', 'currentSmoothingParameters', 'fieldBoundary', 'fieldSolver', 'particleBoundary']

    Efield = fields["E"]
    # Efield.attrs.keys()  # ['axisLabels', 'dataOrder', 'fieldSmoothing', 'geometry', 'geometryParameters', 'gridGlobalOffset', 'gridSpacing', 'gridUnitSI', 'timeOffset', 'unitDimension']

    dset_Er = np.asarray(Efield["r"])
    dset_Et = np.asarray(Efield["t"])
    dset_Ez = np.asarray(Efield["z"])
    dset_rho = np.asarray(fields["rho"])

    Nm = np.shape(dset_Er)[0]
    Nr = np.shape(dset_Er)[1]
    Nz = np.shape(dset_Er)[2]

    # step.geometry = Efield.attrs["geometry"]

    gridGlobalOffset = Efield.attrs["gridGlobalOffset"]

    gridSpacing = Efield.attrs["gridSpacing"]

    gridUnitSI = Efield.attrs["gridUnitSI"]

    # * Density and rho
    rho = np.zeros((2 * Nr, Nz))

    rho[Nr:, :] = np.tensordot([1.0, 1.0, 0.0], dset_rho, axes=(0, 0))[:, :]

    rho[:Nr, :] = np.tensordot([1.0, -1.0, -0.0], dset_rho, axes=(0, 0))[::-1, :]

    step.rho2D = rho

    step.density2D = -step.rho2D / const.e

    # * Er

    Er1D = np.zeros((2 * Nr, Nz))
    Er1D[Nr:, :] = np.tensordot([1.0, 1.0, 0.0], dset_Er, axes=(0, 0))[:, :]
    Er1D[:Nr, :] = np.tensordot([1.0, -1.0, -0.0], dset_Er, axes=(0, 0))[::-1, :]

    n_cells = Nr * 2
    i_cell = int(0.5 * (0 + 1.0) * n_cells)
    i_cell = max(i_cell, 0)
    i_cell = min(i_cell, n_cells - 1)

    Er1D = np.take(Er1D, [i_cell], axis=0)

    step.Er1D = np.squeeze(Er1D)

    Er2D = np.zeros((2 * Nr, Nz))
    Er2D[Nr:, :] = np.tensordot([1, 1, 0], dset_Er, axes=(0, 0))[:, :]
    Er2D[:Nr, :] = np.tensordot([-1, 1, 0], dset_Er, axes=(0, 0))[::-1, :]

    step.Er2D = Er2D

    # * Et

    Et1D = np.zeros((2 * Nr, Nz))
    Et1D[Nr:, :] = np.tensordot([1.0, 1.0, 0.0], dset_Et, axes=(0, 0))[:, :]
    Et1D[:Nr, :] = np.tensordot([1.0, -1.0, -0.0], dset_Et, axes=(0, 0))[::-1, :]

    n_cells = Nr * 2
    i_cell = int(0.5 * (0 + 1.0) * n_cells)
    i_cell = max(i_cell, 0)
    i_cell = min(i_cell, n_cells - 1)

    Et1D = np.take(Et1D, [i_cell], axis=0)
    step.Et1D = np.squeeze(Et1D)

    Et2D = np.zeros((2 * Nr, Nz))
    Et2D[Nr:, :] = np.tensordot([1, 1, 0], dset_Et, axes=(0, 0))[:, :]
    Et2D[:Nr, :] = np.tensordot([-1, 1, 0], dset_Et, axes=(0, 0))[::-1, :]

    step.Et2D = Et2D

    # * Ez

    Ez2D = np.zeros((2 * Nr, Nz))
    Ez2D[Nr:, :] = np.tensordot([1, 1, 0], dset_Ez, axes=(0, 0))[:, :]
    Ez2D[:Nr, :] = np.tensordot([1, -1, 0], dset_Ez, axes=(0, 0))[::-1, :]

    step.Ez2D = Ez2D

    n_cells = Nr * 2
    i_cell = int(0.5 * (0 + 1.0) * n_cells)
    i_cell = max(i_cell, 0)
    i_cell = min(i_cell, n_cells - 1)

    step.Ez1D = np.squeeze(np.take(step.Ez2D, [i_cell], axis=0))

    # Ex and Ey
    theta = 0.0

    step.Ex1D = np.cos(theta) * step.Er1D - np.sin(theta) * step.Et1D

    step.Ey1D = np.sin(theta) * step.Er1D + np.cos(theta) * step.Et1D

    step.Ex2D = np.cos(theta) * step.Er2D - np.sin(theta) * step.Et2D

    step.Ey2D = np.sin(theta) * step.Er2D + np.cos(theta) * step.Et2D

    step.Ey2D_env = np.abs(hilbert(step.Ey2D, axis=1))

    position_check = 0.5
    step.dr = gridSpacing[0] * gridUnitSI
    start = gridGlobalOffset[0] * gridUnitSI + position_check * step.dr
    end = start + (Nr - 1) * step.dr

    rfield = np.linspace(start, end, Nr, endpoint=True)

    step.rfield = np.concatenate((-rfield[::-1], rfield))

    step.Nrfield = len(step.rfield)

    position_check = 0.5
    step.dz = gridSpacing[1] * gridUnitSI
    start = gridGlobalOffset[1] * gridUnitSI + position_check * step.dz
    end = start + (Nz - 1) * step.dz

    step.zfield = np.linspace(start, end, Nz, endpoint=True)

    step.Nzfield = len(step.zfield)

    dt_spectrum = (
        step.zfield[1] - step.zfield[0]
    ) / const.c  # Integration step for the FFT
    fft_field = np.fft.fft(step.Ey1D) * dt_spectrum

    # Take half of the data (positive frequencies only)
    spectrum = abs(fft_field[: int(len(fft_field) / 2)])

    T = (step.zfield[-1] - step.zfield[0]) / const.c
    step_spectrum = 2 * np.pi / T * 1.0
    start = 0.0
    end = start + (np.shape(spectrum)[0] - 1) * step_spectrum
    step.omega = np.linspace(start, end, np.shape(spectrum)[0], endpoint=True)

    i_max = np.argmax(spectrum)

    step.omega0 = step.omega[i_max]

    # * a0
    step.Ey1D_env = np.abs(hilbert(step.Ey1D))

    step.a01D = step.Ey1D_env * const.e / (const.m_e * const.c * step.omega0)

    step.zfield_a0 = step.zfield[np.argmax(np.abs(step.Ey1D_env))]

    step.a0_max = np.amax(step.Ey1D_env) * const.e / (const.m_e * const.c * step.omega0)

    step.a02D = step.Ey2D_env * const.e / (const.m_e * const.c * step.omega0)

    # * FWHM
    ind = np.unravel_index(np.argmax(step.a02D, axis=None), step.a02D.shape)

    r_in_fwhm = step.rfield[
        np.where(step.a02D[:, ind[1]] >= np.max(step.a02D[:, ind[1]]) / 2)
    ]

    step.r_FWHM_intensity_simu = np.max(r_in_fwhm) - np.min(r_in_fwhm)

    z_in_fwhm = step.zfield[
        np.where(step.a02D[ind[0], :] >= np.max(step.a02D[ind[0], :]) / 2)
    ]

    step.z_FWHM_intensity_simu = np.max(z_in_fwhm) - np.min(z_in_fwhm)

    # * waist
    nr = np.size(step.rfield)
    r = np.linspace(np.min(step.rfield), np.max(step.rfield), num=nr)

    nz = np.size(step.zfield)
    z = np.linspace(np.min(step.zfield), np.max(step.zfield), num=nr)

    grid_data = step.Ey2D
    grid_data = np.asarray(grid_data)
    grid_data = grid_data[:, :]

    grid_data = np.square(grid_data)
    Total_energy = np.sum(grid_data)

    for i in range(0, nz):
        grid_data[:, i] = grid_data[:, i] * r[:] * r[:]

    half_waist_intensity = np.sqrt(np.sum(grid_data) / Total_energy)

    step.waist0_max = 2.0 * half_waist_intensity

    return step


def fbpic_particle(step, directory, timestep, species="electrons", **kwargs):
    """Read hdf5 files from FBPIC.

    Args:
        step (Step class): Step class to be used
        directory (str): Path of the data
        timestep (int): Id of the timestep to be taken
        species (str, optional): Species of the beam. Defaults to 'electrons'.

    Returns:
        Step class: Step class to be used
    """
    directory = directory + "/hdf5"
    filename = directory + "/data" + str(timestep).zfill(8) + ".h5"

    if kwargs.get("verbose", True):
        print(f"Read file {filename} for species: {species}")

    # Open file
    f = h5py.File(filename, "r")

    data = f["data"]
    data = data[str(timestep)]

    get_data_attr(step, data, timestep)

    #### Particles
    particles = data["particles"]
    particles = particles[
        species
    ]  # ['charge', 'id', 'mass', 'momentum', 'position', 'positionOffset', 'weighting']

    momentum = particles["momentum"]  # ['x', 'y', 'z']
    position = particles["position"]  # ['x', 'y', 'z']

    # Add beam particles
    step.species = species

    step.w = np.array(particles["weighting"])

    step.ux = np.array(momentum["x"]) / (const.m_e * const.c)

    step.uy = np.array(momentum["y"]) / (const.m_e * const.c)

    step.uz = np.array(momentum["z"]) / (const.m_e * const.c)

    step.x = np.array(position["x"])

    step.y = np.array(position["y"])

    step.z = np.array(position["z"])

    # Compute beam parameters
    step.get_beam(**kwargs)

    return step
