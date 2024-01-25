"""happy.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""


import numpy as np
import scipy.constants as const

try:
    import happi
except ImportError:
    print("WARNING: happi package not found")

# twissed
from ..utils.stats import *


def get_happi_timesteps_Probe0(directory, verbose=False):
    S = happi.Open(directory)
    diag = S.Probe.Probe0("Ex")
    timesteps = diag.getTimesteps()

    timesteps.sort()
    if verbose:
        print(f"timesteps: {timesteps}")

    return timesteps


def get_happi_timesteps_Probe1(directory, verbose=False):
    S = happi.Open(directory)
    diag = S.Probe.Probe1("Ex")
    timesteps = diag.getTimesteps()

    timesteps.sort()
    if verbose:
        print(f"timesteps: {timesteps}")

    return timesteps


def get_happi_timesteps_TrackParticles(directory, verbose=False):
    S = happi.Open(directory)

    track_part = S.TrackParticles(
        species="electronfromion", chunksize=5000000, sort=False
    )
    timesteps = track_part.getTimesteps()

    timesteps.sort()
    if verbose:
        print(f"timesteps: {timesteps}")

    return timesteps


def read_happi_time(step, S, omega0, timestep):
    step.dt = S.namelist.dt / omega0
    step.timestep = timestep
    step.time = step.timestep * step.dt
    return step


def happi_probe1D(step, directory, timestep, lambda0, omega0, nc, **kwargs):
    verbose = kwargs.get("verbose", False)

    S = happi.Open(directory)
    probe = S.Probe

    step = read_happi_time(step, S, omega0, timestep)

    step.zfield_moved = (
        probe.Probe0("Env_E_abs", timesteps=timestep).getXmoved(timestep)
        * lambda0
        / (2 * np.pi)
    )

    zfield = (
        probe.Probe0("Env_E_abs", timesteps=timestep).getAxis("axis1")
        * lambda0
        / (2 * np.pi)
        + step.zfield_moved
    )

    step.zfield = zfield

    step.a01D = np.asarray(probe.Probe0("Env_E_abs", timesteps=timestep).getData())[
        0, :
    ]

    step.zfield_a0 = step.zfield[np.argmax(np.abs(step.a))]

    step.a0_max = np.amax(step.a)

    # Electric field in z
    step.Ez1D = np.asarray(probe.Probe0("Ex", timesteps=timestep).getData())[0, :] * (
        const.m_e * const.c / const.e * omega0
    )

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


def happi_probe2D(step, directory, timestep, lambda0, omega0, nc, **kwargs):
    verbose = kwargs.get("verbose", False)

    S = happi.Open(directory)
    probe = S.Probe

    step = read_happi_time(step, S, omega0, timestep)

    step.zfield_moved = (
        probe.Probe1("Env_E_abs", timesteps=timestep).getXmoved(timestep)
        * lambda0
        / (2 * np.pi)
    )
    zfield = (
        probe.Probe1("Env_E_abs", timesteps=timestep).getAxis("axis1")
        * lambda0
        / (2 * np.pi)
        + step.zfield_moved
    )
    step.zfield = zfield[:, 0]

    step.rfield = (
        probe.Probe1("Env_E_abs", timesteps=timestep).getAxis("axis2")[:, 1]
        * lambda0
        / (2 * np.pi)
    )

    step.Ey2D_env = np.asarray(probe.Probe1("Env_E_abs", timesteps=timestep).getData())[
        0, :, :
    ]
    step.Ey2D_env = step.Ey2D_env.T

    step.Ez2D = np.asarray(probe.Probe1("Ex", timesteps=timestep).getData())[0, :, :]
    step.Ez2D = step.Ez2d.T

    ne = np.asarray(probe.Probe1("Rho", timesteps=timestep).getData())[0, :, :]
    step.density2D = -ne.T * nc

    step.Er2D = np.asarray(probe.Probe1("Ey", timesteps=timestep).getData())[0, :, :]
    step.Er2D = step.Er2D.T

    return step


def happi_particle(
    step,
    directory,
    timestep,
    species,
    lambda0,
    omega0,
    conversion_factor,
    n_crit,
    **kwargs,
):
    chunksize = kwargs.get("chunksize", 50000000)

    S = happi.Open(directory)
    track_part = S.TrackParticles(species=species, chunksize=chunksize, sort=False)

    step = read_happi_time(step, S, omega0, timestep)

    x = []
    y = []
    z = []
    ux = []
    uy = []
    uz = []
    w = []

    for particle_chunk in track_part.iterParticles(timestep, chunksize=chunksize):
        # TODO : Check !!!
        ### Read particles arrays with positions and momenta
        # positions
        z = (
            particle_chunk["moving_x"] * lambda0 / (2 * np.pi)
        )  # takes into account moving window
        y = particle_chunk["y"] * lambda0 / (2 * np.pi)
        x = particle_chunk["z"] * lambda0 / (2 * np.pi)  # Switch between x and z !!!

        # momenta
        ux = particle_chunk["pz"]
        uy = particle_chunk["py"]
        uz = particle_chunk["px"]  # Switch between x and z !!!

        w = particle_chunk["w"] * n_crit * (conversion_factor * 1e-6) ** 3

    # Add beam particles
    step.species = species

    step.w = np.array(w)

    step.ux = np.array(ux)

    step.uy = np.array(uy)

    step.uz = np.array(uz)

    step.x = np.array(x)

    step.y = np.array(y)

    step.z = np.array(z)

    # Compute beam parameters
    step.get_beam(**kwargs)

    return step


# TO DEL !!!!
def getTimesteps(S):
    diag = S.Probe.Probe1("Ex")
    return diag.getTimesteps()


# TO DEL !!!
def getTimesteps_part(S, species_name):
    track_part = S.TrackParticles(species=species_name, chunksize=5000000, sort=False)
    return track_part.getTimesteps()


def read_happi_TrackParticles(
    step,
    S,
    track_part,
    lambda0,
    omega0,
    conversion_factor,
    n_crit,
    timestep,
    chunk_size=50000000,
    **kwargs,
):
    """Create Step class from happi

    Args:
        lambda0 (_type_): _description_
        omega_0 (_type_): _description_
        conversion_factor (_type_): _description_
        timestep (_type_): _description_
        chunk_size (int, optional): _description_. Defaults to 50000000.
        directory (str, optional): _description_. Defaults to "".
        species (str, optional): _description_. Defaults to 'electrons'.
    """

    step = read_happi_time(step, S, omega0, timestep)

    for particle_chunk in track_part.iterParticles(timestep, chunksize=chunk_size):
        ### Read particles arrays with positions and momenta
        # positions
        step.z = (
            particle_chunk["moving_x"] * lambda0 / (2 * np.pi)
        )  # takes into account moving window
        step.y = particle_chunk["y"] * lambda0 / (2 * np.pi)
        step.x = particle_chunk["z"] * lambda0 / (2 * np.pi)

        # momenta
        step.ux = particle_chunk["pz"]
        step.uy = particle_chunk["py"]
        step.uz = particle_chunk["px"]

        step.w = particle_chunk["w"] * n_crit * (conversion_factor * 1e-6) ** 3

        # Compute beam parameters
        step.get_beam(**kwargs)

    return step


def read_happi_Probe0_Env_E_abs(
    step, S, probe, lambda0, omega_0, conversion_factor, n_crit, timestep
):
    step = read_happi_time(step, S, omega_0, timestep)

    step.zfield_moved = (
        probe.Probe0("Env_E_abs", timesteps=timestep).getXmoved(timestep)
        * lambda0
        / (2 * np.pi)
    )
    step.zfield = (
        probe.Probe0("Env_E_abs", timesteps=timestep).getAxis("axis1")
        * lambda0
        / (2 * np.pi)
        + step.zfield_moved
    )

    step.a01D = np.asarray(probe.Probe0("Env_E_abs", timesteps=timestep).getData())[
        0, :
    ]
    step.zfield_a0 = step.zfield[np.argmax(np.abs(step.a))]
    step.a0_max = np.amax(step.a)

    return step


def read_happi_Probe1(
    step, S, probe, lambda0, omega_0, conversion_factor, n_crit, timestep
):
    step = read_happi_time(step, S, omega_0, timestep)

    step.zfield_moved = (
        probe.Probe1("Env_E_abs", timesteps=timestep).getXmoved(timestep)
        * lambda0
        / (2 * np.pi)
    )
    step.zfield = (
        probe.Probe1("Env_E_abs", timesteps=timestep).getAxis("axis1")
        * lambda0
        / (2 * np.pi)
        + step.zfield_moved
    )
    step.zfield = step.zfield[:, 0]

    step.rfield = (
        probe.Probe1("Env_E_abs", timesteps=timestep).getAxis("axis2")[:, 1]
        * lambda0
        / (2 * np.pi)
    )

    step.Ey2D_env = np.asarray(probe.Probe1("Env_E_abs", timesteps=timestep).getData())[
        0, :, :
    ]
    step.Ey2D_env = step.Ey2D_env.T

    step.Ez1D = np.asarray(probe.Probe0("Ex", timesteps=timestep).getData())[0, :]

    density2D = np.asarray(probe.Probe1("Rho", timesteps=timestep).getData())[0, :, :]
    step.density2D = -density2D.T * n_crit

    step.Er2D = np.asarray(probe.Probe1("Ey", timesteps=timestep).getData())[0, :, :]
    step.Er2D = step.Er2D.T

    return step
