"""astra.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np
import scipy.constants as const

# twissed
from ..step.step import Step


def read_astra(file_path, remove_non_standard=True):
    """Reads particle data from ASTRA and returns it in the unis used by
    APtools.

    Parameters
    ----------
    file_path : str
        Path to the file with particle data

    remove_non_standard : bool
        Determines whether non-standard particles (those with a status flag
        other than 5) should be removed from the read data.

    Returns
    -------
    A tuple with 7 arrays containing the 6D phase space and charge of the
    particles.
    """
    data = np.genfromtxt(file_path)
    status_flag = data[:, 9]
    if remove_non_standard:
        data = data[np.where(status_flag == 5)]
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    px = data[:, 3] / (const.m_e * const.c**2 / const.e)
    py = data[:, 4] / (const.m_e * const.c**2 / const.e)
    pz = data[:, 5] / (const.m_e * const.c**2 / const.e)
    z[1:] += z[0]
    pz[1:] += pz[0]
    q = data[:, 7] * 1e-9

    step = Step()
    step.set_new_6D_beam(x, y, z, px, py, pz, q)
    step.get_beam()

    return step


def write_astra(step: Step, filename: str, **kwargs):
    """Saves particle data in ASTRA format.

    Parameters
    ----------
    beam_data : list
        Contains the beam data as [x, y, z, px, py, pz, q], where the positions
        have units of meters, momentun is in non-dimensional units (beta*gamma)
        and q is in Coulomb.

    folder_path : str
        Path to the folder in which to save the data

    file_name : str
        Name of the file to save without extension

    reposition : bool
        Optional. Whether to reposition de particle distribution in space
        and/or momentum centered in the coordinates specified in avg_pos and
        avg_mom

    avg_pos : list
        Optional, only used it reposition=True. Contains the new average
        positions of the beam after repositioning. Should be specified as
        [x_avg, y_avg, z_avg] in meters. Setting a component as None prevents
        repositioning in that coordinate.

    avg_mom : list
        Optional, only used it reposition=True. Contains the new
        average momentum of the beam after repositioning. Should be specified
        as [px_avg, py_avg, pz_avg] in non-dimmesional units (beta*gamma).
        Setting a component as None prevents repositioning in that coordinate.

    n_part : int
        Optional. Number of particles to save. Must be lower than the original
        number of particles. Particles to save are chosen randomly.
    """

    # Get beam data
    x_orig = step.x
    y_orig = step.y
    xi_orig = step.z
    px_orig = step.ux * const.m_e * const.c**2 / const.e
    py_orig = step.uy * const.m_e * const.c**2 / const.e
    pz_orig = step.uz * const.m_e * const.c**2 / const.e
    q_orig = step.w * 1e9  # nC

    # Create arrays
    x = np.zeros(q_orig.size + 1)
    y = np.zeros(q_orig.size + 1)
    xi = np.zeros(q_orig.size + 1)
    px = np.zeros(q_orig.size + 1)
    py = np.zeros(q_orig.size + 1)
    pz = np.zeros(q_orig.size + 1)
    q = np.zeros(q_orig.size + 1)

    # Reference particle
    x[0] = np.average(x_orig, weights=q_orig)
    y[0] = np.average(y_orig, weights=q_orig)
    xi[0] = np.average(xi_orig, weights=q_orig)
    px[0] = np.average(px_orig, weights=q_orig)
    py[0] = np.average(py_orig, weights=q_orig)
    pz[0] = np.average(pz_orig, weights=q_orig)
    q[0] = sum(q_orig) / len(q_orig)

    # Put relative to reference particle
    x[1::] = x_orig
    y[1::] = y_orig
    xi[1::] = xi_orig - xi[0]
    px[1::] = px_orig
    py[1::] = py_orig
    pz[1::] = pz_orig - pz[0]
    q[1::] = q_orig
    t = xi / const.c

    # Add flags and indices
    ind = np.ones(q.size)
    flag = np.ones(q.size) * 5

    # Save to file
    data = np.column_stack((x, y, xi, px, py, pz, t, q, ind, flag))
    np.savetxt(
        filename, data, "%1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %1.12e %i %i"
    )
