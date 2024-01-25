"""tracewin.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""


import numpy as np
import scipy.constants as const
import struct
from typing import Optional

# twissed
from ..step.step import Step


def read_dst(filename: str, charge: Optional[float] = None, **kwargs) -> Step:
    """Read dst file into a Step class. All information on the beam are computed.

    Example
    -------

    .. code-block:: python

        step = twissed.read_dst("treacewin.dst", charge = 100, inverse_x_and_y=True)


    .. todo::

        Check charge in pC.
        Check z distribution.

    Args:
        filename (str): Path of the dst file.
        charge (float, optional): Overwrite the total charge to the given amount in pC.

    Returns:
        Step: class containing beam information :meth:`twissed.step`.

    Other Parameters
    ----------------
    **kwargs : List of properties
        * verbose (bool): Display information. Defaults to True.
        * inverse_x_and_y (bool): Inverse the x and y axis. Defaults to False.
    """

    f = open(filename, "rb")  # opening a binary file
    content = f.read()

    header = list(struct.unpack("<cciddc", content[0:23]))

    N = header[2]
    current = header[3]
    freq = header[4] * 1e6

    # ! Todo check charge in pC !

    charge_applicable = current / freq
    if charge is not None:
        charge_applicable = charge

    if charge != 0:
        weight = charge_applicable / const.e / 1e12 / N
    else:
        weight = 1.0

    last_id = 23

    x = []
    y = []
    t = []
    xp = []
    yp = []
    W = []

    for i in range(N):
        part = list(struct.unpack("<dddddd", content[last_id : last_id + 8 * 6]))

        if kwargs.get("inverse_x_and_y", False):
            x.append(part[2])
            xp.append(part[3])
            y.append(part[0])
            yp.append(part[1])
        else:
            x.append(part[0])
            xp.append(part[1])
            y.append(part[2])
            yp.append(part[3])

        t.append(part[4])
        W.append(part[5])

        last_id = last_id + 8 * 6

    Erest = list(struct.unpack("<d", content[last_id : last_id + 8 * 6]))[0] / 1e-6

    x = np.array(x) / 100.0
    y = np.array(y) / 100.0
    xp = np.array(xp)
    yp = np.array(yp)

    W = np.array(W) / 1e-6
    gammaloc = W / Erest + 1.0

    uz = np.sqrt((gammaloc**2 - 1) / (xp**2 + yp**2 + 1.0))

    t = np.array(t)

    ux = xp * uz
    uy = yp * uz

    z = -t * uz / gammaloc * const.c / 2.0 / const.pi / freq

    step = Step()

    step.set_new_6D_beam(x, y, z, ux, uy, uz, np.ones(N) * weight)

    step.get_beam()

    if kwargs.get("verbose", True):
        print(f"INFO: .dst file read with N particle: {step.N} and charge {charge}")

    return step


def write_dst(step: Step, filename: str, freq: Optional[float] = 1e9, **kwargs) -> None:
    """Write a .dst file for the beam.


    Warning
    -------

    The weigh of particle must be equal!


    Example
    -------

    .. code-block:: python

        twissed.write_dst(step, "treacewin.dst", freq=1e9, inverse_x_and_y=True)


    Args:
        step (Step): class containing beam information :meth:`twissed.step`.
        filename (str): Path + name of the file
        freq (float,optional): Frequency. Default to 1 GHz.


    Other Parameters
    ----------------
    **kwargs : List of properties
        * verbose (bool): Display information. Defaults to True.
        * inverse_x_and_y (bool): Inverse the x and y axis. Defaults to False. Warning: It is not equal to rotation -90° !
    """

    inverse_x_and_y = kwargs.get(
        "inverse_x_and_y", False
    )  # Warning ! Not equal to rotation -90° !

    dtype_TWin = [
        ("x", np.float64),
        ("xp", np.float64),
        ("y", np.float64),
        ("yp", np.float64),
        ("t", np.float64),
        ("W", np.float64),
    ]

    charge = np.sum(step.w) * const.e

    arr_z = step.z - np.average(step.z, weights=step.w)

    me_eV = const.m_e * const.c**2 / const.e
    Erest = me_eV

    tab_end = np.zeros((np.size(step.w)), dtype=dtype_TWin)

    gammaloc = np.sqrt(1.0 + step.ux**2 + step.uy**2 + step.uz**2)

    if not inverse_x_and_y:
        tab_end["x"] = step.x
        tab_end["y"] = step.y
        tab_end["xp"] = step.ux / step.uz
        tab_end["yp"] = step.uy / step.uz
    else:
        tab_end["x"] = step.y
        tab_end["y"] = step.x
        tab_end["xp"] = step.uy / step.uz
        tab_end["yp"] = step.ux / step.uz
    tab_end["t"] = (
        -arr_z / step.uz / const.c * gammaloc * 2.0 * const.pi * freq
    )  # PHASE Phi
    tab_end["W"] = (gammaloc - 1.0) * Erest  # = self.Ek

    cur = charge * freq

    # apert=10e-6
    # apert=0
    # if apert > 0:
    #     mask1 = np.abs(tab_end["x"])<apert
    #     tab_end = tab_end[mask1]
    #     mask1 = np.abs(tab_end["y"])<apert
    #     tab_end = tab_end[mask1]

    center = True
    if center:
        tab_end["x"] -= tab_end["x"].mean()
        tab_end["y"] -= tab_end["y"].mean()
        tab_end["xp"] -= tab_end["xp"].mean()
        tab_end["yp"] -= tab_end["yp"].mean()
        tab_end["t"] -= tab_end["t"].mean()

    with open(filename, "wb") as fout:
        fout.write(
            struct.pack(
                "<cciddc",
                chr(125).encode("ascii"),
                chr(100).encode("ascii"),
                len(tab_end),
                cur * 1e3,
                freq * 1e-6,
                chr(0).encode("ascii"),
            )
        )
        for line_data in tab_end:
            fout.write(
                struct.pack(
                    "<dddddd",
                    line_data["x"] * 100.0,
                    line_data["xp"],
                    line_data["y"] * 100.0,
                    line_data["yp"],
                    line_data["t"],
                    line_data["W"] * 1e-6,
                )
            )
        fout.write(struct.pack("<d", Erest * 1e-6))

    if kwargs.get("verbose", True):
        print(f"INFO: .dst file {filename} wrote with N particle: {step.N}.")
