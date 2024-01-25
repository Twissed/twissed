"""noweight.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np
import random
import scipy.constants as const

# twissed
from ..step.step import Step


def noweight(step: Step, weight_norm: float = 1.0, method: int = 1, **kwargs) -> Step:
    """Remove weight of the distribution. Zero method with random for ratio.

    Might create more weight than original distribution.

    Use
    ---

    * Define a new weight `weight_norm`.
    * For each particle.
        * if w > weight_norm
            * Create N new particles with same position and momenta.
            with `N = int(np.floor(w/weight_norm))`

    Args:
        step (Step class): Step class
        weight_norm (float, optional): New weight. Defaults to 1.
        method (int,optional): Type of method used (1 better than 0).

    Returns:
        Step class: A new Step class
    """

    x_new = []
    y_new = []
    z_new = []
    ux_new = []
    uy_new = []
    uz_new = []
    w_new = []

    if method == 1:
        for id in range(step.N):
            w_t = step.w[id]
            N_int = int(np.floor(w_t / weight_norm))

            for j in range(N_int):
                w_new.append(weight_norm)
                x_new.append(step.x[id])
                y_new.append(step.y[id])
                z_new.append(step.z[id])
                ux_new.append(step.ux[id])
                uy_new.append(step.uy[id])
                uz_new.append(step.uz[id])

            N_rat = (w_t / weight_norm) - np.floor(w_t / weight_norm)

            if N_rat > random.random():
                w_new.append(weight_norm)
                x_new.append(step.x[id])
                y_new.append(step.y[id])
                z_new.append(step.z[id])
                ux_new.append(step.ux[id])
                uy_new.append(step.uy[id])
                uz_new.append(step.uz[id])

    elif method == 0:
        for id in range(step.N):
            if step.w[id] > weight_norm:
                w_t = step.w[id]
                N = int(np.floor(w_t / weight_norm))

                for j in range(N):
                    w_new.append(weight_norm)
                    x_new.append(step.x[id])
                    y_new.append(step.y[id])
                    z_new.append(step.z[id])
                    ux_new.append(step.ux[id])
                    uy_new.append(step.uy[id])
                    uz_new.append(step.uz[id])

    step_new = Step()
    step_new.set_new_6D_beam(
        x_new,
        y_new,
        z_new,
        ux_new,
        uy_new,
        uz_new,
        np.array(w_new) * np.sum(step.w) / np.sum(np.array(w_new)),
    )
    step_new.get_beam(verbose=kwargs.get("verbose", True))

    if kwargs.get("verbose", True):
        print(f"Nparticle  Old: {step.N}, New: {step_new.N}")

        print(
            f"Emittance x - Old: {step.emit_norm_rms_x}, New: {step_new.emit_norm_rms_x}"
        )
        print(
            f"Emittance y - Old: {step.emit_norm_rms_y}, New: {step_new.emit_norm_rms_y}"
        )
        print(f"Energy mean - Old: {step.Ek_avg}, New: {step_new.Ek_avg}")

        print(
            f"Charge - Old: {step.charge}, Saved: {np.sum(np.array(w_new)) * const.e * 1e12 }, New: {step_new.charge}"
        )

    return step_new
