"""steps.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np
import os
import scipy.constants as const
from typing import Any, Optional

# twissed
from ..utils.metadata import MetaData
from ..utils import physics
from ..step.step import Step
from ..io.fbpic import fbpic_particle, fbpic_field, get_fbpic_timesteps
from ..io.happi import (
    happi_particle,
    happi_probe1D,
    happi_probe2D,
    get_happi_timesteps_Probe0,
    get_happi_timesteps_Probe1,
    get_happi_timesteps_TrackParticles,
)
from ..io.smilei import get_smilei_timesteps, smilei_field, smilei_particle

try:
    from .plotbeam import StepsPlotBeam
except ImportError:

    class StepsPlotBeam:
        pass


DATA_BEAM_TO_STEPS = [
    "time",
    "z_avg",
    "charge",
    "Ek_avg",
    "Ek_med",
    "Ek_std",
    "Ek_mad",
    "Ek_std_perc",
    "Ek_mad_perc",
    "emit_rms_x",
    "emit_rms_y",
    "emit_norm_rms_x",
    "emit_norm_rms_y",
    "alpha_x",
    "alpha_y",
    "beta_x",
    "beta_y",
    "gamma_x",
    "gamma_y",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "sigma_xp",
    "sigma_yp",
]


DATA_FIELD_TO_STEPS = [
    "zfield_a0",
    "a0_max",
    "r_FWHM_intensity_simu",
    "z_FWHM_intensity_simu",
    "waist0_max",
]


class Steps(MetaData, StepsPlotBeam):
    """
    General class to investigate a full single LPA run
    """

    def __init__(self) -> None:
        pass

    def find_data(
        self, directory: str = os.getcwd(), source: str = "fbpic", verbose=False
    ):
        self.verbose = verbose

        for name in DATA_BEAM_TO_STEPS:
            self.__dict__[name] = np.array([])

        for name in DATA_FIELD_TO_STEPS:
            self.__dict__[name] = np.array([])

        self.x_charge = []
        self.y_charge = []
        self.z_charge = []

        self.directory = directory

        self.source = source

        if self.source == "fbpic":
            self.timesteps = get_fbpic_timesteps(directory, verbose=self.verbose)

        elif self.source == "smilei":
            self.trackpart_list = [
                i for i in os.listdir(directory) if i.startswith("TrackParticles")
            ]

            self.species_list = [
                i[i.find("_") + 1 : i.find(".h5")] for i in self.trackpart_list
            ]

            self.probe_list = [
                i for i in os.listdir(directory) if i.startswith("Probes")
            ]

            self.diag_list = self.trackpart_list + self.probe_list

            if self.verbose:
                print(f"INFO: files {self.diag_list} found.")

            self.timesteps_all = []
            for filename in self.diag_list:
                file = os.path.join(self.directory, filename)
                self.timesteps_all.append(get_smilei_timesteps(file))

            self.lambda0 = 0.8 * 1e-6

            self.conversion_factor = self.lambda0 / 2.0 / np.pi * 1e6

            self.n_crit = physics.critical_density(self.lambda0)

            self.omega0 = physics.omega_laser(self.lambda0)

        elif self.source == "happi":
            self.timesteps_probe0 = get_happi_timesteps_Probe0(
                directory, verbose=self.verbose
            )
            self.timesteps_probe1 = get_happi_timesteps_Probe1(
                directory, verbose=self.verbose
            )
            self.timesteps_trackpart = get_happi_timesteps_TrackParticles(
                directory, verbose=self.verbose
            )

            self.timesteps = self.timesteps_trackpart

            self.lambda0 = 0.8 * 1e-6

            self.conversion_factor = self.lambda0 / 2.0 / np.pi * 1e6

            self.n_crit = physics.critical_density(self.lambda0)

            self.omega0 = physics.omega_laser(self.lambda0)

    def read_field(self, step: Step, timestep: int, **kwargs) -> Step:
        if self.source == "fbpic":
            step = fbpic_field(
                step, self.directory, timestep, verbose=kwargs.get("verbose", False)
            )
            return step

        elif self.source == "smilei":
            if kwargs.get("probe", False):
                filenames = [
                    x
                    for x in self.diag_list
                    if x == "Probes" + str(kwargs.get("probe", -1)) + ".h5"
                ]
            else:
                filenames = [x for x in self.diag_list if x.startswith("Probes")]

            for filename in filenames:
                file = os.path.join(self.directory, filename)
                if self.verbose:
                    print(f"INFO: Reading {file} at timestep: {timestep}")

                step = smilei_field(
                    step,
                    file,
                    timestep,
                    self.omega0,
                    self.n_crit,
                )
            return step

        elif self.source == "happi":
            if kwargs.get("probe", -1) == 0:
                step = happi_probe1D(
                    step,
                    self.directory,
                    timestep,
                    self.lambda0,
                    self.omega0,
                    self.n_crit,
                    verbose=kwargs.get("verbose", False),
                )
                return step

            elif self.source == "happi" and kwargs.get("probe", -1) == 1:
                step = happi_probe2D(
                    step,
                    self.directory,
                    timestep,
                    self.lambda0,
                    self.omega0,
                    self.n_crit,
                    verbose=kwargs.get("verbose", False),
                )
                return step

    def read_beam(
        self,
        step: Step,
        timestep: int,
        species: str,
        Disordered: Optional[bool] = None,
        **kwargs: Any,
    ) -> Step:
        if self.source == "fbpic":
            step = fbpic_particle(
                step,
                self.directory,
                timestep,
                species=species,
                **kwargs,
            )
            return step

        elif self.source == "smilei":
            if species not in self.species_list:
                raise ValueError("ERROR: species not found")
            else:
                files = [s for s in self.diag_list if species in s]
                if len(files) > 1:
                    if Disordered is None:
                        print(
                            "WARNING: Ordered and disordered file found for the same species. Disordered file selected"
                        )
                        filename = [s for s in files if "Disordered" in s][0]
                        Disordered = True
                    elif Disordered:
                        filename = [s for s in files if "Disordered" in s][0]
                    elif not Disordered:
                        filename = [s for s in files if "Disordered" not in s][0]
                else:
                    filename = files[0]
                    if Disordered is None:
                        if "Disordered" in filename:
                            Disordered = True
                        else:
                            Disordered = False

                file = os.path.join(self.directory, filename)

                # print(self.timesteps_all[self.diag_list.index(filename)])

                if timestep == -1:
                    timestep = self.timesteps_all[self.diag_list.index(filename)][-1]
                elif timestep not in self.timesteps_all[self.diag_list.index(filename)]:
                    raise ValueError(f"ERROR: timestep {timestep} not in the list.")

                if self.verbose:
                    print(f"INFO: Reading {file} at timestep: {timestep}")

                step = smilei_particle(
                    step, file, timestep, species, self.omega0, Disordered=Disordered
                )
                return step

        elif self.source == "happi":
            step = happi_particle(
                step,
                self.directory,
                timestep,
                species,
                self.lambda0,
                self.omega0,
                self.conversion_factor,
                self.n_crit,
                chunksize=kwargs.get("chunksize", 50000000),
                **kwargs,
            )
            return step

    def get_step(
        self,
        step: Step,
        xrange=[None, None],
        xconv=None,
        yrange=[None, None],
        yconv=None,
        zrange=[None, None],
        zconv=None,
        bins=75,
    ) -> None:
        # * Have a beam
        if step.N > 10:
            for name in DATA_BEAM_TO_STEPS:
                self.__dict__[name] = np.append(
                    self.__dict__[name], step.__dict__[name]
                )

            # * Beam size hist
            # x
            H, xedges = step.hist1D(
                "x", xconv=xconv, xrange=xrange, bins=bins, plot=None
            )
            self.x_charge_pos = xedges[:-1]
            self.x_charge.append(H)

            # y
            H, xedges = step.hist1D(
                "y", xconv=yconv, xrange=yrange, bins=bins, plot=None
            )
            self.y_charge_pos = xedges[:-1]
            self.y_charge.append(H)

            # z
            H, xedges = step.hist1D(
                "z_avg", xconv=zconv, xrange=zrange, bins=bins, plot=None
            )
            self.z_charge_pos = xedges[:-1]
            self.z_charge.append(H)

        # * Do not have a beam
        else:
            for name in DATA_BEAM_TO_STEPS:
                self.__dict__[name] = np.append(self.__dict__[name], 0.0)

            self.x_charge.append(np.zeros(bins))
            self.y_charge.append(np.zeros(bins))
            self.z_charge.append(np.zeros(bins))

        # * Fields
        for name in DATA_FIELD_TO_STEPS:
            self.__dict__[name] = np.append(self.__dict__[name], step.__dict__[name])
