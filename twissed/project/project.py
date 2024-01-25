"""project.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""


from datetime import datetime
import h5py
import json
import numpy as np
import pandas as pd
import os
import shutil
from typing import Any, List, Optional, Union

# twissed
from ..step.step import Step
from ..steps.steps import Steps
from ..utils.data import STD_DATABASE
from ..utils.tools import create_folder, make_executable

__version__ = "2.0.2"

PROJECT_PARAMETERS: List[str] = [
    "twissed_version",
    "project_name",
    "project_author",
    "project_description",
    "project_type",
    "project_first_id",
    "project_date",
    "project_directory",
    "project_source",
    "h5filename",
]


RUN_OUTPUT_PARAMETERS: List[str] = [
    "dt",
    "time",
    "timestep",
    "species",
    "charge",
    "N",
    "Ek_avg",
    "Ek_med",
    "Ek_std",
    "Ek_mad",
    "Ek_std_perc",
    "Ek_mad_perc",
    "g_avg",
    "g_med",
    "g_std",
    "g_mad",
    "betaz_avg",
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "sigma_xp",
    "sigma_yp",
    "sigma_dp",
    "x_divergence",
    "y_divergence",
    "x_avg",
    "y_avg",
    "z_avg",
    "xx_avg",
    "yy_avg",
    "ux_avg",
    "uy_avg",
    "uz_avg",
    "xp_avg",
    "yp_avg",
    "xpxp_avg",
    "ypyp_avg",
    "xxp_avg",
    "yyp_avg",
    "emit_rms_x",
    "emit_rms_y",
    "emit_rms_z",
    "emit_norm_rms_x",
    "emit_norm_rms_y",
    "emit_norm_rms_z",
    "emit_norm_rms_4D",
    "emit_norm_rms_6D",
    "beta_x",
    "beta_y",
    "beta_z",
    "gamma_x",
    "gamma_y",
    "gamma_z",
    "alpha_x",
    "alpha_y",
    "alpha_z",
    "Ek_hist_yaxis",
    "Ek_hist_xaxis",
    "Ek_hist_peak",
    "Ek_hist_fwhm",
    "x_dispersion",
    "y_dispersion",
]


class RunControler:
    """Class to control runs."""

    def __init__(self, directory: str = os.getcwd()) -> None:
        # Find run_h5file
        listdir = [i for i in os.listdir(directory) if i.startswith("Run_")]
        if len(listdir) == 0:
            print("No file Run_xxx.sky.h5 found!")
        elif len(listdir) > 1:
            print("Too many file Run_xxx.sky.h5 found!")
        else:
            folder_name = listdir[0]
            self.h5filename = os.path.join(directory, folder_name)
            print(f"File {self.h5filename} found!")

            self.h5file = h5py.File(self.h5filename, "r+")

            for key in self.h5file["input"].keys():
                self.__dict__[key] = self.h5file["input"][key][()]

            if "output_status" in list(self.h5file.keys()):
                self.output_status = self.h5file["output_status"]

                # ! Do it at timestep -1
                # for key in self.h5file["output"].keys():
                #     self.__dict__[key] = self.h5file["output"][key][()]

            else:
                self.output_status = False

    def close(self):
        """Close h5file"""

        try:
            getattr(self, "h5file")
            self.h5file.close
        except AttributeError:
            print("Run does not have attribute 'h5file'.")


class RunReader(object):
    def __init__(
        self,
        id: int,
        h5group: Optional[h5py.File] = None,
        directory: Optional[str] = os.getcwd(),
        timesteps: Optional[Union[int, List[int]]] = None,
    ) -> None:
        self.folder_name = "Run_" + str(id).zfill(8)

        if h5group is None:
            h5filename = os.path.join(directory, self.folder_name)
            self.h5file = h5py.File(h5filename, "r+")
            print(f"File {h5filename} found!")
        else:
            self.h5file = h5group
            print(f"Group {h5group} found!")

        for key in list(self.h5file.keys()):
            if isinstance(self.h5file[key], h5py.Dataset):
                self.__dict__[key] = self.h5file[key][()]

        self.input = {}
        for key in list(self.h5file["input"].keys()):
            self.__dict__[key] = self.h5file["input"][key][()]
            self.input[key] = self.h5file["input"][key][()]

        if "output" in list(self.h5file.keys()):
            self.timesteps_avail = self.h5file["output"]["timesteps"][()]

            if timesteps is None:
                self.timesteps = self.timesteps_avail
            elif timesteps == -1:
                self.timesteps = [self.timesteps_avail[-1]]
            else:
                self.timesteps = (
                    timesteps if isinstance(timesteps, list) else [timesteps]
                )

            self.step = []
            for i, timestep in enumerate(self.timesteps):
                self.step.append(Step())

                for key in list(self.h5file["output"][str(timestep)].keys()):
                    setattr(
                        self.step[i], key, self.h5file["output"][str(timestep)][key][()]
                    )

        # End run
        if h5group is None:
            self.h5file.close()

    def steps_DataFrame(self) -> pd.DataFrame:
        if "timesteps" in self.__dict__.keys():
            if len(self.timesteps) == 1:
                df = self.step[0].DataFrame()
                df["timestep"] = self.timesteps[0]
                for i, (key, val) in enumerate(self.input.items()):
                    df[key] = val
                return df

            elif len(self.timesteps) > 1:
                df = pd.DataFrame()
                for i, timestep in enumerate(self.timesteps):
                    df1 = self.step[i].DataFrame()
                    df1["timestep"] = timestep
                    for i, (key, val) in enumerate(self.input.items()):
                        df1[key] = val
                    df = pd.concat([df, df1])

                df.set_index("timestep", inplace=True)
                return df
        else:
            df = pd.DataFrame(self.input, index=[0])
            return df


def h5set_attr(h5f: h5py.File, key: str) -> None:
    """Create attribute to h5 file

    Args:
        h5f (h5py.File): _description_
        key (str): _description_
    """
    if key in STD_DATABASE:
        for attribute in STD_DATABASE[key].keys():
            h5f[key].attrs[attribute] = STD_DATABASE[key][attribute]


def h5copy_group(name: str, source: h5py.File, destination: h5py.File) -> None:
    source.copy(name, destination)


class Project:
    def __init__(self, **kwargs: Any) -> None:
        for name in kwargs:
            self.__dict__[name] = kwargs.get(name, 0)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def keys(self) -> List:
        """Display all the keys.

        Returns:
            List: List of all the keys.
        """
        return list(self.__dict__.keys())

    def new_project(
        self,
        project_name: str,
        project_author: str,
        project_description: str,
        project_type: str = "LWFA",
        project_directory: Optional[str] = os.getcwd(),
        project_source: Optional[str] = None,
        project_first_id: Optional[int] = 0,
        overwrite: Optional[bool] = False,
    ):
        """_summary_

        Args:
            project_name (str): _description_
            project_author (str): _description_
            project_description (str): _description_
            project_type (str, optional): _description_. Defaults to "LWFA".
            project_directory (_type_, optional): _description_. Defaults to os.getcwd().
            project_source (Any, optional): _description_. Defaults to None.
            id (int, optional): _description_. Defaults to 0.
        """

        # Erase previous project
        if overwrite:
            remove_project()

        self.twissed_version = __version__
        self.project_name = project_name
        self.project_author = project_author
        self.project_description = project_description
        self.project_type = project_type
        self.project_date = datetime.now().strftime("%Y/%m/%d %H:%M:%S:%f")

        self.project_directory = project_directory
        self.project_first_id = project_first_id
        self.project_source = project_source

        # Create h5 file
        self.h5filename = str(self.project_name) + ".sky.h5"
        self.h5file = h5py.File(self.h5filename, "w")

        # Create list of scripts and ids
        self.list_scripts = []
        _ = self.h5file.create_dataset("list_scripts", data=self.list_scripts)
        self.list_ids = []
        _ = self.h5file.create_dataset("list_ids", data=self.list_ids)

        # Create input group
        self.h5file.create_group("input")

        # Copy Attributes
        for key in PROJECT_PARAMETERS:
            _ = self.h5file.create_dataset(key, data=self.__dict__[key])
            _ = self.h5file["input"].create_dataset(key, data=self.__dict__[key])
            h5set_attr(self.h5file["input"], key)

        # Keep track last id used
        self.lastId = project_first_id - 1

    def open_project(self, project_name: str):
        self.h5filename = project_name + ".sky.h5"
        self.h5file = h5py.File(self.h5filename, "r+")

        for key in PROJECT_PARAMETERS:
            self.__dict__[key] = self.h5file[key][()]

        self.project_first_id = int(self.project_first_id)

        self.list_scripts = list(self.h5file["list_scripts"][()])

        self.list_ids = list(self.h5file["list_ids"][()])
        if len(self.list_ids) == 0:
            self.lastId = self.project_first_id - 1
        else:
            self.lastId = self.list_ids[-1]

        # Copy scripts
        for script_name in self.list_scripts:
            self.__dict__[script_name] = self.h5file[script_name][()].decode("utf-8")

    def print(self) -> None:
        """Print main information"""

        print(f"Project: {self.project_name} by {self.project_author}")
        print(f"Date: {self.project_date} (twissed v{self.twissed_version})")
        print(f"Directory: {self.project_directory}")
        print(f"First ID: {self.project_first_id}")
        print(f"{self.project_description}")
        print(f"Number of scripts: {len(self.list_scripts)}")
        for i in range(len(self.list_scripts)):
            print(f"   - {self.list_scripts[i]}")

    def close(self):
        """Close h5file"""

        try:
            getattr(self, "h5file")
            self.h5file.close()
        except AttributeError:
            print("Project does not have attribute 'h5file'.")

    def add_script(self, script_name: str, script_path: str, **kwargs: Any) -> None:
        """Add a new script file to the project.

        Argument under {{ }} are automatically updated later to the given value.

        Example
        -------

        Script form:

        >>> a0 = {{a0}}
            w0 = {{w0}}
            taua = {{taua}}
            zfoc = {{zfoc}}



        Args:
            script_name (str): Name of the file.
            script_path (str): Path of the script.
        """

        # Read script file
        with open(script_path, "r") as file:
            text = file.read()

        # Add keyword arguments to class
        for key in kwargs:
            _ = self.h5file["input"].create_dataset(key, data=kwargs.get(key, 0))
            h5set_attr(self.h5file["input"], key)

        # Copy script
        _ = self.h5file.create_dataset(script_name, data=text.encode("utf-8"))

        # Update class
        self.__dict__[script_name] = text

        # Update h5file
        self.list_scripts.append(script_name)
        del self.h5file["list_scripts"]
        _ = self.h5file.create_dataset("list_scripts", data=self.list_scripts)

    # def project_update_script(self, script_name: str) -> None:
    #     """Complete scripts with given arguments at the project level.

    #     Example
    #     -------

    #     >>> {{a0}} -> 1.3

    #     Args:
    #         script_name (str): Name of the script.

    #     Raises:
    #         ValueError: Check if missing argument.
    #     """

    #     text = self.__dict__[script_name]

    #     for name in self.__dict__:
    #         text = text.replace("{{" + name + "}}", str(self.__dict__[name]))

    #     self.__dict__[script_name] = text

    def add_run(self, id: Optional[int] = -1, **kwargs: Any) -> None:
        if id == -1:
            id = self.lastId + 1

        self.lastId = id

        # Update h5file
        self.list_ids.append(id)
        del self.h5file["list_ids"]
        _ = self.h5file.create_dataset("list_ids", data=self.list_ids)

        # Create group Run_XXXX
        folder_name = "Run_" + str(id).zfill(8)
        self.h5file.create_group(folder_name)
        self.h5file.copy("input", self.h5file[folder_name])

        # Create folder
        folder_path = os.path.join(str(self.project_directory), folder_name)
        create_folder(folder_path, verbose=True)

        # id
        _ = self.h5file[folder_name].create_dataset("id", data=id)
        h5set_attr(self.h5file[folder_name], "id")

        # run attr
        for key in kwargs:
            _ = self.h5file[folder_name]["input"].create_dataset(
                key, data=kwargs.get(key, 0)
            )
            h5set_attr(self.h5file[folder_name]["input"], key)

        # Update scripts
        for script_name in self.list_scripts:
            text = self.__dict__[script_name]

            # id
            text = text.replace("{{id}}", str(id).zfill(8))

            # project attr
            for name in self.h5file["input"].keys():
                text = text.replace(
                    "{{" + name + "}}", str(self.h5file["input"][name][()])
                )

            # run attr
            for name in kwargs:
                text = text.replace("{{" + name + "}}", str(kwargs.get(name, 0)))

            # Security
            check_missing_argument(text, script_name)

            # Create dataset
            _ = self.h5file[folder_name].create_dataset(
                script_name, data=text.encode("utf-8")
            )

            # Create script file
            script_path = os.path.join(folder_path, script_name)
            with open(script_path, "w") as f:
                f.write(text)

            make_executable(script_path)

        # # Create run_h5file
        run_h5filename = os.path.join(folder_path, folder_name + ".sky.h5")

        run_h5file = h5py.File(run_h5filename, "w")

        for key in self.h5file[folder_name].keys():
            self.h5file[folder_name].copy(key, run_h5file)

        run_h5file.close()

    # ! REECRIRE !!
    def run_input(self, id: int) -> dict:
        """Return the inputs of the wanted run.

        Args:
            id (int): Id of the run

        Returns:
            dict: Dictionnary of values
        """

        folder_name = "Run_" + str(id).zfill(8)
        return json.loads(self.h5file[folder_name].attrs["run_input"])

    def write_executable(self, limit: int = 300, cmdrun="ccc_msub"):
        # for id
        # self.list_ids.append(id)
        # del self.h5file["list_ids"]
        # _ = self.h5file.create_dataset("list_ids", data=self.list_ids)

        # # Create group Run_XXXX
        # folder_name = "Run_" + str(id).zfill(8)

        # list_folder = self.df["folder"].tolist()

        if len(self.list_ids) < limit:
            filename = os.path.join(self.project_directory, "run_all.sh")

            with open(filename, "w") as f:
                f.write("#!/bin/bash \n")

                for id in self.list_ids:
                    folder_name = "Run_" + str(id).zfill(8)
                    f.write(
                        "cd " + folder_name + "\n" + f"{cmdrun} run.sh \n" + "cd .. \n"
                    )

            make_executable(filename)

        else:
            filemax = len(self.list_ids) / limit
            j = 0
            i = 0
            while j < filemax:
                filename = os.path.join(
                    self.project_directory, "run_all_" + str(j).zfill(3) + ".sh"
                )
                with open(filename, "w") as f:
                    f.write("#!/bin/bash \n")

                    for k in range(limit):
                        if i < len(self.list_ids):
                            id = self.list_ids[i]
                            folder_name = "Run_" + str(id).zfill(8)

                            f.write(
                                "cd "
                                + folder_name
                                + "\n"
                                + f"{cmdrun} run.sh \n"
                                + "cd .. \n"
                            )
                            i += 1

                j += 1
                make_executable(filename)

    def read_run(
        self, id: int, timesteps: Optional[Union[int, List[int]]] = None
    ) -> RunReader:
        folder_name = "Run_" + str(id).zfill(8)

        run = RunReader(id, h5group=self.h5file[folder_name], timesteps=timesteps)

        return run

    def read_project(self, timestep: Optional[int] = -1) -> pd.DataFrame:
        df = pd.DataFrame()

        for id in self.list_ids:
            df1 = self.read_run(id, timesteps=timestep).steps_DataFrame()
            df1["id"] = id

            df = pd.concat([df, df1])

        df.set_index("id", inplace=True)
        return df


def check_missing_argument(text: str, script_name: str) -> None:
    """_summary_

    Args:
        text (str): _description_
        script_name (str): _description_

    Raises:
        AttributeError: Mission argument
    """
    if text.find("{{") > -1:
        error = (
            "Error: missing argument: '"
            + text[text.find("{{") + 2 : text.find("}}")]
            + "' in script: "
            + script_name
            + "\n"
            + "Please restart project (overwrite = True)"
        )

        raise AttributeError(error)


def end_run(
    directory: Optional[str] = os.getcwd(),
    source: Optional[str] = "fbpic",
    species: Optional[str] = "electrons",
    verbose: Optional[bool] = True,
):
    """Collect data after run.

    In a Run_XXXX folder:

    >>> python3 -c "import twissed;twissed.end_run(source="fbpic",species="electron");"

    """

    # ! Do all timesteps
    timestep = -1

    run = RunControler(directory=directory)

    if run.output_status:
        print(f"INFO: File {run.h5filename} already updated.")

    else:
        print(f"INFO: Starting update...")

        if source == "fbpic":
            directory = os.path.join(directory, "lab_diags")

        # Find all timesteps
        steps = Steps()
        steps.find_data(directory=directory, source=source, verbose=verbose)

        if len(steps.timesteps) >= 1:
            # Create output group
            run.h5file.create_group("output")

            _ = run.h5file["output"].create_dataset("timesteps", data=steps.timesteps)

            for timestep in steps.timesteps:
                # Create output group
                run.h5file["output"].create_group(str(timestep))

                # Creation of the step class
                step = Step()

                # Read data
                step = steps.read_beam(step, timestep, species=species)

                if step.N > 10.0:
                    for key in RUN_OUTPUT_PARAMETERS:
                        _ = run.h5file["output"][str(timestep)].create_dataset(
                            key, data=step.__dict__[key]
                        )
                        h5set_attr(run.h5file["output"][str(timestep)], key)
                else:
                    _ = run.h5file["output"][str(timestep)].create_dataset(
                        "N", data=step.N
                    )
                    h5set_attr(run.h5file["output"][str(timestep)], "N")

            print(f"INFO: File {run.h5filename} updated.")

            _ = run.h5file.create_dataset("output_status", data=True)

        else:
            print(
                f"WARNING: Unable to update {run.h5filename}. Steps have no timesteps."
            )

            _ = run.h5file.create_dataset("output_status", data=False)

    run.h5file.close()


def end_project(
    project_name: str,
    path: Optional[str] = os.getcwd(),
    source: Optional[str] = "fbpic",
    species: Optional[str] = "electrons",
    verbose: Optional[bool] = True,
):
    project = Project()

    project.open_project(project_name=project_name)

    for id in project.list_ids:
        folder_name = "Run_" + str(id).zfill(8)
        directory = os.path.join(path, folder_name)

        if "output" in project.h5file[folder_name]:
            print(
                f"INFO: Project {project_name} update: {folder_name} already updated."
            )
        else:
            end_run(
                directory=directory,
                source=source,
                species=species,
                verbose=verbose,
            )

            run = RunControler(directory=directory)

            if "output" in list(run.h5file.keys()):
                run.h5file.copy("output", project.h5file[folder_name])
            if "output_status" in list(run.h5file.keys()):
                run.h5file.copy("output_status", project.h5file[folder_name])

            run.h5file.close()

            print(f"INFO: Project {project_name} update: {folder_name} updated.")

    project.close()


def remove_project(path: str = os.getcwd()):
    list_runs = [i for i in os.listdir(path) if i.startswith("Run_")]
    for folder in list_runs:
        shutil.rmtree(os.path.join(path, folder))

    list_h5 = [i for i in os.listdir(path) if i.endswith(".sky.h5")]
    for file in list_h5:
        os.remove((os.path.join(path, file)))

    list_runs = [i for i in os.listdir(path) if i.startswith("run_all")]
    for file in list_runs:
        os.remove((os.path.join(path, file)))
