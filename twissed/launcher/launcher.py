"""launcher.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

from datetime import datetime
import numpy as np
import os
import pandas as pd
import scipy.constants as const
import shutil

# twissed
from ..utils.tools import make_executable, deprecated
from ..steps.steps import Steps

__version__ = "0.0"
__codes_available__ = []

keys_getdataRun = [
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


class Run:
    """Run class.

    Create a run from given scripts. The run will be created in the folder `/directory/Run_id`.

    Main function are `addfile` and `save`.
    Each script added with `addfile` will be save in a pandas dataframe named `df_ini_id.csv`.

    Usage
    -----

    .. code-block:: python

        # Initiate new run
        run = twissed.Run('twissed Run',
                        directory=os.getcwd(),
                        id=0,
                        # Add to dict:
                        author='Damien',
                        description='Run for test')

        # Add file to run
        filename = "run.sh"
        filepath = os.path.join(directory, "run_fbpic_topaze.sh")
        run.addfile(filename,
                    filepath,
                    # Add to dict:
                    accronyme="Skytest",
                    slurm_n_core=20,
                    slurm_T_cputime=84000,
                    scriptpath = os.path.join(run.folderpath, "script_fbpic.py"))

        # Add file to run
        filename = "script_fbpic.py"
        filepath = os.path.join(directory, "script_fbpic.py")
        run.addfile(filename,
                    filepath,
                    # Add to dict:
                    a0 = 1.3,
                    w0 = 20.9e-6,
                    taua = 2.5e-14,
                    zfoc = 3.6e-03,
                    z0 = -30.6*1e-6,
                    )

        # Create df_ini_00000000.csv file
        run.save()

    """

    def keys(self):
        return list(self.__dict__.keys())

    @deprecated
    def __init__(self, projectname, id, directory=os.getcwd(), source=None, **kwargs):
        if source == None:
            print(f"Please select an available source: {__codes_available__}")
        else:
            self.twissedversion = __version__
            self.projectname = projectname
            self.directory = directory
            self.id = id
            self.source = source

            for name in kwargs:
                self.__dict__[name] = kwargs.get(name, 0)

            self.folder = "Run_" + str(self.id).zfill(8)
            self.folderpath = os.path.join(directory, self.folder)
            self.createfolder()

            self.date = datetime.now().strftime("%Y/%m/%d %H:%M:%S:%f")

    def addfile(self, filename, filepath, **kwargs):
        with open(filepath, "r") as file:
            text = file.read()

        for name in kwargs:
            self.__dict__[name] = kwargs.get(name, 0)
            text = text.replace("{{" + name + "}}", str(kwargs.get(name, 0)))

        text = text.replace("{{id}}", str(self.id).zfill(8))
        for name in self.__dict__:
            text = text.replace("{{" + name + "}}", str(self.__dict__[name]))

        self.__dict__[filename] = text
        if text.find("{{") > -1:
            print(
                "Warning missing argument: "
                + text[text.find("{{") + 2 : text.find("}}")]
                + "     in file: "
                + filename
            )

        copypath = os.path.join(self.folderpath, filename)
        with open(copypath, "w") as f:
            f.write(text)
        make_executable(copypath)

    def createfolder(self):
        os.mkdir(self.folderpath)

        print(f"Folder (variable folderpath) created: {self.folderpath}")

    def save(self):
        series = pd.Series(self.__dict__)
        path = os.path.join(self.folderpath, "df_ini_" + str(self.id).zfill(8) + ".csv")
        series.to_csv(path)

        print(f"Data series saved: {path}")

    def removefolder(self):
        shutil.rmtree(self.folderpath)


class Launcher_all:
    def keys(self):
        return list(self.__dict__.keys())

    def addrun(self, run):
        # self.df = self.df.append(run.__dict__,ignore_index=True) # deprecated !

        if self.df.empty:
            self.df = pd.Series(run.__dict__)
        else:
            self.df = pd.concat(
                [self.df.transpose(), pd.Series(run.__dict__)],
                axis=1,
                ignore_index=True,
            ).transpose()


class Launcher(Launcher_all):
    """Main launcher class"""

    def __init__(self):
        self.df = pd.DataFrame()

    def save(self, path=os.getcwd(), cmdrun="ccc_msub ", limit=np.infty):
        filename = os.path.join(path, "df_ini.csv")
        self.df.to_csv(filename)

        print(f"Dataframe saved: {filename}")

        list_folder = self.df["folder"].tolist()
        if len(list_folder) < limit:
            filename = os.path.join(path, "run_all.sh")
            with open(filename, "w") as f:
                f.write("#!/bin/bash \n")
                for item in list_folder:
                    f.write("cd " + item + "\n" + f"{cmdrun}run.sh \n" + "cd .. \n")
            make_executable(filename)
        else:
            filemax = len(list_folder) / limit
            j = 0
            i = 0
            while j < filemax:
                filename = os.path.join(path, "run_all_" + str(j).zfill(3) + ".sh")
                with open(filename, "w") as f:
                    f.write("#!/bin/bash \n")
                    for k in range(limit):
                        f.write(
                            "cd "
                            + list_folder[i]
                            + "\n"
                            + f"{cmdrun}run.sh \n"
                            + "cd .. \n"
                        )
                        i += 1

                j += 1
                make_executable(filename)


class LauncherGA(Launcher_all):
    """
    Launcher for genetic algorithm
    """

    @deprecated
    def __init__(self, FIRSTID, POP_SIZE, nInputs, path=os.getcwd()):
        self.path = path
        self.df = pd.DataFrame()

        self.FIRSTID = FIRSTID
        self.POP_SIZE = POP_SIZE
        self.nInputs = nInputs

        self.log_path = os.path.join(self.path, "log.csv")

        if os.path.exists(self.log_path):
            self.log = pd.read_csv(self.log_path)
            self.iGeneration = self.log["iGeneration"].max() + 1
            self.lengthlog = len(self.log.index)
            self.lastids = list(
                map(
                    int,
                    (self.log["ids"].iloc[self.lengthlog - 1])
                    .replace("[", "")
                    .replace("]", "")
                    .split(","),
                )
            )
            self.lastpop = np.reshape(
                list(
                    map(
                        float,
                        (self.log["POP"].iloc[self.lengthlog - 1])
                        .replace("[", "")
                        .replace("]", "")
                        .split(","),
                    )
                ),
                (self.POP_SIZE, self.nInputs),
            )
        else:
            self.log = pd.DataFrame()
            self.iGeneration = 0
            self.lengthlog = 0
            self.lastids = 0
            self.lastpop = 0

    def isLastGenEnded(self):
        path = self.path
        if self.lastids == 0:
            return True
        else:
            for id in self.lastids:
                folder = "Run_" + str(id).zfill(8)
                folderpath = os.path.join(self.path, folder)
                filename = os.path.join(
                    folderpath, "df_end_" + str(id).zfill(8) + ".csv"
                )

                if os.path.exists(filename):
                    print(f"Id: {id} finished!")
                else:
                    print(f"Id: {id} not finished!")
                    return False
            return True

    def save(self, population, cmdrun="ccc_msub "):
        path = self.path
        ids = [i for i in self.df["id"]]

        newlog = pd.DataFrame(
            {
                "iGeneration": self.iGeneration,
                "ids": [ids],
                "FIRSTID": self.FIRSTID,
                "POP_SIZE": self.POP_SIZE,
                "POP": [list(np.reshape(population, (self.POP_SIZE * self.nInputs,)))],
            }
        )

        if self.log.empty:
            self.log = newlog
        else:
            self.log = pd.concat([self.log, newlog])

        self.log.to_csv(self.log_path, index=False)

        filename = os.path.join(path, "df_ini.csv")
        self.df.to_csv(filename)

        print(f"Dataframe saved: {filename}")

        list_folder = self.df["folder"].tolist()
        filename = os.path.join(
            path, "run_all_" + str(self.iGeneration).zfill(4) + ".sh"
        )
        with open(filename, "w") as f:
            f.write("#!/bin/bash \n")
            for item in list_folder:
                f.write("cd " + item + "\n" + f"{cmdrun}run.sh \n" + "cd .. \n")
        make_executable(filename)


@deprecated
def getdatafromrun(
    path: str = os.getcwd(),
    timestep: int = -1,
    source: str = None,
    species: str = "electrons",
) -> pd.DataFrame:
    """Create df_end_id.csv file.

    Args:
        path (str, optional): Path to run. Defaults to os.getcwd().
        timestep (int, optional): Step time wanted. Defaults to -1 (last one).
        source (str, optional) : Set the source of the simulation. If None, will try to find the source from the df_ini_***.csv file. Defaults to None.
        species (str, optional) : Name of the beam species. Defaults to "electrons"

    Returns:
        pd.DataFrame: Pandas dataframe with the dict from Step.
    """

    id = int(path[path.find("Run_") + 4 : path.find("Run_") + 12])
    df = pd.DataFrame()

    # Find source
    if source == None:
        df_ini_path = os.path.join(path, "df_ini_" + str(id).zfill(8) + ".csv")
        df_ini = pd.read_csv(df_ini_path, index_col=0).squeeze()

        if "source" in df_ini.columns or "source" in df_ini.index:
            source = df_ini["source"]
        else:
            source = "fbpic"

    if source == "fbpic":
        datapath = os.path.join(path, "lab_diags")

    if source == "happi":
        datapath = path

    # Find all timesteps
    steps = Steps(directory=datapath, source=source, verbose=False)

    # timestep selection
    if timestep == -1:
        timestep = steps.timesteps[-1]

    # Creation of the step class
    step = Step()

    # Read data
    step = steps.read_beam(step, timestep, species=species)

    if step.N > 10.0:
        dict = {key: step.__dict__[key] for key in keys_getdataRun}
    else:
        dict = {"N": 0.0}

    df = pd.Series(dict)

    filename = os.path.join(path, "df_end_" + str(id).zfill(8) + ".csv")
    df.to_csv(filename)

    return df


def getdata(
    path: str = os.getcwd(), timestep: int = -1, filename: str = "df_full.csv"
) -> pd.DataFrame:
    """Create df_full.csv file from all Run_*** folders.

    Inline usage
    ------------

    .. code-block::

        python -c "import twissed;twissed.getdata();"


    Args:
        path (str, optional): Path of all Run_*** folders. Defaults to os.getcwd().
        timestep (int, optional): timestep selected for data collection. Defaults to -1.
        filename (str, optional): Name of the .csv file. Defaults to 'df_full.csv'.

    Returns:
        pd.DataFrame: Pandas dataframe with the dict from all Runs.
    """

    df_full = pd.DataFrame()

    list_folder = [i for i in os.listdir(path) if i.startswith("Run_")]
    for folder in list_folder:
        id = int(folder[folder.find("Run_") + 4 : folder.find("Run_") + 12])

        df_ini_path = os.path.join(folder, "df_ini_" + str(id).zfill(8) + ".csv")

        df_end = getdatafromrun(path=folder, timestep=timestep)

        df_run = pd.concat([pd.read_csv(df_ini_path, index_col=0).squeeze(), df_end])

        if df_full.empty:
            df_full = df_run
        else:
            df_full = pd.concat(
                [df_full.transpose(), df_run], axis=1, ignore_index=True
            ).transpose()

    filename = os.path.join(path, "df_full.csv")
    df_full.to_csv(filename)

    return df_full


@deprecated
def removelauncher(path=os.getcwd()):
    list_run = [i for i in os.listdir(path) if i.startswith("Run_")]
    for folder in list_run:
        shutil.rmtree(os.path.join(path, folder))

    list_run = [i for i in os.listdir(path) if i.startswith("run_all")]
    for file in list_run:
        os.remove((os.path.join(path, file)))

    if os.path.exists(os.path.join(path, "df_ini.csv")):
        os.remove((os.path.join(path, "df_ini.csv")))

    if os.path.exists(os.path.join(path, "log.csv")):
        os.remove((os.path.join(path, "log.csv")))
