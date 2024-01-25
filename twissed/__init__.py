"""__init__.py file

    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 03/07/2023
"""


# twissed short cut
from .io.astra import *
from .io.fbpic import *
from .io.happi import *
from .io.smilei import *
from .io.tracewin import *
from .laser.laser import *
from .launcher.launcher import *
from .plt.rcParams import rcParams
from .plt.colormap import Cmap

# from .plt.runs import *
from .project.project import *
from .step.step import *
from .steps.steps import *
from .utils.data import *
from .utils.noweight import *
from .utils.physics import *
from .utils.stats import *
from .utils.units import *

from typing import Dict, List

__version__: str = "2.1.1"

__date__: str = "2023/01/25"

__contributors__: Dict = {
    "Damien Minenna (CEA/Irfu)": "damien.minenna@cea.fr",
    "Antoine Chancé (CEA/Irfu)": "antoine.chance@cea.fr",
    "Samuel Marini (CEA/Irfu)": "samuel.marini@cea.fr",
    "Francesco Massimo (LPGP/CNRS)": "francesco.massimo@universite-paris-saclay.fr",
    "Nicolas Pichoff (CEA/Irfu)": "nicolas.pichoff@cea.fr",
}

__codes_available__: List[str] = ["fbpic", "happi", "smilei", "tracewin", "astra"]


# Print logo
LOGO_VERSION = "\n " + f"twissed (v{__version__}, {__date__})" + "\n"
print(LOGO_VERSION)
