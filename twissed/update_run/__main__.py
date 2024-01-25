"""__main__.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

# Usage
# -----
#
#  python -m twissed.update_run -source fbpic -verbose True

import argparse
import os

# twissed
from ..project.project import Run_bis, end_run

parser = argparse.ArgumentParser(
    description="twissed script to collect data from a run"
)
parser.add_argument("-directory", "--directory", help="directory", required=False)
parser.add_argument("-source", "--source", help="source", required=False)
parser.add_argument("-species", "--species", help="species", required=False)
parser.add_argument("-verbose", "--verbose", help="verbose", required=False)
args = vars(parser.parse_args())

if args["directory"] == None:
    directory = os.getcwd()
else:
    directory = str(args["directory"])

if args["source"] == None:
    source = "fbpic"
else:
    source = str(args["source"])

if args["species"] == None:
    species = "electrons"
else:
    species = str(args["species"])

if args["verbose"] == None:
    verbose = True
else:
    verbose = str(args["verbose"])

end_run(
    directory=directory,
    source=source,
    species=species,
    verbose=verbose,
)
