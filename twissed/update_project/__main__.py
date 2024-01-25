"""__main__.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

# Usage
# -----
#
#  python -m twissed.update_project -project_name TwissedProject

import argparse
import os

# twissed
from ..project.project import end_project

parser = argparse.ArgumentParser(
    description="twissed script to collect data from a run"
)
parser.add_argument(
    "-project_name", "--project_name", help="project_name", required=True
)
parser.add_argument("-directory", "--directory", help="directory", required=False)
parser.add_argument("-source", "--source", help="source", required=False)
parser.add_argument("-species", "--species", help="species", required=False)
parser.add_argument("-verbose", "--verbose", help="verbose", required=False)
args = vars(parser.parse_args())

project_name = str(args["project_name"])

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

end_project(
    project_name=project_name,
    path=directory,
    source=source,
    species=species,
    verbose=verbose,
)
