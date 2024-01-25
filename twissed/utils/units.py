"""units.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

from typing import Any

# SI prefixes
quetta: float = 1e30
ronna: float = 1e27
yotta: float = 1e24
zetta: float = 1e21
exa: float = 1e18
peta: float = 1e15
tera: float = 1e12
giga: float = 1e9
mega: float = 1e6
kilo: float = 1e3
hecto: float = 1e2
deca: float = 1e1
deci: float = 1e-1
centi: float = 1e-2
milli: float = 1e-3
micro: float = 1e-6
nano: float = 1e-9
pico: float = 1e-12
femto: float = 1e-15
atto: float = 1e-18
zepto: float = 1e-21
yocto: float = 1e-24
ronto: float = 1e-27
quecto: float = 1e-30

SI_PREFIX: dict = {
    "Q": quetta,
    "R": ronna,
    "Y": yotta,
    "Z": zetta,
    "E": exa,
    "P": peta,
    "T": tera,
    "G": giga,
    "M": mega,
    "k": kilo,
    "h": hecto,
    "da": deca,
    "d": deci,
    "c": centi,
    "m": milli,
    "u": micro,
    "n": nano,
    "p": pico,
    "f": femto,
    "a": atto,
    "z": zepto,
    "y": yocto,
    "r": ronto,
    "q": quecto,
}

AUTORISED_UNITS: list = [
    "A",
    "s",
    "m",
    "rad",
    "V/m",
    "W",
    "C",
    "J",
    "eV",
]  # Unit format as "SI_PREFIX" + "key in AUTORISED_UNITS". Ex: ms (s units)

AUTORISED_UNITS_4_CONVERSION = AUTORISED_UNITS.copy()
AUTORISED_UNITS_4_CONVERSION.append(
    "m-3",
)


def convert_units(ini: str, fin: str) -> float:
    """Convert for initial to final units.

    Args:
        ini (str): Initial units
        fin (str): Final units

    Returns:
        float: Value needed to convert
    """
    if ini is None or fin is None:
        return 1.0

    # Automatic conversion
    for item in AUTORISED_UNITS:
        if ini.endswith(item):
            if len(ini[: -len(item)]) == 0 and len(fin[: -len(item)]) == 0:
                return 1.0
            elif len(ini[: -len(item)]) == 0:
                return 1 / SI_PREFIX[fin[: -len(item)]]
            elif len(fin[: -len(item)]) == 0:
                return SI_PREFIX[ini[: -len(item)]]
            elif ini[: -len(item)] in SI_PREFIX:
                return SI_PREFIX[ini[: -len(item)]] * 1 / SI_PREFIX[fin[: -len(item)]]

    # Density
    item = "m-3"
    if ini.endswith(item):
        if len(ini[: -len(item)]) == 0 and len(fin[: -len(item)]) == 0:
            return 1.0
        elif len(ini[: -len(item)]) == 0:
            return SI_PREFIX[fin[: -len(item)]] ** 3
        elif len(fin[: -len(item)]) == 0:
            return 1 / SI_PREFIX[ini[: -len(item)]] ** 3
        elif ini[: -len(item)] in SI_PREFIX:
            return (
                1
                / SI_PREFIX[ini[: -len(item)]] ** 3
                * SI_PREFIX[fin[: -len(item)]] ** 3
            )

    return 1.0


# ! Fonction a supprimer ! Replacer par la lecture de STD_DATABASE.
def convert_symb(ini: Any, short=False) -> str:
    if ini == None:
        return ""


def units_latex(name: str, short: bool = False) -> str:
    if name is None:
        return ""

    # Automatic conversion
    for item in AUTORISED_UNITS_4_CONVERSION:
        if name.endswith(item):
            si_unit = name[: -len(item)]
            if si_unit in SI_PREFIX:
                if item == "m-3":
                    item = r"m$^{3}$"

                if si_unit == "u":
                    return "\u03bc" + item
                else:
                    return name

    if name == "pi.mm.mrad" and short == True:
        return "\u03bc" + "rad"
    elif name == "pi.mm.mrad":
        return r"$\pi$mm$\cdot$mrad"

    if name == "mm/mrad" and short == True:
        return r"m/rad"

    if name == "mrad/mm" and short == True:
        return r"rad/m"

    if name == "mm/pi/%":
        return r"mm/$\pi$/\%"

    if name == "pi.mm.%":
        return r"$\pi$ mm$\cdot$\%"

    if name == "(pi.mm.mrad)2":
        return r"($\pi$ mm$\cdot$mrad)$^2$"

    if name == "(pi.mm.mrad)3":
        return r"($\pi$ mm$\cdot$mrad)$^3$"

    if name == "m2":
        return r"m$^2$"

    if name == "m.rad":
        return r"m$\cdot$rad"

    if name == "rad2":
        return r"rad$^2$"

    if name == "C/m3":
        return r"C/m$^{3}$"

    return name
