"""metadata.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

# import
import pandas as pd
from typing import Any, Type, Tuple, Dict, Union, List

# twissed
from . import units
from .data import STD_DATABASE


class MetaData(object):
    """Main class to manage metadata.

    Allows to have metadata for attributes.
    """

    _data = {}  # Metadata for the important variables.

    def __setattr__(self, name: str, value: Type[Union[Any, Tuple[Any, Dict]]]) -> None:
        """Set attribute for the MetaData object

        Example
        -------

        .. code-block:: python

            a = MetaData()

            a.x = 12 # x is known in the database
            a.y = 12 # y is known in the database
            a.tmp = 12 # tmp is not known in the database
            a.pos = (
                    14,
                    {
                    'name_latex': '$x$',
                    'units': 'm',
                    'info': 'Positions x of the macro-particle of the beam.',
                    'type': 'np.ndarray'
                    }
                    ) # Create an attribute pos = 14, with metadata defined by the dictionnary.

            a.x = (13, {'units': 'mm', 'new': 'temporary'}) # Update the metadata of x

            print(a._data) # print metadata

        Args:
            name (str): Name of the attribute
            value (Type[Any  |  Tuple[Any  |  Dict]]): The value of the attribute. This argument can be a tuple with the wanted value and metadata for the attribute

        Raises:
            KeyError: Some attributes are reserved.
        """

        if name == "_data":
            raise KeyError("Attribute '_data' is reserved!")

        data = {}
        if isinstance(value, tuple):
            value, data = value

        self.__dict__[name] = value

        if name in STD_DATABASE and bool(data):
            self._data[name] = STD_DATABASE[name] | data
        elif name in STD_DATABASE:
            self._data[name] = STD_DATABASE[name]
        elif bool(data):
            self._data[name] = data

    def __delattr__(self, name: str) -> None:
        """Delate an attribute

        Args:
            name (str): Name of the attribute to delete.

        Raises:
            KeyError: No key in the object.
        """

        if not name in self.__dict__:
            raise KeyError(f"There is no key '{name}' in the object.")
        del self._data[name]
        del self.__dict__[name]

    def keys(self) -> List:
        """Display all the keys.

        Returns:
            List: List of all the keys.
        """
        return list(self.__dict__.keys())

    def get_data(self, name: str) -> Dict:
        """Return _data of the attribute.

        >>> a.get_data('x') # is equivalent to a._data['x']

        Args:
            name (str): key

        Returns:
            Dict: The metadata
        """
        return self._data[name]

    def get_units(self, name: str) -> Union[str, None]:
        """Return _data of the attribute.

        >>> a.get_units('x') # is equivalent to a._data['x']['units']

        Args:
            name (str): key

        Returns:
            str | None: The "units" metadata. Can be None.
        """
        if "units" in self._data[name]:
            return self._data[name]["units"]
        else:
            return None

    def get_info(self, name: str) -> Union[str, None]:
        """Return _data of the attribute.

        >>> a.get_info('x') # is equivalent to a._data['x']['info']

        Args:
            name (str): key

        Returns:
            str | None: The "info" metadata. Can be None.
        """
        if "info" in self._data[name]:
            return self._data[name]["info"]
        else:
            return None

    def DataFrame(self) -> pd.DataFrame:
        """Create a pandas DataFrame from scalar values

        Returns:
            pd.DataFrame: _description_
        """
        data = {}
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], int) or isinstance(
                self.__dict__[key], float
            ):
                data[key] = self.__dict__[key]

        return pd.DataFrame(data, index=[0])

    def print(self, name: str) -> None:
        """Print human readable data.

        Args:
            name (str): key

        Raises:
            KeyError: No key in the object.
        """

        if name == "beam":
            print(f"sigma_x     : {self.sigma_x} m")
            print(f"sigma_y     : {self.sigma_y} m")
            print(f"sigma_z     : {self.sigma_z} m")
            print(f"sigma_xp    : {self.sigma_xp} rad")
            print(f"sigma_yp    : {self.sigma_yp} rad")
            print(f"sigma_dp/p  : {self.sigma_dp * 100} %")
            print("-----------")
            print(f"eps_N_x     : {self.emit_norm_rms_x} pi.mm.mrad")
            print(f"alpha_x     : {self.alpha_x}")
            print(f"beta_x      : {self.beta_x} mm/mrad")
            print(f"gamma_x     : {self.gamma_x} mrad/mm")
            print(f"eps_N_y     : {self.emit_norm_rms_y} pi.mm.mrad")
            print(f"alpha_y     : {self.alpha_y}")
            print(f"beta_y      : {self.beta_y} mm/mrad")
            print(f"gamma_y     : {self.gamma_y} mrad/mm")
            print(f"eps_N_z     : {self.emit_norm_rms_z} pi.mm.%")
            print(f"alpha_z     : {self.alpha_z}")
            print(f"beta_z      : {self.beta_z} mm/pi/%")
            print(f"gamma_z     : {self.gamma_z} mrad/mm")
            print(f"eps_4D      : {self.emit_norm_rms_4D} (pi.mm.mrad)2")
            print(f"eps_6D      : {self.emit_norm_rms_6D} (pi.mm.mrad)3")
            print("-----------")
            print(f"Energy mean : {self.Ek_avg} MeV")
            print(f"Energy med  : {self.Ek_med} MeV")
            print(f"Energy std  : {self.Ek_std} MeV")
            print(f"Energy std  : {self.Ek_std_perc} %")
            print(f"Energy mad  : {self.Ek_mad} MeV")
            print(f"Energy mad  : {self.Ek_mad_perc} %")
            print("-----------")
            print(f"Charge      : {self.charge} pC")
            print(f"Nb particle : {self.N}")

        else:
            if not name in self.__dict__:
                raise KeyError(f"There is no key '{name}' in the object.")

            elif name == "sigma_matrix":
                print(
                    r"   x (m)"
                    + f"| {self.sigma_matrix[0,0]:.5e} {self.sigma_matrix[0,1]:.5e} {self.sigma_matrix[0,2]:.5e} {self.sigma_matrix[0,3]:.5e} {self.sigma_matrix[0,4]:.5e} {self.sigma_matrix[0,5]:.5e} |"
                )
                print(
                    r"x' (rad)"
                    + f"| {self.sigma_matrix[1,0]:.5e} {self.sigma_matrix[1,1]:.5e} {self.sigma_matrix[1,2]:.5e} {self.sigma_matrix[1,3]:.5e} {self.sigma_matrix[1,4]:.5e} {self.sigma_matrix[1,5]:.5e} |"
                )
                print(
                    r"   y (m)"
                    + f"| {self.sigma_matrix[2,0]:.5e} {self.sigma_matrix[2,1]:.5e} {self.sigma_matrix[2,2]:.5e} {self.sigma_matrix[2,3]:.5e} {self.sigma_matrix[2,4]:.5e} {self.sigma_matrix[2,5]:.5e} |"
                )
                print(
                    r"y' (rad)"
                    + f"| {self.sigma_matrix[3,0]:.5e} {self.sigma_matrix[3,1]:.5e} {self.sigma_matrix[3,2]:.5e} {self.sigma_matrix[3,3]:.5e} {self.sigma_matrix[3,4]:.5e} {self.sigma_matrix[3,5]:.5e} |"
                )
                print(
                    r"   z (m)"
                    + f"| {self.sigma_matrix[4,0]:.5e} {self.sigma_matrix[4,1]:.5e} {self.sigma_matrix[4,2]:.5e} {self.sigma_matrix[4,3]:.5e} {self.sigma_matrix[4,4]:.5e} {self.sigma_matrix[4,5]:.5e} |"
                )
                print(
                    r"    dp/p"
                    + f"| {self.sigma_matrix[5,0]:.5e} {self.sigma_matrix[5,1]:.5e} {self.sigma_matrix[5,2]:.5e} {self.sigma_matrix[5,3]:.5e} {self.sigma_matrix[5,4]:.5e} {self.sigma_matrix[5,5]:.5e} |"
                )

            else:
                msg = str(self.__dict__[name])

                if not name in self._data:
                    print(msg)
                else:
                    units = self.get_units(name)
                    if not units == None:
                        msg += " [" + str(units) + "]"

                    info = self.get_info(name)
                    if not info == None:
                        msg += ", " + str(info)

                    print(msg)

    def convert(self, name: str, wanted_units: str) -> float:
        """Convert a variable to another unit

        # ! Experimental function. No security or generalisation implemented.
        # ! Return 1 when it does not know what to do.

        Args:
            name (str): key that must have
            wanted_units (str): Wanted final units

        Returns:
            float: the attribute after conversion.
        """

        base_units = self.get_units(name)
        if base_units == None or base_units == wanted_units or wanted_units == None:
            return self.__dict__[name]

        return self.__dict__[name] * units.convert_units(base_units, wanted_units)

    def label_plot(self, name, conv: str = None, short: bool = False) -> str:
        if name in STD_DATABASE:
            text = STD_DATABASE[name]["name_latex"]
            if conv is None:
                if "units" in STD_DATABASE[name]:
                    text += ", " + units.units_latex(
                        STD_DATABASE[name]["units"], short=short
                    )
            else:
                text += " [" + units.units_latex(conv, short=short) + "]"

            return text
        else:
            return str(name)
