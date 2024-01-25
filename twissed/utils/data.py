"""data.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""


import pandas as pd
import json
from typing import Dict

STD_DATABASE_KEYS = [
    "name_latex",
    "units",
    "info",
    "type",
]

# Beam data type
STD_DATABASE_PROJECT: Dict = {
    # ******************** Projets
    "twissed_version": {
        "name_latex": r"twissed version",
        "info": r"Version of twissed used to generate the data.",
        "type": "str",
    },
    "project_name": {
        "name_latex": r"Project name",
        "info": r"Name of the twissed project.",
        "type": "str",
    },
    "project_author": {
        "name_latex": r"Project name",
        "info": r"Name of the twissed project.",
        "type": "str",
    },
    "project_description": {
        "name_latex": r"Project name",
        "info": r"Name of the twissed project.",
        "type": "str",
    },
    "project_type": {
        "name_latex": r"Project type",
        "info": r"Type of the twissed project (LWFA, ...).",
        "type": "str",
    },
    "project_date": {
        "name_latex": r"Project date",
        "info": r"Date of the project creation.",
        "type": "str",
    },
    "project_first_id": {
        "name_latex": r"Project first ID",
        "info": r"ID of the first run.",
        "type": "int",
    },
    "project_directory": {
        "name_latex": r"Project directory",
        "info": r"Path of the project.",
        "type": "str",
    },
    "project_source": {
        "name_latex": r"Project source",
        "info": r"Type of code used to generate data [fbpic, smilei,...].",
        "type": "str",
    },
    # ******************** Run
    "id": {
        "name_latex": r"id",
        "info": r"Id of the run. Each run should have an unique id.",
        "type": "int",
    },
}

# Beam data type
STD_DATABASE_BEAM: Dict = {
    # ******************** Particles
    "species": {
        "name_latex": r"species",
        "info": r"Type of particles",
        "type": "str",
    },
    # Beam positions
    "x": {
        "name_latex": r"$x$",
        "units": "m",
        "info": r"Positions x of the macro-particle of the beam.",
        "type": "np.ndarray",
    },
    "y": {
        "name_latex": r"$y$",
        "units": "m",
        "info": r"Positions y of the macro-particle of the beam.",
        "type": "np.ndarray",
    },
    "z": {
        "name_latex": r"$z$",
        "units": "m",
        "info": r"Positions z of the macro-particle of the beam.",
        "type": "np.ndarray",
    },
    # Beam momenta
    "ux": {
        "name_latex": r"$u_x$",
        "info": r"Normalised momenta $x$ ($u_x=\gamma*v_x/c$) of the macro-particle of the beam.",
        "type": "np.ndarray",
    },
    "uy": {
        "name_latex": r"$u_y$",
        "info": r"Normalised momenta $y$ ($u_y=\gamma*v_y/c$) of the macro-particle of the beam.",
        "type": "np.ndarray",
    },
    "uz": {
        "name_latex": r"$u_z$",
        "info": r"Normalised momenta $z$ ($u_z=\gamma*v_z/c$) of the macro-particle of the beam.",
        "type": "np.ndarray",
    },
    # Weight
    "w": {
        "name_latex": r"$w$",
        "info": r"Weighs of the macro-particles in term of number of elemetary particles.",
        "type": "np.ndarray",
    },
    # Particle energy, gamma and speed
    "g": {
        "name_latex": r"$\gamma$",
        "info": r"Lorentz factor of every single macro-particles. $\gamma = \sqrt{1 + u_x^2 + u_y^2 + u_z^2}$",
        "type": "np.ndarray",
    },
    "Ek": {
        "name_latex": r"Energy",
        "units": "MeV",
        "info": r"Relativistic kinetic energy of macro-particles. $E_k = (\gamma - 1) m_e c^2 / e / 1e6$",
        "type": "np.ndarray",
    },
    "vz": {
        "name_latex": r"$v_z$",
        "units": "m/s",
        "info": r"Speed z (lab frame) of the macro-particle of the beam. $v_z = u_z c / \gamma$.",
        "type": "np.ndarray",
    },
    "p": {
        "name_latex": r"$p_z$",
        "units": "m/s",
        "info": r"Particle momenta $p_z = \gamma <\beta_z> c$.",
        "type": "np.ndarray",
    },
    "dp": {
        "name_latex": r"$\delta p/p$",
        "info": r"Particle momenta variation $\delta p/p$.",
        "type": "np.ndarray",
    },
    # Trace momenta
    "xp": {
        "name_latex": r"$x'$",
        "units": "rad",
        "info": r"Trace momenta of the particles $x' = u_x / u_z$.",
        "type": "np.ndarray",
    },
    "yp": {
        "name_latex": r"$y'$",
        "units": "rad",
        "info": r"Trace momenta of the particles $y' = u_y / u_z$.",
        "type": "np.ndarray",
    },
    # ******************** Beam
    # Main beam info
    "charge": {
        "name_latex": r"Charge",
        "units": "pC",
        "info": r"Total charge of the beam. For electrons : $Q = (\sum_i w_i e) / 1e-12",
        "type": "float",
    },
    "N": {
        "name_latex": r"Number of particles",
        "info": r"Total number of macro-particle the beam. Note that, if N < 10: no beam assumed.",
        "type": "int",
    },
    "current": {
        "name_latex": r"Current",
        "units": "A",
        "info": r"Current of the beam. For electrons : $I = v_z w e$",
        "type": "float",
    },
    # Beam energies
    "Ek_avg": {
        "name_latex": r"Energy",
        "units": "MeV",
        "info": r"Mean (avg) kinetic energy of the beam. $E_{\mathrm{avg}} = <E_k>$",
        "type": "float",
    },
    "Ek_med": {
        "name_latex": r"Energy",
        "units": "MeV",
        "info": r"Median (med) kinetic energy of the beam.",
        "type": "float",
    },
    "Ek_std": {
        "name_latex": r"$\sigma_{E,\mathrm{std}}$",
        "units": "MeV",
        "info": r"Standard deviation (std) of the kinetic energy of the beam. Same as sigma_Ek.",
        "type": "float",
    },
    "Ek_mad": {
        "name_latex": r"$\sigma_{E,\mathrm{mad}}$",
        "units": "MeV",
        "info": r"Median absolute deviation (mad) of the kinetic energy of the beam.",
        "type": "float",
    },
    "Ek_std_perc": {
        "name_latex": r"$\sigma_{E,\mathrm{std}} / E_{\mathrm{avg}}$",
        "units": "%",
        "info": r"Energy spread Ek_std / Ek_avg * 100 of the beam",
        "type": "float",
    },
    "Ek_mad_perc": {
        "name_latex": r"$\sigma_{E,\mathrm{mad}} / E_{\mathrm{med}}$",
        "units": "%",
        "info": r"Energy spread Ek_mad / Ek_med * 100 of the beam.",
        "type": "float",
    },
    # Beam gamma Lorentz factor
    "g_avg": {
        "name_latex": r"Lorentz factor",
        "info": r"Mean (avg) Lorentz factor of the beam $\gamma_{\mathrm{avg}} = <\gamma>$",
        "type": "float",
    },
    "g_med": {
        "name_latex": r"Lorentz factor",
        "info": r"Median (med) Lorentz factor of the beam.",
        "type": "float",
    },
    "g_std": {
        "name_latex": r"$\sigma_{\gamma,\mathrm{std}}$",
        "units": "MeV",
        "info": r"Standard deviation (std) of the Lorentz factor of the beam",
        "type": "float",
    },
    "g_mad": {
        "name_latex": r"$\sigma_{\gamma,\mathrm{mad}}$",
        "info": r"Median absolute deviation (mad) of the Lorentz factor of the beam.",
        "type": "float",
    },
    # Beam sizes
    "sigma_x": {
        "name_latex": r"$\sigma_{x}$",
        "units": "m",
        "info": r"RMS size in x.",
        "type": "float",
    },
    "sigma_y": {
        "name_latex": r"$\sigma_{y}$",
        "units": "m",
        "info": r"RMS size in y.",
        "type": "float",
    },
    "sigma_z": {
        "name_latex": r"$\sigma_{z}$",
        "units": "m",
        "info": r"RMS size in z.",
        "type": "float",
    },
    "sigma_ux": {
        "name_latex": r"$\sigma_{u_x}$",
        "info": r"RMS size in $u_x$.",
        "type": "float",
    },
    "sigma_uy": {
        "name_latex": r"$\sigma_{u_y}$",
        "info": r"RMS size in $u_y$.",
        "type": "float",
    },
    "sigma_uz": {
        "name_latex": r"$\sigma_{z}$",
        "info": r"RMS size in $u_z$.",
        "type": "float",
    },
    "sigma_Ek": {
        "name_latex": r"$\sigma_{E,\mathrm{std}}$",
        "units": "MeV",
        "info": r"RMS size of the kinetic energy. Same as Ek_std the standard deviation (std) of the kinetic energy of the beam.",
        "type": "float",
    },
    "sigma_dp": {
        "name_latex": r"$\sigma_{\delta p / p}$",
        "units": "%",
        "info": r"Standard deviation (std) of the particle momenta variation in percentage.",
        "type": "float",
    },
    "sigma_xp": {
        "name_latex": r"$\sigma_{x'}$",
        "units": "rad",
        "info": r"RMS angle $x'$ the beam",
        "type": "float",
    },
    "sigma_yp": {
        "name_latex": r"$\sigma_{y'}$",
        "units": "rad",
        "info": r"RMS angle $y'$ the beam",
        "type": "float",
    },
    # Average mean velocity
    "betaz_avg": {
        "name_latex": r"$<\beta_{z}>$",
        "info": r"Average normalized velocity $<\beta_z> = <v_z/c>$ in z.",
        "type": "float",
    },
    # dp/p
    "p_avg": {
        "name_latex": r"$<p>$",
        "units": "m/s",
        "info": r"Average momenta $<p>$.",
        "type": "float",
    },
    "dp_avg": {
        "name_latex": r"$<\delta p/p>$",
        "info": r"Average particle momenta variation $<\delta p/p>$.",
        "type": "float",
    },
    # Divergences
    "x_divergence": {
        "name_latex": r"Horizontal divergence",
        "units": "rad",
        "info": r"Divergence of the trace momentum $x'$ the beam.",
        "type": "float",
    },
    "y_divergence": {
        "name_latex": r"Vertical Divergence",
        "units": "rad",
        "info": r"Divergence of the trace momentum $y'$ the beam.",
        "type": "float",
    },
    # Dispersions
    "x_dispersion": {
        "name_latex": r"Horizontal divergence",
        "units": "m/rad",
        "info": r"Horizontal $x$ dispersion",
        "type": "float",
    },
    "y_dispersion": {
        "name_latex": r"Vertical divergence",
        "units": "m/rad",
        "info": r"Horizontal $y$ dispersion",
        "type": "float",
    },
    # Center of mass
    "x_avg": {
        "name_latex": r"$x$",
        "units": "m",
        "info": r"Beam center of mass (average position) in $x$.",
        "type": "float",
    },
    "y_avg": {
        "name_latex": r"$y$",
        "units": "m",
        "info": r"Beam center of mass (average position) in $y$.",
        "type": "float",
    },
    "z_avg": {
        "name_latex": r"$z$",
        "units": "m",
        "info": r"Beam center of mass (average position) in $z$.",
        "type": "float",
    },
    "ux_avg": {
        "name_latex": r"Average momenta",
        "info": r"Average normalised momentum $<u_x> = <\ gamma v_x / c> = <\gamma \beta_x)>$.",
        "type": "float",
    },
    "uy_avg": {
        "name_latex": r"Average momenta",
        "info": r"Average normalised momentum $<u_y> = <\ gamma v_y / c> = <\gamma \beta_y)>$.",
        "type": "float",
    },
    "uz_avg": {
        "name_latex": r"Average momenta",
        "info": r"Average normalised momentum $<u_z> = <\ gamma v_z / c> = <\gamma \beta_z)>$.",
        "type": "float",
    },
    "xp_avg": {
        "name_latex": r"$<x'>$",
        "units": "rad",
        "info": r"Average divergence $<x'>$.",
        "type": "float",
    },
    "yp_avg": {
        "name_latex": r"$<y'>$",
        "units": "rad",
        "info": r"Average divergence $<y'>$.",
        "type": "float",
    },
    # <x**2>
    "xx_avg": {
        "name_latex": r"$<x^2>$",
        "units": "m2",
        "info": r"$<x^2>$.",
        "type": "float",
    },
    "yy_avg": {
        "name_latex": r"$<y^2>$",
        "units": "m2",
        "info": r"$<y^2>$.",
        "type": "float",
    },
    "xy_avg": {
        "name_latex": r"$<x y>$",
        "units": "m2",
        "info": r"$<x y>$.",
        "type": "float",
    },
    "xz_avg": {
        "name_latex": r"$<x z>$",
        "units": "m2",
        "info": r"$<x z>$.",
        "type": "float",
    },
    "yz_avg": {
        "name_latex": r"$<y z>$",
        "units": "m2",
        "info": r"$<y z>$.",
        "type": "float",
    },
    "zz_avg": {
        "name_latex": r"$<z^2>$",
        "units": "m2",
        "info": r"$<z^2>$.",
        "type": "float",
    },
    # <x'x'>
    "xpxp_avg": {
        "name_latex": r"$<x'^2>$",
        "units": "rad2",
        "info": r"$<x'^2>$.",
        "type": "float",
    },
    "ypyp_avg": {
        "name_latex": r"$<y'^2>$",
        "units": "rad2",
        "info": r"$<y'^2>$.",
        "type": "float",
    },
    "xpyp_avg": {
        "name_latex": r"$<x' y'>$",
        "units": "rad2",
        "info": r"$<x' y'>$.",
        "type": "float",
    },
    "xpdp_avg": {
        "name_latex": r"$<x' \delta p / p>$",
        "units": "rad2",
        "info": r"$<x' \delta p / p>$.",
        "type": "float",
    },
    "ypdp_avg": {
        "name_latex": r"$<y' \delta p / p>$",
        "units": "rad2",
        "info": r"$<y' \delta p / p>$.",
        "type": "float",
    },
    # <x x'>
    "xxp_avg": {
        "name_latex": r"$<x x'>$",
        "units": "m.rad",
        "info": r"$<x x'>$.",
        "type": "float",
    },
    "yyp_avg": {
        "name_latex": r"$<y y'>$",
        "units": "m.rad",
        "info": r"$<y y'>$.",
        "type": "float",
    },
    "xyp_avg": {
        "name_latex": r"$<x y'>$",
        "units": "m.rad",
        "info": r"$<x y'>$.",
        "type": "float",
    },
    "yxp_avg": {
        "name_latex": r"$<x' y>$",
        "units": "m.rad",
        "info": r"$<x' y>$.",
        "type": "float",
    },
    "zxp_avg": {
        "name_latex": r"$<x' z>$",
        "units": "m.rad",
        "info": r"$<x' z>$.",
        "type": "float",
    },
    "zyp_avg": {
        "name_latex": r"$<y' z>$",
        "units": "m.rad",
        "info": r"$<y' z>$.",
        "type": "float",
    },
    "xdp_avg": {
        "name_latex": r"$<x \delta p /p >$",
        "units": "m.rad",
        "info": r"$<x \delta p /p >$.",
        "type": "float",
    },
    "ydp_avg": {
        "name_latex": r"$<y \delta p /p >$",
        "units": "m.rad",
        "info": r"$<y \delta p /p >$.",
        "type": "float",
    },
    "zdp_avg": {
        "name_latex": r"$<z \delta p /p >$",
        "units": "m.rad",
        "info": r"$<z \delta p /p >$.",
        "type": "float",
    },
    "dpdp_avg": {
        "name_latex": r"$<(\delta p /p)^2>$",
        "units": "m.rad",
        "info": r"$<(\delta p /p)^2>$.",
        "type": "float",
    },
    "sigma_matrix": {
        "name_latex": r"\sigma_M",
        "info": r"Beam sigma matrix (not multiplied by 5!).",
        "type": "np.ndarray",
    },
    # Trace emittances
    "emit_rms_x": {
        "name_latex": r"$\epsilon_{xx'}$",
        "units": "pi.mm.mrad",
        "info": r"Unnormalized trace emittance 1-rms $\gamma \beta \epsilon_{xx'}$ in $x$.",
        "type": "float",
    },
    "emit_rms_y": {
        "name_latex": r"$\epsilon_{yy'}$",
        "units": "pi.mm.mrad",
        "info": r"Unnormalized trace emittance 1-rms $\gamma \beta \epsilon_{yy'}$ in $y$.",
        "type": "float",
    },
    "emit_rms_z": {
        "name_latex": r"$\epsilon_{z \delta p /p}$",
        "units": "pi.mm.mrad",
        "info": r"Unnormalized trace emittance 1-rms $\epsilon_{z\delta p/p'}$ in $z$.",
        "type": "float",
    },
    "emit_norm_rms_x": {
        "name_latex": r"$\tilde{\epsilon}_{xx'}$",
        "units": "pi.mm.mrad",
        "info": r"Normalized trace emittance 1-rms $\gamma \beta \epsilon_{xx'}$ in x",
        "type": "float",
    },
    "emit_norm_rms_y": {
        "name_latex": r"$\tilde{\epsilon}_{yy'}$",
        "units": "pi.mm.mrad",
        "info": r"Normalized trace emittance 1-rms $\gamma \beta \epsilon_{yy'}$ in y",
        "type": "float",
    },
    "emit_norm_rms_z": {
        "name_latex": r"$\tilde{\epsilon}_{z\delta p/p'}$",
        "units": "pi.mm.%",
        "info": r"Normalized trace emittance 1-rms $\gamma \beta \epsilon_{z\delta p/p}$ in $z$.",
        "type": "float",
    },
    "emit_norm_rms_4D": {
        "name_latex": r"4D emittance",
        "units": "(pi.mm.mrad)2",
        "info": r"Normalized trace emittance 4D trace 1-rms emittance $\epsilon$ ($x$-$x'$-$y$-$y'$).",
        "type": "float",
    },
    "emit_norm_rms_6D": {
        "name_latex": r"6D emittance",
        "units": "(pi.mm.mrad)3",
        "info": r"Normalized trace emittance 6D trace 1-rms emittance $\epsilon$ ($x$-$x'$-$y$-$y'$-$z$-$\delta p / p$).",
        "type": "float",
    },
    # Twiss parameters
    "beta_x": {
        "name_latex": r"$\beta_x$",
        "units": "mm/mrad",
        "info": r"Beam Twiss coefficient $\beta_x$ in $x$.",
        "type": "float",
    },
    "beta_y": {
        "name_latex": r"$\beta_y$",
        "units": "mm/mrad",
        "info": r"Beam Twiss coefficient $\beta_y$ in $y$.",
        "type": "float",
    },
    "beta_z": {
        "name_latex": r"$\beta_z$",
        "units": "mm/pi/%",
        "info": r"Beam Twiss coefficient $\beta_z$ in $z$.",
        "type": "float",
    },
    "gamma_x": {
        "name_latex": r"$\gamma_x$",
        "units": "mrad/mm",
        "info": r"Beam Twiss coefficient $\gamma_x$ in $x$.",
        "type": "float",
    },
    "gamma_y": {
        "name_latex": r"$\gamma_y$",
        "units": "mrad/mm",
        "info": r"Beam Twiss coefficient $\gamma_y$ in $y$.",
        "type": "float",
    },
    "gamma_z": {
        "name_latex": r"$\gamma_z$",
        "units": "mrad/mm",
        "info": r"Beam Twiss coefficient $\gamma_z$ in $z$.",
        "type": "float",
    },
    "alpha_x": {
        "name_latex": r"$\alpha_x$",
        "info": r"Beam Twiss coefficient $\alpha_x$ in $x$.",
        "type": "float",
    },
    "alpha_y": {
        "name_latex": r"$\alpha_y$",
        "info": r"Beam Twiss coefficient $\alpha_y$ in $y$.",
        "type": "float",
    },
    "alpha_z": {
        "name_latex": r"$\alpha_z$",
        "info": r"Beam Twiss coefficient $\alpha_z$ in $z$.",
        "type": "float",
    },
    # ******************** dQ/dE histogramm
    "Ek_hist_xaxis": {
        "name_latex": r"Energy",
        "units": "MeV",
        "info": r"Abscissa axis for dQ/dE with E the relativistic kinetic energy (dE=1).",
        "type": "np.ndarray",
    },
    "Ek_hist_yaxis": {
        "name_latex": r"Charge",
        "units": "pC",
        "info": r"Ordinate axis for dQ/dE with E the relativistic kinetic energy (dE=1).",
        "type": "np.ndarray",
    },
    "Ek_hist_peak": {
        "name_latex": r"Energy peak",
        "units": "MeV",
        "info": r"Peak of dQ/dE with E the relativistic kinetic energy (dE=1).",
        "type": "float",
    },
    "Ek_hist_fwhm": {
        "name_latex": r"Energy FWHM",
        "units": "MeV",
        "info": r"The full width at half maximum (FWHM) of dQ/dE with E the relativistic kinetic energy (dE=1).",
        "type": "float",
    },
}

STD_DATABASE_BEAM_DF = pd.DataFrame.from_dict(STD_DATABASE_BEAM, orient="index")


# Laser data type
STD_DATABASE_LASER: Dict = {
    # ******************** Laser
    "a0": {
        "name_latex": r"$a_0$",
        "info": r"Maximum laser strength parameter.",
        "type": "float",
    },
    "lambda0": {
        "name_latex": r"$\lambda_0$",
        "units": "m",
        "info": r"Laser wavelength.",
        "type": "float",
    },
    "omega0": {
        "name_latex": r"$\omega_0$",
        "units": "rad/s",
        "info": r"Laser angular frequency $2 \pi c / \lambda_0$",
        "type": "float",
    },
    "waist0": {
        "name_latex": r"$w_0$",
        "units": "m",
        "info": r"Maximum laser waist.",
        "type": "float",
    },
    "duration_FWHM": {
        "name_latex": r"$\tau_0$",
        "units": "s",
        "info": r"Laser FWHM duration.",
        "type": "float",
    },
    "n_crit": {
        "name_latex": r"$\omega_{\mathrm{pe}}$",
        "units": "m-3",
        "info": r"Laser critical density.",
        "type": "float",
    },
    "Efield_laser": {
        "name_latex": r"$\omega_{\mathrm{pe}}$",
        "units": "V/m",
        "info": r"Reference laser electric field $E_{0} = m_{\mathrm{e}} c \omega_0 / e$",
        "type": "float",
    },
    # ******************** Laser simulation
    "z_focalisation": {
        "name_latex": r"$z_{\rm foc}$",
        "units": "m",
        "info": r"Laser focalisation position from the beginning (left border) of the simulation box.",
        "type": "float",
    },
    "z_laser_from_right_border": {
        "name_latex": r"Laser position from right border",
        "units": "m",
        "info": r"Laser focalisation position from the end (right border) of the moving windows box.",
        "type": "float",
    },
    # ******************** Laser-plasma parameters
    "plasma_density_ref": {
        "name_latex": r"$n_{\mathrm{e}}",
        "units": "m-3",
        "info": r"Reference plasma density for laser data.",
        "type": "float",
    },
    "omega_pe": {
        "name_latex": r"$\omega_{\mathrm{pe}}$",
        "units": "rad/s",
        "info": r"Plasma angular frequency.",
        "type": "float",
    },
    "energy_laser": {
        "name_latex": r"Energy",
        "units": "J",
        "info": r"Total laser energy produced.",
        "type": "float",
    },
    "power_laser": {
        "name_latex": r"Power",
        "units": "TW",
        "info": r"Total laser power (through the waist).",
        "type": "float",
    },
    "power_critical": {
        "name_latex": r"Power",
        "units": "TW",
        "info": r"Critical power in the plasma.",
        "type": "float",
    },
    "power_ratio_critical": {
        "name_latex": r"Power",
        "info": r"Ratio of the laser power over the critical power.",
        "type": "float",
    },
    "z_FWHM_intensity": {
        "name_latex": r"Longitudinal FWHM intensity",
        "units": "m",
        "info": r"FWHM length of the intensity in the longitudinal direction.",
        "type": "float",
    },
    "r_FWHM_intensity": {
        "name_latex": r"Transverse FWHM intensity",
        "units": "m",
        "info": r"FWHM length of the intensity in the transverse direction.",
        "type": "float",
    },
    "z_FWHM_field": {
        "name_latex": r"Longitudinal FWHM field amplitude",
        "units": "m",
        "info": r"FWHM length of the laser field in the longitudinal $z$ direction.",
        "type": "float",
    },
    "r_FWHM_field": {
        "name_latex": r"Transverse FWHM field amplitude",
        "units": "m",
        "info": r"FWHM length of the laser field in the transverse $r$ direction.",
        "type": "float",
    },
}


STD_DATABASE_LASER_DF = pd.DataFrame.from_dict(STD_DATABASE_LASER, orient="index")

# LWFA simulation data type
STD_DATABASE_LWFA_STEP: Dict = {
    # ******************** Simulations
    "dt": {
        "name_latex": r"$dt$",
        "units": "s",
        "info": r"Simulation time step.",
        "type": "float",
    },
    "time": {
        "name_latex": r"$t$",
        "units": "s",
        "info": r"Simulation time.",
        "type": "float",
    },
    "timestep": {
        "name_latex": r"timestep",
        "info": r"Simulation time step index.",
        "type": "int",
    },
    # ******************** Grid
    "Nz": {
        "name_latex": r"$N_z$",
        "info": r"Simulation total number of nodes in the longitudinal $z$ direction.",
        "type": "int",
    },
    "Nr": {
        "name_latex": r"$N_r$",
        "info": r"Simulation total number of nodes in the transverse $r$ direction.",
        "type": "int",
    },
    # ******************** Fields
    "zfield": {
        "name_latex": r"$z$",
        "units": "m",
        "info": r"Grid positions z of fields.",
        "type": "np.ndarray",
    },
    "rfield": {
        "name_latex": r"$r$",
        "units": "m",
        "info": r"Grid positions r of fields.",
        "type": "np.ndarray",
    },
    "rho1D": {
        "name_latex": r"$\rho$",
        "units": "C/m3",
        "info": r"1D charge density $\rho$ (negatif for electrons).",
        "type": "np.ndarray",
    },
    "rho2D": {
        "name_latex": r"$\rho$",
        "units": "C/m3",
        "info": r"2D charge density $\rho$ (negatif for electrons).",
        "type": "np.ndarray",
    },
    "density1D": {
        "name_latex": r"$n_{\mathrm{e}}$",
        "units": "m-3",
        "info": r"1D density $n_{\mathrm{e}} = - \rho / e$ (for electrons).",
        "type": "np.ndarray",
    },
    "density2D": {
        "name_latex": r"$n_{\mathrm{e}}$",
        "units": "m-3",
        "info": r"2D density $n_{\mathrm{e}} = - \rho / e$ (for electrons).",
        "type": "np.ndarray",
    },
    "Ex1D": {
        "name_latex": r"$E_{x}$",
        "units": "V/m",
        "info": r"1D electric field in $x$.",
        "type": "np.ndarray",
    },
    "Ex2D": {
        "name_latex": r"$E_{x}$",
        "units": "V/m",
        "info": r"2D electric field in $x$.",
        "type": "np.ndarray",
    },
    "Ey1D": {
        "name_latex": r"$E_{y}$",
        "units": "V/m",
        "info": r"1D electric field in $y$.",
        "type": "np.ndarray",
    },
    "Ey2D": {
        "name_latex": r"$E_{y}$",
        "units": "V/m",
        "info": r"2D electric field in $y$.",
        "type": "np.ndarray",
    },
    "Ey1D_env": {
        "name_latex": r"$E_{y}$",
        "units": "V/m",
        "info": r"Enveloppe of the 1D electric field in $y$.",
        "type": "np.ndarray",
    },
    "Ey2D_env": {
        "name_latex": r"$E_{y}$",
        "units": "V/m",
        "info": r"Enveloppe of the 2D electric field in $y$.",
        "type": "np.ndarray",
    },
    "Ez1D": {
        "name_latex": r"$E_{z}$",
        "units": "V/m",
        "info": r"1D electric wakefield.",
        "type": "np.ndarray",
    },
    "Ez2D": {
        "name_latex": r"$E_{z}$",
        "units": "V/m",
        "info": r"2D electric wakefield.",
        "type": "np.ndarray",
    },
    "Er1D": {
        "name_latex": r"$E_{r}$",
        "units": "V/m",
        "info": r"1D electric field in $r$.",
        "type": "np.ndarray",
    },
    "Er2D": {
        "name_latex": r"$E_{r}$",
        "units": "V/m",
        "info": r"2D electric field in $r$.",
        "type": "np.ndarray",
    },
    "Et1D": {
        "name_latex": r"$E_{\theta}$",
        "units": "V/m",
        "info": r"1D electric field in $\theta$.",
        "type": "np.ndarray",
    },
    "Et2D": {
        "name_latex": r"$E_{\theta}$",
        "units": "V/m",
        "info": r"2D electric field in $\theta$.",
        "type": "np.ndarray",
    },
    "a01D": {
        "name_latex": r"$a_0$",
        "info": r"Laser intensity in 1D",
        "type": "np.ndarray",
    },
    "a02D": {
        "name_latex": r"$a_0$",
        "info": r"Laser intensity in 1D",
        "type": "np.ndarray",
    },
    "a0_max": {
        "name_latex": r"$a_0$",
        "info": r"Maximum laser strength parameter.",
        "type": "float",
    },
    "waist0_max": {
        "name_latex": r"$w_0$",
        "units": "m",
        "info": r"Maximum laser waist.",
        "type": "float",
    },
    "zfield_a0": {
        "name_latex": r"$z$",
        "units": "m",
        "info": r"Grid positions z of a0_max",
        "type": "np.ndarray",
    },
    "z_FWHM_intensity_simu": {
        "name_latex": r"Longitudinal FWHM intensity",
        "units": "m",
        "info": r"FWHM length of the intensity in the longitudinal direction in simulations.",
        "type": "float",
    },
    "r_FWHM_intensity_simu": {
        "name_latex": r"Transverse FWHM intensity",
        "units": "m",
        "info": r"FWHM length of the intensity in the transverse direction in simulations.",
        "type": "float",
    },
}

STD_DATABASE_LASER_LWFA_STEP_DF = pd.DataFrame.from_dict(
    STD_DATABASE_LWFA_STEP, orient="index"
)


# LWFA simulation data type for steps
STD_DATABASE_LWFA_STEPS: Dict = {
    # ******************** Simulations
    "timesteps": {
        "name_latex": r"timesteps",
        "info": r"Simulation time step indexes.",
        "type": "np.ndarray",
    },
}

STD_DATABASE_LASER_LWFA_STEPS_DF = pd.DataFrame.from_dict(
    STD_DATABASE_LWFA_STEPS, orient="index"
)

# Combine dictionaries (python 3.9)
# STD_DATABASE = {}
STD_DATABASE: Dict = (
    STD_DATABASE_PROJECT
    | STD_DATABASE_BEAM
    | STD_DATABASE_LASER
    | STD_DATABASE_LWFA_STEP
    | STD_DATABASE_LWFA_STEPS
)

STD_DATABASE_DF = pd.DataFrame.from_dict(STD_DATABASE, orient="index")
