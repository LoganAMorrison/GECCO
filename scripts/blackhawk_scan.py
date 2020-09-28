#!/bin/python

# Script for scanning over PBH masses between 1e15 and 1e18 and running
# BlackHawk to generate the gamma-ray spectra.

import numpy as np
import os
import pandas as pd

# Path to current directory
CUR_DIR = os.path.abspath(os.getcwd())

# Path to the directory containing BlackHawk code. On my machine, this is
# is the project root directory. Modify according to where you have blackhawk.
BLACKHAWK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "blackhawk_v1.2"
)

# Path to BlackHawk results directory
BLACKHAWK_RESULTS_DIR = os.path.join(BLACKHAWK_DIR, "results")

# Path to BlackHawk parameters file
PAR_FILE = os.path.join(BLACKHAWK_DIR, "parameters.txt")

# Path to where we will store the data from BlackHawk results
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"
)


def write_parameters(exponent, rel_dir):
    """
    Write the BlackHawk parameters file given the exponent of the black-hole
    mass.

    Parameters
    ----------
    exponent: float
        Exponent of the black-hole mass, i.e. M_PBH = 10**exponent g.
    directory: str
        Directory name of the results relative to BlackHawk results directory.
    """
    PAR_FILE = os.path.join(BLACKHAWK_DIR, "parameters.txt")
    with open(PAR_FILE, "w") as f:
        mass = 10.0 ** exponent
        f.write(
            f"""
# name of the output folder in results/
destination_folder = {rel_dir}

# quantity of information displayed (0=less, 1=more)
full_output = 1

# interpolation in the numerical tables (0=linear, 1=logarithmic)
interpolation_method = 0

# number of BH masses (should be the number of tabulated masses if spectrum_choice=5)
BHnumber = 1

# lowest BH mass in g (larger than the Planck mass)
Mmin = {mass}

# highest BH mass in g (larger than the Planck mass)
Mmax = {mass}



# number of Kerr spins
anumber = 1

# lowest Kerr spin
amin = 0

# highest Kerr spin
amax = 0.5



# form of the BH distribution: 0=Dirac, 1=log-normal, 2=power-law, 3=critical collapse, 4=peak theory, 5=uniform -1=user-defined
spectrum_choice = 0

# form of the spin dsitribution for each mass: 0=Dirac, 1=uniform, 2=gaussian
spectrum_choice_a = 0



# amplitude of the log-normal (mass density) distribution in g.cm^-3
amplitude_lognormal = 1.0

# amplitude of the log-normal (number density) distribution in cm^-3
amplitude_lognormal2 = 1.0

# dimensionless variance of the log-normal distribution
stand_dev_lognormal = 1.0

# characteristic mass of the log-normal distribution in g
crit_mass_lognormal = 1.0



# amplitude of the power-law distribution in g^(gamma-1).cm^-3
amplitude_powerlaw = 1.0

# equation of state of the Universe at the BH formation time P = w.rho
eqstate_powerlaw = 0.3333



# amplitude of the critical collapse distribution in g^(-2.85).cm^-3
amplitude_critical_collapse = 1.0

# characteristic mass of the critical collapse distribution in g
crit_mass_critical_collapse = 1.0



# amplitude of the uniform mass distribution in cm^(-3)
amplitude_uniform = 1.0



# standard deviation of the gaussian spin distribution
stand_dev_a_gaussian = 1.0

# mean of the gaussian spin distribution
mean_a_gaussian = 0.5



# table containing the User's BH distribution
table = table.txt



# initial integration time of the evolution of BH in s
tmin = 1e-30

# iteration limit when computing the time evolution of a single BH
limit = 5000



# number of primary particles energies to be simulated
Enumber = 1000

# minimal energy in GeV of the primary particles
Emin = 5e-7 # 0.5 keV

# maximal energy in GeV of the primary particles
Emax = 1.0

# number of primary particles (DO NOT MODIFY)
particle_number = 15

# 0=no graviton, 1=emission of gravitons
grav = 1



# 1=no secondary spectrum, 0=secondary spectrum computed
primary_only = 0



# 0=PYTHIA at the BBN epoch, 1=HERWIG at the BBN epoch, 2=PYTHIA (new) at the present epoch
hadronization_choice = 2
"""
        )


def collect_and_save_data(exponents, rel_dirs):
    """
    Walk through the BlackHawk results directory and collect all the data.
    """
    primary_photon = {"energies": None}
    primary_electron = {"energies": None}
    secondary_photon = {"energies": None}
    secondary_electron = {"energies": None}
    for i, (exponent, d) in enumerate(zip(exponents, rel_dirs)):
        path = os.path.join(
            BLACKHAWK_RESULTS_DIR, d, "instantaneous_primary_spectra.txt"
        )
        prim_engs, prim_dnde_g, prim_dnde_e = (
            np.genfromtxt(path, skip_header=2).T[0],
            np.genfromtxt(path, skip_header=2).T[1],
            np.genfromtxt(path, skip_header=2).T[7],
        )
        path = os.path.join(
            BLACKHAWK_RESULTS_DIR, d, "instantaneous_secondary_spectra.txt"
        )
        sec_engs, sec_dnde_g, sec_dnde_e = (
            np.genfromtxt(path, skip_header=2).T[0],
            np.genfromtxt(path, skip_header=2).T[1],
            np.genfromtxt(path, skip_header=2).T[2],
        )

        if i == 0:
            primary_photon["energies"] = prim_engs
            secondary_photon["energies"] = sec_engs

            primary_electron["energies"] = prim_engs
            secondary_electron["energies"] = sec_engs

        primary_photon[str(10 ** exponent)] = prim_dnde_g
        secondary_photon[str(10 ** exponent)] = sec_dnde_g

        primary_electron[str(10 ** exponent)] = prim_dnde_e
        secondary_electron[str(10 ** exponent)] = sec_dnde_e

    pd.DataFrame(primary_photon).to_csv(
        os.path.join(OUTPUT_DIR, "dnde_photon_primary.csv"), index=False
    )
    pd.DataFrame(secondary_photon).to_csv(
        os.path.join(OUTPUT_DIR, "dnde_photon_secondary.csv"), index=False
    )

    pd.DataFrame(primary_electron).to_csv(
        os.path.join(OUTPUT_DIR, "dnde_electron_primary.csv"), index=False
    )
    pd.DataFrame(secondary_electron).to_csv(
        os.path.join(OUTPUT_DIR, "dnde_electron_secondary.csv"), index=False
    )


if __name__ == "__main__":
    exponents = np.linspace(15.0, 18.6, 73)

    # Change directory to where the BlackHawk code is located
    os.chdir(BLACKHAWK_DIR)

    rel_dirs = [f"MPBH_1e{exponent}g" for exponent in exponents]

    # Run BlackHawk for each black-hole mass
    for i, (exponent, rel_dir) in enumerate(zip(exponents, rel_dirs)):
        write_parameters(exponent, rel_dir)
        os.system("echo 'y' 'y' | ./BlackHawk_inst.x parameters.txt")

    collect_and_save_data(exponents, rel_dirs)

    # Return to original directory
    os.chdir(CUR_DIR)
