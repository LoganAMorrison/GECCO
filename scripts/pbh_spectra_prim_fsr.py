"""
Script for generating plots of the gamma-ray spectra from PBHs
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

ALPHA_EM = 1.0 / 137.035999  # Fine structure constant
M_ELECTRON = 0.5019989500e-3  # Electron mass in GeV

BH_RESULTS_DIR = "/home/logan/Research/GECCO/blackhawk_v1.2/results"
DATA_OUT_DIR = "/home/logan/Research/GECCO/scripts"


def dir_to_mass_str(d):
    """
    Parse the directory and extract the PBH mass as a string
    """
    return d.split("/")[-1].split("_")[1][:-1]


def dir_to_mass(d):
    """
    Hook into the 'parameters.txt' file located in the directory and extract
    the PBH mass.
    """
    with open(os.path.join(d, "parameters.txt")) as f:
        lines = f.readlines()
        for line in lines:
            if line[:4] == "Mmin":
                return float(line.split("=")[1])
        raise RuntimeError(
            "Could not find PBH mass in: " + d + "/parameters.txt"
        )


def get_data_directories():
    """
    Find all of the data directories containing results from BlackHawk.
    """
    dirs = [x[0] for x in os.walk(BH_RESULTS_DIR)][1:]
    dirs.sort(key=dir_to_mass)
    return dirs


def ap_spec(photon_energy, fermion_energy, fermion_mass):
    """
    Compute the Altarelli-Parisi spectrum from a fermion.

    Parameters
    ----------
    photon_energy: float
        Energy of the photon.
    fermion_energy: float
        Energy of the radiating fermion.
    fermion_mass: float
        Mass of the radiating fermion.

    Returns
    -------
    dnde: float
        Photon spectrum at `photon_energy`.
    """
    Q = 2.0 * fermion_energy
    x = 2 * photon_energy / Q
    mu = fermion_mass / Q
    if 0.0 < x < 1.0:
        split = (1.0 + (1.0 - x) ** 2) / x
        log = np.log((1.0 - x) / mu ** 2) - 1.0
        if log < 0.0:
            return 0.0
        else:
            return ALPHA_EM / np.pi * split * log / Q
    else:
        return 0.0


def compute_electron_spectrum(
    photon_energies, electron_energies, dnde_electron
):
    """
    Compute the FSR spectrum off electron evaporated from a PBH.
    """

    integrand = np.array(
        [
            [
                dnde_e * ap_spec(eg, ee, M_ELECTRON)
                for (ee, dnde_e) in zip(electron_energies, dnde_electron)
            ]
            for eg in photon_energies
        ]
    )
    return np.trapz(integrand, electron_energies)


def generate_data():
    """
    Extract data from BlackHawk results and create a pandas DataFrame
    containing the photon energies in the first column and gamma-ray spectra
    for various black-hole masses in the remaining columns.
    """
    dirs = get_data_directories()

    all_data_pri = {"photon_energies": None}

    for i in tqdm(range(len(dirs))):
        d = dirs[i]
        path_pri = os.path.join(d, "instantaneous_primary_spectra.txt")
        engs, spec_photon, dnde_e = (
            np.genfromtxt(path_pri, skip_header=2).T[0],
            np.genfromtxt(path_pri, skip_header=2).T[1],
            np.genfromtxt(path_pri, skip_header=2).T[7],
        )

        if i == 0:
            all_data_pri["photon_energies"] = engs

        spec_electron = compute_electron_spectrum(engs, engs, dnde_e)
        spec = spec_photon + spec_electron
        all_data_pri[dir_to_mass_str(d)] = spec

    return pd.DataFrame(all_data_pri)


def plot_data(df):
    """
    Plot all of the spectra.
    """
    plt.figure(dpi=100)

    energies = df["photon_energies"]
    print(df.columns)
    for col in df.columns:
        if col != "photon_energies":
            spec = df[col]
            plt.plot(energies, energies ** 2 * spec)

    plt.xlim([1e-6, 1])
    plt.ylim([1e1, 1e20])
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    df = generate_data()
    df.to_csv(os.path.join(DATA_OUT_DIR, "spectra.csv"))
    plot_data(df)

