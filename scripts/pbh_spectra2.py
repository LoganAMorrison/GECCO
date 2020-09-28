"""
Script for generating plots of the gamma-ray spectra from PBHs
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import UnivariateSpline
from hazma.decay import neutral_pion, muon, charged_pion
from hazma.parameters import neutral_pion_mass as mpi0
from hazma.parameters import charged_pion_mass as mpi

GOLD_RATIO = (1.0 + np.sqrt(5)) / 2.0


mass_conversion = 5.60958884e23  # g to GeV
time_conversion = 1.519267407e24  # s to GeV^(-1)
leng_conversion = 5.06773058e13  # cm to GeV^(-1)
rate_conversion = (
    1.0e-100 * time_conversion * pow(leng_conversion, 3.0)
)  # cm^(-3)s^(-1)GeV^(-1) to GeV^3, plus the correction of 1.e-100
dens_conversion = 1.0e-100 * pow(
    leng_conversion, 3.0
)  # cm^(-3) to GeV^3, plus the correction of 1.e-100

G_NEWTON = (
    6.67408e-11
    * pow(leng_conversion * 100.0, 3.0)
    / (mass_conversion * 1000.0)
    / pow(time_conversion, 2.0)
)  # Newton constant in GeV
M_PLANK = 1.221e19  # Planck mass in GeV

# masses of the Standard Model particles (PDG 2017) in GeV
M_ELECTRON = 0.5109989461e-3
M_MUON = 105.6583745e-3
M_PION = mpi0 * 1e-3
M_PIONP = mpi * 1e-3
ALPHA_EM = 1.0 / 137.0  # Fine structure constant


def ap_spec(photon_energy, energy, mass, spin2):
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
    Q = 2.0 * energy
    x = 2 * photon_energy / Q
    mu = mass / Q
    if 0.0 < x < 1.0:
        if spin2 == 0:
            split = 2.0 * (1 - x) / x
        elif spin2 == 1:
            split = (1.0 + (1.0 - x) ** 2) / x
        else:
            raise ValueError("Invalid spin2")
        log = np.log((1.0 - x) / mu ** 2) - 1.0
        if log < 0.0:
            return 0.0
        else:
            return ALPHA_EM / np.pi * split * log / Q
    else:
        return 0.0


def get_greybody_factors(spin2):
    """
    Get the greybody factors for a particle with a spin equal to `spin2/2`.

    Parameters
    ----------
    spin2: int
        Twice the spin of the particle.

    Returns
    -------
    xs: np.array
       Scaled energies of the particle: x = 2*G*M*E where G is Newton's
       constant and M is the blackhole mass.
    greybodies: np.array
        Greybody factors corresponding to the particle energy in `energies`.
    """
    if spin2 == 0:
        fname = "/home/logan/Research/GECCO/blackhawk_v1.2/src/tables/gamma_tables/spin_0.txt"
    elif spin2 == 1:
        fname = "/home/logan/Research/GECCO/blackhawk_v1.2/src/tables/gamma_tables/spin_0_5.txt"
    elif spin2 == 2:
        fname = "/home/logan/Research/GECCO/blackhawk_v1.2/src/tables/gamma_tables/spin_1.txt"
    elif spin2 == 4:
        fname = "/home/logan/Research/GECCO/blackhawk_v1.2/src/tables/gamma_tables/spin_2.txt"
    else:
        raise ValueError("Only 2*spin = 0, 1, 2 and 4 are available.")

    with open(fname, "r") as f:
        xline = f.readline().split("   ")[5:]
        xline[-1] = xline[-1].split("\n")[0]
        xs = np.array([float(x) for x in xline])
        gbline = f.readline().split("   ")[2:]
        gbline[-1] = gbline[-1].split("\n")[0]
        gbs = np.array([float(gb) for gb in gbline])

    return xs, gbs


def convert_x_to_energy(x, m_pbh):
    """
    Convert x = 2*G*M*E into energy.

    Parameters
    ----------
    x: float
        Value of x to convert.
    m_pbh: float
        Mass of the black-hole in GeV.

    Returns
    -------
    energy: float
        Value of energy corresponding to `x`.
    """
    return x / (2.0 * G_NEWTON * m_pbh)


def black_hole_temperature(m_pbh):
    """
    Compute the temperature of a black-hole.

    Parameters
    ----------
    m_pbh: float
        Mass of the black-hole in GeV.
    """
    return 1.0 / (8.0 * np.pi * G_NEWTON * m_pbh)


def dnde_neutral_pion(egam, epi):
    """
    Compute the spectrum from the decay of a neutral pion.
    """
    if epi < M_PION:
        return 0.0
    p = np.sqrt(epi ** 2 - M_PION ** 2)
    beta = p / epi
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    eplus = M_PION / (2 * gamma * (1 - beta))
    eminus = M_PION / (2 * gamma * (1 + beta))

    # print("{}, {}, {}".format(eminus, egam, eplus))

    if eminus < egam < eplus:
        return 2 / (gamma * beta * M_PION)
    else:
        return 0.0


def dnde_charged_pion(egam, epi):
    """
    Compute the spectrum from the decay of a neutral pion.
    """
    if epi < M_PIONP:
        return 0.0
    return charged_pion(egam * 1e3, epi * 1e3) * 1e3 + ap_spec(
        egam, epi, M_PIONP, 0
    )


def compute_pion_spectrum(photon_energies, m_pbh):
    """
    compute the gamma-ray spectrum from pions.
    """
    xs, greybodies = get_greybody_factors(0)
    pion_energies = np.array([convert_x_to_energy(x, m_pbh) for x in xs])

    dnde_pis = np.array(
        [gb / (2.0 * np.pi) * time_conversion for gb in greybodies]
    )

    integrand = np.array(
        [
            [
                dnde_pi * dnde_neutral_pion(egam, epi)
                for (epi, dnde_pi) in zip(pion_energies, dnde_pis)
            ]
            for egam in photon_energies
        ]
    )
    return np.trapz(integrand, pion_energies)


def compute_cpion_spectrum(photon_energies, m_pbh):
    """
    Compute the gamma-ray spectrum from pions.
    """
    xs, greybodies = get_greybody_factors(0)
    pion_energies = np.array([convert_x_to_energy(x, m_pbh) for x in xs])

    dnde_pis = np.array(
        [2.0 * gb / (2.0 * np.pi) * time_conversion for gb in greybodies]
    )

    integrand = np.array(
        [
            [
                dnde_pi * dnde_charged_pion(egam, epi)
                for (epi, dnde_pi) in zip(pion_energies, dnde_pis)
            ]
            for egam in photon_energies
        ]
    )
    return np.trapz(integrand, pion_energies)


def compute_electron_spectrum(
    photon_energies, electron_energies, dnde_electron
):
    """
    Compute the FSR spectrum off electron evaporated from a PBH.
    """
    integrand = np.array(
        [
            [
                dnde_e * ap_spec(eg, ee, M_ELECTRON, 1)
                for (ee, dnde_e) in zip(electron_energies, dnde_electron)
            ]
            for eg in photon_energies
        ]
    )
    return np.trapz(integrand, electron_energies)


def compute_muon_spectrum(photon_energies, muon_energies, dnde_muon):
    """
    Compute the FSR + decay spectrum off muon evaporated from a PBH.
    """
    integrand = np.array(
        [
            [
                dnde_mu
                * (
                    ap_spec(eg, emu, M_MUON, 1)
                    + muon(eg * 1e3, emu * 1e3) * 1e3
                )
                for (emu, dnde_mu) in zip(muon_energies, dnde_muon)
            ]
            for eg in photon_energies
        ]
    )
    return np.trapz(integrand, muon_energies)


if __name__ == "__main__":
    # Load the data
    masses = [r"1e15", r"1e16", r"1e17", r"1e18"]
    m_pbhs = np.array([1e15, 1e16, 1e17, 1e18]) * mass_conversion
    pri_files = [
        "/home/logan/Research/GECCO/blackhawk_v1.2/results/MPBH_"
        + mass
        + ".0g/instantaneous_primary_spectra.txt"
        for mass in masses
    ]
    sec_files = [
        "/home/logan/Research/GECCO/blackhawk_v1.2/results/MPBH_"
        + mass
        + ".0g/instantaneous_secondary_spectra.txt"
        for mass in masses
    ]

    plt.figure(dpi=150, figsize=(7, 7 / GOLD_RATIO))

    labels = [r"10^{15}", r"10^{16}", r"10^{17}", r"10^{18}"]
    labels = [
        r"$M_{\mathrm{PBH}}=" + lab + r" \ \mathrm{g}$" for lab in labels
    ]
    colors = ["steelblue", "firebrick", "goldenrod", "mediumorchid"]

    for i, (pri_file, sec_file, lab, m_pbh) in enumerate(
        zip(pri_files, sec_files, labels, m_pbhs)
    ):
        # Photon spectrum
        engs, spec = np.genfromtxt(sec_file, skip_header=2).T[:2]
        # Photon + Electron + muon primary spectra
        engs_pri, dnde_g, dnde_e, dnde_mu = (
            np.genfromtxt(pri_file, skip_header=2).T[0],
            np.genfromtxt(pri_file, skip_header=2).T[1],
            np.genfromtxt(pri_file, skip_header=2).T[7],
            np.genfromtxt(pri_file, skip_header=2).T[8],
        )

        spec_electron = compute_electron_spectrum(engs_pri, engs_pri, dnde_e)
        spec_muon = compute_muon_spectrum(engs_pri, engs_pri, dnde_mu)
        spec_pion = compute_pion_spectrum(engs_pri, m_pbh)
        spec2 = spec_electron + dnde_g + spec_pion

        plt.plot(engs, engs * spec, ls="--", c=colors[i])
        plt.plot(engs_pri, engs_pri * spec2, ls="-", c=colors[i])
        # plt.plot(engs_pri, engs_pri ** 2 * spec_muon, ls=":")
        # plt.plot(engs_pri, engs_pri ** 2 * spec_pion, ls=":")

    plt.xlim([1e-6, 1])
    plt.ylim([1e7, 1e22])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(
        r"$E_{\gamma} \frac{dN_{\gamma}}{dE_{\gamma}dt} \ (\mathrm{GeV}\mathrm{s}^{-1})$",
        fontsize=16,
    )
    plt.xlabel(r"$E_{\gamma} \ (\mathrm{GeV})$", fontsize=16)

    lines = [Line2D([], [], color=colors[i]) for i in range(len(labels))]
    lines.append(Line2D([], [], color="k", ls="-",))
    lines.append(Line2D([], [], color="k", ls="--"))
    labels.append(r"$\mathrm{BH}_{\mathrm{prim}}+\mathrm{FSR}$")
    labels.append(r"$\mathrm{BH}_{\mathrm{sec}}$")
    plt.legend(lines, labels)

    plt.tight_layout()
    # plt.show()
    plt.savefig("/home/logan/Research/GECCO/figures/PBH_spectra.pdf")

