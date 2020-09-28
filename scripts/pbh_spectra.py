"""
Script for generating plots of the gamma-ray spectra from PBHs
"""

import matplotlib.pyplot as plt
import numpy as np

GOLD_RATIO = (1.0 + np.sqrt(5)) / 2.0

if __name__ == "__main__":
    # Load the data
    masses = [r"1e15", r"1e16", r"1e17", r"1e18"]
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
    for (pri_file, sec_file, lab) in zip(pri_files, sec_files, labels):
        # engs, spec = np.genfromtxt(pri_file, skip_header=2).T[:2]
        # plt.plot(engs, engs ** 2 * spec, c="k")
        engs, spec = np.genfromtxt(sec_file, skip_header=2).T[:2]
        label = r"$M_{\mathrm{PBH}}=" + lab + r" \ \mathrm{g}$"
        plt.plot(engs, engs ** 2 * spec, label=label)

    plt.xlim([1e-6, 1])
    plt.ylim([1e1, 1e20])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(
        r"$E_{\gamma}^2 \frac{dN_{\gamma}}{dE_{\gamma}dt} \ (\mathrm{GeV}\mathrm{s}^{-1})$",
        fontsize=16,
    )
    plt.xlabel(r"$E_{\gamma} \ (\mathrm{GeV})$", fontsize=16)

    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/logan/Research/GECCO/figures/PBH_spectra.pdf")
