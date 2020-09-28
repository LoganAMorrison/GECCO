"""
Script for generating a plot of the 511 keV brightness vs. GECCO capabilities
"""

import matplotlib.pyplot as plt
import numpy as np


PHI_MW = (9.6e-4, 0.7e-4, "MW", "steelblue")
PHI_M31 = (5.76e-6, 4.71e-6, "M31", "firebrick")
PHI_M33 = (8.09e-8, 3.58e-8, "M33", "goldenrod")
PHI_DRACO = (1.49e-8, 0.62e-8, "Draco", "mediumorchid")
PHI_URSA_MINOR = (3.85e-8, 1.44e-8, "Ursa Minor", "teal")
PHI_FORNAX_CL = (7.98e-7, 4.55e-7, "Fornax Cl.", "darkslateblue")
PHI_COMA_CL = (1.86e-7, 1.70e-7, "Coma Cl.", "Peru")

PHI_GECCO_BC = 7.4e-8
PHI_GECCO_WC = 3.2e-7


GOLD_RATIO = (1.0 + np.sqrt(5)) / 2.0

if __name__ == "__main__":
    plt.figure(dpi=150, figsize=(7, 7 / GOLD_RATIO))

    phis = [
        PHI_M31,
        PHI_M33,
        PHI_DRACO,
        PHI_URSA_MINOR,
        PHI_FORNAX_CL,
        PHI_COMA_CL,
    ]
    for i, phi in enumerate(phis):
        plt.errorbar(
            2 * i + 1.0,
            phi[0],
            phi[1],
            fmt="o",
            elinewidth=3,
            capsize=5,
            color=phi[3],
        )

    idxs = 2 * np.arange(len(phis)) + 1
    plt.xticks(idxs, [phi[2] for phi in phis])

    X_MIN = 0
    X_MAX = np.max(idxs) + 2
    plt.hlines(PHI_GECCO_BC, X_MIN, X_MAX, colors="k")
    plt.hlines(PHI_GECCO_WC, X_MIN, X_MAX, colors="k")

    plt.text(0.1, PHI_GECCO_BC * 1.2, "GECCO (best-case)")
    plt.text(0.1, PHI_GECCO_WC * 1.2, "GECCO (conservative)")

    plt.xlim([X_MIN, X_MAX])

    plt.yscale("log")

    plt.ylabel(
        r"$\phi_{511} \ (\mathrm{cm}^{-2} \ \mathrm{s}^{-1})$", fontsize=16
    )

    plt.tight_layout()
    plt.savefig("/home/logan/Projects/GECCO/figures/511_source.pdf")
