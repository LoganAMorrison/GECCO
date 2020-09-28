#!/usr/bin/python

"""
Script for generating a plot of GECCO's capabilities of detecting a 511 keV
line.
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
    gecco_bc = 7.4e-8
    gecco_wc = 3.2e-7

    phi_511 = 3e-4
    N_MSP = (2.7e3, 0.9e3)

    ds = np.logspace(-2, 2, 100)
    num_bc = phi_511 / gecco_bc * (8.5 / ds) ** 2
    num_wc = phi_511 / gecco_wc * (8.5 / ds) ** 2

    FIG_WIDTH = 7
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 7 / GOLD_RATIO))
    # plt.figure(dpi=150, figsize=(FIG_WIDTH, FIG_WIDTH / GOLD_RATIO))

    # PLOT 1

    Y_MIN = 1e2
    Y_MAX = 1e8

    ax1.fill_between(
        ds,
        num_bc,
        num_wc,
        alpha=0.5,
        color="steelblue",
        label="GECCO (best-case)",
    )
    ax1.fill_between(
        ds, num_wc, alpha=0.5, color="firebrick", label="GECCO (conservative)"
    )
    # ax1.text(0.7, 6e4, "GECCO (best-case)", rotation=-45, fontsize=10)
    # ax1.text(0.7, 1e4, "GECCO (conservative)", rotation=-45, fontsize=10)

    ax1.plot(ds, [N_MSP[0] for _ in ds], ls="-", lw=2, c="k")
    ax1.fill_between(
        ds,
        [N_MSP[0] + N_MSP[1] for _ in ds],
        [N_MSP[0] - N_MSP[1] for _ in ds],
        # color="mediumorchid",
        color="teal",
        alpha=0.5,
    )

    ax1.text(1, 1e3, "MSP", fontsize=12)

    # Wolf-Rayet
    D_WR = 0.350
    ax1.vlines(D_WR, Y_MIN, Y_MAX, colors="k", linestyles="--")
    ax1.text(D_WR * 0.75, 4.5e3, "Wolf-Rayet", rotation=90, fontsize=9, c="k")
    # LMXB 4U 1700+24
    D_LMXB = 0.42
    ax1.vlines(D_LMXB, Y_MIN, Y_MAX, colors="k", linestyles="--")
    ax1.text(
        D_LMXB * 1.1, 4.5e3, "LMXB 4U 1700+24", rotation=90, fontsize=9, c="k"
    )
    # MSP J0427-4715
    D_MSP = 0.16
    ax1.vlines(D_MSP, Y_MIN, Y_MAX, colors="k", linestyles="--")
    ax1.text(
        D_MSP * 0.75, 4.5e3, "MSP J0427-4715", rotation=90, fontsize=9, c="k"
    )

    ax1.set_xlim([0.1, 20])
    ax1.set_ylim(Y_MIN, Y_MAX)

    ax1.set_yscale("log")
    ax1.set_xscale("log")

    ax1.set_ylabel(r"$N_{\mathrm{src}}$", fontsize=16)
    ax1.set_xlabel(r"$d_{\mathrm{src}} \ (\mathrm{kpc})$", fontsize=16)

    ax1.legend(frameon=False)

    # PLOT 2
    phis = [
        PHI_M31,
        PHI_M33,
        PHI_DRACO,
        PHI_URSA_MINOR,
        PHI_FORNAX_CL,
        PHI_COMA_CL,
    ]
    for i, phi in enumerate(phis):
        ax2.errorbar(
            2 * i + 1.0,
            phi[0],
            phi[1],
            fmt="o",
            elinewidth=3,
            capsize=5,
            color=phi[3],
        )

    idxs = 2 * np.arange(len(phis)) + 1
    ax2.set_xticks(idxs)
    ax2.set_xticklabels([phi[2] for phi in phis])
    plt.setp(ax2.get_xticklabels(), rotation=45)

    X_MIN = 0
    X_MAX = np.max(idxs) + 2
    ax2.hlines(
        PHI_GECCO_BC, X_MIN, X_MAX, colors="k", label="GECCO (best-case)"
    )
    ax2.hlines(
        PHI_GECCO_WC,
        X_MIN,
        X_MAX,
        colors="k",
        label="GECCO (conservative)",
        linestyle="--",
    )

    # ax2.text(0.1, PHI_GECCO_BC * 1.2, "GECCO (best-case)")
    # ax2.text(0.1, PHI_GECCO_WC * 1.2, "GECCO (conservative)")

    ax2.set_xlim([X_MIN, X_MAX])

    ax2.set_yscale("log")

    ax2.set_ylabel(
        r"$\phi_{511} \ (\mathrm{cm}^{-2} \ \mathrm{s}^{-1})$", fontsize=16
    )

    ax2.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("/home/logan/Projects/GECCO/figures/511_combined.pdf")
