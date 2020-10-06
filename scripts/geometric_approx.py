"""
Script for comparing geometric-approximation to greybody factors to BlackHawk.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

from utils import (
    BlackHawk,
    spectrum_geometic_approximation,
    MASS_CONVERSION,
    FIGURES_DIR,
)

if __name__ == "__main__":
    mpbhs_grams = [1e15, 1e16, 1e17, 1e18]
    mpbhs_grams_str = [r"10^{15}", r"10^{16}", r"10^{17}", r"10^{18}"]

    colors = ["steelblue", "firebrick", "goldenrod", "mediumorchid"]

    plt.figure(dpi=150)
    for mpbh, c in zip(mpbhs_grams, colors):
        blackhawk = BlackHawk(mpbh)
        blackhawk.min_primary_eng = 1e-6
        blackhawk.max_primary_eng = 1
        blackhawk.run()

        energies = blackhawk.primary["energies"]
        bh_spec = blackhawk.primary["electron"]
        geom = spectrum_geometic_approximation(
            energies, mpbh * MASS_CONVERSION, 1
        )

        plt.plot(energies * 1e3, bh_spec * 1e-3, ls="-", c=c)
        plt.plot(energies * 1e3, 4.0 * geom * 1e-3, ls="--", c=c)

    plt.vlines(3.5, 1e15, 1e20, colors="k", linestyles="-.")

    lines = []
    labels = []
    for c, m in zip(colors, mpbhs_grams_str):
        lines.append(Line2D([], [], ls="-", color=c))
        labels.append(r"$M={}$".format(m))

    lines.append(Line2D([], [], ls="--", color="k"))
    lines.append(Line2D([], [], ls="-", color="k"))
    lines.append(Line2D([], [], ls="-.", color="k"))
    labels.append("Geometric")
    labels.append("BlackHawk")
    labels.append(r"$E_{e^{\pm}} = 3.5 \ (\mathrm{MeV})$")
    plt.legend(lines, labels, loc=1)

    plt.xlabel(r"$E_{e^{\pm}} \ (\mathrm{MeV})$", fontsize=16)
    plt.ylabel(
        r"$\frac{dN}{dE_{e^{\pm}}} \ (\mathrm{MeV}^{-1}s^{-1})$", fontsize=16
    )
    plt.xlim([1e-1, 1000])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim([1e15, 1e20])
    plt.savefig(os.path.join(FIGURES_DIR, "geom_approx.pdf"))
