import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    fname = '/home/logan/Research/GECCO/blackhawk_v1.2/src/tables/gamma_tables/spin_0.txt'

    with open(fname, 'r') as f:
        xline = f.readline().split('   ')[5:]
        xline[-1] = xline[-1].split('\n')[0]
        xs = np.array([float(x) for x in xline])
        gbline = f.readline().split('   ')[2:]
        gbline[-1] = gbline[-1].split('\n')[0]
        gbs = np.array([float(gb) for gb in gbline])

    plt.plot(xs, gbs)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


