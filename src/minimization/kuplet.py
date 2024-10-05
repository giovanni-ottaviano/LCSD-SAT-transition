

# Generator and test for random doublets

import argparse
import numpy as np
from scipy.special import beta, gamma
from tabulate import tabulate

from argparse_utils import positiveint, floatrange



def rand_couples(N: int, P: int, rho: float, seed: int=None):

    """
        Random doublets generator. 
        Generate P doublets (N-dimensional vectors) with fixed overlap rho.
    """

    np.random.seed(seed)

    xi = np.concatenate((np.zeros(N-1), 1), axis=None)
    xibar = np.array([])

    for mu in range(P):
        bar = np.random.randn(N-1)
        bar = np.concatenate(((bar/np.linalg.norm(bar)) * np.sqrt(1. - rho**2), rho), axis=None)

        randmat = np.random.randn(N,N)
        Q, R = np.linalg.qr(randmat)
        Q *= np.random.choice([-1,1])

        if mu == 0:
            xibar = np.array([Q@xi, Q@bar])
        else:
            xibar = np.vstack((xibar, Q@xi, Q@bar))

    return xibar


def distribution_mean(N: int, R: float):

    """Mean of the distribution for random points on a N-dimensional sphere of radius R"""

    return 2**(N-1) * R * beta(N/2, N/2) / beta(N - 0.5, 0.5)


def distribution_variance(N: int, R: float):
    """Variance of the distribution for random points on a N-dimensional sphere of radius R"""

    # CAMBIA I NOMI DELLE VARIABILI
    numerator = 2**(2*N - 2.) * gamma(N/2)**4
    denominator = np.pi * gamma(N - 0.5)**2

    return (2. - (numerator/denominator))*R*R



def main():

    '''Test the random doublets generator with given parameters'''

    parser = argparse.ArgumentParser(description="Test ")
    parser.add_argument('N_min', help="Number of dimensions (min)", type=positiveint)
    parser.add_argument('N_max', help="Number of dimensions (max)", type=positiveint)
    parser.add_argument('P', help="Number of doublets", type=positiveint)
    parser.add_argument('R', help="Radius of the sphere", type=floatrange(0,10000))
    parser.add_argument('rho', help="Overlap", type=floatrange(-1,1))
    parser.add_argument('--seed', help="Seed for random number generator", type=int, default=None)

    args = parser.parse_args()

    N_min = args.N_min
    N_max = args.N_max
    P = args.P
    R = args.R
    rho = args.rho

    n_couples = P*(2*P - 1)

    if N_min > N_max:
        print("WARNING: Found N_min > N_max...Swapping!")
        N_min, N_max = N_max, N_min

    list_print = []
    for N in range(N_min, N_max + 1):
        distribution_vector = np.zeros(n_couples, dtype=np.float64)
        cumulative_sum, var_calculation = 0., 0.
        index = 0

        v = rand_couples(N, P, rho)

        for i in range(0, 2*P):
            for j in range(i + 1, 2*P):
                distribution_vector[index] = np.sqrt(np.sum((v[i] - v[j])**2))
                cumulative_sum += distribution_vector[index]

                index += 1

        ave_dist = cumulative_sum / n_couples

        for d in distribution_vector:
            var_calculation += (d - ave_dist)**2

        var_calculation /= n_couples - 1.

        list_print.append([N, ave_dist, distribution_mean(N,R), var_calculation, distribution_variance(N,R)])
        
    # Print results
    header = [
        "Dimension",
        "Mean",
        "Calc_mean",
        "Variance",
        "Calc_variance"
    ]

    print("\n")
    print(tabulate(list_print, headers=header, tablefmt="psql"))


if __name__ == '__main__':
    main()
