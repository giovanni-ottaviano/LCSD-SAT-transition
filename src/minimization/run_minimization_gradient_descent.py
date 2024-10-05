# Script for gradient descent minimization algorithm

import os
import argparse
import multiprocessing as mp
import numpy as np

from loss_minimization_utils import (
    compute_feasible_num_pytorch,
    loss_pytorch
)
from argparse_utils import positiveint, floatrange



def main():

    '''Run cost function minimization with gradient descent algorithm and given parameters'''

    parser = argparse.ArgumentParser(description="Solve QCP with non-convex constraints by minimizing a loss function via gradient descent")
    parser.add_argument('N', help="Number of dimensions", type=positiveint)
    parser.add_argument('alpha_min', help="Number of doublets (min)", type=floatrange(0,10000))
    parser.add_argument('alpha_max', help="Number of doublets (max)", type=floatrange(0,10000))
    parser.add_argument('rho', help="Overlap", type=floatrange(-1,1))
    parser.add_argument('--x0', help="Starting point for minimization", type=float, default=5.)
    parser.add_argument('--lr', help="Learning rate", type=floatrange(1e-10,1e+2), default=0.1)
    parser.add_argument('--epochs', help="Number of epochs", type=positiveint, default=500)
    parser.add_argument('--n_seeds', help="Number different models to optimize (for each P)", type=positiveint, default=100)
    parser.add_argument('--cores', help="Number of cores for multiprocessing", type=positiveint, default=1)


    args = parser.parse_args()

    N = args.N
    alpha_min = args.alpha_min
    alpha_max = args.alpha_max
    rho = args.rho
    x0 = args.x0
    lr = args.lr
    n_epochs = args.epochs
    n_seeds = args.n_seeds
    cores = args.cores

    if alpha_min > alpha_max:
        print("WARNING: Found alpha_min > alpha_max...Swapping!")
        alpha_min, alpha_max = alpha_max, alpha_min

    # Check number of available CPUs
    try:
        if cores > len(os.sched_getaffinity(0)) or cores <= 0:
            print(f"WARNING: Invalid number of cores. {cores} will be changed to {len(os.sched_getaffinity(0))}\n")
            cores = len(os.sched_getaffinity(0))
    except AttributeError:
        print("Module 'os' has no attribute 'sched_getaffinity'. Skipping check on cores...")


    P_min = int(alpha_min * N)
    P_max = int(alpha_max * N)
    Ps = list(range(P_min, P_max + 1))

    if cores == 1:
        feasible_num = [
            compute_feasible_num_pytorch(
                cost_function=loss_pytorch,
                N=N,
                P=P,
                rho=rho,
                x0=x0,
                n_seeds=n_seeds,
                lr=lr,
                n_epochs=n_epochs
            ) for P in Ps
        ]
    else:
        try:
            mp.set_start_method('fork')
        except ValueError:
            print("Contex for'fork' cannot be found. Skipping...")

        pool = mp.Pool(processes=cores)
        with pool:
            feasible_num = pool.starmap(
                compute_feasible_num_pytorch, [(loss_pytorch, N, P, rho, x0, n_epochs, lr, n_seeds) for P in Ps]
            )

        pool.join()

    # Make ratio and write results to file
    feasible_fraction = np.array(feasible_num, dtype=float) / n_seeds
    save_vec = [(p, fract) for p, fract in zip(Ps, feasible_fraction)]

    file_name = f"../../results/GD_{N}_{str(rho).replace('.','')}.dat"
    np.savetxt(file_name, save_vec, fmt='%i %1.3f')



if __name__ == '__main__':
    main()
