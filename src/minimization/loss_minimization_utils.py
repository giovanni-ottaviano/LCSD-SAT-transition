from typing import Callable
import numpy as np
from scipy.optimize import minimize
import torch

from kuplet import rand_couples


def loss_minimize(w: np.array, xi: np.array, xibar: np.array) -> float:

    """Function to be optimized via root finding algorithm"""

    assert xi.shape == xibar.shape

    lsum = 0.

    for c, cb in zip(xi, xibar):
        lsum += np.maximum(0, -(w@c) * (w@cb))

    return lsum


def loss_pytorch(w: torch.Tensor, xi: torch.Tensor, xibar: torch.Tensor) -> float:

    """
        Function to be optimized via gradient descent algorithm.
        It's built using only pytorch functions.
    """
    assert xi.shape == xibar.shape
    
    # Compute scalar prod
    tmp_xi = torch.sum(xi * w, 1)
    tmp_xibar = torch.sum(xibar * w, 1)

    # Apply ReLU
    prod = torch.relu(-1. * tmp_xi * tmp_xibar)

    # Compute loss
    lsum = prod.sum()

    return lsum


def compute_feasible_num_constraint(
    cost_function: Callable[[np.array,np.array,np.array], float],
    N: int,
    P: int,
    rho: float,
    x0: float,
    wnorm: float,
    n_seeds: int
) -> int:

    """
        Compute the minimization via root finding algorithm for n_seeds different sets of random doublets
        and return the number of corrected solutions.
        Each solution satiesfies the constraint ||w||^2 >= wnorm.
    """

    n_feasible = 0

    for seed in range(n_seeds):
        v = rand_couples(N, P, rho, seed)
        xi, xibar = np.array(v[0]), np.array(v[1])

        for i in range(2, 2*P, 2):
            xi = np.vstack((xi, v[i]))
            xibar = np.vstack((xibar, v[i+1]))

        # Search feasible solutions (respecting given constraints)
        constr = {
            'type': 'ineq',
            'fun': lambda w: w@w - wnorm
        }
        res = minimize(cost_function, x0=x0*np.ones(N), args=(xi,xibar), constraints=(constr))

        # Solution array
        w = res.x

        # Check proposed solution
        check_constraints = [(w@c) * (w@cb) > 0. for c, cb in zip(xi, xibar)]

        # Count realizations
        if np.all(check_constraints):
            n_feasible += 1

    return n_feasible


def compute_feasible_num_pytorch(
        cost_function: Callable[[torch.Tensor,torch.Tensor,torch.Tensor], float],
        N: int,
        P: int,
        rho: float,
        x0: float,
        n_epochs: int,
        lr: float,
        n_seeds: int
) -> int:

    """
        Compute the minimization via gradient descent algorithm for n_seeds different sets of random doublets
        and return the number of corrected solutions.
        The starting point, number of epochs and learning rate are passed as parameters, but the loss function (L1)
        and the optimizer (Adam) are fixed.
    """

    n_feasible = 0

    for seed in range(n_seeds):
        v = rand_couples(N, P, rho, seed)
        xi, xibar = np.array(v[0]), np.array(v[1])

        for i in range(2, 2*P, 2):
            xi = np.vstack((xi, v[i]))
            xibar = np.vstack((xibar, v[i+1]))

        xi_tensor = torch.as_tensor(xi)
        xibar_tensor = torch.as_tensor(xibar)

        # Set up framework
        w = torch.full((N,), x0, dtype=torch.float, requires_grad=True)
        y = torch.reshape(cost_function(w, xi_tensor, xibar_tensor), shape=(1,)) # Avoid warning on size
        y_target = torch.tensor([0.], dtype=torch.float)

        optimizer = torch.optim.Adam([w], lr=lr)
        loss_function = torch.nn.L1Loss()

        # Run optimization
        for _ in range(n_epochs):
            optimizer.zero_grad()
            y = torch.reshape(cost_function(w, xi_tensor, xibar_tensor), shape=(1,))
            loss = loss_function(y, y_target)
            loss.backward()
            optimizer.step()

        # Check if solution respects constraints
        w_appo = w.detach().numpy()
        check_constraints = [(w_appo@c) * (w_appo@cb) >= 0. for c, cb in zip(xi, xibar)]
        null_array = np.all(w_appo == 0)

        if np.any(np.isnan(check_constraints)):
            print("NaN found in check_constraints")

        if np.all(check_constraints) and not null_array:
            n_feasible += 1

    return n_feasible