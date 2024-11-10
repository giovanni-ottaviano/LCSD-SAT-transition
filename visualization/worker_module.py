# Support module to enable multiprocessing in jupyter notebook

import numpy as np
from decimal import Decimal

# Functions for unstructure data (Cover's problem)
def C_cover_matrix_wm(n,p):

    '''Return number of admissible dichotomies C(n,p) for unstructured data (Cover) using recursion relation'''

    assert isinstance(n, int) and n > 0
    assert p > 0

    # Set the matrix
    grid = np.empty((n+1,p+1), dtype=Decimal)
    grid.fill(Decimal('0'))

    # Fill row n = 0
    for i in range(p+1):
        grid[0][i] = Decimal('0')

    # Fill column p = 0 (useless for this computation)
    for i in range(n+1):
        grid[i][0] = Decimal('0')

    # Fill p = 1 for all n
    for i in range(1,n+1):
        grid[i][1] = Decimal('2')

    # Fill n = 1 using recursione (with cut)
    for i in range(2,p+1):
        grid[1][i] = grid[1][i-1]

    # Fill the matrix column by column (using recursion)
    for P in range(2,p+1):
        for N in range(2,n+1):
            grid[N][P] = grid[N][P-1] + grid[N-1][P-1]


    return grid[n][p]

def c_cover_wm(n: int, p: int) -> float:
    
    '''Return the fraction of admissible dichotomies c(n,p) for fixed rho and k=2'''
    
    return float(Decimal(C_cover_matrix_wm(n,p)) / Decimal(2**p))

# Function for doublets
def psi2(rho: float) -> float:
    
    '''Symmetrized probability psi_2(rho)'''
    
    return (2./np.pi) * np.arctan(np.sqrt((1. + rho) / (1. - rho)))


def krondelta(i: int, j: int) -> int:
    
    '''Kronecker delta element i,j'''
    
    assert isinstance(i, int) and isinstance(j, int)
    
    return 1 if i == j else 0


def C_matrix_wm(n: int, p: int, rho: float) -> int:
    
    '''Return number of admissible dichotomies C(n,p) for fixed rho using a lookup table'''
    
    assert isinstance(n, int) and n > 0
    assert p > 0
    
    # Set table
    grid = np.zeros((n + 1, p + 1))
    grid.fill(np.nan)
    
    # Fill row n = 0
    for i in range(p + 1):
        grid[0][i] = 0
        
    # Fill column p = 0 (even though it's useless for this computation)
    for i in range(n + 1):
        grid[i][0] = 0
    
    # Fill p = 1 for all n
    for i in range(1, n + 1):
        grid[i][1] = 2*(1. - (1. - psi2(rho))*krondelta(i,1))
    
    # Fill n = 1 using recursione (with cut)
    for i in range(2, p + 1):
        grid[1][i] = psi2(rho)*grid[1][i-1]
        
    # Fill the matrix column by column (using recursion)
    for P in range(2, p + 1):
        for N in range(2, n + 1):
            # Consistency check
            if grid[N][P-1] == np.nan or grid[N-1][P-1] == np.nan or grid[N-2][P-1] == np.nan:
                print("WARNING: ", grid[N][P-1], grid[N-1][P-1], grid[N-2][P-1])
                
            grid[N][P] = psi2(rho)*grid[N][P-1] + grid[N-1][P-1] + (1 - psi2(rho))*grid[N-2][P-1]
            
    return grid[n][p]


def c_k2_wm(n: int, p: int, rho: float) -> float:
    
    '''Return the fraction of admissible dichotomies c(n,p) for fixed rho and k=2'''
    
    return float(Decimal(C_matrix_wm(n,p,rho)) / Decimal(2**p))