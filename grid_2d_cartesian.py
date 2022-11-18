import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from time import time

def build_diffusion_matrix(Nx, Ny, dt, alpha_x = 1, alpha_y = 1):
    dx, dy = 1/Nx, 1/Ny

    sys_size = Nx * Ny
    A = sp.lil_matrix((sys_size, sys_size))
    print(A.shape)

    A.setdiag(-2*((alpha_x / dx) + (alpha_y / dy)), k=0)
    A.setdiag(alpha_x / dx, k=1)
    A.setdiag(alpha_x / dx, k=-1)
    A.setdiag(alpha_y / dy, k=Nx)
    A.setdiag(alpha_y / dy, k=-Nx)

    zero_row = sp.lil_matrix(sys_size)

    # ???



    return A.tocsc()

if __name__ == "__main__":
    A = build_diffusion_matrix(8, 8, .1)
    img = plt.imshow(A.todense())
    plt.colorbar(img)
    plt.show()


