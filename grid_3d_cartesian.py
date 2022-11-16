import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from time import time

def grid_to_vec(grid):
    return grid.flatten()

def vec_to_grid(gv, Nx, Ny, Nz):
    return gv.reshape(Nx, Ny, Nz)

def build_diffusion_matrix(Nx, Ny, Nz, dt, alpha_x = 1, alpha_y = 1, alpha_z = 1):

    dx, dy, dz = 1/Nx, 1/Ny, 1/Nz
    # alpha_x, alpha_y, alpha_z = 1, 1, 1

    def at(i, j, k):
        # return i + j*Nx + k*Nx*Ny
        return k + j*Nz + i*Ny*Nz

    sys_size = Nx*Ny*Nz
    A = sp.lil_matrix((sys_size, sys_size))

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                A[at(i, j, k), at(i, j, k)] = -2 * (alpha_x/dx**2 + alpha_y/dy**2 + alpha_z/dz**2)
                if i > 0:
                    A[at(i, j, k), at(i-1, j, k)] = alpha_x/dx**2
                if j > 0:
                    A[at(i, j, k), at(i, j-1, k)] = alpha_y/dy**2
                if k > 0:
                    A[at(i, j, k), at(i, j, k-1)] = alpha_z/dz**2
                if i < Nx-1:
                    A[at(i, j, k), at(i+1, j, k)] = alpha_x/dx**2
                if j < Ny-1:
                    A[at(i, j, k), at(i, j+1, k)] = alpha_y/dy**2
                if k < Nz-1:
                    A[at(i, j, k), at(i, j, k+1)] = alpha_z/dz**2

    for i in range(Nx):
        for j in range(Ny):
            A[at(i, j, 0), :] = 0
            A[:, at(i, j, 0)] = 0
            A[at(i, j, 0), at(i, j, 1)] = alpha_z / dz**2
            A[at(i, j, Nz-1), :] = 0
            A[:, at(i, j, Nz-1)] = 0
            A[at(i, j, Nz-1), at(i, j, Nz-2)] = alpha_z / dz**2

    A = A.tocsc()

    I = sp.eye(Nx*Ny*Nz, format="csc")

    CN_left =  (I - dt * A / 2)
    CN_right = (I + dt * A / 2)

    return CN_left, CN_right, at

def update_diffusion(left_scheme_matrix, right_scheme_matrix, forcing_vector, grid_vector, index_fnc):
    """Performs one step of diffusion using the precomputed matrix for the scheme.
    The scheme: L @ u^{n+1} = R @ u^{n} + f
    Input:
        left_scheme_matrix: Left matrix in the numerical scheme
        right_scheme_matrix: Right matrix in the numerical scheme
        forcing_vector: RHS from the discretized PDE 
        grid_vector: vector containing the grid to find the numerical solution on
        index_fnc: function for correctly indexing the grid_vector
    Returns:
        new_grid_vector: updated grid vector after one iteration
    """

    tmp = right_scheme_matrix @ grid_vector
    new_gv = spla.cg(left_scheme_matrix, tmp + forcing_vector)[0]

    return new_gv

def iterate_system(Nx, Ny, Nz, dt, gv, n_iter, alpha_x, alpha_y, alpha_z):
    t0 = time()
    CN_left, CN_right, index_fnc = build_diffusion_matrix(Nx, Ny, Nz, dt, 
        alpha_x=alpha_x, alpha_y=alpha_y, alpha_z=alpha_z
    )
    t1 = time()

    t2 = time()
    gvs = np.zeros((n_iter+1, gv.shape[0]))
    gvs[0, :] = gv
    t3 = time()
    
    for i in range(1, n_iter+1):
        gvs[i, :] = update_diffusion(CN_left, CN_right, 0, gvs[i-1, :], index_fnc)
    
    t4 = time()

    print(f"Building matrix: {t1-t0:.3f} sec.")
    print(f"Initializing problem: {t3-t2:.3f} sec.")
    print(f"Interation: {t4-t3:.3f} sec.")

    return gvs

if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    Nx, Ny, Nz = 16, 16, 16
    # Nx, Ny, Nz = 4, 4, 4
    g = np.zeros((Nx, Ny, Nz))

    # g[Nx//2, Ny//2, Nz//2] = 10
    g[Nx//2, Ny//2, 1] = 1
    g[:, :, 0] = 1
    g[:, :, -1] = 1
    mass_correction = 2 * np.sum(g[:, :, 0])#+ np.sum(g[:, : -1])
    gv = grid_to_vec(g)

    dt = 1e-4

    n_iter = 20
    gvs = iterate_system(
        Nx, Ny, Nz, dt, gv, n_iter,
        alpha_x = 1, alpha_y = 1, alpha_z = 100
    )


    fig = plt.figure()
    axs = [fig.add_subplot(2, 8, i+1) for i in range(16)]

    def animate(i):
        g = vec_to_grid(gvs[i, :], Nx, Ny, Nz)

        for j, ax in enumerate(axs):
            ax.clear()
            ax.imshow(g[:, :, j], vmin=0, vmax=.01)
            
            
        axs[0].set_title(f"t = {i * dt:.4f}")
        return axs

    anim = FuncAnimation(fig, animate, frames=n_iter, blit=True, interval=10)

    masses = [np.sum(gvs[i, :]) - mass_correction for i in range(n_iter)]
    print(masses)

    plt.show()