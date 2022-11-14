import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def grid_to_vec(grid):
    return grid.flatten()

def vec_to_grid(gv, Nx, Ny, Nz):
    return gv.reshape(Nx, Ny, Nz)

def build_diffusion_matrix(Nx, Ny, Nz, dt):

    dx, dy, dz = 1/Nx, 1/Ny, 1/Nz
    alpha_x, alpha_y, alpha_z = 1, 1, 1

    def at(i, j, k):
        return i + j*Nx + k*Nx*Ny

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

    A *= dt / 2

    return A.tocsc()

def iterate_system(Nx, Ny, Nz, dt, gv, n_iter):
    A = build_diffusion_matrix(Nx, Ny, Nz, dt)
    I = sp.eye(Nx*Ny*Nz, format="csc")
    CN_left = (I - A)
    CN_right = (I + A)

    gvs = np.zeros((n_iter+1, gv.shape[0]))
    gvs[0, :] = gv

    for i in range(1, n_iter+1):
        tmp = CN_right @ gvs[i-1, :]
        gvs[i, :] = spla.spsolve(CN_left, tmp.T)

    return gvs

if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    Nx, Ny, Nz = 8, 8, 8
    g = np.zeros((Nx, Ny, Nz))

    g[Nx//2, Ny//2, Nz//2] = 10
    gv = grid_to_vec(g)

    dt = 1e-3

    n_iter = 10
    gvs = iterate_system(Nx, Ny, Nz, dt, gv, n_iter)

    fig = plt.figure()
    axs = [fig.add_subplot(1, 8, i+1) for i in range(8)]

    def animate(i):
        g = vec_to_grid(gvs[i, :], Nx, Ny, Nz)

        for i, ax in enumerate(axs):
            ax.clear()
            ax.imshow(g[:, :, i], vmin=0, vmax=1)
            

        axs[0].set_title(f"t = {i * dt:.4f}")
        return axs

    anim = FuncAnimation(fig, animate, frames=n_iter, blit=True, interval=10)

    plt.show()