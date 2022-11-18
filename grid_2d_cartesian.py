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

    def at(i, j):
        return i + Nx*j
        # return j + Nx*i

    # Setting the 5-point stencil
    for i in range(Nx):
        for j in range(Ny):
            A[at(i,j), at(i, j)] = -2 * ((alpha_x / dx**2) + (alpha_y / dy**2))
            if i > 0:
                A[at(i,j), at(i-1, j)] = alpha_x / dx**2
            if j > 0:
                A[at(i,j), at(i, j-1)] = alpha_x / dx**2
            if i < Nx-1:
                A[at(i,j), at(i+1, j)] = alpha_y / dy**2
            if j < Ny-1:
                A[at(i,j), at(i, j+1)] = alpha_y / dy**2

    # No-flux in y-direction
    for i in range(Nx):
        A[at(i, 0), :] = 0
        A[:, at(i, 0)] = 0
        A[at(i, 0), at(i, 1)] = alpha_y / dy**2
        A[at(i, Ny-1), :] = 0
        A[:, at(i, Ny-1)] = 0
        A[at(i, Ny-1), at(i, Ny-2)] = alpha_y / dy**2

    # No-flux in x-direction
    for j in range(Ny):
        A[at(0, j), :] = 0
        A[:, at(0, j)] = 0
        A[at(0, j), at(1, j)] = alpha_x / dx**2
        A[at(Nx-1 ,j), :] = 0
        A[:, at(Nx-1 ,j)] = 0
        A[at(Nx-1, j), at(Nx-2, j)] = alpha_x / dx**2

    A[at(0, 0), :] = 0
    A[at(Nx-1, 0), :] = 0
    A[at(0, Ny-1), :] = 0
    A[at(Ny-1, Ny-1), :] = 0
    A[at(0, 0), at(0, 0)] = 1
    A[at(Nx-1, 0), at(Nx-1, 0)] = 1
    A[at(0, Ny-1), at(0, Ny-1)] = 1
    A[at(Ny-1, Ny-1), at(Ny-1, Ny-1)] = 1

    return A.tocsc()



if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    # A = build_diffusion_matrix(8, 8, .001)
    # print(np.linalg.det(A.todense()))
    # img = plt.imshow(A.todense())
    # plt.colorbar(img)
    # plt.show()

    Nx, Ny = 32, 32
    # Nx, Ny = 8, 8
    dt = 1e-5
    timesteps = 1000

    # gv = np.zeros(Nx*Ny)
    # gv[Nx*Ny//2] = 100
    gv = np.zeros((Nx, Ny))
    gv[Nx//2, Ny//2] = 1
    gv[:, 0] = 1
    gv[:, -1] = 1
    gv[0, :] = 1
    gv[-1, :] = 1
    gv = gv.flatten()
    A = build_diffusion_matrix(Nx, Ny, dt)
    I = sp.eye(Nx*Ny)

    img = plt.imshow((dt * A).todense() != 0)
    plt.colorbar(img)
    plt.show()

    CN_left = (I - dt * A / 2)
    CN_right = (I + dt * A / 2)
    bw_euler = (I - dt * A)

    # img = plt.imshow(bw_euler.todense())
    # plt.colorbar(img)
    # plt.show()

    gvs = np.zeros((timesteps, Nx*Ny))
    gvs[0, :] = gv
    print(np.max(gvs))

    for i in range(1, timesteps):
        tmp = CN_right @ gvs[i-1, :]
        gvs[i, :] = spla.spsolve(CN_left, tmp)
        # gvs[i, :] = spla.spsolve(bw_euler, gvs[i-1, :])

    rng = np.linspace(0, 1, num=Nx)
    xx, yy = np.meshgrid(rng, rng)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def animate(i):
        ax.clear()
        ax.set_title(f"t = {i * dt:3f}")
        # ax.plot_trisurf(rng, rng, gvs[i, :])
        ax.imshow(gvs[i, :].reshape((Nx, Ny)))

    maxvals = []
    for i in range(timesteps):
        maxvals.append(np.max(gvs[i, :]))

    print(maxvals)
    
    anim = FuncAnimation(fig, animate, timesteps, interval=10)
    plt.show()




