import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from time import time

def grid_to_vec(grid):
    return grid.flatten()

def vec_to_grid(gv, Nx, Ny, Nz):
    return gv.reshape(Nx, Ny, Nz)

def build_diffusion_matrix(N_space, alphas):

    Nx, Ny, Nz = N_space, N_space, N_space
    dx, dy, dz = 1/Nx, 1/Ny, 1/Nz
    ax, ay, az = alphas

    def at(i, j, k):
        # return i + j*Nx + k*Nx*Ny
        return k + j*Nz + i*Ny*Nz

    sys_size = Nx*Ny*Nz
    A = sp.lil_matrix((sys_size, sys_size))

    t0 = time()
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                A[at(i, j, k), at(i, j, k)] = -2 * (ax/dx**2 + ay/dy**2 + az/dz**2)
                if i > 0:
                    A[at(i, j, k), at(i-1, j, k)] = ax/dx**2
                if j > 0:
                    A[at(i, j, k), at(i, j-1, k)] = ay/dy**2
                if k > 0:
                    A[at(i, j, k), at(i, j, k-1)] = az/dz**2
                if i < Nx-1:
                    A[at(i, j, k), at(i+1, j, k)] = ax/dx**2
                if j < Ny-1:
                    A[at(i, j, k), at(i, j+1, k)] = ay/dy**2
                if k < Nz-1:
                    A[at(i, j, k), at(i, j, k+1)] = az/dz**2
    t1 = time()
    print(f"Setting diagonals took {t1-t0:.3f} seconds.")

    for i in range(Nx):
        for j in range(Ny):
            A[at(i, j, 0), :] = 0
            A[:, at(i, j, 0)] = 0
            A[at(i, j, 0), at(i, j, 1)] = az / dz**2
            A[at(i, j, Nz-1), :] = 0
            A[:, at(i, j, Nz-1)] = 0
            A[at(i, j, Nz-1), at(i, j, Nz-2)] = az / dz**2
    t2 = time()
    print(f"Setting BCs took {t2-t1:.3f} seconds.")

    return A.tocsc()


def update_diffusion(left_scheme_mx, right_scheme_mx , f_vec, grid_vector):
    """Performs one step of diffusion using the precomputed matrix for the scheme.
    The scheme: L @ u^{n+1} = R @ u^{n} + f
    Input:
        left_scheme_mx: Left matrix in the numerical scheme
        right_scheme_mx: Right matrix in the numerical scheme
        f_vec: RHS from the discretized PDE 
        grid_vector: vector containing the grid to find the numerical solution on
    Returns:
        new_gv: updated grid vector after one iteration
    """

    if left_scheme_mx is None:
        return right_scheme_mx @ grid_vector
    elif right_scheme_mx is None:
        return spla.cg(left_scheme_mx, grid_vector)[0]
    else:
        tmp = right_scheme_mx @ grid_vector
        return spla.cg(left_scheme_mx, tmp)[0]

def iterate_system_cn(N_space, N_time, dt, initial_gv, alphas):
    t0 = time()
    A = build_diffusion_matrix(N_space, alphas)
    I = sp.eye(N_space**3)
    CN_left = (I - dt * A / 2)
    CN_right = (I + dt * A / 2)
    t1 = time()

    t2 = time()
    gvs = np.zeros((N_time, initial_gv.shape[0]))
    gvs[0, :] = initial_gv
    t3 = time()
    
    for i in range(1, N_time):
        gvs[i, :] = update_diffusion(CN_left, CN_right, 0, gvs[i-1, :])
    
    t4 = time()

    print(f"Building matrix: {t1-t0:.3f} sec.")
    print(f"Initializing problem: {t3-t2:.3f} sec.")
    print(f"Interation: {t4-t3:.3f} sec.")

    return gvs

def iterate_system_bw_euler(N_space, N_time, dt, initial_gv, alphas):
    t0 = time()
    A = build_diffusion_matrix(N_space, alphas)
    I = sp.eye(N_space**3)
    bw_euler = (I - dt * A)
    t1 = time()
    print(f"Building system took: {t1-t0:.3f} seconds.")

    gvs = np.zeros((N_time, N_space**3))
    gvs[0, :] = initial_gv

    t0 = time()
    for i in range(1, N_time):
        gvs[i, :] = update_diffusion(bw_euler, None, None, gvs[i-1, :])

    t1 = time()
    print(f"Iterating system took: {t1-t0:.3f} seconds.")
    return gvs


if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    # Nx, Ny, Nz = 16, 16, 16
    N_space = 25
    Nx, Ny, Nz = N_space, N_space, N_space
    dx, dy, dz = 1/Nx, 1/Ny, 1/Nz
    # Nx, Ny, Nz = 4, 4, 4
    g = np.zeros((Nx, Ny, Nz))

    # g[Nx//2, Ny//2, Nz//2] = 10
    g[Nx//2, Ny//2, 1] = 1
    g[:, :, 0] = 10
    g[:, :, -1] = 10
    mass_correction = 2 * np.sum(g[:, :, 0])#+ np.sum(g[:, : -1])
    gv = grid_to_vec(g)

    dt = 1e-4
    Lx, Ly, Lz = .22e-6, .22e-6, 15e-9
    A_terminal = np.pi * Lx**2
    D = 8e-7
    N0 = 5000
    rho_R0 = 1000 / 1e-12
    R0 = A_terminal * rho_R0
    ax, ay, az = D * dx**2 / Lx**2, D * dy**2 / Ly**2, D * dz**2 / Lz**2
    ax, ay, az = ax/az, ay/az, az/az
    alphas = (ax, ay, az)
    T = Lz / D
    Tf = 200*T
    N_time = 2000
    dt = Tf / N_time

    print(f"""
    System settings:
    ----------------
        N0 = {N0}, R0 = {R0} => Ratio: {R0 / N0 * 100:.2f}%

        Lx = {Lx:.3e}, Ly = {Ly:.3e}, Lz = {Lz:.3e}
        dx = {dx:.3e}, dy = {dy:.3e}, dz = {dz:.3e}
        ax = {ax:.3e}, ay = {ay:.3e}, az = {az:.3e}

        T =  {T:.3e}   Tf = {Tf}
        dt = {dt:.3e}
    """)

    gvs = iterate_system_cn(Nx, N_time, dt, gv, alphas)
    # gvs = iterate_system_bw_euler(Nx, N_time, dt, gv, alphas)


    fig = plt.figure()
    axs = [fig.add_subplot(1, 9, i+1) for i in range(9)]
    # plot_loc = np.arange(1, Nx, Nx//8)
    plot_loc = [1 + i*Nx//8 for i in range(8)]
    plot_loc.append(Nx-2)
    print(plot_loc)

    def animate(i):
        g = vec_to_grid(gvs[i, :], Nx, Ny, Nz)

        for j, ax in enumerate(axs):
            ax.clear()
            ax.imshow(g[:, :, plot_loc[j]], vmin=0, vmax=R0/N0)
            
            
        axs[0].set_title(f"t = {i * dt:.4f}")
        return axs

    anim = FuncAnimation(fig, animate, frames=N_time, blit=True, interval=10)

    masses = [np.sum(gvs[i, :]) - mass_correction for i in range(N_time)]
    print(masses[-1])

    plt.show()