import numpy as np
import matplotlib.pyplot as plt
from reaction import update_reaction
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from time import time


def build_diffusion_matrix(N, alphas):
    ax, ay, az = alphas
    A = sp.lil_matrix((N**3, N**3))

    def at(i, j, k):
        return i + j*N + k*N**2

    diag = [ax, ay, az, -2 * (ax + ay + az), az, ay, ax]
    offset = [-N**2, -N, -1, 0, 1, N, N**2]

    for (d, o) in zip(diag, offset):
        A.setdiag(d, k=o)
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i == 0 and k > 0:  # North
                    A[at(i, j, k), at(i-1, j, k)] = 0
                if i == N-1 and k < N-2:  # South
                    A[at(i, j, k), at(i+1, j, k)] = 0
                if j == 0 and k > 0:  # West
                    A[at(i, j, k), at(i, j-1, k)] = 0
                if j == N-1 and k < N-2:  # East
                    A[at(i, j, k), at(i, j+1, k)] = 0
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i == 0:  # Neumann, z
                    A[at(i, j, k), :] = 0
                    A[at(i, j, k), at(i, j, k)] =  -az
                    A[at(i, j, k), at(i+1, j, k)] = az
                # if j == 0:  # Neumann, y
                #     A[at(i, j, k), :] = 0
                #     A[at(i, j, k), at(i, j, k)] =  -ay
                #     A[at(i, j, k), at(i, j+1, k)] = ay
                # if k == 0:  # Neumann, x
                #     A[at(i, j, k), :] = 0
                #     A[at(i, j, k), at(i, j, k)] =  -ax
                #     A[at(i, j, k), at(i, j, k+1)] = ax
                if i == N-1:  # Neumann, z
                    A[at(i, j, k), :] = 0
                    A[at(i, j, k), at(i, j, k)] =  -az
                    A[at(i, j, k), at(i-1, j, k)] = az
                # if j == N-1:  # Neumann, y
                #     A[at(i, j, k), :] = 0
                #     A[at(i, j, k), at(i, j, k)] =  -ay
                #     A[at(i, j, k), at(i, j-1, k)] = ay
                # if k == N-1:  # Neumann, x
                #     A[at(i, j, k), :] = 0
                #     A[at(i, j, k), at(i, j, k)] =  -ax
                #     A[at(i, j, k), at(i, j, k-1)] = ax

    return (A.tocsc() * N**2).T


def update_diffusion(left_scheme_mx, right_scheme_mx, grid_vector):
    """Performs one step of diffusion using the precomputed matrix for the scheme.
    The scheme: L @ u^{n+1} = R @ u^{n} + f
    Input:
        left_scheme_mx: Left matrix in the numerical scheme
        right_scheme_mx: Right matrix in the numerical scheme
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


### Deprecated, does not include reaction term
# def iterate_system_cn(N_space, N_time, dt, initial_gv, alphas):
#     t0 = time()
#     A = build_diffusion_matrix(N_space, alphas)
#     I = sp.eye(N_space**3)
#     CN_left = (I - dt * A / 2)
#     CN_right = (I + dt * A / 2)
#     t1 = time()

#     t2 = time()
#     gvs = np.zeros((N_time, initial_gv.shape[0]))
#     gvs[0, :] = initial_gv
#     t3 = time()
    
#     for i in range(1, N_time):
#         gvs[i, :] = update_diffusion(CN_left, CN_right, gvs[i-1, :])
    
#     t4 = time()

#     print(f"Building matrix: {t1-t0:.3f} sec.")
#     print(f"Initializing problem: {t3-t2:.3f} sec.")
#     print(f"Interation: {t4-t3:.3f} sec.")

#     return gvs

def iterate_system_bw_euler(N_space, N_time, dt, initial_state, alphas, reaction_ode):
    n_init, r_init, c_init = initial_state
    t0 = time()
    A = build_diffusion_matrix(N_space, alphas)
    I = sp.eye(N_space**3)
    bw_euler = (I - dt * A)
    t1 = time()
    print(f"Building system took: {t1-t0:.3f} seconds.")

    n_vecs = np.zeros((N_time, N_space**3))
    r_vecs = np.zeros((N_time, N_space**2))
    c_vecs = np.zeros((N_time, N_space**2))
    n_vecs[0, :] = n_init
    r_vecs[0, :] = r_init
    c_vecs[0, :] = c_init

    t0 = time()
    for i in range(1, N_time):
        temp_state = update_diffusion(bw_euler, None, n_vecs[i-1, :])

        n_terminal, r_new, c_new = update_reaction(
            temp_state[N_space**3-N_space**2:], r_vecs[i-1, :], c_vecs[i-1, :],
            dt, reaction_ode
        )

        temp_state[N_space**3 - N_space**2:] = n_terminal[:]
        n_vecs[i, :] = temp_state
        r_vecs[i, :] = r_new
        c_vecs[i, :] = c_new

    t1 = time()
    print(f"Iterating system took: {t1-t0:.3f} seconds.")
    return n_vecs, r_vecs, c_vecs


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
    # g[:, :, 0] = 10
    # g[:, :, -1] = 10
    mass_correction = 2 * np.sum(g[:, :, 0])#+ np.sum(g[:, : -1])
    gv = g.reshape((Nx*Ny*Nz))

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
    Tf = 100*T
    N_time = 3000
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

    # gvs = iterate_system_cn(Nx, N_time, dt, gv, alphas)
    gvs = iterate_system_bw_euler(Nx, N_time, dt, gv, alphas)


    fig = plt.figure()
    axs = [fig.add_subplot(1, 9, i+1) for i in range(9)]
    # plot_loc = np.arange(1, Nx, Nx//8)
    plot_loc = [1 + i*Nx//8 for i in range(8)]
    plot_loc.append(Nx-2)
    print(plot_loc)

    def animate(i):
        g = gvs[i, :].reshape((Nx, Ny, Nz))

        for j, ax in enumerate(axs):
            ax.clear()
            ax.imshow(g[:, :, plot_loc[j]], vmin=0, vmax=R0/N0)
            
            
        axs[0].set_title(f"t = {i * dt:.4f}")
        return axs

    anim = FuncAnimation(fig, animate, frames=N_time, blit=True, interval=10)

    masses = [np.sum(gvs[i, :]) - mass_correction for i in range(N_time)]
    print(masses[-1])

    plt.show()