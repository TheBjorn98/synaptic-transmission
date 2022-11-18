import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def build_diffusion_matrix(N_space, alphas):
    A = sp.lil_matrix((N_space**2, N_space**2))
    ax, ay = alphas

    def at(i, j):
        return i + N_space*j

    # diag = [1, 1, -2*(), 1, 1]
    diag = [ax, ay, -2 * (ax + ay), ay, ax]
    offset = [-N_space, -1, 0, 1, N_space]

    for (d, o) in zip(diag, offset):
        A.setdiag(d, k=o)

    for i in range(N_space):
        for j in range(N_space):
            if i == 0 and j > 0:
                A[at(i, j), at(i-1, j)] = 0
            if i == N_space-1 and j < N_space-2:
                A[at(i, j), at(i+1, j)] = 0

    for i in range(N_space):
        for j in range(N_space):
            # West boundary
            if i == 0:
                A[at(i, j), :] = 0
                A[at(i, j), at(i, j)] =  -ay
                A[at(i, j), at(i+1, j)] = ay
            # East boundary
            if i == N_space-1:
                A[at(i, j), :] = 0
                A[at(i, j), at(i, j)] =  -ay
                A[at(i, j), at(i-1, j)] = ay
            # North boundary
            if j == 0:
                A[at(i, j), :] = 0
                A[at(i, j), at(i, j)] =  -ax
                A[at(i, j), at(i, j+1)] = ax
            # South boundary
            if j == N_space-1:
                A[at(i, j), :] = 0
                A[at(i, j), at(i, j)] =  -ax
                A[at(i, j), at(i, j-1)] = ax

    return A.tocsc() * N_space**2


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
    gv[:, 0] = 1
    gv[:, -1] = 1
    gv[0, :] = 1
    gv[-1, :] = 1
    init_mass = np.sum(gv)
    gv[1, Ny//2] = 1
    gv = gv.flatten("F")
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
        ax.imshow(gvs[i, :].reshape((Nx, Ny))[1:-1, 1:-1])

    maxvals = []
    mass_sum = []
    for i in range(timesteps):
        maxvals.append(np.max(gvs[i, :]))
        mass_sum.append(np.sum(gvs[i, :]))


    print(f"Max value: {maxvals[-1]:.3f}")
    print(f"Balance:  {mass_sum[-1]-init_mass:.3f}")
    
    anim = FuncAnimation(fig, animate, timesteps, interval=10)
    plt.show()




