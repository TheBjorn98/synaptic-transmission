import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def setup_system_matrix():
    pass

def A(u):  # Deprecated
    Ni, Nj = u.shape
    # hi, hj = 1 / Ni, 1 / Nj
    i = np.arange(1, Ni)
    j = np.arange(1, Nj)
    c = np.ix_(i, j)
    n = np.ix_(i-1, j)
    s = np.ix_(i+1, j)
    e = np.ix_(i, j+1)
    w = np.ix_(i, j-1)

    return Ni * (u[n] + u[s] - 2 * u[c]) + Nj * (u[e] + u[w] - 2 * u[c])

def jacobi(u0, f, N_iter):  # Deprecated
    Ni, Nj = u0.shape
    hi, hj = 1 / Ni, 1 / Nj
    i = np.arange(1, Ni-1)
    j = np.arange(1, Nj-1)
    c = np.ix_(i, j)
    n = np.ix_(i-1, j)
    s = np.ix_(i+1, j)
    e = np.ix_(i, j+1)
    w = np.ix_(i, j-1)

    u = np.copy(u0)

    for k in range(N_iter):
        u[c] = 0.25 * (u[n] + u[s] + u[e] + u[w] + hi * hj * f[c])
    
    return u

# TODO Turn this into a matrix, not a function call
def stencil(us, i, dt, dr, dz, D):
    c = 1 - 2 * D * dt * (1/dr**2 + 1/dz**2)
    n = 2 * D * dt * (-1/(2 * i * dr**2) + 1/dr**2)
    s = 2 * D * dt * ( 1/(2 * i * dr**2) + 1/dr**2)
    w = 2 * D * dt / dz**2
    e = 2 * D * dt / dz**2

    return c*us[1, 1] + n*us[0, 1] + s*us[2, 1] + w*us[1, 0] + e*us[1, 2]

# TODO Construct a matrix to apply to a vector representing the domain?
def timestep_system(u, N_iter, dt, dr, dz, D):
    Ni, Nj = u.shape
    space_time = np.zeros((N_iter, Ni, Nj))

    #space_time[0, :, :] = u¨

    space_time[:, ]

    total_mass = np.sum(u)

    for k in range(1, N_iter):
        for i in range(1, Ni-1):
            for j in range(1, Nj-1):
                space_time[k, i, j] = stencil(
                    space_time[k-1, i-1:i+2, j-1:j+2],
                    i, dt, dr, dz, D
                )
        # space_time[k, :, 0] = space_time[0, :, 0]
        # space_time[k, :, -1] = space_time[0, :, -1]
        # space_time[k, 0, :] = space_time[0, 0, :]
        space_time[k, :, :] /= np.sum(space_time[k, :, :]) / total_mass

    
    return space_time



if __name__ == "__main__":
    r = 2
    z = 1

    dr = .05
    dz = .05
    N_r = int(r / dr)
    N_z = int(z / dz)

    Tf = 10
    dt = 1/100
    N_t = int(Tf / dt)
    # N_t = 2

    u0 = np.zeros((N_r, N_z))
    # u0[N_r//2, N_z//2] = 1
    u0[1, 1] = 1
    f = -1 * np.ones((N_r, N_z))

    # u = jacobi(u0, f, 1000)

    u = timestep_system(u0, N_t, dt, dr, dz, D=.05)

    # img = plt.imshow(u[-1, :, :])
    # plt.colorbar(img)
    # plt.show()

    plot_type = "anim"

    if plot_type == "anim":
        fig = plt.figure()
        ax = fig.add_subplot()

        def animate(i):
            ax.clear()
            ax.set_title(f"T = {dt * i:.2f}, Mass = {np.sum(u[i, :, :]):.2f}")
            ax.imshow(u[i, :, :].T)#, extent=[])

        ani = anim.FuncAnimation(fig, animate, N_t, interval=100, blit=False)
        plt.show()
    elif plot_type == "3":
        fig = plt.figure()
        axs = [fig.add_subplot(1, 3, i+1) for i in range(3)]

        for i in range(3):
            axs[i].imshow(u[i, :, :])
            axs[i].set_title(f"T = {i * dt}")

        plt.show()
    else:
        img = plt.imshow(u[-1, :, :])
        plt.colorbar(img)
        plt.show()




