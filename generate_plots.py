import numpy as np
from reaction import make_reaction_ode
import matplotlib.pyplot as plt
from grid_3d_cartesian import iterate_system_bw_euler

def setup_R0(radius, rho_R0):
    area = np.pi * radius**2
    return area / rho_R0

def setup_alphas(Lx, Ly, Lz, D, N_space):
    dp = 1 / N_space
    az = D * dp / Lz

    return (D * dp / Lx / az, D * dp / Ly / az, 1)

def setup_reaction(k_on, k_off, T, R0, N0, divide=True):
    pi1 = k_on * T * R0
    pi2 = (k_off * T * R0) / (N0 if divide else 1)
    return make_reaction_ode(pi1, pi2)

def setup_ic(N_space, R0, N0):
    g= np.zeros((N_space, N_space, N_space))
    g[N_space//2, N_space//2, 1] = 1
    n_init = g.reshape((N_space**3))
    r_init = np.ones((N_space**2)) * R0 / N0 / N_space**2
    c_init = np.zeros((N_space**2))

    return n_init, r_init, c_init

t_scale = 281

def plot_terminal_concentration(N, divide=True, save=False):
    N0 = 5000
    Lx = Ly = .22e-6
    R0 = setup_R0(Lx, 1e-15)
    Lz = 15e-9
    D = 8e-7
    alphas = ax, ay, az = setup_alphas(Lx, Ly, Lz, D, N)
    T = 1 / N
    Tf = 3
    N_time = 3000
    dt = Tf / N_time
    ### Setup for the reaction
    k_on = 4e6
    k_off = 5
    reaction = setup_reaction(k_on, k_off, T, R0, N0, divide=divide)
    ### Setup ICs
    i_state = setup_ic(N, R0, N0)
    gvs, rs, cs = iterate_system_bw_euler(N, N_time, dt, i_state, alphas, reaction)

    ### Plotting concentrations at the post-synaptic terminal
    ts = np.linspace(0, Tf, num=N_time) * t_scale  # 28.125 ns
    rvals = np.array([np.sum(rs[i, :]) for i in range(N_time)])
    cvals = np.array([np.sum(cs[i, :]) for i in range(N_time)])
    nvals = np.array([np.sum(gvs[i, N**3 - N**2:]) for i in range(N_time)])
    plt.figure(figsize=(4, 3))

    plt.plot(ts, rvals, label = "R")
    plt.plot(ts, cvals, label = "C")
    plt.plot(ts, nvals, label = "N")
    plt.plot(ts, rvals + cvals, label = "Balance: R+C", linestyle = "--", color="k", linewidth=1)
    plt.xlabel("Time, ns")
    plt.ylabel("Concentration, relative to N0")
    if divide:
        plt.title("Dividing by N0")
    else:
        plt.title("Not dividing by N0")
    plt.legend(loc=1)
    if save:
        plt.savefig(f"figures/terminal_concentration_{1 if divide else 0}.pdf")
    else:
        plt.show()

def plot_concentration_along_z(N, save=False):
    N0 = 5000
    Lx = Ly = .22e-6
    R0 = setup_R0(Lx, 1e-15)
    Lz = 15e-9
    D = 8e-7
    alphas = ax, ay, az = setup_alphas(Lx, Ly, Lz, D, N)
    T = 1 / N**2
    Tf = 1
    N_time = 1000
    dt = Tf / N_time
    ### Setup for the reaction
    k_on = 4e6
    k_off = 5
    reaction = setup_reaction(k_on, k_off, T, R0, N0)
    ### Setup ICs
    i_state = setup_ic(N, R0, N0)
    gvs, rs, cs = iterate_system_bw_euler(N, N_time, dt, i_state, alphas, reaction)
    ### Plotting z-directional distribution of N
    fig = plt.figure(figsize=(7, 3))
    zs = np.linspace(0, 15, num=N)
    z_dist = []
    for i in range(N_time):
        g = gvs[i, :].reshape((N, N, N))
        z_dist.append([np.sum(g[:, :, j]) for j in range(N)])
    ts = [10, 50, 100, 250, 500]
    for t in ts:
        plt.plot(zs, z_dist[t], label=f"t = {t_scale * t / N_time:.2f} ns")
    plt.axhline(R0/N0, linestyle="--", color="k", linewidth=1, label="R0/N0")
    plt.xlabel("Distance across the cleft, z, [nm]")
    plt.ylabel("Concentration of N, relative to N0")
    plt.legend()
    if save:
        plt.savefig("figures/conc_across_cleft.pdf")
    else:
        plt.show()

def plot_terminal_spread(N, save=False):
    N0 = 5000
    Lx = Ly = .22e-6
    R0 = setup_R0(Lx, 1e-15)
    Lz = 15e-9
    D = 8e-7
    alphas = ax, ay, az = setup_alphas(Lx, Ly, Lz, D, N)
    T = 1 / N**2
    Tf = 1
    N_time = 1000
    dt = Tf / N_time
    ### Setup for the reaction
    k_on = 4e6
    k_off = 5
    reaction = setup_reaction(k_on, k_off, T, R0, N0)
    ### Setup ICs
    i_state = setup_ic(N, R0, N0)
    gvs, rs, cs = iterate_system_bw_euler(N, N_time, dt, i_state, alphas, reaction)

    fig = plt.figure(figsize=(7, 3))
    axs = [fig.add_subplot(1, 4, i+1) for i in range(4)]
    ts = [100, 200, 300, 400]
    for (i, t) in enumerate(ts):
        axs[i].imshow(gvs[t, :].reshape((N, N, N))[:, :, -1], vmax=.0005, vmin=0)
        axs[i].set_title(f"Time: {t * t_scale / N_time:.2f} ns")

    if save:
        plt.savefig("figures/terminal_spread.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    save = False
    plot_concentration_along_z(25, save=save)
    plot_terminal_concentration(25, divide=False, save=save)
    plot_terminal_concentration(25, divide=True , save=save)
    plot_terminal_spread(25, save=save)
