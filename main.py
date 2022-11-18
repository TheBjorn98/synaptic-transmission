import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from reaction import update_reaction, make_reaction_ode
from grid_3d_cartesian import update_diffusion, iterate_system_bw_euler


if __name__ == "__main__":
    ### Setup for the solver
    N_space = 25
    dx, dy, dz = 1/N_space, 1/N_space, 1/N_space
    
    ### Setup for the physical system
    Lx, Ly, Lz = .22e-6, .22e-6, 15e-9
    A_terminal = np.pi * Lx**2
    D = 8e-7
    N0 = 5000
    rho_R0 = 1000 / 1e-12
    R0 = A_terminal * rho_R0
    ax, ay, az = D * dx**2 / Lx**2, D * dy**2 / Ly**2, D * dz**2 / Lz**2
    ax, ay, az = ax/az, ay/az, az/az
    alphas = (ax, ay, az)

    ### Time parameters for the simulation
    N_time = 3000
    T = Lz / D
    Tf = 100*T
    dt = Tf / N_time
    N_time = 2000
    
    ### Setup for the reaction
    k_on = 4e6
    k_off = 5
    pi_1 = k_on * T * R0
    pi_2 = k_off * R0 * T / N0
    reaction = make_reaction_ode(pi_1, pi_2)
    # reaction = make_reaction_ode(1, 1)

    ### Initial conditions
    g = np.zeros((N_space, N_space, N_space))
    g[N_space//2, N_space//2, 1] = 1
    n_init = g.reshape((N_space**3))
    r_init = np.ones((N_space**2)) * R0 / N0 / N_space**2
    c_init = np.zeros((N_space**2))

    ### Printing the state of the system
    print(f"""
    System settings:
    ----------------
        N0 = {N0}, R0 = {R0} => Ratio: {R0 / N0 * 100:.2f}%

        Lx = {Lx:.3e}, Ly = {Ly:.3e}, Lz = {Lz:.3e}
        dx = {dx:.3e}, dy = {dy:.3e}, dz = {dz:.3e}
        ax = {ax:.3e}, ay = {ay:.3e}, az = {az:.3e}

        T =  {T:.3e}   Tf = {Tf}
        dt = {dt:.3e}

        k_on = {k_on:.3e}, k_off = {k_off:.3e}
        pi_1 = {pi_1:.3e}, pi_2 = {pi_2:.3e}
    """)

    # gvs = iterate_system_cn(N_space, N_time, dt, gv, alphas)
    gvs, rs, cs = iterate_system_bw_euler(N_space, N_time, dt, (n_init, r_init, c_init), alphas, reaction)


    fig = plt.figure()
    axs = [fig.add_subplot(1, 9, i+1) for i in range(9)]
    # plot_loc = np.arange(1, N_space, N_space//8)
    plot_loc = [1 + i*N_space//8 for i in range(8)]
    plot_loc.append(N_space-2)
    print(plot_loc)

    def animate(i):
        g = gvs[i, :].reshape((N_space, N_space, N_space))

        for j, ax in enumerate(axs):
            ax.clear()
            ax.imshow(g[:, :, plot_loc[j]], vmin=0, vmax=R0/N0)
            
            
        axs[0].set_title(f"t = {i * dt:.4f}")
        return axs

    anim = FuncAnimation(fig, animate, frames=N_time, blit=True, interval=10)

    plt.show()

    masses = [np.sum(gvs[i, :]) for i in range(N_time)]
    plt.plot(masses)
    plt.axhline(1, linestyle="--", color="k", linewidth=1)
    plt.show()

    rvals = np.array([np.sum(rs[i, :]) for i in range(N_time)])
    cvals = np.array([np.sum(cs[i, :]) for i in range(N_time)])
    nvals = np.array([np.sum(gvs[i, N_space**3 - N_space**2:]) for i in range(N_time)])

    plt.plot(rvals)
    plt.plot(cvals)
    plt.plot(nvals)
    plt.plot(rvals + cvals)
    plt.show()

