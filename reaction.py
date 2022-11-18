import numpy as np
import scipy.integrate as spint

def make_reaction_ode(k_left, k_right):
    def reaction_ode(t, state):
        ds = -k_left * state[0] * state[1] + k_right * state[2]
        return np.array([ds, ds, -ds])
    
    return reaction_ode


def RK4_step(f, t, state, dt):
    k1 = f(t         , state)
    k2 = f(t + dt / 2, state + dt * k1 / 2)
    k3 = f(t + dt / 2, state + dt * k2 / 2)
    k4 = f(t + dt    , state + dt * k3)

    return dt * (k1+ 2 * k2 + 2 * k3 + k4) / 6


def update_reaction(n_vec, r_vec, c_vec, dt, reaction_ode):
    state_vector = np.vstack((n_vec, r_vec, c_vec))
    diff = RK4_step(reaction_ode, 0, state_vector, dt)
    sv_updated = state_vector + diff
    n_vec_n, r_vec_n, c_vec_n = sv_updated[0, :], sv_updated[1, :], sv_updated[2, :]

    # Control mass balance
    eps = (r_vec_n + c_vec_n - r_vec - c_vec) / 2
    n_vec_n += eps

    return n_vec_n, r_vec_n, c_vec_n


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    N = 16
    n_vec = np.random.random(N**2)
    r_vec = np.random.random(N**2)
    c_vec = np.random.random(N**2)

    reaction_ode = make_reaction_ode(1, 1)

    N_time = 1000
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    nvecs, rvecs, cvecs = np.zeros((N_time, N**2)), np.zeros((N_time, N**2)), np.zeros((N_time, N**2))
    nvecs[0, :] = n_vec
    rvecs[0, :] = r_vec
    cvecs[0, :] = c_vec

    for i in range(1, N_time):
        a, b, c = update_reaction(nvecs[i-1, :], rvecs[i-1, :], cvecs[i-1, :], 1e-2, reaction_ode)
        nvecs[i, :] = a
        rvecs[i, :] = b
        cvecs[i, :] = c

    def animate(i):
        ax.clear()
        ax.plot(nvecs[i, :])
        ax.plot(rvecs[i, :])
        ax.plot(cvecs[i, :])
        ax.set_title(f"t = {i}")

    anim = FuncAnimation(fig, animate, N_time, interval=50)
    plt.show()

