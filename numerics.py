import numpy as np
from scipy.integrate import RK45
import scipy.sparse as sp
from scipy.sparse import linalg as spla

def grid_to_vector(grid):
    return grid.flatten()

def vector_to_grid(gridvec, Nr, Nth, Nz):
    return gridvec.reshape((Nr, Nth, Nz))

def RK4_step(f, t_i, w_i, h):
    ''' 
    One step using Runge_kutta
    Intput: 
        w_i: [angle theta, frequency ohmega], 
        t_i: time
        h: step length
        f: function to be solved
    Return: next function value
    '''
    k1 = f(t_i,w_i)
    k2 = f(t_i + h/2, w_i + h*k1/2)
    k3 = f(t_i + h/2, w_i + h*k2/2)
    k4 = f(t_i+ h, w_i + h*k3)

    return t_i + h,  w_i + h/6 *(k1+ 2*k2 + 2*k3 + k4)

def reaction_ode(s, k1=1, k2=1):
    
    Nt = -k1 * s[0] * s[1] + k2 * s[2] 

    return np.array([Nt,Nt,-Nt])

def update_reaction_state(grid_vec, s0, dt):
    
    
    """
    Need to implement code to get N by summing over the reaction area in grid_vec
    """

    #RK4 step:
    k1 = reaction_ode(s0)
    k2 = reaction_ode(s0 + dt*k1/2)
    k3 = reaction_ode(s0 + dt*k2/2)
    k4 = reaction_ode(s0 + dt*k3)

    return s0 + dt/6 *(k1+ 2*k2 + 2*k3 + k4)


def setup_system_matrix(Nr, Nth, Nz, dt):
    dr, dth, dz = 1/Nr, 1/Nth, 1/Nz
    print(f"{dr = }, {dth = }, {dz = }")

    sys_size = Nr * Nth * Nz

    center_diag = np.ones(sys_size) - (dt * ((1/dz**2) + (1/dth**2) + (1/dr**2)))

    north_diag = (dt / 4 / dr) * np.ones(sys_size-1)
    south_diag = (dt / 4 / dr) * np.ones(sys_size-1)

    west_diag = ((dt / dz**2) / (2 * (((np.arange(sys_size - Nr) % Nr) + 1) * dr)**2))
    east_diag = ((dt / dz**2) / (2 * (((np.arange(sys_size - Nr) % Nr) + 1) * dr)**2))
    
    up_diag =   (dt / dz**2) * np.ones(sys_size - Nr*Nth) / 2
    down_diag = (dt / dz**2) * np.ones(sys_size - Nr*Nth) / 2

    print(f"""
    Diagonal lengths:
        center: {center_diag.shape}
        north : {north_diag.shape}
        south : {south_diag.shape}
        east  : {east_diag.shape}
        west  : {west_diag.shape}
        up    : {up_diag.shape}
        down  : {down_diag.shape}
    """
    )

    # print(center_diag)

    return sp.diags(
        [down_diag, west_diag, north_diag, center_diag, south_diag, east_diag, up_diag],
        [-Nr*Nth, -Nr, -1, 0, 1, Nr, Nr*Nth]
    )



def update_diffusion(system_matrix, grid_vector, sigma=0):

    pass

def update_bcs(system_matrix, grid_vector):
    pass

def update_reaction(s, method=RK45):
    

    pass

def store_results(grid):
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Nr, Nth, Nz = 8, 8, 8

    # grid = np.zeros((Nr, Nth, Nz))
    # grid[0, :, :] = 1

    A = setup_system_matrix(Nr, Nth, Nz, 1e-3)

    img = plt.imshow(A.todense())
    plt.colorbar(img)
    plt.show()
    
    # grid_vec = grid_to_vector(grid)

    # fig = plt.figure(figsize=(12,9))
    # axs = [fig.add_subplot(2, 4, i+1) for i in range(8)]

    # for i in range(Nz):
    #     axs[i].imshow(grid[:, :, i])
    #     axs[i].set_title(f"Depth: z={i+1}")
    #     axs[i].set_ylabel("r")
    #     axs[i].set_xlabel("theta")

    # plt.show()


    
    