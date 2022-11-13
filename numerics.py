import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla

def grid_to_vector(grid):
    return grid.flatten()

def vector_to_grid(gridvec, Nr, Nth, Nz):
    return gridvec.reshape((Nr, Nth, Nz))

def setup_system_matrix(Nr, Nth, Nz, dt):
    dr, dth, dz = 1/Nr, 1/Nth, 1/Nz
    print(f"{dr = }, {dth = }, {dz = }")

    sys_size = Nr * Nth * Nz
    # center: sys_size, n/s: sys_size-1
    # w/e: sys_size - Nr, u/d: sys_size - Nr*Nth

    # Setting up diagonals
    center_diag = np.ones(sys_size) - (dt * ((1/dz**2) + (1/dth**2) + (1/dr**2)))

    north_diag = (dt / 4 / dr) * np.ones(sys_size)
    south_diag = (dt / 4 / dr) * np.ones(sys_size)

    west_diag = ((dt / dz**2) / (2 * (((np.arange(sys_size) % Nr) + 1) * dr)**2))
    east_diag = ((dt / dz**2) / (2 * (((np.arange(sys_size) % Nr) + 1) * dr)**2))
    
    up_diag =   (dt / dz**2) * np.ones(sys_size) / 2
    down_diag = (dt / dz**2) * np.ones(sys_size) / 2

    center_grid = vector_to_grid(center_diag, Nr, Nth, Nz)
    north_grid = vector_to_grid(north_diag, Nr, Nth, Nz)
    south_grid = vector_to_grid(south_diag, Nr, Nth, Nz)
    west_grid = vector_to_grid(west_diag, Nr, Nth, Nz)
    east_grid = vector_to_grid(east_diag, Nr, Nth, Nz)
    up_grid = vector_to_grid(up_diag, Nr, Nth, Nz)
    down_grid = vector_to_grid(down_diag, Nr, Nth, Nz)

    # # Decoupling blocks and layers
    north_grid[0, :, :] = 0
    # for i in range(sys_size - 1):
        # if (i+1) % Nr == 0:
            # north_diag[i] = 0
            # south_diag[i] = 0
# 
    # for i in range(sys_size - Nr):
        # if (i+1) % Nr*Nth == 0:
            # west_diag[i] = 0
            # east_diag[i] = 0

    # Enforcing Neumann BCs at z=0 and z=L
    up_grid[:, :, 0] *= -(1 / (1/2 + np.sqrt(3)/3))
    down_grid[:, :, 0] *= -(1 / (1/2 + np.sqrt(3)/3))
    down_grid[:, :, 1] += (1 / (1/2 + np.sqrt(3)/3)) * (dt / dz**2)
    down_grid[:, :, 1] += (1 / (1/2 + np.sqrt(3)/3)) * (dt / dz**2)
    
    up_grid[:, :, -1] *= (1 / (1/2 + np.sqrt(3)/3))
    down_grid[:, :, -1] *= (1 / (1/2 + np.sqrt(3)/3))
    up_grid[:, :, -2] -= (1 / (1/2 + np.sqrt(3)/3)) * (dt / dz**2)
    down_grid[:, :, -2] -= (1 / (1/2 + np.sqrt(3)/3)) * (dt / dz**2)
    # up_diag[0:Nr*Nth] *= (1 / (1/2 + np.sqrt(3)/3))
    # down_diag[0:Nr*Nth] *= (1 / (1/2 + np.sqrt(3)/3))
    # up_diag[-1:-Nr*Nth] *= (1 / (1/2 + np.sqrt(3)/3))
    # down_diag[-1:-Nr*Nth] *= (1 / (1/2 + np.sqrt(3)/3))

    # Enforcing zeros at the edge of the cylinder (BC)
    # for i in range(Nz):
        # center_diag[(Nr-1):Nth()]

    # Enforce BC at the edge of the cylinder
    center_grid[Nr-1, :, :] = 1
    north_grid[Nr-1, :, :] = 0
    south_grid[Nr-1, :, :] = 0
    west_grid[Nr-1, :, :] = 0
    east_grid[Nr-1, :, :] = 0
    up_grid[Nr-1, :, :] = 0
    down_grid[Nr-1, :, :] = 0

    # center_grid[0, :, :] = 0
    # north_grid[0, :, :] = 0
    # south_grid[0, :, :] = 0
    # west_grid[0, :, :] = 0
    # east_grid[0, :, :] = 0
    # up_grid[0, :, :] = 0
    # down_grid[0, :, :] = 0

    center_diag = grid_to_vector(center_grid)
    north_diag = grid_to_vector(north_grid)[1:]
    south_diag = grid_to_vector(south_grid)[:-1]
    west_diag = grid_to_vector(west_grid)[Nr:]
    east_diag = grid_to_vector(east_grid)[:-Nr]
    up_diag = grid_to_vector(up_grid)[Nr*Nth:]
    down_diag = grid_to_vector(down_grid)[:-Nr*Nth]
    

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
    sys_mx_implicit = sp.diags(
            [-down_diag, -west_diag, -north_diag, -(-2 * np.ones(sys_size) - center_diag), -south_diag, -east_diag, -up_diag],
            [-Nr*Nth, -Nr, -1, 0, 1, Nr, Nr*Nth], format="csc"
    )
    sys_mx_explicit = sp.diags(
            [down_diag, west_diag, north_diag, center_diag, south_diag, east_diag, up_diag],
            [-Nr*Nth, -Nr, -1, 0, 1, Nr, Nr*Nth], format = "csc"
    )

    # Fixing r=0
    # sys_mx_explicit[]

    return sys_mx_implicit, sys_mx_explicit




def update_diffusion(system_matrix, grid_vector):
    pass

def update_bcs(system_matrix, grid_vector):
    pass

def update_reaction(system_matrix, grid_vector):
    pass

def store_results(grid):
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Nr, Nth, Nz = 4, 4, 4

    grid = np.zeros((Nr, Nth, Nz))
    grid[0, :, 0] = 1

    A_imp, A_exp = setup_system_matrix(Nr, Nth, Nz, .001)

    img = plt.imshow(A_imp.todense(), cmap="Reds")
    plt.colorbar(img)
    plt.show()    
    
    grid_vec = grid_to_vector(grid)
    zero_vec = np.copy(grid_vec)
    iter_one = A_exp @ grid_vec
    iter_one = spla.spsolve(A_imp, iter_one)
    iter_one = vector_to_grid(iter_one, Nr, Nth, Nz)

    fig = plt.figure(figsize=(12,9))
    axs = [fig.add_subplot(2, 4, i+1) for i in range(8)]

    for i in range(Nz):
        axs[i].imshow(grid[:, :, i])
        axs[i].set_title(f"Depth: z={i+1}")
        axs[i].set_ylabel("r")
        axs[i].set_xlabel("theta")
        axs[i+4].imshow(iter_one[:, :, i])
        axs[i+4].set_title(f"Depth: z={i+1}")

    plt.show()


    
    