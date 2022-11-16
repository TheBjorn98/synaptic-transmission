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
    # north_grid[0, :, :] = 0
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
    # up_grid[:, :, 0] *= -(1 / (1/2 + np.sqrt(3)/3))
    # down_grid[:, :, 0] *= -(1 / (1/2 + np.sqrt(3)/3))
    # down_grid[:, :, 1] += (1 / (1/2 + np.sqrt(3)/3)) * (dt / dz**2)
    # down_grid[:, :, 1] += (1 / (1/2 + np.sqrt(3)/3)) * (dt / dz**2)
    
    # up_grid[:, :, -1] *= (1 / (1/2 + np.sqrt(3)/3))
    # down_grid[:, :, -1] *= (1 / (1/2 + np.sqrt(3)/3))
    # up_grid[:, :, -2] -= (1 / (1/2 + np.sqrt(3)/3)) * (dt / dz**2)
    # down_grid[:, :, -2] -= (1 / (1/2 + np.sqrt(3)/3)) * (dt / dz**2)
    # up_diag[0:Nr*Nth] *= (1 / (1/2 + np.sqrt(3)/3))
    # down_diag[0:Nr*Nth] *= (1 / (1/2 + np.sqrt(3)/3))
    # up_diag[-1:-Nr*Nth] *= (1 / (1/2 + np.sqrt(3)/3))
    # down_diag[-1:-Nr*Nth] *= (1 / (1/2 + np.sqrt(3)/3))

    # Enforcing zeros at the edge of the cylinder (BC)
    # for i in range(Nz):
        # center_diag[(Nr-1):Nth()]

    # Enforce BC at the edge of the cylinder
    # center_grid[Nr-1, :, :] = 1
    # north_grid[Nr-1, :, :] = 0
    # south_grid[Nr-1, :, :] = 0
    # west_grid[Nr-1, :, :] = 0
    # east_grid[Nr-1, :, :] = 0
    # up_grid[Nr-1, :, :] = 0
    # down_grid[Nr-1, :, :] = 0

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


def setup_diffusion_matrix(Nr, Nth, Nz, dt):
    dr, dth, dz = 1/Nr, 1/Nth, 1/Nz

    alpha_r, alpha_th, alpha_z = 1, 10, 1

    def at(i, j, k):
        # return i + j*Nr + k*Nr*Nth
        return k + j*Nz + k*Nz*Nth
    
    # A = np.zeros((Nr*Nth*Nz, Nr*Nth*Nz))
    # A = sp.csc_matrix((Nr*Nth*Nz, Nr*Nth*Nz))
    A = sp.lil_matrix((Nr*Nth*Nz, Nr*Nth*Nz))

    # Set up adjacent blocks
    for i in range(Nr):
        for j in range(Nth):
            for k in range(Nz):
                A[at(i,j,k), at(i,j,k)] = -(alpha_z * 2/dz**2 + alpha_th * 2/dth**2 + alpha_r * 2/dr**2)      # Center, c
                # if i > 0:
                try:
                    A[at(i,j,k), at(i-1,j,k)] = alpha_r / 2 / dr                      # North, r
                except:
                    print("north")
                # if j > 0:
                try:
                    A[at(i,j,k), at(i,j-1,k)] = alpha_th / 2 / dth**2 / (i+1) / dr         # West, th
                except:
                    print("west")
                # if k > 0:
                try:
                    A[at(i,j,k), at(i,j,k-1)] = alpha_z / 2 / dz**2                   # Down, z
                except:
                    print("down")
                # if i < Nr-1:
                try:
                    A[at(i,j,k), at(i+1,j,k)] = alpha_r / 2 / dr                   # South, r
                except:
                    print("south")
                # if j < Nth-1:
                try:
                    A[at(i,j,k), at(i,j+1,k)] = alpha_th / 2 / dth**2 / (i+1) / dr         # East, th
                except:
                    print("east")
                # if k < Nz-1:
                try:
                    A[at(i,j,k), at(i,j,k+1)] = alpha_z / 2 / dz**2                   # Up, z
                except:
                    print("up")
    
    # Fix angular dependency
    for i in range(Nr):
        for k in range(Nz):
            if k < Nz-1:
                A[at(i, Nth-1, k), at(i, Nth, k)] = 0
                A[at(i, Nth-1, k), at(i, 0, k)] = alpha_th / 2 / dth**2 / (i+1) / dr
            if k > 0:
                A[at(i, 0, k), at(i, -1, k)] = 0
                A[at(i, 0, k), at(i, Nth-1, k)] = alpha_th / 2 / dth**2 / (i+1) / dr
    
    # Fix no-flux condition in z-direction
    a, b = 1, 1/2 + np.sqrt(3)/3
    for i in range(Nr):
        for j in range(Nth):
            # Almost border layers
            A[at(i,j, 0), at(i,j, 1)]       += alpha_z * (a / b / dz**2 - 1 / 2 / dz**2)
            A[at(i,j, Nz-1), at(i,j, Nz-2)] += alpha_z * (a / b / dz**2 - 1 / 2 / dz**2)
            # Border layers
            A[at(i,j,0), at(i,j,0)]       -= alpha_z * (a / b / dz**2 - 2 / dz**2)
            A[at(i,j,Nz-1), at(i,j,Nz-1)] -= alpha_z * (a / b / dz**2 - 2 / dz**2)

    # Fix angular dependency in the first and last layers
    # A[at(0, 0, 0), at(0, Nth-1, 0)] = 1 / 2 / dth**2 / dr

    
    A *= dt / 2

    # # Fix edge of cylinder BC (edge = 0)
    # for j in range(Nth):
        # for k in range(Nz):
            # A[at(Nr-1, j, k), :] = 0
            # A[:, at(Nr-1, j, k)] = 0
            # A[at(Nr-1, j, k), at(Nr-1, j, k)] = 1


    # # Fix r=0 entry (sum over r=r1)
    for k in range(Nz):
        for j in range(Nth):
            A[at(0, j, k), :] = 0
            # A[at(0, j, k), at(0, j, k)] = 1
            for _j in range(Nth):
                A[at(0, j, k), at(1, _j, k)] = dth / alpha_th


    return A.tocsc()


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
    # Nr, Nth, Nz = 8, 8, 8
    # Nr, Nth, Nz = 16, 16, 16

    dt = 1e-4
    A_imp, A_exp = setup_system_matrix(Nr, Nth, Nz, .01)
    A = setup_diffusion_matrix(Nr, Nth, Nz, dt)

    img = plt.imshow(A.todense() != 0)
    plt.colorbar(img)
    plt.show()

    grid = np.zeros((Nr, Nth, Nz))
    # grid[1, :, 0] = 1
    # grid[0, :, 0] = 1
    grid[:, Nth//2, 0] = 1

    I = sp.eye(Nr*Nth*Nz, format="csc")
    D = 1/2
    left =  (I - D * A)
    right = (I + D * A)

    _grid = grid.copy()
    gv = grid_to_vector(_grid)
    print(left.shape)
    print(right.shape)
    print(gv.shape)
    print(np.sum(gv))
    for i in range(10):
        interm = right @ gv
        print(interm.shape)
        # gv = np.linalg.solve(left, interm)
        gv = spla.spsolve(left, interm.T)
        print(np.sum(gv))
    
    sol_grid = vector_to_grid(gv, Nr, Nth, Nz)

    fig = plt.figure()
    axs = [fig.add_subplot(2, 4, i+1) for i in range(8)]

    for i in range(4):
        img = axs[i].imshow(grid[:, :, i])#, vmin=0, vmax=1)
        img_sol = axs[i+4].imshow(sol_grid[:, :, i])#, vmin=0, vmax=1)
        plt.colorbar(img, ax=axs[i])
        plt.colorbar(img_sol, ax=axs[i+4])
        axs[i].set_title(f"z = {i}")
        axs[i+4].set_title(f"z = {i}")


    plt.show()

    # img = plt.imshow(A)
    # plt.show()