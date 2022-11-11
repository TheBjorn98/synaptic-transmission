import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla

def grid_to_vector(grid):
    return grid.flatten()

def vector_to_grid(gridvec, Nr, Nth, Nz):
    return grid.reshape((Nr, Nth, Nz))

def setup_system_matrix(Nr, Nth, Nz):
    pass

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

    Nr, Nth, Nz = 8, 8, 8

    grid = np.zeros((Nr, Nth, Nz))

    grid[0, :, :] = 1
    
    # grid_vec = grid_to_vector(grid)

    fig = plt.figure(figsize=(12,9))
    axs = [fig.add_subplot(2, 4, i+1) for i in range(8)]

    for i in range(Nz):
        axs[i].imshow(grid[:, :, i])
        axs[i].set_title(f"Depth: z={i+1}")
        axs[i].set_ylabel("r")
        axs[i].set_xlabel("theta")

    plt.show()
    
    