from numerics import *
from reaction import *
import numpy as np
import matplotlib.pyplot as plt

import time

def main():
    #simulation parameters:
     N_its = 1000
    
    #initial conditions
    reaction_state = np.array([0, 0, 0])
    grid_vec = np.zeros(1000)
    t0 = 0

    
    #run iterations
    for _ in range(N_its):

        #grid_vec = update_diffusion() 
        
        #updating state:
        t, reaction_state = update_reaction_state(t, reaction_state, grid_vec, 0.01)
        
  


if __name__ == "__main__":
    main()
