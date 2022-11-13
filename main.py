from numerics import *
import numpy as np
import matplotlib.pyplot as plt

import time

def main():

    N_iter = 10
    R = 10
    t = 0

    #10 x 10 x 10
    grid_vec = np.random.uniform(0, 1, 10*10*10) #flatten solution from difusion. 
    reaction_state = np.array([100.30, 3.2, 1.1]) #[N, R, C] initial

    state_hist = [reaction_state]
    t_hist = [t]

    t0 = time.time()

    
    for _ in range(10000):

        new_t, new_s = update_reaction_state(grid_vec, t, reaction_state, 0.01)
        
        t += 0.1
        state_hist.append(new_s)
        t_hist.append(new_t)

    t1 = time.time()

    print(f"time: {t1-t0}")

    state_hist = np.array(state_hist)
    N = np.array([s[0] for s in state_hist])
    R = np.array([s[1] for s in state_hist])
    C = np.array([s[2] for s in state_hist])

    plt.plot(t_hist, N)
    plt.plot(t_hist, R)
    plt.plot(t_hist, C)
    plt.show()
    print(N)
     





















if __name__ == "__main__":
    main()