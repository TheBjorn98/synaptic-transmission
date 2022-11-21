from simulationUtils import *
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import time 

def main():
    #grid params
    w, h = 10, 10 #width of grid
    dx, dy = 0.1, 0.1
    numIterations = 100

    numSynapses = 100 #total number of synapses in system
    prop_fired_initially = 0.05 #number of synapses fired initially
    D = 4. #diffusion constant
    N0 = 10 #fire concentration
    fire_treshold = 0.01 #1% of N0 means the synaps will fire.
    radius = 2*dx #radius of synaps

    #initiating system
    synapses = generateSynapses(numSynapses, w, h,
                radius, N0, active=prop_fired_initially)
    #num_synapses * 5 matrix. Each synaps is a 1x5 array. [x, y, radius, N0, active: True, False] 
    
    U0 = np.zeros((int(w/dx), int(h/dy))) #Consentration matrix. Initially.
    U0, synapses = fireInitialSynapses(U0, synapses, dx, dy)
    U = U0.copy()

    history = np.zeros((numIterations, U0.shape[0], U0.shape[1]))
    history[0] = U0

    #iterating:
    dx2 = dx * dx; dy2 = dy * dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))  #stable timestep

    t0 = time.time()
    print(f"Starting simulation...\n synapses: {numSynapses} \t initially fired: {prop_fired_initially * numSynapses} \t fire treshold: {fire_treshold} \n" +
        f"stable timestep: {round(dt, 3)} \t time ellapsed: {round(numIterations * dt,3)}s")

    for i in range(1, numIterations):
        U = timestep(U0, U, D, dt, dx2, dy2) #f

        for j, synaps in enumerate(synapses):
            synaps_N_value = U[int(synaps[0] / dx), int(synaps[1] / dy)]

            if(synaps_N_value >= fire_treshold * synaps[3] and not synaps[4]):
                #synaps is not active and the N is above the treshold -> fire it.
                U, synapses[j] = fireSynaps(U, synaps, dx, dy)
        
        U0 = U.copy()
        history[i] = U0

    t1 = time.time()
    print(f"\nSimulation complete in {round(t1-t0, 3)}...\n")
    fig = plt.figure()
    ax = fig.add_subplot()

    def animate(i):
        ax.clear()
        ax.set_title(f"T = {dt * i:.2f}, Mass = {np.sum(history[i, :, :]):.2f}")
        ax.imshow(history[i, :, :].T, cmap=plt.get_cmap('hot'), vmin=0,vmax=N0)#, extent=[])

    ani = anim.FuncAnimation(fig, animate, numIterations, interval=100, blit=False)
    writergif = anim.PillowWriter(fps=30) 
    ani.save("animation.gif", writer=writergif)
    plt.show()


if __name__=="__main__": main()
