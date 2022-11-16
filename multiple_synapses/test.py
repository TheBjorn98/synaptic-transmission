import matplotlib.pyplot as plt
import matplotlib.animation as anim
from CellularSpace import *

def main():
    #grid params and such
    N0 = 10
    D  = 4
    dx = 0.1; dy = 0.1
    width = 10; height = 10;

    #initiating the simulator object
    cellspace = CellularSpace(width, height, dx, dy, D, fire_treshold=0.001 * N0) 
    
    #adding some cells
    p = 0.1
    for _ in range(500):
        if(np.random.uniform(0,1) <= p):
            cellspace.insertCell(radius=dx, N0=np.random.normal(N0, 2)) #adding a initially active cell
        else:
            cellspace.insertCell(radius=dx, N0=0) #adding a none-active cell

    #simulating
    sim = cellspace.simulate(100)
    #results:
    U = sim["history"]
    dt = sim["timestep"]
    Nt = sim["timesteps"]

    #visualizing
    fig = plt.figure()
    ax = fig.add_subplot()

    def animate(i):
        ax.clear()
        ax.set_title(f"T = {dt * i:.2f}, Mass = {np.sum(U[i, :, :]):.2f}")
        ax.imshow(U[i, :, :].T, cmap=plt.get_cmap('hot'), vmin=0,vmax=N0)#, extent=[])

    ani = anim.FuncAnimation(fig, animate, Nt, interval=100, blit=False)
    writergif = anim.PillowWriter(fps=30) 
    ani.save("animation.gif", writer=writergif)
    plt.show()

    


if __name__=="__main__":
    main()