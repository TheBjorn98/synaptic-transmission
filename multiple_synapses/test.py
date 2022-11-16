import matplotlib.pyplot as plt
import matplotlib.animation as anim
from CellularSpace import *

def main():
    N0 = 700
    D  = 4

    dx = 0.1; dy = 0.1
    width = 10; height = 10;

    cellspace = CellularSpace(width, height, dx, dy, D)
    for _ in range(10): cellspace.insertCell(radius=dx, N0=N0) #adding some random points

    sim = cellspace.simulate(100)
    
    U = sim["history"]
    dt = sim["timestep"]
    Nt = sim["timesteps"]

    fig = plt.figure()
    ax = fig.add_subplot()

    def animate(i):
        ax.clear()
        ax.set_title(f"T = {dt * i:.2f}, Mass = {np.sum(U[i, :, :]):.2f}")
        ax.imshow(U[i, :, :].T, cmap=plt.get_cmap('hot'), vmin=0,vmax=N0)#, extent=[])

    ani = anim.FuncAnimation(fig, animate, Nt, interval=100, blit=False)
    plt.show()

    


if __name__=="__main__":
    main()