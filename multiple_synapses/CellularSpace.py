import numpy as np
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

class CellularSpace:
    def __init__(self, width: int, height: int, dx: float, dy: float,
                 D: float):
        """
        Object holding all the cells
        """
        
        #grid params:
        self.w = width; self.h=height
        self.dx = dx; self.dy = dy
        self.nx = int(width/dx); self.ny = int(height/dy)

        #difusion constant. For now the same for all cells.
        self.D = D

        #Cells will contain array [N, R, C], their molecule structure (Not a chemist).
        self.CellReactions = np.array([])
        self.CellPositions = np.array([])
    
        #initial universal consentration grid.
        self.U0 = np.zeros((self.nx, self.ny))
        

    
    def insertCell(self, point=np.array([]), radius = 1, N0 = 1.0):
        """
        Function for inserting a cell/synaps with a given N-consentration
        of N0. Thinking about adding a random placement feature. Maybe later.
        Can be added as a circle. I dont know if that is usefull.
        """
        if(len(point) == 0):
            point = np.array([int(np.random.uniform(0, 1)*self.w), int(np.random.uniform(0, 1) * self.h)])

            while(point in self.CellPositions):
                point = np.array([int(np.random.uniform(0, 1)*self.w), int(np.random.uniform(0, 1) * self.h)])

        r2 = radius * radius
        for i in range(self.nx):
                for j in range(self.ny):
                    p2 = (i * self.dx - point[0])**2 + (j*self.dy-point[1])**2

                    if( p2 < r2 ):
                        self.U0[i, j] = N0

        self.CellPositions = np.concatenate((self.CellPositions, point), axis=0)
        self.CellReactions = np.concatenate((self.CellReactions, np.array([N0, 0, 0])), axis=0)


    def simulate(self, timesteps):
        """
        Simulates the time evolution given a forward difference in time
        and central difference in space.

        """

        U0 = self.U0
        U = np.zeros((self.nx, self.ny))

        D = self.D

        dx2 = self.dx * self.dx; dy2 = self.dy * self.dy
        dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

        print(f"Starting simulation.\n Stable timestep(dt): {dt} \n simulation time: {dt * timesteps}")

        U0 = self.U0
        U = np.zeros((self.nx, self.ny))

        D = self.D
        U_hist = np.zeros((timesteps, self.nx, self.ny)); U_hist[0] = U0

        for i in range(1, timesteps):
            #propagating 
            U[1:-1, 1:-1] = U0[1:-1, 1:-1] + D * dt * (
                (U0[2:, 1:-1] - 2*U0[1:-1, 1:-1] + U0[:-2, 1:-1])/dx2
                + (U0[1:-1, 2:] - 2*U0[1:-1, 1:-1] + U0[1:-1, :-2])/dy2
            )

            U0 = U.copy()
            U_hist[i] = U0
  
            # TODO add reaction stuff to all the cells. And fire some cells when N is large enough?

        self.U0 = U0 #updating in case we want to simulate more later on.
    
        return {"history": U_hist,
                "timestep": dt,
                "timesteps": timesteps}

    
    



        
