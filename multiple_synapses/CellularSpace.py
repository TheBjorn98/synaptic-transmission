import numpy as np
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

class CellularSpace:
    def __init__(self, width: int, height: int, dx: float, dy: float,
                 D: float, fire_treshold: float = 0.1):
        """
        Object holding all the cells
        """
        
        #grid params:
        self.w = width; self.h=height
        self.dx = dx; self.dy = dy
        self.nx = int(width/dx); self.ny = int(height/dy)

        #difusion constant. For now the same for all cells.
        self.D = D

        #cells will contain  [x, y, N0-value] - elements. Corresponding to cells
        self.cells = np.empty((0, 3))
    
        #initial universal consentration grid.
        self.U0 = np.zeros((self.nx, self.ny))
        
        self.fire_tresh = fire_treshold # treshold for firing cell when N = fire_tresh 
    
    def insertCell(self, point=np.array([]), N0 = 1, radius=None, active=False):
        """
        Function for inserting a cell/synaps with a given N-consentration
        of N0. Thinking about adding a random placement feature. Maybe later.
        Can be added as a circle. I dont know if that is usefull.
        """
        if(len(point) == 0):
            point = np.array([np.random.uniform(0, 1)*self.w, np.random.uniform(0, 1) * self.h])

            while(point in self.cells):
                point = np.array([int(np.random.uniform(0, 1)*self.w), int(np.random.uniform(0, 1) * self.h)])

        r2 = radius * radius
        for i in range(self.nx):
                for j in range(self.ny):
                    p2 = (i * self.dx - point[0])**2 + (j*self.dy-point[1])**2

                    if( p2 < r2 ):
                        self.U0[i, j] = N0

        cell = np.array([point[0], point[1], N0])
        self.cells = np.concatenate((self.cells, [cell]), axis=0)


    def simulate(self, timesteps):
        """
        Simulates the time evolution given a forward difference in time
        and central difference in space.

        """

        D = self.D
        dx2 = self.dx * self.dx; dy2 = self.dy * self.dy
        dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

        cells = self.cells
        is_active = np.array([cell[2] != 0 for cell in cells]) #list containing information if the cell is active or not

        print(f"Starting simulation.\n Stable timestep(dt): {dt} \n simulation time: {dt * timesteps}")

        U0 = self.U0
        U = np.zeros((self.nx, self.ny))
        U_hist = np.zeros((timesteps, self.nx, self.ny)); U_hist[0] = U0
        

        for i in range(1, timesteps):

            #propagating 1 it
            U[1:-1, 1:-1] = U0[1:-1, 1:-1] + D * dt * (
                (U0[2:, 1:-1] - 2*U0[1:-1, 1:-1] + U0[:-2, 1:-1])/dx2
                + (U0[1:-1, 2:] - 2*U0[1:-1, 1:-1] + U0[1:-1, :-2])/dy2
            )
            
            U0 = U.copy()
            U_hist[i] = U0

            #checking if there are unactive cells that should fire
            for i, cell in enumerate(cells):

                grid_N_value = U0[int(cell[0] / self.dx), int(cell[1] / self.dy)] #N value at cell.
                
                #unactive cell check:
                if(grid_N_value >= self.fire_tresh and is_active[i] == False):
                    
                    U0[int(cell[0] / self.dx), int(cell[1] / self.dy)] = 10 # = N0 of that cell. Firing
                
                #active cell check:
                elif(grid_N_value < self.fire_tresh and is_active[i] == True):
                    is_active[i] = False

                


            # TODO add reaction stuff to all the cells. And fire some cells when N is large enough?

        self.U0 = U0 #updating in case we want to simulate more later on.
    
        return {"history": U_hist,
                "timestep": dt,
                "timesteps": timesteps}

    

