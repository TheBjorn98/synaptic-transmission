import numpy as np

def generateSynapses(num, w, h, radius, N0, sigmaN0=0.2, active=0.2):
    #each synaps consist of: [x, y, radius, N0, active: True, False]
    synapses = np.zeros((num, 5))

    for i in range(num):
        synapses[i] = [np.random.uniform(0.1, 0.9)*w, np.random.uniform(0.1, 0.9)*h,
                       radius, np.random.normal(N0, sigmaN0*N0),
                       np.random.uniform(0,1) <= active]
    return synapses

def fireSynaps(U, synaps, dx, dy):
    #synaps: [x, y, radius, N0, active: True, False]
    #dx2 = dx * dx

    #drawing the circle
    r2 = synaps[2] * synaps[2] 
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p2 = (i * dx - synaps[0])**2 + (j*dy-synaps[1])**2
            if( p2 < r2 ):
                U[i, j] = synaps[3] # = N0

    synaps[4] = 1

    return U, synaps

def killSynaps(synaps): synaps[4] = 0; return synaps

def fireInitialSynapses(U, synapses, dx, dy):
    _U = U.copy()
    for i in range(len(synapses)):
        if(synapses[i][4] == 1): #if active synaps
            U, synapses[i] = fireSynaps(U, synapses[i], dx, dy)
            
    return U, synapses        

def timestep(U0, U, D, dt, dx2, dy2):
    U[1:-1, 1:-1] = U0[1:-1, 1:-1] + D * dt * (
        (U0[2:, 1:-1] - 2*U0[1:-1, 1:-1] + U0[:-2, 1:-1])/dx2
        + (U0[1:-1, 2:] - 2*U0[1:-1, 1:-1] + U0[1:-1, :-2])/dy2
    )
    return U
