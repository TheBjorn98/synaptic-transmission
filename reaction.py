import numpy as np

def RK4_step(f, t, s, h):
    
    k1 = f(t, s)
    k2 = f(t+ h/2, s + h*k1/2)
    k3 = f(t + h/2, s + h*k2/2)
    k4 = f(t+ h, s + h*k3)

    return t + h,  s + h/6 *(k1+ 2*k2 + 2*k3 + k4)

def reaction_ode(t, s, k1=1, k2=1):

    Nt = -k1 * s[0] * s[1] + k2 * s[2] 

    return np.array([Nt,Nt,-Nt])

def get_transmitters(grid_vec):
    pass


def update_reaction_state(t0, s0, grid_vec, dt):

    s = s0.copy()
    
    # Grabbing the current amount of transmittors in the reaction domain
    #s[0] = get_transmitters(grid_vec)


    #RK4 step:
    """ 
    method = RK45(reaction_ode, t0=t0, y0=s0, t_bound=t0+dt)

    method.step()
    """

    return RK4_step(reaction_ode, t0, s0, dt)