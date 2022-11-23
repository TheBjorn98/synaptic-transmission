# synaptic-transmission

A numerical project in mathematical modelling aiming to simulate a simplified model of synaptic informational transmission.

The aim of the code is to provide a 3D numerical scheme for the synaptic cleft which couples neurons.
When neurons communicate through axons and dendrites, there is an electrical signal which is passed to the axon terminal.
Between the pre-synaptic and post-synaptic axon terminals is a gap which effectively insulates the two terminals electrically.
In order to send a signal to the other terminal, neurotransmitters are released on the pre-synaptic terminal which diffuse towards the post-synaptic terminal.
At the post-synaptic terminal, neurotransmitters activate receptors which trigger a signal travelling up the post-synaptic axon terminal to the other neuron.

The travelling from pre- to post-synapse happens through a diffusion process, while the activation is modeled with a chemical reaction equation.

## How to run this project

> Summary: 
> * Run `generate_plots.py` to see some results.
> * Main question is numerically implemented in `grid_3d_cartesian.py` and `reaction.py`
> * Followup question 1 numerically implemented (though not complete) in files in `multipleSynapses/`-folder

Ensure that the packages `numpy`, `scipy` and `matplotlib` are installed.
Then consider the following files:

* `generate_plots.py` contains functions for plotting some potentially interesting results from the simulation.
	* If the code runs slowly, pass a smaller `N` as argument to these functions, this will reduce the spatial refinement.
* `grid_3d_cartesian.py` contains functions to build system matrices and iterate the time dependent system.
	* `build_diffusion_matrix` constructs a matrix of the kind "A" as seen in the backward Euler equation further down.
	* `iterate_system_bw_euler` performs timestepping of the system updating both diffusion and reaction terms using backward Euler.
	* `update_diffusion` performs one step of matrix inversion to arrive at the next vector of concentrations.
* `reaction.py` contains functionality for creating an ode-function and updating the reaction term.
	* `make_reaction_ode` takes a forward and backward reaction coefficient and wraps a function with these values as hidden contants.
	* `update_reaction` takes the state of the system and performs an iteration of RK4 to devise the next state of the system after the reaction has taken place.
* `grid_2d_cartesian.py` contains code for performing this simulation in 2D, making use of rotational symmetry.
	* The code does not work at the moment and needs fixing
* `grid_3d_cylindrical.py` contains defunct code attempting to solve the problem using cylinder coordinates instead of Cartesian.
	* This code has been abandoned, but advances in the other files may bear fruits for this code in the future.
* `multipleSynapses/` contains files for numerical methods on the 2D geometric reduction of the intercellular space
	* `simulation.py` runs the multiple synapse simulation
	* `simulationUtils.py` contains helper functions to facilitate this simulation
	* This code is not finished, but a working idea is in place

### Followup question 2: Transporters

Implementation of transporters into the model is not something which was implemented.
The necessary work to implement this is to specify a separate reaction ode on the sides of the cube.
Currently, there is a receptor reaction ode acting on the bottom of the cube, and if the vector of neurotransmitters `n_vec` is sliced properly,
it is possible to add reaction odes taking the north, south, east and west walls of the cube as the vectors to act on.

A sketch of the implementation is as follows:
* Slice the vector correctly for all walls of the cube, alias these to slice-objects which may be called easily by the numpy code
* Store all the wall-slice objects in an array
* Create the corresponding transporter reaction odes (possibly in multiple stages) and store them in an array in the same order as the slice objects
* Initialize transporter-vectors (`t_vec`) and (transporter-neurotransmitter)-vectors (`p_vec`) for all walls
* When updating the reaction term:
	* For each pair of `slice` and `reaction ode` in the arrays
	* `update_reaction` used with `n_vec[slice]`, `reaction_ode` and the appropriate `t_vec`-s and `p_vecs`
	* Update the entire `n_vec` to reflect the result of these odes

### Followup question 3: Flow

Implementation of the flow in the 3D model would be more difficult.
One way to do it may be to modify the directional diffusion coefficients and add more coefficients.
At this moment the code uses the same coefficient in positive directions of `x` and `y`,
but if there were different coefficients, `(ax_pos, ax_neg)`, scaling diffusion in the positive and negative `x`-direction,
we may simulate flow going in the transverse direction.

Which values these different coefficients should take is another matter, and is difficult to estimate without doing more research.

## Modelling equations

The chemical reaction taking place is: $R + N \iff C$, this chemical reaction can be written as:

$$
\partial_t N = - k_1 R N + k_{-1} C
$$

Together with the diffusion equation:

$$
\partial_t N = \nabla^2 N
$$

this system becomes:

$$
\partial_t N = \nabla^2 - k_1 R N + k_{-1} C
$$

Which is solved for an approximation to the function $N(t, x, y, z)$ marking the spatial concentration of neurotransmitters and how it changes over time.

## Setup

We endeavor to solve this equation on the unit cube with directionally dependent coefficients of diffusion.
As initial conditions we have the neurotransmitters as a scaled point source and the receptors as evenly spread on the other side

$$
N(0, .5, .5, 0) = 1, R(0, x, y, 1) = \Tilde{R0}
$$

As boundary conditions we have Dirichlet conditions at the faces in the x- and y-directions and no-flux Neumann conditions in the z-directions:

$$
N(t, 0, y, z) = N(t, 1, y, z) = N(t, x, 0, z) = N(t, x, 1, z) = 0
$$

$$
\partial_z N = 0 \text{ for } z=0, z=1
$$

## Discretization and scheme

We use a second order central difference for the terms in the Laplace operator, while we use a backward difference for the time.

$$
\partial_{xx}N \approx \frac{N_{i-1, j, k}^n - 2 N_{i, j, k}^n + N_{i+1, j, k}^n}{\Delta x^2} \quad \partial_t N \approx \frac{N_{i,j,k}^n - N_{i,j,k}^{n-1}}{\Delta t}
$$

This results in a fully implicit backward Euler scheme in the diffusion part of the equation.

$$
(I - \Delta t A) N^{n+1} = N^{n}
$$

This system is solved using SciPy.
