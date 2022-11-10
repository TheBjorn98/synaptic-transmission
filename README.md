# synaptic-transmission

A numerical project in mathematical modelling aiming to simulate a simplified model of synaptic informational transmission.

The aim of the code is to provide a 3D numerical scheme for the synaptic cleft which couples neurons.
When neurons coomunicate through axons and dendrites, there is an electrical signal which is passed to the axon terminal.
Between the pre-synaptic and post-synaptic axon terminals is a gap which effectively insulates the two terminals electrically.
In order to send a signal to the other terminal, neurotransmitters are released on the pre-synaptic terminal which diffuse towards the post-synaptic terminal.
At the post-synaptic terminal, neurotransmitters activate receptors which trigger a signal travelling up the post-synaptic axon terminal to the other neuron.

The numerical model therefore includes:

1. A release-scheme for neurotransmitters inputted as an initial neurotransmitter distribution on the pre-synaptic terminal
2. Three dimensional (radial, tangential, longitudinal) diffusion of neurotransmitters in the synaptic cleft
3. Chemical equilibrium reactions acting as post-synaptic activation

## Modelling equations

For the concentration of neurotransmitters, denoted $N$, the following is the mathematical model:

**The domain representing the synaptic cleft**

Cylinder:
* $0 < r < R$
* $0 < \theta < 2\pi$
* $0 < z < L$
* Pre-synaptic terminal: $z = 0$
* Post-synaptic terminal: $z = L$

**Diffusion between the terminals**

$$
N_t = D \nabla^2 N = D \left[\frac1r \partial_r \left( r \partial_r \right) + \frac1{r^2} \partial_\theta^2 + \partial_z^2\right] N
$$

**Reaction at the post-synaptic terminal**

Chemical reaction: $R + N \iff C$

Reaction equation ($R, N, C$ are concentrations):

$$
N_t = -k_1 R N + k_{-1} C
$$

## Numerical scheme

We set the concentration:

$$
N(r_i, \theta_j, z_k, t_n) = N_{ijk}^n
$$

Difference equations for all differential operators as follows:

* $\delta_z^2(N_{ijk}^n) = \frac{N_{i,j,k-1}^n - 2 N_{i,j,k}^n + N_{i,j,k+1}^n}{\Delta z^2}$
* $\delta_\theta^2(N_{ijk}^n) = \frac{N_{i,j-1,k}^n - 2 N_{i,j,k}^n + N_{i,j+1,k}^n}{\Delta \theta^2}$
* $\delta_r^2(N_{ijk}^n) = \frac{1}{r_i \Delta r^2} \left[ r_{i-\frac12}\left(N_{i-1,j,k}^n - N_{i,j,k}^n\right) + r_{i+\frac12} \left( N_{i+1,j,k}^n - N_{i,j,k}^n\right)\right]$

* $\delta_t(N_{ijk}^n) = \frac{N_{i,j,k}^{n+1} - N_{i,j,k}^n}{\Delta t}$

The diffusion 

The Crank-Nicholson scheme for the diffusion part of the equation becomes

$$
\delta_t(N_{i,j,k}^n) =
    \frac{\delta_r^2(N_{i,j,k}^{n+1}) + \delta_\theta^2(N_{i,j,k}^{n+1}) + \delta_z^2(N_{i,j,k}^{n+1})}2
    + \frac{\delta_r^2(N_{i,j,k}^n) + \delta_\theta^2(N_{i,j,k}^n) + \delta_z^2(N_{i,j,k}^n)}2
$$

