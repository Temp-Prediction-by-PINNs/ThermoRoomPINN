# ThermoRoomPINN

Master's (M2) project at Le Mans University, aiming to model 2D unsteady thermal diffusion in a room using **Physics-Informed Neural Networks (PINNs)**.

Unlike a classical data-driven model, a PINN embeds the governing partial differential equation (PDE) directly into the network's loss function, allowing it to learn a physically consistent field even with very little (or no) measured data inside the domain.

---

## Members

| Role        | Name            | Email                               | University ID |
|-------------|----------------|--------------------------------------|----------------|
| Developer   | Maelig Pesantez | maelig.pesantez.Etu@univ-lemans.fr | @e2103064      |
| Developer   | Luka Cognard    | Luka.Cognard.Etu@univ-lemans.fr    | @s2200371      |
| Supervisor  | Anthony Larcher | Anthony.Larcher@univ-lemans.fr     | @alarcher (LIUM) |

---

## Context and objectives

The goal is to predict the evolution of temperature T(x, y, t) in a 2D room of 1 m x 1 m:

- At t = 0, the room is at a homogeneous ambient temperature T_amb = 20C, except for a hot object placed at the center (block or disk) at T_obj = 80C.
- The outer walls remain at T_amb = 20C for all t > 0 (Dirichlet boundary conditions).
- The network must learn to predict T(x, y, t) at every point of the domain and at every instant t in [0, t_max].

The phenomenon is governed by the 2D unsteady heat equation with no internal source:

```
dT/dt - alpha * (d2T/dx2 + d2T/dy2) = 0      in Omega x ]0, t_max]
```

where alpha is the thermal diffusivity of the medium (~ 2x10^-5 m^2/s for air, to be non-dimensionalized to stabilize training).

---

## Work plan

1. **Non-dimensionalization and sampling** - non-dimensional reformulation of the PDE, generation of collocation points (initial condition, boundary conditions, internal residual).
2. **PINN architecture and automatic differentiation** - network T_theta(x, y, t) in PyTorch, partial derivatives via `torch.autograd`, multi-objective loss L = w_IC*L_IC + w_BC*L_BC + w_res*L_res.
3. **Training strategy** - hybrid training with Adam then L-BFGS, using at least one advanced technique for the stiffness of the initial condition (adaptive resampling, dynamic loss weighting, or hard constraints).
4. **Numerical validation** - reference solution from a classical method (finite differences or finite elements), comparison against the PINN (MSE, relative L2 error).
5. **Interactive demonstrator** - Gradio interface allowing the hot object's parameters to be varied, the predicted heat map to be visualized via a time slider, and error maps to be displayed.

---

## Reference Solver
A 2D finite difference simulation (FTCS scheme) has been implemented to generate the reference solution.
<img width="5400" height="480" alt="reference_snapshots" src="https://github.com/user-attachments/assets/0a2ab226-0d7f-43be-9f12-3e6e21d88b5c" />

Watch the full animation of heat transfer: 
<video src="https://github.com/user-attachments/assets/e122b4a4-eefe-47c0-beb5-b923bc814db3" controls autoplay loop muted width="100%">
</video>



---

## Deliverables

- Git repository (clean, documented code)
- Project presentation (5 minutes)
- Interactive demonstrator (Gradio)

---

## Resources

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems.* Journal of Computational Physics.
- [PyTorch Autograd documentation](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html)
