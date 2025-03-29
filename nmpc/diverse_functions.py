"""
diverse_functions.py

This module contains various basic functions used in the simulation

1. model of the dynamical system
2. inverse function of the level set for 3D plotting
"""

import casadi as ca
import numpy as np

def nominal_dynamics(x, u, bias=np.zeros(3), mode="SIM"):
    """
    Compute the next state based on the NOMINAL discrete-time model.
    
    Parameters:
        x: casadi.SX or MX, current state vector.
        u: casadi.SX or MX, control input vector.
        bias: list, the bias added to the nominal parameters
        mode: "NLP" or "SIM" depending on the purpose of usage: 
        1) for optimization in MPC or 2) used for simulation
    
    Returns:
        x_next: casadi.SX or MX, next state vector.
    """
    if mode == "NLP":
        f1 = -0.99 * x[1]
        f2 = (0.85 + bias[0]) * ca.tanh(x[0]) + (0.995 + bias[1]) * ca.tanh(x[1]) + (0.01+ bias[2]) * u[0]
        x_next = ca.vertcat(f1, f2)
    elif mode == "SIM":
        f1 = -0.99 * x[1]
        f2 = (0.85 + bias[0]) * np.tanh(x[0]) + (0.995 + bias[1]) * np.tanh(x[1]) + (0.01+ bias[2]) * u[0]
        x_next = np.array([f1, f2])
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")

    return x_next


def inverse_level_set(x, r):
    """
    The inverse level set function, r = x * y, the basic inverse-proportional function
    other complicated functions can also be coded
    """
    return r / x

