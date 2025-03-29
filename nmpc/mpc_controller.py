"""
mpc_controller.py

This module contains
(1) the basic open-loop MPCController class that
implements a Model Predictive Controller (MPC) for a discrete-time system;
(2) the closed-loop MPCSimulator class that
implements the open-loop controller iteratively and runs the closed-loop simulation

The controller uses CasADi for symbolic modeling and the optimizer within CasADi.
"""

import casadi as ca
import numpy as np
from nmpc.utils import generate_uniform_sphere_vectors

class MPCController:
    """
    This is the MPC controller class, it has two parts
    1. Initialization and defining the NLP optimization problem
    2. Solve the NLP problem with a specified initial state and return all outputs for open-loop analysis
    3. Solve the NLP but only return the first input for closed-loop simulation and analysis
    """
    def __init__(self, horizon,
                 x_dim, u_dim,
                 umin, umax,
                 Q, R,
                 dynamics, bias=np.zeros(3)):
        """
        Initialize the MPC controller.

        Parameters:
            horizon: int, prediction horizon.
            x_dim: int, dimension of the state vector.
            u_dim: int, dimension of the control input.
            umin: numpy array, lower bound for control inputs.
            umax: numpy array, upper bound for control inputs.
            Q: numpy array, state weighting matrix.
            R: numpy array, input weighting matrix.
            dynamics: function, discrete-time dynamics function.
            bais: bias added to the nominal value of the parameters
        """
        self.N = horizon
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.umin = umin
        self.umax = umax
        self.Q = Q
        self.R = R
        self.dynamics = dynamics
        self.bias = bias

        # Create an Opti instance
        self.opti = ca.Opti()

        # Decision variables: states over the horizon and control inputs
        self.X = self.opti.variable(x_dim, self.N+1)
        self.U = self.opti.variable(u_dim, self.N)

        # Parameter for the initial state
        self.X0 = self.opti.parameter(x_dim)
        # Initial condition constraint
        self.opti.subject_to(self.X[:,0] == self.X0)

        # Build the cost function and constraints over the horizon
        self.obj = 0  # Initialize objective function
        for k in range(self.N):
            # Stage cost (assuming reference is zero; modify as needed)
            self.obj += ca.mtimes([self.X[:, k].T, Q, self.X[:, k]]) + ca.mtimes([self.U[:, k].T, R, self.U[:, k]])

            # Dynamics constraint: x_{k+1} = f(x_k, u_k)
            x_next = self.dynamics(self.X[:, k], self.U[:, k], self.bias, "NLP")
            self.opti.subject_to(self.X[:, k+1] == x_next)

            # Input constraints (elementwise)
            self.opti.subject_to(self.umin <= self.U[:, k])
            self.opti.subject_to(self.U[:, k] <= self.umax)

        # Terminal cost (the weight is set the same as the stage cost)
        self.obj += ca.mtimes([self.X[:, -1].T, Q, self.X[:, -1]])

        # Set the objective
        self.opti.minimize(self.obj)

        # Configure the solver
        opts = {"print_time": False, "ipopt": {"print_level": 0}}
        self.opti.solver("ipopt", opts)

    def solve_open(self, x0_val):
        """
        Solve the MPC problem for a given initial state and return all info for open-loop analysis.

        Parameters:
            x0_val: numpy array, initial state value.

        Returns:
            u_0: numpy array, the first control action.
            x_traj: numpy array, state trajectory
            u_traj: numpy array, input trajectory
            x_N: the open-loop final state
            V_N: the value of the MPC value function
        """
        # Set the initial state parameter value
        self.opti.set_value(self.X0, x0_val)
        # Solve the optimization problem
        sol = self.opti.solve()
        # Extract the first control input
        u_0 = sol.value(self.U[:, 0])
        # Extract the state and input trajectory
        x_traj = sol.value(self.X)
        u_traj = sol.value(self.U)
        # Extract the objective value
        obj_val = sol.value(self.obj)

        out_mpc = {"u_0": u_0,
                    "x_traj": x_traj, "u_traj": u_traj,
                    "x_N": x_traj[:,-1], "V_N": obj_val}
        return out_mpc

    def solve_closed(self, x0_val):
        """
        Solve the MPC problem for a given initial state
        and return only the first input for closed-loop integration

        Parameters:
            x0_val: numpy array, initial state value.

        Returns:
            u_0: numpy array, the first control action.
        """
        # Set the initial state parameter value
        self.opti.set_value(self.X0, x0_val)
        # Solve the optimization problem
        sol = self.opti.solve()
        # Extract the first control input
        u_0 = sol.value(self.U[:, 0])
        return u_0

    def solve_input_traj(self, x0_val):
        """
        Solve the MPC problem for a given initial state
        and return only the input sequence

        Parameters:
            x0_val: numpy array, initial state value.

        Returns:
            u_traj: numpy array, the input sequence
        """
        # Set the initial state parameter value
        self.opti.set_value(self.X0, x0_val)
        # Solve the optimization problem
        sol = self.opti.solve()
        # Extract the input trajectory
        u_traj = sol.value(self.U)
        return u_traj

    def solve_value(self, x0_val):
        """
        Solve the MPC problem for a given initial state
        and return only the value function

        Parameters:
            x0_val: numpy array, initial state value.

        Returns:
            obj_val: numpy value
        """
        # Set the initial state parameter value
        self.opti.set_value(self.X0, x0_val)
        # Solve the optimization problem
        sol = self.opti.solve()
        # Extract the objective value
        obj_val = sol.value(self.obj)
        return obj_val


class MPCSimulator:
    """
    This is the MPC simulator class, it has two parts
    1. Initialization and defining the controller
    2. Solve the NLP problem iteratively to perform the closed-loop simulation
    """
    def __init__(self, horizon,
                 x_dim, u_dim,
                 u_min, u_max,
                 Q, R,
                 dynamics):
        """
        Initialize the MPC controller.

        Parameters:
            horizon: int, prediction horizon.
            x_dim: int, dimension of the state vector.
            u_dim: int, dimension of the control input.
            u_min: numpy array, lower bound for control inputs.
            u_max: numpy array, upper bound for control inputs.
            Q: numpy array, state weighting matrix.
            R: numpy array, input weighting matrix.
            dynamics: function, discrete-time dynamics function.
        """
        # Extract the useful information
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.u_min = u_min
        self.u_max = u_max
        self.Q = Q
        self.R = R
        self.dynamics = dynamics

        # Specify the basic controller
        self.controller = MPCController(horizon, x_dim, u_dim, u_min, u_max, Q, R, dynamics)

    def simulate_closed_loop(self, x_0, time, bias_true, mode="SIMPLE"):
        """
        Solve the MPC problem for a given initial state and return all info for closed-loop analysis.

        Parameters:
            x_0: numpy array, initial state of the simulation.
            time: int, simulation horizon
            bias_true: the bias added to the nominal model to represent the true model
            mode: char, two options
                1) "SIMPLE": only returns the value (default option)
                2) "COMPLETE": returns all information
        Returns:
            x_traj: numpy array, state trajectory
            u_traj: numpy array, input trajectory
            x_T: numpy array, final state at the end of the simulation
            V_T: numpy array, the closed-loop system performance
        """

        # initialize the state and input trajectory
        x_traj = np.zeros((self.x_dim, time + 1))
        u_traj = np.zeros((self.u_dim, time))

        # assign the first state
        x_traj[:, 0] = x_0

        # initialize the closed-loop performance
        cost_cl = x_0.T @ self.Q @ x_0

        # loop for the simulation
        for k in range(time):
            # 1) solve the MPC at the the current time step
            u_traj[:, k] = self.controller.solve_closed(x_traj[:, k])

            # 2) propagate to the next state usin the true model
            x_traj[:, k+1] = self.dynamics(x_traj[:, k], u_traj[:, k], bias_true, "SIM")

            # 3) Update the cost
            cost_cl += x_traj[:, k+1].T @ self.Q @ x_traj[:, k+1] + u_traj[:, k].T @ self.R @ u_traj[:, k]

        # return the output
        if mode == "SIMPLE":
            out_mpc_cl = cost_cl
        elif mode == "COMPLETE":
            out_mpc_cl = {"x_traj": x_traj, "u_traj": u_traj,
                          "x_T": x_traj[:, -1], "V_T": cost_cl}
        else:
            raise ValueError("Invalid input! Please use 'SIMPLE' for only the value function"
                            " or 'COMPLETE' for all information.")

        return out_mpc_cl

    def simulate_infinite_horizon(self, x_0, bias_true):
        """
        This function simulate the open-loop infinite-horizon control for the true system

        Parameters:
            x_0: the initial state
            bias_true: the true system bias

        Return:
            the infinite-horizon value
        """
        # set the value difference tolerance
        tol = 1e-4
        # set an initial horizon and horizon increments
        horizon = int(1e2)
        horizon_increments = int(10)
        # solve the problem for the initial horizon
        mpc_base = MPCController(horizon, self.x_dim, self.u_dim,
                                 self.u_min, self.u_max,
                                 self.Q, self.R,
                                 self.dynamics, bias_true)
        value_prior = mpc_base.solve_value(x_0)

        # initialize the value difference in a naive way
        err_value = 1

        # looping for value function convergence
        while err_value > tol:
            # increase the horizon
            horizon += horizon_increments
            # solve the problem for the incremented horizon
            mpc_incremented = MPCController(horizon, self.x_dim, self.u_dim,
                                     self.u_min, self.u_max,
                                     self.Q, self.R,
                                     self.dynamics, bias_true)
            # obtain the posterior value function
            value_posterior = mpc_incremented.solve_value(x_0)
            # update the value error and prior value function
            err_value = value_posterior - value_prior
            value_prior = value_posterior

        return value_prior



class MPCSensitivity:
    """
    This is the MPC sensitivity calculator, which computes the input sensitivity.

    For this class, a high-level data structure is used to initialize the class,
    which is the mpc_info, and this data structure will be used here only for simulation
    convenience.
    """
    def __init__(self, mpc_info, horizon, nominal_dynamics, num_vec, num_state):
        """
        The initialization of the MPC value function.

        Parameters:
            mpc_info: dict, it has the following attributes:
            1) x_dim: state dimension
            2) u_dim: input dimension
            3) u_min: input lower bounds
            4) u_max: input upper bounds 
            4) Q: state penalizing matrix
            5) R: input penalizing matrix
            6) dim_para: dimension of the parameter that will be perturbed
            horizon: prediction horizon
            nominal_dynamics: the dynamics function
            num_vec: number of scenarios considered in the sensitivity analysis
            num_state: number of state given a level of state norm
        """
        # Extract the basic information
        self.x_dim = mpc_info["x_dim"]
        self.u_dim = mpc_info["u_dim"]
        self.u_min = np.array([mpc_info["u_min"]])
        self.u_max = np.array([mpc_info["u_max"]])
        self.Q = mpc_info["Q"]
        self.R = mpc_info["R"]
        self.dim_para = mpc_info["dim_para"]
        self.nominal_dynamics = nominal_dynamics
        self.horizon = horizon
        self.num_vec = num_vec
        self.num_state = num_state

        # Obtain the nominal MPC
        self.mpc = MPCController(horizon, self.x_dim, self.u_dim,
                            self.u_min, self.u_max,
                            self.Q, self.R, nominal_dynamics)

    def sensitive_input_single_state(self, e_para, x_0):
        """
        The function computes the input sensitivity for a given perturbation level
        and a specific state.

        Parameters:
            e_para: the norm of the parametric error
            x_0: initial state
        """
        # generate the bias matrix
        error_matrix = generate_uniform_sphere_vectors(self.dim_para, e_para, self.num_vec)
        # initialize the sensitivity vector
        vec_sensitivity = np.zeros(self.num_vec)
        # compute the nominal input
        u_nm = self.mpc.solve_input_traj(x_0)

        # looping to get the input perturbation for each instance
        for i in range(self.num_vec):
            # define the MPC controller for a specific bias
            temp_mpc = MPCController(self.horizon, self.x_dim, self.u_dim,
                                    self.u_min, self.u_max,
                                    self.Q, self.R, self.nominal_dynamics, bias=error_matrix[i, :])
            # solve the problem to obtain the input
            temp_u = temp_mpc.solve_input_traj(x_0)
            # store the input perturbation
            vec_sensitivity[i] = np.linalg.norm(u_nm - temp_u)

        # form the output
        err_u_mean = np.mean(vec_sensitivity) # mean
        err_u_max = np.max(vec_sensitivity) # max
        err_u_min = np.min(vec_sensitivity) # min
        err_u_std = np.std(vec_sensitivity) # variance

        # return the five important values as a dictionary
        out_dict = {"mean": err_u_mean, "max": err_u_max, "min": err_u_min,
                    "var_upper": err_u_mean + err_u_std, "var_lower": err_u_mean - err_u_std}

        return out_dict
    
    def sensitive_input_normed_state(self, e_para, norm_x):
        """
        The function computes the input sensitivity for a given perturbation level
        and a given state norm.

        Parameters:
            e_para: the norm of the parametric error
            norm_x: norm of the initial state
        """
        # generate the bias matrix
        error_matrix = generate_uniform_sphere_vectors(self.dim_para, e_para, self.num_vec)
        state_matrix = generate_uniform_sphere_vectors(self.x_dim, norm_x, self.num_state)
        # initialize the sensitivity vector
        vec_sensitivity = np.zeros((self.num_state, self.num_vec))
        for i in range(self.num_state):
            # compute the nominal input
            u_nm = self.mpc.solve_input_traj(state_matrix[i, :])

            # looping to get the input perturbation for each instance
            for j in range(self.num_vec):
                # define the MPC controller for a specific bias
                temp_mpc = MPCController(self.horizon, self.x_dim, self.u_dim,
                                        self.u_min, self.u_max,
                                        self.Q, self.R, self.nominal_dynamics, bias=error_matrix[j, :])
                # solve the problem to obtain the input
                temp_u = temp_mpc.solve_input_traj(state_matrix[i, :])
                # store the input perturbation
                vec_sensitivity[i, j] = np.linalg.norm(u_nm - temp_u)

        # form the output
        # err_u_mean = np.mean(vec_sensitivity) # mean
        err_u_max = np.max(vec_sensitivity) # max
        # err_u_min = np.min(vec_sensitivity) # min
        # err_u_std = np.std(vec_sensitivity) # variance

        # return the five important values as a dictionary
        # out_dict = {"mean": err_u_mean, "max": err_u_max, "min": err_u_min,
                    # "var_upper": err_u_mean + err_u_std, "var_lower": err_u_mean - err_u_std}

        return err_u_max

    def sensitive_value(self, e_para, x_0):
        """
        The function computes the value function sensitivity for a given perturbation and state.

        Parameters:
            e_para: the norm of the parametric error
            x_0: initial state
        """
        # generate the bias matrix
        error_matrix = generate_uniform_sphere_vectors(self.dim_para, e_para, self.num_vec)
        # initialize the sensitivity vector
        vec_value = np.zeros(self.num_vec)
        # compute the nominal value function
        value_nm = self.mpc.solve_value(x_0)

        # looping to get the input perturbation for each instance
        for i in range(self.num_vec):
            # define the MPC controller for a specific bias
            temp_mpc = MPCController(self.horizon, self.x_dim, self.u_dim,
                                    self.u_min, self.u_max,
                                    self.Q, self.R, self.nominal_dynamics, bias=error_matrix[i, :])
            # solve the problem to obtain the input and store it
            vec_value[i] = temp_mpc.solve_value(x_0)

        # obtain the final ratio vector
        vec_ratio = np.abs(vec_value - value_nm * np.ones(self.num_vec)) / vec_value

        # return the maximum value
        return np.max(vec_ratio)

    def sensitive_closed_loop(self, e_para, x_0):
        """
        This function simulates the final closed-loop sensitivity
        for a given perturbation and state.

        Parameters:
            e_para: the norm of the parametric error
            x_0: initial state
        """
        # generate the bias matrix
        error_matrix = generate_uniform_sphere_vectors(self.dim_para, e_para, self.num_vec)
        # initialize the sensitivity vector
        vec_value_closed_loop = np.zeros(self.num_vec)
        vec_value_open_loop = np.zeros(self.num_vec)
        # initialize the mpc_simulator
        mpc_sim = MPCSimulator(self.horizon, self.x_dim, self.u_dim,
                               self.u_min, self.u_max,
                               self.Q, self.R, self.nominal_dynamics)

        # looping to get the competitive ratio for each instance
        for i in range(self.num_vec):
            # simulate the closed-loop to get the practical MPC value
            vec_value_closed_loop[i] = mpc_sim.simulate_closed_loop(x_0, int(1e2), error_matrix[i, :])

            # simulate the open-loop to get the ideal MPC value
            vec_value_open_loop[i] = mpc_sim.simulate_infinite_horizon(x_0, error_matrix[i, :])

        vec_competitive_ratio = vec_value_closed_loop / vec_value_open_loop

        return np.max(vec_competitive_ratio)
