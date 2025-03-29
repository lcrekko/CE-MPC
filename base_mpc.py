from nmpc.diverse_functions import nominal_dynamics
from nmpc.mpc_controller import MPCController, MPCSimulator
import numpy as np
import matplotlib.pyplot as plt
# Set the font family to serif and choose Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# If you are using LaTeX for text rendering, enable it and load a Times package:
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

# -------- Define parameters for the MPC ---------
horizon = 50  # prediction horizon

x_dim = 2   # state dimension
u_dim = 1   # input dimension

u_lim = 0.05 # input limit (more parameters are needed for multi-input systems)
umin = np.array([-u_lim]) 
umax = np.array([u_lim])

Q = np.eye(x_dim)  # state weight
R = np.eye(u_dim)  # input weight

# bias = np.array([0.02, -0.01, 0.04]) # parametric system bias

# ------ Open-loop Simulation ------

# With bias specified
# mpc = MPCController(horizon, x_dim, u_dim, umin, umax, Q, R, nominal_dynamics, bias)

# No bias specified
mpc = MPCController(horizon, x_dim, u_dim, umin, umax, Q, R, nominal_dynamics)

# Define an initial state (example)
x0 = np.array([-1, 1])

# Solve for the control input and trajectories
out_mpc = mpc.solve_open(x0)
print("Optimal first control input:", out_mpc["u_0"])
print("The final open-loop sate:", out_mpc["x_N"])
print("The value of the open-loop MPC value function:", out_mpc["V_N"])

# Extract trajectories
x_traj = out_mpc["x_traj"]  # Shape: (x_dim, horizon+1)
u_traj = out_mpc["u_traj"]  # Shape: (u_dim, horizon)
# Set the time index
time_steps_input = np.arange(horizon)


# ------ Closed-loop Simulation ------

# Set the bias
bias_nominal = np.zeros(3) # parametric system bias
bias_true = np.array([0.02, -0.01, 0.04]) # the true model

# Initialize the MPC closed-loop simulator
mpc_cl = MPCSimulator(horizon, x_dim, u_dim, umin, umax, Q, R, nominal_dynamics)

# ---- Set other parameters ----
T = 20 # Simulation horizon

# Simulate
out_mpc_cl = mpc_cl.simulate_closed_loop(x0, T, bias_true, mode="COMPLETE")

# ---- Extract the output and figure
print("The final closed-loop sate:", out_mpc_cl["x_T"])
print("The value of the closed-loop MPC value function:", out_mpc_cl["V_T"])

# Extract trajectories
x_traj_cl = out_mpc_cl["x_traj"]  # Shape: (x_dim, horizon+1)
u_traj_cl = out_mpc_cl["u_traj"]  # Shape: (u_dim, horizon)
# Set the time index
time_steps_input_cl = np.arange(T)

# Plotting parameters
Fig_h = 6
Fig_w = Fig_h * (np.sqrt(5) + 1) * 0.5
fontsize_label = 16
fontsize_title = 18
fontsize_legend = 16

# Turn on interactive mode
# plt.ion()

# Plot Input Trajectory in a separate figure
plt.figure(figsize=(Fig_w, Fig_h), num="Open-loop Input")
plt.plot(time_steps_input, u_traj, 'bo-', markersize=8, label='Input')
plt.xlabel(r'MPC Time Step $k$', fontsize=fontsize_label)
plt.ylabel(r'$u$', fontsize=fontsize_label)
plt.title('Input Trajectory', fontsize=fontsize_title)
plt.legend(fontsize = fontsize_legend)
plt.grid(True)

# Plot State Trajectory in a separate figure
plt.figure(figsize=(Fig_h, Fig_h), num="Open-loop State")
plt.plot(x_traj[0, :], x_traj[1, :], 'rs-', markersize=8, label='State Trajectory')
plt.xlabel(r'$x_1$', fontsize=fontsize_label)
plt.ylabel(r'$x_2$', fontsize=fontsize_label)
plt.title('State Trajectory', fontsize=fontsize_title)
plt.legend(fontsize = fontsize_legend)
plt.grid(True)

# Compute the common limit based on the data
x_min, x_max = np.min(x_traj[0, :]), np.max(x_traj[0, :])
y_min, y_max = np.min(x_traj[1, :]), np.max(x_traj[1, :])
common_limit = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

# Set x and y limits to be equal
plt.xlim([-common_limit, common_limit])
plt.ylim([-common_limit, common_limit])
plt.axis('equal')


# Plot Input Trajectory in a separate figure
plt.figure(figsize=(Fig_w, Fig_h), num="Closed-loop Input")
plt.plot(time_steps_input_cl, u_traj_cl[0, :], 'bo-', markersize=8, label='Input')
plt.xlabel(r'Control Time Step $k$', fontsize=fontsize_label)
plt.ylabel(r'$u$', fontsize=fontsize_label)
plt.title('Closed-loop Input Trajectory', fontsize=fontsize_title)
plt.legend(fontsize = fontsize_legend)
plt.grid(True)

# Plot State Trajectory in a separate figure
plt.figure(figsize=(Fig_h, Fig_h), num="Closed-loop State")
plt.plot(x_traj_cl[0, :], x_traj_cl[1, :], 'rs-', markersize=8, label='State Trajectory')
plt.xlabel(r'$x_1$', fontsize=fontsize_label)
plt.ylabel(r'$x_2$', fontsize=fontsize_label)
plt.title('Closed-loop State Trajectory', fontsize=fontsize_title)
plt.legend(fontsize = fontsize_legend)
plt.grid(True)

# Compute the common limit based on the data
x_min, x_max = np.min(x_traj_cl[0, :]), np.max(x_traj_cl[0, :])
y_min, y_max = np.min(x_traj_cl[1, :]), np.max(x_traj_cl[1, :])
common_limit = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

# Set x and y limits to be equal
plt.xlim([-common_limit, common_limit])
plt.ylim([-common_limit, common_limit])
plt.axis('equal')

plt.show()

