"""
This python script is used for verify that the input perturbation admits
a linear upper bound of the joint product of state norm and parametric perturbation norm.

The result is a solid demonstration of Assumption 7, Corollary 6, and Proposition 5,
which further validates the existence of a competitive ratio bound as in Theorem 2
"""
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from nmpc.mpc_controller import MPCSensitivity
from nmpc.diverse_functions import nominal_dynamics, inverse_level_set
from nmpc.plotter import ManifoldPlotter
plt.rcParams.update({
    "text.usetex": True,                  # Use LaTeX for text rendering
    "font.family": "serif",               # Use a serif font
    "font.serif": ["Computer Modern Roman"]              # Set the font to Times New Roman or similar
})

# ------------ Basic Simulation Settings -------------
num_vec = 100 # number of monte carlo simulation
num_err = 10 # number of errors
num_level = 10 # number of levels
num_state = 10 # number of states
norm_x_max = 1

# vector of different error norms
e_seq = np.linspace(1e-3, 1e-2, num_err)

# vector of different level set
r_vec = np.linspace(1e-4, 1e-3, num_level)

# ------------ The MPC Settings -------------
horizon = 10  # prediction horizon

x_dim = 2   # state dimension
u_dim = 1   # input dimension

u_lim = 0.05 # input limit (more parameters are needed for multi-input systems)
u_min = np.array([-u_lim])
u_max = np.array([u_lim])

Q = np.eye(x_dim)  # state weight
R = np.eye(u_dim)  # input weight

# form the MPC information dictionary
mpc_info = {"x_dim": x_dim, "u_dim": u_dim,
            "u_min": u_min, "u_max": u_max,
            "Q": Q, "R": R, "dim_para": 3}

# ----------- Sensitivity Simulation -------------
# Uncomment the following lines to generate new data
# mpc_sensitivity = MPCSensitivity(mpc_info, horizon, nominal_dynamics, num_vec, num_state)

# surface_plotter = ManifoldPlotter(r_vec, e_seq,
#                                   inverse_level_set, mpc_sensitivity.sensitive_input_normed_state,
#                                   norm_x_max)
# dump(surface_plotter, "class_surface_plotter.joblib")

# -------- Use the saved class for fast simulation -------------
surface_plotter = load("class_surface_plotter.joblib")

# ----------- Plotter info & Plotting -------------
fig_width = 6
gold_ratio = 0.5 * (np.sqrt(5) - 1)
info_text = {"x_label": r'$\varepsilon_\theta = \|\theta^\ast - \hat{\theta}\|$',
             "y_label": r'$\|x\|$',
             "legend": r'$||\mathbf{u}^\star_N(\theta^\ast) - \mathbf{u}^\star_N(\hat{\theta})||$'}
info_font = {"ft_type": "Computer Modern Roman",
             "ft_size_label": fig_width * 4,
             "ft_size_legend": fig_width * 4,
             "ft_size_tick": fig_width * 3}

# Create the figure and 3D axis
fig = plt.figure(figsize=(fig_width * 1.2, fig_width))
ax = fig.add_subplot(111, projection='3d')

surface_plotter.plotter_basic(ax, info_text, info_font)
plt.tight_layout()
plt.savefig("joint_input_state_3D_plot.pdf", format="pdf",
            dpi=800, bbox_inches='tight', pad_inches=0.8)
plt.show()
