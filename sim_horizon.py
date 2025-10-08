"""
This python script is used for verify the closed-loop performance when
varying the prediction horizon, which will be used to demonstrate both
Theorem 1 and Theorem 2.
"""
import numpy as np
import matplotlib.pyplot as plt
from nmpc.mpc_controller import MPCSimulator
from nmpc.plotter import MonteCarloPlotter
from nmpc.diverse_functions import nominal_dynamics, cr_array_horizon
from nmpc.utils import generate_uniform_sphere_vectors, default_color_generator
plt.rcParams.update({
    "text.usetex": True,                  # Use LaTeX for text rendering
    "font.family": "serif",               # Use a serif font
    "font.serif": ["Computer Modern Roman"]              # Set the font to Times New Roman or similar
})

# The two error norm
err_1 = 1e-3
err_2 = 1e-2
err_3 = 1e-1

# number of error vectors
num_vec = 100

# form the error matrix
vec_err_1 = generate_uniform_sphere_vectors(3, err_1, num_vec, seed=42)
vec_err_2 = generate_uniform_sphere_vectors(3, err_2, num_vec, seed=24)
vec_err_3 = generate_uniform_sphere_vectors(3, err_3, num_vec, seed=42)


# specify the horizon range
horizon_min = 5
horizon_max = 30
# generate the horizon vector
vec_horizon = np.arange(horizon_min, horizon_max + 1)  # to include the last horizon
num_horizon = horizon_max - horizon_min + 1

# -------- Define parameters for the MPC ---------
x_dim = 2   # state dimension
u_dim = 1   # input dimension

blfx = 1.005
blfu = 0.02

u_lim = 0.05 # input limit (more parameters are needed for multi-input systems)
u_min = np.array([-u_lim])
u_max = np.array([u_lim])

Q = np.eye(x_dim)  # state weight
R = np.eye(u_dim)  # input weight

# the initial state (we here consider only 1)
x0 = np.array([-1, 1])

# -------- Closed-loop simulation ----------
# initialize the output table
out_tab_1 = np.zeros((num_vec, num_horizon))
out_tab_2 = np.zeros((num_vec, num_horizon))
out_tab_3 = np.zeros((num_vec, num_horizon))

# Specify a uniform closed-loop simulation time
# T_closed_loop = 10

# for i in range(num_vec):
#     for j in range(num_horizon):
#         temp_simulator = MPCSimulator(vec_horizon[j],
#                                       x_dim, u_dim,
#                                       u_min, u_max,
#                                       Q, R, nominal_dynamics)
#         out_tab_1[i,j] = temp_simulator.simulate_closed_loop(x0, T_closed_loop, vec_err_1[i,:])
#         out_tab_2[i,j] = temp_simulator.simulate_closed_loop(x0, T_closed_loop, vec_err_2[i,:])
#         # out_tab_3[i,j] = temp_simulator.simulate_closed_loop(x0, T_closed_loop, vec_err_3[i,:])
#     print(f"{i+1}-th finished")

# np.save("cl_tab_1.npy", out_tab_1)
# np.save("cl_tab_2.npy", out_tab_2)
# np.save("new_tab_3.npy", out_tab_3) # reminder of the tension curve

# for i in range(num_vec):
#     for j in range(num_horizon):
#         temp_simulator = MPCSimulator(vec_horizon[j],
#                                       x_dim, u_dim,
#                                       u_min, u_max,
#                                       Q, R, nominal_dynamics)
#         out_tab_1[i,j] = temp_simulator.simulate_infinite_horizon(x0, vec_err_1[i,:])
#         out_tab_2[i,j] = temp_simulator.simulate_infinite_horizon(x0, vec_err_2[i,:])
#         # out_tab_3[i,j] = temp_simulator.simulate_closed_loop(x0, T_closed_loop, vec_err_3[i,:])
#     print(f"{i+1}-th finished")

# np.save("true_tab_1.npy", out_tab_1)
# np.save("true_tab_2.npy", out_tab_2)

# Remark: the table true_tab_i is redundant since it is infinite-horizon simulation

# ----------------- Bound Computation -----------------
# # compute the competitive ratio bound
# cr_n_1 = cr_array_horizon(x0, err_1, vec_horizon, blfx, blfu)
# cr_n_2 = cr_array_horizon(x0, err_2, vec_horizon, blfx, blfu)

# # load the true infinite-horizon performance
# true_tab_1 = np.load("true_tab_1.npy")
# true_tab_2 = np.load("true_tab_2.npy")

# # compute the performance bound
# bound_tab_1 = np.outer(true_tab_1[:,0], cr_n_1)
# bound_tab_2 = np.outer(true_tab_2[:,0], cr_n_2)

# np.save("bound_tab_1", bound_tab_1)
# np.save("bound_tab_2", bound_tab_2)

# ---------- Instance Identification -----------
# ini_tab_1 = np.load("new_tab_1.npy")
# ini_tab_2 = np.load("new_tab_2.npy")
# iut_tab_3 = np.load("new_tab_3.npy")

# indices_1 = np.where((out_tab_1.min(axis=1) < out_tab_1[:, -1]) & (out_tab_1[:, 0] > out_tab_1[:, -1]))[0]
# indices_2 = np.where((out_tab_2.min(axis=1) < out_tab_2[:, -1]) & (out_tab_2[:, 0] > out_tab_2[:, -1]))[0]

# indices_1 = np.where(out_tab_1[:, 0] > out_tab_1[:, -1])[0]
# indices_2 = np.where(out_tab_2[:, 0] < out_tab_2[:, -1])[0]

# print(indices_1)
# print(indices_2)

# ------- specific case simulation --------
# out_tab_1 = np.load("new_tab_1.npy")
# out_tab_2 = np.load("new_tab_2.npy")

# np.save("monotone_1.npy", out_tab_1)
# np.save("monotone_2.npy", out_tab_2)

# load the performance bound
bound_tab_1 = np.load("bound_tab_1.npy")
bound_tab_2 = np.load("bound_tab_2.npy")

# load the true performance
cl_tab_1 = np.load("cl_tab_1.npy")
cl_tab_2 = np.load("cl_tab_2.npy")

# -------- Plotting --------
fig_width = 8
gold_ratio = 0.5 * (np.sqrt(5) - 1)
fig_size = (fig_width, fig_width * gold_ratio)
tab_color = default_color_generator()
info_text = {"x_label": r'$N$',
             "y_label": r'$J_{{\infty}}(x;\mu_{{N,\hat{{\theta}}}},\theta^\ast)$',
             "legend": fr'$\varepsilon_\theta = {err_1}$'}
info_text2 = {"x_label": r'$N$',
             "y_label": r'$J_{{\infty}}(x;\mu_{{N,\hat{{\theta}}}},\theta^\ast)$',
             "legend": fr'$\varepsilon_\theta = {err_2}$'}
info_font = {"ft_type": "Computer Modern Roman",
             "ft_size_label": fig_width * 4, "ft_size_legend": fig_width * 4, "ft_size_tick": fig_width * 3}

# do the plotting
fig, ax = plt.subplots(1, 2, figsize=(fig_size[0] * 2, fig_size[1] * 1))
# myp_11 = MonteCarloPlotter(ax[0], vec_horizon, bound_tab_1,
#                           info_text, info_font, info_color=tab_color['C2'], 
#                           marker_type='*', marker=True)
# myp_11.plot_basic(set_x_ticks=True)
myp_12 = MonteCarloPlotter(ax[0], vec_horizon, cl_tab_1,
                          info_text, info_font, info_color=tab_color['C0'], marker=True)
myp_12.plot_basic(set_x_ticks=True)
# myp_21 = MonteCarloPlotter(ax[1], vec_horizon, bound_tab_2,
#                           info_text2, info_font, info_color=tab_color['C3'], 
#                           marker_type='*', marker=True)
# myp_21.plot_basic(set_x_ticks=True)
myp_22 = MonteCarloPlotter(ax[1], vec_horizon, cl_tab_2,
                          info_text2, info_font, info_color=tab_color['C1'], marker=True)
myp_22.plot_basic(set_x_ticks=True)
plt.tight_layout()
plt.savefig("horizon_perturbation_multi_traj.pdf", format="pdf",
            dpi=800, bbox_inches='tight', pad_inches=0.3)
plt.show()

