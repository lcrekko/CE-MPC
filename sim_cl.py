"""
This python script is used for verication of the competitive ratio

The result is a direct and solid demonstration of Theorem 2
"""
import numpy as np
import matplotlib.pyplot as plt
from nmpc.mpc_controller import MPCSensitivity
from nmpc.diverse_functions import nominal_dynamics
from nmpc.plotter import MonteCarloPlotter
from nmpc.utils import generate_uniform_sphere_vectors
# # Set the font family to serif and choose Times New Roman
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']

# # If you are using LaTeX for text rendering, enable it and load a Times package:
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'

plt.rcParams.update({
    "text.usetex": True,                  # Use LaTeX for text rendering
    "font.family": "serif",               # Use a serif font
    "font.serif": ["Computer Modern Roman"]              # Set the font to Times New Roman or similar
})

# ------------ Basic Simulation Settings -------------
num_vec = 100 # number of monte carlo simulation
num_err = 10 # number of errors
num_state = 10 # number of states

# vector of different error norms
e_seq = np.linspace(1e-3, 1e-2, num_err)

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
# (uncomment the following part to generate new data)
mpc_sensitivity = MPCSensitivity(mpc_info, horizon, nominal_dynamics, num_vec, num_state)

# Define an initial state (example)
norm_x0 = np.sqrt(2)
norm_x0_2 = np.array(2) / 25

state_vec = generate_uniform_sphere_vectors(x_dim, norm_x0, num_state)
state_vec_2 = generate_uniform_sphere_vectors(x_dim, norm_x0_2, num_state)

# cr_table = np.zeros((num_state, num_err))
# cr_table_2 = np.zeros((num_state, num_err))

# for i in range(num_state):
#     for j in range(num_err):
#         cr_table[i,j] = mpc_sensitivity.sensitive_closed_loop(e_seq[j], state_vec[i,:])
#         cr_table_2[i,j] = mpc_sensitivity.sensitive_closed_loop(e_seq[j], state_vec_2[i,:])
#     print(f"the {i}-th round is finished")

# np.save("table_x0_cr.npy", cr_table)
# np.save("table_x0_2_cr.npy", cr_table_2)

# here we directly load the pre-generated data
cr_table = np.load("table_x0_cr.npy")
cr_table_2 = np.load("table_x0_2_cr.npy")

# -------- Plotting --------
fig_width = 8
gold_ratio = 0.5 * (np.sqrt(5) - 1)
fig_size = (fig_width, fig_width * gold_ratio)
info_text = {"x_label": r'$\varepsilon_{\theta} = \|\theta - \hat{\theta}\|$',
             "y_label": r'$\mathcal{R}_{\mathrm{cr},N}(\varepsilon_{\theta})$',
             "legend": r'$\|x\| = \sqrt{2}$'}
info_text2 = {"x_label": r'$\varepsilon_{\theta} = \|\theta - \hat{\theta}\|$',
             "y_label": r'$\mathcal{R}_{\mathrm{cr},N}(\varepsilon_{\theta})$',
             "legend": r'$\|x\| = \sqrt{0.08}$'}
info_font = {"ft_type": "Computer Modern Roman",
             "ft_size_label": fig_width * 4, "ft_size_legend": fig_width * 4, "ft_size_tick": fig_width * 3}

# do the plotting
fig, ax = plt.subplots(1, 2, figsize=(fig_size[0] * 2, fig_size[1] * 1))
myp_1 = MonteCarloPlotter(ax[0], e_seq, cr_table_2,
                          info_text, info_font, info_color=(0.15, 0.15, 0.15), marker=True)
myp_1.plot_basic_multi_traj(set_x_ticks=True)
myp_2 = MonteCarloPlotter(ax[1], e_seq, cr_table,
                          info_text2, info_font, info_color=(0.15, 0.15, 0.15), marker=True)
myp_2.plot_basic_multi_traj(set_x_ticks=True)
plt.tight_layout()
plt.savefig("competitive_ratio_multi_traj.pdf", format="pdf",
            dpi=800, bbox_inches='tight', pad_inches=0.3)
plt.show()
