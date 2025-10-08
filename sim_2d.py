"""
This python script is used for verify that the input perturbation admits
a linear upper bound of the joint product of state norm and parametric perturbation norm.

The result first serves as a solid verifation of Assumption 5 and Lemma 2

The result is a solid demonstration of Assumption 7, Corollary 6, and Proposition 5,
which further validates the existence of a competitive ratio bound as in Theorem 2
"""
import numpy as np
import matplotlib.pyplot as plt
from nmpc.mpc_controller import MPCController
from nmpc.plotter import MonteCarloPlotter
from nmpc.diverse_functions import nominal_dynamics
from nmpc.utils import generate_multiple_sphere_vectors, default_color_generator
plt.rcParams.update({
    "text.usetex": True,                  # Use LaTeX for text rendering
    "font.family": "serif",               # Use a serif font
    "font.serif": ["Computer Modern Roman"]              # Set the font to Times New Roman or similar
})

num_vec = 100
num_err = 10
e_seq = np.linspace(1e-3, 1e-2, num_err)
error = generate_multiple_sphere_vectors(3, e_seq, num_vec)

# -------- Define parameters for the MPC ---------
horizon = 10  # prediction horizon

x_dim = 2   # state dimension
u_dim = 1   # input dimension

u_lim = 0.05 # input limit (more parameters are needed for multi-input systems)
umin = np.array([-u_lim])
umax = np.array([u_lim])

Q = np.eye(x_dim)  # state weight
R = np.eye(u_dim)  # input weight

# -------- Nominal MPC ---------
# No bias added
mpc = MPCController(horizon, x_dim, u_dim, umin, umax, Q, R, nominal_dynamics)

# Define an initial state (example)
x0 = np.array([-1, 1])
x0_2 = np.array([-0.2, 0.2])

u_nm = mpc.solve_input_traj(x0)
u_nm2 = mpc.solve_input_traj(x0_2)

# -------- Sensitivity Analysis --------
out_tab = np.zeros((num_vec, num_err))
out_tab2 = np.zeros((num_vec, num_err))

# for i in range(num_err):
#     for j in range(num_vec):
#         temp_mpc = MPCController(horizon, x_dim, u_dim, umin, umax,
#                                  Q, R, nominal_dynamics, bias=error[j,:,i])
#         temp_out = temp_mpc.solve_input_traj(x0)
#         temp_out2 = temp_mpc.solve_input_traj(x0_2)

#         out_tab[j, i] = np.linalg.norm(temp_out - u_nm)
#         out_tab2[j, i] = np.linalg.norm(temp_out2 - u_nm2)

# np.save("table_2d.npy", out_tab)
# np.save("table_2d_2.npy", out_tab2)

out_tab = np.load("table_2d.npy")
out_tab2 = np.load("table_2d_2.npy")


# -------- Plotting --------
fig_width = 8
gold_ratio = 0.5 * (np.sqrt(5) - 1)
fig_size = (fig_width, fig_width * gold_ratio)
tab_color = default_color_generator()
info_text = {"x_label": r'$\varepsilon_{\theta} = ||\theta^\ast - \hat{\theta}||$',
             "y_label": r'$||\mathbf{u}^\star_N(\theta^\ast) - \mathbf{u}^\star_N(\hat{\theta})||$',
             "legend": r'$x = [-1,1]^\top$'}
info_text2 = {"x_label": r'$\varepsilon_{\theta} = ||\theta^\ast - \hat{\theta}||$',
             "y_label": r'$||\mathbf{u}^\star_N(\theta^\ast) - \mathbf{u}^\star_N(\hat{\theta})||$',
             "legend": r'$x = [0.2,-0.2]^\top$'}
info_font = {"ft_type": "Computer Modern Roman",
             "ft_size_label": fig_width * 4, "ft_size_legend": fig_width * 4, "ft_size_tick": fig_width * 3}

# do the plotting
fig, ax = plt.subplots(1, 2, figsize=(fig_size[0] * 2, fig_size[1] * 1))
myp_1 = MonteCarloPlotter(ax[0], e_seq, out_tab,
                          info_text, info_font, tab_color['C0'], marker=True)
myp_1.plot_basic(set_x_ticks=True)
myp_2 = MonteCarloPlotter(ax[1], e_seq, out_tab2,
                          info_text2, info_font, tab_color['C1'], marker=True)
myp_2.plot_basic(set_x_ticks=True)
plt.tight_layout()
plt.savefig("linear_input_perturbation.pdf", format="pdf",
            dpi=800, bbox_inches='tight', pad_inches=0.3)
plt.show()
