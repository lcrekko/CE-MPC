from nmpc.diverse_functions import cr_array_error 
import numpy as np

x = np.array([-1, 1])
e_seq = np.linspace(1e-3, 1e-2, 10)
n = 10
blfx = 1.005
blfu = 0.02


print(cr_array_error(x, e_seq, n, blfx, blfu))