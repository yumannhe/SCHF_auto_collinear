import sys
import os
# Add the project root to Python path for direct script execution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    # Move project_root to the front of sys.path
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

import numpy as np
from functions_parameters.schf import bond_orders, non_interacting_outputs, mean_field_h_k_independent, mean_field_u, k_dependent_bond_mean_field_h
from functions_parameters.universal_parameters import a, b
from functions_parameters.bond_table_generator import build_buckets_per_shell
from itertools import product

'''
parameters:
'''
# order is important. for sublattice a, b and c respectively
basis_frac = np.array([[1/2, 0], [1/2, 1/2], [0, 1/2]])
norb = basis_frac.shape[0]
num_k_mesh = 60
b_0 = np.linspace(-b[0] / 2, b[0] / 2, num_k_mesh, endpoint=False)
b_1 = np.linspace(-b[1] / 2, b[1] / 2, num_k_mesh, endpoint=False)
k_mesh_points = np.vstack([v1 + v2 for v1, v2 in product(b_0, b_1)])
num_k_points = k_mesh_points.shape[0]

radii, a_lists, deltas = build_buckets_per_shell(a, basis_frac, 2)

# calculate the correlation matrix
temperature = 4E-4
filling = 1/6
mu = 2
t_nn = 1
t_nnn = -0.025
t_arr = np.array([t_nn, t_nnn])

corr_k_arr, hk_o_arr, ground_state_e = non_interacting_outputs(k_mesh_points, deltas, a_lists, mu, t_arr, a, filling, temperature)

'''
obtain the bond orders and initial mean-field decompositions
'''
bond_orders_o = bond_orders(deltas, a_lists, corr_k_arr, k_mesh_points, a)
density_list = np.diag(np.sum(corr_k_arr, axis=0)/num_k_points)
h_mean_o_u = mean_field_u(np.stack((density_list, density_list)), 1)
# the factor 2 for density list indicate the spin degeneracy
h_mean_o_v2 = mean_field_h_k_independent(a_lists[:,1:2,:,:], np.array([1]), density_list*2)
h_mean_o_v1 = mean_field_h_k_independent(a_lists[:,0:1,:,:], np.array([1]), density_list*2)

h_mean_v1_arr = np.zeros((num_k_points, norb, norb), dtype=np.complex128)
h_mean_v2_arr = np.zeros((num_k_points, norb, norb), dtype=np.complex128)
e_u_o = 0
e_v1_o = 0
e_v2_o = 0
for i in range(num_k_points):
    k = k_mesh_points[i]
    h_mean_v1 = k_dependent_bond_mean_field_h(deltas, bond_orders_o[:,0:1,:,:], k, np.array([1]), a)
    h_mean_v2 = k_dependent_bond_mean_field_h(deltas, bond_orders_o[:,1:2,:,:], k, np.array([1]), a)
    h_mean_v1_arr[i] = h_mean_v1 + h_mean_o_v1
    h_mean_v2_arr[i] = h_mean_v2 + h_mean_o_v2
    # the multiplier factor 2 for density list indicate the spin degeneracy
    # the divider factor 2 for the trace is because of the double counting in the mean-field decomposition
    e_u_o += np.real(np.trace(h_mean_o_u[0].T @ corr_k_arr[i]))/2*2
    e_v1_o += np.real(np.trace(h_mean_v1_arr[i].T @ corr_k_arr[i]))/2*2
    e_v2_o += np.real(np.trace(h_mean_v2_arr[i].T @ corr_k_arr[i]))/2*2






