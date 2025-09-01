'''Test of the a single grid of SCHF'''
import jax
import jax.numpy as jnp
import numpy as np
import os
from itertools import product

from functions_parameters.jax_schf_kernel import schf_fixed_filling_pmap_over_u, schf_single_job
from functions_parameters.jax_schf_helpers import *
from functions_parameters.universal_parameters import a, b
from functions_parameters.bond_table_generator import build_buckets_per_shell

# ---- numeric mode ----
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)  # use float64 by default

Array = jnp.ndarray
PyTree = Any


'''
preparation:
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

# in this case, as both TB model and interaction, we include up to NNN, the a_list and deltas are the same
# calculate the correlation matrix
temperature = 4E-4
filling = 1/2
mu = 2
t_nn = 1
t_nnn = -0.025
t_arr = np.array([t_nn, t_nnn])

phase_pos, phase_neg = precompute_k_phase_tables(deltas, a, k_mesh_points)
Htb, e_all, v_all, v_all_dagger = hk_all_k_from_phases(mu, a_lists, t_arr, phase_neg)
dict_ref = prepare_reference_state(filling, a_lists, Htb, e_all, v_all, v_all_dagger, phase_pos, phase_neg, temperature)

'''
SCHF setup
'''
# read the basis input from the written file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_basis = np.load(os.path.join(project_root, 'functions_parameters', 'filling_1_rerun_basis_v1_v2.npz'))
input_d_tot = input_basis["d"]
input_d_tot = jnp.asarray(input_d_tot, dtype=jnp.complex128)
input_bond_tot = input_basis["bond"]
input_bond_tot = jnp.asarray(input_bond_tot, dtype=jnp.complex128)

# test explicitly for the neighboring point and see what is the output
input_d = input_d_tot[3]
input_bond = input_bond_tot[3]

u_arr = jnp.array([0.2])
v1_arr = jnp.array([0.75])
v2_arr = v1_arr.copy()
v_arr = jnp.concatenate((v1_arr, v2_arr))
nshell = v_arr.shape[0]
ndeltas = deltas.shape[0]
Htb = jnp.stack((jnp.asarray(Htb), jnp.asarray(Htb)), axis=1)
a_lists = jnp.asarray(a_lists)
phase_pos = jnp.asarray(phase_pos)
phase_neg = jnp.asarray(phase_neg)
# double the filling to get the correct number of electrons
filling = filling * 2

res = schf_single_job(Htb, a_lists, phase_pos, phase_neg, dict_ref, input_d, input_bond, filling, u_arr, v_arr, temperature)

