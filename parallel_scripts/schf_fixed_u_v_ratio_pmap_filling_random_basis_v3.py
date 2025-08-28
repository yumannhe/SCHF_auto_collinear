import jax
import jax.numpy as jnp
import numpy as np
import os
from itertools import product

from functions_parameters.jax_schf_kernel import schf_fixed_u_v_pair_pmap_over_filling, schf_single_job
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

nshell_tb = 2
radii, a_lists, deltas = build_buckets_per_shell(a, basis_frac, nshell_tb)

# in this case, as both TB model and interaction, we include up to NNN, the a_list and deltas are the same
# calculate the correlation matrix
temperature = 4E-4
mu = 2
t_nn = 1
t_nnn = -0.025
t_arr = np.array([t_nn, t_nnn])

phase_pos, phase_neg = precompute_k_phase_tables(deltas, a, k_mesh_points)
Htb, e_all, v_all, v_all_dagger = hk_all_k_from_phases(mu, a_lists, t_arr, phase_neg)

nshell_v = 3
radii_v, a_lists_v, deltas_v = build_buckets_per_shell(a, basis_frac, nshell_v)
phase_pos_v, phase_neg_v = precompute_k_phase_tables(deltas_v, a, k_mesh_points)
'''
SCHF parameters:
'''
# Get the project root directory (parent of parallel_scripts)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_d_tot = np.load(os.path.join(project_root, 'functions_parameters', 'random_basis_arr.npy'))
input_d_tot = input_d_tot/20.0
input_d_tot = jnp.asarray(input_d_tot, dtype=jnp.complex128)
num_channel = input_d_tot.shape[0]

num_filling_points = 20
filling_arr = -jnp.linspace(-1, 0, num_filling_points, endpoint=False)[::-1]
num_u_points = 16
u_arr = jnp.linspace(0, 0.75, num_u_points)
v1_arr = u_arr/2
v2_arr = u_arr/2
v3_arr = u_arr/2
v_arr = jnp.stack((v1_arr, v2_arr, v3_arr), axis=1)
ndeltas = deltas.shape[0]
input_bond_tot = jnp.zeros((num_channel, ndeltas, 2, nshell_v, norb, norb), dtype=jnp.complex128)
Htb = jnp.stack((jnp.asarray(Htb), jnp.asarray(Htb)), axis=1)
a_lists_v = jnp.asarray(a_lists_v)
phase_pos_v = jnp.asarray(phase_pos_v)
phase_neg_v = jnp.asarray(phase_neg_v)
# double the filling to get the correct number of electrons
filling_arr = filling_arr * 2

'''
SCHF parallel run:
'''
res = schf_fixed_u_v_pair_pmap_over_filling(schf_single_job, Htb, a_lists_v, e_all, v_all, v_all_dagger, phase_pos_v, phase_neg_v, input_d_tot, input_bond_tot, filling_arr, u_arr, v_arr, temperature)

host_res = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), res)

np.savez_compressed(
    os.path.join(project_root, "u_v1_v2_v3_t_4_em4_random_basis_mesh_60_pmap_over_filling.npz"),
    filling=np.asarray(filling_arr),
    u=np.asarray(u_arr),
    v=np.asarray(v_arr),
    d=host_res["d"],
    bond=host_res["bond"],
    e_diff=host_res["e_diff"],
    c_diff=host_res["c_diff"],
    gse=host_res["gse"],
    e_fermi=host_res["e_fermi"],
    any_bi_fail=host_res["any_bi_fail"],
    iters=host_res["iters"],
)

