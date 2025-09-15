import numpy as np
import numpy.linalg as la
from itertools import product
from functions_parameters.bond_table_generator import build_buckets_per_shell
from functions_parameters.jax_schf_helpers import precompute_k_phase_tables, hk_all_k_from_phases

"""
most universal parameters
"""
# lattice vector
# convention: each row is a lattice vector in Cartesian coordinate systems
a = np.array([[5.279, 0],
              [-2.6395, 4.571748]])
b = (la.inv(a) * 2 * np.pi).T

# including Pauli matrices
sigma_0 = np.array([[1.0, 0], [0, 1.0]], np.complex128)
sigma_1 = np.array([[0, 1.0], [1.0, 0]], np.complex128)
sigma_2 = np.array([[0, -1.0 * 1j], [1.0 * 1j, 0]], np.complex128)
sigma_3 = np.array([[1.0, 0], [0, -1.0]], np.complex128)
pauli_matrices = np.stack([sigma_0, sigma_1, sigma_2, sigma_3])

temperature = 4E-4


"""
The TB parameters
"""
# document the TB parameters
mu = 2
t_nn = 1
t_nnn = -0.025
t_arr = np.array([t_nn, t_nnn])

# ----------------------------document Htb in 1*1 unit cell----------------------------
a1_basis = np.ones(3)/np.sqrt(3)
e2_basis1 = np.array([-1, 2, -1]) / np.sqrt(6)
e2_basis2 = np.array([-1, 0, 1]) / np.sqrt(2)

# document k_mesh and basis frac for 1*1 unit cell
# order is important. for sublattice a, b and c respectively
basis_frac = np.array([[1/2, 0], [1/2, 1/2], [0, 1/2]])
norb = basis_frac.shape[0]
num_k_mesh = 60
b_0 = np.linspace(-b[0] / 2, b[0] / 2, num_k_mesh, endpoint=False)
b_1 = np.linspace(-b[1] / 2, b[1] / 2, num_k_mesh, endpoint=False)
k_mesh_points = np.vstack([v1 + v2 for v1, v2 in product(b_0, b_1)])
num_k_points = k_mesh_points.shape[0]

radii, a_lists, deltas = build_buckets_per_shell(a, basis_frac, t_arr.shape[0])
phase_pos, phase_neg = precompute_k_phase_tables(deltas, a, k_mesh_points)
Htb, e_all, v_all, v_all_dagger = hk_all_k_from_phases(mu, a_lists, t_arr, phase_neg)

# ----------------------------document Htb in 2*2 unit cell----------------------------
# order is important. for unit cell (0,0), (0,1), (1,0), (1,1), sublattice a, b and c respectively
basis_frac_o = basis_frac/2
basis_frac_1 = basis_frac_o + np.array([0, 1/2])
basis_frac_2 = basis_frac_o + np.array([1/2, 0])
basis_frac_3 = basis_frac_o + np.array([1/2, 1/2])
basis_frac_2_2 = np.concatenate((basis_frac_o, basis_frac_1, basis_frac_2, basis_frac_3), axis=0)
norb_2_2 = basis_frac_2_2.shape[0]


num_k_mesh_2_2 = num_k_mesh//2
a_2_2 = a*2
b_2_2 = b/2
b_0_2_2 = np.linspace(-b_2_2[0] / 2, b_2_2[0] / 2, num_k_mesh_2_2, endpoint=False)
b_1_2_2 = np.linspace(-b_2_2[1] / 2, b_2_2[1] / 2, num_k_mesh_2_2, endpoint=False)
k_mesh_points_2_2 = np.vstack([v1 + v2 for v1, v2 in product(b_0_2_2, b_1_2_2)])
num_k_points_2_2 = k_mesh_points_2_2.shape[0]
radii_2_2, a_lists_2_2, deltas_2_2 = build_buckets_per_shell(a_2_2, basis_frac_2_2, t_arr.shape[0])
phase_pos_2_2, phase_neg_2_2 = precompute_k_phase_tables(deltas_2_2, a_2_2, k_mesh_points_2_2)
Htb_2_2, e_all_2_2, v_all_2_2, v_all_dagger_2_2 = hk_all_k_from_phases(mu, a_lists_2_2, t_arr, phase_neg_2_2)

"""
For phase diagram analysis
"""
# universal parameters for phase diagram analysis
threshold = 1E-2

# translation symmetry check
translation_a1 = np.kron(np.array([[0,0,1,0],
                           [0,0,0,1],
                           [1,0,0,0],
                           [0,1,0,0]]), np.eye(3))
translation_a2 = np.kron(np.array([[0,1,0,0],
                           [1,0,0,0],
                           [0,0,0,1],
                           [0,0,1,0]]), np.eye(3))
translation_a3 = translation_a1 @ translation_a2

# rotation symmetry check
# ------------------------1*1 unit cell------------------------
c_6_uc = np.array([[0,0,1],
                   [1,0,0],
                   [0,1,0]])
c_3_uc = c_6_uc@c_6_uc
c_2_uc = c_6_uc @ c_3_uc
c_6_uc = c_6_uc[None, :, :]
c_3_uc = c_3_uc[None, :, :]
c_2_uc = c_2_uc[None, :, :]


# ------------------------2*2 unit cell------------------------
c_6_2uc_o = np.zeros((12, 12))
c_6_2uc_o[0, 2] = 1
c_6_2uc_o[1, 6] = 1
c_6_2uc_o[2, 7] = 1
c_6_2uc_o[3, 8] = 1
c_6_2uc_o[4, 0] = 1
c_6_2uc_o[5, 1] = 1
c_6_2uc_o[6, -1] = 1
c_6_2uc_o[7, 3] = 1
c_6_2uc_o[8, 4] = 1
c_6_2uc_o[9, 5] = 1
c_6_2uc_o[10, -3] = 1
c_6_2uc_o[11, -2] = 1

c_6_2uc_o = np.moveaxis(np.reshape(c_6_2uc_o, (2, 2, 3, 2, 2, 3)), (0, 3), (1, 4)).reshape(12, 12)
c_3_2uc_o = c_6_2uc_o @ c_6_2uc_o
c_2_2uc_o = c_3_2uc_o @ c_6_2uc_o

c_6_2uc_gamma = np.zeros((12, 12))
c_6_2uc_gamma[1, 0] = 1
c_6_2uc_gamma[2, 1] = 1
c_6_2uc_gamma[6, 2] = 1
c_6_2uc_gamma[7, 3] = 1
c_6_2uc_gamma[8, 4] = 1
c_6_2uc_gamma[0, 5] = 1
c_6_2uc_gamma[10, 6] = 1
c_6_2uc_gamma[11, 7] = 1
c_6_2uc_gamma[3, 8] = 1
c_6_2uc_gamma[4, 9] = 1
c_6_2uc_gamma[5, 10] = 1
c_6_2uc_gamma[9, 11] = 1

c_3_2uc_gamma = c_6_2uc_gamma @ c_6_2uc_gamma
c_2_2uc_gamma = c_3_2uc_gamma @ c_6_2uc_gamma


c_6_2uc_y=np.zeros((12, 12))
c_6_2uc_y[10, 0] = 1
c_6_2uc_y[11, 1] = 1
c_6_2uc_y[3, 2] = 1
c_6_2uc_y[4, 3] = 1
c_6_2uc_y[5, 4] = 1
c_6_2uc_y[9, 5] = 1
c_6_2uc_y[1, 6] = 1
c_6_2uc_y[2, 7] = 1
c_6_2uc_y[6, 8] = 1
c_6_2uc_y[7, 9] = 1
c_6_2uc_y[8, 10] = 1
c_6_2uc_y[0, 11] = 1

c_3_2uc_y = c_6_2uc_y @ c_6_2uc_y
c_2_2uc_y = c_3_2uc_y @ c_6_2uc_y


c_6_2uc_x=np.zeros((12, 12))
c_6_2uc_x[4, 0] = 1
c_6_2uc_x[5, 1] = 1
c_6_2uc_x[9, 2] = 1
c_6_2uc_x[10, 3] = 1
c_6_2uc_x[11, 4] = 1
c_6_2uc_x[3, 5] = 1
c_6_2uc_x[7, 6] = 1
c_6_2uc_x[8, 7] = 1
c_6_2uc_x[0, 8] = 1
c_6_2uc_x[1, 9] = 1
c_6_2uc_x[2, 10] = 1
c_6_2uc_x[6, 11] = 1

c_3_2uc_x = c_6_2uc_x @ c_6_2uc_x
c_2_2uc_x = c_3_2uc_x @ c_6_2uc_x

c_6_2_uc = np.stack((c_6_2uc_o, c_6_2uc_x, c_6_2uc_y, c_6_2uc_gamma), axis=0)
c_3_2_uc = np.stack((c_3_2uc_o, c_3_2uc_x, c_3_2uc_y, c_3_2uc_gamma), axis=0)
c_2_2_uc = np.stack((c_2_2uc_o, c_2_2uc_x, c_2_2uc_y, c_2_2uc_gamma), axis=0)

ts_phase = ['', 'stripe', '2*2']
nematic_phase = [' R symmetric', ' C3 charge nematic', ' C2 charge nematic', ' no R']
magnetism_phase = [' +', ' FM', ' AFri']

phase_tot = list(i+j+m for i,j,m in product(ts_phase, nematic_phase, magnetism_phase))

