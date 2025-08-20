import numpy as np
from functools import reduce
import numpy.linalg as la
try:
    from .universal_parameters import pauli_matrices
except ImportError:
    from universal_parameters import pauli_matrices
try:
    from .tools import fermi_level_bisection, fermi_dirac_electron_count
except ImportError:
    from tools import fermi_level_bisection, fermi_dirac_electron_count
from itertools import product
import time


'''
--------- convention of bond orders: ------------
the bond orders are defined as c_r_i_d*c_(r+dr)_j


--------- convention of FT: ------------
c_r_d = \sum_k c_k_d e^{ikr}/sqrt(N)
c_k_d = \sum_r c_r_d e^{-ikr}/sqrt(N)
where N is the number of lattice sites.
This is because the FT of a real function is Hermitian.

# Hamiltonian:
h_r_dr_ij = t_ij*c_r_i_d*c_(r+dr)_j
h_k_ij = \sum_dr h_r_dr_ij e^{-ikdr} c_k_i_d*c_k_j

# Correlation matrix:
corr_dr_ij = c_r_dr_ij*c_r_i_d*c_(r+dr)_j
           = \sum_k corr_k_ij e^{-ikdr}/N
corr_k_ij = c_k_ij*c_k_i_d*c_k_j

--------- note: ------------
Since we define bond orders as c_r_i_d*c_(r+dr)_j, the mean-field "companion" shoud have the opposite order
as c_(r+dr)_j_d*c_r_i, which should have opposite Fourier phase factor 
'''

def bond_orders(deltas, a_list, corr_k, k_mesh_points, a):
    '''
    obtain the bond orders magnitude from correlation matrix
    deltas: (L, 2)
    a_list: (L, nshells, norb, norb)
    corr_k: (N, norb, norb)
    k_mesh_points: (N, 2)
    a: (2, 2)
    return: bond_orders: (L, nshells, norb, norb)
    '''
    bond_orders = np.zeros((a_list.shape), dtype = np.complex128)
    for i in range(deltas.shape[0]):
        delta = deltas[i]
        dr = delta@a
        corr_i = np.sum(corr_k*np.exp(-1j*k_mesh_points@dr)[:,np.newaxis,np.newaxis], axis=0)/k_mesh_points.shape[0]
        for j in range(a_list.shape[1]):
            bond_orders[i,j] = corr_i*a_list[i,j]
    return bond_orders


def k_dependent_bond_mean_field_h(deltas, bond_orders, k, v_arr, a):
    '''
    calculate the k-dependent mean-field Hamiltonian from bond orders
    deltas: (L, 2)
    bond_orders: (L, nshells, norb, norb)
    k: (2,)
    v_arr: (nshells,)
    a: (2, 2)
    return: h_mean: (norb, norb)
    '''
    # Check if bond_orders dim1 and v_arr dim0 are the same
    assert bond_orders.shape[1] == v_arr.shape[0], f"Dimension mismatch: bond_orders.shape[1]={bond_orders.shape[1]} != v_arr.shape[0]={v_arr.shape[0]}"

    bond_orders_k_d = np.zeros((bond_orders.shape[0], bond_orders.shape[2],bond_orders.shape[3]), dtype = np.complex128)
    for i in range(v_arr.shape[0]):
        bond_orders_k_d += np.transpose(bond_orders[:,i,:,:]*v_arr[i], (0,2,1))
    drs = deltas@a
    bond_order_k = np.sum(bond_orders_k_d*np.exp(1j*drs@k)[:,np.newaxis,np.newaxis], axis=0)
    # return the opposite value as Fock terms should have a negative sign
    return -bond_order_k


def mean_field_h_k_independent(a_list, v_arr, density_list):
    '''
    from bond orders and density list, calculate the k-independent mean-field Hamiltonian
    a_list: (L, nshells, norb, norb)
    v_arr: (nshells,)
    density_list: (norb,)
    return: h_mean: (norb, norb)
    '''
    h_mean_diagonal = np.zeros_like(density_list)
    for i in range(a_list.shape[0]):
        for j in range(a_list.shape[1]):
            h_mean_diagonal += a_list[i,j,:,:]@(v_arr[j]*density_list)
    h_mean = np.diag(h_mean_diagonal)
    return h_mean


def mean_field_u(density_s, u):
    '''
    calculate the mean-field interaction term
    density_s: (2, norb)
    u: float
    return: h_mean: (2, norb, norb)
    '''
    h_mean = np.stack((np.diag(u*density_s[1]), np.diag(u*density_s[0])))
    return h_mean


def hk(mu, t_arr, deltas, a_list, k, a_lattice):
    '''
    calculate the k-dependent Hamiltonian based on the bond orders
    mu: on-site potential
    t_arr: (nshells,)
    deltas: (L, 2)
    a_list: (L, nshells, norb, norb)
    k: (2,)
    a_lattice: (2, 2)
    return: h_k: (norb, norb)
    '''
    norb = a_list.shape[-1]
    # we assume same mu for all orb as this is true for symmetric kagome
    h_k = np.eye(norb, dtype=np.complex128)*mu
    drs = deltas@a_lattice
    h_k += np.sum(np.exp(-1j*drs@k)[:,np.newaxis,np.newaxis]*np.sum(a_list*t_arr[np.newaxis,:,np.newaxis,np.newaxis], axis=1), axis=0)
    return h_k


def non_interacting_outputs(k_mesh_points, deltas, a_list, mu, t_arr, a_lattice, filling, temperature):
    '''
    calculate the non-interacting outputs
    k_mesh_points: (N, 2)
    deltas: (L, 2)
    a_list: (L, nshells, norb, norb)
    mu: on-site potential
    a_lattice: (2, 2)
    filling: float
    temperature: float
    return: eigvals: (N, norb)
    '''
    num_k_points = k_mesh_points.shape[0]
    norb = a_list.shape[-1]
    corr_k_arr = np.zeros((num_k_points, norb, norb), dtype=np.complex128)
    hk_o_arr = np.zeros((num_k_points, norb, norb), dtype=np.complex128)
    e_dic = np.zeros((num_k_points, norb))
    v_dic = np.zeros((num_k_points, norb, norb), dtype=np.complex128)
    for i in range(num_k_points):
        k = k_mesh_points[i]
        h_k = hk(mu, t_arr, deltas, a_list, k, a_lattice)
        hk_o_arr[i] = h_k
        eigvals, eigvecs = la.eigh(h_k)
        e_dic[i] = eigvals
        v_dic[i] = eigvecs
        total_e = np.sort(e_dic.reshape(-1))
    e_fermi_iterated = fermi_level_bisection(total_e, filling, temperature)
    ground_state_e = 0
    for m in range(num_k_points):
        k = k_mesh_points[m]
        e_kpoint = e_dic[m]
        v_kpoint = v_dic[m]
        electron_count = fermi_dirac_electron_count(e_kpoint, e_fermi_iterated, temperature)
        corr_k = np.transpose(v_kpoint @ np.diag(electron_count) @ v_kpoint.conj().T)
        corr_k_arr[m] = corr_k
        # this doesn't cound the spin degeneracy
        ground_state_e += np.real(np.trace(hk_o_arr[m].T @ corr_k))
    return corr_k_arr, hk_o_arr, ground_state_e

def collinear_schf_iteration(input_d, input_bond, deltas, a_list, k_points, a_lattice, h_k_o, h_mean_initial_u, h_mean_initial_v_arr, 
                   gse_o, e_correction_u, e_correction_v_arr, u, v_arr, filling, temperature, e_threshold = 1E-8, 
                   c_threshold = 1E-7, max_iter = 1000, c_log_length=10):
    '''
    perform the self-consistent field iteration
    input_d: (2, norb)
    input_bond: (2, L, nshells, norb, norb)
    k_points: (N, 2)
    a_lattice: (2, 2)
    h_k_o: (N, norb, norb)
    h_mean_initial_u: (norb, norb)
    h_mean_initial_v_arr: (N, nshells, norb, norb)
    gse_o: float
    e_correction_o: float
    '''
    norb = input_d.shape[1]
    num_k_points = k_points.shape[0]
    nshells = v_arr.shape[0]
    iteration = 0
    e_difference = 1.0
    c_difference = 1.0
    gse = gse_o
    c_log = np.arange(c_log_length, dtype=np.float64)
    while (np.abs(e_difference) > e_threshold or c_difference > c_threshold) and iteration < max_iter:
        e_dic = np.zeros((num_k_points, 2, norb))
        v_dic = np.zeros((num_k_points, 2, norb, norb), np.complex128)
        h_mean_input_k_arr = np.zeros((num_k_points, 2, norb, norb), np.complex128)
        h_mean_input = mean_field_h_k_independent(a_list, v_arr, input_d[0] + input_d[1])
        h_mean_input = np.stack((h_mean_input, h_mean_input)) + mean_field_u(input_d, u)
        corr_k_arr = np.zeros((num_k_points, 2, norb, norb), dtype=np.complex128)
        for m in range(num_k_points):
            k = k_points[m]
            h_mean_input_k = h_mean_input + np.stack((k_dependent_bond_mean_field_h(deltas, input_bond[0], k, v_arr, a_lattice), 
                                       k_dependent_bond_mean_field_h(deltas, input_bond[1], k, v_arr, a_lattice)))
            h_mean_initial_k = h_mean_initial_u*u
            for i in range(nshells):
                h_mean_initial_k += h_mean_initial_v_arr[m,i]*v_arr[i]
            h_mean_input_k_arr[m] = h_mean_input_k
            for i in range(2):
                h_k_i = h_k_o[m] + h_mean_input_k[i] - h_mean_initial_k
                eigvals, eigvecs = la.eigh(h_k_i)
                e_dic[m,i] = eigvals
                v_dic[m,i] = eigvecs
        total_e = np.sort(e_dic.reshape(-1))
        e_fermi_iterated = fermi_level_bisection(total_e, filling, temperature)
        ground_state_e = 0
        for m in range(num_k_points):
            k = k_points[m]
            for i in range(2):
                e_kpoint = e_dic[m,i]
                v_kpoint = v_dic[m,i]
                electron_count = fermi_dirac_electron_count(e_kpoint, e_fermi_iterated, temperature)
                corr_k = np.transpose(v_kpoint @ np.diag(electron_count) @ v_kpoint.conj().T)
                corr_k_arr[m,i] = corr_k
                # this doesn't cound the spin degeneracy
                ground_state_e += np.real(np.trace((h_k_o[m] + h_mean_input_k_arr[m,i]/2 - h_mean_initial_u*u).T @ corr_k))
                for l in range(nshells):
                    ground_state_e -= np.real(np.trace((h_mean_initial_v_arr[m,l]*v_arr[l]).T @ corr_k))
        for m in range(nshells):
            ground_state_e += e_correction_v_arr[m]*v_arr[m]
        ground_state_e += e_correction_u*u
        # obtain the new bond orders
        input_bond_iterated = np.stack((bond_orders(deltas, a_list, corr_k_arr[:,0,:,:], k_points, a_lattice), 
                                bond_orders(deltas, a_list, corr_k_arr[:,1,:,:], k_points, a_lattice)))
        # obtain the new density list
        input_d_iterated = np.stack((np.diag(np.sum(corr_k_arr[:,0,:,:], axis=0))/num_k_points, 
                            np.diag(np.sum(corr_k_arr[:,1,:,:], axis=0))/num_k_points))
        # obtain the new mean-field Hamiltonian
        e_difference = np.abs(ground_state_e - gse)/num_k_points
        c_difference_d = np.max(np.abs(input_d - input_d_iterated))
        c_difference_b = np.max(np.abs(input_bond - input_bond_iterated))
        c_difference = np.max(np.array([c_difference_d, c_difference_b]))
        c_log[iteration % c_log_length] = c_difference
        c_log_diff = np.max(np.abs(np.diff(c_log)))
        # print('For iteration ' + str(iteration) +
        # ', the e difference is %.3E and c difference is %.3E, the ground state energy difference '
        # 'is %.3E, ' % (e_difference, c_difference,
        #                                     (ground_state_e - gse) / num_k_points) +
        # 'the c_log_diff is %.3E' % c_log_diff, flush=True)
        gse = ground_state_e
        input_d = input_d_iterated
        input_bond = input_bond_iterated
        iteration += 1
        if c_log_diff < c_threshold/10:
            break
    return input_d, input_bond, e_difference, c_difference, (gse-gse_o)/num_k_points, e_fermi_iterated, c_log, iteration
    

def schf_fixed_filling_parallel_u(u, input_d_tot, input_bond_tot, deltas, a_list, k_points, a_lattice, h_k_o, h_mean_initial_u, h_mean_initial_v_arr, 
                   gse_o, e_correction_u, e_correction_v_arr, v_arr, filling, temperature, e_threshold = 1E-8, 
                   c_threshold = 1E-7, max_iter = 1000, c_log_length=10):
    '''
    perform the self-consistent field iteration for fixed filling and u
    '''
    norb = input_d_tot.shape[-1]
    num_v1_points = v_arr.shape[0]
    num_channel = input_d_tot.shape[0]
    final_d = np.zeros((num_v1_points, num_channel, 2, norb))
    final_bond = np.zeros((num_v1_points, num_channel, 2, deltas.shape[0], v_arr.shape[-1], norb, norb), dtype=np.complex128)
    final_e_difference = np.zeros((num_v1_points, num_channel))
    final_c_difference = np.zeros((num_v1_points, num_channel))
    final_gse = np.zeros((num_v1_points, num_channel))
    final_e_fermi_iterated = np.zeros((num_v1_points, num_channel))
    final_c_log = np.zeros((num_v1_points, num_channel, c_log_length))
    final_iteration = np.zeros((num_v1_points, num_channel))
    for i in range(num_v1_points):
        v_arr_i = v_arr[i]
        for j in range(num_channel):
            d_value = (u + np.sum(v_arr_i))/10
            input_d = input_d_tot[j]*d_value
            input_bond = input_bond_tot[j]*d_value
            final_d[i,j], final_bond[i,j], final_e_difference[i,j], final_c_difference[i,j], final_gse[i,j], final_e_fermi_iterated[i,j], final_c_log[i,j], final_iteration[i,j] = \
                collinear_schf_iteration(input_d, input_bond, deltas, a_list, k_points, a_lattice, h_k_o, h_mean_initial_u, h_mean_initial_v_arr, 
                                        gse_o, e_correction_u, e_correction_v_arr, u, v_arr_i, filling, temperature, e_threshold = e_threshold, 
                                        c_threshold = c_threshold, max_iter = max_iter, c_log_length=c_log_length)
        print(f"u = {u}, v_arr = {v_arr_i}", flush=True)
    return final_d, final_bond, final_e_difference, final_c_difference, final_gse, final_e_fermi_iterated, final_c_log, final_iteration
            

    







