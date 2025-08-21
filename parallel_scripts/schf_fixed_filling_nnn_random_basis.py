import time
import os as os
from functions_parameters.universal_parameters import a, b, a1_basis, e2_basis1, e2_basis2
from functions_parameters.bond_table_generator import build_buckets_per_shell
import numpy.linalg as la
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from itertools import product
from functions_parameters.schf import non_interacting_outputs, schf_fixed_filling_parallel_u, bond_orders, \
    mean_field_u, mean_field_h_k_independent, k_dependent_bond_mean_field_h
from functools import partial
import pickle

if __name__ == "__main__":
    '''
    parameters
    '''
    basis_frac = np.array([[1/2, 0], [1/2, 1/2], [0, 1/2]])
    norb = basis_frac.shape[0]
    num_k_mesh = 60
    b_0 = np.linspace(-b[0] / 2, b[0] / 2, num_k_mesh, endpoint=False)
    b_1 = np.linspace(-b[1] / 2, b[1] / 2, num_k_mesh, endpoint=False)
    k_mesh_points = np.vstack([v1 + v2 for v1, v2 in product(b_0, b_1)])
    num_k_points = k_mesh_points.shape[0]

    '''
    non-interacting case
    '''
    temperature = 4E-4
    filling = 1/6
    mu = 2
    t_nn = 1
    t_nnn = -0.025
    t_arr = np.array([t_nn, t_nnn])

    radii, a_lists, deltas = build_buckets_per_shell(a, basis_frac, t_arr.shape[0])

    corr_k_arr, hk_o_arr, ground_state_e = \
        non_interacting_outputs(k_mesh_points, deltas, a_lists, 
                                mu, t_arr, a, filling, temperature)

    '''
    SCHF preperation
    '''
    # generate the random basis for nematic channels
    num_nematic_p_channel = 3
    num_channel = num_nematic_p_channel * 3 + 1
    ferro_basis = np.stack((a1_basis, -a1_basis))
    rng = np.random.default_rng(111)          # reproducible
    random_coefficients_p = rng.uniform(-1, 1, size=(num_nematic_p_channel, 2))
    random_coefficients_m = rng.uniform(0, 1, size=(num_nematic_p_channel, 1))
    random_coefficients_ferri = rng.uniform(0, 1, size=(num_nematic_p_channel, 3))
    random_coefficients_ferri = random_coefficients_ferri/la.norm(random_coefficients_ferri, axis=1)[:,np.newaxis]
    random_coefficients_p = random_coefficients_p/la.norm(random_coefficients_p, axis=1)[:,np.newaxis]
    random_basis_p = random_coefficients_p[:,0][:,np.newaxis] * e2_basis1 + random_coefficients_p[:,1][:,np.newaxis] * e2_basis2
    random_basis_p = np.stack((random_basis_p, random_basis_p), axis=1)
    ferri_basis = np.stack((random_coefficients_ferri, -random_coefficients_ferri), axis=1)
    random_coefficients_m = np.concatenate((random_coefficients_m, random_coefficients_p),axis=1)
    random_coefficients_m = random_coefficients_m/la.norm(random_coefficients_m, axis=1)[:,np.newaxis]
    random_basis_m_p = random_coefficients_m[:,1][:,np.newaxis] * e2_basis1 + random_coefficients_m[:,2][:,np.newaxis] * e2_basis2
    random_basis_m_p = np.stack((random_basis_m_p, random_basis_m_p), axis=1)
    random_basis_m = random_coefficients_m[:,0,np.newaxis,np.newaxis] * ferro_basis[np.newaxis,:,:] + random_basis_m_p
    total_channel_arr = np.concatenate((ferro_basis[np.newaxis,:,:], random_basis_m, ferri_basis, random_basis_p), axis=0)
    
    # num_u_points = 16
    # u_arr = np.linspace(0, 0.75, num_u_points)
    num_u_points = 1
    u_arr = np.array([0.4])
    v1_arr = u_arr
    v2_arr = np.zeros(num_u_points)
    v_arr = np.stack((v1_arr, v2_arr), axis=1)
    c_log_length = 10

    bond_orders_o = bond_orders(deltas, a_lists, corr_k_arr, k_mesh_points, a)
    density_list = np.diag(np.sum(corr_k_arr, axis=0)/num_k_points)
    h_mean_o_u = mean_field_u(np.stack((density_list, density_list)), 1)
    # the factor 2 for density list indicate the spin degeneracy
    h_mean_o_v_arr = np.zeros((v_arr.shape[-1], norb, norb), dtype=np.complex128)
    for i in range(v_arr.shape[-1]):
        h_mean_o_v_arr[i] = mean_field_h_k_independent(a_lists[:,i:i+1,:,:], np.array([1]), density_list*2)

    h_mean_v_arr = np.zeros((num_k_points, v_arr.shape[-1], norb, norb), dtype=np.complex128)
    e_u_o = 0
    e_v_o = np.zeros(v_arr.shape[-1])
    for i in range(num_k_points):
        k = k_mesh_points[i]
        for j in range(v_arr.shape[-1]):
            h_mean_v = k_dependent_bond_mean_field_h(deltas, bond_orders_o[:,j:j+1,:,:], k, np.array([1]), a)
            h_mean_v_arr[i,j] = h_mean_v + h_mean_o_v_arr[j]
        # the multiplier factor 2 for density list indicate the spin degeneracy
        # the divider factor 2 for the trace is because of the double counting in the mean-field decomposition
        e_u_o += np.real(np.trace(h_mean_o_u[0].T @ corr_k_arr[i]))/2*2
        for j in range(v_arr.shape[-1]):
            e_v_o[j] += np.real(np.trace(h_mean_v_arr[i,j].T @ corr_k_arr[i]))/2*2
    # count the spin degeneracy
    ground_state_e = ground_state_e*2
    input_bond_tot = np.zeros((total_channel_arr.shape[0], 2, deltas.shape[0], v_arr.shape[-1], norb, norb))
    
    '''
    SCHF iteration
    '''
    partial_nnn_colinear_hf_filling_v1_v2_fd_bi_parallel = \
        partial(schf_fixed_filling_parallel_u, input_d_tot=total_channel_arr,
                input_bond_tot=input_bond_tot, deltas=deltas, a_list=a_lists, k_points=k_mesh_points, a_lattice=a, h_k_o=hk_o_arr, h_mean_initial_u=h_mean_o_u[0],
                h_mean_initial_v_arr=h_mean_v_arr, gse_o=ground_state_e, e_correction_u=e_u_o, e_correction_v_arr=e_v_o,
                v_arr=v_arr, filling=filling, temperature=temperature, e_threshold=1E-8, c_threshold=1E-7, max_iter=1000, c_log_length=c_log_length)
    POOL_SIZE = num_u_points
    results = [None] * u_arr.shape[0]
    futures = {}
    with ProcessPoolExecutor(max_workers=POOL_SIZE) as executor:
        for idx, f in enumerate(u_arr):
            fut = executor.submit(partial_nnn_colinear_hf_filling_v1_v2_fd_bi_parallel, f)
            futures[fut] = idx               # remember where this result belongs
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()  # slot straight into its place
                print(f"Process finished u_arr[{idx}] = {u_arr[idx]}", flush=True)
            except Exception as e:
                print(f"error on u_arr[{idx}] ({u_arr[idx]}): {e}", flush=True)

    final_d = np.stack([i[0] for i in results])
    final_bond = np.stack([i[1] for i in results])
    final_e_difference = np.stack([i[2] for i in results])
    final_c_difference = np.stack([i[3] for i in results])
    final_gse = np.stack([i[4] for i in results])
    final_e_fermi = np.stack([i[5] for i in results])
    final_c_log = np.stack([i[6] for i in results])
    final_iteration = np.stack([i[7] for i in results])
    final_result = {'density': final_d, 'bond': final_bond, 'e difference': final_e_difference,
                    'c difference': final_c_difference,
                    'ground state energy density difference': final_gse, 'kmesh': num_k_mesh,
                    'u': u_arr, 'v':v_arr, 'filling': filling, 'temperature': temperature,
                    'final_fermi_energy': final_e_fermi, 'c log': final_c_log, 'iteration': final_iteration}

    print(final_gse)
    # with open('colinear_kagome_fd_filling_1o6_mesh60_u_v1_diagram_t_4Em4_hpc4_random_basis.pkl', 'wb') as file:
    #     pickle.dump(final_result, file)
