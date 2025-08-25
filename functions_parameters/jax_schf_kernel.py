from typing import Callable, Any, Dict
from functions_parameters.jax_schf_helpers import *

import jax
import jax.numpy as jnp
from jax import lax

# ---- numeric mode ----
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)  # use float64 by default

Array = jnp.ndarray
PyTree = Any

def schf_single_job(
    Htb: Array,
    a_list: Array,
    phase_pos: Array,
    phase_neg: Array,
    dict_ref: PyTree,
    input_d: Array,
    input_bond: Array,
    filling: float,
    u: float,
    v_arr: Array,
    temperature: float,
    e_threshold: float = 1e-8,
    c_threshold: float = 1e-7,
    max_iter: int = 500,
):
    '''
    perform the self-consistent field iteration for a single job using lax.while_loop
    '''
    norb = input_d.shape[1]
    nk = Htb.shape[0]
    nshells = v_arr.shape[-1]
    # unpack the reference state
    gse_o = dict_ref['gs_e']
    h_mean_u = dict_ref['h_mean_u']
    h_mean_v = dict_ref['h_mean_v']
    e_u_o = dict_ref['e_u_o']
    e_v_o_arr = dict_ref['e_v_o_arr']
    e_fermi_initial = dict_ref['e_fermi']

    # prepare the h_mean_initial, which only depends on filling and u,v
    h_mean_initial_u = h_mean_u*u
    h_mean_initial = h_mean_initial_u[None, :, :, :] + jnp.einsum('Nsvmn,v->Nsmn', h_mean_v,v_arr)

    # prepare the e_correction, which only depends on filling and u,v
    e_correction_u = e_u_o*u
    e_correction_v_arr = e_v_o_arr@v_arr
    e_correction_initial = e_correction_u + e_correction_v_arr

    # prepare the initial input depends on the u,v_arr to treat the input as a perturbation
    scaled_factor = (u+jnp.sum(v_arr))/10.0
    input_d = input_d * scaled_factor
    input_bond = input_bond * scaled_factor

    def cond(carry):
        d, bond, e_diff, c_diff, gse, e_fermi, any_bi_fail, iters = carry
        not_converged = jnp.logical_or(e_diff > e_threshold, c_diff > c_threshold)
        return jnp.logical_and(not_converged, iters < max_iter)
    
    def body(carry):
        d, bond, e_diff, c_diff, gse, e_fermi, any_bi_fail, iters = carry

        # from input order parameters obtain the mean field decomposition
        # k_independent part
        h_mean_input_u = mean_field_u(d, u)
        h_mean_input_v_o = hartree_spinor(a_list, v_arr, d)
        h_mean_input_o = h_mean_input_u + h_mean_input_v_o
        # k-dependent part
        h_mean_input_v_k = fock_spinor(bond, v_arr, phase_pos)
        h_mean_input = h_mean_input_o[None, :, :, :] + h_mean_input_v_k

        # contrcut the Hamiltonian:
        hk = Htb + h_mean_input - h_mean_initial
        
        # # a more GPU safer way to diagonalize the Hamiltonian to avoid overloads
        # def sequential_diagonalize(hk):
        #     # enforce Hermitian
        #     hk = 0.5*(hk + jnp.swapaxes(hk, -1, -2).conj())
        #     def process_one(carry, x):
        #         eigvals, eigvecs = jax.vmap(jnp.linalg.eigh)(x)
        #         return carry, (eigvals, eigvecs)

        #     _, results = lax.scan(process_one, None, hk)
        #     eigvals, eigvecs = results
        #     return eigvals, eigvecs

        # eigvals, eigvecs = sequential_diagonalize(hk)

        # GPU is not that efficient in this case, so we use the naive way
        hk = 0.5*(hk + jnp.swapaxes(hk, -1, -2).conj())
        eigvals, eigvecs = jnp.linalg.eigh(hk)

        # obtain the new state
        e_fermi_new, _, bi_converged_new, _ = fermi_level_bisection_core(eigvals, filling, temperature)
        electron_count = occ(eigvals, e_fermi_new, temperature)
        # spin-resolved and transposed to (N, 2, norb, norb)
        corr_k = jnp.einsum('Nsni,Nsi,Nsmi->Nsmn', eigvecs, electron_count, eigvecs.conj())
        d_new = jnp.einsum('Nsii->si', corr_k)/nk
        bond_new = bond_orders_from_phases(a_list, corr_k, phase_neg)

        # obtain the new ground state energy
        gse_new = (jnp.einsum('Nsji, Nsji->', Htb + h_mean_input/2 - h_mean_initial, corr_k)/nk).real
        
        # obtain the difference
        c_diff_d_new = jnp.max(jnp.abs(d - d_new))
        c_diff_bond_new = jnp.max(jnp.abs(bond - bond_new))
        c_diff_new = jnp.max(jnp.array([c_diff_d_new, c_diff_bond_new]))
        e_diff_new = jnp.abs(gse_new - gse)

        # check if any bisection fails
        any_bi_fail_new = jnp.logical_or(any_bi_fail, jnp.logical_not(bi_converged_new))
        # update the carry
        carry = d_new, bond_new, e_diff_new, c_diff_new, gse_new, e_fermi_new, any_bi_fail_new, iters + 1
        return carry

    carry_init = input_d, input_bond, jnp.array(1.0, dtype=jnp.float64), jnp.array(1.0, dtype=jnp.float64), gse_o, e_fermi_initial, \
        jnp.array(False, dtype=jnp.bool_), jnp.array(0, dtype=jnp.int32)
    d, bond, e_diff, c_diff, gse, e_fermi, any_bi_fail, iters = lax.while_loop(cond, body, carry_init)
    
    gse = gse + e_correction_initial - gse_o

    return dict(
        d=d, 
        bond=bond, 
        e_diff=e_diff, 
        c_diff=c_diff, 
        gse=gse, 
        e_fermi=e_fermi, 
        any_bi_fail=any_bi_fail, 
        iters=iters,
    )


# ----------------------
# parallelization for fixed filling
# ----------------------
def schf_fixed_filling_prallel_u_v(
    schf_core: Callable,
    Htb: Array,
    a_list: Array,
    phase_pos: Array,
    phase_neg: Array,
    dict_ref: PyTree,
    input_d_tot: Array,
    input_bond_tot: Array,
    filling: float,
    u: Array,
    v_arr: Array,
    temperature: float,
    e_threshold: float = 1e-8,
    c_threshold: float = 1e-7,
    max_iter: int = 500,
):
    """
    parallelize the schf_core over u, v and input_d_tot, input_bond_tot
    input_d_tot: (nchannels, 2, norb)
    input_bond_tot: (nchannels, 2, nshells, norb, norb)
    u: (nU,)
    v_arr: (nV, nshells)
    temperature: float
    e_threshold: float
    c_threshold: float
    max_iter: int
    """

    # one job (works like partial in python)
    def run_one(u_i, v_i, input_d, input_bond):
        return schf_core(Htb, a_list, phase_pos, phase_neg, dict_ref, input_d, input_bond, filling, u_i, v_i, temperature, e_threshold, c_threshold, max_iter)

    # map over input channels
    # density and bonds come in pairs
    over_channel = jax.vmap(run_one, in_axes=(None, None, 0, 0))
    # map over v
    over_v = jax.vmap(over_channel, in_axes=(None, 0, None, None))
    # map over u
    over_u = jax.vmap(over_v, in_axes=(0, None, None, None))

    # run the schf
    output = over_u(u, v_arr, input_d_tot, input_bond_tot)

    return output
