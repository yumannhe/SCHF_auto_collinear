from typing import Callable, Any, Dict
from functions_parameters.jax_schf_helpers import *

import jax
import jax.numpy as jnp
from jax import lax
from jax import pmap, vmap, local_device_count
from jax.tree_util import tree_map

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
        
        # a more GPU safer way to diagonalize the Hamiltonian to avoid overloads
        def sequential_diagonalize(hk):
            # enforce Hermitian
            hk = 0.5*(hk + jnp.swapaxes(hk, -1, -2).conj())
            def process_one(carry, x):
                eigvals, eigvecs = jax.vmap(jnp.linalg.eigh)(x)
                return carry, (eigvals, eigvecs)

            _, results = lax.scan(process_one, None, hk)
            eigvals, eigvecs = results
            return eigvals, eigvecs

        eigvals, eigvecs = sequential_diagonalize(hk)

        # # CPU OK with this but still slow
        # hk = 0.5*(hk + jnp.swapaxes(hk, -1, -2).conj())
        # eigvals, eigvecs = jnp.linalg.eigh(hk)

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
# vmap singly is not true parallelization, and still quite slow
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
    over_channel = vmap(run_one, in_axes=(None, None, 0, 0))
    # map over v
    over_v = vmap(over_channel, in_axes=(None, 0, None, None))
    # map over u
    over_u = vmap(over_v, in_axes=(0, None, None, None))

    # run the schf
    output = over_u(u, v_arr, input_d_tot, input_bond_tot)

    return output


# ----------------------
# parallelization for fixed filling using pmap
# ----------------------
def schf_fixed_filling_pmap_over_u(
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
    parallelize the schf_core over u, v and input_d_tot, input_bond_tot using pmap
    input_d_tot: (nchannels, 2, norb)
    input_bond_tot: (nchannels, 2, nshells, norb, norb)
    u: (nU,)
    v_arr: (nV, nshells)
    temperature: float
    e_threshold: float
    c_threshold: float
    max_iter: int
    """
    nU = u.shape[0]
    nV, nshells = v_arr.shape
    nCh = input_d_tot.shape[0]

    # JIT once to avoid recompiles inside pmap
    schf_core_jit = jax.jit(schf_core, static_argnames=('max_iter',))

    def run_one(u_i, v_i, input_d, input_bond):
        return schf_core_jit(
            Htb, a_list, phase_pos, phase_neg, dict_ref, 
            input_d, input_bond, filling, u_i, v_i, temperature, 
            e_threshold, c_threshold, max_iter
        )

    # For a single u_i, sweep all v and all channels locally (keeps d/bond paired)
    def run_for_one_u(u_i):
        # over channels (pair d and bond via same channel index)
        def over_channel(v_i):
            return vmap(run_one, in_axes=(None, None, 0, 0))(
                u_i, v_i, input_d_tot, input_bond_tot
            )  # → (nCh, ...) dict/tree
        # then over v
        return vmap(over_channel, in_axes=(0,))(v_arr)  # → (nV, nCh, ...) dict/tree
    
    D = local_device_count()

    # avoiding recompiles of pmap inside loop for chunking 
    per_device = pmap(run_for_one_u, in_axes=0, out_axes=0)

    # Jax requires the size of pmap mapping axis matches exactly with D
    # so we meed tp safely chunk just in case. 
    def pmap_chunk(u_chunk):
        L = u_chunk.shape[0]
        if L<D:
            u_chunk = jnp.pad(u_chunk, (0, D-L))
        out = per_device(u_chunk)
        return tree_map(lambda x: x[:L], out) 
    
    if nU <=D:
        return pmap_chunk(u)
    
    # nU > D: loop over chunks of size D on the host; compiled pmap is reused
    pieces = [pmap_chunk(u[i:i+D]) for i in range(0, nU, D)]
    return tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *pieces)


# ----------------------
# parallelization for fixed u-v pairing but different filling
# ----------------------
def schf_fixed_u_v_pair_pmap_over_filling(
    schf_core: Callable,
    Htb: Array,
    a_list: Array,
    e_all: Array, 
    v_all: Array, 
    v_all_dagger: Array,
    phase_pos: Array,
    phase_neg: Array,
    input_d_tot: Array,
    input_bond_tot: Array,
    filling: Array,
    u: Array,
    v_arr: Array,
    temperature: float,
    e_threshold: float = 1e-8,
    c_threshold: float = 1e-7,
    max_iter: int = 500,
):
    """
    parallelize the schf_core over filling using pmap
    filling: (nfilling,)
    u: (nU,)
    v_arr: (nV, nshells)
    temperature: float
    e_threshold: float
    c_threshold: float
    max_iter: int
    """
    # nU should be the same as nV
    nU = u.shape[0]
    nV, nshells = v_arr.shape
    nCh = input_d_tot.shape[0]
    nfilling = filling.shape[0]

    # JIT once to avoid recompiles inside pmap
    schf_core_jit = jax.jit(schf_core, static_argnames=('max_iter',))

    def run_one(filling_i, dict_ref_i, u_i, v_i, input_d, input_bond):
        return schf_core_jit(
            Htb, a_list, phase_pos, phase_neg, dict_ref_i,
            input_d, input_bond, filling_i, u_i, v_i, temperature, 
            e_threshold, c_threshold, max_iter
        )
    
    # For a single filling, sweep all u,v and all channels locally (keeps d/bond paired)
    def run_for_one_filling(filling_i):
        dict_ref_i = prepare_reference_state(filling_i, a_list, Htb[:,0,:,:], e_all, v_all, v_all_dagger, phase_pos, phase_neg, temperature)
        # over channels (pair d and bond via same channel index)
        def over_channel(dict_ref_i, u_i, v_i):
            return vmap(run_one, in_axes=(None, None, None, None, 0, 0))(
                filling_i, dict_ref_i, u_i, v_i, input_d_tot, input_bond_tot
            )  # → (nCh, ...) dict/tree
        # then over u and v
        return vmap(over_channel, in_axes=(None, 0, 0))(dict_ref_i, u, v_arr)

    D = local_device_count()

    # avoiding recompiles of pmap inside loop for chunking 
    per_device = pmap(run_for_one_filling, in_axes=0, out_axes=0)

    def pmap_chunk(filling_chunk):
        L = filling_chunk.shape[0]
        if L<D:
            filling_chunk = jnp.pad(filling_chunk, (0, max(0, D-L)))
        out = per_device(filling_chunk)
        return tree_map(lambda x: x[:L], out) 

    if nfilling <=D:
        return pmap_chunk(filling)
    
    # nfilling > D: loop over chunks of size D on the host; compiled pmap is reused
    pieces = [pmap_chunk(filling[i:i+D]) for i in range(0, nfilling, D)]
    return tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *pieces)


def schf_fixed_filling_u_pmap_over_v1_v2(
    schf_core: Callable,
    Htb: Array,
    a_list: Array,
    phase_pos: Array,
    phase_neg: Array,
    dict_ref: PyTree,
    input_d_tot: Array,
    input_bond_tot: Array,
    filling: float,
    u: float,
    v_1: Array,
    v_2: Array,
    temperature: float,
    e_threshold: float = 1e-8,
    c_threshold: float = 1e-7,
    max_iter: int = 500,
):
    """
    parallelize the schf_core over v1, v2 using pmap
    This is used for the two-shell schf
    nshells = 2
    """
    nshells = 2

    nV1, nV2 = v_1.shape[0], v_2.shape[0]
    nCh = input_d_tot.shape[0]

    schf_core_jit = jax.jit(schf_core, static_argnames=('max_iter',))

    def run_one(v_i, input_d, input_bond):
        return schf_core_jit(
            Htb, a_list, phase_pos, phase_neg, dict_ref, 
            input_d, input_bond, filling, u, v_i, temperature, 
            e_threshold, c_threshold, max_iter
        )

    
    def run_for_one_v1(v1_i):
        v1_arr = jnp.ones((nV2,))*v1_i
        v_i = jnp.stack((v1_arr, v_2), axis=1)
        # over channels (pair d and bond via same channel index)
        def over_channel(v_i):
            return vmap(run_one, in_axes=(None, 0, 0))(
                v_i, input_d_tot, input_bond_tot
            )  # → (nCh, ...) dict/tree
        # then over u and v
        return vmap(over_channel, in_axes=0)(v_arr)

    D = local_device_count()

    # avoiding recompiles of pmap inside loop for chunking 
    per_device = pmap(run_for_one_v1, in_axes=0, out_axes=0)

    def pmap_chunk(v1_chunk):
        L = v1_chunk.shape[0]
        if L<D:
            v1_chunk = jnp.pad(v1_chunk, (0, max(0, D-L)))
        out = per_device(v1_chunk)
        return tree_map(lambda x: x[:L], out) 

    if nV1 <=D:
        return pmap_chunk(v_1)
    
    # nV1 > D: loop over chunks of size D on the host; compiled pmap is reused
    pieces = [pmap_chunk(v_1[i:i+D]) for i in range(0, nV1, D)]
    return tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *pieces)   


    