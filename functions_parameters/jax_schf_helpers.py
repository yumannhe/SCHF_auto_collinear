from functools import partial
from typing import Callable, Optional, Dict, Any

import jax
import jax.numpy as jnp
from jax import lax

# ---- numeric mode ----
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)  # use float64 by default

Array = jnp.ndarray
PyTree = Any

def occ(evals: Array, mu: Array, kT: float) -> Array:
    """Fermi-Dirac occupations at finite temperature only (kT > 0).
    evals: (...,) energies
    mu: scalar (Array) chemical potential
    returns: same shape as evals
    """
    beta = 1.0 / kT
    x = jnp.clip((evals - mu) * beta, -60.0, 60.0)
    return 1.0 / (1.0 + jnp.exp(x))


def fermi_level_bisection_core(
    evals: Array,        # (nk, nb)
    target_ne: float,    # electrons per cell
    kT: float,
    tol: float = 1e-9,
    max_iter: int = 100,
):
    """Pure-JAX bisection to find mu at finite T.
    Returns (mu, ne, converged, iters) as JAX values.
    Assumes uniform k-weights; electrons per cell is (sum_k,n f_kn)/nk.
    """
    nk = evals.shape[0]
    e_min, e_max = jnp.min(evals), jnp.max(evals)
    lo = e_min - 10.0 * kT
    hi = e_max + 10.0 * kT

    def count(mu):
        # electrons per cell
        return jnp.sum(occ(evals, mu, kT)) / nk

    def cond_fun(carry):
        lo, hi, it = carry
        mu = 0.5 * (lo + hi)
        ne = count(mu)
        diff = ne - target_ne
        return jnp.logical_and(jnp.abs(diff) >= tol, it < max_iter)

    def body_fun(carry):
        lo, hi, it = carry
        mu = 0.5 * (lo + hi)
        ne = count(mu)
        diff = ne - target_ne
        lo = jnp.where(diff < 0.0, mu, lo)
        hi = jnp.where(diff >= 0.0, mu, hi)
        return (lo, hi, it + 1)

    lo, hi, it = lax.while_loop(cond_fun, body_fun, (lo, hi, jnp.array(0, jnp.int32)))
    mu = 0.5 * (lo + hi)
    ne = count(mu)
    converged = jnp.abs(ne - target_ne) < tol
    return mu, ne, converged, it


'''
--------- convention of bond orders: ------------
the bond orders are defined as c_r_i_d*c_(r+dr)_j


--------- convention of FT: ------------
c_r_d = '\\sum_k c_k_d e^{ikr}/sqrt(N)'
c_k_d = '\\sum_r c_r_d e^{-ikr}/sqrt(N)'
where N is the number of lattice sites.
This is because the FT of a real function is Hermitian.

# Hamiltonian:
h_r_dr_ij = t_ij*c_r_i_d*c_(r+dr)_j
h_k_ij = '\\sum_dr h_r_dr_ij e^{-ikdr} c_k_i_d*c_k_j'

# Correlation matrix:
corr_dr_ij = c_r_dr_ij*c_r_i_d*c_(r+dr)_j
           = '\\sum_k corr_k_ij e^{-ikdr}/N'
corr_k_ij = c_k_ij*c_k_i_d*c_k_j

--------- note: ------------
Since we define bond orders as c_r_i_d*c_(r+dr)_j, the mean-field "companion" shoud have the opposite order
as c_(r+dr)_j_d*c_r_i, which should have opposite Fourier phase factor 
'''

# ----------------------
# Phase precomputation
# ----------------------
def precompute_k_phase_tables(deltas, a_lattice, k_mesh_points):
    """
    deltas: (L, 2)
    a_lattice: (2, 2)
    k_mesh_points: (N, 2)
    drs: (L, 2)
    returns:
      k_phase_pos: (N, L) with exp(+i dr · k)
      k_phase_neg: (N, L) with exp(-i dr · k)  (conj of k_phase_pos)
    """
    deltas = jnp.asarray(deltas)
    a_lattice = jnp.asarray(a_lattice)
    k_mesh_points = jnp.asarray(k_mesh_points)
    drs = deltas @ a_lattice
    # (N,2) @ (2,L) -> (N, L)
    dot = k_mesh_points @ drs.T
    k_phase_pos = jnp.exp(1j * dot)
    k_phase_neg = jnp.conj(k_phase_pos)
    return k_phase_pos, k_phase_neg

# ----------------------
# Bond orders (uses precomputed phases)
# ----------------------
def bond_orders_from_phases(a_list, corr_k, k_phase_neg):
    """
    obtain bond orders magnitude from correlation matrices using precomputed phases

    a_list: (L, C, norb, norb)    # C = "nshells" or interaction channels in your code
    corr_k: (N, ..., norb, norb)  # possibly spin-resolved
    k_phase_neg: (N, L)           # exp(-i k · dr)
    return: bond_orders: (L, C, ..., norb, norb)

    Equivalent to your original loop, but vectorized:
      corr_i = (1/N) * sum_k [ corr_k(k) * exp(-i k · dr_i) ]
      bond_orders[i, j] = corr_i * a_list[i, j]
    """
    N = corr_k.shape[0]

    # Average correlation for each delta i: (L, ..., norb, norb)
    # einsum: (L,N) x (N,...,norb,norb) -> (L,...,norb,norb)
    corr_by_delta = jnp.einsum('lN,N...ab->l...ab', k_phase_neg.T, corr_k) / float(N)

    bond_orders = jnp.einsum('l...ab,lcab->l...cab', corr_by_delta, a_list)
    # result: (L, ..., C, n, n)
    return bond_orders

# ----------------------
# k-dependent Fock (bond) mean field from precomputed phases
# ----------------------
def fock_spinor(bond_orders, v_arr, k_phase_pos):
    """
    Vectorized over all k:
      bond_orders: (L, ..., C, norb, norb)
      v_arr: (C,)
      k_phase_pos: (N, L)
    return: h_mean_bond: (N, ..., norb, norb)
    """
    W_T = jnp.einsum('l...cnm,c->l...mn', bond_orders, v_arr)  # (L,...,norb,norb)
    # (N,L) & (L,...,norb,norb) -> (N,...,norb,norb)
    h_mean_bond = jnp.einsum('Nl,l...nm->N...nm', k_phase_pos, W_T)
    return -h_mean_bond

# ----------------------
# k-independent mean-field diagonal (Hartree-like) from bond form factors
# ----------------------
def hartree_spinor(a_list, v_arr, density_vec):
    """
    a_list: (L, C, norb, norb)
    v_arr: (C,)
    density_vec: (2, norb,)          # spin-resolved input
    return: h_mean: (..., norb, norb)  # diagonal matrix
    """
    # found out all the "density pairing" from bond tables
    density_vec = jnp.sum(density_vec, axis=0)
    diag_vec = jnp.einsum('lcnm,c,m->n', a_list, v_arr, density_vec)
    diag_matrix = jnp.diag(diag_vec)
    return jnp.stack((diag_matrix, diag_matrix))

# ----------------------
# On-site U (collinear two-spin case)
# ----------------------
def mean_field_u(density_s, u):
    """
    density_s: (2, norb)   # spin-resolved densities
    u: float
    return: (2, norb, norb) with up sees n_down and vice versa
    """
    return jnp.stack((jnp.diag(u * density_s[1]), jnp.diag(u * density_s[0])))


# ----------------------
# non-interacting hk, e_all, v_all for all k
# ----------------------
def hk_all_k_from_phases(mu, a_list, t_arr, k_phase_neg):
    """
    Vectorized over all k:
      k_phase_neg: (N, L)
    return: H(k), e_all, v_all for all k: (N, norb, norb), (N, norb), (N, norb, norb)
    """
    a_list = jnp.asarray(a_list)
    t_arr = jnp.asarray(t_arr)
    norb = a_list.shape[-1]
    M = jnp.einsum('lcnm,c->lnm', a_list, t_arr)      # (L,norb,norb)
    Htb = jnp.einsum('Nl,lnm->Nnm', k_phase_neg, M)     # (N,norb,norb)
    Htb = Htb + jnp.eye(norb, dtype=jnp.complex128)[None, :, :] * mu 
    e_all, v_all = jnp.linalg.eigh(Htb)
    return Htb, jnp.real(e_all).astype(jnp.float64), v_all, jnp.swapaxes(v_all, -1, -2).conj()




# ----------------------
# reference state information
# ----------------------
def prepare_reference_state(
        filling:float, 
        a_list: Array,
        hk_o_arr: Array,
        e_all: Array, 
        v_all: Array, 
        v_all_dagger: Array,
        k_phase_pos: Array, 
        k_phase_neg: Array,
        kT:float):
    """
    filling: float
    a_list: (L, C, norb, norb)
    hk_o_arr: (N, norb, norb)
    e_all: (N, norb)
    v_all: (N, norb, norb)
    v_all_dagger: (N, norb, norb)
    k_phase_pos: (N, L)
    k_phase_neg: (N, L)
    kT: float
    Note: the input does not count the spin degeneracy, but all the output has taken care of the spin degeneracy
    """
    # first obtain the correlation matrix for each k and the ground state energy
    nk = e_all.shape[0]
    norb = e_all.shape[1]
    e_fermi_iterated, ne, converged, iters = fermi_level_bisection_core(e_all, filling, kT)
    electron_count = occ(e_all, e_fermi_iterated, kT)
    # (U @ diag(electron_count) @ U_d).T
    corr_k = jnp.einsum('Nni,Ni,Nim->Nmn', v_all, electron_count, v_all_dagger)
    # count the spin degeneracy
    gs_e = jnp.einsum('Nji, Nji->', hk_o_arr, corr_k)/nk*2

    # then obtain the mean field decomposition and corresponding 
    bond_orders = bond_orders_from_phases(a_list, corr_k, k_phase_neg)
    density_vec = jnp.einsum('Nii->i', corr_k)/nk
    h_mean_u = mean_field_u(jnp.stack((density_vec, density_vec)), 1)
    # count the spin degeneracy
    h_mean_v_o = jnp.einsum('lcmn,n->cm',a_list, density_vec*2)
    h_mean_v_o = jnp.einsum('cm,mn->cmn', h_mean_v_o, jnp.eye(norb, dtype=h_mean_v_o.dtype))
    # mean field from the tranpose of bond orders
    h_mean_v_k = jnp.einsum('Nl,lcmn->Ncnm', k_phase_pos, bond_orders)
    h_mean_v = h_mean_v_o[None, :, :, :] - h_mean_v_k
    # make the shape of (N, 2, nshells, norb, norb)
    h_mean_v = jnp.stack((h_mean_v, h_mean_v), axis=1)
    # the factor 2 is to avoid the double counting
    # h_mean_u has shape (2, norb, norb), need to sum over spin dimension
    e_u_o = jnp.einsum('sji, Nji->', h_mean_u, corr_k)/nk/2
    e_v_o_arr = jnp.einsum('Nsvji, Nji->v', h_mean_v, corr_k)/nk/2
    return {
        'gs_e': jnp.real(gs_e).astype(jnp.float64),
        'h_mean_u': h_mean_u,
        'h_mean_v': h_mean_v,
        'e_u_o': jnp.real(e_u_o).astype(jnp.float64),
        'e_v_o_arr': jnp.real(e_v_o_arr).astype(jnp.float64),
        'e_fermi': jnp.real(e_fermi_iterated).astype(jnp.float64),
        'ne': jnp.real(ne).astype(jnp.float64),
        'converged': converged,
        'iters': jnp.real(iters).astype(jnp.float64),
        "bond_orders": bond_orders,
        "density_vec": density_vec,
    }


