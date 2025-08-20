
import numpy as np
from typing import List, Tuple, Dict

def frac_to_cart(frac: np.ndarray, a_lattice: np.ndarray) -> np.ndarray:
    return frac @ a_lattice

def enumerate_pairs(a_lattice: np.ndarray,
                    basis_frac: np.ndarray,
                    search_R: int = 2):
    norb = basis_frac.shape[0]
    out = []
    for i in range(norb):
        for j in range(norb):
            for d1 in range(-search_R, search_R+1):
                for d2 in range(-search_R, search_R+1):
                    if i == j and d1 == 0 and d2 == 0:
                        continue
                    dR = np.array([d1, d2], dtype=float)
                    delta_frac = basis_frac[j] + dR - basis_frac[i]
                    delta_cart = frac_to_cart(delta_frac, a_lattice)
                    r = float(np.linalg.norm(delta_cart))
                    out.append((i, j, (int(d1), int(d2)), r, delta_cart))
    return out

def cluster_shells(distances: List[float], tol_shell: float = 1e-6) -> List[float]:
    uniq = []
    for r in sorted(distances):
        if not uniq or abs(r - uniq[-1]) > tol_shell:
            uniq.append(r)
    return uniq

def build_shell_catalog_indexed(a_lattice: np.ndarray,
                                basis_frac: np.ndarray,
                                search_R: int = 2,
                                tol_shell: float = 1e-6):
    """
    Returns:
      radii: list of shell radii (ascending; for info/logging)
      shells: list where shells[s] is a list of entries [ ((i,j), ΔR), ... ]
    """
    pairs = enumerate_pairs(a_lattice, basis_frac, search_R=search_R)
    radii = cluster_shells([r for *_ , r, _ in pairs], tol_shell=tol_shell)

    def assign_shell(r):
        idx = int(np.argmin([abs(r - R) for R in radii]))
        if abs(r - radii[idx]) > tol_shell:
            return None
        return idx

    shells = [[] for _ in radii]
    for i, j, dR, r, _ in pairs:
        s = assign_shell(r)
        if s is not None:
            shells[s].append(((i, j), dR))

    for s in range(len(shells)):
        shells[s].sort(key=lambda x: (x[0][0], x[0][1], x[1][0], x[1][1]))

    return radii, shells

def _buckets_from_entries(entries, norb: int):
    """
    From entries [((i,j), ΔR), ...], build ΔR buckets with boolean adjacency masks A (norb x norb).
    Ensures Hermitian partner (-ΔR, A^T) exists. Returns (deltas, A_list).
    """
    buckets = {}
    for (i, j), dR in entries:
        key = tuple(dR)
        if key not in buckets:
            buckets[key] = np.zeros((norb, norb), dtype=bool)
        buckets[key][i, j] = True

    keys = set(buckets.keys())
    for dR, A in list(buckets.items()):
        dR_neg = (-dR[0], -dR[1])
        if dR_neg not in keys:
            buckets[dR_neg] = A.T.copy()
            keys.add(dR_neg)

    deltas = sorted(buckets.keys())
    A_list = [buckets[d] for d in deltas]
    return deltas, A_list

def build_buckets_for_shell(a_lattice: np.ndarray,
                            basis_frac: np.ndarray,
                            shell_index: int,
                            search_R: int = 2,
                            tol_shell: float = 1e-6):
    """
    One-shot convenience: given a shell index (0=NN, 1=NNN, ...), return
      radius, deltas, A_list
    for that shell only. No merging, no V_list.
    """
    radii, shells = build_shell_catalog_indexed(a_lattice, basis_frac, search_R=search_R, tol_shell=tol_shell)
    if shell_index < 0 or shell_index >= len(shells):
        raise IndexError(f"shell_index {shell_index} out of range [0, {len(shells)-1}]")
    norb = basis_frac.shape[0]
    deltas_s, A_s = _buckets_from_entries(shells[shell_index], norb)
    return radii[shell_index], deltas_s, A_s

def build_buckets_per_shell(a_lattice: np.ndarray,
                            basis_frac: np.ndarray,
                            nshells: int,
                            search_R: int = 2,
                            tol_shell: float = 1e-6):
    """
    Build aligned bond tables for multiple shells using the method from consistency test.
    
    Returns:
        radii: list of shell radii (ascending)
        A_bonds: aligned bond matrices (L, nshells, norb, norb) where L is number of unique deltas
        deltas_aligned: aligned delta vectors (L, 2) corresponding to A_bonds
    """
    # Get shell catalog
    radii, shells = build_shell_catalog_indexed(a_lattice, basis_frac, search_R=search_R, tol_shell=tol_shell)
    
    # Limit to requested number of shells
    nshells = min(nshells, len(shells))
    radii = radii[:nshells]
    shells = shells[:nshells]
    
    norb = basis_frac.shape[0]
    
    # Build dictionaries for each shell
    shell_dicts = []
    for s in range(nshells):
        deltas_s, A_s = _buckets_from_entries(shells[s], norb)
        D_s = {d: A for d, A in zip(deltas_s, A_s)}
        shell_dicts.append(D_s)
    
    # Get all unique delta keys across all shells
    all_keys = set()
    for D_s in shell_dicts:
        all_keys.update(D_s.keys())
    keys = sorted(all_keys)
    
    # Create zero matrix for missing entries
    Z = np.zeros((norb, norb), dtype=bool)
    
    # Build aligned stacks for each shell
    A_aligned_per_shell = []
    for s in range(nshells):
        D_s = shell_dicts[s]
        A_s_aligned = np.stack([D_s.get(d, Z) for d in keys], axis=0)  # (L, norb, norb)
        A_aligned_per_shell.append(A_s_aligned)
    
    # Stack all shells along a new axis: (L, nshells, norb, norb)
    A_bonds = np.stack(A_aligned_per_shell, axis=1)
    
    # Convert keys to numpy array
    deltas_aligned = np.array(keys, dtype=int)
    
    return radii, A_bonds, deltas_aligned
