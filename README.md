SCHF Auto Collinear
===================

Self‑consistent Hartree–Fock (SCHF) calculations for collinear spin systems, written with JAX for fast CPU/GPU execution. The code builds tight‑binding Hamiltonians from precomputed bond tables, supports multi‑shell density–density interactions, and offers vectorized and pmap‑parallel solvers for large parameter sweeps and phase-diagram generation.


Key Features
------------
- Flexible hopping/interactions: build aligned bond tables for NN, NNN, … shells and extend systematically to longer range.
- Two unit‑cell choices: 1×1 and 2×2 supercell workflows are included.
- JAX acceleration: single‑job solver plus `vmap`/`pmap` runners for parameter grids and multi‑device scaling.
- Analysis notebooks: phase diagrams and plotting utilities with consistent aesthetics.


Project Layout
--------------
- `functions_parameters/`: core numerics and helpers
  - `jax_schf_kernel.py`: SCHF single‑job kernel and parallel runners (`schf_single_job`, `schf_fixed_filling_pmap_over_u`, …)
  - `jax_schf_helpers.py`: precomputation (k‑phases, H(k)), reference state, mean‑field pieces
  - `bond_table_generator.py`: build aligned per‑shell bond tables from lattice + basis
  - `universal_parameters.py`: default lattice, k‑mesh, TB parameters, symmetry ops
  - `phase_plot.py`: consistent plotting for phase‑maps
- `parallel_scripts/`: ready‑to‑run examples for 1×1 and 2×2 setups
- `data_analysis/`: Jupyter notebooks for phase diagrams; `.npz` results live alongside
- `test_codes/`: minimal checks and references


Installation
------------
Prereqs: Python 3.11+ (JAX ≥ 0.7). For GPU, install the matching `jaxlib` for your CUDA/ROCm.

Option A: quick dev install
1) Create/activate a virtual environment (conda or venv)
2) From the repo root:
   - `pip install -e .`
   - Optional: notebooks and dev tools → `pip install -e .[jupyter]` and/or `pip install -e .[dev]`

Option B: use the helper script
- `bash install.sh`

Notes on JAX
- CPU‑only: `pip install --upgrade jax jaxlib` is sufficient.
- GPU: follow JAX’s official wheel selector to install a CUDA/ROCm‑compatible `jaxlib`.


Quick Start (Scripts)
---------------------
All commands run from the repo root. The scripts save compressed results (`.npz`) into the repo so notebooks can pick them up.

- 1×1 unit cell, fixed filling sweep over `u` and `v1` (with `v2=0`):
  - `python parallel_scripts/schf_fixed_filling_pmap_u_v1_v2_random_basis.py`

- 2×2 supercell, fixed filling sweep over `u` and `(v1, v2)`:
  - `python parallel_scripts/schf_fixed_filling_pmap_u_v1_v2_random_basis_2_2.py`

- Fixed `u:v` ratio, sweep over filling (examples vary by file name):
  - `python parallel_scripts/schf_fixed_u_v_ratio_pmap_filling_random_basis.py`
  - `python parallel_scripts/schf_fixed_u_v_ratio_pmap_filling_random_basis_2_2.py`

Typical output keys in `.npz`
- `u`, `v`: scanned interaction grids
- `d`: converged on‑site densities `(channels, 2, norb)`
- `bond`: bond orders `(channels, deltas, 2, nshells, norb, norb)`
- `gse`: total ground‑state energy (with corrections)
- `e_fermi`: Fermi level; `e_diff`, `c_diff`: energy/order convergence deltas
- `iters`: iteration counts; `any_bi_fail`: any Fermi‑level bisection failures


Running the Notebooks
---------------------
Notebooks under `data_analysis/` expect the `.npz` files produced by the scripts above.

Recommended flow
1) Generate data with a script (see Quick Start).
2) Start Jupyter Lab/Notebook in the repo: `jupyter lab`.
3) Open a notebook such as `data_analysis/over_v/phase_diagram_filling_vhs_t_4em4_2_2_over_v1_v2_seed_121.ipynb` and run.

Plotting
- The helper `functions_parameters/phase_plot.py` enforces consistent colormap, symbol shapes, and magnetic ring encoding.
- Adjust global aesthetics once via `PlotConfig` for paper‑wide consistency.


Minimal API Example (single job)
--------------------------------
Below is a compact recipe to run one SCHF solve programmatically. Adapt to your lattice and shells.

```python
import jax, jax.numpy as jnp
import numpy as np
from functions_parameters.bond_table_generator import build_buckets_per_shell
from functions_parameters.jax_schf_helpers import (
    precompute_k_phase_tables, hk_all_k_from_phases, prepare_reference_state
)
from functions_parameters.jax_schf_kernel import schf_single_job

# Lattice + basis (example 1×1, 3 orbitals per cell)
from functions_parameters.universal_parameters import a, b
basis_frac = np.array([[1/2, 0], [1/2, 1/2], [0, 1/2]])

# k‑mesh (square grid over BZ)
num_k = 60
b0 = np.linspace(-b[0]/2, b[0]/2, num_k, endpoint=False)
b1 = np.linspace(-b[1]/2, b[1]/2, num_k, endpoint=False)
k_points = np.vstack([x + y for x in b0 for y in b1])

# Build aligned bond tables up to NNN (nshells=2)
radii, a_list, deltas = build_buckets_per_shell(a, basis_frac, nshells=2)
phase_pos, phase_neg = precompute_k_phase_tables(deltas, a, k_points)

# Tight‑binding H(k) and eigenpairs
mu = 2.0; t_nn = 1.0; t_nnn = -0.025
t_arr = np.array([t_nn, t_nnn])
Htb, e_all, v_all, v_all_dagger = hk_all_k_from_phases(mu, a_list, t_arr, phase_neg)

# Reference state at given filling and temperature
filling = 1/2   # electrons per cell (spin included in helper)
kT = 4e-4
ref = prepare_reference_state(filling, a_list, Htb, e_all, v_all, v_all_dagger, phase_pos, phase_neg, kT)

# Initial order parameters (random basis or heuristics)
norb = Htb.shape[-1]
ndelta = deltas.shape[0]
nshells = 2
init_d = jnp.zeros((2, norb), dtype=jnp.complex128)
init_bond = jnp.zeros((ndelta, 2, nshells, norb, norb), dtype=jnp.complex128)

# Interactions and solve
u = 0.4
v_vec = jnp.array([0.2, 0.0])  # (v1, v2)
out = schf_single_job(
    Htb=jnp.stack((jnp.asarray(Htb), jnp.asarray(Htb))),
    a_list=jnp.asarray(a_list),
    phase_pos=jnp.asarray(phase_pos),
    phase_neg=jnp.asarray(phase_neg),
    dict_ref=ref,
    input_d=init_d,
    input_bond=init_bond,
    filling=filling*2,   # count spin explicitly in kernel
    u=u,
    v_arr=v_vec,
    temperature=kT,
)
print(out["gse"], out["e_diff"], out["c_diff"], out["iters"])
```


Extending Range and Unit Cells
------------------------------
- Longer‑range hopping/interactions: increase `nshells` and the length of `t_arr`; `build_buckets_per_shell(...)` returns aligned delta buckets and bond matrices that scale to any finite range. See `functions_parameters/bond_table_generator.py`.
- Supercells: follow the 2×2 examples in `parallel_scripts/` to redefine the basis, lattice vectors, and k‑mesh consistently.


Performance Tips
----------------
- Prefer the `pmap` runners for multi‑device GPUs; they chunk work to match `jax.local_device_count()`.
- Start small (few k‑points, short grids) to JIT‑compile once; then scale up.
- If you hit GPU memory limits, reduce `num_k`, sweep granularity, or channels; or run CPU‑only for smoke tests.


Troubleshooting
---------------
- JAX install: ensure `jax`/`jaxlib` versions match (and match CUDA/ROCm if used).
- Entry points: a CLI stub may be present but not wired for every script; prefer running the Python files in `parallel_scripts/` directly.
- Reproducibility: some inputs use precomputed random bases (e.g., `..._121.npy`, `..._269.npy`). Use those for consistent comparisons.


License and Citation
--------------------
- License: MIT (see `pyproject.toml`).
- If you use this project in academic work, please cite the repository and the specific notebooks/scripts used to generate figures.


Acknowledgements
----------------
This codebase builds on the idea of precomputing bond tables and aligning per‑shell adjacency to decouple physics choices (range, unit cell) from solver mechanics. With the pregenerated bond tables, one can systematically extend the Hamiltonian from NN/NNN to arbitrary finite range; the same applies to interaction shells. Enlarging unit cells (e.g., 2×2) follows the same pattern.


