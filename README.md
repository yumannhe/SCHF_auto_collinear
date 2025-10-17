SCHF Auto Collinear
===================
**Paper:** arXiv: [2510.14593](https://arxiv.org/abs/2510.14593)


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


