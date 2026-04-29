# HFB Toolkit — 2D Attractive Hubbard Model

Self-consistent mean-field study of superconductivity in the 2D Hubbard model,
including static ground-state HFB, finite-temperature extensions, and real-time
linear response via TDHFB.  Results are stored in HDF5 and analyzed with a
collection of Python visualization scripts.

---

## Theory Overview

The attractive Hubbard model on an L×L lattice,

```
H = -t Σ_{<ij>,σ} c†_{iσ} c_{jσ}  +  U Σ_i n_{i↑} n_{i↓}      (U < 0)
```

is treated at the mean-field level via Hartree-Fock-Bogoliubov theory.  The
key objects are the single-particle density matrix ρ and the anomalous (pairing)
density κ, packed into the 4N×4N generalized density matrix R.  Diagonalizing
the Bogoliubov-de Gennes (BdG) Hamiltonian H_BdG(ρ, κ) self-consistently gives
the quasiparticle spectrum and the superconducting gap Δ ∝ U⟨κ⟩.

See the companion documents for detailed physics:

- [hfb.md](DOCS/hfb.md) — static T=0 ground state (COMPLETE)
- [fthfb.md](DOCS/fthfb.md) — finite-temperature extension (INCOMPLETE, functions but is not production ready, can be run anyway)
- [tdhfb.md](DOCS/tdhfb.md) — real-time dynamics and linear response (INCOMPLETE, functions somewhat, does not return good answers or converge well. Very hard problem)

---

## Dependencies

| Library | Purpose | Install (Ubuntu) |
|---------|---------|------------------|
| Eigen3 (≥ 3.4, with `unsupported/MatrixFunctions`) | Linear algebra, matrix exponential | `apt install libeigen3-dev` |
| HDF5 C library | Binary result files | `apt install libhdf5-dev` |
| CMake ≥ 3.16 | Build system | `apt install cmake` |
| g++ with C++17 | Compiler | `apt install g++` |
| Python ≥ 3.8 | Visualizations | system or venv |
| h5py, numpy, matplotlib | Python analysis | `pip install -r visualizations/requirements.txt` |

---

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This produces four executables in `build/`:

| Executable | Description |
|-----------|-------------|
| `hfb` | T=0 ground state for a sweep of U values |
| `fthfb` | Finite-temperature ground state at a given kT |
| `tdhfb` | Real-time linear response (kick → propagate → S(E)) |
| `tdhfb_quench` | Imaginary-time relaxation after a parameter quench |

All executables must be run from the `build/` directory (or with explicit output
paths), so that HDF5 results land where the visualization scripts expect them.

---

## Quick Start

### Static ground state (T=0)

```bash
cd build
./hfb           # entropy-seeded
./hfb 42        # reproducible run with seed 42
```

Sweeps U ∈ {0, -1, …, -10} on a 4×4 lattice at half-filling.
Output: `hfb_results.h5`.

### Finite temperature

```bash
cd build
./fthfb 42 0.5   # seed=42, kT=0.5 t
```

Output: `fthfb_results.h5`.

### Real-time linear response

```bash
cd build
./tdhfb --x 4 --y 4 --U -2 --steps 8000 --kick density_q_pi_pi --output response.h5
```

Output: `response.h5` (ground state + time series + strength function S(E)).

### Imaginary-time quench

```bash
cd build
./tdhfb_quench --U-initial 0 --U-final -3 --dtau 0.005 --output quench.h5
```

Output: `quench.h5` (relaxation trajectory + converged ground state at U_final).

---

## Executables — Full Options

### `hfb`

```
./hfb [seed]
```

| Hardcoded parameter | Value |
|--------------------|-------|
| Lattice | 4×4 (16 sites) |
| Particle number | 16 (half-filling) |
| Hopping | t = 1.0 |
| U sweep | 0, -1, …, -10 |
| Temperature | T = 0 |
| Output | `hfb_results.h5` |

### `fthfb`

```
./fthfb [seed [kT]]
```

| Argument | Default | Meaning |
|---------|---------|---------|
| seed | entropy | RNG seed |
| kT | 0 | Temperature in units of t |

Same lattice and U sweep as `hfb`.
Output: `fthfb_results.h5`.

### `tdhfb`

```
./tdhfb [options]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--x N` | 4 | Lattice x dimension |
| `--y N` | 4 | Lattice y dimension |
| `--U V` | -2 | Hubbard U (negative = attractive) |
| `--t V` | 1.0 | Hopping amplitude |
| `--N V` | x*y | Particle number |
| `--dt V` | 0.005 | Time step (ℏ/t) |
| `--steps N` | 8000 | Number of propagation steps |
| `--eta V` | 1e-3 | Kick amplitude (keep small for linear response) |
| `--kick KIND` | `density_q_pi_pi` | See kick operators below |
| `--gamma V` | 0.5 | Lorentzian broadening for S(E) |
| `--emin V` | 0 | Lower bound of energy grid |
| `--emax V` | 8 | Upper bound of energy grid |
| `--negrid N` | 800 | Energy grid points |
| `--integrator S` | `expmid` | `expmid` (unitary, 2nd-order) or `euler` |
| `--sample-every N` | 1 | Record observables every N steps |
| `--output PATH` | `tdhfb_results.h5` | HDF5 output path |
| `--seed N` | random | RNG seed for HFB initialization |
| `--quiet` | — | Suppress per-step progress |

**Kick operators:**

| Kind | Operator | Use |
|------|---------|-----|
| `density_q_pi_pi` | Σ_i (-1)^(xi+yi) n_i | Staggered charge (antiferromagnetic wavevector) |
| `density_q_pi_0` | Σ_i (-1)^xi n_i | Stripe charge |
| `density_uniform` | Σ_i n_i | Total particle number |
| `pairing_real` | Σ_i (c†_i↑c†_i↓ + h.c.) | Higgs / pairing amplitude |
| `current_x` | i Σ_{⟨ij⟩,σ} (c†_i c_j − h.c.) | Optical current along x |

### `tdhfb_quench`

```
./tdhfb_quench [options]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--x N` | 4 | Lattice x dimension |
| `--y N` | 4 | Lattice y dimension |
| `--U-initial V` | 0 | Ground-state U before quench |
| `--U-final V` | -3 | U after quench |
| `--t V` | 1.0 | Hopping |
| `--N V` | x*y | Particle number |
| `--dtau V` | 0.005 | Imaginary-time step |
| `--max-steps N` | 100000 | Maximum iterations |
| `--tol V` | 1e-8 | Energy-convergence tolerance |
| `--save-every N` | 4 | Matrix snapshot frequency |
| `--mu-lr V` | 0.1 | Chemical-potential learning rate |
| `--output PATH` | `tdhfb_quench_results.h5` | HDF5 output path |
| `--seed N` | random | RNG seed |
| `--quiet` | — | Suppress progress |

---

## HDF5 Output Structure

All files share common top-level attributes: `kind`, `created_utc`, `x_sites`,
`y_sites`, `n_states`, `particle_number`, `t_hop`.

Complex matrices are stored as `(rows, cols, 2)` real datasets where the last
axis carries `[real, imag]`.

### Ground-state groups (`/hfb/U_*`)

| Item | Shape | Content |
|------|-------|---------|
| attr `U` | scalar | Hubbard parameter |
| attr `mu` | scalar | Converged chemical potential |
| attr `energy_real` | scalar | Ground-state energy |
| attr `iteration_count` | int | SCF iterations to convergence |
| attr `converged` | bool | Whether the SCF loop converged |
| `p` | (2N,2N,2) | Density matrix ρ |
| `k` | (2N,2N,2) | Pairing matrix κ |
| `R` | (4N,4N,2) | Generalized density matrix |
| `H_BdG` | (4N,4N,2) | BdG Hamiltonian |
| `H_BdG_evals` | (4N,) | BdG eigenvalues (real) |
| `H_BdG_evecs` | (4N,4N,2) | BdG eigenvectors |

Finite-temperature files additionally store `entropy` and `grand_potential`
per group.

### TDHFB groups (`/tdhfb/U_*__<kick_op>`)

Time series (length n_steps/sample_every): `times`, `F_real`, `F_imag`,
`energy`, `kinetic_energy`, `interaction_energy`, `particle_number`,
`idempotency_err`, `hermiticity_err`, `pairing_gap`.

Spectrum (length negrid): `E_grid`, `S_E`, `f_real`, `f_imag`.

Snapshots (4N×4N complex): `R_initial` (immediately after kick), `R_final`.

### Quench groups (`/quench/U_*`)

Time series over imaginary time: `tau`, `energy`, `particle_number`,
`idempotency_err`, `pairing_gap`, `mu_history`, `kinetic_energy`,
`interaction_energy`.

Matrix snapshots (4-D, shape `(n_snapshots, 4N, 4N, 2)`): stored every
`--save-every` steps.

Final state: `R_final`.

---

## Reading Results in Python

```python
import h5py
import numpy as np

def load_complex(ds):
    arr = np.asarray(ds)
    return arr[..., 0] + 1j * arr[..., 1]

with h5py.File("hfb_results.h5", "r") as f:
    grp = f["hfb"]["U_-2.000000"]
    p = load_complex(grp["p"])      # 2N×2N density matrix
    k = load_complex(grp["k"])      # 2N×2N pairing matrix
    evals = np.asarray(grp["H_BdG_evals"])
    mu = grp.attrs["mu"]
    print(f"mu = {mu:.4f},  gap = {evals[evals > 0].min():.4f}")
```

---

## Visualizations

Install dependencies once:

```bash
pip install -r visualizations/requirements.txt
```

All scripts accept `--file PATH` (default: the natural `.h5` name for that
script), `--nx N`, `--ny N`, `--u GROUP`, and `--animate-u` (sweep over all
U values as an animation).  Run from the project root.

| Script | What it shows |
|--------|--------------|
| `plot_phase.py` | Order parameter Δ̄ vs U — the phase diagram |
| `plot_energies.py` | BdG eigenvalue spectrum vs U — gap opening |
| `plot_quasiparticle.py` | Lowest quasiparticle density on the lattice |
| `plot_ph_mixing.py` | Particle-hole decomposition W_u / W_v vs U |
| `plot_coherence.py` | Off-diagonal coherence &#124;ρ_{ij}&#124; vs distance |
| `plot_momentum.py` | Momentum distribution n(k_x, k_y) |
| `plot_kappa.py` | Top pairing bonds drawn on the lattice |
| `plot_density.py` | Full density and pairing matrices as heatmaps |

Example — animate the quasiparticle density as U decreases:

```bash
python visualizations/plot_quasiparticle.py \
    --file build/hfb_results.h5 --animate-u --nx 4 --ny 4 \
    --save qp_animation.gif
```

Example — phase diagram:

```bash
python visualizations/plot_phase.py --file build/hfb_results.h5
```


## Disclosure

   A Few different LLM's were used to spruce up the documentation and comments for public use and understanding. All code is my own.