# TDHFB on the 2D Hubbard Model

Time-Dependent Hartree-Fock-Bogoliubov implementation, building on the
existing static HFB code (`hfb_numeric_current.cpp`).

## Files

| File | Role |
|---|---|
| `hfb_numeric_current.cpp` | Static HFB solver (patched: stores μ + lattice metadata; original `main` guarded by `HFB_STANDALONE_MAIN`) |
| `hfb_api.h` | Header exposing HFB internals to the TDHFB module |
| `data_output.h` | `HFBResult` and `TDHFBResult` struct definitions |
| `data_output.cpp` | HDF5 writers for both result types |
| `tdhfb.h` | TDHFB API: `TDHFBParams`, `TDHFBResult`, `run_tdhfb`, kick/propagator helpers |
| `tdhfb.cpp` | TDHFB implementation: matrix-exp midpoint integrator, kick, FT |
| `tdhfb_main.cpp` | CLI driver |

## Build

Requires Eigen3 (with the `unsupported/MatrixFunctions` module) and the
HDF5 C library. On Ubuntu:

```
apt-get install libeigen3-dev libhdf5-dev
```

Compile and link:

```
g++ -O2 -std=c++17 \
    -I/usr/include/eigen3 -I/usr/include/hdf5/serial \
    hfb_numeric_current.cpp data_output.cpp tdhfb.cpp tdhfb_main.cpp \
    -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 \
    -o tdhfb
```

For a faster build, use `-O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG`.

## Run

```
./tdhfb --x 4 --y 4 --U -2 --steps 8000 --dt 0.005 --output run.h5
```

Full options:

```
--x N              lattice x dim (default 4)
--y N              lattice y dim (default 4)
--U V              on-site Hubbard U (default -2; negative = attractive)
--t V              hopping (default 1.0)
--N V              particle number (default = STATES = x*y)
--dt V             time step (default 0.005, units ℏ/t)
--steps N          number of steps (default 8000; T = dt·n_steps)
--eta V            kick amplitude (default 1e-3, must be small for linearity)
--kick KIND        density_q_pi_pi (default), density_q_pi_0,
                   density_uniform, pairing_real, current_x
--gamma V          Lorentzian smoothing for S(E) (default 0.5)
--emin V           min E in S(E) grid (default 0)
--emax V           max E in S(E) grid (default 8)
--negrid N         number of E points (default 800)
--integrator S     "expmid" (default; unitary, 2nd order) or "euler"
--sample-every N   downsample observable storage (default 1)
--output PATH      output HDF5 file (default tdhfb_results.h5)
--seed N           RNG seed for HFB initialization
--quiet            suppress per-step progress
```

## HDF5 output schema

Top-level attributes:
- `kind` ("hfb_ground_state" or "tdhfb_run"), `created_utc`
- `x_sites`, `y_sites`, `n_states`, `particle_number`, `t_hop`

### `/hfb/U_*` — ground state
- attrs: `U`, `temperature`, `mu`, `energy_real`, `energy_imag`,
  `iteration_count`, `converged`
- datasets: `p`, `k`, `R`, `H_BdG`, `H_BdG_evecs` (complex,
  shape `(rows, cols, 2)` with last axis = real/imag),
  `H_BdG_evals` (real)

### `/tdhfb/U_*__<kick_op>` — TDHFB run
- attrs: `U`, `dt`, `n_steps`, `T_total`, `eta_kick`, `Gamma_smooth`,
  `mu_fixed`, `kick_operator`, `integrator`, `m1_FT`, `m1_commutator`
- 1-D time series: `times`, `F_real`, `F_imag`,
  `energy`, `kinetic_energy`, `interaction_energy`,
  `particle_number`, `idempotency_err`, `hermiticity_err`,
  `pairing_gap`
- 1-D spectra: `E_grid`, `S_E`, `f_real`, `f_imag`
- 3-D snapshots: `R_initial` (post-kick), `R_final`

## Reading in Python

```python
import h5py, numpy as np
f = h5py.File("run.h5", "r")
td = f["tdhfb"][list(f["tdhfb"].keys())[0]]
E = np.asarray(td["E_grid"]); S = np.asarray(td["S_E"])
# complex matrix:
R0 = np.asarray(td["R_initial"]); R0 = R0[...,0] + 1j*R0[...,1]
```

## Validation status

On 4x4 lattice, U = -2, t = 1, half-filling:

- **T1 stationarity (eta=0):** energy drift 1.4e-10 over 1000 steps
- **T2 conservation:** dE = 1.7e-9, dN = 4.5e-10, |R^2 - R| = 1.5e-10 over T=40
- **T3 linearity:** S(E) at eta=1e-3 vs 2.5e-4 agrees to 4e-4 relative
- **T8 sum rule:** m1(FT)/m1(commutator) = 0.99 (within 1%)

The matrix-exp midpoint integrator is unitary by construction; energy,
particle number, and idempotency are preserved to ~1e-10 over thousands
of steps. Sum rule converges to the analytic value as T grows.

## Cost

Per step: 2 eigendecompositions of the 4N x 4N BdG matrix, where
N = x*y. For 4x4 (BdG dim 64), ~5 ms/step. For larger lattices,
cost scales as O(N^3) per step. 16x16 (BdG dim 1024) will take
~hours per long run; consider increasing `--sample-every`,
reducing `--steps`, or using `--integrator euler` (faster but
not unitary; verify conservation in the diagnostics).

## Quick analysis snippet

```python
import h5py, numpy as np, matplotlib.pyplot as plt
f = h5py.File("run.h5", "r")
td = f["tdhfb"][list(f["tdhfb"].keys())[0]]

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0,0].plot(td["times"], td["F_real"]); ax[0,0].set(title="<F>(t) - <F>(0)", xlabel="t")
ax[0,1].plot(td["E_grid"], td["S_E"]);   ax[0,1].set(title="S(E; F)", xlabel="E")
ax[1,0].plot(td["times"], np.asarray(td["energy"]) - td["energy"][0])
ax[1,0].set(title="energy drift", xlabel="t", yscale="symlog", linthresh=1e-12)
ax[1,1].plot(td["times"], td["idempotency_err"])
ax[1,1].set(title="|R^2 - R|_F", xlabel="t", yscale="log")
plt.tight_layout(); plt.savefig("tdhfb_diagnostics.png", dpi=120)
```
