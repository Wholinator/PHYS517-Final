# Time-Dependent HFB

The `tdhfb` and `tdhfb_quench` executables extend static HFB to real-time
dynamics.  `tdhfb` computes linear-response strength functions by real-time
propagation.  `tdhfb_quench` finds the new ground state after a sudden
parameter quench via imaginary-time evolution.

---

## TDHFB Equation of Motion

The generalized density matrix R(t) satisfies the TDHFB equation

```
i ∂_t R = [ H[R(t)] − μ N̂_BdG ,  R(t) ]
```

where H[R(t)] is the BdG Hamiltonian rebuilt self-consistently at each
instant from the current R(t), and N̂_BdG = diag(1, …, 1, −1, …, −1) is
the particle-number operator in Nambu space.

This is the time-dependent analog of the static self-consistency condition.
The static ground state satisfies [H[R], R] = 0 and is therefore stationary
under this equation — a useful sanity check.

The TDHFB equation preserves:
- **Energy**: d/dt ⟨H⟩ = 0 (Hamiltonian is autonomous)
- **Particle number**: d/dt Tr(ρ) = 0
- **Idempotency**: R² = R is preserved exactly (by unitarity)

---

## Linear Response

The standard use case is linear-response spectroscopy: kick the ground
state R₀ with a one-body operator F, propagate for time T, and
Fourier-transform the expectation value ⟨F(t)⟩.

**Step 1 — Kick**

Apply a unitary perturbation at t = 0:

```
R(0) = e^{iηF̃} R₀ e^{-iηF̃}
```

where F̃ = block_diag(F, −F̄) extends the single-particle operator F
to Nambu space, and η is a small amplitude (η = 1e-3 by default).

**Step 2 — Propagate**

Evolve R(t) under the TDHFB equation for T = n_steps × dt.

**Step 3 — Fourier transform**

The linear-response strength function is extracted as

```
S(E; F) = −(1/π) Im [ ∫_0^T dt e^{iEt} e^{-Γt} δ⟨F(t)⟩ ]
```

where δ⟨F(t)⟩ = ⟨F(t)⟩ − ⟨F(0)⟩ and Γ is a Lorentzian broadening
parameter (--gamma, default 0.5 t).  The exponential damping windowing
suppresses Gibbs oscillations from the finite propagation time.

S(E; F) gives the spectral weight that F couples to at energy E.  For a
density operator at wavevector q, this is the charge structure factor
S(q, E).  For the pairing operator, it is the Higgs / amplitude mode spectrum.

**Sum rule check**

The first moment of S(E) satisfies the exact sum rule

```
m₁ = ∫ E S(E) dE  =  ½ ⟨0| [F, [H, F]] |0⟩
```

The code computes both sides and reports the ratio.  Agreement to within ~1%
over T = 40 ℏ/t confirms the calculation is in the linear regime and the
integrator is accurate.

---

## Unitary Integrator (expmid)

The default integrator is the matrix-exponential midpoint rule, which
propagates R by a unitary transformation:

```
R(t + dt) = U R(t) U†,     U = exp(-i dt H_mid)
```

The midpoint Hamiltonian H_mid is estimated by a predictor-corrector step:

```
R_pred  = exp(-i dt H[R(t)])  R(t)  exp(+i dt H[R(t)])   (Euler predictor)
H_mid   = ½ ( H[R(t)] + H[R_pred] )                       (midpoint average)
```

This is a second-order, symplectic-like scheme.  Because the propagator is
unitary (U†U = 1), R² = R is preserved to machine precision regardless of
step size.  Energy and particle number are conserved to O(dt²) per step,
accumulating to O(dt²) drift over a run of fixed total time T.

Cost per step: two eigendecompositions of the 4N×4N BdG matrix.  For the
default 4×4 lattice (4N = 64), each step takes ~5 ms.  A run of 8000 steps
takes ~40 seconds.  Cost scales as O(N³) per step.

**Euler integrator** (--integrator euler)

The forward Euler scheme R(t+dt) = R(t) − i dt [H[R(t)], R(t)] is available
for debugging.  It is first-order and non-unitary: idempotency is not
preserved and energy drifts at O(dt).  It is faster per step (one fewer
eigendecomposition) but unsuitable for long runs or quantitative results.

---

## Imaginary-Time Quench (tdhfb_quench)

The quench driver computes the new ground state after an instantaneous
change U_initial → U_final by evolving in imaginary time τ = it.

The imaginary-time TDHFB equation is

```
∂_τ R = -[ H[R], [H[R], R] ]
```

This is gradient descent on the BdG energy functional.  The double commutator
is the Riemannian gradient of E[R] on the manifold of idempotent matrices.
R converges monotonically to the ground state of H[R_∞] as τ → ∞.

After each step R is explicitly purified (eigenvalues projected to {0, 1})
to maintain the idempotency constraint R² = R.  The chemical potential μ
is adjusted at every step by gradient descent to conserve Tr(ρ) = N, using
a learning rate of --mu-lr (default 0.1).

Convergence is declared when |ΔE / Δτ| < --tol (default 1e-8) between
successive steps.

The quench trajectory records the relaxation of the pairing gap, energy,
and idempotency error over imaginary time, which characterizes how quickly
the system finds the new superconducting ground state.

---

## Running — tdhfb

Basic real-time run (4×4, U=-2, staggered charge kick):

```bash
cd build
./tdhfb --x 4 --y 4 --U -2 --steps 8000 --kick density_q_pi_pi \
        --output response.h5
```

Higgs mode (pairing channel):

```bash
./tdhfb --x 4 --y 4 --U -5 --steps 16000 --dt 0.002 \
        --kick pairing_real --gamma 0.2 --emax 4 --output higgs.h5
```

Larger lattice with downsampled recording:

```bash
./tdhfb --x 8 --y 8 --U -2 --steps 4000 --sample-every 4 \
        --output response_8x8.h5
```

Reproducible run with progress suppressed:

```bash
./tdhfb --seed 42 --quiet --output run42.h5
```

Full option reference: see [README.md](README.md).

---

## Running — tdhfb_quench

Quench from normal metal (U=0) to superconductor (U=-3):

```bash
cd build
./tdhfb_quench --U-initial 0 --U-final -3 --dtau 0.005 \
               --max-steps 100000 --output quench.h5
```

Quench between two superconducting states:

```bash
./tdhfb_quench --U-initial -2 --U-final -8 --dtau 0.01 \
               --tol 1e-10 --save-every 10 --output quench_deep.h5
```

---

## Output Schema

### tdhfb_results.h5

Ground-state data is stored under `/hfb/U_*` exactly as for `hfb_results.h5`.

TDHFB results are stored under `/tdhfb/U_*__<kick_operator>/`:

**Attributes:**
`U`, `dt`, `n_steps`, `T_total`, `eta_kick`, `Gamma_smooth`,
`mu_fixed`, `kick_operator`, `integrator`, `m1_FT`, `m1_commutator`.

**Time series** (length = n_steps / sample_every):

| Dataset | Content |
|---------|---------|
| `times` | t values |
| `F_real`, `F_imag` | Re/Im ⟨F(t)⟩ |
| `energy` | Total HFB energy |
| `kinetic_energy` | Hopping contribution |
| `interaction_energy` | U contribution |
| `particle_number` | Tr(ρ) — should be constant |
| `idempotency_err` | ‖R² − R‖_F — should stay ~1e-10 |
| `hermiticity_err` | ‖R − R†‖_F |
| `pairing_gap` | mean|κ_{ii}| |

**Spectrum:**

| Dataset | Content |
|---------|---------|
| `E_grid` | Energy grid |
| `S_E` | Strength function S(E; F) |
| `f_real`, `f_imag` | Complex spectrum before symmetrization |

**Snapshots:**

| Dataset | Content |
|---------|---------|
| `R_initial` | 4N×4N R immediately after kick |
| `R_final` | 4N×4N R at end of run |

### tdhfb_quench_results.h5

Data stored under `/quench/U_0_to_U_-3/` (keys encode U_initial and U_final).

**Attributes:** `U_initial`, `U_final`, `dtau`, `tol`, `n_steps_run`.

**Imaginary-time series:**

| Dataset | Content |
|---------|---------|
| `tau` | Imaginary time values |
| `energy` | Total energy (monotonically decreasing) |
| `particle_number` | Should remain ≈ N |
| `idempotency_err` | ‖R² − R‖_F after purification |
| `pairing_gap` | mean|κ_{ii}| — builds up from zero |
| `mu_history` | Chemical potential trajectory |
| `kinetic_energy` | Hopping contribution |
| `interaction_energy` | U contribution |

**Matrix snapshots** (shape `(n_snapshots, 4N, 4N, 2)`):
stored every `--save-every` steps; last axis is real/imag.

**Final state:** `R_final` — converged ground state at U_final.

---

## Reading Results in Python

```python
import h5py, numpy as np

def load_complex(ds):
    arr = np.asarray(ds)
    return arr[..., 0] + 1j * arr[..., 1]

with h5py.File("response.h5", "r") as f:
    key = list(f["tdhfb"].keys())[0]
    td  = f["tdhfb"][key]

    t   = np.asarray(td["times"])
    Ft  = np.asarray(td["F_real"]) + 1j * np.asarray(td["F_imag"])
    E   = np.asarray(td["E_grid"])
    SE  = np.asarray(td["S_E"])

    print(f"m1_FT  = {td.attrs['m1_FT']:.4f}")
    print(f"m1_com = {td.attrs['m1_commutator']:.4f}")
    print(f"ratio  = {td.attrs['m1_FT'] / td.attrs['m1_commutator']:.4f}")
```

Quick diagnostic plot:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes[0, 0].plot(t, Ft.real)
axes[0, 0].set(title="Re⟨F(t)⟩", xlabel="t (ℏ/t)")

axes[0, 1].plot(E, SE)
axes[0, 1].set(title="S(E; F)", xlabel="E (t)")

energy = np.asarray(td["energy"])
axes[1, 0].plot(t, energy - energy[0])
axes[1, 0].set(title="Energy drift", xlabel="t", yscale="symlog", linthresh=1e-12)

idem = np.asarray(td["idempotency_err"])
axes[1, 1].plot(t, idem)
axes[1, 1].set(title="‖R² − R‖_F", xlabel="t", yscale="log")

plt.tight_layout()
plt.savefig("diagnostics.png", dpi=120)
```

---

## Validation

On the 4×4 lattice, U = −2, t = 1, half-filling, with the `expmid` integrator:

| Test | Result |
|------|--------|
| Stationarity (η=0): energy drift over 1000 steps | 1.4 × 10⁻¹⁰ |
| Conservation (T=40): ΔE | 1.7 × 10⁻⁹ |
| Conservation (T=40): ΔN | 4.5 × 10⁻¹⁰ |
| Conservation (T=40): ‖R²−R‖_F | 1.5 × 10⁻¹⁰ |
| Linearity: S(E) at η=1e-3 vs 2.5e-4 relative difference | 4 × 10⁻⁴ |
| Sum rule: m₁(FT) / m₁(commutator) | 0.99 |

These confirm that the unitary integrator conserves all constants of motion
to near-machine-precision, the response is linear at η = 1e-3, and the
sum rule converges to the analytic value as T grows.

---

## Cost Estimates

For the `expmid` integrator (2 eigendecompositions per step):

| Lattice | BdG dim | ms / step | 8000-step run |
|---------|---------|-----------|--------------|
| 4×4 | 64 | ~5 | ~40 s |
| 6×6 | 144 | ~30 | ~4 min |
| 8×8 | 256 | ~130 | ~18 min |
| 16×16 | 1024 | ~8000 | ~18 hr |

For large runs, reduce cost by:
- Increasing `--sample-every` (fewer snapshots stored, same propagation cost)
- Reducing `--steps` (shorter T, coarser energy resolution ΔE ≈ 2π/T)
- Using `--integrator euler` (1 eigendecomposition per step; verify conservation)
- Increasing `--dt` (fewer steps for same T; check idempotency and energy drift)
