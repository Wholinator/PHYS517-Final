# Finite-Temperature HFB

The `fthfb` executable extends the zero-temperature HFB calculation to finite
temperature using the Goodman finite-temperature BCS formalism.  At T > 0
quasiparticle modes are thermally occupied according to Fermi-Dirac statistics,
which smooths the Fermi surface, reduces the gap, and drives a
superconductor-to-normal transition at the critical temperature T_c.

---

## Why Finite Temperature?

At T=0 the generalized density matrix R = Σ_{E_n<0} |ψ_n⟩⟨ψ_n| is a
projector (R² = R).  This is a sharp Fermi surface.  At T > 0 each mode
acquires a partial occupation given by the Fermi-Dirac distribution at
energy E_n.  The BdG equations remain exactly the same; only the way
we populate the quasiparticle modes changes.

---

## Thermal Occupation

For a BdG eigenmode with energy E_n > 0, define the quasi-Fermi weight

```
f_n = 1 / (1 + exp(E_n / kT))
```

At T=0, f_n → 0 for all positive modes (fully unpaired).  At high T,
f_n → 1/2 (equipartition).

The generalized density matrix at finite T is

```
R(T) = Σ_n f_n |ψ_n⟩⟨ψ_n|
```

summed over **all** modes (both positive and negative energy).  This
replaces the T=0 step-function occupancy with a smooth Fermi edge.

Note that R(T)² ≠ R(T) for T > 0 — idempotency is broken by thermal
fluctuations, which is expected and correct.

---

## Self-Consistency at Finite T

The self-consistency loop is identical to the T=0 case (see [hfb.md](hfb.md))
except that step 6 uses the thermal R(T) above instead of a projector.
The pairing field Δ is thus reduced as T increases because the off-diagonal
κ is smaller when positive-energy modes are partially occupied.

The calculation is performed at fixed kT throughout — there is no
annealing from high T down to the target temperature.  Each SCF cycle is
evaluated directly at the requested kT.

---

## Thermodynamic Quantities

The code computes two additional observables not present in the T=0 output:

**Configuration entropy** (von Neumann / BCS entropy)

```
S = -kB Σ_n [ f_n ln f_n + (1-f_n) ln(1-f_n) ]
```

**Grand potential** (Goodman eq. 3.23, 3.27)

```
Ω = E_HFB  −  T S  −  μ N
```

where E_HFB is the mean-field energy evaluated with the thermal density
matrix.  Ω is the relevant thermodynamic potential at fixed μ and T;
its minimum over μ determines the equilibrium particle number.

---

## Phase Transition

The superconducting gap Δ̄ = mean_i|κ_{i↑,i↓}| decreases monotonically
with temperature and vanishes at the mean-field critical temperature T_c.
For the attractive Hubbard model at half-filling and moderate |U|/t this
is the BCS transition temperature.  The mean-field T_c overestimates the
true Berezinskii-Kosterlitz-Thouless transition temperature in 2D, but
captures the qualitative physics and the correct U-dependence.

---

## Running

```bash
cd build
./fthfb                # T=0, entropy-seeded
./fthfb 42             # T=0, seed 42
./fthfb 42 0.5         # kT = 0.5 t, seed 42
```

The program sweeps U ∈ {0, -1, …, -10} at the specified temperature.
Output: `fthfb_results.h5`.

Typical usage for a temperature sweep (run each T separately):

```bash
for kT in 0.1 0.2 0.5 1.0 2.0; do
    ./fthfb 42 $kT
    mv fthfb_results.h5 fthfb_kT${kT}.h5
done
```

---

## Output

Same HDF5 structure as `hfb_results.h5`, with two additional datasets
per group:

| Dataset / Attribute | Shape | Content |
|--------------------|-------|---------|
| attr `temperature` | scalar | kT in units of t |
| `entropy` | scalar | Configuration entropy S/k_B |
| `grand_potential` | scalar | Ω = E − TS − μN |

All other fields (p, k, R, H_BdG, H_BdG_evals, H_BdG_evecs, μ, …)
have the same meaning and layout as in the T=0 output.

---

## Key Differences from T=0

| Property | T=0 (hfb) | Finite T (fthfb) |
|---------|-----------|-----------------|
| Occupation | Step function | Fermi-Dirac f_n |
| Idempotency | R² = R exactly | R² ≠ R |
| Gap | Maximum for given U | Reduced; zero above T_c |
| Entropy | 0 | S > 0 |
| Grand potential | Not computed | Ω stored |
| Chemical potential bisection | Same algorithm | Same algorithm |

---

## Visualization

The same visualization scripts used for `hfb_results.h5` work without
modification on `fthfb_results.h5`:

```bash
# Phase diagram at this temperature
python visualizations/plot_phase.py --file build/fthfb_results.h5

# Momentum distribution
python visualizations/plot_momentum.py --file build/fthfb_results.h5 \
    --animate-u --nx 4 --ny 4

# Particle-hole mixing
python visualizations/plot_ph_mixing.py --file build/fthfb_results.h5 \
    --animate-u --nx 4 --ny 4
```

To compare T=0 and finite-T results:

```python
import h5py, numpy as np

def get_gap(fname):
    with h5py.File(fname, "r") as f:
        u_vals, gaps = [], []
        for key in sorted(f["hfb"].keys()):
            grp = f["hfb"][key]
            k = np.asarray(grp["k"])
            k_c = k[..., 0] + 1j * k[..., 1]
            N = k_c.shape[0] // 2
            delta = np.mean(np.abs([k_c[i, N+i] for i in range(N)]))
            u_vals.append(grp.attrs["U"])
            gaps.append(delta)
    return np.array(u_vals), np.array(gaps)

u0, g0 = get_gap("build/hfb_results.h5")
u1, g1 = get_gap("build/fthfb_kT0.5.h5")
```
