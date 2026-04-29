# Static HFB — T=0 Ground State

The `hfb` executable solves the self-consistent Hartree-Fock-Bogoliubov
equations for the 2D attractive Hubbard model at zero temperature.

---

## Physical Setup

The model is the Hubbard Hamiltonian on an L×L square lattice with
periodic boundary conditions,

```
H = -t Σ_{<ij>,σ} c†_{iσ} c_{jσ}  +  U Σ_i n_{i↑} n_{i↓}
```

with U < 0 (attractive interaction).  We work at fixed particle number N
(chemical potential μ adjusted self-consistently to enforce ⟨N̂⟩ = N).

Energies are in units of the hopping t = 1.  The default run uses a 4×4
lattice at half-filling (N = 16).

---

## The HFB Ansatz

HFB theory decouples the interaction at the level of a Bogoliubov
transformation.  Instead of tracking the full many-body state, we track
two mean-field objects:

**Density matrix**
```
ρ_{iσ,jσ'} = ⟨c†_{jσ'} c_{iσ}⟩
```

**Anomalous (pairing) density**
```
κ_{iσ,jσ'} = ⟨c_{jσ'} c_{iσ}⟩
```

These are combined into the 4N×4N **generalized density matrix**

```
R = ( ρ     κ  )
    ( κ†  1−ρ* )
```

which satisfies R² = R (idempotency) and 0 ≤ R ≤ 1 at T=0.

---

## BdG Hamiltonian

From ρ and κ we build the mean-field single-particle Hamiltonian and
pairing field:

**Fock (kinetic + exchange) matrix**
```
h_{iσ,jσ'} = -t δ_{⟨ij⟩} δ_{σσ'}  −  U ρ_{jσ',iσ}
```
The Hartree shift is zero by spin symmetry at half-filling.

**Pairing (gap) field**
```
Δ_{iσ,jσ'} = U κ_{iσ,jσ'}
```
The on-site singlet gap amplitude is Δ_i = U κ_{i↑, i↓}.

These enter the **4N×4N BdG Hamiltonian**

```
H_BdG = ( h − μ       Δ   )
        ( Δ†      −h* + μ  )
```

Diagonalizing H_BdG gives quasiparticle energies E_n and
Bogoliubov amplitudes u_n, v_n:

```
H_BdG ( u_n ) = E_n ( u_n )
       ( v_n )       ( v_n )
```

---

## Self-Consistency Loop

The HFB equations are solved iteratively:

```
1.  Initialize R randomly (or from previous iteration)
2.  Extract ρ, κ from R
3.  Build h[ρ] and Δ[κ]
4.  Bisect μ to satisfy Tr(ρ) = N
5.  Diagonalize H_BdG(μ)
6.  Reconstruct R from occupied quasiparticle modes
7.  Check convergence:  ||H_BdG(new) − H_BdG(old)||_F < 1e-9
8.  If not converged, return to step 2 (with DIIS update)
```

The T=0 density matrix is reconstructed by occupying all quasiparticle
modes with E_n < 0:

```
R = Σ_{E_n < 0} |ψ_n⟩⟨ψ_n|
```

where |ψ_n⟩ is the corresponding BdG eigenvector.

---

## DIIS Acceleration

Bare fixed-point iteration is slow near the solution.  DIIS (Direct
Inversion in the Iterative Subspace) stores the last K BdG Hamiltonians
{H_k} and their residuals {r_k = H_k − H_{k−1}}, then solves for
coefficients {c_k} that minimize the norm of the extrapolated residual.
The next iterate is

```
H_next = Σ_k c_k H_k    subject to  Σ_k c_k = 1
```

This is equivalent to solving a small (K+1)×(K+1) linear system at
each step.  In practice K = 8 to 12 suffices and convergence typically
reaches the 1e-9 threshold in O(100) iterations rather than O(1000).

The DIIS subspace is discarded and restarted whenever the residual
increases — a safeguard against instability when the iterate is far
from the solution.

---

## Chemical Potential and Particle Number

At T=0 and U=0 the density of states on a finite lattice is a set of
discrete levels.  The chemical potential must sit in a gap between
levels to fix N exactly; if N falls on a degenerate shell the bisection
converges to an arbitrary value in the gap and the particle number will
be off by a small integer.  This is normal and harmless — it is an
artifact of finite-size level spacing, not a code error.  The U=0 case
is for reference only; the physically interesting regime is U ≪ 0.

---

## Running

```bash
cd build
./hfb         # random seed
./hfb 42      # seed 42 for reproducibility
```

The program prints convergence diagnostics for each U value:

```
U = -2.000  mu = -0.341  E = -23.847  iters = 124  converged
```

Output is written to `hfb_results.h5` in the working directory.

---

## Output

The HDF5 file contains one group per U value under `/hfb/`:

```
/hfb/U_0.000000/
/hfb/U_-1.000000/
...
/hfb/U_-10.000000/
```

Each group has:

| Dataset / Attribute | Shape | Content |
|--------------------|-------|---------|
| attr `U` | scalar | Hubbard interaction |
| attr `temperature` | scalar | 0 |
| attr `mu` | scalar | Chemical potential |
| attr `energy_real` | scalar | Total HFB energy |
| attr `iteration_count` | int | SCF iterations |
| attr `converged` | bool | Convergence flag |
| `p` | (2N,2N,2) | Density matrix ρ (real/imag split) |
| `k` | (2N,2N,2) | Pairing matrix κ |
| `R` | (4N,4N,2) | Generalized density matrix |
| `H_BdG` | (4N,4N,2) | Converged BdG Hamiltonian |
| `H_BdG_evals` | (4N,) | BdG eigenvalues |
| `H_BdG_evecs` | (4N,4N,2) | BdG eigenvectors |

Matrices with shape `(rows, cols, 2)` store `[real, imag]` in the last axis.
Reconstruct in Python as `arr[..., 0] + 1j * arr[..., 1]`.

---

## Observables

From the converged matrices one can extract:

**Superconducting gap**
```python
N = k.shape[0] // 2
kappa_diag = np.array([k[i, N+i] for i in range(N)])  # on-site κ_{i↑,i↓}
Delta_bar = np.mean(np.abs(kappa_diag))                 # mean gap amplitude
```

**BdG spectrum and gap**
```python
evals = np.sort(np.real(grp["H_BdG_evals"]))
gap = evals[evals > 0].min()    # lowest positive quasiparticle energy
```

**Quasiparticle density** (lowest excitation, index 0)
```python
evecs = load_complex(grp["H_BdG_evecs"])   # 4N×4N
pos = np.where(evals > 0)[0][np.argsort(evals[evals > 0])]
evec = evecs[:, pos[0]]
ns = N
u_up, u_dn = evec[:ns], evec[ns:2*ns]
v_up, v_dn = evec[2*ns:3*ns], evec[3*ns:]
density = np.abs(u_up)**2 + np.abs(u_dn)**2 + np.abs(v_up)**2 + np.abs(v_dn)**2
```

**Pairing bonds** (strongest off-diagonal κ)
```python
kappa = k[:N, N:]               # top-right N×N block
# upper triangle gives unique (i,j) pairs with i < j
i_idx, j_idx = np.triu_indices(N, k=1)
magnitudes = np.abs(kappa[i_idx, j_idx])
```

---

## Visualization

```bash
# Phase diagram
python visualizations/plot_phase.py --file build/hfb_results.h5

# BdG spectrum (gap opening)
python visualizations/plot_energies.py --file build/hfb_results.h5

# Pairing bonds animated over U
python visualizations/plot_kappa.py --file build/hfb_results.h5 \
    --animate-u --nx 4 --ny 4

# Quasiparticle density
python visualizations/plot_quasiparticle.py --file build/hfb_results.h5 \
    --animate-u --nx 4 --ny 4
```
