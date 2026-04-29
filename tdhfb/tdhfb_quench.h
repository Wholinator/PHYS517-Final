// tdhfb_quench.h
//
// Imaginary-time TDHFB for parameter quench relaxation on the 2D Hubbard model.
//
// Starting from a converged HFB ground state at U_initial, the interaction
// is suddenly changed to U_final at τ=0.  The system then relaxes to the new
// ground state via gradient descent on the energy functional using the
// double-commutator imaginary-time equation:
//
//     ∂_τ R = -[H[R], [H[R], R]]
//
// After each step: (1) μ is dynamically adjusted to conserve Tr(ρ) = N, and
// (2) R is purified (R²→R) by eigenvalue projection.
//
// Convergence is declared when |ΔE| between successive steps drops below
// energy_tol.  Matrix snapshots (ρ, κ, U_bdg, V_bdg) are written every
// save_every steps.

#pragma once

#include "data_output.h"
#include "hfb_api.h"

#include <string>

// ===================================================================
//  Parameters for an imaginary-time quench run
// ===================================================================
struct TDHFBQuenchParams {
    // Quench
    double U_initial  = 0.0;      // Hubbard U of the starting ground state
    double U_final    = -3.0;     // Hubbard U after the quench

    // Imaginary time grid
    double dtau       = 0.005;    // imaginary time step (units: ℏ/t_hop)
    double energy_tol = 1e-8;     // stop when |E(τ) − E(τ−dτ)| < tol
    int    max_steps  = 100000;   // hard upper limit on iterations

    // Output
    int    save_every = 4;        // save matrix snapshots every N steps

    // Chemical-potential tuning
    double mu_lr      = 0.1;      // learning rate: μ += lr * (N_target − N_actual)

    // Diagnostics
    bool   verbose       = true;
    int    verbose_every = 200;
};

// ===================================================================
//  Driver: gs (ground state at U_initial) → quench to U_final →
//          imaginary-time relax → return TDHFBQuenchResult.
// ===================================================================
TDHFBQuenchResult run_tdhfb_quench(const HFBResult& gs,
                                   const TDHFBQuenchParams& params);
