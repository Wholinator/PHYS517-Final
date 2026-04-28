// tdhfb.h
//
// Time-Dependent Hartree-Fock-Bogoliubov on the 2D Hubbard model.
//
// Propagates the generalized density matrix R(t) under the TDHFB equation
//
//     i ∂_t R = [ H[R(t)] − μ N̂_BdG, R(t) ]
//
// using a unitary integrator (matrix-exponential midpoint by default,
// forward Euler for debugging).  The mean-field h[ρ] and pairing field
// Δ[κ] are rebuilt every step from the current R(t), reusing the static
// HFB code's build_fock_matrix() and build_delta() routines verbatim.
//
// The standard use case is linear response: kick the HFB ground state
// with a one-body operator F, propagate, Fourier-transform ⟨F(t)⟩,
// recover the strength function S(E; F).

#pragma once

#include "data_output.h"
#include "hfb_api.h"

#include <string>

// ===================================================================
//  Parameters for a TDHFB run
// ===================================================================
struct TDHFBParams {
    // Time grid
    double dt        = 0.005;     // time step (units: ℏ/t_hop)
    int    n_steps   = 10000;     // total propagation steps; T = dt·n_steps

    // Kick (linear-response perturbation at t = 0)
    double      eta_kick = 1e-3;       // kick amplitude; small for linearity
    std::string kick_operator = "density_q_pi_pi";
        // Supported:
        //   "density_q_pi_pi"  — staggered (π,π) charge operator, Σ_i (-1)^(xi+yi) n_i
        //   "density_q_pi_0"   — stripe (π,0) charge operator,    Σ_i (-1)^xi n_i
        //   "density_uniform"  — Σ_i n_i  (couples to N̂; trivial test, only U=0 movements)
        //   "pairing_real"     — Σ_i (c†_i↑ c†_i↓ + h.c.) — Higgs/pairing channel
        //   "current_x"        — i Σ_<ij,σ> (c†_i c_j − h.c.) along x;  optical-like

    // Smoothing for the strength function
    double Gamma_smooth = 0.5;    // Lorentzian half-width in damped FT
    int    n_E_grid     = 800;
    double E_min        = 0.0;
    double E_max        = 8.0;

    // Integrator
    std::string integrator = "expmid";   // "expmid" or "euler"

    // Diagnostics frequency
    int    sample_every = 1;       // record observables every N steps (1 = every step)

    // Sum-rule check (computes ½⟨0|[F,[H,F]]|0⟩)
    bool   compute_m1_commutator = true;

    // Verbose progress to stdout
    bool   verbose = true;
    int    verbose_every = 500;
};

// ===================================================================
//  Build a one-body kick operator F as a 2N x 2N complex Hermitian
//  matrix in spin-orbital space.  The convention is the same as the
//  static HFB code:  basis indices  0..STATES-1 are spin-↑ at sites,
//                                  STATES..2*STATES-1 are spin-↓ at sites.
// ===================================================================
MatXcd build_kick_operator(const std::string& kind,
                           int x, int y, int STATES);

// ===================================================================
//  Apply a one-body kick to a generalized density matrix R.
//  R' = exp(i η F̃) R exp(-i η F̃)   where F̃ = block_diag(F, -F̄).
//  R is 4N x 4N, F is 2N x 2N.
// ===================================================================
MatXcd apply_kick(const MatXcd& R, const MatXcd& F, double eta);

// ===================================================================
//  One propagation step (unitary, second-order midpoint).
//  R(t+dt) = U R(t) U†, with U = exp(-i dt H_mid),
//  H_mid = ½ ( H[R(t)] + H[R_pred] ),
//  R_pred = exp(-i dt H[R(t)]) R(t) exp(+i dt H[R(t)]).
//  μ_fixed enters as a chemical-potential shift of h.
// ===================================================================
MatXcd propagate_step_expmid(const MatXcd& R, const MatXd& H_t,
                             double U_int, int STATES, double mu_fixed,
                             double dt);

// Forward-Euler step for debugging:  R(t+dt) = R(t) − i dt [H[R(t)], R(t)]
// Not unitary — drift and idempotency loss are the price.
MatXcd propagate_step_euler(const MatXcd& R, const MatXd& H_t,
                            double U_int, int STATES, double mu_fixed,
                            double dt);

// ===================================================================
//  Driver: ground state R₀ → kick → propagate → strength function.
//  Writes per-step diagnostics into the result.
// ===================================================================
TDHFBResult run_tdhfb(const HFBResult& gs, const TDHFBParams& params);
