// data_output.h
//
// HDF5 output for HFB ground-state results and TDHFB time-series results.
// Self-contained header — defines HFBResult (used by hfb_numeric_current.cpp)
// and TDHFBResult (used by tdhfb.cpp).

#pragma once

#include <Eigen/Dense>
#include <complex>
#include <string>
#include <vector>

using cd     = std::complex<double>;
using MatXd  = Eigen::MatrixXd;
using MatXcd = Eigen::MatrixXcd;
using VecXd  = Eigen::VectorXd;

// ===================================================================
//  HFBResult — produced by run_hfb_hubbard() in hfb_numeric_current.cpp
// ===================================================================
struct HFBResult {
    // Core converged state
    MatXcd p;             // density matrix ρ, size 2N x 2N
    MatXcd k;             // pairing tensor κ, size 2N x 2N
    cd     energy;        // total energy
    MatXcd R;             // generalized density matrix, 4N x 4N
    MatXcd H_BdG;         // BdG Hamiltonian at convergence, 4N x 4N
    VecXd  H_BdG_evals;   // ascending eigenvalues
    MatXcd H_BdG_evecs;   // corresponding eigenvectors (columns)
    MatXcd U_bdg;         // top half of evecs   (size 2N x 4N)
    MatXcd V_bdg;         // bottom half of evecs (size 2N x 4N)

    // Metadata
    double U_param      = 0.0;
    double temperature  = 0.0;
    double mu           = 0.0;   // chemical potential at convergence (NEW: needed by TDHFB)
    int    iteration_count = 0;
    bool   converged    = false;

    // Lattice metadata (for reproducibility / TDHFB consumption)
    int    x_sites      = 0;
    int    y_sites      = 0;
    int    n_states     = 0;     // STATES = x*y
    int    particle_number = 0;
    double t_hop        = 1.0;
};

// Writes an HFB ground-state result to an HDF5 file.  If a group for this
// U value already exists, it is overwritten.  Otherwise a new group is
// created.  Top-level attributes include lattice dims and timestamp.
void write_converged_result_hdf5(const HFBResult& res, const std::string& path);

// ===================================================================
//  TDHFBResult — produced by run_tdhfb() in tdhfb.cpp
// ===================================================================
struct TDHFBResult {
    // Per-step time series
    std::vector<double> times;            // t at step i
    std::vector<double> F_real;           // Re ⟨F̃⟩(t)  −  Re ⟨F̃⟩(0⁻)
    std::vector<double> F_imag;           // Im ⟨F̃⟩(t)  (should be small)
    std::vector<double> energy;           // E(t), should be ~constant
    std::vector<double> particle_number;  // N(t), should be ~constant
    std::vector<double> idempotency_err;  // ‖R²−R‖_F, should be small
    std::vector<double> hermiticity_err;  // ‖R−R†‖_F, should be small
    std::vector<double> pairing_gap;      // (1/N) Σ_i |κ_i,i+N|, the on-site singlet gap
    std::vector<double> kinetic_energy;
    std::vector<double> interaction_energy;

    // Strength function S(E; F)  computed from damped FT of (F_real + i F_imag)
    std::vector<double> E_grid;
    std::vector<double> S_E;              // -Im f(E) / (π η)
    std::vector<double> f_real;           // Re f(E),  for diagnostics
    std::vector<double> f_imag;           // Im f(E),  for diagnostics

    // Initial (kicked) and final R, for restart / inspection
    MatXcd R_initial;     // R at t = 0⁺ (after kick)
    MatXcd R_final;       // R at t = T

    // Metadata
    double dt           = 0.0;
    int    n_steps      = 0;
    double T_total      = 0.0;
    double eta_kick     = 0.0;
    double Gamma_smooth = 0.0;
    double mu_fixed     = 0.0;
    double U_param      = 0.0;
    double t_hop        = 1.0;
    int    x_sites      = 0;
    int    y_sites      = 0;
    int    n_states     = 0;
    int    n_particles  = 0;
    std::string kick_operator;     // e.g. "density_q_pi_pi", "pairing_real", "current_x"
    std::string integrator;        // "expmid" or "euler"

    // Sum-rule diagnostic
    double m1_FT      = 0.0;       // ∫ E·S(E) dE  from FT
    double m1_commutator = 0.0;    // ½ ⟨0|[F,[H,F]]|0⟩  (when computable)
};

void write_tdhfb_result_hdf5(const TDHFBResult& res, const std::string& path);

// ===================================================================
//  TDHFBQuenchResult — produced by run_tdhfb_quench() in tdhfb_quench.cpp
// ===================================================================
struct TDHFBQuenchResult {
    // Per-step diagnostics (recorded every step)
    std::vector<double> tau;               // imaginary time τ
    std::vector<double> energy;            // E(τ), should decrease monotonically
    std::vector<double> particle_number;   // Tr(ρ), should stay ≈ N
    std::vector<double> idempotency_err;   // ‖R²−R‖_F, should be ≈ 0 after purify
    std::vector<double> hermiticity_err;   // ‖R−R†‖_F
    std::vector<double> pairing_gap;       // (1/N) Σ_i |κ_{i,i+N}|
    std::vector<double> mu_history;        // μ(τ)
    std::vector<double> kinetic_energy;
    std::vector<double> interaction_energy;

    // Matrix snapshots (every save_every steps)
    // Stored as (n_snap × 2N × 2N) or (n_snap × 2N × 4N) — see HDF5 dims
    std::vector<double> snap_tau;          // τ values at snapshot steps
    std::vector<MatXcd> snap_rho;          // ρ (2N×2N) at each snapshot
    std::vector<MatXcd> snap_kappa;        // κ (2N×2N) at each snapshot
    std::vector<MatXcd> snap_U_bdg;        // U_bdg (2N×2N) occupied amplitudes
    std::vector<MatXcd> snap_V_bdg;        // V_bdg (2N×2N) occupied amplitudes

    // Final generalized density matrix
    MatXcd R_final;

    // Metadata
    double U_initial   = 0.0;
    double U_final_val = 0.0;
    double dtau        = 0.0;
    double energy_tol  = 0.0;
    double mu_final    = 0.0;
    double mu_lr       = 0.0;
    int    n_steps_done = 0;
    bool   converged   = false;
    int    x_sites     = 0;
    int    y_sites     = 0;
    int    n_states    = 0;
    int    n_particles = 0;
    double t_hop       = 1.0;
    int    save_every  = 4;
};

void write_quench_result_hdf5(const TDHFBQuenchResult& res, const std::string& path);
