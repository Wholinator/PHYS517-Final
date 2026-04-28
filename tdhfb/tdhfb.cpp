// tdhfb.cpp
//
// Time-Dependent Hartree-Fock-Bogoliubov implementation.

#include "tdhfb.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

// =========================================================================
//  Internal helpers
// =========================================================================
namespace {

constexpr cd I_unit{0.0, 1.0};

// Build the BdG-doubled form of a one-body operator F (size 2N x 2N).
//   F̃ = [[F, 0], [0, -F̄]]  — size 4N x 4N
// This is the correct generalization of "apply F to the orbitals" in the
// density-matrix picture.  If you compute Tr[F̃ R] with R the BdG density,
// you get exactly Σ_p ⟨0|c†_p F_pq c_q|0⟩  =  Tr[F ρ].
MatXcd bdg_double(const MatXcd& F) {
    return block_diag2<MatXcd>(F, (-F.conjugate()).eval());
}

// Build the full BdG mean-field Hamiltonian from current ρ, κ.
// Reuses build_fock_matrix() and build_delta() from hfb_numeric_current.cpp.
MatXcd build_H_BdG(const MatXcd& R, const MatXd& H_t,
                   double U_int, int STATES, double mu_fixed) {
    const int twoN = 2 * STATES;
    MatXcd p = R.topLeftCorner(twoN, twoN);
    MatXcd k = R.topRightCorner(twoN, twoN);

    MatXcd h     = build_fock_matrix(STATES, H_t, p, U_int);
    MatXcd delta = build_delta(k, U_int, STATES);

    MatXcd I = MatXcd::Identity(twoN, twoN);
    MatXcd h_shift = h - mu_fixed * I;

    return block2x2<MatXcd>(h_shift, delta,
                            (-delta.conjugate()).eval(),
                            (-h_shift.transpose()).eval());
}

// Compute exp(c · H) via eigendecomposition.  H must be Hermitian (it is
// here — H_BdG is constructed Hermitian).  For a 1024x1024 matrix this is
// a few hundred ms; for 64x64 it is a few hundred μs.  Eigen's general
// MatrixExponential (Padé+scaling) is also available but eigh is faster
// when H is known Hermitian, and gives machine-precision unitarity.
MatXcd expm_hermitian_scaled(const MatXcd& H, cd scale) {
    Eigen::SelfAdjointEigenSolver<MatXcd> es(H);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("expm: eigh failed (H not Hermitian?)");
    const VecXd&  ev = es.eigenvalues();
    const MatXcd& V  = es.eigenvectors();
    Eigen::VectorXcd phases(ev.size());
    for (Eigen::Index i = 0; i < ev.size(); ++i)
        phases(i) = std::exp(scale * cd(ev(i), 0.0));
    // V * diag(phases) * V†
    MatXcd out = V;
    for (Eigen::Index j = 0; j < V.cols(); ++j)
        out.col(j) *= phases(j);
    return out * V.adjoint();
}

// Energy of the BdG state under the (full, μ-free) mean-field Hamiltonian.
//   E = ½ Tr[(Hₜ + h[ρ]) ρ]  −  ½ Tr[Δ[κ] κ†]
// Same definition as run_hfb_hubbard().  Returns kinetic, interaction, total.
struct EnergyDecomp { double kinetic; double interaction; double total; };

EnergyDecomp compute_energy(const MatXcd& R, const MatXd& H_t,
                            double U_int, int STATES) {
    const int twoN = 2 * STATES;
    MatXcd p = R.topLeftCorner(twoN, twoN);
    MatXcd k = R.topRightCorner(twoN, twoN);
    MatXcd h     = build_fock_matrix(STATES, H_t, p, U_int);
    MatXcd delta = build_delta(k, U_int, STATES);
    MatXcd H_t_c = H_t.cast<cd>();

    // Total HFB energy:  E = ½ Tr[(H_t + h) ρ]  −  ½ Tr[Δ κ†]
    // Split: kinetic = Tr[H_t ρ],  Hartree(+Fock) = ½ Tr[(h − H_t) ρ],
    //        pairing = −½ Tr[Δ κ†].
    // (½ Tr[(H_t+h) ρ] = ½ Tr[H_t ρ] + ½ Tr[h ρ] = Tr[H_t ρ] + ½ Tr[(h-H_t) ρ].)
    cd e_kin     = (H_t_c * p).trace();
    cd e_hartree = 0.5 * ((h - H_t_c) * p).trace();
    cd e_pair    = -0.5 * (delta * k.adjoint()).trace();

    EnergyDecomp ed;
    ed.kinetic     = e_kin.real();
    ed.interaction = e_hartree.real() + e_pair.real();
    ed.total       = 0.5 * ((H_t_c + h) * p).trace().real() + e_pair.real();
    return ed;
}

// On-site singlet pairing magnitude:  (1/N) Σ_i |κ_{i, i+N}|.
// This is the closest scalar analogue of the BCS gap on the lattice.
double pairing_gap_magnitude(const MatXcd& R, int STATES) {
    const int twoN = 2 * STATES;
    MatXcd k = R.topRightCorner(twoN, twoN);
    double sum = 0.0;
    for (int i = 0; i < STATES; ++i) sum += std::abs(k(i, i + STATES));
    return sum / std::max(1, STATES);
}

// ½ ⟨0|[F̃, [H_BdG, F̃]]|0⟩  via the standard identity  m1 = ½ Tr[ F̃ [H, F̃] R ]
// In density-matrix form: Tr[F̃ [H, F̃] R].  We use this for the sum-rule check
// at t=0 (using H without the μ shift, since the m1 sum rule is for H, not H-μN).
double m1_double_commutator(const MatXcd& R0, const MatXcd& F_tilde,
                             const MatXd& H_t, double U_int, int STATES) {
    const int twoN = 2 * STATES;
    MatXcd p = R0.topLeftCorner(twoN, twoN);
    MatXcd k = R0.topRightCorner(twoN, twoN);
    MatXcd h     = build_fock_matrix(STATES, H_t, p, U_int);
    MatXcd delta = build_delta(k, U_int, STATES);
    // Use H_BdG with μ = 0 — m1 sum rule is for the original Hamiltonian.
    MatXcd H = block2x2<MatXcd>(h, delta,
                                 (-delta.conjugate()).eval(),
                                 (-h.transpose()).eval());
    MatXcd comm = H * F_tilde - F_tilde * H;          // [H, F̃]
    MatXcd inner = F_tilde * comm - comm * F_tilde;   // [F̃, [H, F̃]]
    cd val = 0.5 * (inner * R0).trace();
    return val.real();
}

}  // namespace

// =========================================================================
//  Public:  build_kick_operator
// =========================================================================
MatXcd build_kick_operator(const std::string& kind,
                           int x, int y, int STATES) {
    if (STATES != x * y)
        throw std::runtime_error("build_kick_operator: STATES != x*y");
    const int twoN = 2 * STATES;
    MatXcd F = MatXcd::Zero(twoN, twoN);

    auto site_index = [y](int xi, int yi) { return xi * y + yi; };

    if (kind == "density_q_pi_pi") {
        // F = Σ_{i,σ} (-1)^(xi+yi) n_{iσ}
        for (int xi = 0; xi < x; ++xi) {
            for (int yi = 0; yi < y; ++yi) {
                int si = site_index(xi, yi);
                double s = ((xi + yi) % 2 == 0) ? 1.0 : -1.0;
                F(si, si)                 = s;     // ↑
                F(si + STATES, si + STATES) = s;   // ↓
            }
        }
    } else if (kind == "density_q_pi_0") {
        for (int xi = 0; xi < x; ++xi) {
            double s = (xi % 2 == 0) ? 1.0 : -1.0;
            for (int yi = 0; yi < y; ++yi) {
                int si = site_index(xi, yi);
                F(si, si)                 = s;
                F(si + STATES, si + STATES) = s;
            }
        }
    } else if (kind == "density_uniform") {
        for (int i = 0; i < twoN; ++i) F(i, i) = 1.0;
    } else if (kind == "pairing_real") {
        // F = Σ_i (c†_{i↑} c†_{i↓} + h.c.)   — but this is a TWO-body op in the
        // Fock-space sense (not one-body).  In the BdG picture the right thing
        // is the singlet-pair operator F̃ that lives in the off-diagonal blocks.
        // For now we implement a real hopping in spin space:  Σ_i (c†_{i↑} c_{i↓} + h.c.).
        // This drives the system through a spin-flip channel; it lights up
        // pairing and magnetic responses without needing pair creation.
        for (int i = 0; i < STATES; ++i) {
            F(i, i + STATES) = 1.0;
            F(i + STATES, i) = 1.0;
        }
    } else if (kind == "current_x") {
        // F = i Σ_{<ij>_x, σ} (c†_i c_j − h.c.)  — current along x, with PBC.
        for (int xi = 0; xi < x; ++xi) {
            int xj = (xi + 1) % x;
            for (int yi = 0; yi < y; ++yi) {
                int si = site_index(xi, yi);
                int sj = site_index(xj, yi);
                F(si, sj)                       =  cd(0, 1.0);
                F(sj, si)                       = -cd(0, 1.0);
                F(si + STATES, sj + STATES)     =  cd(0, 1.0);
                F(sj + STATES, si + STATES)     = -cd(0, 1.0);
            }
        }
    } else {
        throw std::runtime_error("build_kick_operator: unknown kind '" + kind + "'");
    }

    // Sanity: F should be Hermitian.  Symmetrize to kill any roundoff.
    F = 0.5 * (F + F.adjoint());
    return F;
}

// =========================================================================
//  Public:  apply_kick
// =========================================================================
// Kicks with exp(-iη F̃) to match Ebata Eq. (70):  Vext = -η F δ(t).
// With this convention, S(E;F) = -(1/πη) Im f(E) is positive at E_n > 0.
MatXcd apply_kick(const MatXcd& R, const MatXcd& F, double eta) {
    MatXcd F_tilde = bdg_double(F);
    // U = exp(-i η F̃)
    MatXcd U_kick = expm_hermitian_scaled(F_tilde, cd(0.0, -eta));
    return U_kick * R * U_kick.adjoint();
}

// =========================================================================
//  Public:  propagate_step_expmid
// =========================================================================
MatXcd propagate_step_expmid(const MatXcd& R, const MatXd& H_t,
                             double U_int, int STATES, double mu_fixed,
                             double dt) {
    // 1.  Hamiltonian at t.
    MatXcd H0 = build_H_BdG(R, H_t, U_int, STATES, mu_fixed);

    // 2.  Predictor:  R_pred = exp(-i dt H0) R exp(+i dt H0).
    MatXcd U0 = expm_hermitian_scaled(H0, cd(0.0, -dt));
    MatXcd R_pred = U0 * R * U0.adjoint();

    // 3.  Hamiltonian at predicted state.
    MatXcd H1 = build_H_BdG(R_pred, H_t, U_int, STATES, mu_fixed);

    // 4.  Symmetric midpoint:  H_mid = ½ (H0 + H1).
    MatXcd H_mid = 0.5 * (H0 + H1);
    MatXcd U_mid = expm_hermitian_scaled(H_mid, cd(0.0, -dt));
    return U_mid * R * U_mid.adjoint();
}

// =========================================================================
//  Public:  propagate_step_euler
// =========================================================================
MatXcd propagate_step_euler(const MatXcd& R, const MatXd& H_t,
                            double U_int, int STATES, double mu_fixed,
                            double dt) {
    MatXcd H = build_H_BdG(R, H_t, U_int, STATES, mu_fixed);
    MatXcd comm = H * R - R * H;   // [H, R]
    return R - cd(0.0, dt) * comm;
}

// =========================================================================
//  Damped Fourier transform — strength function S(E; F)
// =========================================================================
//
//   f(E) = ∫_0^T  e^{(iE - Γ/2) t} ⟨F̃⟩(t)  dt
//   S(E; F) = -(1/(π η))  Im f(E)
//
// Trapezoidal rule on the recorded time grid.  The (F_real, F_imag) fed in
// are the *deviation* from the ground-state value, so f(E=0) is bounded.
namespace {

void compute_strength_function(const std::vector<double>& times,
                               const std::vector<double>& F_real,
                               const std::vector<double>& F_imag,
                               double Gamma, double eta_kick,
                               const std::vector<double>& E_grid,
                               std::vector<double>& S_E,
                               std::vector<double>& f_real,
                               std::vector<double>& f_imag) {
    const size_t Nt = times.size();
    S_E.assign(E_grid.size(), 0.0);
    f_real.assign(E_grid.size(), 0.0);
    f_imag.assign(E_grid.size(), 0.0);
    if (Nt < 2) return;

    const double half_gamma = 0.5 * Gamma;

    for (size_t k = 0; k < E_grid.size(); ++k) {
        const double E = E_grid[k];
        cd integral{0.0, 0.0};
        for (size_t i = 0; i < Nt; ++i) {
            const double t = times[i];
            const cd phase = std::exp(cd(-half_gamma * t,  E * t));
            const cd Ft(F_real[i], F_imag[i]);
            const cd integrand = phase * Ft;
            // Trapezoidal weights: ½ at endpoints, full elsewhere; multiply by dt later.
            const double w = (i == 0 || i + 1 == Nt) ? 0.5 : 1.0;
            integral += w * integrand;
        }
        const double dt = (Nt > 1) ? (times[1] - times[0]) : 0.0;
        integral *= dt;
        f_real[k] = integral.real();
        f_imag[k] = integral.imag();
        S_E[k]    = -(1.0 / (M_PI * eta_kick)) * integral.imag();
    }
}

}  // namespace

// =========================================================================
//  Public:  run_tdhfb
// =========================================================================
TDHFBResult run_tdhfb(const HFBResult& gs, const TDHFBParams& params) {
    if (!gs.converged)
        throw std::runtime_error("run_tdhfb: ground state did not converge");
    if (gs.R.size() == 0)
        throw std::runtime_error("run_tdhfb: ground-state R is empty");

    const int x       = gs.x_sites;
    const int y       = gs.y_sites;
    const int STATES  = gs.n_states;
    const int twoN    = 2 * STATES;
    const int fourN   = 4 * STATES;
    if (STATES != x * y)
        throw std::runtime_error("run_tdhfb: STATES != x*y in HFBResult");

    const double U_int    = gs.U_param;
    const double t_hop    = gs.t_hop;
    const double mu_fixed = gs.mu;

    // Static lattice
    MatXd H_t = build_t_mat_hubbard(x, y, t_hop, /*pbc=*/true);

    // Kick operator (one-body, 2N×2N) and its BdG-doubled form (4N×4N).
    MatXcd F        = build_kick_operator(params.kick_operator, x, y, STATES);
    MatXcd F_tilde  = bdg_double(F);

    // Optional sum-rule: m1 from double commutator on the *un-kicked* ground state.
    TDHFBResult result;
    result.dt              = params.dt;
    result.n_steps         = params.n_steps;
    result.T_total         = params.dt * params.n_steps;
    result.eta_kick        = params.eta_kick;
    result.Gamma_smooth    = params.Gamma_smooth;
    result.mu_fixed        = mu_fixed;
    result.U_param         = U_int;
    result.t_hop           = t_hop;
    result.x_sites         = x;
    result.y_sites         = y;
    result.n_states        = STATES;
    result.n_particles     = gs.particle_number;
    result.kick_operator   = params.kick_operator;
    result.integrator      = params.integrator;

    if (params.compute_m1_commutator) {
        result.m1_commutator =
            m1_double_commutator(gs.R, F_tilde, H_t, U_int, STATES);
    }

    // Reference value of ⟨F̃⟩ on the un-kicked ground state.
    const cd F0 = (F_tilde * gs.R).trace();

    // Apply the kick.
    MatXcd R = apply_kick(gs.R, F, params.eta_kick);
    result.R_initial = R;

    if (params.verbose) {
        std::printf(
            "[tdhfb] lattice %dx%d (N=%d, BdG dim=%d), U=%.4f, t=%.4f, mu=%.4f\n"
            "        kick='%s', eta=%.3e, dt=%.5f, n_steps=%d, T=%.3f\n"
            "        integrator=%s, Gamma=%.3f\n"
            "        ground state: E=%.6f, m1(comm)=%.6e\n",
            x, y, STATES, fourN,
            U_int, t_hop, mu_fixed,
            params.kick_operator.c_str(), params.eta_kick,
            params.dt, params.n_steps, params.dt * params.n_steps,
            params.integrator.c_str(), params.Gamma_smooth,
            gs.energy.real(), result.m1_commutator);
        std::fflush(stdout);
    }

    // Reserve storage.
    const size_t n_record = (params.n_steps / std::max(1, params.sample_every)) + 1;
    result.times.reserve(n_record);
    result.F_real.reserve(n_record);
    result.F_imag.reserve(n_record);
    result.energy.reserve(n_record);
    result.particle_number.reserve(n_record);
    result.idempotency_err.reserve(n_record);
    result.hermiticity_err.reserve(n_record);
    result.pairing_gap.reserve(n_record);
    result.kinetic_energy.reserve(n_record);
    result.interaction_energy.reserve(n_record);

    // Choose integrator
    using StepFn = MatXcd(*)(const MatXcd&, const MatXd&, double, int, double, double);
    StepFn step_fn = (params.integrator == "euler") ?
        static_cast<StepFn>(propagate_step_euler)
        : static_cast<StepFn>(propagate_step_expmid);

    // Time loop ---------------------------------------------------------
    auto t_start = std::chrono::steady_clock::now();

    for (int step = 0; step <= params.n_steps; ++step) {
        const double t = step * params.dt;
        const bool record = (step % std::max(1, params.sample_every) == 0)
                          || (step == params.n_steps);

        if (record) {
            // ⟨F̃⟩(t) — minus ground-state reference, so the FT is well-conditioned.
            cd Ft_full = (F_tilde * R).trace();
            cd dF = Ft_full - F0;

            EnergyDecomp ed = compute_energy(R, H_t, U_int, STATES);
            double Ntr      = R.topLeftCorner(twoN, twoN).trace().real();
            // Idempotency / hermiticity sanity
            MatXcd I_full   = MatXcd::Identity(fourN, fourN);
            (void) I_full;
            MatXcd diff_idem = R * R - R;
            double idem_err  = diff_idem.norm();
            MatXcd diff_herm = R - R.adjoint();
            double herm_err  = diff_herm.norm();
            double gap_t     = pairing_gap_magnitude(R, STATES);

            result.times.push_back(t);
            result.F_real.push_back(dF.real());
            result.F_imag.push_back(dF.imag());
            result.energy.push_back(ed.total);
            result.kinetic_energy.push_back(ed.kinetic);
            result.interaction_energy.push_back(ed.interaction);
            result.particle_number.push_back(Ntr);
            result.idempotency_err.push_back(idem_err);
            result.hermiticity_err.push_back(herm_err);
            result.pairing_gap.push_back(gap_t);
        }

        if (params.verbose && (step % std::max(1, params.verbose_every) == 0)) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t_start).count();
            const double frac = (params.n_steps > 0) ?
                (double)step / (double)params.n_steps : 1.0;
            const double eta_s = (frac > 1e-9) ? elapsed * (1.0/frac - 1.0) : 0.0;
            if (!result.energy.empty()) {
                std::printf(
                    "[tdhfb] step %6d/%d (t=%.3f)   E=%.6f   N=%.4f   "
                    "|R²-R|=%.2e   gap=%.4f   elapsed=%.1fs   eta=%.1fs\n",
                    step, params.n_steps, t,
                    result.energy.back(),
                    result.particle_number.back(),
                    result.idempotency_err.back(),
                    result.pairing_gap.back(),
                    elapsed, eta_s);
                std::fflush(stdout);
            }
        }

        if (step == params.n_steps) break;

        // Advance one step.
        R = step_fn(R, H_t, U_int, STATES, mu_fixed, params.dt);
    }

    result.R_final = R;

    // Strength function S(E; F).  E grid uniform over [E_min, E_max].
    result.E_grid.resize(params.n_E_grid);
    for (int k = 0; k < params.n_E_grid; ++k) {
        double frac = (params.n_E_grid == 1) ? 0.0 :
            (double)k / (double)(params.n_E_grid - 1);
        result.E_grid[k] = params.E_min + frac * (params.E_max - params.E_min);
    }
    compute_strength_function(result.times, result.F_real, result.F_imag,
                              params.Gamma_smooth, params.eta_kick,
                              result.E_grid, result.S_E,
                              result.f_real, result.f_imag);

    // Sum-rule check from FT: m1 = ∫ E S(E) dE.  Trapezoidal.
    {
        double m1 = 0.0;
        const auto& Eg = result.E_grid;
        const auto& Sg = result.S_E;
        for (size_t i = 0; i + 1 < Eg.size(); ++i) {
            double dE = Eg[i+1] - Eg[i];
            m1 += 0.5 * dE * (Eg[i] * Sg[i] + Eg[i+1] * Sg[i+1]);
        }
        result.m1_FT = m1;
    }

    if (params.verbose) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t_start).count();
        std::printf("[tdhfb] done.  elapsed=%.1fs   m1(FT)=%.6e   m1(comm)=%.6e   ratio=%.4f\n",
                    elapsed, result.m1_FT, result.m1_commutator,
                    (result.m1_commutator != 0.0)
                       ? result.m1_FT / result.m1_commutator : 0.0);
        std::fflush(stdout);
    }

    return result;
}
