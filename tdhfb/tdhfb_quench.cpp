// tdhfb_quench.cpp
//
// Imaginary-time TDHFB relaxation after a parameter quench.
//
// Algorithm per step:
//   1. Build H_BdG from current R using the *quenched* U.
//   2. Apply the double-commutator imaginary-time step:
//        R_new = R − dτ [H, [H, R]]
//   3. Purify: project R_new eigenvalues to {0,1} keeping the top 2N,
//      restoring R²=R exactly.
//   4. Measure N = Tr(ρ_new); tune μ for the next step via
//        μ ← μ + lr * (N_target − N_actual)
//
// Convergence: halt when |E(step) − E(step−1)| < energy_tol.

#include "tdhfb_quench.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// =========================================================================
//  Internal helpers
// =========================================================================
namespace {

// Build BdG Hamiltonian — identical logic to tdhfb.cpp but local.
MatXcd build_H_BdG_q(const MatXcd& R, const MatXd& H_t,
                     double U_int, int STATES, double mu) {
    const int twoN = 2 * STATES;
    MatXcd p = R.topLeftCorner(twoN, twoN);
    MatXcd k = R.topRightCorner(twoN, twoN);

    MatXcd h     = build_fock_matrix(STATES, H_t, p, U_int);
    MatXcd delta = build_delta(k, U_int, STATES);

    MatXcd I2N    = MatXcd::Identity(twoN, twoN);
    MatXcd h_shift = h - mu * I2N;

    return block2x2<MatXcd>(h_shift, delta,
                            (-delta.conjugate()).eval(),
                            (-h_shift.transpose()).eval());
}

// Forward-Euler imaginary-time step using the double commutator:
//   R_new = R − dτ [H, [H, R]]
MatXcd it_step_dc(const MatXcd& R, const MatXcd& H, double dtau) {
    MatXcd c1 = H * R - R * H;          // [H, R]
    MatXcd c2 = H * c1 - c1 * H;       // [H, [H, R]]
    return R - dtau * c2;
}

// Purification result: idempotent R plus the occupied BdG amplitudes.
// U_bdg = top-2N rows of the occupied eigenvectors  (2N × 2N)
// V_bdg = bottom-2N rows of the occupied eigenvectors (2N × 2N)
struct PurifyResult {
    MatXcd R;
    MatXcd U_bdg;
    MatXcd V_bdg;
};

// Project R to idempotency by retaining the 2*STATES largest eigenvectors.
// Enforces R²=R and R†=R exactly.
PurifyResult purify(const MatXcd& R_raw, int STATES) {
    const int twoN  = 2 * STATES;
    const int n_occ = twoN;  // BdG always occupies 2N quasiparticle states

    // Symmetrize before diagonalizing to control round-off.
    MatXcd R_sym = 0.5 * (R_raw + R_raw.adjoint());

    Eigen::SelfAdjointEigenSolver<MatXcd> es(R_sym);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("purify: SelfAdjointEigenSolver failed");

    // Eigenvalues are ascending → rightmost n_occ columns are "most occupied".
    const MatXcd& V = es.eigenvectors();
    const int n = (int)V.cols();   // = 4N

    // Occupied BdG states = eigenvectors with the n_occ largest eigenvalues.
    MatXcd Q = V.rightCols(n_occ);  // 4N × 2N

    PurifyResult pr;
    pr.R     = Q * Q.adjoint();         // = projector, rank n_occ
    pr.U_bdg = Q.topRows(twoN);         // 2N × 2N  (particle amplitudes)
    pr.V_bdg = Q.bottomRows(twoN);      // 2N × 2N  (hole amplitudes)
    return pr;
}

// Particle number: Tr(ρ) where ρ is the top-left 2N×2N block of R.
double count_particles(const MatXcd& R, int STATES) {
    const int twoN = 2 * STATES;
    return R.topLeftCorner(twoN, twoN).trace().real();
}

// HFB energy E = ½ Tr[(H_t + h)ρ] − ½ Tr[Δκ†].
struct EnergyParts { double kinetic; double interaction; double total; };

EnergyParts energy_q(const MatXcd& R, const MatXd& H_t,
                     double U_int, int STATES) {
    const int twoN = 2 * STATES;
    MatXcd p = R.topLeftCorner(twoN, twoN);
    MatXcd k = R.topRightCorner(twoN, twoN);
    MatXcd h     = build_fock_matrix(STATES, H_t, p, U_int);
    MatXcd delta = build_delta(k, U_int, STATES);
    MatXcd H_t_c = H_t.cast<cd>();

    cd e_pair = -0.5 * (delta * k.adjoint()).trace();
    double e_kin  = (H_t_c * p).trace().real();
    double e_int  = (0.5 * ((h - H_t_c) * p).trace() + e_pair).real();
    double e_tot  = (0.5 * ((H_t_c + h) * p).trace()).real() + e_pair.real();
    return { e_kin, e_int, e_tot };
}

// On-site singlet pairing magnitude: (1/N) Σ_i |κ_{i, i+N}|.
double pairing_gap_q(const MatXcd& R, int STATES) {
    const int twoN = 2 * STATES;
    MatXcd k = R.topRightCorner(twoN, twoN);
    double s = 0.0;
    for (int i = 0; i < STATES; ++i) s += std::abs(k(i, i + STATES));
    return s / std::max(1, STATES);
}

void apply_kappa_perturbation(MatXcd& R, int STATES, double eps) {
    if (eps <= 0.0) {
        return;
    }

    const int twoN = 2 * STATES;
    MatXcd k = R.topRightCorner(twoN, twoN);

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    MatXcd randm(twoN, twoN);
    for (int i = 0; i < twoN; ++i) {
        for (int j = 0; j < twoN; ++j) {
            randm(i, j) = cd(dist(rng), dist(rng));
        }
    }

    MatXcd anti = 0.5 * (randm - randm.transpose());
    k += eps * anti;

    R.topRightCorner(twoN, twoN) = k;
    R.bottomLeftCorner(twoN, twoN) = -k.conjugate();
}

}  // namespace

// =========================================================================
//  Public:  run_tdhfb_quench
// =========================================================================
TDHFBQuenchResult run_tdhfb_quench(const HFBResult& gs,
                                   const TDHFBQuenchParams& params) {
    if (!gs.converged)
        throw std::runtime_error("run_tdhfb_quench: ground state did not converge");
    if (gs.R.size() == 0)
        throw std::runtime_error("run_tdhfb_quench: ground-state R is empty");

    const int STATES   = gs.n_states;
    const int twoN     = 2 * STATES;
    const int fourN    = 4 * STATES;
    const int N_target = gs.particle_number;
    const double t_hop = gs.t_hop;

    if (STATES != gs.x_sites * gs.y_sites)
        throw std::runtime_error("run_tdhfb_quench: STATES != x*y");

    // Static kinetic Hamiltonian (lattice doesn't change).
    MatXd H_t = build_t_mat_hubbard(gs.x_sites, gs.y_sites, t_hop, /*pbc=*/true);

    // ── Apply quench: all subsequent evolution uses U_final. ──────────────
    const double U_q = params.U_final;

    MatXcd R = gs.R;
    double mu = gs.mu;

    if (std::abs(params.U_initial) < 1e-12 && std::abs(params.U_final) > 0.0) {
        apply_kappa_perturbation(R, STATES, 1e-6);
    }

    // Populate result metadata
    TDHFBQuenchResult result;
    result.U_initial    = params.U_initial;
    result.U_final_val  = params.U_final;
    result.dtau         = params.dtau;
    result.energy_tol   = params.energy_tol;
    result.mu_lr        = params.mu_lr;
    result.x_sites      = gs.x_sites;
    result.y_sites      = gs.y_sites;
    result.n_states     = STATES;
    result.n_particles  = N_target;
    result.t_hop        = t_hop;
    result.save_every   = params.save_every;

    if (params.verbose) {
        std::printf(
            "[quench] lattice %dx%d  N=%d  BdG dim=%d\n"
            "         U_initial=%.4f → U_final=%.4f   t=%.4f   μ₀=%.4f\n"
            "         dτ=%.5f   energy_tol=%.2e   max_steps=%d\n"
            "         save_every=%d   μ_lr=%.4f\n",
            gs.x_sites, gs.y_sites, STATES, fourN,
            params.U_initial, params.U_final, t_hop, mu,
            params.dtau, params.energy_tol, params.max_steps,
            params.save_every, params.mu_lr);
        std::fflush(stdout);
    }

    auto t_wall = std::chrono::steady_clock::now();
    double E_prev = 0.0;

    // ── Main imaginary-time loop ──────────────────────────────────────────
    for (int step = 0; step <= params.max_steps; ++step) {
        const double tau    = step * params.dtau;
        const bool do_snap  = (step % std::max(1, params.save_every) == 0);

        // ── Diagnostics ───────────────────────────────────────────────────
        auto ep     = energy_q(R, H_t, U_q, STATES);
        double N_act = count_particles(R, STATES);

        // Idempotency and hermiticity (after purification should be ~0)
        MatXcd dI   = R * R - R;
        double idem = dI.norm();
        MatXcd dH   = R - R.adjoint();
        double herm = dH.norm();
        double gap  = pairing_gap_q(R, STATES);

        result.tau.push_back(tau);
        result.energy.push_back(ep.total);
        result.particle_number.push_back(N_act);
        result.idempotency_err.push_back(idem);
        result.hermiticity_err.push_back(herm);
        result.pairing_gap.push_back(gap);
        result.mu_history.push_back(mu);
        result.kinetic_energy.push_back(ep.kinetic);
        result.interaction_energy.push_back(ep.interaction);

        // ── Matrix snapshot ───────────────────────────────────────────────
        if (do_snap) {
            // R is already purified; re-diagonalise for U_bdg / V_bdg.
            PurifyResult pr_snap = purify(R, STATES);

            result.snap_tau.push_back(tau);
            result.snap_rho.push_back(R.topLeftCorner(twoN, twoN));
            result.snap_kappa.push_back(R.topRightCorner(twoN, twoN));
            result.snap_U_bdg.push_back(pr_snap.U_bdg);
            result.snap_V_bdg.push_back(pr_snap.V_bdg);
        }

        // ── Verbose progress ──────────────────────────────────────────────
        if (params.verbose && (step % std::max(1, params.verbose_every) == 0)) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_wall).count();
            double dE = (step > 0) ? std::abs(ep.total - E_prev) : 0.0;
            std::printf(
                "[quench] step %7d (τ=%8.3f)  E=%12.8f  |ΔE|=%.2e"
                "  N=%.4f  ‖R²-R‖=%.1e  gap=%.4f  μ=%.4f  %.1fs\n",
                step, tau, ep.total, dE,
                N_act, idem, gap, mu, elapsed);
            std::fflush(stdout);
        }

        // ── Convergence check ─────────────────────────────────────────────
        // Check after the first step so dE is meaningful.
        if (step > 0 && std::abs(ep.total - E_prev) < params.energy_tol) {
            result.converged     = true;
            result.n_steps_done  = step;
            if (params.verbose) {
                std::printf(
                    "[quench] CONVERGED at step %d (τ=%.3f)  "
                    "|ΔE|=%.2e < %.2e\n",
                    step, tau,
                    std::abs(ep.total - E_prev), params.energy_tol);
                std::fflush(stdout);
            }
            break;
        }

        if (step == params.max_steps) {
            result.converged    = false;
            result.n_steps_done = params.max_steps;
            if (params.verbose) {
                std::printf("[quench] reached max_steps=%d without convergence\n",
                            params.max_steps);
                std::fflush(stdout);
            }
            break;
        }

        E_prev = ep.total;

        // ── Propagation ───────────────────────────────────────────────────
        // 1. Build H_BdG with the quenched U and current μ.
        MatXcd H = build_H_BdG_q(R, H_t, U_q, STATES, mu);

        // 2. Imaginary-time double-commutator step.
        MatXcd R_step = it_step_dc(R, H, params.dtau);

        // 3. Purify: restore R²=R.
        PurifyResult pr = purify(R_step, STATES);
        R = pr.R;

        // 4. Tune μ: correct for particle-number drift introduced by the step.
        double N_new = count_particles(R, STATES);
        mu += params.mu_lr * ((double)N_target - N_new);
    }

    result.R_final  = R;
    result.mu_final = mu;
    return result;
}
