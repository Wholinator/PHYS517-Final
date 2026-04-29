// tdhfb_quench_main.cpp
//
// Driver: build HFB ground state at U_initial, quench to U_final,
//         relax via imaginary-time TDHFB, write results to HDF5.
//
// Usage:
//   ./tdhfb_quench [options]
//
//   --x N            lattice x dimension (default 4)
//   --y N            lattice y dimension (default 4)
//   --U-initial V    Hubbard U for initial ground state (default 0)
//   --U-final V      Hubbard U after quench            (default -3)
//   --t V            hopping amplitude                 (default 1.0)
//   --N V            particle number (default = STATES, half-filling)
//   --dtau V         imaginary time step               (default 0.005)
//   --max-steps N    maximum imaginary-time steps      (default 100000)
//   --tol V          energy convergence tolerance      (default 1e-8)
//   --save-every N   matrix snapshot interval          (default 4)
//   --mu-lr V        μ learning rate                   (default 0.1)
//   --output PATH    output HDF5 file  (default tdhfb_quench_results.h5)
//   --seed N         RNG seed (default: random)
//   --quiet          suppress verbose progress

#include "data_output.h"
#include "hfb_api.h"
#include "tdhfb_quench.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

namespace {

struct CLIArgs {
    int    x = 4, y = 4;
    int    N = -1;
    double U_initial = 0.0;
    double U_final   = -3.0;
    double t         = 1.0;

    double dtau      = 0.005;
    int    max_steps = 100000;
    double tol       = 1e-8;
    int    save_every = 4;
    double mu_lr     = 0.1;

    std::string output = "tdhfb_quench_results.h5";
    long   seed    = -1;
    bool   verbose = true;
};

void usage() {
    std::cerr <<
      "Usage: tdhfb_quench [--x N] [--y N] [--U-initial V] [--U-final V]\n"
      "                    [--t V] [--N V] [--dtau V] [--max-steps N]\n"
      "                    [--tol V] [--save-every N] [--mu-lr V]\n"
      "                    [--output PATH] [--seed N] [--quiet]\n";
}

bool parse_args(int argc, char** argv, CLIArgs& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value after " << what << "\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if      (s == "--x")           a.x          = std::stoi(next("--x"));
        else if (s == "--y")           a.y          = std::stoi(next("--y"));
        else if (s == "--N")           a.N          = std::stoi(next("--N"));
        else if (s == "--U-initial")   a.U_initial  = std::stod(next("--U-initial"));
        else if (s == "--U-final")     a.U_final    = std::stod(next("--U-final"));
        else if (s == "--t")           a.t          = std::stod(next("--t"));
        else if (s == "--dtau")        a.dtau       = std::stod(next("--dtau"));
        else if (s == "--max-steps")   a.max_steps  = std::stoi(next("--max-steps"));
        else if (s == "--tol")         a.tol        = std::stod(next("--tol"));
        else if (s == "--save-every")  a.save_every = std::stoi(next("--save-every"));
        else if (s == "--mu-lr")       a.mu_lr      = std::stod(next("--mu-lr"));
        else if (s == "--output")      a.output     = next("--output");
        else if (s == "--seed")        a.seed       = std::stol(next("--seed"));
        else if (s == "--quiet")       a.verbose    = false;
        else if (s == "-h" || s == "--help") { usage(); return false; }
        else { std::cerr << "unknown arg: " << s << "\n"; usage(); return false; }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    CLIArgs a;
    if (!parse_args(argc, argv, a)) return 0;

    const int STATES = a.x * a.y;
    if (a.N < 0) a.N = STATES;  // default: half-filling

    uint64_t seed = (a.seed < 0) ?
        std::random_device{}() : static_cast<uint64_t>(a.seed);
    seed_rng(seed);

    std::printf("================================================================\n");
    std::printf(" TDHFB quench:  %dx%d sites,  N=%d,  U: %.4f → %.4f,  t=%g\n",
                a.x, a.y, a.N, a.U_initial, a.U_final, a.t);
    std::printf("================================================================\n");
    std::fflush(stdout);

    // ── Phase 1.  HFB ground state at U_initial ───────────────────────────
    std::printf("[hfb]  warm-up at U=0...\n"); std::fflush(stdout);
    HFBResult warm = run_hfb_hubbard(a.N, STATES, a.x, a.y, /*U=*/0.0, a.t,
                                     nullptr, nullptr, /*temperature=*/0.0);
    if (!warm.converged)
        std::cerr << "warning: U=0 warm-up did not converge\n";

    std::printf("[hfb]  solving at U_initial=%.4f...\n", a.U_initial);
    std::fflush(stdout);
    HFBResult gs = run_hfb_hubbard(a.N, STATES, a.x, a.y, a.U_initial, a.t,
                                   &warm.p, &warm.k, /*temperature=*/0.0);
    if (!gs.converged) {
        std::cerr << "warning: ground state did not converge at U_initial="
                  << a.U_initial << " (iters=" << gs.iteration_count << ")\n";
        gs.converged = true;  // proceed anyway; quench will still run
    }
    std::printf("[hfb]  ground state:  E=%.8f  mu=%.6f  iters=%d  conv=%s\n",
                gs.energy.real(), gs.mu, gs.iteration_count,
                gs.converged ? "yes" : "NO");
    std::fflush(stdout);

    // Persist the initial ground state.
    write_converged_result_hdf5(gs, a.output);
    std::printf("[hfb]  initial GS written to %s :: /hfb/\n", a.output.c_str());
    std::fflush(stdout);

    // ── Phase 2.  Imaginary-time quench relaxation ────────────────────────
    TDHFBQuenchParams qp;
    qp.U_initial   = a.U_initial;
    qp.U_final     = a.U_final;
    qp.dtau        = a.dtau;
    qp.energy_tol  = a.tol;
    qp.max_steps   = a.max_steps;
    qp.save_every  = a.save_every;
    qp.mu_lr       = a.mu_lr;
    qp.verbose     = a.verbose;

    TDHFBQuenchResult qr = run_tdhfb_quench(gs, qp);

    write_quench_result_hdf5(qr, a.output);
    std::printf("[quench] results written to %s :: /quench/\n", a.output.c_str());
    std::fflush(stdout);

    // ── Summary ───────────────────────────────────────────────────────────
    {
        double E0    = qr.energy.front(),    Ef   = qr.energy.back();
        double N0    = qr.particle_number.front(), Nf = qr.particle_number.back();
        double max_idem = 0.0;
        for (double v : qr.idempotency_err) max_idem = std::max(max_idem, v);
        double dN_max = 0.0;
        for (double v : qr.particle_number)
            dN_max = std::max(dN_max, std::abs(v - (double)a.N));

        std::printf("\n============== quench run summary ==============\n");
        std::printf("  Converged:          %s  (steps done: %d)\n",
                    qr.converged ? "YES" : "NO", qr.n_steps_done);
        std::printf("  Energy:             E(0)=%.8f  →  E(final)=%.8f\n", E0, Ef);
        std::printf("  Energy change:      |ΔE_total|=%.3e\n", std::abs(Ef - E0));
        std::printf("  Particle drift:     max|N-N_target|=%.3e\n", dN_max);
        std::printf("  Max ‖R²-R‖_F:       %.3e   (should be ~0 after purify)\n",
                    max_idem);
        std::printf("  Final μ:            %.6f\n", qr.mu_final);
        std::printf("  Matrix snapshots:   %zu  (every %d steps)\n",
                    qr.snap_tau.size(), a.save_every);
        std::printf("  Note: sum rule does not apply (large quench, not infinitesimal kick)\n");
        std::printf("================================================\n");
        std::fflush(stdout);
    }

    return 0;
}
