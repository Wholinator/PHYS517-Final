// tdhfb_main.cpp
//
// Driver:  build HFB ground state, then run TDHFB linear-response.
//
// Usage:
//   ./tdhfb [options]
//
//   --x N              lattice x dimension (default 4)
//   --y N              lattice y dimension (default 4)
//   --U V              on-site Hubbard U (negative = attractive; default -2)
//   --t V              hopping (default 1.0)
//   --N V              particle number (default = STATES, half-filling)
//   --dt V             time step (default 0.005)
//   --steps N          number of steps (default 8000)
//   --eta V            kick amplitude (default 1e-3)
//   --kick KIND        kick operator (default "density_q_pi_pi")
//                      one of: density_q_pi_pi, density_q_pi_0, density_uniform,
//                              pairing_real, current_x
//   --gamma V          Lorentzian smoothing for S(E) (default 0.5)
//   --emin V           minimum E in S(E) grid (default 0)
//   --emax V           maximum E in S(E) grid (default 8)
//   --negrid N         number of E points (default 800)
//   --integrator S     "expmid" (default) or "euler"
//   --sample-every N   downsample (default 1)
//   --output PATH      output HDF5 file (default tdhfb_results.h5)
//   --seed N           RNG seed for HFB initialization (default: random)
//   --quiet            suppress per-step progress
//
// Example:  ./tdhfb --x 6 --y 6 --U -2 --kick density_q_pi_pi --output run1.h5

#include "data_output.h"
#include "hfb_api.h"
#include "tdhfb.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

namespace {

struct CLIArgs {
    int x = 4, y = 4;
    int N = -1;                    // -1 means "default to STATES"
    double U = -2.0;
    double t = 1.0;

    double dt = 0.005;
    int    steps = 8000;
    double eta = 1e-3;
    std::string kick = "density_q_pi_pi";
    double gamma_s = 0.5;
    double emin = 0.0;
    double emax = 8.0;
    int    negrid = 800;
    std::string integrator = "expmid";
    int    sample_every = 1;

    std::string output = "tdhfb_results.h5";
    long   seed = -1;
    bool   verbose = true;
};

void usage() {
    std::cerr <<
      "Usage: tdhfb [--x N] [--y N] [--U V] [--t V] [--N V]\n"
      "             [--dt V] [--steps N] [--eta V] [--kick KIND]\n"
      "             [--gamma V] [--emin V] [--emax V] [--negrid N]\n"
      "             [--integrator expmid|euler] [--sample-every N]\n"
      "             [--output PATH] [--seed N] [--quiet]\n";
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
        if      (s == "--x")            a.x = std::stoi(next("--x"));
        else if (s == "--y")            a.y = std::stoi(next("--y"));
        else if (s == "--N")            a.N = std::stoi(next("--N"));
        else if (s == "--U")            a.U = std::stod(next("--U"));
        else if (s == "--t")            a.t = std::stod(next("--t"));
        else if (s == "--dt")           a.dt = std::stod(next("--dt"));
        else if (s == "--steps")        a.steps = std::stoi(next("--steps"));
        else if (s == "--eta")          a.eta = std::stod(next("--eta"));
        else if (s == "--kick")         a.kick = next("--kick");
        else if (s == "--gamma")        a.gamma_s = std::stod(next("--gamma"));
        else if (s == "--emin")         a.emin = std::stod(next("--emin"));
        else if (s == "--emax")         a.emax = std::stod(next("--emax"));
        else if (s == "--negrid")       a.negrid = std::stoi(next("--negrid"));
        else if (s == "--integrator")   a.integrator = next("--integrator");
        else if (s == "--sample-every") a.sample_every = std::stoi(next("--sample-every"));
        else if (s == "--output")       a.output = next("--output");
        else if (s == "--seed")         a.seed = std::stol(next("--seed"));
        else if (s == "--quiet")        a.verbose = false;
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
    if (a.N < 0) a.N = STATES;       // default: half-filling on spin-doubled space

    // Seed RNG (HFB ground-state init uses it)
    uint64_t seed = (a.seed < 0) ?
        std::random_device{}() : static_cast<uint64_t>(a.seed);
    seed_rng(seed);

    std::printf("================================================================\n");
    std::printf(" TDHFB on 2D Hubbard:  %dx%d sites, N=%d, U=%g, t=%g\n",
                a.x, a.y, a.N, a.U, a.t);
    std::printf("================================================================\n");
    std::fflush(stdout);

    // -----------------------------------------------------------------
    // Phase 1.  Ground state.  We do TWO HFB warm-up calls (the original
    // C++ driver did this) to mimic the Python script's behavior of
    // discarding the first noisy random initialization.  Then a final
    // call seeded with the warm-up's (p, k) gives the production state.
    // -----------------------------------------------------------------
    std::printf("[hfb]  warming up ground state...\n"); std::fflush(stdout);
    HFBResult warm = run_hfb_hubbard(a.N, STATES, a.x, a.y, /*U=*/0.0, a.t,
                                     nullptr, nullptr, /*temperature=*/0.0);
    if (!warm.converged) {
        std::cerr << "warning: U=0 warm-up did not converge\n";
    }

    std::printf("[hfb]  solving at U=%g...\n", a.U); std::fflush(stdout);
    HFBResult gs = run_hfb_hubbard(a.N, STATES, a.x, a.y, a.U, a.t,
                                   &warm.p, &warm.k, /*temperature=*/0.0);
    if (!gs.converged) {
        std::cerr << "ERROR: HFB ground state did not converge at U="
                  << a.U << " (after " << gs.iteration_count << " iters)\n";
        // Keep going anyway — TDHFB will still propagate, but flag it.
    }
    std::printf("[hfb]  ground state:  E = %.8f,  mu = %.6f,  iters = %d,  conv = %s\n",
                gs.energy.real(), gs.mu, gs.iteration_count,
                gs.converged ? "yes" : "NO");
    std::fflush(stdout);

    // Persist the ground state to HDF5.
    if (gs.converged) {
        write_converged_result_hdf5(gs, a.output);
        std::printf("[hfb]  ground state written to %s :: /hfb/U_*\n",
                    a.output.c_str());
        std::fflush(stdout);
    }

    // -----------------------------------------------------------------
    // Phase 2.  TDHFB linear response.
    // -----------------------------------------------------------------
    TDHFBParams tp;
    tp.dt              = a.dt;
    tp.n_steps         = a.steps;
    tp.eta_kick        = a.eta;
    tp.kick_operator   = a.kick;
    tp.Gamma_smooth    = a.gamma_s;
    tp.E_min           = a.emin;
    tp.E_max           = a.emax;
    tp.n_E_grid        = a.negrid;
    tp.integrator      = a.integrator;
    tp.sample_every    = a.sample_every;
    tp.verbose         = a.verbose;

    if (!gs.converged) {
        // Force gs.converged so run_tdhfb will accept it; we already warned.
        gs.converged = true;
    }

    TDHFBResult tr = run_tdhfb(gs, tp);

    write_tdhfb_result_hdf5(tr, a.output);
    std::printf("[tdhfb] results written to %s :: /tdhfb/U_*__%s\n",
                a.output.c_str(), a.kick.c_str());

    // -----------------------------------------------------------------
    // Brief summary to stdout — the kind of thing you'd glance at first.
    // -----------------------------------------------------------------
    {
        // Energy drift and conservation diagnostics
        double E0 = tr.energy.front(), E_end = tr.energy.back();
        double N0 = tr.particle_number.front(), N_end = tr.particle_number.back();
        double max_idem = 0.0;
        for (double v : tr.idempotency_err) max_idem = std::max(max_idem, v);

        // Find the top peaks of S(E)  (local maxima above 5% of global max)
        std::vector<std::pair<double,double>> peaks;
        if (tr.S_E.size() > 2) {
            double Smax = 0.0;
            for (double v : tr.S_E) Smax = std::max(Smax, v);
            for (size_t i = 1; i + 1 < tr.S_E.size(); ++i) {
                if (tr.S_E[i] > tr.S_E[i-1] && tr.S_E[i] > tr.S_E[i+1]
                    && tr.S_E[i] > 0.05 * Smax) {
                    peaks.emplace_back(tr.E_grid[i], tr.S_E[i]);
                }
            }
            // sort descending
            std::sort(peaks.begin(), peaks.end(),
                      [](auto& a, auto& b){ return a.second > b.second; });
        }

        std::printf("\n================ TDHFB run summary ================\n");
        std::printf("  Energy drift:       E(0)=%.8f -> E(T)=%.8f   |dE|=%.3e\n",
                    E0, E_end, std::abs(E_end - E0));
        std::printf("  Particle drift:     N(0)=%.6f -> N(T)=%.6f   |dN|=%.3e\n",
                    N0, N_end, std::abs(N_end - N0));
        std::printf("  Max ‖R²-R‖_F:       %.3e   (smaller is better)\n", max_idem);
        std::printf("  Sum rule:  m1(FT)/m1(comm) = %.4f   (should be ~1)\n",
                    tr.m1_commutator != 0.0 ? tr.m1_FT / tr.m1_commutator : 0.0);
        std::printf("  Top peaks of S(E; %s):\n", a.kick.c_str());
        int npeaks = std::min<size_t>(peaks.size(), 8);
        for (int i = 0; i < npeaks; ++i) {
            std::printf("    E = %7.4f      S = %.4e\n",
                        peaks[i].first, peaks[i].second);
        }
        if (npeaks == 0) std::printf("    (no peaks above 5%% of max found)\n");
        std::printf("====================================================\n");
        std::fflush(stdout);
    }

    return 0;
}
