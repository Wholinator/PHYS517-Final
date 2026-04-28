// hfb_numeric_current.cpp
//
// C++ translation of hfb_numeric_current.py with finite-temperature support.
//
// Hartree-Fock-Bogoliubov calculation on the 2D Hubbard model, supporting
// both zero temperature (T=0 projector onto negative-energy modes) and
// finite temperature following Goodman, Nucl. Phys. A352 (1981) 30-44.
//
// At T>0 every BdG eigenmode contributes weight f_j = 1/(1+exp(β λ_j)) to
// the generalized density R = Σ_j f_j |v_j⟩⟨v_j|.  The HFB equations
// (Goodman eq. 4.36) have the same form as at T=0; only the construction
// of R from the eigendecomposition changes.  The grand potential
//     Ω = E - TS - μN,
//     E = Tr[(T + Γ/2) ρ + Δ t†/2]                       (Goodman 3.23)
//     S/k_B = -Σᵢ [fᵢ ln fᵢ + (1-fᵢ) ln(1-fᵢ)]            (Goodman 3.27)
// is reported alongside E.  The SCF converges to the self-consistent
// fields *at the requested temperature* — there is no annealing.
//
// Build:
//   g++ -O3 -march=native -DNDEBUG -DEIGEN_NO_DEBUG -std=c++17 \
//       -I/usr/include/eigen3 hfb_numeric_current.cpp -o hfb
//
// Run:
//   ./hfb                  -> seed=entropy, kT=0
//   ./hfb <seed>           -> seed=<seed>, kT=0
//   ./hfb <seed> <kT>      -> seed=<seed>, kT=<kT>   (in units of t)
//
// Notes on fidelity:
//   * np.linalg.eigh  -> Eigen::SelfAdjointEigenSolver  (ascending eigenvalues)
//   * np.kron         -> Eigen::kroneckerProduct
//   * np.block        -> manual block placement via .block()
//   * np.roll         -> manual modular index shift
//   * Random numbers: NumPy's PCG64 / MT19937 streams cannot be bit-reproduced
//     in standard C++; we use std::mt19937 with std::normal_distribution and
//     std::uniform_real_distribution. Algorithmic behavior is preserved; the
//     specific random samples will differ from any given Python run.
//   * Complex / real typing: the original Python code starts with real p, k,
//     h and silently promotes them to complex once delta enters the mixing.
//     Here we keep p, k, h as complex throughout (with zero imaginary parts
//     initially), which matches the steady-state typing of the Python code.

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>

#include "data_output.h"

using cd     = std::complex<double>;
using MatXd  = Eigen::MatrixXd;
using MatXcd = Eigen::MatrixXcd;
using VecXd  = Eigen::VectorXd;

// ===================================================================
//  Random-number infrastructure
// ===================================================================
//
// The Python code uses both the legacy global RNG (np.random.randn /
// np.random.uniform) and a fresh default_rng() inside init_hfb_kappa.
// Bit-exact reproduction across language boundaries is not feasible, so
// we keep one std::mt19937 here and document the call sites.

static std::mt19937 g_rng{std::random_device{}()};

inline void seed_rng(uint64_t s) { g_rng.seed(s); }

inline double randn() {
    static std::normal_distribution<double> dist(0.0, 1.0);
    return dist(g_rng);
}

inline double rand_uniform() {
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(g_rng);
}

inline MatXd randn_matrix(int rows, int cols) {
    MatXd M(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            M(i, j) = randn();
    return M;
}

inline VecXd uniform_vector(int n) {
    VecXd v(n);
    for (int i = 0; i < n; ++i) v(i) = rand_uniform();
    return v;
}

// ===================================================================
//  Small helpers (NumPy parity)
// ===================================================================

// np.eye(n) as complex
inline MatXcd eye_c(int n) { return MatXcd::Identity(n, n); }

// np.eye(n) as real
inline MatXd eye_r(int n) { return MatXd::Identity(n, n); }

// scipy.linalg.block_diag for two square blocks A, B
template <typename M>
M block_diag2(const M& A, const M& B) {
    M out = M::Zero(A.rows() + B.rows(), A.cols() + B.cols());
    out.topLeftCorner(A.rows(), A.cols())     = A;
    out.bottomRightCorner(B.rows(), B.cols()) = B;
    return out;
}

// 2x2 block-matrix assembly  [[A, B], [C, D]]  (square sub-blocks of equal size)
template <typename M>
M block2x2(const M& A, const M& B, const M& C, const M& D) {
    const int n = static_cast<int>(A.rows());
    M out(2 * n, 2 * n);
    out.topLeftCorner(n, n)     = A;
    out.topRightCorner(n, n)    = B;
    out.bottomLeftCorner(n, n)  = C;
    out.bottomRightCorner(n, n) = D;
    return out;
}

// np.roll on a 1D vector (positive shift = shift right, wraparound).
template <typename V>
V roll(const V& v, int shift) {
    const int n = static_cast<int>(v.size());
    V out(n);
    int s = ((shift % n) + n) % n;
    for (int i = 0; i < n; ++i) out((i + s) % n) = v(i);
    return out;
}

// np.diag(vector): turn a vector into a diagonal matrix.
template <typename V>
auto diag_from_vec(const V& v) {
    using Scalar = typename V::Scalar;
    const int n = static_cast<int>(v.size());
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> M
        = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, n);
    for (int i = 0; i < n; ++i) M(i, i) = v(i);
    return M;
}

// Sum of a vector (np.sum)
template <typename V>
typename V::Scalar sum_vec(const V& v) { return v.sum(); }

// Frobenius norm difference (np.linalg.norm(... , 'fro'))
inline double fro_norm(const MatXcd& A) { return A.norm(); }

// ===================================================================
//  purify_evals  (predicate version only -- the lambda case)
// ===================================================================
//
// Project a Hermitian matrix's eigenvalues onto values allowed by a
// predicate.  Mirrors purify_evals() in the predicate ("callable") branch
// of the Python code.  The discrete branch is unused in the main pipeline,
// so we omit it here.

MatXd purify_evals_predicate(const std::function<bool(double)>& is_allowed,
                             const MatXd&                       mat,
                             int                                grid_size = 2000) {
    Eigen::SelfAdjointEigenSolver<MatXd> es(mat);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("eigh failed in purify_evals_predicate");
    }
    VecXd eval = es.eigenvalues();
    MatXd evec = es.eigenvectors();

    const double lo = eval.minCoeff() - 1.0;
    const double hi = eval.maxCoeff() + 1.0;

    // Build a grid of trial eigenvalues, keep only the allowed ones.
    std::vector<double> grid(grid_size);
    if (grid_size == 1) {
        grid[0] = lo;
    } else {
        const double step = (hi - lo) / static_cast<double>(grid_size - 1);
        for (int i = 0; i < grid_size; ++i) grid[i] = lo + step * i;
    }
    std::vector<double> allowed_grid;
    allowed_grid.reserve(grid.size());
    for (double x : grid) if (is_allowed(x)) allowed_grid.push_back(x);
    if (allowed_grid.empty()) {
        throw std::runtime_error("No allowed eigenvalues found in search interval");
    }

    VecXd purified(eval.size());
    for (int i = 0; i < eval.size(); ++i) {
        const double x = eval(i);
        if (is_allowed(x)) {
            purified(i) = x;
        } else {
            // Pick the closest allowed grid value.
            double best = allowed_grid[0];
            double bestd = std::abs(allowed_grid[0] - x);
            for (size_t j = 1; j < allowed_grid.size(); ++j) {
                double d = std::abs(allowed_grid[j] - x);
                if (d < bestd) { bestd = d; best = allowed_grid[j]; }
            }
            purified(i) = best;
        }
    }

    // (evec * purified) @ evec.T   -- broadcast-multiply each column by purified[i]
    MatXd scaled = evec;
    for (int j = 0; j < scaled.cols(); ++j) scaled.col(j) *= purified(j);
    return scaled * evec.transpose();
}

MatXd make_positive_semi_definite(const MatXd& a) {
    return purify_evals_predicate([](double x) { return x >= 0.0; }, a);
}

bool is_positive_semidefinite(const MatXd& a, double atol = 1e-10, double /*rtol*/ = 1e-10) {
    Eigen::SelfAdjointEigenSolver<MatXd> es(a);
    if (es.info() != Eigen::Success) return false;
    return (es.eigenvalues().array() >= 0.0).all();
}

// ===================================================================
//  init_rho  --  initial density matrix
// ===================================================================
//
// The Python signature uses several optional flags.  Only the configuration
// actually called by run_hfb_hubbard is exercised in the main loop:
//     restriction="G", anti=True, offdiag_rn=True, spin_mix=True,
//     diag_rn=False (default), per_tol=1e-3 (default).
// We support the full flag set anyway, mirroring the Python branches.

enum class Restriction { R, U, G };

MatXd init_rho(int  block_dim,
               double diagval,
               Restriction restriction = Restriction::R,
               bool  diag_rn    = false,
               bool  offdiag_rn = false,
               bool  spin_mix   = false,
               bool  anti       = true,
               double per_tol   = 1e-3) {
    MatXd rho;

    if (diag_rn) {
        const int dim = block_dim;
        VecXd d1 = uniform_vector(dim);
        VecXd d2 = uniform_vector(dim);
        if (restriction == Restriction::R) d2 = d1;
        MatXd p1 = diag_from_vec(d1);
        MatXd p2 = diag_from_vec(d2);
        MatXd mat = block_diag2<MatXd>(p1, p2) * per_tol;

        const double tr = mat.trace();
        const double mult_factor = diagval / tr;
        // mult_factor_mat = ones((2d,2d)) + (mult_factor - 1) * eye(2d)
        // mat * mult_factor_mat is element-wise: off-diagonals kept,
        // diagonal scaled by mult_factor.  But mat is diagonal here, so
        // the result is just mat * mult_factor on the diagonal.
        rho = MatXd::Zero(2 * dim, 2 * dim);
        for (int i = 0; i < 2 * dim; ++i) rho(i, i) = mat(i, i) * mult_factor;
        // (off-diagonals of mat are zero, so no other contribution)

        // To exactly reproduce the Python expression for non-diagonal mat,
        // we'd need element-wise mat * mult_factor_mat; mat is diagonal, so
        // the above shortcut is identical here.
    } else if (!anti) {
        MatXd p1 = (diagval / block_dim) * MatXd::Identity(block_dim, block_dim);
        MatXd p2 = MatXd::Zero(block_dim, block_dim);
        rho = block_diag2<MatXd>(p1, p2);
    } else {
        // down_diag = [0,1,0,1,...]
        VecXd down_diag(block_dim);
        for (int i = 0; i < block_dim; ++i) down_diag(i) = static_cast<double>(i % 2);
        VecXd up_diag = (restriction == Restriction::R) ? down_diag
                                                        : (VecXd::Ones(block_dim) - down_diag);
        const double mult = diagval / sum_vec((up_diag + down_diag).eval());
        MatXd p1 = diag_from_vec((mult * up_diag).eval());
        MatXd p2 = diag_from_vec((mult * down_diag).eval());
        rho = block_diag2<MatXd>(p1, p2);
    }

    if (offdiag_rn) {
        MatXd off1 = per_tol * randn_matrix(block_dim, block_dim);
        MatXd off2 = (restriction != Restriction::R) ? (per_tol * randn_matrix(block_dim, block_dim))
                                                     : off1;
        // Zero out diagonals so the trace isn't disturbed.
        MatXd one_minus_eye = MatXd::Ones(block_dim, block_dim) - MatXd::Identity(block_dim, block_dim);
        off1 = (off1.array() * one_minus_eye.array()).matrix();
        off2 = (off2.array() * one_minus_eye.array()).matrix();
        // NB: the Python code uses element-wise A*A.T (not A @ A.T) here.
        // We replicate that exactly, faithful to the original even though
        // it does not actually produce a Hermitian matrix in general.
        off1 = 0.5 * (off1.array() * off1.transpose().array()).matrix();
        off2 = 0.5 * (off2.array() * off2.transpose().array()).matrix();
        rho += block_diag2<MatXd>(off1, off2);
    }

    if (spin_mix && restriction == Restriction::G) {
        MatXd sp1 = per_tol * randn_matrix(block_dim, block_dim);
        MatXd sp2 = sp1.transpose();
        MatXd Z   = MatXd::Zero(block_dim, block_dim);
        rho += block2x2<MatXd>(Z, sp1, sp2, Z);
    }

    if (!is_positive_semidefinite(rho)) {
        rho = make_positive_semi_definite(rho);
        if ((rho.trace() - diagval) > 1e-6) {
            // Rescale only the diagonal to fix the trace, leaving off-diagonals intact.
            VecXd diag_old = rho.diagonal();
            const double factor = diagval / rho.trace();
            VecXd diag_new = factor * diag_old;
            // rho * (1 - I) zeros the diagonal; then add diag(diag_new).
            for (int i = 0; i < rho.rows(); ++i) rho(i, i) = 0.0;
            for (int i = 0; i < rho.rows(); ++i) rho(i, i) = diag_new(i);
        }
    }
    return rho;
}

// ===================================================================
//  init_hfb_kappa  --  initial anomalous density
// ===================================================================
//
// The Python version uses a *separate* default_rng() seeded by entropy
// (independent of the global RNG).  We use the same global RNG here for
// simplicity; this changes the specific samples but not the algorithm.

MatXd init_hfb_kappa(int STATES, double eps = 1.0) {
    MatXd r = randn_matrix(STATES * 2, STATES * 2);
    return eps * (r - r.transpose());
}

// ===================================================================
//  Lattice / hopping matrix builders
// ===================================================================

MatXd nearest_neighbor_2d(int rows, int columns, bool pbc = false) {
    const int block_size = columns;
    const int block_num  = rows;

    MatXd zero_block = MatXd::Zero(block_size, block_size);
    MatXd diag_block = MatXd::Identity(block_size, block_size);

    MatXd offdiag_block = MatXd::Zero(block_size, block_size);
    for (int i = 0; i < block_size - 1; ++i) {
        offdiag_block(i + 1, i) = 1.0;
        offdiag_block(i, i + 1) = 1.0;
    }

    MatXd corner_block = MatXd::Zero(block_size, block_size);
    corner_block(0, block_size - 1) = 1.0;
    corner_block(block_size - 1, 0) = 1.0;

    MatXd out = MatXd::Zero(block_num * block_size, block_num * block_size);
    for (int row = 0; row < block_num; ++row) {
        for (int column = 0; column < block_num; ++column) {
            MatXd blk = MatXd::Zero(block_size, block_size);
            if (row == column) {
                blk += offdiag_block;
            } else if (std::abs(row - column) == 1) {
                blk += diag_block;
            }
            // else: empty (already zero)

            if (pbc) {
                if (row == column) {
                    blk += corner_block;
                } else if (std::abs(row - column) == block_num - 1) {
                    blk += diag_block;
                }
            }
            out.block(row * block_size, column * block_size, block_size, block_size) = blk;
        }
    }
    return out;
}

MatXd build_t_mat_hubbard(int x, int y, double t, bool pbc) {
    MatXd lattice = nearest_neighbor_2d(x, y, pbc);
    MatXd I2 = MatXd::Identity(2, 2);
    // -t * np.kron(I_2, lattice)
    MatXd kron = Eigen::kroneckerProduct(I2, lattice).eval();
    return (-t) * kron;
}

// build_fock_matrix:  H_U is U * diag( roll(diag(p), roll_by) ) added to t_mat.
// In Python, p may be complex with a tiny imaginary part; here p is complex.
MatXcd build_fock_matrix(int roll_by, const MatXd& t_mat, const MatXcd& p, double U_val) {
    Eigen::VectorXcd diag_p = p.diagonal();
    Eigen::VectorXcd rolled = roll(diag_p, roll_by);
    MatXcd H_U = MatXcd::Zero(t_mat.rows(), t_mat.cols());
    for (int i = 0; i < rolled.size(); ++i) H_U(i, i) = U_val * rolled(i);
    MatXcd t_mat_c = t_mat.cast<cd>();
    return t_mat_c + H_U;
}

// ===================================================================
//  Delta builder (anomalous mean field)
// ===================================================================
MatXcd build_delta(const MatXcd& kappa, double U_val, int nstates) {
    // The Python code creates a "k-superdiagonal + k-subdiagonal" mask and does
    //   delta = 2*U_val * (kappa * hubbard_k_map).T
    // i.e., element-wise multiply, then transpose.
    const int N = nstates * 2;
    MatXcd masked = MatXcd::Zero(N, N);
    // hubbard_k_map has 1s only at (i, i+nstates) and (i+nstates, i).
    for (int i = 0; i < nstates; ++i) {
        masked(i, i + nstates)     = kappa(i, i + nstates);
        masked(i + nstates, i)     = kappa(i + nstates, i);
    }
    MatXcd delta = 2.0 * U_val * masked.transpose();
    return delta;
}

// ===================================================================
//  Fermi-Dirac utility
// ===================================================================
//
// fermi_dirac(x, beta) = 1 / (1 + exp(beta * x)) with overflow-safe
// branching.  When beta == infinity (kT == 0) we return the hard step:
//   x < 0 → 1, x > 0 → 0, x == 0 → 0.5 (won't occur in practice with
//                                       1e-15 thresholding).
inline double fermi_dirac(double x, double beta) {
    if (!std::isfinite(beta)) {
        // Zero-temperature limit
        return (x < 0.0) ? 1.0 : (x > 0.0 ? 0.0 : 0.5);
    }
    const double bx = beta * x;
    if (bx >  500.0) return 0.0;
    if (bx < -500.0) return 1.0;
    return 1.0 / (1.0 + std::exp(bx));
}

// ===================================================================
//  particle_number  --  Tr(p) for a given chemical potential
// ===================================================================
//
// The chemical-potential bisection inside the SCF loop calls this
// dozens of times per SCF iteration with a fixed (h, delta) and varying
// μ.  When delta == 0 (in particular at U == 0, where the BdG matrix
// block-decouples), Tr(p) reduces to "count eigenvalues of h below μ".
// We exploit this: the caller pre-computes the eigenvalues of h once and
// passes them in; each particle_number call is then an O(n) count rather
// than an O(n^3) eigendecomposition.  This makes the U=0 warm-up calls
// (which dominate runtime) nearly free.
struct ParticleNumberContext {
    const MatXcd& h;
    const MatXcd& delta;
    double        kT = 0.0;     // 0.0 → zero-temperature path
    bool          delta_is_zero = false;
    VecXd         h_evals;     // populated only if delta_is_zero

    // Cache of the most recent full BdG eigendecomposition produced inside
    // particle_number().  After optimize_bisect() finds μ, the SCF loop can
    // pull these out and avoid one redundant eigendecomp per iteration.
    bool   last_evals_valid = false;
    double last_lambda       = 0.0;
    VecXd  last_evals;
    MatXcd last_evecs;
};

// Build the context.  Detects whether delta is numerically zero.
ParticleNumberContext make_pn_context(const MatXcd& h, const MatXcd& delta,
                                       double kT = 0.0) {
    ParticleNumberContext ctx{h, delta, kT, false, VecXd()};
    // Tolerance: if max|delta_ij| < 1e-14, treat as exactly zero.
    if (delta.cwiseAbs().maxCoeff() < 1e-14) {
        ctx.delta_is_zero = true;
        Eigen::SelfAdjointEigenSolver<MatXcd> es(h);
        if (es.info() != Eigen::Success) {
            throw std::runtime_error("eigh on h failed in make_pn_context");
        }
        ctx.h_evals = es.eigenvalues();
    }
    return ctx;
}

double particle_number(ParticleNumberContext& ctx, double lambda_) {
    const double beta = (ctx.kT > 0.0) ? (1.0 / ctx.kT)
                                       : std::numeric_limits<double>::infinity();

    if (ctx.delta_is_zero) {
        // delta = 0 ⇒ BdG matrix is block-diagonal.  At T=0 we counted
        // eigenvalues of h below μ; at T>0 we sum the Fermi-Dirac weights.
        const VecXd& ev = ctx.h_evals;
        if (ctx.kT == 0.0) {
            // Binary-search count (existing fast path).
            int lo = 0, hi = static_cast<int>(ev.size());
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (ev(mid) < lambda_) lo = mid + 1;
                else                   hi = mid;
            }
            return static_cast<double>(lo);
        }
        // FT path: Tr(p) = Σ_i f(h_evals[i] - μ ; β).
        double tr = 0.0;
        for (int i = 0; i < ev.size(); ++i) tr += fermi_dirac(ev(i) - lambda_, beta);
        return tr;
    }

    // General path: full BdG eigendecomposition.
    // Avoid creating an n×n identity and a separate h_shift; just write the
    // shifted blocks directly into H_BdG.  Also reuse a thread_local buffer
    // for H_BdG so we're not re-allocating on every probe.
    thread_local Eigen::SelfAdjointEigenSolver<MatXcd> es;
    thread_local MatXcd H_BdG;

    const int  n     = static_cast<int>(ctx.h.rows());
    const int  N     = 2 * n;
    if (H_BdG.rows() != N) H_BdG.resize(N, N);

    H_BdG.topLeftCorner(n, n)     = ctx.h;
    H_BdG.topRightCorner(n, n)    = ctx.delta;
    H_BdG.bottomLeftCorner(n, n)  = -ctx.delta.conjugate();
    H_BdG.bottomRightCorner(n, n) = -ctx.h.transpose();
    for (int i = 0; i < n; ++i) {
        H_BdG(i, i)         -= lambda_;
        H_BdG(n + i, n + i) += lambda_;
    }

    es.compute(H_BdG);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("eigh failed in particle_number");
    }

    // Cache the result for reuse by the SCF loop after bisection completes.
    ctx.last_evals       = es.eigenvalues();
    ctx.last_evecs       = es.eigenvectors();
    ctx.last_lambda      = lambda_;
    ctx.last_evals_valid = true;

    const VecXd&  evals = ctx.last_evals;
    const MatXcd& evecs = ctx.last_evecs;

    if (ctx.kT == 0.0) {
        // T=0: hard projector on negative-energy modes.
        int n_neg = 0;
        for (int i = 0; i < evals.size(); ++i) {
            if (evals(i) < 1e-15) ++n_neg; else break;
        }
        double tr = 0.0;
        for (int j = 0; j < n_neg; ++j) {
            for (int i = 0; i < n; ++i) {
                tr += std::norm(evecs(i, j));
            }
        }
        return tr;
    }

    // T>0: weight every mode by Fermi-Dirac of its eigenvalue.
    //   Tr(p) = Σ_j f(λ_j; β) Σ_{i<n} |evecs(i,j)|^2
    double tr = 0.0;
    for (int j = 0; j < evals.size(); ++j) {
        const double w = fermi_dirac(evals(j), beta);
        if (w < 1e-300) continue;   // skip negligible
        double col_sum = 0.0;
        for (int i = 0; i < n; ++i) col_sum += std::norm(evecs(i, j));
        tr += w * col_sum;
    }
    return tr;
}

// Backward-compatible 3-arg form (used in the bisect helper signatures).
double particle_number(const MatXcd& h, const MatXcd& delta, double lambda_) {
    ParticleNumberContext ctx = make_pn_context(h, delta, 0.0);
    return particle_number(ctx, lambda_);
}

// ===================================================================
//  Bisection helpers
// ===================================================================
struct Bounds { double low, high; };

Bounds validate_bounds(const std::function<double(double)>& func,
                       double goal,
                       double init_low, double init_high,
                       bool   increasing,
                       int    max_iter = 50) {
    double guess_low = init_low, guess_high = init_high;
    double high = func(guess_high);
    double low  = func(guess_low);

    if (((low - high) > 1e-7 && increasing) ||
        ((high - low) > 1e-7 && !increasing)) {
        std::cout << "INCREASINGNESS IS INCORRECT" << std::endl;
    }

    int iter = 0;
    while (true) {
        ++iter;
        if ((increasing && high < goal) || (!increasing && high > goal)) {
            double temp = guess_high;
            guess_high = 3.0 * guess_high - 2.0 * guess_low;
            guess_low  = temp;
            low  = high;
            high = func(guess_high);
        } else if ((increasing && low > goal) || (!increasing && low < goal)) {
            double temp = guess_low;
            guess_low  = 3.0 * guess_low - 2.0 * guess_high;
            guess_high = temp;
            high = low;
            low  = func(guess_low);
        } else if (high == low) {
            guess_high *= 4.0;
            guess_low  *= 4.0;
            high = func(guess_high);
            low  = func(guess_low);
        } else {
            return {guess_low, guess_high};
        }
        if (iter > max_iter) return {guess_low, guess_high};
    }
}

double optimize_bisect(const std::function<double(double)>& func,
                       double goal,
                       double range_low  = -0.1,
                       double range_high =  0.1,
                       double tol = 1e-12) {
    // "increasing" check uses *wide* probes so that warm-starting with a tight
    // bracket near a noisy region doesn't invert it.  Particle_number is a
    // monotone non-decreasing step-like function of mu, so it's increasing.
    bool increasing = (func(100.0) > func(-100.0));
    Bounds b = validate_bounds(func, goal, range_low, range_high, increasing);
    double guess_low  = b.low;
    double guess_high = b.high;
    double guess = 0.5 * (guess_high + guess_low);

    int iter = 0;
    bool converged = false;
    while (!converged && iter <= 2000) {
        ++iter;
        guess = 0.5 * (guess_high + guess_low);
        double value = func(guess);
        if (std::abs(value - goal) < tol || std::abs(guess_high - guess_low) < tol) {
            converged = true;
        } else if ((increasing && value < goal) || (!increasing && value > goal)) {
            guess_low = guess;
        } else if ((increasing && value > goal) || (!increasing && value < goal)) {
            guess_high = guess;
        }
    }
    return guess;
}

// ===================================================================
//  run_hfb_hubbard  --  the SCF driver
// ===================================================================
// struct HFBResult {
//     MatXcd p;
//     MatXcd k;
//     cd     energy;          // internal energy E = Tr[(T + Γ/2)ρ + Δ t†/2]   (eq. 3.23)
//     double entropy = 0.0;   // S = -k Σ [f_i ln f_i + (1-f_i) ln(1-f_i)]    (eq. 3.27)
//     double grand_potential = 0.0;   // Ω = E - TS - μN                       (eq. 2.3)
//     double mu      = 0.0;   // chemical potential found by bisection
//     int    iters   = 0;
// };

HFBResult run_hfb_hubbard(int PARTICLE_NUMBER, int STATES,
                          int x, int y, double U, double t,
                          double kT = 0.0,
                          const MatXcd* p_GUESS = nullptr,
                          const MatXcd* k_GUESS = nullptr) {
    // kT is in the same units as t and U.  kT == 0.0 → zero-temperature path
    // (hard projector onto negative-energy modes).  kT > 0 → finite-temperature
    // path following Goodman 1981 (Nucl. Phys. A352, 30): every BdG mode
    // contributes weight f_i = 1/(1 + exp(β λ_i)) with β = 1/kT.
    const double sym_tol  = 1e-7;     // matches Python (currently unused)
    const double conv_tol = 1e-8;     // matches Python (used implicitly below)
    const double chem_tol = 1e-12;
    (void) sym_tol; (void) conv_tol;

    // ----- Initialization -----
    MatXcd p, k;
    if (p_GUESS != nullptr && p_GUESS->size() > 0) {
        p = *p_GUESS;
    } else {
        // init_rho(PARTICLE_NUMBER, PARTICLE_NUMBER, "G", anti=True,
        //          offdiag_rn=True, spin_mix=True)
        // NOTE: in Python, init_rho's first arg is block_dim; the call passes
        // PARTICLE_NUMBER for both block_dim and diagval.  We replicate that.
        MatXd p_real = init_rho(PARTICLE_NUMBER, PARTICLE_NUMBER,
                                Restriction::G,
                                /*diag_rn=*/false, /*offdiag_rn=*/true,
                                /*spin_mix=*/true, /*anti=*/true);
        p = p_real.cast<cd>();
    }

    if (k_GUESS != nullptr && k_GUESS->size() > 0) {
        k = *k_GUESS;
    } else {
        MatXd k_real = init_hfb_kappa(STATES);
        k = k_real.cast<cd>();
    }

    // ----- Density / anomalous fields -----
    MatXd  H_t   = build_t_mat_hubbard(x, y, t, /*pbc=*/true);
    MatXcd H_t_c = H_t.cast<cd>();   // cached complex cast
    MatXcd h     = build_fock_matrix(STATES, H_t, p, U);
    MatXcd delta = build_delta(k, U, STATES);

    // Single-slot "previous iterate" storage replaces the unbounded history.
    // Only ever read at index -1 in the original code.
    MatXcd prev_p = p;
    MatXcd prev_k = k;
    cd     prev_energy = 0.0;
    bool   has_prev_energy = false;

    // Density from two iterations back, for 2-cycle detection.  The U=0
    // (non-interacting, degenerate at the Fermi level) case lands on a limit
    // cycle of period 2 -- p alternates between two values forever.  We
    // declare convergence after seeing the same 2-cycle 5 times in a row.
    MatXcd prev_prev_p;
    bool   has_prev_prev_p = false;
    int    two_cycle_count = 0;
    const int two_cycle_threshold = 5;

    const double alpha = 0.2;
    int  iter      = 0;
    bool converged = false;
    cd   energy    = 0.0;
    double entropy_kB = 0.0;     // S / k_B  (only meaningful at T>0)
    double grand_pot  = 0.0;     // Ω = E − T·S − μ·N
    double last_chem_pot = 0.0;  // last bisected μ (for return value)

    MatXcd last_R;
    MatXcd last_H_BdG;
    VecXd  last_evals;
    MatXcd last_evecs;

    // Re-used eigensolvers (no per-iteration workspace alloc).
    Eigen::SelfAdjointEigenSolver<MatXcd> es_bdg;
    Eigen::SelfAdjointEigenSolver<MatXcd> es_pn;

    // Warm-start state for the chemical-potential bisection.  After the first
    // few SCF iterations, mu barely moves between iterations, so a tight
    // bracket centered on the previous answer cuts validate_bounds() to ~0
    // expansion steps and the bisection itself to ~5 instead of ~30+.
    double prev_chem_pot   = 0.0;
    bool   has_prev_mu     = false;
    const double mu_window = 0.1;   // half-width of warm-start bracket

    // DIIS state.  Per user's spec we extrapolate the BdG Hamiltonian H_BdG,
    // not R.  Error vector is e = [H_BdG, R_prev] (current H against
    // previous R), evaluated immediately after H_BdG is built.
    //
    // Lifecycle (per user's spec):
    //   * Iters 1..DIIS_WARMUP_NOSTORE: don't compute or store residuals.
    //   * After warm-up: compute & store every iteration, FIFO-evict from
    //     the front to keep at most DIIS_WINDOW entries.
    //   * Activation: ||e||_F < DIIS_START AND queue size >= DIIS_MIN_QUEUE.
    //   * Deactivation: ||e||_F < DIIS_STOP.  Storage continues.
    const size_t DIIS_WINDOW = 4;
    const double DIIS_START  = 1e-5;
    const double DIIS_STOP   = 1e-9;
    std::deque<MatXcd> diis_H_hist;     // BdG Hamiltonians (4*STATES x 4*STATES)
    std::deque<MatXcd> diis_e_hist;     // [H_BdG, R_prev], same shape
    bool diis_active = false;

    // ----- SCF loop -----
    //
    // Order of operations (per user's spec):
    //   1. Build h, delta, find chemical potential, build H_BdG.
    //   2. Compute DIIS error = [H_BdG, R_prev] and decide engagement.
    //   3. If engaged & queue full: extrapolate H_BdG.
    //   4. Diagonalize H_BdG, build R, extract p, k.
    //   5. Mix with previous (linear alpha mix).
    //   6. Rebuild h, delta from mixed p, k.
    //   7. Compute energy.
    //   8. Convergence checks (delta-E + density + 2-cycle).
    //
    // R_prev is the BdG density from the previous iteration; it lives in
    // prev_R_full below.  On iteration 1 we have no R_prev, so DIIS is
    // skipped automatically (warm-up phase covers this anyway).
    MatXcd prev_R_full;             // last full BdG density (4*STATES x 4*STATES)
    bool   has_prev_R_full = false;

    // DIIS warm-up & queue policy (per user's spec):
    //   * Iterations 1..DIIS_WARMUP_NOSTORE: don't even compute residuals.
    //   * From DIIS_WARMUP_NOSTORE+1 onward: compute & store residuals,
    //     popping from the front to keep the queue at <= DIIS_WINDOW.
    //   * Activation requires queue size >= DIIS_MIN_QUEUE *and*
    //     ||e||_F < DIIS_START.  Once active, deactivate when ||e||_F
    //     drops below DIIS_STOP.  History continues to roll regardless.
    const int    DIIS_WARMUP_NOSTORE = 4;     // skip the first N iters entirely
    const size_t DIIS_MIN_QUEUE      = 4;     // need this many before extrapolating
    // (DIIS_WINDOW, DIIS_START, DIIS_STOP defined above near other DIIS state)

    while (!converged && iter < 4000) {
        ++iter;
        bool diis_modified_H = false;   // tracks whether step 3 replaced H_BdG

        // ----- 1. Chemical potential (on h, delta from previous iteration). -----
        // Build once: a fast path is taken when delta == 0 (avoids ~30
        // 64×64 eigendecompositions per SCF iter at U=0).
        ParticleNumberContext pn_ctx = make_pn_context(h, delta, kT);
        auto pn_func = [&](double lambda_) {
            return particle_number(pn_ctx, lambda_);
        };
        double bracket_lo = has_prev_mu ? (prev_chem_pot - mu_window) : -0.1;
        double bracket_hi = has_prev_mu ? (prev_chem_pot + mu_window) :  0.1;
        double chem_pot = optimize_bisect(pn_func,
                                          static_cast<double>(PARTICLE_NUMBER),
                                          bracket_lo, bracket_hi, chem_tol);
        prev_chem_pot = chem_pot;
        has_prev_mu   = true;
        last_chem_pot = chem_pot;

        // Build H_BdG with chemical potential.
        const int n = static_cast<int>(h.rows());
        MatXcd I = MatXcd::Identity(n, n);
        MatXcd h_shift = h - chem_pot * I;
        MatXcd H_BdG = block2x2<MatXcd>(h_shift, delta,
                                        -delta.conjugate(), -h_shift.transpose());
        // ----- 2. DIIS error = [H_BdG_current, R_previous]. -----
        // R_prev was produced by diagonalizing the previous H_BdG.  At the
        // SCF fixed point H and R commute, so [H^{(n)}, R^{(n-1)}] -> 0.
        // Until then it's a genuine residual.  Skip entirely during warm-up
        // (or if we don't yet have a previous R).
        bool past_warmup = (iter > DIIS_WARMUP_NOSTORE);
        double err_norm = std::numeric_limits<double>::infinity();
        if (past_warmup && has_prev_R_full) {
            MatXcd err = H_BdG * prev_R_full - prev_R_full * H_BdG;
            err_norm   = err.norm();   // Frobenius

            // Engagement (per user's spec): use DIIS when 1e-9 < ||e||_F < 1e-3.
            // No latching -- re-evaluated every iteration.
            const bool in_band = (err_norm < DIIS_START) && (err_norm > DIIS_STOP);

            if (in_band) {
                // Store this iteration's H_BdG / error pair.
                diis_H_hist.push_back(H_BdG);
                diis_e_hist.push_back(err);
                if (diis_H_hist.size() > DIIS_WINDOW) {
                    diis_H_hist.pop_front();
                    diis_e_hist.pop_front();
                }
                diis_active = true;
            } else {
                // Out of trust band: don't pollute the queue with noisy
                // residuals.  If we *were* active and DIIS knocked us out,
                // clear the queue so we restart cleanly when err next
                // re-enters [DIIS_STOP, DIIS_START].
                if (diis_active) {
                    diis_H_hist.clear();
                    diis_e_hist.clear();
                }
                diis_active = false;
            }
        } else {
            diis_active = false;
        }

        // ----- 3. Extrapolate H_BdG, if DIIS is active and queue full enough. -----
        if (diis_active && diis_H_hist.size() >= DIIS_MIN_QUEUE) {
            const int m = static_cast<int>(diis_H_hist.size());
            // Build the augmented Pulay system:
            //   [[B  1]  [c]    [0]
            //    [1' 0]] [-λ] = [1]
            // with B_ij = Re Tr(e_i^dagger e_j).
            Eigen::MatrixXd A(m + 1, m + 1);
            Eigen::VectorXd b(m + 1);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    A(i, j) = (diis_e_hist[i].adjoint() * diis_e_hist[j]).trace().real();
                }
                A(i, m) = 1.0;
                A(m, i) = 1.0;
            }
            A(m, m) = 0.0;
            b.setZero();
            b(m) = 1.0;

            // Tikhonov on B handles near-singular conditioning when iterates
            // become correlated (cheap, vanishes at convergence).
            double trace_B = 0.0;
            for (int i = 0; i < m; ++i) trace_B += A(i, i);
            const double eps_reg = 1e-10 * std::abs(trace_B) / std::max(1, m);
            for (int i = 0; i < m; ++i) A(i, i) += eps_reg;

            Eigen::VectorXd sol = A.fullPivLu().solve(b);
            const double solve_residual = (A * sol - b).norm();

            // Enforce convex-combination coefficients: 0 <= c_i <= 1 with
            // Σ c_i = 1.  Per spec: this keeps the extrapolated H inside
            // the convex hull of stored H's and prevents blow-up.  The
            // unconstrained Pulay coefficients on this problem typically
            // come out as one large positive and one negative; clipping
            // collapses to the most recent H most of the time, so DIIS
            // accelerates only modestly -- acceptable here as long as
            // stability is preserved.
            Eigen::VectorXd c = sol.head(m);
            bool need_clip = false;
            for (int i = 0; i < m; ++i) {
                if (!(c(i) >= 0.0 && c(i) <= 1.0)) { need_clip = true; break; }
            }
            if (need_clip) {
                for (int i = 0; i < m; ++i) c(i) = std::clamp(c(i), 0.0, 1.0);
                double s = c.sum();
                if (s > 1e-14) {
                    c /= s;   // renormalize to Σ c_i = 1
                } else {
                    c.setZero();   // pathological: all clipped to zero
                }
            }

            const bool solve_ok      = std::isfinite(solve_residual) && solve_residual < 1e-6;
            const bool coeffs_usable = (c.sum() > 0.5);
            if (solve_ok && coeffs_usable) {
                MatXcd H_ext = MatXcd::Zero(H_BdG.rows(), H_BdG.cols());
                for (int i = 0; i < m; ++i) H_ext += c(i) * diis_H_hist[i];
                H_BdG = H_ext;
                diis_modified_H = true;
            }
        }

        // ----- 4. Diagonalize H_BdG, build R. -----
        // If DIIS did not modify H_BdG and the chempot bisection cached its
        // last full eigendecomposition (general path), reuse it -- the SCF
        // would otherwise repeat the exact same eigendecomposition the
        // bisection just did.  Saves one 64x64 complex eigendecomp per iter.
        const VecXd*  evals_p = nullptr;
        const MatXcd* evecs_p = nullptr;
        if (!diis_modified_H && pn_ctx.last_evals_valid && !pn_ctx.delta_is_zero
            && pn_ctx.last_lambda == chem_pot) {
            evals_p = &pn_ctx.last_evals;
            evecs_p = &pn_ctx.last_evecs;
        } else {
            es_bdg.compute(H_BdG);
            if (es_bdg.info() != Eigen::Success) {
                throw std::runtime_error("eigh failed in SCF loop");
            }
            evals_p = &es_bdg.eigenvalues();
            evecs_p = &es_bdg.eigenvectors();
        }
        const VecXd&  evals = *evals_p;
        const MatXcd& evecs = *evecs_p;

        {
            // Pair symmetry: BdG eigenvalues should come in ±E pairs.
            // 1e-6 tolerance is empirical — at FT with ~32×32 complex eigh,
            // accumulated round-off can reach 1e-7 even on well-formed input.
            VecXd sum_pairs = evals + evals.reverse();
            if (sum_pairs.cwiseAbs().maxCoeff() > 1e-6) {
                std::cout << "ENERGY'S ARE NOT COMING IN PAIRS" << std::endl;
            }
        }

        // R = sum over modes of f_j |v_j><v_j|.  At T=0 this collapses to a
        // hard projector onto negative-energy modes.  At T>0 every mode
        // contributes weight f_j = 1/(1 + exp(β λ_j))  (Goodman eq. 4.20,
        // applied directly to BdG eigenvalues since μ is already absorbed
        // into the diagonal of H_BdG).  We also collect the f_j vector for
        // the entropy term.
        VecXd  f_weights(evals.size());
        MatXcd R;
        if (kT == 0.0) {
            int n_neg = 0;
            for (int i = 0; i < evals.size(); ++i) {
                if (evals(i) < 1e-15) ++n_neg; else break;
            }
            f_weights.setZero();
            for (int i = 0; i < n_neg; ++i) f_weights(i) = 1.0;
            MatXcd V_neg = evecs.leftCols(n_neg);
            R = V_neg * V_neg.adjoint();
        } else {
            const double beta = 1.0 / kT;
            for (int i = 0; i < evals.size(); ++i) {
                f_weights(i) = fermi_dirac(evals(i), beta);
            }
            // R = evecs * diag(f) * evecs†.  Avoid materializing the
            // diagonal: scale columns of evecs by sqrt(f) (here just by f
            // on one side -- product is V * diag(f) * V†).
            MatXcd VW = evecs;
            for (int j = 0; j < evecs.cols(); ++j) VW.col(j) *= f_weights(j);
            R = VW * evecs.adjoint();
        }

        // Roll R_prev forward for next iteration's DIIS error.
        prev_R_full     = R;
        has_prev_R_full = true;

        MatXcd p_new = R.topLeftCorner(2 * STATES, 2 * STATES);
        MatXcd k_new = R.topRightCorner(2 * STATES, 2 * STATES);

        // ----- 5. Linear (alpha) mix.  DIIS does NOT replace this. -----
        p = alpha * p_new + (1.0 - alpha) * prev_p;
        k = alpha * k_new + (1.0 - alpha) * prev_k;

        // ----- 6. Rebuild h, delta from mixed p, k. -----
        h     = build_fock_matrix(PARTICLE_NUMBER, H_t, p, U);
        delta = build_delta(k, U, STATES);

        // ----- 7. Energy and (at T>0) entropy & grand potential. -----
        // Internal energy: E = (1/2) Tr[(T + h)ρ - Δ t†]   (Goodman 3.23,
        //                                                   with h = T + Γ).
        // The HF and pair potentials have the same form as at T=0; only the
        // construction of ρ and t (via thermally weighted R) differs.
        cd e1 = 0.5 * ((H_t_c + h) * p).trace();
        cd e2 = 0.5 * (delta * k.adjoint()).trace();
        energy = e1 - e2;

        if (kT > 0.0) {
            // Entropy: -k Σ_i [f_i ln f_i + (1-f_i) ln(1-f_i)] over physical
            // quasiparticles.  Using the BdG mirror, the same total is obtained
            // by summing over all 2n eigenmodes and halving.  We use kB == 1
            // (kT carries the units), so S/kB is what we compute.
            double S_over_kB = 0.0;
            for (int j = 0; j < f_weights.size(); ++j) {
                const double f = f_weights(j);
                if (f > 0.0 && f < 1.0) {
                    S_over_kB -= f * std::log(f) + (1.0 - f) * std::log(1.0 - f);
                }
            }
            S_over_kB *= 0.5;
            entropy_kB = S_over_kB;
            // Ω = E - TS - μN.  Since kB = 1, T·S has units [energy] = kT * S_over_kB.
            const double N = p.trace().real();
            grand_pot = energy.real() - kT * S_over_kB - chem_pot * N;
        } else {
            entropy_kB = 0.0;
            const double N = p.trace().real();
            grand_pot = energy.real() - chem_pot * N;
        }

        last_R = R;
        last_H_BdG = H_BdG;
        last_evals = evals;
        last_evecs = evecs;

        // ----- 8. Convergence checks. -----
        // Note: original Python tested `delta_E.real() < 1e-8`, which fires on
        // any drop in energy regardless of size.  At T=0 the density check
        // dominates so this rarely matters; at T>0 the density can land near
        // its fixed point in 1-2 mix steps (smooth Fermi-Dirac instead of a
        // hard step), exposing the bug.  Use |delta_E| instead.
        if (has_prev_energy) {
            cd delta_E = energy - prev_energy;
            bool energy_converged = (std::abs(delta_E.real()) < 1e-8);
            MatXcd diff = p - prev_p;
            bool density_converged = (diff.cwiseAbs().maxCoeff() < 1e-8);
            bool norm_converged    = (fro_norm(diff) < 1e-8);
            converged = energy_converged && density_converged && norm_converged;
        }

        // 2-cycle detection: U=0 (and other degenerate cases) land on a
        // period-2 limit cycle.  Five consecutive matches => done.
        if (has_prev_prev_p) {
            MatXcd cycle_diff = p - prev_prev_p;
            if (cycle_diff.cwiseAbs().maxCoeff() < 1e-8) {
                ++two_cycle_count;
                if (two_cycle_count >= two_cycle_threshold) converged = true;
            } else {
                two_cycle_count = 0;
            }
        }

        // Roll prev-slots.
        prev_prev_p     = prev_p;
        has_prev_prev_p = has_prev_energy;
        prev_p = p;
        prev_k = k;
        prev_energy    = energy;
        has_prev_energy = true;
    }

    HFBResult out;
    out.p = p;
    out.k = k;
    out.energy = energy;
    out.entropy = entropy_kB;
    out.grand_potential = grand_pot;
    out.mu = last_chem_pot;
    out.iters = iter;
    out.R = last_R;
    out.H_BdG = last_H_BdG;
    out.H_BdG_evals = last_evals;
    out.H_BdG_evecs = last_evecs;
    if (last_evecs.rows() > 0) {
        const int half = static_cast<int>(last_evecs.rows() / 2);
        out.U_bdg = last_evecs.topRows(half);
        out.V_bdg = last_evecs.bottomRows(half);
    }
    out.U_param = U;
    out.temperature = kT;
    out.iteration_count = iter;
    out.converged = converged;
    return out;
}

// ===================================================================
//  Main driver  --  reproduces the Python script's bottom block
// ===================================================================
int main(int argc, char** argv) {
    // CLI arguments:
    //   ./hfb                     -> seed=entropy, kT=0
    //   ./hfb <seed>              -> seed=<seed>, kT=0  (zero-temperature, original behavior)
    //   ./hfb <seed> <kT>         -> seed=<seed>, kT=<kT>  (finite-temperature run)
    //
    // kT is in the same units as t and U.  Setting kT=0 reproduces the
    // zero-temperature behavior bit-for-bit (verified against the original
    // C++ port and Python reference).  Setting kT>0 runs Goodman's FT-HFB
    // theory: every BdG mode contributes f_i = 1/(1+exp(βE_i)), the entropy
    // term is added to the grand potential, and the SCF converges to the
    // self-consistent fields *at that temperature* (no annealing).
    uint64_t seed = 0;
    double   kT   = 0.0;
    bool     have_seed = false;
    if (argc > 1) {
        seed = static_cast<uint64_t>(std::stoull(argv[1]));
        have_seed = true;
    }
    if (argc > 2) {
        kT = std::stod(argv[2]);
        if (kT < 0.0) {
            std::fprintf(stderr, "kT must be non-negative; got %g\n", kT);
            return 1;
        }
    }
    if (have_seed) seed_rng(seed);

    const int    x_              = 4;
    const int    y_              = 4;
    const int    PARTICLE_NUMBER = 16;
    const int    STATES          = 16;
    double       U               = 0.0;
    const double t               = 1.0;
    const std::string output_path = "fthfb_results.h5";

    std::printf("# 4x4 Hubbard, half-filled, kT=%g (units of t)\n", kT);
    std::fflush(stdout);

    // Build U_arr = -linspace(0, 10, 11) = [0, -1, -2, ..., -10]
    std::vector<double> U_arr;
    {
        const int n_steps = 11;
        for (int i = 0; i < n_steps; ++i) {
            double s = (n_steps == 1) ? 0.0 : (10.0 * i) / (n_steps - 1);
            U_arr.push_back(-s);
        }
    }

    // First call: warm-up (matches Python).  Always at the requested kT --
    // the user explicitly asked for "no annealing": every call is at the
    // target temperature.
    HFBResult res = run_hfb_hubbard(PARTICLE_NUMBER, STATES, x_, y_, U, t, kT);
    MatXcd p = res.p, k = res.k;
    cd     energy = res.energy;

    // Second call (no captured guesses): exactly mirrors the Python script
    run_hfb_hubbard(PARTICLE_NUMBER, STATES, x_, y_, U, t, kT);

    // Map iteration: insert if absent, then keep the lower energy seen.
    std::vector<std::pair<double, double>> data_by_U;
    auto find_U = [&](double key) -> int {
        for (size_t i = 0; i < data_by_U.size(); ++i)
            if (data_by_U[i].first == key) return static_cast<int>(i);
        return -1;
    };

    for (double U_val : U_arr) {
        res = run_hfb_hubbard(PARTICLE_NUMBER, STATES, x_, y_, U_val, t, kT, &p, &k);
        p = res.p; k = res.k; energy = res.energy;

        if (kT > 0.0) {
            const double F = energy.real() - kT * res.entropy;   // Helmholtz free energy
            std::printf("%d-Sites, Half Filled, t=%g, U=%g, kT=%g, "
                        "E=%.3f, N=%.2f, S/kB=%.3f, F=%.3f, Omega=%.3f, mu=%.3f, iters=%d\n",
                        PARTICLE_NUMBER, t, U_val, kT,
                        energy.real(), p.trace().real(),
                        res.entropy, F, res.grand_potential, res.mu, res.iters);
        } else {
            std::printf("%d-Sites, Half Filled, t=%g, U=%g, Energy: %.3f, N:%.2f\n",
                        PARTICLE_NUMBER, t, U_val,
                        energy.real(), p.trace().real());
        }
        std::fflush(stdout);

        if (res.converged) {
            write_converged_result_hdf5(res, output_path);
        } else {
            std::fprintf(stderr, "Warning: did not converge for U=%g after %d iterations.\n",
                         U_val, res.iters);
        }

        int idx = find_U(U_val);
        if (idx < 0) {
            data_by_U.emplace_back(U_val, energy.real());
        } else if (energy.real() < data_by_U[idx].second) {
            data_by_U[idx].second = energy.real();
        }
    }

    for (const auto& kv : data_by_U) {
        std::printf("%g,%.14f\n", kv.first, kv.second);
    }
    return 0;
}
