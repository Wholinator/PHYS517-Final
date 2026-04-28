// hfb_numeric_current.cpp
//
// C++ translation of hfb_numeric_current.py
//
// Hartree-Fock-Bogoliubov calculation on the 2D Hubbard model.
// Logic mirrors the Python implementation as closely as practical.
//
// Build:
//   g++ -O2 -std=c++17 -I/usr/include/eigen3 hfb_numeric_current.cpp -o hfb
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
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

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
//  particle_number  --  Tr(p) for a given chemical potential
// ===================================================================
double particle_number(const MatXcd& h, const MatXcd& delta, double lambda_) {
    const int n = static_cast<int>(h.rows());
    MatXcd I = MatXcd::Identity(n, n);
    MatXcd h_shift = h - lambda_ * I;
    MatXcd H_BdG = block2x2<MatXcd>(h_shift, delta,
                                    -delta.conjugate(), -h_shift.transpose());

    Eigen::SelfAdjointEigenSolver<MatXcd> es(H_BdG);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("eigh failed in particle_number");
    }
    VecXd  evals = es.eigenvalues();
    MatXcd evecs = es.eigenvectors();

    // P = diag( evals < 1e-15 )
    MatXcd P = MatXcd::Zero(2 * n, 2 * n);
    for (int i = 0; i < evals.size(); ++i) {
        if (evals(i) < 1e-15) P(i, i) = 1.0;
    }
    MatXcd R = evecs * P * evecs.adjoint();
    // Top-left n x n is p
    return R.topLeftCorner(n, n).trace().real();
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
struct HFBResult {
    MatXcd p;
    MatXcd k;
    cd     energy;
};

HFBResult run_hfb_hubbard(int PARTICLE_NUMBER, int STATES,
                          int x, int y, double U, double t,
                          const MatXcd* p_GUESS = nullptr,
                          const MatXcd* k_GUESS = nullptr) {
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
    MatXcd h     = build_fock_matrix(STATES, H_t, p, U);
    MatXcd delta = build_delta(k, U, STATES);

    // ----- History -----
    std::vector<cd>     energies;
    std::vector<MatXcd> focks{ h }, densities{ p }, deltas{ delta }, kappas{ k };
    std::vector<cd>     num;

    const double alpha = 0.2;
    int  iter      = 0;
    bool converged = false;
    cd   energy    = 0.0;

    // ----- SCF loop -----
    while (!converged && iter < 4000) {
        ++iter;

        // Bisect for chemical potential.  Capture h and delta by reference
        // so the lambda matches Python's functools.partial(particle_number, h, delta).
        auto pn_func = [&](double lambda_) {
            return particle_number(h, delta, lambda_);
        };
        double chem_pot = optimize_bisect(pn_func,
                                          static_cast<double>(PARTICLE_NUMBER),
                                          -0.1, 0.1, chem_tol);

        // Build H_BdG with chemical potential
        const int n = static_cast<int>(h.rows());
        MatXcd I = MatXcd::Identity(n, n);
        MatXcd h_shift = h - chem_pot * I;
        MatXcd H_BdG = block2x2<MatXcd>(h_shift, delta,
                                        -delta.conjugate(), -h_shift.transpose());

        // Diagonalize
        Eigen::SelfAdjointEigenSolver<MatXcd> es(H_BdG);
        if (es.info() != Eigen::Success) {
            throw std::runtime_error("eigh failed in SCF loop");
        }
        VecXd  evals = es.eigenvalues();
        MatXcd evecs = es.eigenvectors();

        // Pair-symmetry sanity check: evals should satisfy evals + reverse(evals) ≈ 0
        {
            VecXd sum_pairs = evals + evals.reverse();
            if (sum_pairs.cwiseAbs().maxCoeff() > 1e-7) {
                std::cout << "ENERGY'S ARE NOT COMING IN PAIRS" << std::endl;
            }
        }

        // P = diag(evals < 1e-15)
        MatXcd P = MatXcd::Zero(2 * n, 2 * n);
        for (int i = 0; i < evals.size(); ++i) {
            if (evals(i) < 1e-15) P(i, i) = 1.0;
        }
        MatXcd R = evecs * P * evecs.adjoint();

        MatXcd p_new = R.topLeftCorner(2 * STATES, 2 * STATES);
        MatXcd k_new = R.topRightCorner(2 * STATES, 2 * STATES);

        // Mix with previous (alpha * new + (1-alpha) * prev)
        p = alpha * p_new + (1.0 - alpha) * densities.back();
        k = alpha * k_new + (1.0 - alpha) * kappas.back();
        densities.push_back(p);
        kappas.push_back(k);

        h     = build_fock_matrix(PARTICLE_NUMBER, H_t, p, U);
        delta = build_delta(k, U, STATES);

        // E = 0.5 Tr((H_t + h) p) - 0.5 Tr(delta k.conj().T)
        MatXcd H_t_c = H_t.cast<cd>();
        cd e1 = 0.5 * ((H_t_c + h) * p).trace();
        cd e2 = 0.5 * (delta * k.adjoint()).trace();
        energy = e1 - e2;
        energies.push_back(energy);
        num.push_back(p.trace());

        // Convergence check
        if (energies.size() > 1) {
            cd delta_E = energies[energies.size() - 1] - energies[energies.size() - 2];
            // Python: (energies[-1] - energies[-2]) < 1e-8 -- on a complex value
            // numpy compares the real part for "<" of complex.  We do the same.
            bool energy_converged = (delta_E.real() < 1e-8);
            MatXcd diff = densities[densities.size() - 1] - densities[densities.size() - 2];
            bool density_converged = (diff.cwiseAbs().maxCoeff() < 1e-8);
            bool norm_converged    = (fro_norm(diff) < 1e-8);
            converged = energy_converged && density_converged && norm_converged;
        }
    }

    return {p, k, energy};
}

// ===================================================================
//  Main driver  --  reproduces the Python script's bottom block
// ===================================================================
int main(int argc, char** argv) {
    // Optional CLI seed for reproducibility:
    //   ./hfb           -> entropy-seeded (matches Python default behavior)
    //   ./hfb 42        -> deterministic
    if (argc > 1) {
        seed_rng(static_cast<uint64_t>(std::stoull(argv[1])));
    }

    const int    x_              = 4;
    const int    y_              = 4;
    const int    PARTICLE_NUMBER = 16;
    const int    STATES          = 16;
    double       U               = 0.0;
    const double t               = 1.0;

    // Build U_arr = -linspace(0, 10, 11) = [0, -1, -2, ..., -10]
    std::vector<double> U_arr;
    {
        const int n_steps = 11;
        for (int i = 0; i < n_steps; ++i) {
            double s = (n_steps == 1) ? 0.0 : (10.0 * i) / (n_steps - 1);
            U_arr.push_back(-s);
        }
    }

    // First call: warm-up (matches Python)
    HFBResult res = run_hfb_hubbard(PARTICLE_NUMBER, STATES, x_, y_, U, t);
    MatXcd p = res.p, k = res.k;
    cd     energy = res.energy;

    // Second call (no captured guesses): exactly mirrors the Python script
    run_hfb_hubbard(PARTICLE_NUMBER, STATES, x_, y_, U, t);

    // Map iteration: insert if absent, then keep the lower energy seen.
    std::vector<std::pair<double, double>> data_by_U;
    auto find_U = [&](double key) -> int {
        for (size_t i = 0; i < data_by_U.size(); ++i)
            if (data_by_U[i].first == key) return static_cast<int>(i);
        return -1;
    };

    for (double U_val : U_arr) {
        res = run_hfb_hubbard(PARTICLE_NUMBER, STATES, x_, y_, U_val, t, &p, &k);
        p = res.p; k = res.k; energy = res.energy;

        std::printf("%d-Sites, Half Filled, t=%g, U=%g, Energy: %.3f, N:%.2f\n",
                    PARTICLE_NUMBER, t, U_val,
                    energy.real(), p.trace().real());

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