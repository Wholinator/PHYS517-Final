#include <armadillo>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using cd = std::complex<double>;
using Mat = arma::cx_mat;
using Vec = arma::cx_vec;
using RealVec = arma::vec;

namespace {

std::mt19937_64& global_rng() {
    static std::mt19937_64 rng{std::random_device{}()};
    return rng;
}

Mat as_square_matrix(const Mat& a) {
    if (a.n_rows != a.n_cols) {
        throw std::invalid_argument("Matrix must be 2D and square.");
    }
    return a;
}

Mat eye_mat(arma::uword n) {
    return arma::eye<Mat>(n, n);
}

Mat zeros_mat(arma::uword rows, arma::uword cols) {
    return arma::zeros<Mat>(rows, cols);
}

Mat ones_mat(arma::uword rows, arma::uword cols) {
    return arma::ones<Mat>(rows, cols);
}

Mat diag_from_real(const RealVec& d) {
    return arma::diagmat(arma::conv_to<Vec>::from(d));
}

Mat block_diag(const Mat& a, const Mat& b) {
    Mat out(a.n_rows + b.n_rows, a.n_cols + b.n_cols, arma::fill::zeros);
    out.submat(0, 0, a.n_rows - 1, a.n_cols - 1) = a;
    out.submat(a.n_rows, a.n_cols, a.n_rows + b.n_rows - 1, a.n_cols + b.n_cols - 1) = b;
    return out;
}

Mat block2x2(const Mat& a, const Mat& b, const Mat& c, const Mat& d) {
    if (a.n_rows != b.n_rows || c.n_rows != d.n_rows ||
        a.n_cols != c.n_cols || b.n_cols != d.n_cols) {
        throw std::invalid_argument("Incompatible block dimensions.");
    }
    return arma::join_cols(arma::join_rows(a, b), arma::join_rows(c, d));
}

Mat real_normal_mat(arma::uword rows, arma::uword cols, std::mt19937_64& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    Mat out(rows, cols);
    for (arma::uword i = 0; i < rows; ++i) {
        for (arma::uword j = 0; j < cols; ++j) {
            out(i, j) = dist(rng);
        }
    }
    return out;
}

Mat real_normal_mat(arma::uword rows, arma::uword cols) {
    return real_normal_mat(rows, cols, global_rng());
}

Mat random_uniform_diag(arma::uword n) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Mat out(n, n, arma::fill::zeros);
    for (arma::uword i = 0; i < n; ++i) {
        out(i, i) = dist(global_rng());
    }
    return out;
}

RealVec linspace(double lo, double hi, arma::uword n) {
    return arma::linspace<RealVec>(lo, hi, n);
}

Vec roll(const Vec& v, int shift) {
    const arma::uword n = v.n_elem;
    if (n == 0) return v;

    int s = shift % static_cast<int>(n);
    if (s < 0) s += static_cast<int>(n);

    Vec out(n);
    for (arma::uword i = 0; i < n; ++i) {
        out((i + static_cast<arma::uword>(s)) % n) = v(i);
    }
    return out;
}

Mat purify_evals(const std::vector<double>& allowed_evals, const Mat& mat) {
    if (allowed_evals.empty()) {
        throw std::invalid_argument("allowed_evals cannot be empty.");
    }

    RealVec evals;
    Mat evecs;
    arma::eig_sym(evals, evecs, as_square_matrix(mat));

    RealVec purified(evals.n_elem);
    for (arma::uword i = 0; i < evals.n_elem; ++i) {
        const double x = evals(i);
        auto nearest = std::min_element(
            allowed_evals.begin(), allowed_evals.end(),
            [x](double a, double b) { return std::abs(x - a) < std::abs(x - b); });
        purified(i) = *nearest;
    }

    return evecs * diag_from_real(purified) * evecs.t();
}

Mat purify_evals(const std::function<bool(double)>& is_allowed,
                 const Mat& mat,
                 arma::uword grid_size = 2000) {
    RealVec evals;
    Mat evecs;
    arma::eig_sym(evals, evecs, as_square_matrix(mat));

    const double lo = evals.min() - 1.0;
    const double hi = evals.max() + 1.0;
    const RealVec grid = linspace(lo, hi, grid_size);

    std::vector<double> allowed_grid;
    allowed_grid.reserve(grid_size);
    for (arma::uword i = 0; i < grid.n_elem; ++i) {
        if (is_allowed(grid(i))) allowed_grid.push_back(grid(i));
    }

    if (allowed_grid.empty()) {
        throw std::runtime_error("No allowed eigenvalues found in search interval.");
    }

    RealVec purified(evals.n_elem);
    for (arma::uword i = 0; i < evals.n_elem; ++i) {
        const double x = evals(i);
        if (is_allowed(x)) {
            purified(i) = x;
        } else {
            auto nearest = std::min_element(
                allowed_grid.begin(), allowed_grid.end(),
                [x](double a, double b) { return std::abs(x - a) < std::abs(x - b); });
            purified(i) = *nearest;
        }
    }

    return evecs * diag_from_real(purified) * evecs.t();
}

Mat make_positive_semidefinite(const Mat& a) {
    return purify_evals([](double x) { return x >= 0.0; }, as_square_matrix(a));
}

bool is_positive_semidefinite(const Mat& a, double atol = 1e-10, double rtol = 1e-10) {
    (void)atol;
    (void)rtol;

    RealVec evals;
    Mat evecs;
    arma::eig_sym(evals, evecs, as_square_matrix(a));
    return arma::all(evals >= 0.0);
}

Mat init_rho(int block_dim,
             double diagval,
             const std::string& restriction = "R",
             bool diag_rn = false,
             bool offdiag_rn = false,
             bool spin_mix = false,
             bool anti = true,
             double per_tol = 1e-3) {
    Mat rho;

    if (diag_rn) {
        const int dim = block_dim;
        Mat p1 = random_uniform_diag(dim);
        Mat p2 = random_uniform_diag(dim);
        if (restriction == "R") p2 = p1;

        Mat mat = block_diag(p1, p2) * per_tol;
        const double mult_factor = diagval / std::real(arma::trace(mat));

        Mat mult_factor_mat = ones_mat(2 * dim, 2 * dim);
        mult_factor_mat.diag() += (mult_factor - 1.0);
        rho = mat % mult_factor_mat;
    } else if (!anti) {
        Mat p1 = (diagval / static_cast<double>(block_dim)) * eye_mat(block_dim);
        Mat p2 = zeros_mat(block_dim, block_dim);
        rho = block_diag(p1, p2);
    } else {
        RealVec down_diag(block_dim);
        for (int i = 0; i < block_dim; ++i) {
            down_diag(i) = static_cast<double>(i % 2);
        }

        RealVec up_diag;
        if (restriction == "R") {
            up_diag = down_diag;
        } else {
            up_diag = arma::ones<RealVec>(block_dim) - down_diag;
        }

        const double mult = diagval / arma::sum(up_diag + down_diag);
        Mat p1 = diag_from_real(mult * up_diag);
        Mat p2 = diag_from_real(mult * down_diag);
        rho = block_diag(p1, p2);
    }

    if (offdiag_rn) {
        Mat offdiag1 = per_tol * real_normal_mat(block_dim, block_dim);
        Mat offdiag2 = (restriction != "R")
            ? per_tol * real_normal_mat(block_dim, block_dim)
            : offdiag1;

        Mat mask = ones_mat(block_dim, block_dim) - eye_mat(block_dim);
        offdiag1 %= mask;
        offdiag2 %= mask;

        // This preserves the original NumPy operation exactly:
        //     offdiag = 0.5 * (offdiag * offdiag.T)
        // where * is elementwise multiplication and .T is a non-conjugating transpose.
        offdiag1 = 0.5 * (offdiag1 % offdiag1.st());
        offdiag2 = 0.5 * (offdiag2 % offdiag2.st());

        rho += block_diag(offdiag1, offdiag2);
    }

    if (spin_mix && restriction == "G") {
        Mat sp1 = per_tol * real_normal_mat(block_dim, block_dim);
        Mat sp2 = sp1.st();
        Mat z = zeros_mat(block_dim, block_dim);
        rho += block2x2(z, sp1, sp2, z);
    }

    if (!is_positive_semidefinite(rho)) {
        rho = make_positive_semidefinite(rho);
        if ((std::real(arma::trace(rho)) - diagval) > 1e-6) {
            Vec d = (diagval / std::real(arma::trace(rho))) * rho.diag();
            Mat offdiag_mask = ones_mat(rho.n_rows, rho.n_cols) - eye_mat(rho.n_rows);
            rho = (rho % offdiag_mask) + arma::diagmat(d);
        }
    }

    return rho;
}

std::pair<double, double> validate_bounds(const std::function<double(double)>& func,
                                          double goal,
                                          std::pair<double, double> r1,
                                          bool increasing,
                                          int max_iter = 50) {
    bool valid_bounds = false;
    double guess_low = r1.first;
    double guess_high = r1.second;

    double high = func(guess_high);
    double low = func(guess_low);

    if (((low - high) > 1e-7 && increasing) ||
        ((high - low) > 1e-7 && !increasing)) {
        std::cout << "INCREASINGNESS IS INCORRECT\n";
    }

    int iter = 0;
    while (!valid_bounds) {
        ++iter;
        if ((increasing && high < goal) || (!increasing && high > goal)) {
            const double temp = guess_high;
            guess_high = 3.0 * guess_high - 2.0 * guess_low;
            guess_low = temp;
            low = high;
            high = func(guess_high);
        } else if ((increasing && low > goal) || (!increasing && low < goal)) {
            const double temp = guess_low;
            guess_low = 3.0 * guess_low - 2.0 * guess_high;
            guess_high = temp;
            high = low;
            low = func(guess_low);
        } else if (high == low) {
            guess_high *= 4.0;
            guess_low *= 4.0;
            high = func(guess_high);
            low = func(guess_low);
        } else {
            valid_bounds = true;
        }

        if (iter > max_iter) {
            return {guess_low, guess_high};
        }
    }

    return {guess_low, guess_high};
}

double optimize_bisect(const std::function<double(double)>& func,
                       double goal,
                       std::pair<double, double> guess_range = {-0.1, 0.1},
                       double tol = 1e-12) {
    bool converged = false;
    const bool increasing = func(100.0) > func(-100.0);

    auto [guess_low, guess_high] = validate_bounds(func, goal, guess_range, increasing);

    int iter = 0;
    double guess = 0.5 * (guess_high + guess_low);
    while (!converged && iter <= 2000) {
        ++iter;
        guess = 0.5 * (guess_high + guess_low);
        const double value = func(guess);

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

Mat nearest_neighbor_2d(int rows, int columns, bool pbc = false) {
    const int block_size = columns;
    const int block_num = rows;

    Mat zero_block = zeros_mat(block_size, block_size);
    Mat diag_block = eye_mat(block_size);

    Mat offdiag_block = zeros_mat(block_size, block_size);
    for (int i = 0; i < block_size - 1; ++i) {
        offdiag_block(i + 1, i) = 1.0;
        offdiag_block(i, i + 1) = 1.0;
    }

    Mat corner_block = zero_block;
    corner_block(0, block_size - 1) = 1.0;
    corner_block(block_size - 1, 0) = 1.0;

    Mat out(block_num * block_size, block_num * block_size, arma::fill::zeros);
    for (int row = 0; row < block_num; ++row) {
        for (int col = 0; col < block_num; ++col) {
            Mat block = zero_block;
            if (row == col) {
                block += offdiag_block;
            } else if (std::abs(row - col) == 1) {
                block += diag_block;
            }

            if (pbc) {
                if (row == col) {
                    block += corner_block;
                } else if (std::abs(row - col) == block_num - 1) {
                    block += diag_block;
                }
            }

            out.submat(row * block_size,
                       col * block_size,
                       (row + 1) * block_size - 1,
                       (col + 1) * block_size - 1) = block;
        }
    }

    return out;
}

Mat build_t_mat_hubbard(int x, int y, double t, bool pbc) {
    Mat lattice = nearest_neighbor_2d(x, y, pbc);
    return -t * arma::kron(eye_mat(2), lattice);
}

Mat build_fock_matrix(int roll_by, const Mat& t_mat, const Mat& p, double U_val) {
    Vec rolled_diag = roll(p.diag(), roll_by);
    Mat h_u = U_val * arma::diagmat(rolled_diag);
    return t_mat + h_u;
}

Mat init_hfb_kappa(int states, double eps = 1.0, std::optional<std::uint64_t> seed = std::nullopt) {
    std::mt19937_64 local_rng;
    std::mt19937_64* rng = &global_rng();
    if (seed.has_value()) {
        local_rng.seed(*seed);
        rng = &local_rng;
    }

    const int n = 2 * states;
    Mat rnd = real_normal_mat(n, n, *rng);
    return eps * (rnd - rnd.st());
}

Mat build_delta(const Mat& kappa, double U_val, int nstates) {
    const int n = 2 * nstates;
    Mat hubbard_k_map = zeros_mat(n, n);
    for (int i = 0; i < nstates; ++i) {
        hubbard_k_map(i, i + nstates) = 1.0;
        hubbard_k_map(i + nstates, i) = 1.0;
    }

    return 2.0 * U_val * (kappa % hubbard_k_map).st();
}

double particle_number(const Mat& h, const Mat& delta, double lambda_) {
    const arma::uword n = h.n_rows;
    Mat shifted_h = h - lambda_ * eye_mat(n);
    Mat h_bdg = block2x2(shifted_h, delta, -arma::conj(delta), -shifted_h.st());

    RealVec evals;
    Mat evecs;
    arma::eig_sym(evals, evecs, h_bdg);

    RealVec occ(evals.n_elem, arma::fill::zeros);
    for (arma::uword i = 0; i < evals.n_elem; ++i) {
        occ(i) = (evals(i) < 1e-15) ? 1.0 : 0.0;
    }

    Mat r = evecs * diag_from_real(occ) * evecs.t();
    Mat p = r.submat(0, 0, n - 1, n - 1);
    return std::real(arma::trace(p));
}

struct HfbResult {
    Mat p;
    Mat k;
    double energy;
};

HfbResult run_hfb_hubbard(double particle_number_goal,
                          int states,
                          int x,
                          int y,
                          double U,
                          double t,
                          const Mat& p_guess = Mat(),
                          const Mat& k_guess = Mat()) {
    const double conv_tol = 1e-8;
    const double chem_tol = 1e-12;

    Mat p;
    if (p_guess.n_elem != 0) {
        p = p_guess;
    } else {
        // Preserves the Python call:
        //     init_rho(PARTICLE_NUMBER, PARTICLE_NUMBER, "G", anti=True, offdiag_rn=True, spin_mix=True)
        // If you later run away from half-filling, the first argument should probably be `states`.
        p = init_rho(static_cast<int>(particle_number_goal),
                     particle_number_goal,
                     "G",
                     false,
                     true,
                     true,
                     true);
    }

    Mat k;
    if (k_guess.n_elem != 0) {
        k = k_guess;
    } else {
        k = init_hfb_kappa(states);
    }

    Mat h_t = build_t_mat_hubbard(x, y, t, true);
    Mat h = build_fock_matrix(states, h_t, p, U);
    Mat delta = build_delta(k, U, states);

    std::vector<double> energies;
    std::vector<Mat> densities{p};
    std::vector<Mat> kappas{k};

    const double alpha = 0.2;
    int iter = 0;
    bool converged = false;
    double energy = std::numeric_limits<double>::quiet_NaN();

    while (!converged && iter < 4000) {
        ++iter;

        auto n_of_lambda = [&](double lambda) {
            return particle_number(h, delta, lambda);
        };

        const double chem_pot = optimize_bisect(n_of_lambda,
                                                particle_number_goal,
                                                {-0.1, 0.1},
                                                chem_tol);

        const arma::uword n = h.n_rows;
        Mat shifted_h = h - chem_pot * eye_mat(n);
        Mat h_bdg = block2x2(shifted_h, delta, -arma::conj(delta), -shifted_h.st());

        RealVec evals;
        Mat evecs;
        arma::eig_sym(evals, evecs, h_bdg);

        bool bad_pairs = false;
        for (arma::uword i = 0; i < evals.n_elem; ++i) {
            if (std::abs(evals(i) + evals(evals.n_elem - 1 - i)) > 1e-7) {
                bad_pairs = true;
                break;
            }
        }
        if (bad_pairs) {
            std::cout << "ENERGY'S ARE NOT COMING IN PAIRS\n";
        }

        RealVec occ(evals.n_elem, arma::fill::zeros);
        for (arma::uword i = 0; i < evals.n_elem; ++i) {
            occ(i) = (evals(i) < 1e-15) ? 1.0 : 0.0;
        }

        Mat r = evecs * diag_from_real(occ) * evecs.t();
        p = r.submat(0, 0, 2 * states - 1, 2 * states - 1);
        k = r.submat(0, 2 * states, 2 * states - 1, 4 * states - 1);

        p = alpha * p + (1.0 - alpha) * densities.back();
        k = alpha * k + (1.0 - alpha) * kappas.back();
        densities.push_back(p);
        kappas.push_back(k);

        // Preserves the Python update:
        //     h = build_fock_matrix(PARTICLE_NUMBER, H_t, p, U)
        h = build_fock_matrix(static_cast<int>(particle_number_goal), h_t, p, U);
        delta = build_delta(k, U, states);

        cd e_complex = 0.5 * arma::trace((h_t + h) * p)
                     - 0.5 * arma::trace(delta * k.t());
        energy = std::real(e_complex);
        energies.push_back(energy);

        if (energies.size() > 1) {
            const bool energy_converged = (energies.back() - energies[energies.size() - 2]) < conv_tol;
            const Mat density_diff = densities.back() - densities[densities.size() - 2];
            const bool density_converged = arma::all(arma::vectorise(arma::abs(density_diff) < conv_tol));
            const bool norm_converged = arma::norm(density_diff, "fro") < conv_tol;
            converged = energy_converged && density_converged && norm_converged;
        }
    }

    return {p, k, energy};
}

} // namespace

int main() {
    const int x = 4;
    const int y = 4;
    const double PARTICLE_NUMBER = 16.0;
    const int STATES = 16;
    const double t = 1.0;
    double U = 0.0;

    std::vector<double> steps;
    for (int i = 0; i <= 10; ++i) steps.push_back(static_cast<double>(i));

    std::vector<double> U_arr;
    for (double s : steps) U_arr.push_back(-s);

    std::vector<std::pair<double, double>> data_by_U;

    HfbResult result = run_hfb_hubbard(PARTICLE_NUMBER, STATES, x, y, U, t);
    run_hfb_hubbard(PARTICLE_NUMBER, STATES, x, y, U, t);

    Mat p = result.p;
    Mat k = result.k;

    for (double u_value : U_arr) {
        HfbResult next = run_hfb_hubbard(PARTICLE_NUMBER, STATES, x, y, u_value, t, p, k);
        p = next.p;
        k = next.k;
        const double energy = next.energy;
        const double n_trace = std::real(arma::trace(p));

        std::cout << std::fixed
                  << std::setprecision(0) << PARTICLE_NUMBER
                  << "-Sites, Half Filled, t=" << std::setprecision(0) << t
                  << ", U=" << std::setprecision(0) << u_value
                  << ", Energy: " << std::setprecision(3) << energy
                  << ", N:" << std::setprecision(2) << n_trace << "\n";

        auto it = std::find_if(
            data_by_U.begin(), data_by_U.end(),
            [u_value](const auto& kv) { return kv.first == u_value; });
        if (it == data_by_U.end()) {
            data_by_U.emplace_back(u_value, energy);
        } else if (energy < it->second) {
            it->second = energy;
        }
    }

    std::cout << std::setprecision(12);
    for (const auto& [key, value] : data_by_U) {
        std::cout << key << "," << value << "\n";
    }

    return 0;
}
