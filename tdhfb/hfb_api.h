// hfb_api.h
//
// Public interface to functions defined in hfb_numeric_current.cpp,
// so that tdhfb.cpp (and tdhfb_main.cpp) can reuse them without
// duplicating code.

#pragma once

#include "data_output.h"   // pulls Eigen, std::complex, MatXd, MatXcd, HFBResult
#include <Eigen/Dense>
#include <complex>
#include <functional>

// ---- Random-number infrastructure ---------------------------------
void seed_rng(uint64_t s);
double randn();
double rand_uniform();
MatXd randn_matrix(int rows, int cols);

// ---- Lattice and mean-field builders ------------------------------
MatXd nearest_neighbor_2d(int rows, int columns, bool pbc = false);
MatXd build_t_mat_hubbard(int x, int y, double t, bool pbc);
MatXcd build_fock_matrix(int roll_by, const MatXd& t_mat, const MatXcd& p, double U_val);
MatXcd build_delta(const MatXcd& kappa, double U_val, int nstates);

// ---- Block helpers ------------------------------------------------
template <typename M>
M block_diag2(const M& A, const M& B) {
    M out = M::Zero(A.rows() + B.rows(), A.cols() + B.cols());
    out.topLeftCorner(A.rows(), A.cols())     = A;
    out.bottomRightCorner(B.rows(), B.cols()) = B;
    return out;
}

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

// ---- HFB driver ---------------------------------------------------
HFBResult run_hfb_hubbard(int PARTICLE_NUMBER, int STATES,
                          int x, int y, double U, double t,
                          const MatXcd* p_GUESS = nullptr,
                          const MatXcd* k_GUESS = nullptr,
                          double temperature = 0.0);
