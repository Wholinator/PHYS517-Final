#pragma once

#include <Eigen/Dense>

#include <complex>
#include <string>

struct HFBResult {
    Eigen::MatrixXcd p;
    Eigen::MatrixXcd k;
    std::complex<double> energy;

    double entropy = 0.0;
    double grand_potential = 0.0;
    double mu = 0.0;
    int iters = 0;

    Eigen::MatrixXcd R;
    Eigen::MatrixXcd H_BdG;
    Eigen::VectorXd  H_BdG_evals;
    Eigen::MatrixXcd H_BdG_evecs;
    Eigen::MatrixXcd U_bdg;
    Eigen::MatrixXcd V_bdg;

    double U_param = 0.0;
    double temperature = 0.0;
    int iteration_count = 0;
    bool converged = false;
};

void write_converged_result_hdf5(const HFBResult& result, const std::string& path);
