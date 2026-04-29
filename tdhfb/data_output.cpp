// data_output.cpp
//
// HDF5 writers for HFBResult and TDHFBResult using the HDF5 C API.

#include "data_output.h"

#include <hdf5.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---------------- error helpers ----------------
[[noreturn]] void h5_die(const std::string& what) {
    throw std::runtime_error("HDF5 error: " + what);
}

// ---------------- file open/create ----------------
hid_t open_or_create_file(const std::string& path) {
    H5E_BEGIN_TRY {
        hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (f >= 0) return f;
    } H5E_END_TRY;
    hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0) h5_die("could not create " + path);
    return f;
}

// ---------------- group helpers ----------------
hid_t open_or_create_group(hid_t parent, const std::string& name) {
    H5E_BEGIN_TRY {
        hid_t g = H5Gopen2(parent, name.c_str(), H5P_DEFAULT);
        if (g >= 0) return g;
    } H5E_END_TRY;
    hid_t g = H5Gcreate2(parent, name.c_str(),
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (g < 0) h5_die("could not create group " + name);
    return g;
}

// Delete a dataset/group if it exists, so we can rewrite it.
void unlink_if_exists(hid_t loc, const std::string& name) {
    H5E_BEGIN_TRY {
        H5Ldelete(loc, name.c_str(), H5P_DEFAULT);
    } H5E_END_TRY;
}

// ---------------- scalar attribute writers ----------------
void write_attr_double(hid_t loc, const char* name, double v) {
    hid_t s   = H5Screate(H5S_SCALAR);
    H5E_BEGIN_TRY { H5Adelete(loc, name); } H5E_END_TRY;
    hid_t a   = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, s,
                           H5P_DEFAULT, H5P_DEFAULT);
    if (a < 0) h5_die(std::string("attr ") + name);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &v);
    H5Aclose(a); H5Sclose(s);
}

void write_attr_int(hid_t loc, const char* name, int v) {
    hid_t s = H5Screate(H5S_SCALAR);
    H5E_BEGIN_TRY { H5Adelete(loc, name); } H5E_END_TRY;
    hid_t a = H5Acreate2(loc, name, H5T_NATIVE_INT, s,
                         H5P_DEFAULT, H5P_DEFAULT);
    if (a < 0) h5_die(std::string("attr ") + name);
    H5Awrite(a, H5T_NATIVE_INT, &v);
    H5Aclose(a); H5Sclose(s);
}

void write_attr_string(hid_t loc, const char* name, const std::string& v) {
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, v.size() == 0 ? 1 : v.size());
    H5Tset_strpad(t, H5T_STR_NULLTERM);
    hid_t s = H5Screate(H5S_SCALAR);
    H5E_BEGIN_TRY { H5Adelete(loc, name); } H5E_END_TRY;
    hid_t a = H5Acreate2(loc, name, t, s, H5P_DEFAULT, H5P_DEFAULT);
    if (a < 0) h5_die(std::string("attr ") + name);
    const char* buf = v.empty() ? "" : v.c_str();
    H5Awrite(a, t, buf);
    H5Aclose(a); H5Sclose(s); H5Tclose(t);
}

// ---------------- 1D dataset writers ----------------
void write_dset_1d_double(hid_t loc, const std::string& name,
                          const double* data, size_t n) {
    unlink_if_exists(loc, name);
    hsize_t dims[1] = { n };
    hid_t s = H5Screate_simple(1, dims, nullptr);
    hid_t d = H5Dcreate2(loc, name.c_str(), H5T_NATIVE_DOUBLE, s,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (d < 0) h5_die("create dset " + name);
    if (n > 0)
        H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(d); H5Sclose(s);
}

void write_dset_1d_double(hid_t loc, const std::string& name,
                          const std::vector<double>& v) {
    write_dset_1d_double(loc, name, v.data(), v.size());
}

// ---------------- 2D dataset writers (real / complex) ----------------
//
// Real matrices are written row-major as a 2D dataset.
// Complex matrices are written as a 3D dataset of shape (rows, cols, 2),
// with [...,0] = real and [...,1] = imag.  This is widely-supported and
// trivial to read in numpy: np.asarray(f["..."]).view(np.complex128) etc.

void write_dset_2d_real(hid_t loc, const std::string& name, const MatXd& M) {
    unlink_if_exists(loc, name);
    hsize_t dims[2] = { (hsize_t)M.rows(), (hsize_t)M.cols() };
    hid_t s = H5Screate_simple(2, dims, nullptr);
    hid_t d = H5Dcreate2(loc, name.c_str(), H5T_NATIVE_DOUBLE, s,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (d < 0) h5_die("create dset " + name);
    // Eigen is column-major by default; copy to row-major buffer.
    std::vector<double> buf((size_t)M.rows() * (size_t)M.cols());
    for (Eigen::Index i = 0; i < M.rows(); ++i)
        for (Eigen::Index j = 0; j < M.cols(); ++j)
            buf[(size_t)i * (size_t)M.cols() + (size_t)j] = M(i, j);
    H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(d); H5Sclose(s);
}

void write_dset_2d_complex(hid_t loc, const std::string& name, const MatXcd& M) {
    unlink_if_exists(loc, name);
    hsize_t dims[3] = { (hsize_t)M.rows(), (hsize_t)M.cols(), 2 };
    hid_t s = H5Screate_simple(3, dims, nullptr);
    hid_t d = H5Dcreate2(loc, name.c_str(), H5T_NATIVE_DOUBLE, s,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (d < 0) h5_die("create dset " + name);
    std::vector<double> buf((size_t)M.rows() * (size_t)M.cols() * 2);
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        for (Eigen::Index j = 0; j < M.cols(); ++j) {
            size_t idx = ((size_t)i * (size_t)M.cols() + (size_t)j) * 2;
            buf[idx]     = M(i, j).real();
            buf[idx + 1] = M(i, j).imag();
        }
    }
    H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(d); H5Sclose(s);
}

// ---------------- 4D dataset writer (stack of complex matrices) ----------------
// Writes vector<MatXcd> as a dataset of shape (n_snap, rows, cols, 2),
// where dim[3] carries the real (0) and imaginary (1) parts.
void write_dset_4d_complex(hid_t loc, const std::string& name,
                           const std::vector<MatXcd>& mats) {
    if (mats.empty()) return;
    unlink_if_exists(loc, name);
    const hsize_t n_snap = mats.size();
    const hsize_t rows   = (hsize_t)mats[0].rows();
    const hsize_t cols   = (hsize_t)mats[0].cols();
    hsize_t dims[4] = {n_snap, rows, cols, 2};
    hid_t s = H5Screate_simple(4, dims, nullptr);
    hid_t d = H5Dcreate2(loc, name.c_str(), H5T_NATIVE_DOUBLE, s,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (d < 0) h5_die("create dset " + name);
    std::vector<double> buf(n_snap * rows * cols * 2);
    for (hsize_t si = 0; si < n_snap; ++si) {
        for (hsize_t i = 0; i < rows; ++i) {
            for (hsize_t j = 0; j < cols; ++j) {
                size_t idx = ((si * rows + i) * cols + j) * 2;
                buf[idx]     = mats[si]((Eigen::Index)i, (Eigen::Index)j).real();
                buf[idx + 1] = mats[si]((Eigen::Index)i, (Eigen::Index)j).imag();
            }
        }
    }
    H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(d); H5Sclose(s);
}

// ---------------- timestamp helper ----------------
std::string utc_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return std::string(buf);
}

// Make a group name that is HDF5-safe (no slashes etc.) and ordering-friendly.
std::string group_name_from_U(double U) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "U_%+010.4f", U);
    // replace + with p, - with m, . with _ to be safe
    std::string s(buf);
    for (auto& c : s) {
        if (c == '+') c = 'p';
        else if (c == '-') c = 'm';
        else if (c == '.') c = '_';
    }
    return s;
}

}  // namespace

// ============================================================
//  Public:  write_converged_result_hdf5
// ============================================================
void write_converged_result_hdf5(const HFBResult& res, const std::string& path) {
    hid_t f = open_or_create_file(path);

    // Top-level attrs (only set if not already set; safe to overwrite anyway)
    write_attr_string(f, "kind", std::string("hfb_ground_state"));
    write_attr_string(f, "created_utc", utc_timestamp());
    write_attr_int   (f, "x_sites", res.x_sites);
    write_attr_int   (f, "y_sites", res.y_sites);
    write_attr_int   (f, "n_states", res.n_states);
    write_attr_int   (f, "particle_number", res.particle_number);
    write_attr_double(f, "t_hop", res.t_hop);

    // Per-U group
    hid_t hfb_root = open_or_create_group(f, "hfb");
    hid_t g = open_or_create_group(hfb_root, group_name_from_U(res.U_param));

    write_attr_double(g, "U", res.U_param);
    write_attr_double(g, "temperature", res.temperature);
    write_attr_double(g, "mu", res.mu);
    write_attr_double(g, "energy_real", res.energy.real());
    write_attr_double(g, "energy_imag", res.energy.imag());
    write_attr_int   (g, "iteration_count", res.iteration_count);
    write_attr_int   (g, "converged", res.converged ? 1 : 0);

    write_dset_2d_complex(g, "p",         res.p);
    write_dset_2d_complex(g, "k",         res.k);
    write_dset_2d_complex(g, "R",         res.R);
    write_dset_2d_complex(g, "H_BdG",     res.H_BdG);
    write_dset_2d_complex(g, "H_BdG_evecs", res.H_BdG_evecs);
    write_dset_1d_double (g, "H_BdG_evals",
                          res.H_BdG_evals.data(), (size_t)res.H_BdG_evals.size());

    H5Gclose(g);
    H5Gclose(hfb_root);
    H5Fclose(f);
}

// ============================================================
//  Public:  write_tdhfb_result_hdf5
// ============================================================
void write_tdhfb_result_hdf5(const TDHFBResult& res, const std::string& path) {
    hid_t f = open_or_create_file(path);

    write_attr_string(f, "kind", std::string("tdhfb_run"));
    write_attr_string(f, "created_utc", utc_timestamp());
    write_attr_int   (f, "x_sites", res.x_sites);
    write_attr_int   (f, "y_sites", res.y_sites);
    write_attr_int   (f, "n_states", res.n_states);
    write_attr_int   (f, "particle_number", res.n_particles);
    write_attr_double(f, "t_hop", res.t_hop);

    hid_t tdhfb_root = open_or_create_group(f, "tdhfb");

    // Each TDHFB run is a separate group keyed by (U, kick_operator).
    char gname[256];
    std::snprintf(gname, sizeof(gname), "U_%+010.4f__%s",
                  res.U_param, res.kick_operator.c_str());
    std::string safe(gname);
    for (auto& c : safe) {
        if (c == '+') c = 'p';
        else if (c == '-') c = 'm';
        else if (c == '.') c = '_';
        else if (c == ' ') c = '_';
    }

    hid_t g = open_or_create_group(tdhfb_root, safe);

    // Run metadata
    write_attr_double(g, "U", res.U_param);
    write_attr_double(g, "dt", res.dt);
    write_attr_int   (g, "n_steps", res.n_steps);
    write_attr_double(g, "T_total", res.T_total);
    write_attr_double(g, "eta_kick", res.eta_kick);
    write_attr_double(g, "Gamma_smooth", res.Gamma_smooth);
    write_attr_double(g, "mu_fixed", res.mu_fixed);
    write_attr_string(g, "kick_operator", res.kick_operator);
    write_attr_string(g, "integrator", res.integrator);
    write_attr_double(g, "m1_FT", res.m1_FT);
    write_attr_double(g, "m1_commutator", res.m1_commutator);

    // Time series — the meat of the run
    write_dset_1d_double(g, "times",            res.times);
    write_dset_1d_double(g, "F_real",           res.F_real);
    write_dset_1d_double(g, "F_imag",           res.F_imag);
    write_dset_1d_double(g, "energy",           res.energy);
    write_dset_1d_double(g, "particle_number",  res.particle_number);
    write_dset_1d_double(g, "idempotency_err",  res.idempotency_err);
    write_dset_1d_double(g, "hermiticity_err",  res.hermiticity_err);
    write_dset_1d_double(g, "pairing_gap",      res.pairing_gap);
    write_dset_1d_double(g, "kinetic_energy",   res.kinetic_energy);
    write_dset_1d_double(g, "interaction_energy", res.interaction_energy);

    // Strength function
    write_dset_1d_double(g, "E_grid", res.E_grid);
    write_dset_1d_double(g, "S_E",    res.S_E);
    write_dset_1d_double(g, "f_real", res.f_real);
    write_dset_1d_double(g, "f_imag", res.f_imag);

    // Snapshots of R (initial = post-kick, final).  These are the heaviest
    // datasets but invaluable for restart and offline inspection.
    if (res.R_initial.size() > 0) write_dset_2d_complex(g, "R_initial", res.R_initial);
    if (res.R_final.size()   > 0) write_dset_2d_complex(g, "R_final",   res.R_final);

    H5Gclose(g);
    H5Gclose(tdhfb_root);
    H5Fclose(f);
}

// ============================================================
//  Public:  write_quench_result_hdf5
// ============================================================
void write_quench_result_hdf5(const TDHFBQuenchResult& res, const std::string& path) {
    hid_t f = open_or_create_file(path);

    write_attr_string(f, "kind",            std::string("tdhfb_quench"));
    write_attr_string(f, "created_utc",     utc_timestamp());
    write_attr_int   (f, "x_sites",         res.x_sites);
    write_attr_int   (f, "y_sites",         res.y_sites);
    write_attr_int   (f, "n_states",        res.n_states);
    write_attr_int   (f, "particle_number", res.n_particles);
    write_attr_double(f, "t_hop",           res.t_hop);

    hid_t g = open_or_create_group(f, "quench");

    write_attr_double(g, "U_initial",   res.U_initial);
    write_attr_double(g, "U_final",     res.U_final_val);
    write_attr_double(g, "dtau",        res.dtau);
    write_attr_double(g, "energy_tol",  res.energy_tol);
    write_attr_double(g, "mu_final",    res.mu_final);
    write_attr_double(g, "mu_lr",       res.mu_lr);
    write_attr_int   (g, "n_steps_done",res.n_steps_done);
    write_attr_int   (g, "converged",   res.converged ? 1 : 0);
    write_attr_int   (g, "save_every",  res.save_every);

    // Per-step time series
    write_dset_1d_double(g, "tau",                res.tau);
    write_dset_1d_double(g, "energy",             res.energy);
    write_dset_1d_double(g, "particle_number",    res.particle_number);
    write_dset_1d_double(g, "idempotency_err",    res.idempotency_err);
    write_dset_1d_double(g, "hermiticity_err",    res.hermiticity_err);
    write_dset_1d_double(g, "pairing_gap",        res.pairing_gap);
    write_dset_1d_double(g, "mu_history",         res.mu_history);
    write_dset_1d_double(g, "kinetic_energy",     res.kinetic_energy);
    write_dset_1d_double(g, "interaction_energy", res.interaction_energy);
    write_dset_1d_double(g, "snap_tau",           res.snap_tau);

    // Matrix snapshots: shape (n_snap, 2N, 2N, 2) in HDF5
    if (!res.snap_rho.empty())   write_dset_4d_complex(g, "snap_rho",   res.snap_rho);
    if (!res.snap_kappa.empty()) write_dset_4d_complex(g, "snap_kappa", res.snap_kappa);
    if (!res.snap_U_bdg.empty()) write_dset_4d_complex(g, "snap_U_bdg", res.snap_U_bdg);
    if (!res.snap_V_bdg.empty()) write_dset_4d_complex(g, "snap_V_bdg", res.snap_V_bdg);

    if (res.R_final.size() > 0)
        write_dset_2d_complex(g, "R_final", res.R_final);

    H5Gclose(g);
    H5Fclose(f);
}
