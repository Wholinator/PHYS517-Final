#include "data_output.h"

#include <hdf5.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

bool file_exists(const std::string& path) {
	std::ifstream in(path.c_str());
	return in.good();
}

void ensure_hid(hid_t id, const char* msg) {
	if (id < 0) {
		throw std::runtime_error(msg);
	}
}

hid_t open_or_create_file(const std::string& path) {
	if (file_exists(path)) {
		hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		ensure_hid(file, "H5Fopen failed");
		return file;
	}
	hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	ensure_hid(file, "H5Fcreate failed");
	return file;
}

hid_t create_or_open_group(hid_t loc, const std::string& name) {
	if (H5Lexists(loc, name.c_str(), H5P_DEFAULT) > 0) {
		hid_t group = H5Gopen2(loc, name.c_str(), H5P_DEFAULT);
		ensure_hid(group, "H5Gopen2 failed");
		return group;
	}
	hid_t group = H5Gcreate2(loc, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	ensure_hid(group, "H5Gcreate2 failed");
	return group;
}

void delete_link_if_exists(hid_t loc, const std::string& name) {
	if (H5Lexists(loc, name.c_str(), H5P_DEFAULT) > 0) {
		H5Ldelete(loc, name.c_str(), H5P_DEFAULT);
	}
}

void write_attr_double(hid_t obj, const std::string& name, double value) {
	if (H5Aexists(obj, name.c_str()) > 0) {
		H5Adelete(obj, name.c_str());
	}
	hid_t space = H5Screate(H5S_SCALAR);
	ensure_hid(space, "H5Screate scalar failed (double attr)");
	hid_t attr = H5Acreate2(obj, name.c_str(), H5T_NATIVE_DOUBLE, space,
							H5P_DEFAULT, H5P_DEFAULT);
	ensure_hid(attr, "H5Acreate2 failed (double attr)");
	herr_t status = H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
	if (status < 0) {
		H5Aclose(attr);
		H5Sclose(space);
		throw std::runtime_error("H5Awrite failed (double attr)");
	}
	H5Aclose(attr);
	H5Sclose(space);
}

void write_attr_int(hid_t obj, const std::string& name, int value) {
	if (H5Aexists(obj, name.c_str()) > 0) {
		H5Adelete(obj, name.c_str());
	}
	hid_t space = H5Screate(H5S_SCALAR);
	ensure_hid(space, "H5Screate scalar failed (int attr)");
	hid_t attr = H5Acreate2(obj, name.c_str(), H5T_NATIVE_INT, space,
							H5P_DEFAULT, H5P_DEFAULT);
	ensure_hid(attr, "H5Acreate2 failed (int attr)");
	herr_t status = H5Awrite(attr, H5T_NATIVE_INT, &value);
	if (status < 0) {
		H5Aclose(attr);
		H5Sclose(space);
		throw std::runtime_error("H5Awrite failed (int attr)");
	}
	H5Aclose(attr);
	H5Sclose(space);
}

void write_scalar_complex(hid_t group, const std::string& name,
						  const std::complex<double>& value) {
	delete_link_if_exists(group, name);
	const hsize_t dims[1] = {2};
	hid_t space = H5Screate_simple(1, dims, nullptr);
	ensure_hid(space, "H5Screate_simple failed (complex scalar)");
	hid_t dset = H5Dcreate2(group, name.c_str(), H5T_NATIVE_DOUBLE, space,
							H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	ensure_hid(dset, "H5Dcreate2 failed (complex scalar)");
	double buf[2] = {value.real(), value.imag()};
	herr_t status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
							 H5P_DEFAULT, buf);
	if (status < 0) {
		H5Dclose(dset);
		H5Sclose(space);
		throw std::runtime_error("H5Dwrite failed (complex scalar)");
	}
	H5Dclose(dset);
	H5Sclose(space);
}

void write_vector_double(hid_t group, const std::string& name,
						 const Eigen::VectorXd& vec) {
	delete_link_if_exists(group, name);
	const hsize_t dims[1] = {static_cast<hsize_t>(vec.size())};
	hid_t space = H5Screate_simple(1, dims, nullptr);
	ensure_hid(space, "H5Screate_simple failed (vector)");
	hid_t dset = H5Dcreate2(group, name.c_str(), H5T_NATIVE_DOUBLE, space,
							H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	ensure_hid(dset, "H5Dcreate2 failed (vector)");
	std::vector<double> buf(static_cast<size_t>(vec.size()));
	for (int i = 0; i < vec.size(); ++i) {
		buf[static_cast<size_t>(i)] = vec(i);
	}
	herr_t status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
							 H5P_DEFAULT, buf.data());
	if (status < 0) {
		H5Dclose(dset);
		H5Sclose(space);
		throw std::runtime_error("H5Dwrite failed (vector)");
	}
	H5Dclose(dset);
	H5Sclose(space);
}

void write_matrix_complex(hid_t group, const std::string& name,
						  const Eigen::MatrixXcd& mat) {
	delete_link_if_exists(group, name);
	const hsize_t dims[3] = {
		static_cast<hsize_t>(mat.rows()),
		static_cast<hsize_t>(mat.cols()),
		2
	};
	hid_t space = H5Screate_simple(3, dims, nullptr);
	ensure_hid(space, "H5Screate_simple failed (complex matrix)");
	hid_t dset = H5Dcreate2(group, name.c_str(), H5T_NATIVE_DOUBLE, space,
							H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	ensure_hid(dset, "H5Dcreate2 failed (complex matrix)");

	const size_t rows = static_cast<size_t>(mat.rows());
	const size_t cols = static_cast<size_t>(mat.cols());
	std::vector<double> buf(rows * cols * 2);
	for (size_t r = 0; r < rows; ++r) {
		for (size_t c = 0; c < cols; ++c) {
			const size_t idx = (r * cols + c) * 2;
			const std::complex<double> val = mat(static_cast<int>(r), static_cast<int>(c));
			buf[idx]     = val.real();
			buf[idx + 1] = val.imag();
		}
	}

	herr_t status = H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
							 H5P_DEFAULT, buf.data());
	if (status < 0) {
		H5Dclose(dset);
		H5Sclose(space);
		throw std::runtime_error("H5Dwrite failed (complex matrix)");
	}
	H5Dclose(dset);
	H5Sclose(space);
}

std::string format_u_group(double u) {
	double val = (std::abs(u) < 1e-12) ? 0.0 : u;
	std::ostringstream oss;
	oss << "U_" << std::fixed << std::setprecision(6) << val;
	return oss.str();
}

} // namespace

void write_converged_result_hdf5(const HFBResult& result, const std::string& path) {
	if (!result.converged) {
		return;
	}

	hid_t file = open_or_create_file(path);
	hid_t results_group = create_or_open_group(file, "results");
	std::string group_name = format_u_group(result.U_param);
	hid_t u_group = create_or_open_group(results_group, group_name);

	write_attr_double(u_group, "U_param", result.U_param);
	write_attr_double(u_group, "temperature", result.temperature);
	write_attr_int(u_group, "iteration_count", result.iteration_count);
	write_attr_double(u_group, "entropy", result.entropy);
	write_attr_double(u_group, "grand_potential", result.grand_potential);
	write_attr_double(u_group, "mu", result.mu);
	write_attr_int(u_group, "iters", result.iters);

	// Complex matrices are stored with a trailing dimension of size 2: [real, imag].
	write_scalar_complex(u_group, "energy", result.energy);
	write_matrix_complex(u_group, "p", result.p);
	write_matrix_complex(u_group, "k", result.k);
	write_matrix_complex(u_group, "R", result.R);
	write_matrix_complex(u_group, "H_BdG", result.H_BdG);
	write_matrix_complex(u_group, "U_bdg", result.U_bdg);
	write_matrix_complex(u_group, "V_bdg", result.V_bdg);
	write_vector_double(u_group, "H_BdG_evals", result.H_BdG_evals);
	write_matrix_complex(u_group, "H_BdG_evecs", result.H_BdG_evecs);

	H5Gclose(u_group);
	H5Gclose(results_group);
	H5Fclose(file);
}
