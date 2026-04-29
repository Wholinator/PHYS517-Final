"""Plot spin vectors (quiver) extracted from density matrix R in HDF5 results.

Usage example:
python plot_spins.py --file ../hfb_results.h5 --u U_0.500000 --nx 8 --ny 8
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from read_h5 import load_result


def spin_vectors_from_R(R: np.ndarray) -> np.ndarray:
    # R is 2N x 2N generalized density matrix; rho is the particle block
    N2 = R.shape[0]
    N = N2 // 2
    rho = R[:N, :N]
    Ns = N // 2
    S = np.zeros((Ns, 3), dtype=float)
    for i in range(Ns):
        rho_uu = rho[i, i]
        rho_dd = rho[i + Ns, i + Ns]
        rho_ud = rho[i, i + Ns]
        rho_du = rho[i + Ns, i]
        Sz = 0.5 * (rho_uu.real - rho_dd.real)
        Sx = 0.5 * ((rho_ud + rho_du).real)
        Sy = 0.5 * (1j * (rho_ud - rho_du)).real
        S[i, 0] = Sx
        S[i, 1] = Sy
        S[i, 2] = Sz
    return S


def _infer_grid(N: int, nx: int, ny: int) -> tuple:
    if nx is not None and ny is not None:
        if nx * ny < N:
            raise ValueError('nx * ny must be >= number of sites')
        return nx, ny
    root = int(np.sqrt(N))
    if root * root != N:
        raise ValueError('provide --nx and --ny for non-square lattices')
    return root, root


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True)
    p.add_argument('--u', default=None)
    p.add_argument('--temperature', default=None)
    p.add_argument('--group', default=None)
    p.add_argument('--nx', type=int, default=None)
    p.add_argument('--ny', type=int, default=None)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--plot3d', action='store_true')
    args = p.parse_args()

    res = load_result(args.file, args.u, temperature=args.temperature, group_path=args.group)
    if 'R' not in res:
        raise KeyError('R not found in result')
    R = res['R']
    N = R.shape[0] // 2
    Ns = N // 2

    S = spin_vectors_from_R(R)

    nx, ny = _infer_grid(Ns, args.nx, args.ny)

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()[:Ns]
    Y = Y.flatten()[:Ns]

    U = S[:, 0]
    V = S[:, 1]
    W = S[:, 2]

    fig = plt.figure(figsize=(6, 6))
    if args.plot3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, np.zeros_like(X), U, V, W, length=args.scale)
        ax.set_title('Spin vectors (3D)')
    else:
        ax = fig.add_subplot(111)
        q = ax.quiver(X, Y, U, V, color=None, cmap='viridis', scale=1.0/args.scale)
        q.set_array(W)
        ax.set_aspect('equal')
        ax.set_title('Spin quiver (color=Sz)')
        plt.colorbar(q, label='Sz')
    plt.show()

if __name__ == '__main__':
    main()
