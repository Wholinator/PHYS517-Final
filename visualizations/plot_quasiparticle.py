"""Plot quasiparticle density for a chosen Bogoliubov eigenvector.

Uses U/V matrices when available and sums spin components per spatial site.
"""

## visualizations/plot_quasiparticle.py --file build/hfb_results.h5 --u=U_-1.000000 --index=15
import argparse
import numpy as np
import matplotlib.pyplot as plt
from read_h5 import load_result


def _infer_grid(N: int, nx: int, ny: int) -> tuple:
    if nx is not None and ny is not None:
        if nx * ny < N:
            raise ValueError('nx * ny must be >= number of sites')
        return nx, ny
    root = int(np.sqrt(N))
    if root * root != N:
        raise ValueError('provide --nx and --ny for non-square lattices')
    return root, root


def qp_density_from_uv(U: np.ndarray, V: np.ndarray, idx: int) -> np.ndarray:
    # U,V are (2Ns) x (2Ns*?) with columns as eigenvectors
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError('U and V must be 2D matrices')
    if U.shape != V.shape:
        raise ValueError('U and V shapes must match')
    if idx < 0 or idx >= U.shape[1]:
        raise IndexError('eigenvector index out of range')
    u = U[:, idx]
    v = V[:, idx]
    L = u.shape[0]
    if L % 2 != 0:
        raise ValueError('U/V row count must be even (spin up/down)')
    Ns = L // 2
    dens = np.zeros(Ns)
    for i in range(Ns):
        dens[i] = (np.abs(u[i])**2 + np.abs(u[i + Ns])**2 +
                   np.abs(v[i])**2 + np.abs(v[i + Ns])**2)
    return dens


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True)
    p.add_argument('--u', default=None)
    p.add_argument('--temperature', default=None)
    p.add_argument('--group', default=None)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--nx', type=int, default=None)
    p.add_argument('--ny', type=int, default=None)
    args = p.parse_args()

    res = load_result(args.file, args.u, temperature=args.temperature, group_path=args.group)
    if 'U_bdg' not in res or 'V_bdg' not in res:
        raise KeyError('U_bdg and V_bdg not found in result')
    U = res['U_bdg']
    V = res['V_bdg']
    dens = qp_density_from_uv(U, V, args.index)
    nx, ny = _infer_grid(dens.size, args.nx, args.ny)
    data = dens.reshape(ny, nx)

    plt.figure(figsize=(5, 4))
    im = plt.imshow(data, origin='lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Quasiparticle density (index {args.index})')
    plt.colorbar(im)
    plt.show()

if __name__ == '__main__':
    main()
