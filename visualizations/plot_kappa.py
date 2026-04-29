"""Plot pairing (kappa) bond magnitude/phase on a lattice.

Usage:
python plot_kappa.py --file ../hfb_results.h5 --u U_0.500000 --singlet --nx 8 --ny 8
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from read_h5 import load_result


def singlet_kappa_from_k(k: np.ndarray) -> np.ndarray:
    # k is 2N x 2N; singlet uses block [0:Ns, Ns:2Ns]
    N2 = k.shape[0]
    if N2 % 2 != 0:
        return k
    Ns = N2 // 2
    return k[:Ns, Ns:]


def _infer_grid(N: int, nx: int, ny: int) -> tuple:
    if nx is not None and ny is not None:
        if nx * ny < N:
            raise ValueError('nx * ny must be >= number of sites')
        return nx, ny
    root = int(np.sqrt(N))
    if root * root != N:
        raise ValueError('provide --nx and --ny for non-square lattices')
    return root, root


def _neighbors(nx: int, ny: int, periodic: bool):
    bonds = []
    for y in range(ny):
        for x in range(nx):
            i = y * nx + x
            if x + 1 < nx:
                bonds.append((i, i + 1))
            elif periodic:
                bonds.append((i, y * nx))
            if y + 1 < ny:
                bonds.append((i, i + nx))
            elif periodic:
                bonds.append((i, x))
    return bonds


def _all_pairs(N: int):
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default="build/hfb_results.h5")
    p.add_argument('--u', default="U_-3.000000")
    p.add_argument('--temperature', default=None)
    p.add_argument('--group', default=None)
    p.add_argument('--singlet', action='store_true')
    p.add_argument('--nx', type=int, default=None)
    p.add_argument('--ny', type=int, default=None)
    p.add_argument('--periodic', action='store_true')
    p.add_argument('--nn', action='store_true')
    p.add_argument('--threshold', type=float, default=0.0)
    p.add_argument('--maxlw', type=float, default=4.0)
    args = p.parse_args()

    res = load_result(args.file, args.u, temperature=args.temperature, group_path=args.group)
    if 'k' not in res:
        raise KeyError('k (pairing) not found in result')
    k = res['k']
    if args.singlet:
        kapp = singlet_kappa_from_k(k)
    else:
        kapp = k

    N = kapp.shape[0] if args.singlet else kapp.shape[0] // 2
    nx, ny = _infer_grid(N, args.nx, args.ny)

    bonds = _neighbors(nx, ny, args.periodic) if args.nn else _all_pairs(N)
    mags = []
    phases = []
    segments = []
    for i, j in bonds:
        if i >= N or j >= N:
            continue
        val = kapp[i, j]
        mag = np.abs(val)
        if mag <= args.threshold:
            continue
        phase = np.angle(val)
        x0, y0 = i % nx, i // nx
        x1, y1 = j % nx, j // nx
        segments.append([(x0, y0), (x1, y1)])
        mags.append(mag)
        phases.append(phase)

    if not segments:
        raise ValueError('no bonds above threshold; lower --threshold')

    mags = np.array(mags)
    phases = np.array(phases)
    widths = args.maxlw * (mags / mags.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    lc = LineCollection(segments, cmap='twilight', linewidths=widths)
    lc.set_array(phases)
    ax.add_collection(lc)
    ax.scatter([i % nx for i in range(N)], [i // nx for i in range(N)], s=10, c='k')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_title('Pairing bonds (color=phase, width=|kappa|)')
    plt.colorbar(lc, label='arg(kappa)')
    plt.show()

if __name__ == '__main__':
    main()
