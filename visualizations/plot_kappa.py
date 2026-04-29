"""Plot the strongest pairing (kappa) bonds on a lattice.

The k matrix is 2N×2N.  The top-right N×N block is κ[i,j] (spin-up site i
pairs with spin-down site j).  The upper triangle of that block (i < j) gives
one entry per pair; we sort by |κ[i,j]| and keep the top --topn bonds.

Usage:
    python plot_kappa.py --file ../hfb_results.h5 --u -2.0 --nx 4 --ny 4
    python plot_kappa.py --file ../hfb_results.h5 --animate-u --nx 4 --ny 4
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import animation
from read_h5 import load_result


def extract_bonds(k: np.ndarray, nx: int, ny: int, topn: int):
    """Return the top-n bonds from the upper triangle of the top-right block of k.

    Each bond is (magnitude, i, j, xi, yi, xj, yj), sorted descending by magnitude.
    """
    N = k.shape[0] // 2
    kappa = k[:N, N:]  # top-right N×N block

    bonds = []
    for i in range(N):
        for j in range(i + 1, N):
            mag = abs(kappa[i, j])
            bonds.append((mag, i, j, i % nx, i // nx, j % nx, j // nx))

    bonds.sort(key=lambda b: b[0], reverse=True)
    return bonds[:topn]


def bonds_to_lc_data(bonds):
    """Convert bond list to (segments, magnitudes) for LineCollection."""
    if not bonds:
        return np.empty((0, 2, 2)), np.array([])
    segments = np.array([[[b[3], b[4]], [b[5], b[6]]] for b in bonds], dtype=float)
    mags     = np.array([b[0] for b in bonds])
    return segments, mags


def load_k(path, u_value=None, u_group=None):
    res = load_result(path, u_group=u_group, u_value=u_value)
    if 'k' not in res:
        raise KeyError('k (pairing matrix) not found in result')
    return res['k']


def infer_grid(N, nx, ny):
    if nx is not None and ny is not None:
        return nx, ny
    root = int(np.sqrt(N))
    if root * root != N:
        raise ValueError('lattice is non-square; provide --nx and --ny')
    return root, root


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file',     default='build/hfb_results.h5')
    p.add_argument('--u',        default=None,  help='U value (float) or group name')
    p.add_argument('--nx',       type=int,  default=None)
    p.add_argument('--ny',       type=int,  default=None)
    p.add_argument('--topn',     type=int,  default=20,   help='strongest bonds to show')
    p.add_argument('--maxlw',    type=float, default=4.0,  help='max line width')
    p.add_argument('--limit',    type=float, default=None, help='fix colorbar max')
    p.add_argument('--animate-u', action='store_true',    help='animate U from 0 to -10')
    p.add_argument('--stride',   type=int,  default=1,    help='frame stride for --animate-u')
    p.add_argument('--interval', type=int,  default=200,  help='ms per frame')
    p.add_argument('--save',     default=None,            help='save animation to file')
    args = p.parse_args()

    if args.animate_u:
        # ── animated mode ────────────────────────────────────────────────
        u_values = list(np.arange(0, -10.05, -0.05))[::max(1, args.stride)]
        all_bonds = []
        valid_u   = []
        nx = ny = N = None

        for u_val in u_values:
            try:
                k = load_k(args.file, u_value=u_val)
            except KeyError:
                continue
            if N is None:
                N = k.shape[0] // 2
                nx, ny = infer_grid(N, args.nx, args.ny)
            all_bonds.append(extract_bonds(k, nx, ny, args.topn))
            valid_u.append(u_val)

        if not valid_u:
            raise ValueError('no valid U frames found; check --file')

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter([i % nx for i in range(N)], [i // nx for i in range(N)],
                   s=30, c='k', zorder=5)
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)

        lc   = LineCollection([], cmap='viridis')
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, label='|κ|')

        def update(fi):
            bonds            = all_bonds[fi]
            u_val            = valid_u[fi]
            segments, mags   = bonds_to_lc_data(bonds)

            if mags.size == 0:
                lc.set_segments([])
                lc.set_linewidths([])
                lc.set_array(np.array([0.0]))
                lc.set_clim(0.0, 1.0)
                ax.set_title(f'Pairing bonds  U={u_val:.2f}  (none found)')
                return lc,

            scale = mags.max() or 1.0
            lc.set_segments(segments)
            lc.set_linewidths(args.maxlw * mags / scale)
            lc.set_array(mags)
            vmax = abs(args.limit) if args.limit is not None else mags.max()
            lc.set_clim(0.0, vmax)
            cbar.update_normal(lc)
            ax.set_title(f'Pairing bonds  U={u_val:.2f}  top {len(bonds)}')
            return lc,

        anim = animation.FuncAnimation(
            fig, update, frames=len(valid_u), interval=args.interval, blit=False,
        )
        if args.save:
            anim.save(args.save)
        else:
            plt.show()

    else:
        # ── static single-frame mode ──────────────────────────────────────
        u_val   = None
        u_group = None
        if args.u is not None:
            try:
                u_val = float(args.u)
            except ValueError:
                u_group = args.u

        k    = load_k(args.file, u_value=u_val, u_group=u_group)
        N    = k.shape[0] // 2
        nx, ny = infer_grid(N, args.nx, args.ny)

        bonds            = extract_bonds(k, nx, ny, args.topn)
        segments, mags   = bonds_to_lc_data(bonds)

        if mags.size == 0:
            raise ValueError('no bonds found; check --file or --topn')

        scale = mags.max() or 1.0
        lc    = LineCollection(segments, cmap='viridis',
                               linewidths=args.maxlw * mags / scale)
        lc.set_array(mags)
        if args.limit is not None:
            lc.set_clim(0.0, abs(args.limit))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.add_collection(lc)
        ax.scatter([i % nx for i in range(N)], [i // nx for i in range(N)],
                   s=30, c='k', zorder=5)
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
        ax.set_title(f'Pairing bonds  top {len(bonds)} |κ|')
        fig.colorbar(lc, ax=ax, label='|κ|')
        plt.show()


if __name__ == '__main__':
    main()
