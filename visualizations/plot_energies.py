"""Plot BdG eigenvalue spectrum vs U.

Reads H_BdG_evals from every U group in an HDF5 file and draws one line
per eigenvalue level, showing how the quasiparticle gap opens as U decreases.

Usage:
    python plot_energies.py --file ../hfb_results.h5
    python plot_energies.py --file build/hfb_results.h5 --umin -6 --umax 0
"""
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from read_h5 import list_results, load_result


def collect_spectra(path: str, umin: float, umax: float):
    """Return (u_vals, evals_matrix) where evals_matrix[i] is the sorted
    eigenvalue array at u_vals[i].  Only U groups with H_BdG_evals are kept."""
    groups = list_results(path)

    rows = []
    for key, attrs in groups.items():
        u = attrs.get('U_param')
        if u is None:
            continue
        u = float(u)
        if not (umin <= u <= umax):
            continue
        try:
            res = load_result(path, group_path=key)
        except Exception:
            continue
        if 'H_BdG_evals' not in res:
            continue
        evals = np.sort(np.real(-res['H_BdG_evals']))
        rows.append((u, evals))

    if not rows:
        raise ValueError('no H_BdG_evals data found; check --file and --umin/--umax')

    rows.sort(key=lambda r: r[0])
    u_vals = np.array([r[0] for r in rows])
    evals_matrix = np.array([r[1] for r in rows])   # shape (n_U, n_levels)
    return u_vals, evals_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file',   default='build/hfb_results_16.h5')
    p.add_argument('--umin',   type=float, default=-10.0, help='minimum U to include')
    p.add_argument('--umax',   type=float, default=0.0,   help='maximum U to include')
    p.add_argument('--alpha',  type=float, default=0.5,   help='line opacity')
    p.add_argument('--lw',     type=float, default=1.5,   help='line width')
    p.add_argument('--save',   default=None,              help='save figure to file')
    args = p.parse_args()

    u_vals, evals_matrix = collect_spectra(args.file, args.umin, args.umax)
    n_levels = evals_matrix.shape[1]

    fig, ax = plt.subplots(figsize=(8, 5))
    for level_idx in range(n_levels):
        ax.plot(u_vals, evals_matrix[:, level_idx],
                color='steelblue', alpha=args.alpha, lw=args.lw)

    ax.set_xlabel('U')
    ax.set_ylabel('BdG eigenvalue')
    ax.set_title(f'BdG spectrum vs U  ({n_levels} levels)')
    ax.axhline(0, color='k', lw=0.5, ls='--')

    ax.invert_xaxis()

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == '__main__':
    main()
