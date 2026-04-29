"""Plot the superconducting order parameter \Delta ∝ |k| vs U.

For each U group the scalar gap is the mean on-site pairing amplitude:
    Delta = mean_i |k[i,i]|   where k = k[:N, N:]  (top-right NxN block of k).

This is the quantity that appears in  build_delta: Delta_i = 2U k[i, N+i].
It is zero at U=0 and grows as U becomes more attractive.

Usage:
    python plot_phase.py --file build/hfb_results_16.h5
    python plot_phase.py --file build/hfb_results_36.h5 --umin -8
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from read_h5 import list_results, load_result


def collect_gap(path: str, umin: float, umax: float):
    """Return (u_vals, delta_vals) sorted by U."""
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
        if 'k' not in res:
            continue

        k = res['k']
        N = k.shape[0] // 2
        kappa_diag = np.array([k[i, N + i] for i in range(N)])
        delta = float(np.mean(np.abs(kappa_diag)))
        rows.append((u, delta))

    if not rows:
        raise ValueError('no k data found; check --file and --umin/--umax')

    rows.sort(key=lambda r: r[0])
    u_vals     = np.array([r[0] for r in rows])
    delta_vals = np.array([r[1] for r in rows])
    return u_vals, delta_vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file',  default='build/hfb_results_36_30.h5')
    p.add_argument('--umin',  type=float, default=-30.0, help='minimum U to include')
    p.add_argument('--umax',  type=float, default=0.0,   help='maximum U to include')
    p.add_argument('--lw',    type=float, default=2.0,   help='line width')
    p.add_argument('--save',  default=None,              help='save figure to file')
    args = p.parse_args()

    u_vals, delta_vals = collect_gap(args.file, args.umin, args.umax)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(u_vals, delta_vals, color='steelblue', lw=args.lw)
    ax.set_xlabel('U')
    ax.set_ylabel(r'$\bar{\Delta} \propto \langle|\kappa_{ii}|\rangle$')
    ax.set_title('Superconducting order parameter vs U')
    ax.invert_xaxis()

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == '__main__':
    main()
