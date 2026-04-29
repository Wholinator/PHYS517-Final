"""Plot the lowest-energy quasiparticle density on the 2D lattice.

For each U the code finds the minimum positive BdG eigenvalue, extracts the
corresponding eigenvector from H_BdG_evecs, and computes the site-resolved
density  n_i = |u_up[i]|² + |u_dn[i]|² + |v_up[i]|² + |v_dn[i]|².

Static mode:   plot for a single U value.
Animated mode: slide U from 0 to -10, fixed color scale across all frames.

Usage:
    python plot_quasiparticle.py --file ../hfb_results.h5 --u -2.0 --nx 4 --ny 4
    python plot_quasiparticle.py --file build/hfb_results_16.h5 --animate-u --nx 4 --ny 4
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from read_h5 import load_result, list_results


def infer_grid(N: int, nx: int, ny: int) -> tuple:
    if nx is not None and ny is not None:
        return nx, ny
    root = int(np.sqrt(N))
    if root * root != N:
        raise ValueError('lattice is non-square; provide --nx and --ny')
    return root, root


def qp_density(res: dict, qp_index: int = 0) -> np.ndarray:
    """Return site density of the qp_index-th lowest-energy quasiparticle.

    H_BdG_evecs is 4N×4N, columns sorted ascending by eigenvalue.
    Positive eigenvalues start at column 2N; qp_index=0 is the gap edge.
    Top half of that column = particle amplitudes (u_up, u_dn).
    Bottom half               = hole amplitudes   (v_up, v_dn).
    """
    if 'H_BdG_evals' not in res or 'H_BdG_evecs' not in res:
        raise KeyError('H_BdG_evals / H_BdG_evecs not found in result')

    evals = np.real(res['H_BdG_evals'])
    evecs = res['H_BdG_evecs']          # 4N × 4N complex

    # indices of positive eigenvalues sorted by energy (ascending)
    pos_indices = np.where(evals > 0)[0]
    if len(pos_indices) == 0:
        pos_indices = np.array([int(np.argmax(evals))])  # fallback
    pos_indices = pos_indices[np.argsort(evals[pos_indices])]

    qp_index = min(qp_index, len(pos_indices) - 1)
    idx = int(pos_indices[qp_index])

    n4 = evecs.shape[0]
    n2 = n4 // 2                        # 2 * n_sites
    ns = n2 // 2                        # n_sites (includes spin)

    evec   = evecs[:, idx]
    u_up   = evec[0:ns]
    u_dn   = evec[ns:n2]
    v_up   = evec[n2:n2 + ns]
    v_dn   = evec[n2 + ns:n4]

    return np.abs(u_up)**2 + np.abs(u_dn)**2 + np.abs(v_up)**2 + np.abs(v_dn)**2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file',      default='build/hfb_results_36_obc.h5')
    p.add_argument('--u',         default=None,  help='U value (float) or group name')
    p.add_argument('--nx',        type=int, default=None)
    p.add_argument('--ny',        type=int, default=None)
    p.add_argument('--qp-index',  type=int,   default=0,   help='which quasiparticle (0=lowest)')
    p.add_argument('--animate-u', action='store_true', help='animate U from 0 to -10')
    p.add_argument('--stride',    type=int,   default=1,   help='frame stride')
    p.add_argument('--interval',  type=int,   default=100, help='ms per frame')
    p.add_argument('--save',      default=None,            help='save animation to file')
    args = p.parse_args()

    if args.animate_u:
        # ── collect all frames ────────────────────────────────────────────
        u_values = list(np.arange(0, -10.05, -0.05))[::max(1, args.stride)]
        frames   = []
        valid_u  = []
        nx = ny = None

        for u_val in u_values:
            try:
                res = load_result(args.file, u_value=u_val)
            except KeyError:
                continue
            try:
                dens = qp_density(res, args.qp_index)
            except (KeyError, IndexError):
                continue

            if nx is None:
                ns     = dens.size
                nx, ny = infer_grid(ns, args.nx, args.ny)

            frames.append(dens.reshape(ny, nx))
            valid_u.append(u_val)

        if not frames:
            raise ValueError('no valid U frames found; check --file')

        # fixed color scale across all frames
        vmin = min(f.min() for f in frames)
        vmax = max(f.max() for f in frames)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(frames[0], origin='lower', vmin=vmin, vmax=vmax,
                       cmap='viridis', interpolation='nearest')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        title = ax.set_title(f'QP {args.qp_index} density  U={valid_u[0]:.2f}')
        fig.colorbar(im, ax=ax, label='|u|² + |v|²')
        fig.tight_layout()

        def update(fi):
            im.set_data(frames[fi])
            title.set_text(f'QP {args.qp_index} density  U={valid_u[fi]:.2f}')
            return im, title

        anim = animation.FuncAnimation(
            fig, update, frames=len(valid_u), interval=args.interval, blit=False,
        )
        if args.save:
            anim.save(args.save)
        else:
            plt.show()

    else:
        # ── static single-frame ───────────────────────────────────────────
        u_val   = None
        u_group = None
        if args.u is not None:
            try:
                u_val = float(args.u)
            except ValueError:
                u_group = args.u

        res  = load_result(args.file, u_group=u_group, u_value=u_val)
        dens = qp_density(res, args.qp_index)
        ns   = dens.size
        nx, ny = infer_grid(ns, args.nx, args.ny)
        data = dens.reshape(ny, nx)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(data, origin='lower', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        u_label = args.u if args.u is not None else 'default'
        ax.set_title(f'QP {args.qp_index} density  U={u_label}')
        fig.colorbar(im, ax=ax, label='|u|² + |v|²')
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
