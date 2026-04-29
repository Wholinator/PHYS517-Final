"""Plot heatmaps of the density (p) and pairing (k) matrices.

Shows Re(p) and Re(k) as separate figures with a diverging colormap:
red = negative, white = 0, blue = positive.
Color scale is symmetric about zero and fixed globally across all frames.

Static mode:   plot for a single U value.
Animated mode: slide U from 0 to -10.

Usage:
    python plot_density.py --file build/hfb_results_16.h5
    python plot_density.py --file build/hfb_results_16.h5 --u -3.0
    python plot_density.py --file build/hfb_results_16.h5 --animate-u
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from read_h5 import load_result, list_results


CMAP = 'RdBu'   # red = negative, white = 0, blue = positive


def load_pk(path, u_value=None, u_group=None):
    res = load_result(path, u_group=u_group, u_value=u_value)
    if 'p' not in res or 'k' not in res:
        raise KeyError('p or k not found in result')
    return np.real(res['p']), np.real(res['k'])


def sym_vlim(*arrays):
    """Symmetric vmin/vmax: vmax = max absolute value, vmin = -vmax."""
    vmax = max(np.abs(a).max() for a in arrays)
    return -vmax, vmax


def split_save_path(path, label):
    """Insert a plot label before the extension for paired animation exports."""
    if path is None:
        return None
    root, dot, ext = path.rpartition('.')
    if not dot:
        return f'{path}_{label}'
    return f'{root}_{label}.{ext}'


def make_heatmap_figure(matrix, title, vmin, vmax, u_label):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=CMAP, vmin=vmin, vmax=vmax,
                   interpolation='nearest', origin='upper')
    ax.set_title(title)
    ax.set_xlabel('j')
    ax.set_ylabel('i')
    fig.colorbar(im, ax=ax)
    fig.suptitle(f'U = {u_label}', fontsize=13)
    fig.tight_layout()
    return fig, im


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file',      default='build/hfb_results_16.h5')
    p.add_argument('--u',         default=None,  help='U value (float) or group name')
    p.add_argument('--animate-u', action='store_true', help='animate U from 0 to -10')
    p.add_argument('--stride',    type=int,   default=1,   help='frame stride')
    p.add_argument('--interval',  type=int,   default=100, help='ms per frame')
    p.add_argument('--save',      default=None,            help='save animation to file')
    args = p.parse_args()

    if args.animate_u:
        # ── pre-load all frames ───────────────────────────────────────────
        u_values  = list(np.arange(0, -10.05, -0.05))[::max(1, args.stride)]
        p_frames  = []
        k_frames  = []
        valid_u   = []

        for u_val in u_values:
            try:
                pm, km = load_pk(args.file, u_value=u_val)
            except KeyError:
                continue
            p_frames.append(pm)
            k_frames.append(km)
            valid_u.append(u_val)

        if not valid_u:
            raise ValueError('no valid U frames found; check --file')

        # global symmetric color limits
        vmin_p, vmax_p = sym_vlim(*p_frames)
        vmin_k, vmax_k = sym_vlim(*k_frames)

        fig_p, im_p = make_heatmap_figure(
            p_frames[0], 'Re(ρ)  —  density matrix',
            vmin_p, vmax_p, f'{valid_u[0]:.2f}',
        )
        fig_k, im_k = make_heatmap_figure(
            k_frames[0], 'Re(κ)  —  pairing matrix',
            vmin_k, vmax_k, f'{valid_u[0]:.2f}',
        )

        def update_p(fi):
            im_p.set_data(p_frames[fi])
            fig_p.suptitle(f'U = {valid_u[fi]:.2f}', fontsize=13)
            return im_p,

        def update_k(fi):
            im_k.set_data(k_frames[fi])
            fig_k.suptitle(f'U = {valid_u[fi]:.2f}', fontsize=13)
            return im_k,

        anim_p = animation.FuncAnimation(
            fig_p, update_p, frames=len(valid_u), interval=args.interval, blit=False,
        )
        anim_k = animation.FuncAnimation(
            fig_k, update_k, frames=len(valid_u), interval=args.interval, blit=False,
        )
        if args.save:
            anim_p.save(split_save_path(args.save, 'density'))
            anim_k.save(split_save_path(args.save, 'pairing'))
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

        pm, km = load_pk(args.file, u_value=u_val, u_group=u_group)
        vmin_p, vmax_p = sym_vlim(pm)
        vmin_k, vmax_k = sym_vlim(km)

        u_label = args.u if args.u is not None else 'default'
        make_heatmap_figure(pm, 'Re(ρ)  —  density matrix',
                            vmin_p, vmax_p, u_label)
        make_heatmap_figure(km, 'Re(κ)  —  pairing matrix',
                            vmin_k, vmax_k, u_label)
        plt.show()


if __name__ == '__main__':
    main()
