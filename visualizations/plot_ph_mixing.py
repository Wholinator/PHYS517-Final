"""Plot particle-hole mixing for a quasiparticle near the Fermi surface.

Left panel  — bar chart:  total particle weight W_u = Σ|u_i|²
                           vs hole weight         W_v = Σ|v_i|²  as U varies.
              At U=0 the state is a pure electron or pure hole (one bar = 1).
              As U → -10 BCS pairing drives it to a 50/50 superposition.

Middle panel — particle density heatmap: |u_up[i]|² + |u_dn[i]|²  per site.
Right  panel — hole    density heatmap: |v_up[i]|² + |v_dn[i]|²  per site.

Both heatmaps use a fixed color scale determined globally across all frames.

Usage:
    python plot_ph_mixing.py --file build/hfb_results_16.h5
    python plot_ph_mixing.py --file build/hfb_results_16.h5 --qp-index 1
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from read_h5 import load_result


def infer_grid(N: int, nx: int, ny: int) -> tuple:
    if nx is not None and ny is not None:
        return nx, ny
    root = int(np.sqrt(N))
    if root * root != N:
        raise ValueError('lattice is non-square; provide --nx and --ny')
    return root, root


def extract_ph(res: dict, qp_index: int):
    """Return (W_u, W_v, u_density, v_density) for the qp_index-th quasiparticle.

    W_u, W_v  — scalar total particle / hole weights (sum to ~1).
    u_density — per-site particle density, shape (ns,).
    v_density — per-site hole    density, shape (ns,).
    """
    if 'H_BdG_evals' not in res or 'H_BdG_evecs' not in res:
        raise KeyError('H_BdG_evals / H_BdG_evecs not found in result')

    evals = np.real(res['H_BdG_evals'])
    evecs = res['H_BdG_evecs']

    pos_indices = np.where(evals > 0)[0]
    if len(pos_indices) == 0:
        pos_indices = np.array([int(np.argmax(evals))])
    pos_indices = pos_indices[np.argsort(evals[pos_indices])]
    idx = int(pos_indices[min(qp_index, len(pos_indices) - 1)])

    n4 = evecs.shape[0]
    n2 = n4 // 2
    ns = n2 // 2

    evec = evecs[:, idx]
    u_up = evec[0:ns]
    u_dn = evec[ns:n2]
    v_up = evec[n2:n2 + ns]
    v_dn = evec[n2 + ns:n4]

    u_dens = np.abs(u_up)**2 + np.abs(u_dn)**2   # per-site particle density
    v_dens = np.abs(v_up)**2 + np.abs(v_dn)**2   # per-site hole density

    W_u = float(u_dens.sum())
    W_v = float(v_dens.sum())

    return W_u, W_v, u_dens, v_dens


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file',      default='build/hfb_results_256.h5')
    p.add_argument('--nx',        type=int,   default=None)
    p.add_argument('--ny',        type=int,   default=None)
    p.add_argument('--qp-index',  type=int,   default=0,   help='quasiparticle index (0=lowest)')
    p.add_argument('--stride',    type=int,   default=1,   help='frame stride')
    p.add_argument('--interval',  type=int,   default=1000, help='ms per frame')
    p.add_argument('--save',      default=None,            help='save animation to file')
    args = p.parse_args()

    # ── pre-load all frames ───────────────────────────────────────────────────
    u_values = list(np.arange(0, -9, -0.05))[::max(1, args.stride)]
    W_u_list  = []
    W_v_list  = []
    u_maps    = []
    v_maps    = []
    valid_u   = []
    nx = ny   = None

    for u_val in u_values:
        try:
            res = load_result(args.file, u_value=u_val)
        except KeyError:
            continue
        try:
            W_u, W_v, u_dens, v_dens = extract_ph(res, args.qp_index)
        except (KeyError, IndexError):
            continue

        if nx is None:
            nx, ny = infer_grid(u_dens.size, args.nx, args.ny)

        W_u_list.append(W_u)
        W_v_list.append(W_v)
        u_maps.append(u_dens.reshape(ny, nx))
        v_maps.append(v_dens.reshape(ny, nx))
        valid_u.append(u_val)

    if not valid_u:
        raise ValueError('no valid U frames found; check --file')

    # fixed heatmap color scale across all frames
    vmax_u = max(m.max() for m in u_maps)
    vmax_v = max(m.max() for m in v_maps)
    vmax   = max(vmax_u, vmax_v)   # shared scale so the two maps are comparable

    # ── figure layout: bar chart | particle map | hole map ────────────────────
    fig, (ax_bar, ax_u, ax_v) = plt.subplots(1, 3, figsize=(13, 4.5))

    # — bar chart —
    bar_colors = ['#4C72B0', '#DD8452']
    bars = ax_bar.bar(['Particle\n(u)', 'Hole\n(v)'],
                      [W_u_list[0], W_v_list[0]],
                      color=bar_colors, width=0.5)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel('weight  Σ|·|²')
    bar_title = ax_bar.set_title(f'QP {args.qp_index}  U={valid_u[0]:.2f}')
    ax_bar.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.5)

    # — heatmaps —
    im_u = ax_u.imshow(u_maps[0], origin='lower', vmin=0, vmax=vmax,
                       cmap='Blues', interpolation='nearest')
    ax_u.set_title('Particle density  |u|²')
    ax_u.set_xlabel('x');  ax_u.set_ylabel('y')
    fig.colorbar(im_u, ax=ax_u, label='|u|²')

    im_v = ax_v.imshow(v_maps[0], origin='lower', vmin=0, vmax=vmax,
                       cmap='Reds', interpolation='nearest')
    ax_v.set_title('Hole density  |v|²')
    ax_v.set_xlabel('x');  ax_v.set_ylabel('y')
    fig.colorbar(im_v, ax=ax_v, label='|v|²')

    fig.tight_layout()

    def update(fi):
        # update bar heights
        bars[0].set_height(W_u_list[fi])
        bars[1].set_height(W_v_list[fi])
        bar_title.set_text(f'QP {args.qp_index}  U={valid_u[fi]:.2f}')

        # update heatmaps
        im_u.set_data(u_maps[fi])
        im_v.set_data(v_maps[fi])

        return bars[0], bars[1], bar_title, im_u, im_v

    anim = animation.FuncAnimation(
        fig, update, frames=len(valid_u), interval=args.interval, blit=False,
    )

    if args.save:
        anim.save(args.save)
    else:
        plt.show()


if __name__ == '__main__':
    main()
