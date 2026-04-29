"""Plot single-particle coherence |ρ_{ij}| vs inter-site distance r_{ij}.

Data source: the p matrix (2N×2N density matrix).  The spin-up block
p[:N, :N] = ρ↑↑ gives ⟨c†_{i↑} c_{j↑}⟩.  For each ordered pair (i,j)
we plot:

    x = |r_i - r_j|  (Euclidean distance in lattice units)
    y = |ρ↑↑[i,j]|   (log scale)

Points with |ρ| below a numerical floor (1e-14) are dropped.
The y-axis range is fixed across all frames so the decay is visually comparable.

Usage:
    python plot_coherence.py --file build/hfb_results_16.h5
    python plot_coherence.py --file build/hfb_results_36.h5 --nx 6 --ny 6
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from read_h5 import load_result


FLOOR = 1e-14   # drop points below this to keep the log axis clean


def infer_grid(N: int, nx: int, ny: int) -> tuple:
    if nx is not None and ny is not None:
        return nx, ny
    root = int(np.sqrt(N))
    if root * root != N:
        raise ValueError('lattice is non-square; provide --nx and --ny')
    return root, root


def site_distances(nx: int, ny: int):
    """Return flat arrays (dist, ii, jj) for every ordered pair i < j."""
    N = nx * ny
    dist, ii, jj = [], [], []
    for i in range(N):
        xi, yi = i % nx, i // nx
        for j in range(i + 1, N):
            xj, yj = j % nx, j // nx
            dist.append(np.sqrt((xi - xj)**2 + (yi - yj)**2))
            ii.append(i)
            jj.append(j)
    return np.array(dist), np.array(ii), np.array(jj)


def coherence_values(p: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    """|ρ↑↑[i,j]| for each pair, using the top-left N×N spin-up block."""
    N = p.shape[0] // 2
    rho = p[:N, :N]
    return np.abs(rho[ii, jj])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file',      default='build/hfb_results_36_30.h5')
    p.add_argument('--nx',        type=int,   default=None)
    p.add_argument('--ny',        type=int,   default=None)
    p.add_argument('--stride',    type=int,   default=1,    help='frame stride')
    p.add_argument('--interval',  type=int,   default=100,  help='ms per frame')
    p.add_argument('--alpha',     type=float, default=0.4,  help='dot opacity')
    p.add_argument('--ms',        type=float, default=3.0,  help='marker size')
    p.add_argument('--save',      default=None,             help='save animation to file')
    args = p.parse_args()

    # ── pre-load all frames ───────────────────────────────────────────────────
    u_values = list(np.arange(0, -30.05, -0.05))[::max(1, args.stride)]
    all_mags = []
    valid_u  = []
    dist_arr = ii_arr = jj_arr = None
    nx = ny = None

    for u_val in u_values:
        try:
            res = load_result(args.file, u_value=u_val)
        except KeyError:
            continue
        if 'p' not in res:
            continue

        pk = res['p']
        N  = pk.shape[0] // 2

        if nx is None:
            nx, ny      = infer_grid(N, args.nx, args.ny)
            dist_arr, ii_arr, jj_arr = site_distances(nx, ny)

        mags = coherence_values(pk, ii_arr, jj_arr)
        all_mags.append(mags)
        valid_u.append(u_val)

    if not valid_u:
        raise ValueError('no valid U frames found; check --file')

    # ── fixed log-axis range across all frames ────────────────────────────────
    all_vals = np.concatenate([m[m > FLOOR] for m in all_mags])
    ymin = 10 ** np.floor(np.log10(all_vals.min()))
    ymax = 10 ** np.ceil( np.log10(all_vals.max()))

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.1, dist_arr.max() + 0.5)
    ax.set_xlabel(r'$r = |r_i - r_j|$  (lattice units)')
    ax.set_ylabel(r'$|\rho_{ij}|$')
    title = ax.set_title(f'Single-particle coherence  U={valid_u[0]:.2f}')

    # initial frame: mask out sub-floor points
    mask0   = all_mags[0] > FLOOR
    scat    = ax.scatter(dist_arr[mask0], all_mags[0][mask0],
                         s=args.ms**2, alpha=args.alpha,
                         c='steelblue', linewidths=0)

    fig.tight_layout()

    def update(fi):
        mags = all_mags[fi]
        mask = mags > FLOOR
        xy   = np.column_stack([dist_arr[mask], mags[mask]])
        scat.set_offsets(xy)
        title.set_text(f'Single-particle coherence  U={valid_u[fi]:.2f}')
        return scat, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(valid_u), interval=args.interval, blit=False,
    )

    if args.save:
        anim.save(args.save)
    else:
        plt.show()


if __name__ == '__main__':
    main()
