"""Plot the momentum distribution n(k_x, k_y) as U sweeps from 0 to -10.

The density matrix ρ = p[:N, :N]  (spin-up block of the 2N×2N p matrix).

    n(k) = (1/N) Σ_{ij} e^{-ik·(r_i - r_j)} ρ_{ij}
         = (1/N) φ_k† ρ φ_k,   φ_k[j] = e^{i k·r_j}

k-grid: k = 2π (m_x/nx, m_y/ny), m running over the first Brillouin zone.
The heatmap is fftshifted so k=(0,0) sits at the centre.

Color scale is fixed globally across all frames.

Usage:
    python plot_momentum.py --file build/hfb_results_16.h5
    python plot_momentum.py --file build/hfb_results_36.h5 --nx 6 --ny 6
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


def build_phase_matrix(nx: int, ny: int) -> np.ndarray:
    """Return phase[k_flat, site] = exp(i k·r_site) for all BZ k-points.

    k = 2π (mx/nx, my/ny), mx in fftfreq order.
    Shape: (ny*nx, ny*nx).
    """
    N = nx * ny
    sites_x = np.array([i % nx for i in range(N)], dtype=float)
    sites_y = np.array([i // nx for i in range(N)], dtype=float)

    mx = np.fft.fftfreq(nx) * nx   # [0, 1, ..., nx/2, -nx/2+1, ..., -1]
    my = np.fft.fftfreq(ny) * ny
    MX, MY = np.meshgrid(mx, my)   # shape (ny, nx)

    mx_flat = MX.ravel()[:, np.newaxis]   # (Nk, 1)
    my_flat = MY.ravel()[:, np.newaxis]

    phase = np.exp(2j * np.pi * (
        mx_flat * sites_x[np.newaxis, :] / nx +
        my_flat * sites_y[np.newaxis, :] / ny
    ))                                     # (Nk, N)
    return phase


def momentum_distribution(rho: np.ndarray, phase: np.ndarray,
                           nx: int, ny: int) -> np.ndarray:
    """Compute n(k) and return as (ny, nx) real array, fftshifted."""
    N    = rho.shape[0]
    # n(k) = (1/N) conj(phase) @ rho @ phase.T  (diagonal of outer product)
    # equivalent to row-wise dot: for each k, conj(phase[k]) · (rho @ phase[k])
    rho_phi    = rho @ phase.conj().T          # (N, Nk)
    n_k_flat   = np.real(np.einsum('ij,ji->i', phase, rho_phi)) / N
    n_k        = n_k_flat.reshape(ny, nx)
    return np.fft.fftshift(n_k)               # center k=(0,0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file',      default='build/hfb_results_256.h5')
    p.add_argument('--nx',        type=int,   default=None)
    p.add_argument('--ny',        type=int,   default=None)
    p.add_argument('--stride',    type=int,   default=1,    help='frame stride')
    p.add_argument('--interval',  type=int,   default=500,  help='ms per frame')
    p.add_argument('--cmap',      default='viridis',        help='colormap')
    p.add_argument('--save',      default=None,             help='save animation to file')
    args = p.parse_args()

    # ── pre-load all frames ───────────────────────────────────────────────────
    u_values = list(np.arange(0, -10.05, -0.05))[::max(1, args.stride)]
    frames   = []
    valid_u  = []
    phase_mat = nx = ny = None

    for u_val in u_values:
        try:
            res = load_result(args.file, u_value=u_val)
        except KeyError:
            continue
        if 'p' not in res:
            continue

        pk  = res['p']
        N   = pk.shape[0] // 2
        rho = pk[:N, :N]                       # spin-up block

        if nx is None:
            nx, ny    = infer_grid(N, args.nx, args.ny)
            phase_mat = build_phase_matrix(nx, ny)

        nk = momentum_distribution(rho, phase_mat, nx, ny)
        frames.append(nk)
        valid_u.append(u_val)

    if not frames:
        raise ValueError('no valid U frames found; check --file')

    # fixed color scale
    vmin = min(f.min() for f in frames)
    vmax = max(f.max() for f in frames)

    # k-axis ticks in units of π
    kx_ticks     = np.fft.fftshift(np.fft.fftfreq(nx) * nx)  # pixel positions after shift
    ky_ticks     = np.fft.fftshift(np.fft.fftfreq(ny) * ny)
    kx_vals_pi   = np.fft.fftshift(np.fft.fftfreq(nx)) * 2   # in units of π: 2π*m/nx / π
    ky_vals_pi   = np.fft.fftshift(np.fft.fftfreq(ny)) * 2

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))

    extent = [-np.pi, np.pi, -np.pi, np.pi]
    im = ax.imshow(frames[0], origin='lower', vmin=vmin, vmax=vmax,
                   cmap=args.cmap, interpolation='nearest', extent=extent,
                   aspect='equal')

    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    title = ax.set_title(f'Momentum distribution  U={valid_u[0]:.2f}')
    fig.colorbar(im, ax=ax, label=r'$n(k)$')
    fig.tight_layout()

    def update(fi):
        im.set_data(frames[fi])
        title.set_text(f'Momentum distribution  U={valid_u[fi]:.2f}')
        return im, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(valid_u), interval=args.interval, blit=False,
    )

    if args.save:
        anim.save(args.save)
    else:
        plt.show()


if __name__ == '__main__':
    main()
