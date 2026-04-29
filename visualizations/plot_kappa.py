"""Plot the strongest pairing (kappa) bonds on a lattice.

Usage:
python plot_kappa.py --file ../hfb_results.h5 --u U_0.500000 --singlet --nx 8 --ny 8
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import animation
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


def _parse_group_selector(u_arg: str, temp_arg: str) -> tuple:
    u_group = u_arg
    u_value = None
    if u_arg is not None:
        try:
            u_value = float(u_arg)
        except ValueError:
            pass
        else:
            u_group = None

    temperature = temp_arg
    temp_value = None
    if temp_arg is not None:
        try:
            temp_value = float(temp_arg)
        except ValueError:
            pass
        else:
            temperature = None

    return u_group, u_value, temperature, temp_value


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


def _kappa_for_plot(res: dict, use_singlet: bool) -> np.ndarray:
    if 'k' not in res:
        raise KeyError('k (pairing) not found in result')
    k = res['k']
    return singlet_kappa_from_k(k) if use_singlet else k


def _bond_segments_from_kappa(
    kapp: np.ndarray,
    nx: int,
    ny: int,
    threshold: float,
    topn: int,
    nn_only: bool,
    periodic: bool,
):
    N = kapp.shape[0] if kapp.shape[0] % 2 != 0 else kapp.shape[0] // 2
    bonds = _neighbors(nx, ny, periodic) if nn_only else _all_pairs(N)

    bond_data = []
    for i, j in bonds:
        if i >= N or j >= N:
            continue
        val = kapp[i, j]
        mag = np.abs(val)
        if mag <= threshold:
            continue
        x0, y0 = i % nx, i // nx
        x1, y1 = j % nx, j // nx
        bond_data.append((mag, [(x0, y0), (x1, y1)]))

    if not bond_data and threshold != float('-inf'):
        for i, j in bonds:
            if i >= N or j >= N:
                continue
            val = kapp[i, j]
            mag = np.abs(val)
            x0, y0 = i % nx, i // nx
            x1, y1 = j % nx, j // nx
            bond_data.append((mag, [(x0, y0), (x1, y1)]))

    if not bond_data:
        return np.array([]), np.array([])

    bond_data.sort(key=lambda item: item[0], reverse=True)
    bond_data = bond_data[:max(1, topn)]
    mags = np.array([item[0] for item in bond_data])
    segments = np.array([item[1] for item in bond_data], dtype=float)
    return segments, mags


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='build/hfb_results.h5')
    p.add_argument('--u', default=None)
    p.add_argument('--temperature', default=None)
    p.add_argument('--group', default=None)
    p.add_argument('--singlet', action='store_true')
    p.add_argument('--nx', type=int, default=None)
    p.add_argument('--ny', type=int, default=None)
    p.add_argument('--periodic', action='store_true')
    p.add_argument('--nn', action='store_true')
    p.add_argument('--all-pairs', action='store_true', help='Use all site pairs instead of nearest neighbors')
    p.add_argument('--threshold', type=float, default=0.0)
    p.add_argument('--topn', type=int, default=20)
    p.add_argument('--interval', type=int, default=200)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--limit', type=float, default=None)
    p.add_argument('--animate-u', action='store_true', help='Animate across U values 0 to -10')
    p.add_argument('--maxlw', type=float, default=4.0)
    p.add_argument('--save', default=None)
    args = p.parse_args()

    group_path = args.group
    if group_path is not None and group_path.startswith('results/'):
        group_path = group_path.split('results/', 1)[1]

    use_nn = args.nn# or not args.all_pairs

    if args.animate_u:
        u_values = list(np.arange(0, -10.05, -0.05))[::max(1, args.stride)]
        frame_segments = []
        frame_mags = []
        valid_u = []
        nx = ny = None
        N = None

        for u_val in u_values:
            try:
                res = load_result(
                    args.file,
                    u_value=u_val,
                    temperature=args.temperature,
                    group_path=group_path,
                )
            except KeyError:
                continue

            kapp = _kappa_for_plot(res, args.singlet)
            if N is None:
                N = kapp.shape[0] if args.singlet else kapp.shape[0] // 2
                nx, ny = _infer_grid(N, args.nx, args.ny)

            segments, mags = _bond_segments_from_kappa(
                kapp,
                nx,
                ny,
                args.threshold,
                args.topn,
                use_nn,
                args.periodic,
            )
            frame_segments.append(segments)
            frame_mags.append(mags)
            valid_u.append(u_val)

        if not valid_u:
            raise ValueError('no valid U frames found in file for --animate-u')

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter([i % nx for i in range(N)], [i // nx for i in range(N)], s=10, c='k')
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)

        lc = LineCollection([], cmap='viridis')
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, label='|kappa|')

        def update(frame_idx: int):
            segments = frame_segments[frame_idx]
            mags = frame_mags[frame_idx]
            u_val = valid_u[frame_idx]
            if mags.size == 0:
                lc.set_segments([])
                lc.set_linewidths([])
                lc.set_array(np.array([0.0]))
                lc.set_clim(0.0, 1.0)
                ax.set_title(f'Pairing bonds (U={u_val:.2f}, none above threshold)')
                return lc,

            scale = mags.max() if mags.max() > 0 else 1.0
            widths = args.maxlw * (mags / scale)
            segs = np.array(segments, dtype=float)
            lc.set_segments(segs)
            lc.set_linewidths(widths)
            lc.set_array(mags)
            if args.limit is not None:
                vmax = abs(args.limit)
            else:
                vmax = mags.max()
            lc.set_clim(0.0, vmax)
            cbar.update_normal(lc)
            try:
                allpts = segs.reshape(-1, 2)
                minxy = allpts.min(axis=0)
                maxxy = allpts.max(axis=0)
                ax.set_xlim(min(-0.5, minxy[0] - 0.5), max(nx - 0.5, maxxy[0] + 0.5))
                ax.set_ylim(min(-0.5, minxy[1] - 0.5), max(ny - 0.5, maxxy[1] + 0.5))
            except Exception:
                pass
            ax.set_title(f'Pairing bonds (U={u_val:.2f}, top {len(segments)} by |kappa|)')
            return lc,

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(valid_u),
            interval=args.interval,
            blit=False,
        )

        if args.save:
            anim.save(args.save)
        else:
            plt.show()

    else:
        u_group, u_value, temperature, temp_value = _parse_group_selector(args.u, args.temperature)
        res = load_result(
            args.file,
            u_group=u_group,
            temperature=temperature,
            group_path=group_path,
            u_value=u_value,
            temp_value=temp_value,
        )
        kapp = _kappa_for_plot(res, args.singlet)

        N = kapp.shape[0] if args.singlet else kapp.shape[0] // 2
        nx, ny = _infer_grid(N, args.nx, args.ny)
        segments, mags = _bond_segments_from_kappa(
            kapp,
            nx,
            ny,
            args.threshold,
            args.topn,
            use_nn,
            args.periodic,
        )

        if mags.size == 0:
            raise ValueError('no bonds above threshold; lower --threshold')

        scale = mags.max() if mags.max() > 0 else 1.0
        widths = args.maxlw * (mags / scale)
        fig, ax = plt.subplots(figsize=(6, 6))
        lc = LineCollection(segments, cmap='viridis', linewidths=widths)
        lc.set_array(mags)
        if args.limit is not None:
            lc.set_clim(0.0, abs(args.limit))
        ax.add_collection(lc)
        ax.scatter([i % nx for i in range(N)], [i // nx for i in range(N)], s=10, c='k')
        ax.set_aspect('equal')
        try:
            allpts = np.array(segments, dtype=float).reshape(-1, 2)
            minxy = allpts.min(axis=0)
            maxxy = allpts.max(axis=0)
            ax.set_xlim(min(-0.5, minxy[0] - 0.5), max(nx - 0.5, maxxy[0] + 0.5))
            ax.set_ylim(min(-0.5, minxy[1] - 0.5), max(ny - 0.5, maxxy[1] + 0.5))
        except Exception:
            ax.set_xlim(-0.5, nx - 0.5)
            ax.set_ylim(-0.5, ny - 0.5)
        ax.set_title(f'Pairing bonds (top {len(segments)} by |kappa|)')
        fig.colorbar(lc, ax=ax, label='|kappa|')
        plt.show()


if __name__ == '__main__':
    main()
