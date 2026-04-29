"""Plot spin vectors and Sz heatmaps from HFB results.

Usage example:
python plot_spins.py --file ../hfb_results.h5 --u U_0.500000 --nx 8 --ny 8
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from read_h5 import load_result


def _to_complex(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    return arr


def _extract_rho_from_R(R: np.ndarray) -> np.ndarray:
    n2 = R.shape[0]
    if n2 % 2 != 0:
        raise ValueError('R must be 2N x 2N')
    n = n2 // 2
    return _to_complex(R[:n, :n])


def spin_vectors_from_rho(rho: np.ndarray) -> np.ndarray:
    # rho is 2Ns x 2Ns particle-density block
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError('rho must be a square matrix')
    if rho.shape[0] % 2 != 0:
        raise ValueError('rho dimension must be even')
    ns = rho.shape[0] // 2
    spins = np.zeros((ns, 3), dtype=float)
    for i in range(ns):
        rho_uu = rho[i, i]
        rho_dd = rho[i + ns, i + ns]
        rho_ud = rho[i, i + ns]
        rho_du = rho[i + ns, i]
        spins[i, 0] = 0.5 * (rho_ud + rho_du).real
        spins[i, 1] = 0.5 * (1j * (rho_ud - rho_du)).real
        spins[i, 2] = 0.5 * (rho_uu.real - rho_dd.real)
    return spins


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


def sz_heatmap_from_rho(rho: np.ndarray) -> np.ndarray:
    spins = spin_vectors_from_rho(rho)
    return spins[:, 2]


def _load_result_for_group(args, group_path, u_value=None):
    return load_result(
        args.file,
        u_value=u_value,
        temperature=args.temperature,
        group_path=group_path,
    )


def _heatmap_limits(data: np.ndarray, limit: float | None) -> tuple:
    if limit is not None:
        bound = abs(limit)
    else:
        bound = float(np.max(np.abs(data))) if data.size else 1.0
    if bound == 0:
        bound = 1.0
    return -bound, bound


def _apply_group_path(group: str | None) -> str | None:
    if group is not None and group.startswith('results/'):
        return group.split('results/', 1)[1]
    return group


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='build/hfb_results.h5')
    p.add_argument('--u', default=None)
    p.add_argument('--temperature', default=None)
    p.add_argument('--group', default=None)
    p.add_argument('--nx', type=int, default=None)
    p.add_argument('--ny', type=int, default=None)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--interval', type=int, default=100)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--limit', type=float, default=None)
    p.add_argument('--animate-u', action='store_true')
    p.add_argument('--animate-3d', action='store_true')
    p.add_argument('--plot3d', action='store_true')
    p.add_argument('--save', default=None)
    p.add_argument('--save-3d', default=None)
    p.add_argument('--spin-rate', type=float, default=1.5)
    p.add_argument('--elev', type=float, default=25.0)
    args = p.parse_args()

    group_path = _apply_group_path(args.group)

    if args.animate_u:
        u_values = list(np.arange(0, -10.05, -0.05))[::max(1, args.stride)]
        frames = []
        x_grid = y_grid = None
        nx = ny = None

        for u_val in u_values:
            try:
                res = _load_result_for_group(args, group_path, u_value=u_val)
            except KeyError:
                continue
            if 'R' not in res:
                continue
            rho = _extract_rho_from_R(res['R'])
            sz = sz_heatmap_from_rho(rho)
            if x_grid is None:
                nx, ny = _infer_grid(sz.size, args.nx, args.ny)
                x_grid = np.arange(nx)
                y_grid = np.arange(ny)
            frames.append((u_val, sz.reshape(ny, nx)))

        if not frames:
            raise ValueError('no valid U frames found in file for --animate-u')

        fig, ax = plt.subplots(figsize=(6, 5))
        lo, hi = _heatmap_limits(frames[0][1], args.limit)
        im = ax.imshow(frames[0][1], origin='lower', cmap='coolwarm', vmin=lo, vmax=hi)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('Sz')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        def update(frame_idx: int):
            u_val, data = frames[frame_idx]
            im.set_data(data)
            lo, hi = _heatmap_limits(data, args.limit)
            im.set_clim(lo, hi)
            ax.set_title(f'Sz heatmap (U={u_val:.2f})')
            cb.update_normal(im)
            return (im,)

        anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=args.interval, blit=False)
        if args.save:
            anim.save(args.save)
        else:
            plt.show()
        return

    u_group, u_value, temperature, temp_value = _parse_group_selector(args.u, args.temperature)
    res = load_result(
        args.file,
        u_group=u_group,
        temperature=temperature,
        group_path=group_path,
        u_value=u_value,
        temp_value=temp_value,
    )
    if 'R' not in res:
        raise KeyError('R not found in result')
    rho = _extract_rho_from_R(res['R'])
    spins = spin_vectors_from_rho(rho)
    sz = sz_heatmap_from_rho(rho)

    nx, ny = _infer_grid(spins.shape[0], args.nx, args.ny)
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()[:spins.shape[0]]
    Y = Y.flatten()[:spins.shape[0]]
    U = spins[:, 0]
    V = spins[:, 1]
    W = spins[:, 2]
    heat = sz.reshape(ny, nx)

    if args.plot3d:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, np.zeros_like(X), U, V, W, length=args.scale)
        ax.set_xlim(0, nx - 1)
        ax.set_ylim(0, ny - 1)
        zmax = max(1e-9, float(np.max(np.abs(W))))
        ax.set_zlim(-zmax, zmax)
        ax.set_title('Spin vectors (3D)')
        plt.show()
        return

    if args.animate_3d:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('Spin vectors (3D)')
        ax.set_xlim(0, nx - 1)
        ax.set_ylim(0, ny - 1)
        zmax = max(1e-9, float(np.max(np.abs(W))))
        ax.set_zlim(-zmax, zmax)
        ax.view_init(elev=args.elev, azim=0.0)

        def update(frame_idx: int):
            ax.cla()
            ax.quiver(X, Y, np.zeros_like(X), U, V, W, length=args.scale)
            ax.set_title('Spin vectors (3D)')
            ax.set_xlim(0, nx - 1)
            ax.set_ylim(0, ny - 1)
            ax.set_zlim(-zmax, zmax)
            ax.view_init(elev=args.elev, azim=frame_idx * args.spin_rate)
            return ()

        anim = animation.FuncAnimation(fig, update, frames=180, interval=args.interval, blit=False)
        if args.save_3d:
            anim.save(args.save_3d)
        elif args.save:
            anim.save(args.save)
        else:
            plt.show()
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax0, ax1 = axes
    q = ax0.quiver(X, Y, U, V, W, cmap='viridis', scale=1.0 / max(args.scale, 1e-9))
    q.set_array(W)
    ax0.set_aspect('equal')
    ax0.set_xlim(-0.5, nx - 0.5)
    ax0.set_ylim(-0.5, ny - 0.5)
    ax0.set_title('Spin vectors (color=Sz)')
    fig.colorbar(q, ax=ax0, label='Sz')

    lo, hi = _heatmap_limits(heat, args.limit)
    im = ax1.imshow(heat, origin='lower', cmap='coolwarm', vmin=lo, vmax=hi)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Sz heatmap')
    fig.colorbar(im, ax=ax1, label='Sz')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
