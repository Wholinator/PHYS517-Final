"""Helpers to read HFB HDF5 results files.

Functions:
- list_results(path)
- open_results(path, u=None, temperature=None, group=None)
- read_complex(h5obj, name) -> np.ndarray[complex]

Assumptions: complex arrays are stored as real/imag trailing dimension of size 2.
"""

import h5py
import numpy as np
from typing import Optional, Tuple, Dict, Any


def _to_complex(arr: np.ndarray) -> np.ndarray:
    # arr expected shape (..., 2)
    return arr[..., 0] + 1j * arr[..., 1]


def read_complex(h5obj: h5py.Group, name: str) -> np.ndarray:
    if name not in h5obj:
        raise KeyError(f"{name} not found in HDF5 group")
    ds = h5obj[name]
    data = np.array(ds)
    if data.ndim >= 1 and data.shape[-1] == 2:
        return _to_complex(data)
    # scalar complex stored as [2]
    if data.size == 2:
        return data[0] + 1j * data[1]
    # already real
    return data


def _iter_groups(results: h5py.Group):
    for key in results.keys():
        obj = results[key]
        if isinstance(obj, h5py.Group):
            yield key, obj
            for subkey in obj.keys():
                sub = obj[subkey]
                if isinstance(sub, h5py.Group):
                    yield f"{key}/{subkey}", sub


def list_results(path: str) -> Dict[str, Any]:
    out = {}
    with h5py.File(path, 'r') as f:
        if 'results' not in f:
            return out
        results = f['results']
        for key, grp in _iter_groups(results):
            attrs = dict(grp.attrs)
            out[key] = attrs
    return out


def _maybe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _find_group_by_attrs(results: h5py.Group,
                         u_value: Optional[float],
                         temperature: Optional[float],
                         tol: float = 1e-6) -> Optional[h5py.Group]:
    for _, grp in _iter_groups(results):
        attrs = grp.attrs
        u_attr = _maybe_float(attrs.get('U_param'))
        t_attr = _maybe_float(attrs.get('temperature'))
        if u_value is not None and (u_attr is None or abs(u_attr - u_value) > tol):
            continue
        if temperature is not None and (t_attr is None or abs(t_attr - temperature) > tol):
            continue
        return grp
    return None


def open_result_group(path: str,
                      u_group: Optional[str] = None,
                      temperature: Optional[str] = None,
                      group_path: Optional[str] = None,
                      u_value: Optional[float] = None,
                      temp_value: Optional[float] = None) -> h5py.Group:
    f = h5py.File(path, 'r')
    if 'results' not in f:
        f.close()
        raise KeyError('results group not found')
    results = f['results']
    if group_path is not None:
        if group_path not in results:
            f.close()
            raise KeyError(f"{group_path} not found in results")
        return results[group_path]
    if u_group is None and temperature is None and u_value is None and temp_value is None:
        name = list(results.keys())[0]
        return results[name]
    if u_group is not None:
        if u_group not in results:
            f.close()
            raise KeyError(f"{u_group} not found in results")
        grp = results[u_group]
        if temperature is None and temp_value is None:
            return grp
        if temperature is not None and temperature in grp:
            return grp[temperature]
        sub = _find_group_by_attrs(grp, None, _maybe_float(temp_value), tol=1e-6)
        if sub is not None:
            return sub
        f.close()
        raise KeyError("temperature subgroup not found under u_group")
    # fallback to attribute search
    sub = _find_group_by_attrs(results, _maybe_float(u_value), _maybe_float(temp_value), tol=1e-6)
    if sub is None:
        f.close()
        raise KeyError("no matching group by attributes")
    return sub


def load_result(path: str,
                u_group: Optional[str] = None,
                temperature: Optional[str] = None,
                group_path: Optional[str] = None,
                u_value: Optional[float] = None,
                temp_value: Optional[float] = None) -> Dict[str, Any]:
    g = open_result_group(path,
                          u_group=u_group,
                          temperature=temperature,
                          group_path=group_path,
                          u_value=u_value,
                          temp_value=temp_value)
    # gather scalars
    meta = dict(g.attrs)
    data = {}
    for name in g.keys():
        if isinstance(g[name], h5py.Dataset):
            # heuristics: complex if last dim ==2
            arr = np.array(g[name])
            if arr.ndim >= 1 and arr.shape[-1] == 2:
                data[name] = _to_complex(arr)
            else:
                data[name] = arr
    g.file.close()
    return {**meta, **data}
