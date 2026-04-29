Visualization helpers for HFB results

Place your HDF5 files (e.g. `hfb_results.h5`) in the project root or point to them with `--file`.

Quick examples:

Spin quiver (auto grid):

```bash
python visualizations/plot_spins.py --file hfb_results.h5 --u U_0.500000
```

Pairing magnitude/phase:

```bash
python visualizations/plot_kappa.py --file hfb_results.h5 --u U_0.500000 --singlet
```

Quasiparticle density (lowest eigenvector):

```bash
python visualizations/plot_quasiparticle.py --file hfb_results.h5 --u U_0.500000
```

Dependencies: see `visualizations/requirements.txt` (install with `pip install -r visualizations/requirements.txt`).
