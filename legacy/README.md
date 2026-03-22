# Legacy Code

This directory contains code that predates the current Python-first rewrite.
It is kept for reference and backwards-compatibility but is **not required** for
normal use of iQSM+.

## `matlab/`

MATLAB wrappers and utility functions. These call the Python backend via
`matlab/fcns/PythonRecon.m` and were the primary interface before the web app
and CLI (`app.py` / `run.py`) were introduced.

To use the MATLAB wrappers, add `legacy/matlab/` and `legacy/matlab/fcns/` to
your MATLAB path, then call e.g.:

```matlab
addpath('legacy/matlab', 'legacy/matlab/fcns');
QSM = iQSM_plus(phase, TE, 'mask', mask, 'voxel_size', [1,1,1], 'B0', 3);
```

## `python/`

Original evaluation and training scripts from the research phase of the project.
The active inference code has been consolidated into `inference.py` at the repo
root; these scripts are kept for reproducibility of published results.
