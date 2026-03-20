# iQSM+ – Orientation-Adaptive Quantitative Susceptibility Mapping

**Plug-and-Play Latent Feature Editing for Orientation-Adaptive Quantitative Susceptibility Mapping Neural Networks**

[MIA 2024](https://doi.org/10.1016/j.media.2024.103160) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2311.07823) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

iQSM+ enables direct QSM reconstruction from raw MRI phase acquired at **arbitrary orientations**, using orientation-adaptive latent feature editing (OA-LFE) blocks that learn the encoding of acquisition orientation vectors and integrate them seamlessly into the network.

> **Update (March 2025):** New user-friendly MATLAB wrappers for iQSM+/iQSM/iQFM/xQSM/xQSM+ with simpler syntax — all available in this repo.

> **Windows users:** Run `iQSM_fcns/ConfigurePython.m` first and update the `pyExec` variable to your conda Python path.

---

## Overview

### Framework

![Whole Framework](https://github.com/sunhongfu/iQSM_Plus/blob/master/figs/fig1.png)

Fig. 1: The overall structure of the proposed Orientation-Adaptive Neural Network, incorporating OA-LFE blocks that learn orientation encoding and integrate it into latent features.

### Representative Results

![Representative Results](https://github.com/sunhongfu/iQSM_Plus/blob/master/figs/fig3.png)

Fig. 2: Comparison of iQSM, iQSM-Mixed, and iQSM+ on simulated brains at different acquisition orientations and in vivo 3T scans.

---

## Requirements

- Python 3.7+, PyTorch 1.8+
- NVIDIA GPU (CUDA 10.0+)
- MATLAB R2017b+
- FSL (for BET brain mask extraction)

Tested on: Windows 11 (RTX 4090 / A4000), macOS (M1 Pro Max), CentOS 7.8 (Tesla V100).

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/sunhongfu/iQSM_Plus.git
cd iQSM_Plus

conda create -n iQSM_Plus python=3.8
conda activate iQSM_Plus
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install scipy
```

### 2. Run reconstruction (MATLAB)

```matlab
QSM = iQSM_plus(phase, TE, 'mag', mag, 'mask', mask, 'voxel_size', [1,1,1], 'B0', 3, 'B0_dir', [0,0,1], 'eroded_rad', 3, 'output_dir', pwd);
```

For xQSM+ reconstruction:

```matlab
QSM = xQSM_plus(lfs, 'mask', mask, 'B0_dir', [0,0,1], 'voxel_size', [1,1,1], 'output_dir', pwd);
```

---

## MATLAB Wrapper: iQSM_plus

**Compulsory inputs:**
- `phase` — 3D (single-echo) or 4D (multi-echo) GRE phase volume
- `TE` — echo time(s) in seconds, e.g. `20e-3` or `[4,8,12,16,20,24,28]*1e-3`

**Optional inputs:**
- `mag` — magnitude volume (default: ones)
- `mask` — brain mask (default: ones)
- `voxel_size` — resolution in mm (default: `[1 1 1]`)
- `B0_dir` — B0 field direction (default: `[0 0 1]` for axial)
- `B0` — field strength in Tesla (default: `3`)
- `eroded_rad` — brain mask erosion radius in voxels (default: `3`)
- `output_dir` — output folder (default: current directory)

Outputs: `iQSM_plus.nii` (NIfTI) and `iQSM.mat` saved to `output_dir`.

---

## How to Calculate B0_dir from DICOM

```matlab
Xz = dicom_info.ImageOrientationPatient(3);
Yz = dicom_info.ImageOrientationPatient(6);
Zxyz = cross(dicom_info.ImageOrientationPatient(1:3), dicom_info.ImageOrientationPatient(4:6));
Zz = Zxyz(3);
B0_dir = [Xz, Yz, Zz];
```

---

## Available Reconstruction Functions

| Function | Task |
|----------|------|
| `iQSM_plus` | Single-step QSM at arbitrary orientation |
| `iQSM` | Single-step QSM (axial) |
| `iQFM` | Single-step tissue field mapping |
| `xQSM_plus` | xQSM dipole inversion at arbitrary orientation |
| `xQSM` | xQSM dipole inversion (axial) |
| `SQNet` | — |

---

## Citation

```bibtex
@article{iqsmplus2024,
  title={Plug-and-Play Latent Feature Editing for Orientation-Adaptive Quantitative Susceptibility Mapping Neural Networks},
  journal={Medical Image Analysis},
  year={2024},
  doi={10.1016/j.media.2024.103160}
}
```

---

[⬆ top](#iqsm--orientation-adaptive-quantitative-susceptibility-mapping) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)
