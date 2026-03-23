# iQSM+ – Orientation-Adaptive Quantitative Susceptibility Mapping

**Plug-and-Play Latent Feature Editing for Orientation-Adaptive Quantitative Susceptibility Mapping Neural Networks**

[MIA 2024](https://doi.org/10.1016/j.media.2024.103160) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2311.07823) &nbsp;|&nbsp; [HuggingFace](https://huggingface.co/sunhongfu/iQSM_Plus) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

iQSM+ extends iQSM to handle **arbitrary acquisition orientations** — not just axial scans. It uses orientation-adaptive latent feature editing (OA-LFE) blocks that learn the encoding of acquisition orientation vectors and integrate them into the network. It supports single-echo and multi-echo phase inputs.

---

## Overview

![Framework](https://github.com/sunhongfu/iQSM_Plus/blob/master/figs/fig1.png)

*Fig. 1: Orientation-Adaptive Neural Network with OA-LFE blocks.*

![Results](https://github.com/sunhongfu/iQSM_Plus/blob/master/figs/fig3.png)

*Fig. 2: Comparison of iQSM, iQSM-Mixed, and iQSM+ at different acquisition orientations.*

---

## Which Setup Should I Use?

| I want to… | Best option |
|---|---|
| Just try it quickly, no coding | **Docker** (Option 1) |
| Use the web app on a shared server | **Docker** or **Conda** |
| Run from the command line / scripts | **Conda** or **pip** |
| Call from MATLAB | **MATLAB wrapper** (requires Conda or pip) |
| Use an NVIDIA GPU | **Docker** (GPU mode) or **Conda/pip** |

---

## Option 1 — Docker (Web App, Recommended)

**Best for:** Windows, macOS (including Apple Silicon), Linux. No Python setup needed.

**Requirements:** [Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine on Linux).

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sunhongfu/iQSM_Plus.git
cd iQSM_Plus

# 2. Download model weights (run once, on the host — not inside Docker)
python run.py --download-checkpoints

# 3. (Optional) Download demo data to try the app
python run.py --download-demo

# 4. Start the app
docker compose up
```

Open **http://localhost:7860** in your browser.

> The `demo/` and `checkpoints/` folders are bind-mounted into the container — files downloaded on the host are immediately visible inside Docker without a restart.

### Enable NVIDIA GPU (Linux only)

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
2. Edit `docker-compose.yml`: set `TORCH_VARIANT: cu121`.
3. Uncomment the GPU block at the bottom of `docker-compose.yml`.
4. Rebuild and start:

```bash
docker compose build
docker compose up
```

---

## Option 2 — Conda (Web App + CLI)

**Best for:** Users with Anaconda or Miniconda already installed.

**Requirements:** [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/).

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sunhongfu/iQSM_Plus.git
cd iQSM_Plus

# 2. Create and activate the environment
conda env create -f environment.yml
conda activate iqsm-plus

# 3a. Launch the web app
python app.py
#     → open http://localhost:7860

# 3b. Or use the command line directly
python run.py --download-demo          # download demo data
python run.py --download-checkpoints   # download model weights
python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz
```

> **GPU note:** `environment.yml` installs `pytorch-cuda=12.1` by default. To install CPU-only, remove that line from `environment.yml` before running `conda env create`.

---

## Option 3 — pip (Web App + CLI)

**Best for:** Users who prefer pip, or already have a Python environment.

**Requirements:** Python 3.10+.

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sunhongfu/iQSM_Plus.git
cd iQSM_Plus

# 2. Install PyTorch (choose one):
pip install torch                                                        # Apple Silicon or CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu      # Linux/Windows CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cu121    # Linux/Windows NVIDIA GPU

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4a. Launch the web app
python app.py
#     → open http://localhost:7860

# 4b. Or use the command line directly
python run.py --download-demo          # download demo data
python run.py --download-checkpoints   # download model weights
python run.py --phase ph.nii.gz --te 0.0032 0.0065 0.0098 --mag mag.nii.gz
```

---

## Option 4 — MATLAB Wrapper

**Best for:** Users already working in MATLAB who want to call iQSM+ directly.

**Requirements:** MATLAB R2017b+, and a working Python environment (Conda or pip — see Options 2/3 above).

> **Windows users:** Run `iQSM_fcns/ConfigurePython.m` first and update the `pyExec` variable to your Python executable path.

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/sunhongfu/iQSM_Plus.git

# 2. Set up Python environment (Conda or pip, see Options 2/3 above)
#    Model weights are downloaded automatically on first inference.
```

```matlab
% Run the demo
demo_multi_echo
```

### Function signatures

```matlab
% iQSM+ — arbitrary orientation QSM
QSM = iQSM_plus(phase, TE, 'mag', mag, 'mask', mask, ...
                'voxel_size', [1,1,1], 'B0', 3, 'B0_dir', [0,0,1], ...
                'eroded_rad', 3, 'output_dir', pwd);

% xQSM+ — dipole inversion at arbitrary orientation
QSM = xQSM_plus(lfs, 'mask', mask, 'B0_dir', [0,0,1], ...
                'voxel_size', [1,1,1], 'output_dir', pwd);
```

### Parameters — iQSM_plus

| Parameter | Required | Description |
|---|---|---|
| `phase` | ✓ | 3D (single-echo) or 4D (multi-echo) GRE phase volume |
| `TE` | ✓ | Echo time(s) in seconds — e.g. `20e-3` or `[4,8,12,16,20,24,28]*1e-3` |
| `mag` | | Magnitude volume (default: ones) |
| `mask` | | Brain mask (default: ones) |
| `voxel_size` | | Resolution in mm (default: `[1 1 1]`) |
| `B0_dir` | | B0 direction unit vector (default: `[0 0 1]` for axial) |
| `B0` | | Field strength in Tesla (default: `3`) |
| `eroded_rad` | | Brain mask erosion radius in voxels (default: `3`) |
| `output_dir` | | Output folder (default: current directory) |

### How to derive B0_dir from DICOM

```matlab
Xz = dicom_info.ImageOrientationPatient(3);
Yz = dicom_info.ImageOrientationPatient(6);
Zxyz = cross(dicom_info.ImageOrientationPatient(1:3), dicom_info.ImageOrientationPatient(4:6));
B0_dir = [Xz, Yz, Zxyz(3)];
```

### Available MATLAB functions

| Function | Description |
|---|---|
| `iQSM_plus` | Single-step QSM at arbitrary orientation |
| `iQSM` | Single-step QSM (axial only) |
| `iQFM` | Single-step tissue field mapping |
| `xQSM_plus` | xQSM dipole inversion at arbitrary orientation |
| `xQSM` | xQSM dipole inversion (axial only) |

---

## Downloading Checkpoints and Demo Data

Model weights and demo data are hosted on [Hugging Face Hub](https://huggingface.co/sunhongfu/iQSM_Plus).

```bash
# Download model weights into checkpoints/
python run.py --download-checkpoints

# Download demo data into demo/ and print the run command
python run.py --download-demo
```

> **Docker users:** Run these commands on the **host machine** before or after starting the container — not inside Docker. The folders are bind-mounted, so files appear immediately without a restart.

Files are also cached in `~/.cache/huggingface/hub/` for reuse.

---

## Web App Features

- Upload phase NIfTI (`.nii` / `.nii.gz`) or DICOM — single-echo (3D) or multi-echo (4D)
- Echo times auto-extracted from DICOM headers
- Optionally upload magnitude and brain mask
- Set B0 direction for non-axial acquisitions
- Click **⬇ Load Demo Data** to auto-fill all fields with the demo dataset
- Click **▶ Run Reconstruction** to generate the QSM map
- Download output NIfTI — view in [FSLeyes](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes), [ITK-SNAP](http://www.itksnap.org/), or [3D Slicer](https://www.slicer.org/)

---

## Command Line Reference

```bash
# Show all options
python run.py --help

# Download demo data (prints example run command)
python run.py --download-demo

# Single-echo reconstruction
python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz

# Multi-echo reconstruction
python run.py --phase ph.nii.gz --te 0.0032 0.0065 0.0098 --mag mag.nii.gz

# Non-axial acquisition (specify B0 direction)
python run.py --phase ph.nii.gz --te 0.020 --b0-dir 0.1 0.0 0.995

# Use a YAML config file
python run.py --config config.yaml

# Override output directory
python run.py --phase ph.nii.gz --te 0.020 --output ./my_output/
```

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
