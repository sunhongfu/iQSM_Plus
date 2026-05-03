# iQSM+ – Orientation-Adaptive Quantitative Susceptibility Mapping

**Plug-and-Play Latent Feature Editing for Orientation-Adaptive Quantitative Susceptibility Mapping Neural Networks**

[MIA 2024](https://doi.org/10.1016/j.media.2024.103160) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2311.07823) &nbsp;|&nbsp; [HuggingFace](https://huggingface.co/sunhongfu/iQSM_Plus) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

iQSM+ extends iQSM to handle **arbitrary acquisition orientations** — not just axial scans. It uses orientation-adaptive latent feature editing (OA-LFE) blocks that learn the encoding of acquisition orientation vectors and integrate them into the network. It accepts single-echo and multi-echo phase inputs.

## Highlights

- **DICOM folder input** — drop your raw multi-echo GRE folder; phase and magnitude are auto-separated by `ImageType`, echoes are grouped by `EchoTime`, and TEs / voxel size / B0 strength / **B0 direction** are read from headers. Works in both the **web app** and the **CLI** (`--dicom_dir`).
- **NIfTI / MAT input** — multiple 3D phase echoes or a single 4D volume (`.nii`, `.nii.gz`, `.mat` v5/v7.3 all supported).
- **Orientation aware** — the OA-LFE encoder consumes a B0 direction vector so non-axial scans (oblique, sagittal, coronal) reconstruct correctly without re-sampling.
- **Browser-based UI** — collapsible sections, live progress, slice slider, shape verification, per-run "equivalent CLI command" log entry, dark-mode auto-open with port auto-fallback.
- **Single output** — `iQSM_plus.nii.gz` (susceptibility, χ).

## Layout

- `app.py` — Gradio web app for browser-based inference.
- `run.py` — command-line driver (DICOM folder, 4D NIfTI, per-echo files, or YAML config).
- `inference.py` — pure-Python iQSM+ inference pipeline (handles both single- and multi-echo).
- `data_utils.py` — NIfTI / MAT / DICOM loaders and shape utilities (DICOM splits phase from magnitude and computes B0 direction in image coordinates).
- `models/` — OA-LFE LoT-Unet architecture.

---

## Overview

![Framework](figs/fig1.png)

*Fig. 1: Orientation-Adaptive Neural Network with OA-LFE blocks.*

![Results](figs/fig3.png)

*Fig. 2: Comparison of iQSM, iQSM-Mixed, and iQSM+ at different acquisition orientations.*

---

## Quick Start

### 1. Get the code

**Option A — Git**

```bash
git clone https://github.com/sunhongfu/iQSM_Plus.git
cd iQSM_Plus
```

**Option B — Download ZIP**

1. Open the GitHub repository page.
2. Click **Code** → **Download ZIP**.
3. Unzip and open a terminal in the folder.

---

### 2. Install dependencies

A fresh virtual environment is the recommended way — it isolates iQSM+'s dependencies from anything else on your system and avoids version conflicts.

You need Python 3.10 or 3.11. Check your version:

```bash
python --version
```

If Python is not installed, download it from [python.org](https://www.python.org/downloads/). On Windows, tick **Add Python to PATH** during installation.

**Create and activate a virtual environment:**

macOS / Linux:
```bash
python -m venv venv
source venv/bin/activate
```

Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your prompt. Run this activation command each time you open a new terminal.

**Install PyTorch.** Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select your OS and CUDA version, and copy the install command. For example:

CUDA 12.4 (recommended if you have an NVIDIA GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

CPU only (slower, but works without a GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Install remaining dependencies.** Pick one of the two options below depending on whether you want the browser-based web app:

- **Web app + Command-Line** (recommended for most users):

  ```bash
  pip install -r requirements-webapp.txt
  ```

  Adds Gradio and Matplotlib for the browser UI and slice previews.

- **Command-Line only** (lighter install, fewer dependencies, no web stack):

  ```bash
  pip install -r requirements.txt
  ```

  Skips Gradio and its ~18 transitive packages. Recommended for headless servers, HPC clusters, or environments where Gradio's deps conflict with other tools.

---

### 3. Download checkpoints (and optionally demo data)

Large files (checkpoints and demo data) are excluded from git and hosted on Hugging Face: [sunhongfu/iQSM_Plus](https://huggingface.co/sunhongfu/iQSM_Plus/tree/main).

**Download checkpoints** (required, one-time):

```bash
python run.py --download-checkpoints
```

**Optional — download demo data:**

```bash
python run.py --download-demo
```

This places sample NIfTI files in `demo/`. See [Run Demo Examples](#run-demo-examples) below.

**Manual download (optional).** If the auto-download fails (e.g. behind a firewall), grab the files from Hugging Face and place them as follows:

```text
iQSM_Plus/
├── checkpoints/
│   ├── iQSM_plus.pth
│   └── LoTLayer_chi.pth
└── demo/
    ├── ph_multi_echo.nii.gz
    ├── mag_multi_echo.nii.gz
    ├── mask_multi_echo.nii.gz
    └── params.json
```

---

### 4. Run

Choose the web app (recommended) or the command-line interface.

---

## Web App

```bash
python app.py
```

The app picks port `7860` by default; if it's busy it falls back automatically (`7861`, `7862`, …). Your default browser opens at the dark-themed URL once the server is ready.

### Usage walk-through

The page is organised top-to-bottom; each section is a collapsible accordion.

#### 1. Phase Input

Two tabs — choose **one**:

##### 📁 DICOM Folder *(recommended)*
Click **Select DICOM Folder** and pick the folder containing your multi-echo GRE DICOMs. The app:
- walks the folder recursively (any extension or none — `.dcm`, `.ima`, `.dicom`, …);
- splits **phase** vs **magnitude** via `ImageType` (`P`/`PHASE` vs `M`/`MAGNITUDE`);
- groups by `EchoTime`, sorts each echo by `ImagePositionPatient`;
- normalises phase values to radians where needed;
- builds an LPS→RAS NIfTI affine and writes one NIfTI per modality (`dcm_converted_phase[_4d].nii.gz` and, when present, `dcm_converted_magnitude[_4d].nii.gz`);
- auto-fills **Echo Times**, **Voxel size**, **B0 strength**, and **B0 direction** (in image coordinates) from the headers.

Mixed-study folders, single-echo phase folders, or folders with no phase image are rejected with a clear message instead of producing wrong results.

##### 📄 NIfTI / MAT files *(advanced)*
Click **Add Phase NIfTI / MAT** and pick:
- multiple 3D files (one per echo) — `.nii`, `.nii.gz`, or `.mat`; or
- a single 4D volume of shape `(X, Y, Z, n_echoes)`.

`.mat` files are accepted in **both v5 and v7.3** formats.

#### 2. Processing Order
Lists every phase file that will be fed to the pipeline (sorted in natural numeric order: `mag1`, `mag2`, …, `mag10`). Below the list, a one-line shape summary tells you whether all files share the same volume dimensions.

#### 3. Echo Times (ms)
A single textbox accepts two equivalent formats:

- **Comma-separated values** — one per echo: `3.2, 6.5, 9.8, 13.0`
- **Compact `first_TE : spacing : count`** — uniform spacing only: `3.2 : 3.3 : 8` expands to 8 evenly-spaced echoes

Auto-filled when you use the DICOM Folder tab.

#### 4. Magnitude *(optional)*
Used internally by iQSM+ for magnitude × TE² weighted multi-echo fitting. Auto-filled when DICOM input includes magnitude images.

#### 5. Brain Mask *(optional)*
Click **Select Brain Mask** to provide a BET (or any binary) mask. Supported: `.nii`, `.nii.gz`, `.mat`. Without a mask, all voxels are processed.

#### 6. Acquisition & Hyper-parameters *(collapsed by default)*
- **Voxel size** (mm) — overrides NIfTI header.
- **B0** (Tesla) — defaults to 3.0.
- **B0 direction** — unit vector for non-axial acquisitions. Auto-filled from DICOM headers; for NIfTI input, leave blank for axial scans (defaults to `[0 0 1]`).
- **Mask erosion radius** (voxels) — defaults to 3.
- **Reverse phase sign** — enable if iron-rich deep grey matter appears dark.

#### 7. Run Reconstruction
Hit the green **Run Reconstruction** button. Three sections appear in sequence:

- **Log** — streaming console output, including a *RUN CONFIGURATION* block that prints the equivalent command-line invocation so you can reproduce the run from a terminal.
- **Results** — `iQSM_plus.nii.gz` downloadable when the run completes.
- **Visualisation** — middle-slice grayscale preview with a **Z-slice slider**. Display window (± 0.2 ppm) is editable.

GPU memory is released between runs.

---

## Command-Line Interface

The pipeline can also be driven from the terminal using a YAML config file or explicit arguments.

### Config file

```bash
python run.py --config config.yaml
```

Example `config.yaml`:

```yaml
data_dir: demo

# Pick one input style:
# Style A — DICOM folder:
# dicom_dir: dicoms
# Style B1 — multiple 3D phase echoes:
# echo_files: [phase_e1.nii, phase_e2.nii, phase_e3.nii]
# te_ms: [3.2, 6.5, 9.8]
# Style B2 — single 4D phase NIfTI:
echo_4d: ph_multi_echo.nii.gz
te_ms: [3.2, 6.5, 9.8, 13.1, 16.4, 19.7, 23.1, 26.4]

mag: mag_multi_echo.nii.gz
mask: mask_multi_echo.nii.gz
b0: 3.0
b0_dir: null            # null = read from header / [0,0,1]
eroded_rad: 3
output: ./iqsm_plus_output
```

### Direct Command-Line

DICOM folder (multi-echo phase + magnitude GRE — TEs/voxel size/B0/B0-direction auto-detected):

```bash
python run.py --dicom_dir Data/your_subject_dicoms --mask BET_mask.nii
```

The folder is walked recursively; phase vs magnitude is split via `ImageType`, grouped by `EchoTime`, sorted by `ImagePositionPatient`, and saved as `dcm_converted_phase[_4d].nii.gz` and `dcm_converted_magnitude[_4d].nii.gz` inside `<output>/dicom_converted_nii/`.

Multiple 3D phase NIfTI echoes (combined into a 4D volume internally):

```bash
python run.py \
  --data_dir Data/your_subject \
  --echo_files ph1.nii ph2.nii ph3.nii \
  --te_ms 4 8 12 \
  --mag mag_4d.nii.gz \
  --mask BET_mask.nii
```

Single 4D phase NIfTI:

```bash
python run.py \
  --echo_4d phase_4d.nii.gz \
  --te_ms 3.2 6.5 9.8 13.1 16.4 19.7 23.1 26.4 \
  --mag mag_multi_echo.nii.gz \
  --mask mask_multi_echo.nii.gz
```

Non-axial acquisition — provide B0 direction in image coordinates:

```bash
python run.py --echo_4d phase_4d.nii.gz --te_ms 4 8 12 \
  --b0-dir 0.1 0.0 0.995
```

Legacy single-echo (3D phase, TE in **seconds**):

```bash
python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz
```

`.mat` inputs (v5 or v7.3) must contain a single numeric array per file.

### Arguments at a glance

- Input (mutually exclusive): `--dicom_dir`, `--echo_files`, `--echo_4d`, `--phase`.
- Echo times: `--te_ms` (preferred, milliseconds) or `--te` (seconds, legacy).
- Optional: `--mag`, `--mask` (or `--bet_mask`), `--voxel-size`, `--b0`, `--b0-dir`, `--eroded-rad`, `--reverse-phase-sign`.
- Output: `--output` (default `./iqsm_plus_output`).
- Setup: `--download-checkpoints`, `--download-demo`.
- `--data_dir` is **optional** (defaults to current working directory) — relative input paths are resolved against it.

---

## Run Demo Examples

Once you've downloaded the demo data (`python run.py --download-demo`), you can try iQSM+ in any of the following ways. The demo is an 8-echo GRE acquisition (3 T, 1×1×1 mm isotropic).

### Option 1 — Web app

```bash
python app.py
```

Drop the demo files into the **NIfTI / MAT files** tab, paste the TEs into Echo Times, add the magnitude and mask, and hit **Run Reconstruction**.

### Option 2 — Command line (multi-echo)

```bash
python run.py \
  --echo_4d demo/ph_multi_echo.nii.gz \
  --te_ms 3.2 6.5 9.8 13.1 16.4 19.7 23.1 26.4 \
  --mag demo/mag_multi_echo.nii.gz \
  --mask demo/mask_multi_echo.nii.gz
```

### Option 3 — YAML config

```bash
python run.py --config config.yaml
```

All three options produce the same output: `iQSM_plus.nii.gz` (susceptibility, ppm). View in [FSLeyes](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes), [ITK-SNAP](http://www.itksnap.org/), or [3D Slicer](https://www.slicer.org/).

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
