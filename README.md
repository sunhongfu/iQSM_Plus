# iQSM+ – Orientation-Adaptive Quantitative Susceptibility Mapping

**Plug-and-Play Latent Feature Editing for Orientation-Adaptive Quantitative Susceptibility Mapping Neural Networks**

[MIA 2024](https://doi.org/10.1016/j.media.2024.103160) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2311.07823) &nbsp;|&nbsp; [HuggingFace](https://huggingface.co/sunhongfu/iQSM_Plus) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

iQSM+ extends iQSM to handle **arbitrary acquisition orientations** — not just axial scans. It uses orientation-adaptive latent feature editing (OA-LFE) blocks that learn the encoding of acquisition-orientation vectors and integrate them into the network. iQSM+ accepts single-echo and multi-echo phase + (optional) magnitude inputs.

## Highlights

- **Orientation-aware reconstruction** — the OA-LFE encoder consumes a B0 direction vector, so non-axial scans (oblique, sagittal, coronal) reconstruct correctly without any re-sampling.
- **NIfTI / MAT input** — phase + (optional) magnitude as 3D-per-echo files or a single 4D volume; `.nii`, `.nii.gz`, `.mat` v5/v7.3 all supported.
- **Native multi-echo handling inside the model** — iQSM+'s pipeline ingests all echoes at once and combines them internally with magnitude × TE² weighting (uniform magnitude when none is supplied).
- **Browser-based UI** — collapsible sections, live progress log, slice slider, orientation-preview panels (last-echo phase / magnitude / mask) for quick alignment checks, shape verification, port auto-fallback, SSH-aware launch.
- **Standalone DICOM helper** — `dicom_to_nifti.py` converts raw GRE DICOMs (phase + magnitude **or** real + imaginary, with optional GE slice-direction chopper correction) into the NIfTI files the web app and CLI consume — and computes B0 direction from `ImageOrientationPatient` for you.
- **Single output** — `iQSM_plus.nii.gz` (susceptibility, χ).

## Layout

| File / folder | Purpose |
|---|---|
| `app.py` | Gradio web app for browser-based inference. |
| `run.py` | Command-line driver (per-echo files, 4D NIfTI, converted folder, or YAML config). |
| `dicom_to_nifti.py` | Standalone DICOM → NIfTI converter (run once, before `app.py` / `run.py`). |
| `inference.py` | Pure-Python iQSM+ pipeline (handles single- and multi-echo natively). |
| `data_utils.py` | NIfTI / MAT loaders, shape utilities, DICOM splitter (also computes B0 direction). |
| `models/` | OA-LFE LoT-Unet architecture. |
| `config.yaml` | Example YAML config for `run.py --config`. |

---

## Overview

![Framework](figs/fig1.png)

*Fig. 1: Orientation-adaptive neural network with OA-LFE blocks.*

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

A fresh virtual environment isolates iQSM+'s dependencies and avoids version conflicts. You need **Python 3.10 or 3.11**.

```bash
python --version    # check
```

If Python is missing, install from [python.org](https://www.python.org/downloads/). On Windows, tick **Add Python to PATH** during installation.

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

Re-activate each new terminal.

**Install PyTorch.** Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), pick your OS / CUDA version, and run the command it gives you. Examples:

```bash
# NVIDIA GPU (CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only (works without GPU, slower):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Install the rest:**

- **Web app + CLI** (recommended):
  ```bash
  pip install -r requirements-webapp.txt
  ```

- **CLI only** (lighter, no Gradio / FastAPI / Pydantic / Uvicorn):
  ```bash
  pip install -r requirements.txt
  ```

---

### 3. Download checkpoints (and optionally demo data)

Checkpoints and demo data are excluded from git and hosted on Hugging Face: [sunhongfu/iQSM_Plus](https://huggingface.co/sunhongfu/iQSM_Plus/tree/main).

**Checkpoints** (required, one-time):
```bash
python run.py --download-checkpoints
```

**Demo data** (optional, see [Run Demo Examples](#run-demo-examples)):
```bash
python run.py --download-demo
```

**Manual download** (if behind a firewall) — grab the files from Hugging Face and place them as:

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

If you're starting from raw DICOMs, do the [DICOM → NIfTI conversion step](#dicom--nifti-conversion) first. Then choose the [Web App](#web-app) (recommended) or the [Command-Line Interface](#command-line-interface).

---

## DICOM → NIfTI conversion

The web app and CLI consume **NIfTI** files, not raw DICOMs. `dicom_to_nifti.py` walks one or more folders, splits modalities by the DICOM `ImageType` tag (with fall-backs to `ComplexImageComponent` and the GE private tag `(0043, 102f)`), groups slices by `EchoTime`, computes the **B0 direction in image coordinates** from `ImageOrientationPatient`, and emits ready-to-use NIfTIs plus a `params.json` you can copy values out of.

It accepts **either** of the modality combinations a scanner may export:

- **phase (P/PHASE) + magnitude (M/MAGNITUDE)** — used directly, or
- **real (R/REAL) + imaginary (I/IMAGINARY)** — phase and magnitude are derived from the complex signal:
  ```
  phase     = angle(R + 1j·I)
  magnitude = |R + 1j·I|
  ```

When both pairs are present in the same folder, **real + imaginary is preferred**.

### Combined folder (any/all modalities mixed together)

```bash
python dicom_to_nifti.py --dicom_dir /path/to/dicoms --out_dir ./converted
```

### Separate folders, phase + magnitude

```bash
python dicom_to_nifti.py \
  --phase_dir /path/to/phase_dicoms \
  --mag_dir   /path/to/magnitude_dicoms \
  --out_dir   ./converted
```

### Separate folders, real + imaginary (phase + magnitude derived)

```bash
python dicom_to_nifti.py \
  --real_dir /path/to/real_dicoms \
  --imag_dir /path/to/imaginary_dicoms \
  --out_dir  ./converted
```

### GE slice-direction chopper (real + imaginary only)

GE 3D-GRE recon inserts an alternating ±1 along the slice direction in image space (a missing `fftshift` in their pipeline). Magnitude is invariant under this, but phase derived from raw real + imag shows π flips on every other slice — looking like massive wrapping. The script automatically applies the `(-1)^z` chopper when `Manufacturer = GE MEDICAL SYSTEMS`. Override:

```bash
python dicom_to_nifti.py --dicom_dir /path/to/dicoms --chopper on   # always apply
python dicom_to_nifti.py --dicom_dir /path/to/dicoms --chopper off  # never apply
```

`--chopper auto` (default) only fires for GE. `--chopper` has no effect when phase + magnitude DICOMs are used directly.

### After conversion

You'll see a copy-paste-friendly summary you can paste into the iQSM+ web app form:

```text
─── Acquisition values ───
  Echo Times (ms)  : 3.2, 6.5, 9.8, 13.1, 16.4, 19.7, 23.1, 26.4
  Voxel size (mm)  : 1 1 1
  B0 (Tesla)       : 3.0
  B0 direction     : 0.0  0.0  1.0
──────────────────────────
```

The output folder will contain:

```text
converted/
├── dcm_converted_phase[_4d].nii.gz
├── dcm_converted_magnitude[_4d].nii.gz
└── params.json
```

`python dicom_to_nifti.py --help` lists all flags. The output folder feeds directly into:

- the **web app** (drop the NIfTIs into the upload buttons; copy values from `params.json` into the form), or
- the **CLI** (`run.py --from_converted ./converted` reads `params.json` automatically — no retyping).

---

## Web App

```bash
python app.py
```

The app picks port `7860` by default; if it's busy it falls back automatically (`7861`, `7862`, …). Your default browser opens once the server is ready.

### Usage walk-through

#### 1. Phase + Magnitude Input

Two side-by-side upload buttons:

- **Add Phase NIfTI / MAT** (left, required)
- **Add Magnitude NIfTI / MAT (optional)** (right)

Each accepts one 4D file or multiple 3D files (one per echo). Supported: `.nii`, `.nii.gz`, `.mat` (v5 or v7.3).

**Phase is required.** **Magnitude is optional** — used internally by iQSM+ for magnitude × TE² weighted multi-echo fitting; without it iQSM+ falls back to uniform magnitude (TE²-only weighting).

Have raw DICOMs? See [DICOM → NIfTI conversion](#dicom--nifti-conversion).

#### 2. Processing Order

Two parallel lists (Phase left, Magnitude right) showing every uploaded file in natural numeric order (`mag1`, `mag2`, …, `mag10`). Each list shows a **shape summary** so mismatches are obvious before you run, plus an explicit "✕ Remove all …" button per modality. When both modalities are supplied, the two columns must have the same echo count.

#### 3. Echo Times (ms)

A single textbox accepts two equivalent formats:

- Comma-separated values (any spacing) — `2.4, 3.6, 9.2, 20.8`
- Compact `first_TE : spacing : count` — `4.5 : 5.0 : 5` expands to `4.5, 9.5, 14.5, 19.5, 24.5`

Voxel size is auto-filled from the first NIfTI's header on upload (overridable below).

#### 4. Brain Mask *(optional but recommended)*

This section is **open by default**. A brain mask improves **iQSM+ reconstruction quality** by concentrating the network on tissue voxels.

Default mask erosion is 3 voxels; adjust under **Acquisition & Hyper-parameters** below if you'd rather keep more cortical brain region.

⚠️ **Make sure the mask is oriented and aligned to the phase / magnitude volumes.** After the run finishes you can confirm in the **Visualisation** panel — the brain-mask preview shares the same slice slider as the phase / magnitude previews.

#### 5. Acquisition & Hyper-parameters *(collapsed by default)*

| Field | Notes |
|---|---|
| Voxel size (mm) | Overrides NIfTI header. Auto-filled on upload. |
| B0 (Tesla) | Defaults to 3.0. |
| **B0 direction** (unit vector) | Auto-filled from `params.json` when using a converted DICOM folder; for hand-prepared NIfTIs, leave blank for axial scans (defaults to `[0, 0, 1]`). For oblique / sagittal / coronal scans, enter the unit vector — this is what makes iQSM+ orientation-aware. |
| Mask erosion radius (voxels) | Disabled (and 0) when no mask is provided; defaults to 3 once a mask is supplied. |
| Reverse phase sign | Enable if iron-rich deep grey matter appears dark (rather than bright) in the QSM output. |

#### 6. Run Reconstruction

Click the green **Run Reconstruction** button. Below it you'll see, in order:

- **Log** — streaming console output, including a *RUN CONFIGURATION* block that prints the equivalent CLI invocation so you can reproduce the run from a terminal.
- **Visualisation** — middle-slice grayscale preview of `iQSM_plus.nii.gz` with a **Z-slice slider** and editable display window (± 0.2 ppm). Below it, three **orientation-preview** panels show the last-echo raw phase, last-echo raw magnitude, and brain mask (auto-windowed) sharing the same slice slider so you can verify alignment.
- **Results** — `iQSM_plus.nii.gz` is listed for download. Click the file size to download a single file, or click **📦 Download all (ZIP)** at the bottom for the whole bundle.

GPU memory is released between runs, so you can upload a new dataset and re-run without restarting the page.

The web app accepts files up to **5 GB** (`max_file_size="5gb"` on launch).

#### Running over SSH

The launch script detects `SSH_CONNECTION` and skips the auto-open browser step (which would only try to launch a browser on the remote box). It prints the URL and a port-forward hint instead:

```text
Running over SSH — auto-open skipped.
Open this URL in your local browser:
  http://127.0.0.1:7860/
If the host isn't reachable from your laptop, forward the port:
  ssh -L 7860:127.0.0.1:7860 <user>@<host>
```

---

## Command-Line Interface

`run.py` can be driven via flags, a YAML config, or the converted-folder shortcut. It always validates inputs and prints an *Equivalent command-line invocation* line so a web-app run can be reproduced verbatim from a terminal.

### One-step from a converted DICOM folder

After [DICOM conversion](#dicom--nifti-conversion):

```bash
python run.py --from_converted ./converted --mask BET_mask.nii
```

`--from_converted` reads `phase.nii.gz`, `magnitude.nii.gz`, and `params.json` (TEs, voxel size, B0, **and B0 direction**) automatically — no retyping.

### Multiple 3D phase NIfTI echoes

(stacked into a 4D volume internally; iQSM+ takes 4D phase natively):

```bash
python run.py \
  --data_dir Data/your_subject \
  --echo_files ph1.nii ph2.nii ph3.nii \
  --te_ms 4 8 12 \
  --mag mag_4d.nii.gz \
  --mask BET_mask.nii
```

### Single 4D phase NIfTI

```bash
python run.py \
  --echo_4d phase_4d.nii.gz \
  --te_ms 3.2 6.5 9.8 13.1 16.4 19.7 23.1 26.4 \
  --mag mag_multi_echo.nii.gz \
  --mask mask_multi_echo.nii.gz
```

### Non-axial acquisition

Provide B0 direction in image coordinates:

```bash
python run.py --echo_4d phase_4d.nii.gz --te_ms 4 8 12 \
  --mag mag.nii.gz --b0-dir 0.1 0.0 0.995
```

### Single-echo (3D phase, TE in **seconds**)

```bash
python run.py --phase ph.nii.gz --te 0.020 --mag mag.nii.gz --mask mask.nii.gz
```

### MATLAB inputs

```bash
python run.py \
  --data_dir Data/your_subject \
  --echo_files ph1.mat ph2.mat ph3.mat \
  --te_ms 4 8 12 \
  --mag mag.mat \
  --mask BET_mask.mat
```

`.mat` files (v5 or v7.3) must contain a single numeric array per file.

### YAML config

```bash
python run.py --config config.yaml
```

Example `config.yaml`:

```yaml
data_dir: demo
echo_4d: ph_multi_echo.nii.gz
te_ms: [3.2, 6.5, 9.8, 13.1, 16.4, 19.7, 23.1, 26.4]
mag: mag_multi_echo.nii.gz
mask: mask_multi_echo.nii.gz
b0: 3.0
b0_dir: null            # null → read from params.json or [0, 0, 1]
eroded_rad: 3
output: ./iqsm_plus_output
```

### Arguments at a glance

- **Input** (mutually exclusive): `--from_converted`, `--echo_files`, `--echo_4d`, `--phase`.
- **Echo times**: `--te_ms` (preferred, milliseconds) or `--te` (seconds, legacy).
- **Required**: a phase-input flag (above).
- **Optional**: `--mag` (omit → uniform magnitude / TE²-only weighting), `--mask` (or `--bet_mask`), `--voxel-size`, `--b0`, `--b0-dir`, `--eroded-rad`, `--reverse-phase-sign`.
- **Output**: `--output` (default `./iqsm_plus_output`).
- **Setup**: `--download-checkpoints`, `--download-demo`.
- `--data_dir` is **optional** (defaults to current working directory) — relative input paths are resolved against it.

`python run.py --help` lists everything.

---

## Run Demo Examples

After `python run.py --download-demo`, try iQSM+ in any of the following ways. The bundled demo is an 8-echo GRE acquisition (3 T, 1×1×1 mm isotropic).

### Option 1 — Web app

```bash
python app.py
```

Drop the demo files into the upload buttons (Phase + Magnitude), paste the TEs into Echo Times, add the mask. Hit **Run Reconstruction**.

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

## Troubleshooting

- **Web app upload fails with "Connection errored out / Load failed"** — usually a stale browser tab from before the server was restarted. Close all tabs from earlier sessions and open a fresh one.
- **SSH "channel N: open failed: connect failed: Connection refused"** — comes from your local SSH client, not Gradio. Browser is hitting the forwarded port before Gradio has bound it (race during startup), or a stale tab is polling. After `* Running on local URL` prints, refresh once. To silence the noise add `-q -o LogLevel=ERROR` to your `ssh` command.
- **Phase from real + imaginary looks "wrapped" every other slice** — that's the GE FFT-shift quirk. Re-run conversion with `--chopper on`.
- **Iron-rich deep grey matter appears dark in the QSM output** — flip the phase sign convention with `--reverse-phase-sign 1` (CLI) or the *Reverse phase sign* checkbox (web app).
- **Non-axial acquisition produces unrealistic susceptibility values** — make sure B0 direction is set. Use `--from_converted` (auto), or pass `--b0-dir x y z` (CLI), or fill the *B0 direction* field (web app).
- **Checkpoint download fails behind a firewall** — manually grab the two `.pth` files from [Hugging Face](https://huggingface.co/sunhongfu/iQSM_Plus/tree/main) and drop them in `checkpoints/`.

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
