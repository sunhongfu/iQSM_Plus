# iQSM+ – Orientation-Adaptive Quantitative Susceptibility Mapping

**Plug-and-Play Latent Feature Editing for Orientation-Adaptive Quantitative Susceptibility Mapping Neural Networks**

[MIA 2024](https://doi.org/10.1016/j.media.2024.103160) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2311.07823) &nbsp;|&nbsp; [HuggingFace](https://huggingface.co/sunhongfu/iQSM_Plus) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

iQSM+ extends iQSM to handle **arbitrary acquisition orientations** — not just axial scans. It uses orientation-adaptive latent feature editing (OA-LFE) blocks that learn the encoding of acquisition orientation vectors and integrate them into the network. It supports single-echo and multi-echo phase inputs.

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

  Includes all base dependencies plus Gradio and Matplotlib for the browser UI and slice previews.

- **Command-Line only** (lighter install, no web stack):

  ```bash
  pip install -r requirements.txt
  ```

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

Then open [http://localhost:7860](http://localhost:7860) in your browser.

### Usage

1. **Upload phase file** — select a phase NIfTI (`.nii` / `.nii.gz`) or DICOM series. For multi-echo data upload a 4D NIfTI. Echo times are auto-extracted from DICOM headers or the NIfTI description field.
2. **Set parameters** — verify echo time(s), voxel size, B0 field strength, and mask erosion radius. For non-axial acquisitions enter the B0 direction unit vector.
3. **Brain mask** — optionally upload a BET mask. If omitted, all voxels are processed.
4. **Run** — click **▶ Run Reconstruction**. Use the slice slider to browse the QSM result volume.
5. **Download** — output NIfTI files appear in the Results panel when complete. View in [FSLeyes](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes), [ITK-SNAP](http://www.itksnap.org/), or [3D Slicer](https://www.slicer.org/).

---

## Command-Line Interface

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

## Run Demo Examples

Once you have downloaded the demo data (`python run.py --download-demo`), you can try iQSM+ in any of the following ways. The demo is an 8-echo multi-echo GRE acquisition (3 T, 1×1×1 mm isotropic).

### Option 1 — Web app (one click)

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860), click **⬇ Load Demo Data** to pre-fill all inputs (phase, magnitude, mask, echo times), then click **▶ Run Reconstruction**.

### Option 2 — Command line (multi-echo)

```bash
python run.py \
  --phase demo/ph_multi_echo.nii.gz \
  --te 0.0032 0.0065 0.0098 0.0130 0.0163 0.0195 0.0228 0.0260 \
  --mag demo/mag_multi_echo.nii.gz \
  --mask demo/mask_multi_echo.nii.gz
```

Output (`QSM.nii.gz`) is written to the current directory by default. Use `--output ./my_output/` to redirect it.

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
