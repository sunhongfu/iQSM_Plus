"""
iQSM+ – Gradio Web Interface
=====================================
Clinician-friendly web UI for Quantitative Susceptibility Mapping (QSM).

Launch:
    python app.py                   # CPU
    python app.py --share           # public Gradio link
    python app.py --server-port 8080

Docker:
    docker compose up               # see docker-compose.yml
"""

import argparse
import os
import re
import tempfile
import traceback
import urllib.request

import gradio as gr
import nibabel as nib
import numpy as np

from inference import run_iqsm_plus


# ---------------------------------------------------------------------------
# Demo data – multi-echo in-vivo brain, 1×1×1 mm, B0=3T, 8 echoes
# ---------------------------------------------------------------------------
_DEMO_BASE = (
    "https://github.com/sunhongfu/iQSM_Plus/releases/download/v1.0-demo"
)
_DEMO_PHASE = f"{_DEMO_BASE}/ph_multi_echo.nii.gz"
_DEMO_MAG   = f"{_DEMO_BASE}/mag_multi_echo.nii.gz"
_DEMO_MASK  = f"{_DEMO_BASE}/mask_multi_echo.nii.gz"
_DEMO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "iqsm_plus_demo")

# Demo acquisition parameters
_DEMO_TE         = [0.0032, 0.0065, 0.0098, 0.0131, 0.0164, 0.0197, 0.0231, 0.0264]
_DEMO_B0         = 3.0
_DEMO_VOX        = "1 1 1"
_DEMO_B0DIR      = ""
_DEMO_ERODED_RAD = 0
_DEMO_PHASE_SIGN = False


def _download_demo() -> tuple[str, str, str]:
    """Download demo files (cached after first run). Returns (phase, mag, mask) paths."""
    os.makedirs(_DEMO_CACHE_DIR, exist_ok=True)
    phase_path = os.path.join(_DEMO_CACHE_DIR, "ph_multi_echo.nii.gz")
    mag_path   = os.path.join(_DEMO_CACHE_DIR, "mag_multi_echo.nii.gz")
    mask_path  = os.path.join(_DEMO_CACHE_DIR, "mask_multi_echo.nii.gz")

    for url, path in [
        (_DEMO_PHASE, phase_path),
        (_DEMO_MAG,   mag_path),
        (_DEMO_MASK,  mask_path),
    ]:
        if not os.path.exists(path):
            print(f"Downloading demo file: {url}")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:
                raise gr.Error(
                    f"Could not download demo data from GitHub Releases.\n{exc}\n\n"
                    "Please upload your own phase NIfTI file instead."
                )
    return phase_path, mag_path, mask_path


def load_and_run_demo(progress=gr.Progress(track_tqdm=True)):
    """Download demo data, fill all form fields, and run reconstruction."""
    def _progress(frac, msg):
        progress(frac, desc=msg)

    _progress(0.0, "Downloading demo data …")
    try:
        phase_path, mag_path, mask_path = _download_demo()
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc))

    output_dir = tempfile.mkdtemp(prefix="iqsm_plus_demo_")
    try:
        out_path = run_iqsm_plus(
            phase_nii_path=phase_path,
            te_values=_DEMO_TE,
            mag_nii_path=mag_path,
            mask_nii_path=mask_path,
            voxel_size=[1, 1, 1],
            b0_dir=None,
            b0=_DEMO_B0,
            eroded_rad=_DEMO_ERODED_RAD,
            phase_sign=-1,
            output_dir=output_dir,
            progress_fn=_progress,
        )
    except Exception:
        raise gr.Error("Demo reconstruction failed.\n\n" + traceback.format_exc())

    try:
        ax_img, cor_img, sag_img = _make_slice_figure(out_path)
    except Exception:
        ax_img = cor_img = sag_img = None

    te_str = ", ".join(f"{te:.4g}" for te in _DEMO_TE)
    demo_info = (
        f"Demo data cached at: {_DEMO_CACHE_DIR}\n"
        f"  Phase:  ph_multi_echo.nii.gz\n"
        f"  Mag:    mag_multi_echo.nii.gz\n"
        f"  Mask:   mask_multi_echo.nii.gz\n"
        "Parameters: 64×64×32 crop, 1×1×1 mm, 8 echoes, B0=3T"
    )
    status = "✅ Demo complete! Download QSM NIfTI below."

    return (
        # --- input field updates ---
        phase_path,                               # phase_file
        mag_path,                                 # mag_file
        mask_path,                                # mask_file
        te_str,                                   # te_str
        _DEMO_VOX,                                # voxel_str
        _DEMO_B0DIR,                              # b0dir_str
        _DEMO_B0,                                 # b0_val
        _DEMO_ERODED_RAD,                         # eroded_rad
        _DEMO_PHASE_SIGN,                         # negate_phase
        gr.update(value=demo_info, visible=True), # demo_info_box
        # --- output results ---
        status, out_path, ax_img, cor_img, sag_img,
    )


# ---------------------------------------------------------------------------
# Metadata extraction from uploaded NIfTI
# ---------------------------------------------------------------------------

def extract_nii_metadata(file_obj):
    """
    Called when a phase NIfTI is uploaded.
    Returns updates for: te_str, voxel_str, b0_val
    Auto-fills voxel size from the header; attempts to parse TE from descrip.
    """
    if file_obj is None:
        return gr.update(), gr.update(), gr.update()
    try:
        img = nib.load(file_obj.name)
        zooms = img.header.get_zooms()
        voxel_str = f"{zooms[0]:.4g} {zooms[1]:.4g} {zooms[2]:.4g}"

        te_update = gr.update()
        try:
            descrip = img.header.get("descrip", b"")
            if isinstance(descrip, (bytes, bytearray)):
                descrip = descrip.decode("utf-8", errors="ignore")
            descrip = descrip.strip()
            m = re.search(r"TE\s*=\s*([\d.]+)\s*(ms)?", descrip, re.IGNORECASE)
            if m:
                te_val = float(m.group(1))
                if m.group(2) and m.group(2).lower() == "ms":
                    te_val /= 1000.0
                te_update = gr.update(value=str(te_val))
        except Exception:
            pass

        return te_update, gr.update(value=voxel_str), gr.update()
    except Exception:
        return gr.update(), gr.update(), gr.update()


# ---------------------------------------------------------------------------
# DICOM helper callbacks
# ---------------------------------------------------------------------------

def _auto_fill_from_dicom(dcm_files):
    """
    Called when DICOM phase files are uploaded.
    Returns updates for: te_str, voxel_str, b0_val
    """
    if not dcm_files:
        return gr.update(), gr.update(), gr.update()
    try:
        from dicom_utils import extract_te_from_dicoms, extract_metadata_from_dicoms
        paths = [f.name for f in dcm_files]

        te_update  = gr.update()
        vox_update = gr.update()
        b0_update  = gr.update()

        te_values = extract_te_from_dicoms(paths)
        if te_values:
            te_update = gr.update(value=", ".join(f"{te:.6g}" for te in te_values))

        try:
            meta = extract_metadata_from_dicoms(paths)
            if meta.get("voxel_size"):
                v = meta["voxel_size"]
                vox_update = gr.update(value=f"{v[0]:.4g} {v[1]:.4g} {v[2]:.4g}")
            if meta.get("b0"):
                b0_update = gr.update(value=float(meta["b0"]))
        except Exception:
            pass

        return te_update, vox_update, b0_update
    except Exception:
        return gr.update(), gr.update(), gr.update()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_floats(text: str, name: str, n: int | None = None) -> list[float]:
    try:
        vals = [float(v) for v in text.replace(",", " ").split()]
    except ValueError:
        raise gr.Error(f"'{name}' must be numbers separated by spaces or commas.")
    if n is not None and len(vals) != n:
        raise gr.Error(f"'{name}' must have exactly {n} values, got {len(vals)}.")
    return vals


def _make_slice_figure(nii_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = nib.load(nii_path).get_fdata(dtype=np.float32)
    vmin, vmax = np.percentile(vol, [2, 98])
    vol_n = np.clip((vol - vmin) / max(vmax - vmin, 1e-6), 0, 1)

    slices = {
        "Axial":    vol_n[:, :, vol_n.shape[2] // 2].T,
        "Coronal":  vol_n[:, vol_n.shape[1] // 2, :].T,
        "Sagittal": vol_n[vol_n.shape[0] // 2, :, :].T,
    }

    imgs = []
    for title, sl in slices.items():
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        imgs.append(buf[:, :, :3].copy())
        plt.close(fig)

    return imgs[0], imgs[1], imgs[2]


# ---------------------------------------------------------------------------
# Core reconstruction callback
# ---------------------------------------------------------------------------

def reconstruct(
    input_mode,
    phase_file,
    mag_file,
    phase_dcm,
    mag_dcm,
    te_str,
    mask_file,
    voxel_str,
    b0dir_str,
    b0_val,
    eroded_rad,
    negate_phase,
    progress=gr.Progress(track_tqdm=True),
):
    dcm_tmp = None

    if input_mode == "DICOM series":
        if not phase_dcm:
            raise gr.Error("Please upload phase DICOM files.")
        try:
            from dicom_utils import dicoms_to_nifti
        except ImportError as exc:
            raise gr.Error(str(exc))

        dcm_tmp = tempfile.mkdtemp(prefix="iqsm_dcm_in_")

        try:
            phase_nii_path, dcm_te = dicoms_to_nifti(
                [f.name for f in phase_dcm], dcm_tmp, label="phase"
            )
        except Exception as exc:
            raise gr.Error(f"Failed to read phase DICOM files: {exc}")

        mag_nii_path = None
        if mag_dcm:
            try:
                mag_nii_path, _ = dicoms_to_nifti(
                    [f.name for f in mag_dcm], dcm_tmp, label="mag"
                )
            except Exception as exc:
                raise gr.Error(f"Failed to read magnitude DICOM files: {exc}")

        if te_str.strip():
            te_values = _parse_floats(te_str, "Echo time(s) (TE)")
        else:
            te_values = dcm_te
            if not te_values:
                raise gr.Error(
                    "Could not extract echo times from the DICOM files. "
                    "Please enter them manually in the TE field."
                )
    else:
        if phase_file is None:
            raise gr.Error("Please upload a phase NIfTI file.")
        if not te_str.strip():
            raise gr.Error("Please enter at least one echo time (TE).")
        phase_nii_path = phase_file if isinstance(phase_file, str) else phase_file.name
        mag_nii_path   = (mag_file if isinstance(mag_file, str) else mag_file.name) if mag_file else None
        te_values      = _parse_floats(te_str, "Echo time(s) (TE)")

    if any(t <= 0 for t in te_values):
        raise gr.Error("Echo times must be positive. Enter values in seconds (e.g. 0.020).")

    voxel_size = None
    if voxel_str.strip():
        voxel_size = _parse_floats(voxel_str, "Voxel size", n=3)
        if any(v <= 0 for v in voxel_size):
            raise gr.Error("Voxel sizes must be positive.")

    b0_dir = None
    if b0dir_str.strip():
        b0_dir = _parse_floats(b0dir_str, "B0 direction", n=3)
        if np.linalg.norm(b0_dir) == 0:
            raise gr.Error("B0 direction must not be the zero vector.")

    phase_sign = 1 if negate_phase else -1
    output_dir = tempfile.mkdtemp(prefix="iqsm_plus_out_")

    def _progress(frac, msg):
        progress(frac, desc=msg)

    try:
        out_path = run_iqsm_plus(
            phase_nii_path=phase_nii_path,
            te_values=te_values,
            mag_nii_path=mag_nii_path,
            mask_nii_path=(mask_file if isinstance(mask_file, str) else mask_file.name) if mask_file else None,
            voxel_size=voxel_size,
            b0_dir=b0_dir,
            b0=float(b0_val),
            eroded_rad=int(eroded_rad),
            phase_sign=phase_sign,
            output_dir=output_dir,
            progress_fn=_progress,
        )
    except Exception:
        raise gr.Error(
            "Reconstruction failed. Check the log for details.\n\n"
            + traceback.format_exc()
        )

    try:
        ax_img, cor_img, sag_img = _make_slice_figure(out_path)
    except Exception:
        ax_img = cor_img = sag_img = None

    status = (
        "✅ Reconstruction complete! "
        "Use the Download button below to save the QSM NIfTI."
    )
    return status, out_path, ax_img, cor_img, sag_img


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

TITLE = "iQSM+ – QSM Reconstruction"
DESCRIPTION = """
**Quantitative Susceptibility Mapping (QSM)** from MRI phase data
using the *iQSM+* deep learning model ([paper](https://doi.org/10.1016/j.media.2024.103160)).

**Quick-start:**
1. Upload phase data as **NIfTI** or select the **DICOM** tab.
2. Voxel size and TE are auto-filled from the file — verify before running.
3. Click **▶ Run Reconstruction**. Or try **⚡ Run demo** instantly.
"""

HELP_TE    = "Echo time(s) in **seconds**. Single: `0.020`. Multi: `0.004, 0.008, 0.012`. Auto-filled from DICOM or NIfTI header."
HELP_VOX   = "Voxel size in mm (x y z). Auto-filled from file header. Override if needed."
HELP_B0DIR = "B0 field direction vector. Leave blank for default `0 0 1` (axial)."
HELP_NEGATE = "Tick if QSM looks inverted (veins appear bright). Flips the phase sign convention."


def build_ui():
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Phase & magnitude input")

                input_tabs = gr.Tabs()
                with input_tabs:
                    with gr.Tab("NIfTI files"):
                        phase_file = gr.File(
                            label="Phase NIfTI (.nii / .nii.gz) — 3D single-echo or 4D multi-echo",
                            file_types=[".nii", ".gz"],
                        )
                        mag_file = gr.File(
                            label="Magnitude NIfTI (optional)",
                            file_types=[".nii", ".gz"],
                        )

                    with gr.Tab("DICOM series"):
                        phase_dcm = gr.File(
                            label="Phase DICOM files (select all slices / all echoes)",
                            file_count="multiple",
                            file_types=[".dcm", ".ima", "."],
                        )
                        mag_dcm = gr.File(
                            label="Magnitude DICOM files (optional)",
                            file_count="multiple",
                            file_types=[".dcm", ".ima", "."],
                        )

                input_mode = gr.State("NIfTI files")
                input_tabs.change(
                    fn=lambda tab: tab,
                    inputs=[input_tabs],
                    outputs=[input_mode],
                )

                gr.Markdown("### Echo time(s)")
                te_str = gr.Textbox(
                    label="TE (seconds) — auto-filled from file",
                    placeholder="e.g.  0.020   or   0.004, 0.008, 0.012",
                )
                gr.Markdown(f"<small>{HELP_TE}</small>")

                negate_phase = gr.Checkbox(
                    label="Reverse phase sign (opposite scanner convention)",
                    value=False,
                )
                gr.Markdown(f"<small>{HELP_NEGATE}</small>")

                gr.Markdown("### Optional inputs")
                mask_file = gr.File(
                    label="Brain mask NIfTI (optional)",
                    file_types=[".nii", ".gz"],
                )

                gr.Markdown("### Acquisition parameters")
                with gr.Row():
                    b0_val = gr.Number(
                        label="B0 field strength (Tesla)",
                        value=3.0,
                        minimum=0.1,
                        maximum=14.0,
                        step=0.5,
                    )
                    eroded_rad = gr.Slider(
                        label="Mask erosion radius (voxels)",
                        minimum=0,
                        maximum=10,
                        step=1,
                        value=3,
                    )

                voxel_str = gr.Textbox(
                    label="Voxel size (mm) — auto-filled from file header",
                    placeholder="e.g.  1 1 2",
                )
                gr.Markdown(f"<small>{HELP_VOX}</small>")

                b0dir_str = gr.Textbox(
                    label="B0 direction override (optional)",
                    placeholder="e.g.  0 0 1",
                )
                gr.Markdown(f"<small>{HELP_B0DIR}</small>")

                with gr.Row():
                    run_btn  = gr.Button("▶ Run Reconstruction", variant="primary", size="lg")
                    demo_btn = gr.Button("⚡ Run demo", variant="secondary", size="lg")

                demo_info_box = gr.Textbox(
                    label="Demo data info",
                    lines=5,
                    interactive=False,
                    visible=False,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Results")
                status_box = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False,
                    placeholder="Reconstruction output will appear here …",
                )
                download_file = gr.File(label="⬇ Download QSM NIfTI")

                gr.Markdown("#### Preview (middle slice)")
                with gr.Row():
                    axial_img    = gr.Image(label="Axial",    show_label=True)
                    coronal_img  = gr.Image(label="Coronal",  show_label=True)
                    sagittal_img = gr.Image(label="Sagittal", show_label=True)

        # Auto-fill from NIfTI upload
        phase_file.change(
            fn=extract_nii_metadata,
            inputs=[phase_file],
            outputs=[te_str, voxel_str, b0_val],
        )

        # Auto-fill TE, voxel size, B0 from DICOM upload
        phase_dcm.upload(
            fn=_auto_fill_from_dicom,
            inputs=[phase_dcm],
            outputs=[te_str, voxel_str, b0_val],
        )

        _run_outputs  = [status_box, download_file, axial_img, coronal_img, sagittal_img]
        _demo_outputs = [
            phase_file, mag_file, mask_file,
            te_str, voxel_str, b0dir_str, b0_val, eroded_rad, negate_phase,
            demo_info_box,
        ] + _run_outputs

        run_btn.click(
            fn=reconstruct,
            inputs=[
                input_mode,
                phase_file, mag_file,
                phase_dcm, mag_dcm,
                te_str, mask_file,
                voxel_str, b0dir_str,
                b0_val, eroded_rad, negate_phase,
            ],
            outputs=_run_outputs,
        )

        demo_btn.click(
            fn=load_and_run_demo,
            inputs=[],
            outputs=_demo_outputs,
        )

        gr.Markdown(
            "---\n"
            "**Citation:** Gao Y, et al. *Plug-and-Play latent feature editing for "
            "orientation-adaptive QSM neural networks.* "
            "Medical Image Analysis, 2024. "
            "[doi:10.1016/j.media.2024.103160](https://doi.org/10.1016/j.media.2024.103160)\n\n"
            "**Source code:** [github.com/sunhongfu/iQSM_Plus](https://github.com/sunhongfu/iQSM_Plus)"
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iQSM+ Gradio server")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        theme=gr.themes.Soft(),
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        allowed_paths=[tempfile.gettempdir()],
    )
