"""
iQSM+ – Gradio Web Interface

Launch:
    python app.py                   # CPU
    python app.py --server-port 8080

Docker:
    docker compose up               # see docker-compose.yml
"""

import argparse
import os
import re
import tempfile
import traceback

import gradio as gr
import nibabel as nib
import numpy as np

from inference import run_iqsm_plus, CheckpointNotFoundError


# ---------------------------------------------------------------------------
# Demo data – multi-echo in-vivo brain, 1×1×1 mm, B0=3T, 8 echoes
# ---------------------------------------------------------------------------
_HF_REPO  = "sunhongfu/iQSM_Plus"
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.join(_HERE, "demo")

_DEMO_FILES = [
    "ph_multi_echo.nii.gz",
    "mag_multi_echo.nii.gz",
    "mask_multi_echo.nii.gz",
    "params.json",
]

_DEMO_NOT_FOUND_MSG = (
    "Demo data not found in demo/.\n\n"
    "Run this on the host before starting Docker:\n\n"
    "    python run.py --download-demo\n\n"
    "Then restart the app or reload this page."
)


def _load_demo_files() -> tuple[str, str, str, dict]:
    """Load demo NIfTIs + params.json from local demo/ folder."""
    import json
    missing = [f for f in _DEMO_FILES
               if not os.path.exists(os.path.join(_DEMO_DIR, f))]
    if missing:
        raise gr.Error(_DEMO_NOT_FOUND_MSG)
    phase_path = os.path.join(_DEMO_DIR, "ph_multi_echo.nii.gz")
    mag_path   = os.path.join(_DEMO_DIR, "mag_multi_echo.nii.gz")
    mask_path  = os.path.join(_DEMO_DIR, "mask_multi_echo.nii.gz")
    with open(os.path.join(_DEMO_DIR, "params.json")) as f:
        params = json.load(f)
    return phase_path, mag_path, mask_path, params


def load_demo_data(progress=gr.Progress(track_tqdm=True)):
    """Load demo files and populate all input fields. Does not run reconstruction."""
    try:
        phase_path, mag_path, mask_path, params = _load_demo_files()
    except Exception as exc:
        raise gr.Error(str(exc))

    te      = params["TE_seconds"]
    te_str  = str(te) if isinstance(te, (int, float)) else ", ".join(f"{v:.4g}" for v in te)
    vox     = params["voxel_size_mm"]
    vox_str = " ".join(f"{v:.4g}" for v in vox)
    b0      = params["B0_Tesla"]
    eroded  = params.get("eroded_rad", 3)
    negate  = params["phase_sign_convention"] == 1
    mat     = params.get("matrix_size", "")
    mat_str = "×".join(str(x) for x in mat) if mat else ""
    n_echo  = params.get("num_echoes", "")

    demo_info = (
        f"demo/ph_multi_echo.nii.gz    (phase, 4D)\n"
        f"demo/mag_multi_echo.nii.gz   (magnitude, 4D)\n"
        f"demo/mask_multi_echo.nii.gz  (mask)\n"
        f"Matrix: {mat_str} · Voxel: {vox_str} mm · {n_echo} echoes · B0: {b0} T\n"
        f"Ready — click ▶ Run Reconstruction to proceed."
    )

    return (
        phase_path, mag_path, mask_path,
        te_str, vox_str, "",
        b0, eroded, negate,
        gr.update(value=demo_info, visible=True),
    )


# ---------------------------------------------------------------------------
# Metadata extraction from uploaded NIfTI
# ---------------------------------------------------------------------------

def extract_nii_metadata(file_obj):
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
            m = re.search(r"TE\s*=\s*([\d.]+)\s*(ms)?", descrip.strip(), re.IGNORECASE)
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


_DISPLAY_VMIN = -0.2   # ppm
_DISPLAY_VMAX =  0.2   # ppm


def _make_slice_figure(nii_path: str, vmin: float, vmax: float):
    """Render axial/coronal/sagittal middle slices; return list of (path, caption) for Gallery."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = nib.load(nii_path).get_fdata(dtype=np.float32)

    slices = [
        (vol[:, :, vol.shape[2] // 2].T, "Axial"),
        (vol[:, vol.shape[1] // 2, :].T, "Coronal"),
        (vol[vol.shape[0] // 2, :, :].T, "Sagittal"),
    ]

    out_dir = tempfile.mkdtemp(prefix="iqsm_preview_")
    result = []
    for i, (sl, caption) in enumerate(slices):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
        ax.axis("off")
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        path = os.path.join(out_dir, f"slice_{i}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0,
                    facecolor="#111827")
        plt.close(fig)
        result.append((path, caption))

    return result


# ---------------------------------------------------------------------------
# Core reconstruction callback
# ---------------------------------------------------------------------------

def reconstruct(
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

    if phase_dcm:
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
        raise gr.Error("Echo times must be positive (enter seconds, e.g. 0.020).")

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
    except CheckpointNotFoundError as exc:
        return str(exc), None, []
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return "Reconstruction failed — check the terminal / Docker log for the full error.", None, []

    try:
        gallery_data = _make_slice_figure(out_path, _DISPLAY_VMIN, _DISPLAY_VMAX)
    except Exception:
        gallery_data = []

    status = "✅ Done — download QSM file below."
    return status, out_path, gallery_data


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_CSS = """
/* ── Typography ──────────────────────────────────────────────── */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter",
                 Roboto, "Helvetica Neue", Arial, sans-serif !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
}

/* ── App header ──────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0c2340 0%, #1a4f8a 55%, #2471b5 100%);
    border-radius: 10px;
    padding: 22px 28px;
    margin-bottom: 4px;
}
.app-header h1 {
    color: #ffffff !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    margin: 0 0 5px 0 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.2 !important;
}
.app-header p {
    color: rgba(255,255,255,0.72) !important;
    font-size: 0.875rem !important;
    margin: 0 !important;
    line-height: 1.55 !important;
}
.app-header a { color: #93c5fd !important; text-decoration: none; }
.app-header a:hover { text-decoration: underline !important; }

/* ── Section labels ──────────────────────────────────────────── */
.sec-label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
    margin: 0 0 6px 0 !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #e2e8f0 !important;
    display: block !important;
}

/* ── Action buttons ──────────────────────────────────────────── */
#run-btn > button {
    font-size: 0.975rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    height: 52px !important;
    border-radius: 8px !important;
}
#demo-btn > button {
    height: 52px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ── Demo info box ────────────────────────────────────────────── */
#demo-info textarea {
    background: #f0f9ff !important;
    border-color: #bae6fd !important;
    font-size: 0.82rem !important;
    font-family: ui-monospace, "Cascadia Code", "Fira Code", monospace !important;
    color: #0c4a6e !important;
}

/* ── Status box ──────────────────────────────────────────────── */
#status-box textarea {
    font-size: 0.875rem !important;
    color: #1e293b !important;
}

/* ── Preview images ──────────────────────────────────────────── */
#preview-row .image-container { border-radius: 6px !important; }

/* ── Footer ──────────────────────────────────────────────────── */
.app-footer {
    font-size: 0.775rem !important;
    color: #94a3b8 !important;
    text-align: center !important;
    padding: 12px 0 2px 0 !important;
    border-top: 1px solid #e2e8f0 !important;
    margin-top: 4px !important;
    line-height: 1.6 !important;
}
.app-footer a { color: #64748b !important; text-decoration: none; }
.app-footer a:hover { text-decoration: underline !important; }

/* ── Theme toggle button ─────────────────────────────────────── */
.app-header { position: relative !important; }
#theme-toggle {
    position: absolute !important;
    top: 16px !important;
    right: 16px !important;
    background: rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
#theme-toggle:hover { background: rgba(255,255,255,0.28) !important; }

/* ── Hide Gradio share button ─────────────────────────────────── */
.share-button { display: none !important; }

/* ── Gallery lightbox: hide close (X) button while in browser fullscreen ──
   Clicking X while the browser is in fullscreen leaves the page stuck in
   fullscreen mode. Users must press Escape or click exit-fullscreen instead. */
:fullscreen button[aria-label="Close"],
:fullscreen button[aria-label="close"],
:fullscreen .close-btn,
:fullscreen .close-button,
:-webkit-full-screen button[aria-label="Close"],
:-webkit-full-screen button[aria-label="close"],
:-webkit-full-screen .close-btn,
:-webkit-full-screen .close-button { display: none !important; }
"""

_THEME = gr.themes.Default(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
)

_JS = """
function() {
    const key = 'iqsm-theme';
    const saved = localStorage.getItem(key) || 'light';
    if (saved === 'dark') {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
    function updateBtn(theme) {
        const btn = document.getElementById('theme-toggle');
        if (btn) btn.textContent = theme === 'dark' ? '\u2600 Light mode' : '\u263d Dark mode';
    }
    updateBtn(saved);
    document.addEventListener('click', function(e) {
        if (e.target && e.target.id === 'theme-toggle') {
            const next = document.documentElement.classList.contains('dark') ? 'light' : 'dark';
            document.documentElement.classList.toggle('dark', next === 'dark');
            localStorage.setItem(key, next);
            updateBtn(next);
        }
    });
}
"""

TITLE = "iQSM+ — Quantitative Susceptibility Mapping"


def build_ui():
    with gr.Blocks(title=TITLE, js=_JS) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-header">
          <button id="theme-toggle">&#x263D; Dark mode</button>
          <h1>iQSM+ &mdash; Quantitative Susceptibility Mapping</h1>
          <p>
            Deep learning QSM reconstruction from single- or multi-echo MRI phase
            (<a href="https://doi.org/10.1016/j.media.2024.103160">Gao et al., MedIA 2024</a>).
            Upload NIfTI or DICOM data, verify parameters, and reconstruct.
            New here? Click <strong style="color:#bfdbfe">⬇ Load Demo Data</strong> to prefill all fields, then click Run.
          </p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left column: Inputs ──────────────────────────────────────
            with gr.Column(scale=5, min_width=340):

                gr.HTML('<p class="sec-label">Input</p>')

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

                te_str = gr.Textbox(
                    label="Echo time(s) — TE (seconds)",
                    placeholder="Single: 0.020   Multi: 0.004, 0.008, 0.012",
                    info="Auto-filled from file header. Separate multiple echoes with commas or spaces.",
                )
                negate_phase = gr.Checkbox(
                    label="Reverse phase sign",
                    value=False,
                    info="Enable if using the opposite scanner convention "
                         "(veins appear bright in the QSM output).",
                )

                mask_file = gr.File(
                    label="Brain mask NIfTI (optional — full volume used if omitted)",
                    file_types=[".nii", ".gz"],
                )
                with gr.Row():
                    b0_val = gr.Number(
                        label="B0 field strength (Tesla)",
                        value=3.0, minimum=0.1, maximum=14.0, step=0.5,
                    )
                    eroded_rad = gr.Slider(
                        label="Mask erosion (voxels)",
                        minimum=0, maximum=10, step=1, value=3,
                    )
                voxel_str = gr.Textbox(
                    label="Voxel size — x y z (mm)",
                    placeholder="e.g.  1 1 2",
                    info="Auto-filled from file header. Override if the values look wrong.",
                )
                b0dir_str = gr.Textbox(
                    label="B0 direction override (optional)",
                    placeholder="e.g.  0 0 1",
                    info="Unit vector of B0 field direction. Leave blank to use [0 0 1] (axial).",
                )

                with gr.Row():
                    run_btn  = gr.Button(
                        "▶  Run Reconstruction", variant="primary",
                        size="lg", elem_id="run-btn", scale=3,
                    )
                    demo_btn = gr.Button(
                        "⬇  Load Demo Data", variant="secondary",
                        size="lg", elem_id="demo-btn", scale=1,
                    )

                demo_info_box = gr.Textbox(
                    label="Demo dataset",
                    lines=5, interactive=False, visible=False, elem_id="demo-info",
                )

            # ── Right column: Results ────────────────────────────────────
            with gr.Column(scale=5, min_width=340):

                gr.HTML('<p class="sec-label">Results</p>')

                status_box = gr.Textbox(
                    label="Status",
                    lines=4, interactive=False,
                    placeholder="Results will appear here after reconstruction …",
                    elem_id="status-box",
                )
                download_file = gr.File(label="QSM — susceptibility map (.nii.gz)")

                gr.HTML('<p class="sec-label" style="margin-top:14px">Preview — QSM middle slices</p>')
                qsm_gallery = gr.Gallery(
                    columns=3, rows=1, height=230,
                    object_fit="contain", show_label=False,
                    elem_id="preview-row",
                )

        # ── Footer ──────────────────────────────────────────────────────
        gr.HTML("""
        <div class="app-footer">
          Gao Y, et al. <em>Plug-and-Play latent feature editing for orientation-adaptive
          QSM neural networks.</em> Med Image Anal, 2024.
          <a href="https://doi.org/10.1016/j.media.2024.103160">doi:10.1016/j.media.2024.103160</a>
          &nbsp;·&nbsp;
          <a href="https://github.com/sunhongfu/iQSM_Plus">github.com/sunhongfu/iQSM_Plus</a>
        </div>
        """)

        # ── Wiring ───────────────────────────────────────────────────────
        phase_file.change(
            fn=extract_nii_metadata,
            inputs=[phase_file],
            outputs=[te_str, voxel_str, b0_val],
        )
        phase_dcm.upload(
            fn=_auto_fill_from_dicom,
            inputs=[phase_dcm],
            outputs=[te_str, voxel_str, b0_val],
        )

        _run_outputs  = [status_box, download_file, qsm_gallery]
        _demo_outputs = [
            phase_file, mag_file, mask_file,
            te_str, voxel_str, b0dir_str, b0_val, eroded_rad, negate_phase,
            demo_info_box,
        ]

        run_btn.click(
            fn=reconstruct,
            inputs=[
                phase_file, mag_file,
                phase_dcm, mag_dcm,
                te_str, mask_file,
                voxel_str, b0dir_str,
                b0_val, eroded_rad, negate_phase,
            ],
            outputs=_run_outputs,
        )
        demo_btn.click(
            fn=load_demo_data,
            inputs=[],
            outputs=_demo_outputs,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iQSM+ Gradio server")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        theme=_THEME,
        css=_CSS,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        allowed_paths=[tempfile.gettempdir(), _DEMO_DIR],
    )
