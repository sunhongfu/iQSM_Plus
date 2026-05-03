"""
iQSM+ — Gradio web app for orientation-adaptive Quantitative Susceptibility Mapping.

Layout mirrors DeepRelaxo's web app:
  - DICOM Folder tab (recommended) + NIfTI / MAT files tab
  - Processing Order panel with per-file shape verification
  - Echo Times dual-format input (comma list OR `first:spacing:count`)
  - Optional magnitude (improves multi-echo fitting in iQSM+ inference)
  - Optional brain mask with shape comparison
  - Acquisition + Hyper-parameters (B0, B0 direction, voxel size, mask erosion)
  - Run Pipeline → Log / Results / Visualisation panels
  - Slice slider, dark-mode auto-open, port auto-fallback, GPU cleanup
"""

import re
import sys
import queue
import shutil
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from inference import run_iqsm_plus, CheckpointNotFoundError
from data_utils import (
    load_array_with_affine,
    file_shape,
    shape_summary,
    load_dicom_qsm_folder,
)

_pipeline_lock = threading.Lock()

_DEMO_DIR = REPO_ROOT / "demo"

_QSM_VMIN = -0.2  # ppm
_QSM_VMAX =  0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _QueueWriter:
    def __init__(self, q, orig):
        self._q = q
        self._orig = orig

    def write(self, text):
        if text.strip():
            self._q.put(text.rstrip())
        if self._orig:
            self._orig.write(text)
        return len(text)

    def flush(self):
        if self._orig:
            self._orig.flush()

    def isatty(self):
        return False


_RED_WAIT = (
    "<span style='color: #dc2626; font-weight: 700; font-size: 1.05em;'>{msg}</span>"
)


def _parse_te_input(s):
    """Accept either 'TE1, TE2, TE3, ...' (ms) or compact 'first:spacing:count' (ms)."""
    s = (s or "").strip()
    if not s:
        return []
    if ":" in s and "," not in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "TE compact form must be 'first_TE : spacing : count' "
                f"(got {len(parts)} parts in '{s}')"
            )
        try:
            first = float(parts[0])
            spacing = float(parts[1])
            n = int(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"Invalid number in TE compact form '{s}'. "
                "Use 'first_TE : spacing : count' (e.g. '4 : 4 : 8')."
            ) from exc
        if n < 1:
            raise ValueError(f"TE count must be ≥ 1 (got {n})")
        return [round(first + i * spacing, 6) for i in range(n)]
    return [float(t.strip()) for t in s.replace(",", " ").split() if t.strip()]


def _to_path(f):
    if f is None:
        return None
    if isinstance(f, str):
        return Path(f)
    if hasattr(f, "path"):
        return Path(f.path)
    if hasattr(f, "name"):
        return Path(f.name)
    return Path(str(f))


def _natural_key(f):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", _to_path(f).name)]


def _sort_paths(paths):
    return sorted(paths, key=lambda p: _natural_key(p))


def _detect_echoes_from_paths(paths):
    """If a single 4D NIfTI/MAT is uploaded, return its 4th-dim length; otherwise len(paths)."""
    if not paths:
        return 0
    paths = [p for p in paths if p]
    if len(paths) == 1:
        p = Path(paths[0])
        name = p.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            try:
                shape = nib.load(str(p)).shape
                if len(shape) == 4:
                    return int(shape[-1])
            except Exception:
                pass
        elif name.endswith(".mat"):
            try:
                arr, _ = load_array_with_affine(p)
                if arr.ndim == 4:
                    return int(arr.shape[-1])
            except Exception:
                pass
        return 1
    return len(paths)


# ---------------------------------------------------------------------------
# Slice-image rendering
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _volume_array(nii_path):
    return nib.load(str(nii_path)).get_fdata().astype(np.float32)


def _make_slice_image(nii_path, slice_idx=None, vmin=-0.2, vmax=0.2):
    if not nii_path:
        return None
    data = _volume_array(str(nii_path))
    depth = data.shape[2]
    if slice_idx is None:
        slice_idx = depth // 2
    slice_idx = max(0, min(int(slice_idx), depth - 1))
    sl = np.rot90(data[:, :, slice_idx])
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.imshow(sl, cmap="gray", vmin=float(vmin), vmax=float(vmax), aspect="equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, bbox_inches="tight", pad_inches=0, dpi=150, facecolor="black")
    plt.close(fig)
    return tmp.name


# ---------------------------------------------------------------------------
# RUN CONFIGURATION log block
# ---------------------------------------------------------------------------

def _print_run_config(work_dir, mode, phase_paths, te_list_ms, mag_path, mask_path,
                      voxel_size, b0, b0_dir, eroded_rad, phase_sign):
    print("============================")
    print("RUN CONFIGURATION")
    print("============================")
    if mode == "4d":
        print(f"Input mode      : Single 4D phase ({len(te_list_ms)} echoes)")
        print(f"Phase file      : {Path(phase_paths[0]).name}")
    elif len(phase_paths) == 1:
        print(f"Input mode      : Single 3D phase")
        print(f"Phase file      : {Path(phase_paths[0]).name}")
    else:
        print(f"Input mode      : {len(phase_paths)} 3D phase echoes")
        for i, (p, te) in enumerate(zip(phase_paths, te_list_ms), 1):
            print(f"  Echo {i}: {Path(p).name}    TE = {te} ms")
    print(f"TE values (ms)  : {', '.join(str(t) for t in te_list_ms)}")
    print(f"Magnitude       : {Path(mag_path).name if mag_path else '(none)'}")
    print(f"Brain mask      : {Path(mask_path).name if mask_path else '(none)'}")
    if voxel_size:
        print(f"Voxel size (mm) : {' '.join(f'{v:.4g}' for v in voxel_size)}")
    else:
        print(f"Voxel size (mm) : (from NIfTI header)")
    print(f"B0 (T)          : {b0}")
    print(f"B0 direction    : {b0_dir if b0_dir else '[0 0 1] (default)'}")
    print(f"Mask erosion    : {eroded_rad} voxels")
    print(f"Reverse phase   : {'yes' if phase_sign == 1 else 'no'}")
    print(f"Working dir     : {work_dir}")
    cmd = ["python run.py"]
    cmd.append(f"--data_dir {work_dir}")
    if mode == "4d":
        cmd.append(f"--echo_4d {Path(phase_paths[0]).name}")
    elif len(phase_paths) == 1:
        cmd.append(f"--phase {Path(phase_paths[0]).name}")
    else:
        cmd.append("--echo_files " + " ".join(Path(p).name for p in phase_paths))
    cmd.append("--te_ms " + " ".join(str(t) for t in te_list_ms))
    if mag_path:
        cmd.append(f"--mag {Path(mag_path).name}")
    if mask_path:
        cmd.append(f"--mask {Path(mask_path).name}")
    if voxel_size:
        cmd.append("--voxel-size " + " ".join(f"{v:.4g}" for v in voxel_size))
    if b0_dir:
        cmd.append("--b0-dir " + " ".join(f"{v:.4g}" for v in b0_dir))
    if b0 != 3.0:
        cmd.append(f"--b0 {b0}")
    if eroded_rad != 3:
        cmd.append(f"--eroded-rad {eroded_rad}")
    if phase_sign == 1:
        cmd.append("--reverse-phase-sign 1")
    sep = "-" * 56
    print()
    print(sep)
    print("  Equivalent command-line invocation:")
    print(sep)
    print("  " + " \\\n      ".join(cmd))
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Background inference thread
# ---------------------------------------------------------------------------

def _gpu_cleanup():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _run_thread(job, work_dir, mode, phase_paths, te_list_ms, mag_path, mask_path,
                voxel_size, b0, b0_dir, eroded_rad, phase_sign, vmin, vmax):
    log_q = job["log_queue"]
    orig = sys.stdout
    sys.stdout = _QueueWriter(log_q, orig)
    try:
        with _pipeline_lock:
            job["status"] = "running"
            te_list_s = [t / 1000.0 for t in te_list_ms]

            _print_run_config(work_dir, mode, phase_paths, te_list_ms, mag_path,
                              mask_path, voxel_size, b0, b0_dir, eroded_rad, phase_sign)

            # iQSM+ accepts a single 3D NIfTI (single-echo) or 4D NIfTI (multi-echo).
            # If the user supplied per-echo 3D files, stack them into a 4D volume.
            if mode == "multi" and len(phase_paths) > 1:
                print("============================")
                print(f"COMBINING {len(phase_paths)} 3D ECHOES → 4D VOLUME")
                print("============================")
                imgs = [nib.load(str(p)) for p in phase_paths]
                affine = imgs[0].affine
                data_4d = np.stack(
                    [img.get_fdata(dtype=np.float32) for img in imgs], axis=-1,
                )
                combined = work_dir / "phase_4d.nii.gz"
                nib.save(nib.Nifti1Image(data_4d, affine), str(combined))
                phase_input_path = str(combined)
                print(f"  Combined shape: {data_4d.shape}")
            else:
                phase_input_path = str(phase_paths[0])

            out_dir = work_dir / "iqsmplus_output"
            out_dir.mkdir(exist_ok=True)

            print("============================")
            print("RECONSTRUCTION")
            print("============================")
            qsm_path = run_iqsm_plus(
                phase_nii_path=phase_input_path,
                te_values=te_list_s,
                mag_nii_path=mag_path,
                mask_nii_path=mask_path,
                voxel_size=voxel_size,
                b0_dir=b0_dir,
                b0=float(b0),
                eroded_rad=int(eroded_rad),
                phase_sign=phase_sign,
                output_dir=str(out_dir),
            )

            print("\n✅ Pipeline complete!")
            job["status"] = "done"
            job["qsm_path"] = qsm_path
            job["depth"] = _volume_array(qsm_path).shape[2]
            job["qsm_image"] = _make_slice_image(qsm_path, vmin=vmin, vmax=vmax)
    except CheckpointNotFoundError as exc:
        print(f"\n❌ {exc}")
        job["status"] = "error"
    except Exception as exc:
        import traceback
        print(f"\n❌ Error: {exc}")
        print(traceback.format_exc())
        job["status"] = "error"
    finally:
        _gpu_cleanup()
        sys.stdout = orig
        log_q.put(None)


def _result_files(job):
    files = [p for p in (job.get("qsm_path"),) if p]
    return files or None


def _result_info_md(job):
    files = _result_files(job)
    return shape_summary(files) if files else ""


def _state_and_slider_update(job):
    state = (job.get("qsm_path"),)
    depth = job.get("depth")
    if depth and not job.get("_slider_init"):
        job["_slider_init"] = True
        slider = gr.update(visible=True, minimum=0, maximum=depth - 1,
                           value=depth // 2, interactive=True)
    else:
        slider = gr.update()
    return state, slider


def _visibility_updates(job):
    return (
        gr.update(visible=True),                             # log_group
        gr.update(visible=bool(_result_files(job))),         # results_group
        gr.update(visible=bool(job.get("qsm_image"))),       # viz_group
    )


def _stream_job(job):
    log = ""
    while True:
        msg = job["log_queue"].get()
        if msg is None:
            break
        log += msg + "\n"
        state, slider = _state_and_slider_update(job)
        log_v, res_v, viz_v = _visibility_updates(job)
        yield (log, _result_files(job), _result_info_md(job),
               job.get("qsm_image"), state, slider, log_v, res_v, viz_v)
    state, slider = _state_and_slider_update(job)
    log_v, res_v, viz_v = _visibility_updates(job)
    yield (log, _result_files(job), _result_info_md(job),
           job.get("qsm_image"), state, slider, log_v, res_v, viz_v)


# ---------------------------------------------------------------------------
# Pipeline callback
# ---------------------------------------------------------------------------

def run_pipeline(phase_files, te_ms_str, mag_file, mask_file,
                 voxel_str, b0_val, b0dir_str, eroded_rad, negate_phase,
                 vmin, vmax):
    _noop = (
        None, "", None, (None,), gr.update(),
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
    )
    try:
        te_list_ms = _parse_te_input(te_ms_str)
    except ValueError as exc:
        yield (f"❌ {exc}", *_noop)
        return
    if not te_list_ms:
        yield ("❌ Enter echo times (ms)", *_noop)
        return

    if not phase_files:
        yield ("❌ Provide phase file(s) — use the DICOM Folder or NIfTI / MAT tab.", *_noop)
        return

    files = _sort_paths([str(_to_path(f)) for f in
                        (phase_files if isinstance(phase_files, list) else [phase_files])])
    missing = [p for p in files if not Path(p).exists()]
    if missing:
        yield ("❌ Some uploaded files no longer exist on disk:\n  " + "\n  ".join(missing), *_noop)
        return

    n_detected = _detect_echoes_from_paths(files)
    if len(files) == 1 and n_detected and n_detected > 1:
        mode = "4d"
        n_echoes = n_detected
    else:
        mode = "single" if len(files) == 1 else "multi"
        n_echoes = len(files)

    if n_echoes != len(te_list_ms):
        yield (f"❌ {n_echoes} echo(es) but {len(te_list_ms)} TE value(s) — counts must match", *_noop)
        return

    work_dir = Path(tempfile.mkdtemp(prefix="iqsmplus_"))

    # Stage phase files (convert .mat → NIfTI if needed)
    phase_paths = []
    for f in files:
        src = _to_path(f)
        if src.suffix.lower() == ".mat":
            arr, _ = load_array_with_affine(src)
            dst = work_dir / (src.stem + ".nii.gz")
            nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(dst))
        else:
            dst = work_dir / src.name
            shutil.copy(src, dst)
        phase_paths.append(dst)

    # Stage magnitude
    mag_path = None
    if mag_file:
        src = _to_path(mag_file)
        if src and src.exists():
            if src.suffix.lower() == ".mat":
                arr, _ = load_array_with_affine(src)
                dst = work_dir / (src.stem + "_mag.nii.gz")
                nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(dst))
            else:
                dst = work_dir / src.name
                shutil.copy(src, dst)
            mag_path = str(dst)

    # Stage mask
    mask_path = None
    if mask_file:
        src = _to_path(mask_file)
        if src and src.exists():
            if src.suffix.lower() == ".mat":
                arr, _ = load_array_with_affine(src)
                dst = work_dir / (src.stem + "_mask.nii.gz")
                nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(dst))
            else:
                dst = work_dir / src.name
                shutil.copy(src, dst)
            mask_path = str(dst)

    # Voxel size
    voxel_size = None
    if voxel_str and voxel_str.strip():
        try:
            voxel_size = [float(v) for v in voxel_str.replace(",", " ").split()]
        except ValueError:
            yield ("❌ Invalid voxel size — enter three numbers, e.g. 1 1 2", *_noop)
            return
        if len(voxel_size) != 3:
            yield ("❌ Voxel size must have exactly 3 values (x y z)", *_noop)
            return
    elif all(_to_path(f).suffix.lower() == ".mat" for f in files):
        voxel_size = [1.0, 1.0, 1.0]

    # B0 direction
    b0_dir = None
    if b0dir_str and b0dir_str.strip():
        try:
            b0_dir = [float(v) for v in b0dir_str.replace(",", " ").split()]
        except ValueError:
            yield ("❌ Invalid B0 direction — enter three numbers, e.g. 0 0 1", *_noop)
            return
        if len(b0_dir) != 3:
            yield ("❌ B0 direction must have exactly 3 values (x y z)", *_noop)
            return

    phase_sign = 1 if negate_phase else -1
    log_q: queue.Queue = queue.Queue()
    job = {"status": "queued", "log_queue": log_q}
    threading.Thread(
        target=_run_thread,
        args=(job, work_dir, mode, phase_paths, te_list_ms, mag_path, mask_path,
              voxel_size, float(b0_val), b0_dir, int(eroded_rad), phase_sign,
              float(vmin), float(vmax)),
        daemon=True,
    ).start()

    yield from _stream_job(job)


# ---------------------------------------------------------------------------
# DICOM parsing handler
# ---------------------------------------------------------------------------

def parse_dicom(files, progress=gr.Progress()):
    """Convert dropped DICOM folder → phase + (optional) magnitude NIfTIs, plus B0 direction."""
    base_noop = (
        gr.update(),  # accumulated_phase (state)
        gr.update(),  # sorted_files
        gr.update(),  # sorted_info
        gr.update(),  # te_ms
        gr.update(),  # voxel_str
        gr.update(),  # b0_val
        gr.update(),  # b0dir_str
        gr.update(),  # mag_file
        gr.update(),  # phase_input
        None,         # dicom_input — clear
        "",           # dicom_info
        gr.update(),  # order_group
        gr.update(),  # clear_order_btn
    )
    if not files:
        return tuple(list(base_noop[:10]) + ["", base_noop[11], base_noop[12]])

    raw_list = files if isinstance(files, list) else [files]
    progress(0.05, desc=f"Reading {len(raw_list)} upload entry(ies)…")

    file_paths = []
    for f in raw_list:
        p = _to_path(f)
        if p is None:
            continue
        if p.is_dir():
            file_paths.extend(str(c) for c in p.rglob("*") if c.is_file())
        elif p.exists():
            file_paths.append(str(p))

    if not file_paths:
        return tuple(list(base_noop[:9]) + [None, "❌ No readable files in the upload."]
                     + list(base_noop[11:]))

    if len(file_paths) == 1:
        name = Path(file_paths[0]).name
        msg = (f"⚠️ Only one file was uploaded (`{name}`). iQSM+ needs the **entire folder** "
               "of DICOMs (phase, optionally with magnitude). It looks like you may have "
               "navigated into the folder and selected a single file by mistake.\n\n"
               "**How to fix:** in the OS folder dialog, click the folder name **once** "
               "and confirm — don't enter the folder and click a file inside.")
        return tuple(list(base_noop[:9]) + [None, msg] + list(base_noop[11:]))

    progress(0.25, desc=f"Parsing {len(file_paths)} DICOM files…")

    out_dir = Path(tempfile.mkdtemp(prefix="iqsmplus_dicom_"))
    try:
        result = load_dicom_qsm_folder(file_paths, out_dir)
    except Exception as exc:
        return tuple(list(base_noop[:9]) + [None, f"❌ DICOM parsing failed:\n{exc}"]
                     + list(base_noop[11:]))

    progress(0.95, desc="Building summary…")

    phase_path = result["phase_path"]
    mag_path = result["mag_path"]
    te_s = result["te_values_s"]
    voxel = result["voxel_size"]
    b0 = result["b0"]
    b0_dir = result["b0_dir"]
    shape = result["phase_shape"]

    accumulated_paths = [str(phase_path)]
    sorted_paths = _sort_paths(accumulated_paths)
    te_ms_str = ", ".join(f"{t * 1000:g}" for t in te_s)
    info_lines = [
        f"✅ Parsed {len(te_s)} echo(es) from DICOM:",
        f"  Phase NIfTI : {Path(phase_path).name}    shape {shape}",
    ]
    if mag_path:
        info_lines.append(f"  Magnitude   : {Path(mag_path).name}")
    info_lines.append("")
    info_lines.append(f"  TE (ms)     : {te_ms_str}")
    info_lines.append(f"  Voxel (mm)  : {' '.join(f'{v:.4g}' for v in voxel)}")
    if b0 is not None:
        info_lines.append(f"  B0 (T)      : {b0}")
    if b0_dir is not None:
        info_lines.append(f"  B0 direction: {' '.join(f'{v:.4g}' for v in b0_dir)}")
    info_lines.append("")
    info_lines.append(f"NIfTI files written to: {out_dir}")
    info_html = (
        "<pre style='font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; "
        "white-space: pre; margin: 0; line-height: 1.5;'>"
        + "\n".join(info_lines)
        + "</pre>"
    )

    progress(1.0, desc="Done")
    return (
        accumulated_paths,                                                   # accumulated_phase
        sorted_paths,                                                         # sorted_files
        shape_summary(sorted_paths),                                          # sorted_info
        te_ms_str,                                                            # te_ms
        " ".join(f"{v:.4g}" for v in voxel) if voxel else gr.update(),       # voxel_str
        float(b0) if b0 is not None else gr.update(),                         # b0_val
        " ".join(f"{v:.4g}" for v in b0_dir) if b0_dir else gr.update(),     # b0dir_str
        str(mag_path) if mag_path else None,                                  # mag_file
        None,                                                                 # phase_input — clear
        None,                                                                 # dicom_input — clear
        info_html,                                                            # dicom_info
        gr.update(open=True),                                                 # order_group
        gr.update(visible=True),                                              # clear_order_btn
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container {
    --text-md: 17px;
    --block-info-text-size: 15px;
    --block-label-text-size: 16px;
    --block-title-text-size: 18px;
    font-size: 17px !important;
    line-height: 1.55 !important;
}
.gradio-container button { font-size: 16px !important; }

/* Section titles */
.gradio-container h3 {
    font-size: 1.4rem !important;
    padding: 4px 0 6px 14px !important;
    color: #1d4ed8 !important;
    border-left: 5px solid #1d4ed8 !important;
    margin: 8px 0 14px 4px !important;
}
.dark .gradio-container h3 {
    color: #60a5fa !important;
    border-left-color: #60a5fa !important;
}

/* Section panels */
.dr-section {
    margin-bottom: 24px !important;
    padding: 16px 20px !important;
    border: 2px solid #4b5563 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
    overflow: hidden !important;
}
.dark .dr-section {
    border-color: #9ca3af !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.30) !important;
}

/* Run Pipeline — green */
#dr-run-btn,
#dr-run-btn button {
    background: #16a34a !important;
    background-image: linear-gradient(180deg, #22c55e, #15803d) !important;
    color: #ffffff !important;
    border-color: #15803d !important;
    font-size: 18px !important;
    padding: 14px 28px !important;
    font-weight: 700 !important;
    width: 100% !important;
}
#dr-run-btn:hover,
#dr-run-btn button:hover {
    background: #15803d !important;
    background-image: linear-gradient(180deg, #16a34a, #14532d) !important;
}

/* Explicit Remove buttons */
#dr-clear-order-btn,
#dr-clear-order-btn button,
#dr-mask-clear-btn,
#dr-mask-clear-btn button,
#dr-mag-clear-btn,
#dr-mag-clear-btn button {
    width: 100% !important;
    margin-top: 4px !important;
}

/* Input-method tabs */
.dr-input-tabs button[role="tab"] {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    padding: 20px 36px !important;
    border: 2px solid #9ca3af !important;
    border-bottom: 2px solid #4b5563 !important;
    border-radius: 10px 10px 0 0 !important;
    background: #e5e7eb !important;
    color: #374151 !important;
    margin-right: 8px !important;
    box-shadow: 0 -1px 0 rgba(0, 0, 0, 0.08) inset !important;
}
.dr-input-tabs button[role="tab"]:hover {
    background: #d1d5db !important;
    color: #111827 !important;
}
.dr-input-tabs button[role="tab"][aria-selected="true"] {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    border-color: #1d4ed8 !important;
    box-shadow: 0 -3px 8px rgba(29, 78, 216, 0.35) !important;
    transform: translateY(-1px) !important;
}
.dr-input-tabs button[role="tab"]::after,
.dr-input-tabs button[role="tab"]::before {
    display: none !important;
    content: none !important;
}
.dark .dr-input-tabs button[role="tab"] {
    background: #374151 !important;
    color: #e5e7eb !important;
    border-color: #6b7280 !important;
}
.dark .dr-input-tabs button[role="tab"][aria-selected="true"] {
    background: #60a5fa !important;
    color: #0b1220 !important;
    border-color: #60a5fa !important;
}

/* Accordion label — match h3 */
.dr-accordion > .label-wrap,
.dr-accordion button.label-wrap,
.dr-accordion > .label-wrap span,
.dr-accordion button.label-wrap span {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #1d4ed8 !important;
}
.dr-accordion > .label-wrap,
.dr-accordion button.label-wrap {
    border-left: 5px solid #1d4ed8 !important;
    padding: 6px 0 6px 14px !important;
    margin: 4px 0 8px 4px !important;
    background: transparent !important;
}
.dark .dr-accordion > .label-wrap,
.dark .dr-accordion button.label-wrap,
.dark .dr-accordion > .label-wrap span,
.dark .dr-accordion button.label-wrap span {
    color: #60a5fa !important;
    border-left-color: #60a5fa !important;
}
.dr-accordion,
.dr-accordion > div,
.dr-accordion > div > div,
.dr-accordion .accordion-content {
    background: transparent !important;
}
.dr-section.dr-accordion {
    background: var(--block-background-fill) !important;
}

#dr-sorted-files .upload-container,
#dr-sorted-files .wrap.svelte-12ioyct,
#dr-sorted-files .upload-button,
#dr-sorted-files button:has(svg.feather-upload),
#dr-sorted-files .icon-button-wrapper.top-panel {
    display: none !important;
}
#dr-mask-file .label-clear-button,
#dr-mask-file .icon-button-wrapper.top-panel,
#dr-mag-file .label-clear-button,
#dr-mag-file .icon-button-wrapper.top-panel {
    display: none !important;
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="iQSM+") as app:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>", padding=False)
    gr.Markdown(
        "# iQSM+ — Orientation-Adaptive Quantitative Susceptibility Mapping\n"
        "<span style='font-size: 1.2em'>"
        "🐙 [GitHub](https://github.com/sunhongfu/iQSM_Plus)"
        " &nbsp;·&nbsp; "
        "📄 [Paper (MIA 2024)](https://doi.org/10.1016/j.media.2024.103160)"
        " &nbsp;·&nbsp; "
        "🌐 [Hongfu Sun](https://sunhongfu.github.io)"
        "</span>"
    )

    accumulated_phase = gr.State([])

    with gr.Column():
        # ── 1. Phase Input ────────────────────────────────────────────
        with gr.Accordion("Phase Input", open=True,
                          elem_classes=["dr-section", "dr-accordion"]):
            gr.Markdown("Wrapped MRI phase. Pick one input method below.")
            with gr.Tabs(elem_classes="dr-input-tabs"):
                with gr.Tab("📁 DICOM Folder  (recommended)") as tab_dicom:
                    gr.Markdown(
                        "**Easiest path.** Click the button and select the folder containing "
                        "the DICOM series for your GRE acquisition. Phase images are detected "
                        "via `ImageType` (containing `P`/`PHASE`); magnitude images, if present, "
                        "are auto-detected too. Echoes, TE values, voxel size, B0 strength, and "
                        "**B0 direction** (for non-axial scans) are read from headers."
                    )
                    dicom_input = gr.UploadButton(
                        "📁  Select DICOM Folder",
                        file_count="directory",
                        variant="primary",
                    )
                    dicom_info = gr.Markdown("")

                with gr.Tab("📄 NIfTI / MAT files  (advanced)") as tab_nifti:
                    gr.Markdown(
                        "Pick pre-converted **wrapped phase** files — multiple 3D echoes "
                        "(one per file) or a single 4D volume. Supported: `.nii`, `.nii.gz`, "
                        "`.mat` (v5 or v7.3). **You'll need to enter Echo Times below.**"
                    )
                    phase_input = gr.UploadButton(
                        "📄  Add Phase NIfTI / MAT",
                        file_count="multiple",
                        file_types=[".nii", ".nii.gz", ".mat"],
                        variant="primary",
                    )
                    phase_status = gr.Markdown("")

        # ── 2. Processing Order ─────────────────────────────────────
        with gr.Accordion("Processing Order", open=False,
                          elem_classes=["dr-section", "dr-accordion"]) as order_group:
            gr.Markdown(
                "Phase files in processing order (sorted naturally by filename: "
                "`mag1`, `mag2`, …, `mag10`)."
            )
            sorted_files = gr.File(
                file_count="multiple",
                show_label=False,
                interactive=True,
                height=180,
                elem_id="dr-sorted-files",
            )
            sorted_info = gr.Markdown("")
            clear_order_btn = gr.Button(
                "✕  Remove all phase files",
                variant="stop",
                visible=False,
                elem_id="dr-clear-order-btn",
            )

        # ── 3. Echo Times ────────────────────────────────────────────
        with gr.Accordion("Echo Times (ms)", open=True,
                          elem_classes=["dr-section", "dr-accordion"]):
            gr.Markdown(
                "Two accepted formats:\n"
                "- Comma-separated values (one per echo, irregular spacings allowed): "
                "`3.2, 6.5, 9.8, 13.0`\n"
                "- Compact `first_TE : spacing : count` (uniform spacing): "
                "`3.2 : 3.3 : 8` → 8 evenly-spaced echoes\n\n"
                "*Auto-filled when you use the DICOM Folder tab.*"
            )
            te_ms = gr.Textbox(
                show_label=False,
                placeholder="e.g.  3.2, 6.5, 9.8, 13.0    or compact:  3.2 : 3.3 : 8",
            )

        # ── 4. Magnitude (optional) ─────────────────────────────────
        with gr.Accordion("Magnitude (optional)", open=False,
                          elem_classes=["dr-section", "dr-accordion"]) as mag_group:
            gr.Markdown(
                "Optional. Used internally by iQSM+ for magnitude × TE² weighted fitting "
                "in multi-echo reconstruction.\n\n"
                "Supported: `.nii`, `.nii.gz`, `.mat`. Auto-filled when DICOM input includes "
                "magnitude images."
            )
            mag_button = gr.UploadButton(
                "🧲  Select Magnitude",
                file_count="single",
                file_types=[".nii", ".nii.gz", ".mat"],
                variant="primary",
            )
            mag_file = gr.File(
                file_count="single",
                show_label=False,
                interactive=True,
                visible=False,
                elem_id="dr-mag-file",
                height=70,
            )
            mag_info = gr.Markdown("")
            mag_clear_btn = gr.Button(
                "✕  Remove Magnitude",
                variant="stop",
                visible=False,
                elem_id="dr-mag-clear-btn",
            )

        # ── 5. Brain Mask (optional) ─────────────────────────────────
        with gr.Accordion("Brain Mask (optional)", open=False,
                          elem_classes=["dr-section", "dr-accordion"]) as mask_group:
            gr.Markdown(
                "Optional — if omitted, all voxels are processed. Supplying a brain mask "
                "concentrates the reconstruction on tissue and applies the configured erosion.\n\n"
                "Supported: `.nii`, `.nii.gz`, `.mat`."
            )
            mask_button = gr.UploadButton(
                "🧠  Select Brain Mask",
                file_count="single",
                file_types=[".nii", ".nii.gz", ".mat"],
                variant="primary",
            )
            mask_file = gr.File(
                file_count="single",
                show_label=False,
                interactive=True,
                visible=False,
                elem_id="dr-mask-file",
                height=70,
            )
            mask_info = gr.Markdown("")
            mask_clear_btn = gr.Button(
                "✕  Remove Brain Mask",
                variant="stop",
                visible=False,
                elem_id="dr-mask-clear-btn",
            )

        # ── 6. Acquisition + Hyper-parameters ────────────────────────
        with gr.Accordion("Acquisition & Hyper-parameters", open=False,
                          elem_classes=["dr-section", "dr-accordion"]):
            with gr.Row():
                voxel_str = gr.Textbox(
                    label="Voxel size (mm) — x y z",
                    placeholder="e.g. 1 1 2  (blank → from NIfTI header)",
                )
                b0_val = gr.Number(value=3.0, label="B0 (Tesla)",
                                   minimum=0.1, maximum=14.0, step=0.5)
            b0dir_str = gr.Textbox(
                label="B0 direction — x y z (unit vector)",
                placeholder="e.g.  0.1 0.0 0.995   (blank → [0 0 1] / read from DICOM)",
                info="iQSM+ adapts to non-axial acquisitions via this vector. "
                     "Auto-filled from DICOM headers; for NIfTI input, leave blank for axial scans.",
            )
            with gr.Row():
                eroded_rad = gr.Slider(
                    label="Mask erosion radius (voxels)",
                    minimum=0, maximum=10, step=1, value=3,
                )
                negate_phase = gr.Checkbox(
                    label="Reverse phase sign",
                    value=False,
                    info="Enable if iron-rich deep grey matter appears dark "
                         "(rather than bright) in the QSM output.",
                )

        run_btn = gr.Button("Run Reconstruction", variant="primary",
                            elem_id="dr-run-btn")

        # ── 7. Log (hidden until run) ───────────────────────────────
        with gr.Accordion("Log", open=True, visible=False,
                          elem_classes=["dr-section", "dr-accordion"]) as log_group:
            log_out = gr.Textbox(show_label=False, lines=8, max_lines=20,
                                 interactive=False, autoscroll=True)

        # ── 8. Results (hidden until run) ───────────────────────────
        with gr.Accordion("Results", open=True, visible=False,
                          elem_classes=["dr-section", "dr-accordion"]) as results_group:
            gr.Markdown("Click the file size on the right to download.")
            result_file = gr.File(show_label=False, file_count="multiple")
            result_info = gr.Markdown("")

        # ── 9. Visualisation (hidden until run) ─────────────────────
        with gr.Accordion("Visualisation", open=True, visible=False,
                          elem_classes=["dr-section", "dr-accordion"]) as viz_group:
            img_qsm = gr.Image(
                label="QSM (susceptibility, ppm)",
                show_download_button=False,
                show_fullscreen_button=False,
                height=420,
            )
            with gr.Row(equal_height=True):
                prev_btn = gr.Button("◀ Prev", scale=1)
                slice_slider = gr.Slider(
                    minimum=0, maximum=0, value=0, step=1,
                    label="Slice (Z)", show_label=False,
                    container=False, interactive=True, scale=8,
                )
                next_btn = gr.Button("Next ▶", scale=1)
            with gr.Row():
                vmin_input = gr.Number(value=_QSM_VMIN, label="QSM min (ppm)", precision=3)
                vmax_input = gr.Number(value=_QSM_VMAX, label="QSM max (ppm)", precision=3)
        output_state = gr.State((None,))

    # ── Handlers ─────────────────────────────────────────────────────────

    def _clear_btn_update(count):
        return gr.update(visible=count >= 1)

    def add_files(new_files, current, progress=gr.Progress()):
        if not new_files:
            srt = _sort_paths(current) if current else []
            return (current, srt or None, shape_summary(srt), None, gr.update(),
                    "", _clear_btn_update(len(srt)))
        files = new_files if isinstance(new_files, list) else [new_files]
        progress(0.1, desc=f"Reading {len(files)} uploaded file(s)…")
        accepted = (".nii", ".nii.gz", ".mat")
        new_paths, rejected = [], []
        for f in files:
            p = _to_path(f)
            if p is None:
                continue
            name = p.name.lower()
            if any(name.endswith(ext) for ext in accepted):
                new_paths.append(str(p))
            else:
                rejected.append(p.name)
        if rejected:
            gr.Warning("Ignored unsupported file(s): " + ", ".join(rejected)
                       + ". Only .nii, .nii.gz and .mat files are accepted.")
        if not new_paths:
            srt = _sort_paths(current) if current else []
            return (current, srt or None, shape_summary(srt), None, gr.update(),
                    "⚠️ No supported files in this upload." if rejected else "",
                    _clear_btn_update(len(srt)))
        # Drop any DICOM-converted leftovers when uploading via NIfTI tab
        current = [p for p in current if not Path(p).name.startswith("dcm_converted_")]
        progress(0.5, desc="Merging into list…")
        new_names = {Path(p).name for p in new_paths}
        kept = [p for p in current if Path(p).name not in new_names]
        updated = kept + new_paths
        srt = _sort_paths(updated)
        progress(0.85, desc="Computing shape summary…")
        summary = shape_summary(srt)
        progress(1.0, desc="Done")
        added_names = [Path(p).name for p in new_paths]
        if len(added_names) == 1:
            status = f"✅ Added file: `{added_names[0]}`"
        else:
            status = (f"✅ Added {len(added_names)} files:\n\n"
                      + "\n".join(f"- `{n}`" for n in added_names))
        return (updated, srt or None, summary, None, gr.update(open=True),
                status, _clear_btn_update(len(srt)))

    phase_input.click(
        lambda: _RED_WAIT.format(
            msg="⏳ Waiting for phase file selection / upload — Processing Order will populate "
                "after the file transfer completes…"
        ),
        outputs=phase_status,
    )
    phase_input.upload(
        add_files,
        inputs=[phase_input, accumulated_phase],
        outputs=[accumulated_phase, sorted_files, sorted_info, phase_input,
                 order_group, phase_status, clear_order_btn],
    )

    dicom_input.click(
        lambda: _RED_WAIT.format(
            msg="⏳ Waiting for DICOM folder selection / upload — parsing will start "
                "once the file transfer completes…"
        ),
        outputs=dicom_info,
    )
    dicom_input.upload(
        parse_dicom,
        inputs=[dicom_input],
        outputs=[accumulated_phase, sorted_files, sorted_info, te_ms,
                 voxel_str, b0_val, b0dir_str, mag_file, phase_input, dicom_input,
                 dicom_info, order_group, clear_order_btn],
    )

    def sync_after_remove(visible_files):
        files = (visible_files if isinstance(visible_files, list)
                 else ([] if visible_files is None else [visible_files]))
        files = [f for f in files if f is not None]
        if not files:
            return [], None, "", gr.update(), _clear_btn_update(0)
        paths = [str(_to_path(f)) for f in files]
        return (paths, paths, shape_summary(paths),
                gr.update(open=True), _clear_btn_update(len(paths)))

    sorted_files.change(
        sync_after_remove, inputs=[sorted_files],
        outputs=[accumulated_phase, sorted_files, sorted_info, order_group, clear_order_btn],
    )
    sorted_files.delete(
        sync_after_remove, inputs=[sorted_files],
        outputs=[accumulated_phase, sorted_files, sorted_info, order_group, clear_order_btn],
    )

    def on_clear_order():
        return [], None, "", gr.update(), _clear_btn_update(0)

    clear_order_btn.click(
        on_clear_order,
        outputs=[accumulated_phase, sorted_files, sorted_info, order_group, clear_order_btn],
    )

    # Magnitude
    def on_mag_upload(uploaded, progress=gr.Progress()):
        if uploaded is None:
            return gr.update(value=None, visible=False), "", gr.update(visible=False)
        progress(0.3, desc="Reading magnitude file…")
        path = _to_path(uploaded)
        if path is None or not path.exists():
            return gr.update(value=None, visible=False), "", gr.update(visible=False)
        try:
            arr, _ = load_array_with_affine(path)
            shape_str = " × ".join(str(s) for s in arr.shape)
            info = (f"&nbsp;&nbsp;**Loaded:** `{path.name}` &nbsp;·&nbsp; "
                    f"**Shape:** {shape_str} &nbsp;·&nbsp; **dtype:** `{arr.dtype}`")
        except Exception as exc:
            info = f"⚠️ Could not read magnitude: {exc}"
        progress(1.0, desc="Done")
        return (gr.update(value=str(uploaded.path) if hasattr(uploaded, "path")
                          else (uploaded if isinstance(uploaded, str) else None),
                          visible=True),
                info, gr.update(visible=True))

    mag_button.click(
        lambda: _RED_WAIT.format(
            msg="⏳ Waiting for magnitude selection / upload — info will appear once "
                "the file transfer completes…"
        ),
        outputs=mag_info,
    )
    mag_button.upload(on_mag_upload, inputs=[mag_button],
                      outputs=[mag_file, mag_info, mag_clear_btn])

    def on_mag_change(value):
        if value is None:
            return gr.update(value=None, visible=False), "", gr.update(visible=False)
        return gr.update(), gr.update(), gr.update()
    mag_file.change(on_mag_change, inputs=mag_file,
                    outputs=[mag_file, mag_info, mag_clear_btn])
    mag_file.delete(
        lambda: (gr.update(value=None, visible=False), "", gr.update(visible=False)),
        outputs=[mag_file, mag_info, mag_clear_btn],
    )
    mag_clear_btn.click(
        lambda: (gr.update(value=None, visible=False), "", gr.update(visible=False)),
        outputs=[mag_file, mag_info, mag_clear_btn],
    )

    # Mask
    def show_mask_info(mask, accumulated_paths):
        if mask is None:
            return ""
        path = _to_path(mask)
        if path is None or not path.exists():
            return ""
        try:
            arr, _ = load_array_with_affine(path)
            mask_shape = tuple(arr.shape)
            shape_str = " × ".join(str(s) for s in mask_shape)
            base = (f"&nbsp;&nbsp;**Loaded:** `{path.name}` &nbsp;·&nbsp; "
                    f"**Shape:** {shape_str} &nbsp;·&nbsp; **dtype:** `{arr.dtype}`")
        except Exception as exc:
            return f"⚠️ Could not read mask: {exc}"
        if not accumulated_paths:
            return base + " &nbsp;·&nbsp; *(load phase to verify shape match)*"
        spatials = set()
        for p in accumulated_paths:
            s = file_shape(p)
            if s and len(s) >= 3:
                spatials.add(tuple(s[:3]))
        if not spatials:
            return base
        if len(spatials) > 1:
            return base + " &nbsp;·&nbsp; ⚠️ phase files have mismatched shapes — cannot compare"
        expected = next(iter(spatials))
        spatial = mask_shape[:3] if len(mask_shape) >= 3 else mask_shape
        if spatial == expected:
            return base + " &nbsp;·&nbsp; ✓ **matches phase**"
        exp_str = " × ".join(str(s) for s in expected)
        return base + f" &nbsp;·&nbsp; ⚠️ **does not match phase** (expected {exp_str})"

    def on_mask_upload(uploaded, accumulated_paths, progress=gr.Progress()):
        if uploaded is None:
            return gr.update(value=None, visible=False), "", gr.update(visible=False)
        progress(0.3, desc="Reading mask file…")
        info = show_mask_info(uploaded, accumulated_paths)
        progress(1.0, desc="Done")
        return gr.update(value=uploaded, visible=True), info, gr.update(visible=True)

    mask_button.click(
        lambda: _RED_WAIT.format(
            msg="⏳ Waiting for mask selection / upload — info will appear once "
                "the file transfer completes…"
        ),
        outputs=mask_info,
    )
    mask_button.upload(on_mask_upload, inputs=[mask_button, accumulated_phase],
                       outputs=[mask_file, mask_info, mask_clear_btn])

    def on_mask_change(value):
        if value is None:
            return gr.update(value=None, visible=False), "", gr.update(visible=False)
        return gr.update(), gr.update(), gr.update()
    mask_file.change(on_mask_change, inputs=mask_file,
                     outputs=[mask_file, mask_info, mask_clear_btn])
    mask_file.delete(
        lambda: (gr.update(value=None, visible=False), "", gr.update(visible=False)),
        outputs=[mask_file, mask_info, mask_clear_btn],
    )
    mask_clear_btn.click(
        lambda: (gr.update(value=None, visible=False), "", gr.update(visible=False)),
        outputs=[mask_file, mask_info, mask_clear_btn],
    )

    # Run pipeline
    run_btn.click(
        run_pipeline,
        inputs=[accumulated_phase, te_ms, mag_file, mask_file, voxel_str,
                b0_val, b0dir_str, eroded_rad, negate_phase,
                vmin_input, vmax_input],
        outputs=[log_out, result_file, result_info, img_qsm,
                 output_state, slice_slider, log_group, results_group, viz_group],
    )

    # Slice navigation
    def render_slice(state, idx, vmin, vmax):
        qsm_path = state[0] if state else None
        return _make_slice_image(qsm_path, idx, vmin, vmax) if qsm_path else None

    def step_slice(current, state, delta):
        if state is None or state[0] is None:
            return gr.update()
        depth = _volume_array(state[0]).shape[2]
        return max(0, min(int(current) + delta, depth - 1))

    _ri = [output_state, slice_slider, vmin_input, vmax_input]
    _ro = [img_qsm]
    prev_btn.click(lambda c, s: step_slice(c, s, -1),
                   inputs=[slice_slider, output_state], outputs=slice_slider)
    next_btn.click(lambda c, s: step_slice(c, s, +1),
                   inputs=[slice_slider, output_state], outputs=slice_slider)
    slice_slider.change(render_slice, inputs=_ri, outputs=_ro)
    vmin_input.change(  render_slice, inputs=_ri, outputs=_ro)
    vmax_input.change(  render_slice, inputs=_ri, outputs=_ro)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def _find_free_port(preferred=7860, max_tries=20, host="127.0.0.1"):
    import socket
    for offset in range(max_tries):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in {preferred}–{preferred + max_tries - 1}")


if __name__ == "__main__":
    host = "127.0.0.1"
    port = _find_free_port(7860, host=host)
    if port != 7860:
        print(f"⚠️  Port 7860 is in use — falling back to {port}")
    import webbrowser, socket, threading, time
    def _open_when_ready():
        url = f"http://{host}:{port}/?__theme=dark"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex((host, port)) == 0:
                    webbrowser.open(url)
                    return
            time.sleep(0.2)
    threading.Thread(target=_open_when_ready, daemon=True).start()
    app.launch(server_name=host, server_port=port)
