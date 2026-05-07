"""
iQSM+ — Gradio web app for orientation-adaptive Quantitative Susceptibility Mapping.

Layout mirrors DeepRelaxo's web app:
  - DICOM Folder tab (recommended) + NIfTI / MAT files tab
  - Processing Order panel with per-file shape verification
  - Echo Times dual-format input (comma list OR `first:spacing:count`)
  - Optional magnitude (used for weighted multi-echo combination only)
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
import zipfile
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

@lru_cache(maxsize=16)
def _volume_array(nii_path):
    return nib.load(str(nii_path)).get_fdata().astype(np.float32)


@lru_cache(maxsize=32)
def _auto_window(nii_path, echo_key):
    """Robust 1–99 percentile window for the volume (or one echo of a 4D volume)."""
    data = _volume_array(nii_path)
    if data.ndim == 4 and echo_key != "none":
        data = data[..., int(echo_key)]
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _make_slice_image(nii_path, slice_idx=None, vmin=-0.2, vmax=0.2,
                      echo_idx=None, auto_window=False):
    if not nii_path:
        return None
    data = _volume_array(str(nii_path))
    if data.ndim == 4:
        ei = echo_idx if echo_idx is not None else data.shape[-1] - 1
        ei = max(0, min(int(ei), data.shape[-1] - 1))
        data = data[..., ei]
    depth = data.shape[2]
    if slice_idx is None:
        slice_idx = depth // 2
    slice_idx = max(0, min(int(slice_idx), depth - 1))
    sl = np.rot90(data[:, :, slice_idx])
    if auto_window:
        echo_key = str(echo_idx) if echo_idx is not None else "none"
        vmin_v, vmax_v = _auto_window(str(nii_path), echo_key)
    else:
        vmin_v, vmax_v = float(vmin), float(vmax)
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="black")
    ax.imshow(sl, cmap="gray", vmin=vmin_v, vmax=vmax_v, aspect="equal")
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

            # Expand 4D phase NIfTI into per-echo 3D files (matches iQSM)
            if mode == "4d":
                print("============================")
                print("EXTRACTING 4D PHASE VOLUMES")
                print("============================")
                img_4d = nib.load(str(phase_paths[0]))
                data_4d = img_4d.get_fdata(dtype=np.float32)
                echo_paths = []
                for i in range(data_4d.shape[3]):
                    p = work_dir / f"phase_echo{i+1}.nii.gz"
                    nib.save(nib.Nifti1Image(data_4d[:, :, :, i], img_4d.affine), str(p))
                    echo_paths.append(p)
                    print(f"  Echo {i+1}: TE = {te_list_ms[i]} ms")
                phase_paths = echo_paths

            # Expand 4D magnitude similarly (per-echo 3D mag for the combiner)
            mag_paths_per_echo = None
            if mag_path:
                mag_img = nib.load(mag_path)
                if len(mag_img.shape) == 4:
                    print("Expanding 4D magnitude into per-echo volumes …")
                    mag_data = mag_img.get_fdata(dtype=np.float32)
                    mag_paths_per_echo = []
                    for i in range(mag_data.shape[3]):
                        p = work_dir / f"mag_echo{i+1}.nii.gz"
                        nib.save(nib.Nifti1Image(mag_data[:, :, :, i], mag_img.affine), str(p))
                        mag_paths_per_echo.append(str(p))

            out_dir = work_dir / "iqsmplus_output"
            out_dir.mkdir(exist_ok=True)

            per_echo_qsm_paths = []
            if len(phase_paths) == 1:
                print("============================")
                print("RECONSTRUCTION")
                print("============================")
                qsm_path = run_iqsm_plus(
                    phase_nii_path=str(phase_paths[0]),
                    te=te_list_s[0],
                    mask_nii_path=mask_path,
                    voxel_size=voxel_size,
                    b0_dir=b0_dir,
                    b0=float(b0),
                    eroded_rad=int(eroded_rad),
                    phase_sign=phase_sign,
                    output_dir=str(out_dir),
                )
            else:
                print("============================")
                print(f"MULTI-ECHO RECONSTRUCTION ({len(phase_paths)} echoes)")
                print("============================")
                qsm_maps = []
                affine = None
                for i, (ppath, te_s) in enumerate(zip(phase_paths, te_list_s)):
                    print(f"\nEcho {i+1}/{len(phase_paths)}  (TE = {te_list_ms[i]} ms)")
                    echo_out = work_dir / f"echo{i+1}_output"
                    q_path = run_iqsm_plus(
                        phase_nii_path=str(ppath),
                        te=te_s,
                        mask_nii_path=mask_path,
                        voxel_size=voxel_size,
                        b0_dir=b0_dir,
                        b0=float(b0),
                        eroded_rad=int(eroded_rad),
                        phase_sign=phase_sign,
                        output_dir=str(echo_out),
                    )
                    q_img = nib.load(q_path)
                    if affine is None:
                        affine = q_img.affine
                    qsm_maps.append(q_img.get_fdata(dtype=np.float32))

                    # Save per-echo iQSM+ in the main output dir with clear names
                    new_q = out_dir / f"iQSM_plus_e{i+1}.nii.gz"
                    shutil.copy(q_path, new_q)
                    per_echo_qsm_paths.append(str(new_q))

                print(f"\nAveraging {len(qsm_maps)} echoes …")
                qsm_stack = np.stack(qsm_maps, axis=-1)
                te_bc = np.array(te_list_s, dtype=np.float32).reshape(1, 1, 1, -1)
                if mag_path:
                    print("  Using magnitude × TE² weighted averaging")
                    if mag_paths_per_echo:
                        mag_data = np.stack(
                            [nib.load(mp).get_fdata(dtype=np.float32) for mp in mag_paths_per_echo],
                            axis=-1,
                        )
                    else:
                        mag_3d = nib.load(mag_path).get_fdata(dtype=np.float32)
                        if mag_3d.ndim == 3:
                            mag_data = np.repeat(mag_3d[..., np.newaxis], len(qsm_maps), axis=-1)
                        else:
                            mag_data = mag_3d
                    weights = (mag_data * te_bc) ** 2
                    denom = weights.sum(axis=-1, keepdims=True)
                    denom[denom == 0] = 1.0
                    qsm_avg = (weights * qsm_stack).sum(axis=-1) / denom.squeeze(-1)
                else:
                    print("  No magnitude provided — using TE² weighted averaging (uniform magnitude)")
                    weights = te_bc ** 2
                    qsm_avg = (qsm_stack * weights).sum(axis=-1) / weights.sum()
                qsm_path = str(out_dir / "iQSM_plus.nii.gz")
                nib.save(nib.Nifti1Image(qsm_avg.astype(np.float32), affine), qsm_path)

            # Orientation-preview inputs (last-echo phase, last-echo magnitude, mask)
            last_phase_path = str(phase_paths[-1])
            last_phase_echo = None  # phase_paths is per-echo 3D after staging

            last_mag_path, last_mag_echo = None, None
            if mag_paths_per_echo:
                last_mag_path = mag_paths_per_echo[-1]
                last_mag_echo = None
            elif mag_path:
                try:
                    mag_shape = nib.load(mag_path).shape
                    last_mag_path = mag_path
                    last_mag_echo = (int(mag_shape[-1]) - 1
                                     if len(mag_shape) == 4 else None)
                except Exception:
                    last_mag_path = mag_path
                    last_mag_echo = None

            print("\n✅ Pipeline complete!")
            job["status"] = "done"
            job["qsm_path"] = qsm_path
            job["out_dir"] = str(out_dir)
            job["depth"] = _volume_array(qsm_path).shape[2]
            job["qsm_image"] = _make_slice_image(qsm_path, vmin=vmin, vmax=vmax)

            job["last_phase_path"] = last_phase_path
            job["last_phase_echo"] = last_phase_echo
            job["last_mag_path"]   = last_mag_path
            job["last_mag_echo"]   = last_mag_echo
            job["mask_path"]       = mask_path

            job["phase_image"] = _make_slice_image(
                last_phase_path, echo_idx=last_phase_echo, auto_window=True,
            ) if last_phase_path else None
            job["mag_image"] = _make_slice_image(
                last_mag_path, echo_idx=last_mag_echo, auto_window=True,
            ) if last_mag_path else None
            job["mask_image"] = _make_slice_image(
                mask_path, vmin=0, vmax=1,
            ) if mask_path else None

            # Per-echo info — used by the "Echo Selection" recombine panel.
            # te lists are populated for both single- and multi-echo so the
            # panel renders for single-echo too (in a disabled state).
            job["per_echo_qsm_paths"] = per_echo_qsm_paths or None
            job["per_echo_te_s"]      = list(te_list_s)
            job["per_echo_te_ms"]     = list(te_list_ms)
            job["mag_paths_per_echo"] = list(mag_paths_per_echo) if mag_paths_per_echo else None
            job["mag_3d_path"]        = (mag_path if (mag_path and not mag_paths_per_echo)
                                         else None)
            job["render_vmin"]        = float(vmin)
            job["render_vmax"]        = float(vmax)
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
    """Order: recombined first (when present), then the all-echoes combined
    iQSM_plus.nii.gz, then per-echo iQSM_plus_e<i>.nii.gz files."""
    out_dir = job.get("out_dir")
    files = []
    # Recombined output (if Echo Selection has been used) — qsm_path is
    # updated to point at it, distinguished by filename pattern.
    qsm = job.get("qsm_path")
    if qsm and "iQSM_plus_recombined_" in Path(qsm).name:
        files.append(qsm)
    # Original all-echoes combination
    if out_dir:
        all_echoes = Path(out_dir) / "iQSM_plus.nii.gz"
        if all_echoes.exists() and str(all_echoes) not in files:
            files.append(str(all_echoes))
    elif qsm and qsm not in files:
        files.append(qsm)
    # Per-echo files
    for p in (job.get("per_echo_qsm_paths") or []):
        if p and p not in files:
            files.append(p)
    return files or None


def _echo_choices(job):
    """Build (label, value) pairs for the echo-selection CheckboxGroup, including
    single-echo runs (so the panel can render disabled instead of hidden)."""
    tes_ms = job.get("per_echo_te_ms") or []
    if not tes_ms:
        return []
    return [(f"Echo {i+1} — TE = {te_ms:g} ms", i) for i, te_ms in enumerate(tes_ms)]


def _build_results_zip(job):
    """Bundle all output files into a single ZIP for one-click download."""
    files = _result_files(job) or []
    if not files:
        return None
    out_dir = job.get("out_dir") or tempfile.gettempdir()
    zip_path = Path(out_dir) / "iqsmplus_results.zip"
    try:
        zip_path.unlink()
    except FileNotFoundError:
        pass
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=Path(p).name)
    return str(zip_path)


def _result_info_md(job):
    files = _result_files(job)
    return shape_summary(files) if files else ""


def _state_dict(job):
    return {
        "qsm_path":             job.get("qsm_path"),
        "last_phase_path":      job.get("last_phase_path"),
        "last_phase_echo":      job.get("last_phase_echo"),
        "last_mag_path":        job.get("last_mag_path"),
        "last_mag_echo":        job.get("last_mag_echo"),
        "mask_path":            job.get("mask_path"),
        "per_echo_qsm_paths":   job.get("per_echo_qsm_paths"),
        "per_echo_te_s":        job.get("per_echo_te_s"),
        "per_echo_te_ms":       job.get("per_echo_te_ms"),
        "mag_paths_per_echo":   job.get("mag_paths_per_echo"),
        "mag_3d_path":          job.get("mag_3d_path"),
        "out_dir":              job.get("out_dir"),
        "render_vmin":          job.get("render_vmin"),
        "render_vmax":          job.get("render_vmax"),
    }


def _state_and_slider_update(job):
    state = _state_dict(job)
    depth = job.get("depth")
    if depth and not job.get("_slider_init"):
        job["_slider_init"] = True
        slider = gr.update(visible=True, minimum=0, maximum=depth - 1,
                           value=depth // 2, interactive=True)
    else:
        slider = gr.update()
    return state, slider


def _visibility_updates(job):
    has_results = bool(_result_files(job))
    zip_path = _build_results_zip(job) if has_results else None
    choices = _echo_choices(job)
    has_run = bool(choices)                # any run done (single or multi)
    is_multi = len(choices) >= 2           # only multi-echo can recombine
    return (
        gr.update(visible=True),                             # log_group
        gr.update(visible=has_results),                      # results_group
        gr.update(visible=bool(job.get("qsm_image"))),       # viz_group
        gr.update(visible=bool(job.get("phase_image"))),     # phase_panel
        gr.update(visible=bool(job.get("mag_image"))),       # mag_panel
        gr.update(visible=bool(job.get("mask_image"))),      # mask_panel
        gr.update(visible=bool(job.get("phase_image") or job.get("mag_image")
                               or job.get("mask_image"))),  # orientation_row
        (gr.update(value=zip_path, visible=True)
         if zip_path else gr.update(visible=False)),        # download_all_btn
        gr.update(visible=has_run),                          # echo_select_group
        (gr.update(choices=choices,
                   value=[c[1] for c in choices],
                   interactive=is_multi)
         if has_run else gr.update()),                       # echo_checkboxes
        gr.update(interactive=is_multi),                     # recombine_btn
    )


def _stream_job(job):
    log = ""
    while True:
        msg = job["log_queue"].get()
        if msg is None:
            break
        log += msg + "\n"
        state, slider = _state_and_slider_update(job)
        v = _visibility_updates(job)
        yield (log, _result_files(job), _result_info_md(job),
               job.get("qsm_image"),
               job.get("phase_image"), job.get("mag_image"), job.get("mask_image"),
               state, slider, *v)
    state, slider = _state_and_slider_update(job)
    v = _visibility_updates(job)
    yield (log, _result_files(job), _result_info_md(job),
           job.get("qsm_image"),
           job.get("phase_image"), job.get("mag_image"), job.get("mask_image"),
           state, slider, *v)


# ---------------------------------------------------------------------------
# Pipeline callback
# ---------------------------------------------------------------------------

def run_pipeline(phase_files, te_ms_str, mag_files, mask_file,
                 voxel_str, b0_val, b0dir_str, eroded_rad, negate_phase,
                 vmin, vmax):
    _noop = (
        None, "",                                        # files, info
        None,                                            # qsm_image
        None, None, None,                                # phase_image, mag_image, mask_image
        {}, gr.update(),                                 # state, slider
        gr.update(visible=True),                         # log_group
        gr.update(visible=False),                        # results_group
        gr.update(visible=False),                        # viz_group
        gr.update(visible=False),                        # phase_panel
        gr.update(visible=False),                        # mag_panel
        gr.update(visible=False),                        # mask_panel
        gr.update(visible=False),                        # orientation_row
        gr.update(visible=False),                        # download_all_btn
        gr.update(visible=False),                        # echo_select_group
        gr.update(),                                     # echo_checkboxes
        gr.update(),                                     # recombine_btn
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
        yield ("❌ Provide phase file(s) — use the Phase upload in the NIfTI / MAT tab.", *_noop)
        return
    # Magnitude is optional — multi-echo falls back to TE²-only weighting when omitted.

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

    # Stage magnitude (optional). Multiple 3D → stacked into 4D; single 4D/3D → as-is.
    # When omitted, iQSM+ uses mag = 1 internally (TE²-only weighting in multi-echo).
    mag_files_list = mag_files if isinstance(mag_files, list) else [mag_files]
    mag_files_list = [str(_to_path(f)) for f in (mag_files_list or []) if f]
    mag_files_list = _sort_paths([p for p in mag_files_list if p and Path(p).exists()])

    mag_path = None
    if mag_files_list:
        mag_n_detected = _detect_echoes_from_paths(mag_files_list)
        if len(mag_files_list) == 1 and mag_n_detected and mag_n_detected > 1:
            mag_n_echoes = mag_n_detected
        else:
            mag_n_echoes = len(mag_files_list)

        if mag_n_echoes != len(te_list_ms):
            yield (f"❌ Magnitude has {mag_n_echoes} echo(es) but {len(te_list_ms)} TE value(s) — "
                   "phase, magnitude, and TEs must all match.", *_noop)
            return

        staged_mag_paths = []
        for f in mag_files_list:
            src = _to_path(f)
            if src.suffix.lower() == ".mat":
                arr, _ = load_array_with_affine(src)
                dst = work_dir / (src.stem + "_mag.nii.gz")
                nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(dst))
            else:
                dst = work_dir / ("mag_" + src.name if not src.name.startswith("mag_") else src.name)
                shutil.copy(src, dst)
            staged_mag_paths.append(dst)

        if len(staged_mag_paths) > 1:
            imgs = [nib.load(str(p)) for p in staged_mag_paths]
            affine = imgs[0].affine
            data_4d = np.stack([img.get_fdata(dtype=np.float32) for img in imgs], axis=-1)
            mag_combined = work_dir / "magnitude_4d.nii.gz"
            nib.save(nib.Nifti1Image(data_4d, affine), str(mag_combined))
            mag_path = str(mag_combined)
        else:
            mag_path = str(staged_mag_paths[0])

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
            yield ("❌ Invalid voxel size — enter three numbers separated by spaces or commas, e.g. 1 1 2 or 1, 1, 2", *_noop)
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
            yield ("❌ Invalid B0 direction — enter three numbers separated by spaces or commas, e.g. 0 0 1 or 0, 0, 1", *_noop)
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

/* Common height for all action buttons so the form looks consistent. */
#dr-run-btn,         #dr-run-btn         button,
#dr-download-all-btn,#dr-download-all-btn button,
#dr-clear-order-btn, #dr-clear-order-btn button,
#dr-clear-mag-btn,   #dr-clear-mag-btn   button,
#dr-mask-clear-btn,  #dr-mask-clear-btn  button {
    height: 56px !important;
    min-height: 56px !important;
    max-height: 56px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    font-weight: 600 !important;
}

/* Run Pipeline — green */
#dr-run-btn,
#dr-run-btn button {
    background: #16a34a !important;
    background-image: linear-gradient(180deg, #22c55e, #15803d) !important;
    color: #ffffff !important;
    border-color: #15803d !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}
#dr-run-btn:hover,
#dr-run-btn button:hover {
    background: #15803d !important;
    background-image: linear-gradient(180deg, #16a34a, #14532d) !important;
}

/* Download all (ZIP) — slate / neutral. */
#dr-download-all-btn,
#dr-download-all-btn button {
    background: #475569 !important;
    background-image: linear-gradient(180deg, #64748b, #334155) !important;
    color: #ffffff !important;
    border-color: #334155 !important;
}
#dr-download-all-btn:hover,
#dr-download-all-btn button:hover {
    background: #334155 !important;
    background-image: linear-gradient(180deg, #475569, #1e293b) !important;
}

/* Clear / Remove buttons keep their stop-variant red but get the common height. */
#dr-clear-order-btn,
#dr-clear-order-btn button,
#dr-mask-clear-btn,
#dr-mask-clear-btn button,
#dr-clear-mag-btn,
#dr-clear-mag-btn button,
#dr-mag-clear-btn,
#dr-mag-clear-btn button {
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

/* Phase + Magnitude upload buttons — fixed size and horizontally aligned. */
.dr-upload-row { align-items: stretch !important; }
.dr-upload-btn,
.dr-upload-btn button {
    height: 56px !important;
    min-height: 56px !important;
    max-height: 56px !important;
    width: 100% !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    box-sizing: border-box !important;
}

#dr-sorted-files .upload-container,
#dr-sorted-files .wrap.svelte-12ioyct,
#dr-sorted-files .upload-button,
#dr-sorted-files button:has(svg.feather-upload),
#dr-sorted-files .icon-button-wrapper.top-panel,
#dr-sorted-files .label-clear-button,
#dr-sorted-mag-files .upload-container,
#dr-sorted-mag-files .wrap.svelte-12ioyct,
#dr-sorted-mag-files .upload-button,
#dr-sorted-mag-files button:has(svg.feather-upload),
#dr-sorted-mag-files .icon-button-wrapper.top-panel,
#dr-sorted-mag-files .label-clear-button {
    display: none !important;
}
#dr-mask-file .label-clear-button,
#dr-mask-file .icon-button-wrapper.top-panel {
    display: none !important;
}
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="iQSM+", analytics_enabled=False) as app:
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
    accumulated_mag   = gr.State([])

    with gr.Column():
        # ── 1. Phase + Magnitude Input ───────────────────────────────
        with gr.Accordion("Phase + Magnitude Input", open=True,
                          elem_classes=["dr-section", "dr-accordion"]):
            gr.Markdown(
                "Raw (wrapped) phase is required. Magnitude is optional "
                "(used for weighted multi-echo combination only). "
                "Have raw DICOMs? See "
                "[DICOM → NIfTI conversion]"
                "(https://github.com/sunhongfu/iQSM_Plus#dicom--nifti-conversion) "
                "in the GitHub repo, or run `python dicom_to_nifti.py --help`."
            )
            with gr.Row(equal_height=True, elem_classes="dr-upload-row"):
                with gr.Column():
                    phase_input = gr.UploadButton(
                        "Add Phase NIfTI / MAT",
                        file_count="multiple",
                        file_types=[".nii", ".nii.gz", ".gz", ".mat"],
                        variant="primary",
                        elem_classes="dr-upload-btn",
                    )
                    phase_status = gr.Markdown("")
                with gr.Column():
                    mag_button = gr.UploadButton(
                        "Add Magnitude NIfTI / MAT  (optional)",
                        file_count="multiple",
                        file_types=[".nii", ".nii.gz", ".gz", ".mat"],
                        variant="primary",
                        elem_classes="dr-upload-btn",
                    )
                    mag_status = gr.Markdown("")

        # ── 2. Processing Order ─────────────────────────────────────
        with gr.Accordion("Processing Order", open=False,
                          elem_classes=["dr-section", "dr-accordion"]) as order_group:
            gr.Markdown(
                "Phase and magnitude files in processing order — sorted naturally "
                "by filename (`mag1`, `mag2`, …, `mag10`). When both modalities are "
                "supplied, the two columns must have matching echo counts."
            )
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("**Phase**")
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
                with gr.Column():
                    gr.Markdown("**Magnitude**")
                    sorted_mag_files = gr.File(
                        file_count="multiple",
                        show_label=False,
                        interactive=True,
                        height=180,
                        elem_id="dr-sorted-mag-files",
                    )
                    sorted_mag_info = gr.Markdown("")
                    clear_mag_btn = gr.Button(
                        "✕  Remove all magnitude files",
                        variant="stop",
                        visible=False,
                        elem_id="dr-clear-mag-btn",
                    )

        # ── 3. Echo Times ────────────────────────────────────────────
        with gr.Accordion("Echo Times (ms)", open=True,
                          elem_classes=["dr-section", "dr-accordion"]):
            gr.Markdown(
                "Two accepted formats:\n"
                "- Comma-separated values (one per echo, any spacing): "
                "`2.4, 3.6, 9.2, 20.8`\n"
                "- Compact `first_TE : spacing : count` (uniform spacing): "
                "`4.5 : 5.0 : 5` → `4.5, 9.5, 14.5, 19.5, 24.5`"
            )
            te_ms = gr.Textbox(
                show_label=False,
                placeholder="e.g.  2.4, 3.6, 9.2, 20.8    or compact:  4.5 : 5.0 : 5",
            )

        # ── 4. Brain Mask (optional but recommended) ─────────────────
        with gr.Accordion("Brain Mask (optional but recommended)", open=True,
                          elem_classes=["dr-section", "dr-accordion"]) as mask_group:
            gr.Markdown(
                "A brain mask improves **iQSM+ reconstruction quality** by "
                "concentrating the network on tissue voxels.\n\n"
                "Default mask erosion is **3 voxels**; adjust under "
                "**Acquisition & Hyper-parameters** below if you'd rather retain "
                "more of the cortical brain region.\n\n"
                "⚠️ Make sure the mask is **oriented and aligned** to the phase / "
                "magnitude volumes — once the run finishes, you can confirm in the "
                "**Visualisation** panel below (the brain-mask preview shares the "
                "same slice slider as the phase / magnitude previews)."
            )
            mask_button = gr.UploadButton(
                "🧠  Select Brain Mask",
                file_count="single",
                file_types=[".nii", ".nii.gz", ".gz", ".mat"],
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

        # ── 5. Acquisition + Hyper-parameters ────────────────────────
        with gr.Accordion("Acquisition & Hyper-parameters", open=False,
                          elem_classes=["dr-section", "dr-accordion"]):
            with gr.Row():
                voxel_str = gr.Textbox(
                    label="Voxel size (mm) — x y z",
                    placeholder="e.g.  1 1 2    or    1, 1, 2",
                    info="Three numbers, comma- or space-separated. "
                         "Leave blank to read from the NIfTI header.",
                )
                b0_val = gr.Number(value=3.0, label="B0 (Tesla)",
                                   minimum=0.1, maximum=14.0, step=0.5)
            b0dir_str = gr.Textbox(
                label="B0 direction — x y z (unit vector)",
                placeholder="e.g.  0.1 0.0 0.995    or    0.1, 0.0, 0.995",
                info="Three numbers, comma- or space-separated. "
                     "iQSM+ adapts to non-axial acquisitions via this vector. "
                     "Auto-filled from DICOM headers; leave blank for axial scans "
                     "(defaults to 0 0 1).",
            )
            with gr.Row():
                eroded_rad = gr.Slider(
                    label="Mask erosion radius (voxels)",
                    minimum=0, maximum=10, step=1, value=0,
                    interactive=False,
                    info="Disabled when no brain mask is provided "
                         "(erosion has no effect without a mask).",
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

        # ── 8. Echo Selection — recombine multi-echo iQSM+ ──────────
        with gr.Accordion("Echo Selection (refine combination)", open=True, visible=False,
                          elem_classes=["dr-section", "dr-accordion"]) as echo_select_group:
            gr.Markdown(
                "Multi-echo iQSM+ combines all echoes via magnitude × TE² weighted averaging "
                "(or TE² weighting only when no magnitude is supplied). Early echoes with "
                "short TEs may produce artifacts — uncheck any echoes you want to "
                "**exclude**, then click **Recombine**. The per-echo files above are "
                "reused, so this is fast (no re-inference)."
            )
            echo_checkboxes = gr.CheckboxGroup(
                choices=[], value=[],
                label="Include in combination",
            )
            recombine_btn = gr.Button(
                "🔁  Recombine selected echoes",
                variant="primary",
                elem_id="dr-recombine-btn",
            )
            recombine_status = gr.Markdown("")

        # ── 9. Results (hidden until run) ───────────────────────────
        with gr.Accordion("Results", open=True, visible=False,
                          elem_classes=["dr-section", "dr-accordion"]) as results_group:
            gr.Markdown(
                "Recombined output (`iQSM_plus_recombined_e<…>.nii.gz`) appears first when "
                "Echo Selection has been used; the original all-echoes combination "
                "(`iQSM_plus.nii.gz`) and per-echo files (`iQSM_plus_e1.nii.gz`, …) follow. "
                "Click a file size on the right to download a single file, or use "
                "**Download all (ZIP)** below for the whole bundle."
            )
            result_file = gr.File(show_label=False, file_count="multiple")
            result_info = gr.Markdown("")
            download_all_btn = gr.DownloadButton(
                "📦  Download all (ZIP)",
                visible=False,
                elem_id="dr-download-all-btn",
            )

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

            # Inputs preview — to confirm orientation alignment of phase / magnitude / mask.
            # Auto-windowed; not affected by the QSM slider window.
            with gr.Row(visible=False) as orientation_row:
                img_phase = gr.Image(
                    label="Raw phase — last echo (no mask)",
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=300, visible=False,
                )
                img_mag = gr.Image(
                    label="Raw magnitude — last echo (no mask)",
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=300, visible=False,
                )
                img_mask = gr.Image(
                    label="Brain mask",
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=300, visible=False,
                )
        output_state = gr.State({})

    # ── Handlers ─────────────────────────────────────────────────────────

    def _clear_btn_update(count):
        return gr.update(visible=count >= 1)

    def _voxel_from_first_nii(paths):
        """Read voxel sizes (mm) from the first NIfTI in the upload.
        Returns a string like '1 1 2' or None when nothing usable was found."""
        for p in paths:
            name = Path(p).name.lower()
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                try:
                    zooms = nib.load(str(p)).header.get_zooms()
                    if len(zooms) >= 3:
                        return " ".join(f"{float(z):.4g}" for z in zooms[:3])
                except Exception:
                    continue
        return None

    def add_files(new_files, current, current_voxel, progress=gr.Progress()):
        if not new_files:
            srt = _sort_paths(current) if current else []
            return (current, srt or None, shape_summary(srt), None, gr.update(),
                    "", _clear_btn_update(len(srt)), gr.update())
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
                    _clear_btn_update(len(srt)), gr.update())
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
        # Auto-fill voxel size from the first NIfTI header — but only if the field is empty
        voxel_update = gr.update()
        if not (current_voxel and current_voxel.strip()):
            v = _voxel_from_first_nii(new_paths)
            if v:
                voxel_update = gr.update(value=v)
        return (updated, srt or None, summary, None, gr.update(open=True),
                status, _clear_btn_update(len(srt)), voxel_update)

    phase_input.click(
        lambda: _RED_WAIT.format(
            msg="⏳ Waiting for phase file selection / upload — Processing Order will populate "
                "after the file transfer completes…"
        ),
        outputs=phase_status,
    )
    phase_input.upload(
        add_files,
        inputs=[phase_input, accumulated_phase, voxel_str],
        outputs=[accumulated_phase, sorted_files, sorted_info, phase_input,
                 order_group, phase_status, clear_order_btn, voxel_str],
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

    # Magnitude — accumulates uploaded files like phase. Multiple 3D files are
    # kept as a list and stacked into 4D at run time.
    def add_mag_files(new_files, current, progress=gr.Progress()):
        if not new_files:
            srt = _sort_paths(current) if current else []
            return (current, srt or None, shape_summary(srt), None, gr.update(),
                    "", _clear_btn_update(len(srt)))
        files = new_files if isinstance(new_files, list) else [new_files]
        progress(0.1, desc=f"Reading {len(files)} uploaded magnitude file(s)…")
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
        current = [p for p in current if not Path(p).name.startswith("dcm_converted_")]
        new_names = {Path(p).name for p in new_paths}
        kept = [p for p in current if Path(p).name not in new_names]
        updated = kept + new_paths
        srt = _sort_paths(updated)
        progress(1.0, desc="Done")
        added_names = [Path(p).name for p in new_paths]
        if len(added_names) == 1:
            status = f"✅ Added file: `{added_names[0]}`"
        else:
            status = (f"✅ Added {len(added_names)} magnitude files:\n\n"
                      + "\n".join(f"- `{n}`" for n in added_names))
        return (updated, srt or None, shape_summary(srt), None, gr.update(open=True),
                status, _clear_btn_update(len(srt)))

    mag_button.click(
        lambda: _RED_WAIT.format(
            msg="⏳ Waiting for magnitude file selection / upload — Processing Order will "
                "populate after the file transfer completes…"
        ),
        outputs=mag_status,
    )
    mag_button.upload(
        add_mag_files, inputs=[mag_button, accumulated_mag],
        outputs=[accumulated_mag, sorted_mag_files, sorted_mag_info, mag_button,
                 order_group, mag_status, clear_mag_btn],
    )

    def sync_mag_after_remove(visible_files):
        files = (visible_files if isinstance(visible_files, list)
                 else ([] if visible_files is None else [visible_files]))
        files = [f for f in files if f is not None]
        if not files:
            return [], None, "", gr.update(), _clear_btn_update(0)
        paths = [str(_to_path(f)) for f in files]
        return (paths, paths, shape_summary(paths),
                gr.update(open=True), _clear_btn_update(len(paths)))

    sorted_mag_files.change(
        sync_mag_after_remove, inputs=[sorted_mag_files],
        outputs=[accumulated_mag, sorted_mag_files, sorted_mag_info,
                 order_group, clear_mag_btn],
    )
    sorted_mag_files.delete(
        sync_mag_after_remove, inputs=[sorted_mag_files],
        outputs=[accumulated_mag, sorted_mag_files, sorted_mag_info,
                 order_group, clear_mag_btn],
    )
    clear_mag_btn.click(
        lambda: ([], None, "", gr.update(), _clear_btn_update(0)),
        outputs=[accumulated_mag, sorted_mag_files, sorted_mag_info,
                 order_group, clear_mag_btn],
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
            return (gr.update(value=None, visible=False), "", gr.update(visible=False),
                    gr.update(value=0, interactive=False))
        progress(0.3, desc="Reading mask file…")
        info = show_mask_info(uploaded, accumulated_paths)
        progress(1.0, desc="Done")
        # Mask now available → enable erosion slider, restore default 3.
        return (gr.update(value=uploaded, visible=True), info, gr.update(visible=True),
                gr.update(value=3, interactive=True))

    mask_button.click(
        lambda: _RED_WAIT.format(
            msg="⏳ Waiting for mask selection / upload — info will appear once "
                "the file transfer completes…"
        ),
        outputs=mask_info,
    )
    mask_button.upload(on_mask_upload, inputs=[mask_button, accumulated_phase],
                       outputs=[mask_file, mask_info, mask_clear_btn, eroded_rad])

    def on_mask_change(value):
        if value is None:
            return (gr.update(value=None, visible=False), "", gr.update(visible=False),
                    gr.update(value=0, interactive=False))
        return gr.update(), gr.update(), gr.update(), gr.update()
    mask_file.change(on_mask_change, inputs=mask_file,
                     outputs=[mask_file, mask_info, mask_clear_btn, eroded_rad])

    _mask_clear_outputs = [mask_file, mask_info, mask_clear_btn, eroded_rad]
    _mask_clear_returns = (gr.update(value=None, visible=False), "",
                           gr.update(visible=False),
                           gr.update(value=0, interactive=False))
    mask_file.delete(    lambda: _mask_clear_returns, outputs=_mask_clear_outputs)
    mask_clear_btn.click(lambda: _mask_clear_returns, outputs=_mask_clear_outputs)

    # Run pipeline
    run_btn.click(
        run_pipeline,
        inputs=[accumulated_phase, te_ms, accumulated_mag, mask_file, voxel_str,
                b0_val, b0dir_str, eroded_rad, negate_phase,
                vmin_input, vmax_input],
        outputs=[log_out, result_file, result_info,
                 img_qsm, img_phase, img_mag, img_mask,
                 output_state, slice_slider,
                 log_group, results_group, viz_group,
                 img_phase, img_mag, img_mask, orientation_row,
                 download_all_btn,
                 echo_select_group, echo_checkboxes, recombine_btn],
    )

    # ── Recombine multi-echo iQSM+ from per-echo files (no re-inference) ──
    def recombine_echoes(selected, state, slice_idx, vmin, vmax):
        if not state or not state.get("per_echo_qsm_paths"):
            return ("⚠️ No multi-echo run available to recombine.",
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        if not selected:
            return ("⚠️ Select at least one echo to recombine.",
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

        idx = sorted(int(i) for i in selected)
        per_qsm = state["per_echo_qsm_paths"]
        te_s    = state["per_echo_te_s"]
        te_ms   = state["per_echo_te_ms"]
        out_dir = Path(state["out_dir"])

        qsm_vols = [_volume_array(per_qsm[i]) for i in idx]
        affine = nib.load(per_qsm[idx[0]]).affine

        mag_per_echo = state.get("mag_paths_per_echo")
        mag_3d_path  = state.get("mag_3d_path")
        sel_mags = None
        if mag_per_echo and len(mag_per_echo) >= len(per_qsm):
            sel_mags = [_volume_array(mag_per_echo[i]) for i in idx]
        elif mag_3d_path:
            mag_3d = _volume_array(mag_3d_path)
            sel_mags = [mag_3d for _ in idx]

        sel_te_s = np.array([te_s[i] for i in idx], dtype=np.float32)
        te_bc = sel_te_s.reshape(1, 1, 1, -1)
        if sel_mags:
            mag_data = np.stack(sel_mags, axis=-1)
            weights = (mag_data * te_bc) ** 2
            denom = weights.sum(axis=-1, keepdims=True)
            denom[denom == 0] = 1.0
            qsm_avg = (weights * np.stack(qsm_vols, axis=-1)).sum(axis=-1) / denom.squeeze(-1)
            method = "magnitude × TE² weighted"
        else:
            weights = te_bc ** 2
            qsm_avg = (np.stack(qsm_vols, axis=-1) * weights).sum(axis=-1) / weights.sum()
            method = "TE² weighted (uniform magnitude)"

        sel_tag = "_".join(f"e{i+1}" for i in idx)
        for old in out_dir.glob("iQSM_plus_recombined_*.nii.gz"):
            try: old.unlink()
            except OSError: pass
        qsm_out = out_dir / f"iQSM_plus_recombined_{sel_tag}.nii.gz"
        nib.save(nib.Nifti1Image(qsm_avg.astype(np.float32), affine), str(qsm_out))
        _volume_array.cache_clear()

        new_state = dict(state)
        new_state["qsm_path"] = str(qsm_out)
        qsm_img = _make_slice_image(str(qsm_out), slice_idx, vmin, vmax)

        included = ", ".join(f"Echo {i+1} (TE={te_ms[i]:g} ms)" for i in idx)
        excluded = [i for i in range(len(per_qsm)) if i not in idx]
        excluded_str = (", ".join(f"Echo {i+1}" for i in excluded)
                        if excluded else "(none)")
        status = (f"✅ Recombined {len(idx)}/{len(per_qsm)} echoes via {method}.\n\n"
                  f"**Included:** {included}\n\n"
                  f"**Excluded:** {excluded_str}\n\n"
                  f"Saved as `{Path(qsm_out).name}`")

        # Refresh Results panel listing — recombined first, then all-echoes,
        # then per-echo files.
        files = [str(qsm_out)]
        all_echoes_qsm = out_dir / "iQSM_plus.nii.gz"
        if all_echoes_qsm.exists():
            files.append(str(all_echoes_qsm))
        for p in per_qsm:
            if p:
                files.append(p)

        zip_path = out_dir / "iqsmplus_results.zip"
        try: zip_path.unlink()
        except FileNotFoundError: pass
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in files:
                zf.write(p, arcname=Path(p).name)

        return (status, files, qsm_img, new_state,
                shape_summary(files), gr.update(value=str(zip_path), visible=True))

    recombine_btn.click(
        recombine_echoes,
        inputs=[echo_checkboxes, output_state, slice_slider,
                vmin_input, vmax_input],
        outputs=[recombine_status, result_file, img_qsm,
                 output_state, result_info, download_all_btn],
    )

    # Slice navigation — re-renders QSM at user window plus phase / mag / mask
    # at auto-window / [0,1] for orientation verification.
    def render_slice(state, idx, vmin, vmax):
        if not state:
            return None, None, None, None
        qsm_path = state.get("qsm_path")
        ph_path  = state.get("last_phase_path")
        ph_echo  = state.get("last_phase_echo")
        mg_path  = state.get("last_mag_path")
        mg_echo  = state.get("last_mag_echo")
        mk_path  = state.get("mask_path")
        return (
            _make_slice_image(qsm_path, idx, vmin, vmax) if qsm_path else None,
            _make_slice_image(ph_path, idx, echo_idx=ph_echo,
                              auto_window=True) if ph_path else None,
            _make_slice_image(mg_path, idx, echo_idx=mg_echo,
                              auto_window=True) if mg_path else None,
            _make_slice_image(mk_path, idx, vmin=0, vmax=1) if mk_path else None,
        )

    def step_slice(current, state, delta):
        if not state or not state.get("qsm_path"):
            return gr.update()
        depth = _volume_array(state["qsm_path"]).shape[2]
        return max(0, min(int(current) + delta, depth - 1))

    _ri = [output_state, slice_slider, vmin_input, vmax_input]
    _ro = [img_qsm, img_phase, img_mag, img_mask]
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
    import os
    host = "127.0.0.1"
    port = _find_free_port(7860, host=host)
    if port != 7860:
        print(f"⚠️  Port 7860 is in use — falling back to {port}")
    url = f"http://{host}:{port}/"
    is_remote = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT")
                     or os.environ.get("SSH_TTY"))
    if is_remote:
        print()
        print("=" * 60)
        print("Running over SSH — auto-open skipped.")
        print(f"Open this URL in your local browser:\n  {url}")
        print("If the host isn't reachable from your laptop, forward the port:")
        print(f"  ssh -L {port}:127.0.0.1:{port} <user>@<host>")
        print("=" * 60)
        print()
    else:
        import webbrowser, socket, threading, time
        def _open_when_ready():
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.3)
                    if s.connect_ex((host, port)) == 0:
                        try:
                            webbrowser.open(url)
                        except Exception:
                            pass
                        return
                time.sleep(0.2)
        threading.Thread(target=_open_when_ready, daemon=True).start()
    app.launch(server_name=host, server_port=port, max_file_size="5gb")
