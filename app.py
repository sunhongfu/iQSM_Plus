"""
iQSM+ – Gradio Web Interface

Launch:
    python app.py                   # CPU
    python app.py --server-port 8080
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

_pipeline_lock = threading.Lock()

_HERE     = REPO_ROOT
_DEMO_DIR = _HERE / "demo"

_DISPLAY_VMIN = -0.2   # ppm  (QSM)
_DISPLAY_VMAX =  0.2


# ---------------------------------------------------------------------------
# Streaming log writer
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


# ---------------------------------------------------------------------------
# Path / file helpers
# ---------------------------------------------------------------------------

def _natural_key(f):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", _to_path(f).name)]


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


def _mat_to_nii(mat_path: str) -> str:
    """Load a single-array .mat file and save as a temporary NIfTI. Returns path."""
    import scipy.io
    mat = scipy.io.loadmat(mat_path)
    arrays = {k: v for k, v in mat.items()
              if not k.startswith("_") and isinstance(v, np.ndarray) and v.ndim >= 3}
    if not arrays:
        raise ValueError(f"No 3D+ numeric array found in {mat_path}")
    arr = max(arrays.values(), key=lambda a: a.size).squeeze().astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(nib.Nifti1Image(arr, np.eye(4)), tmp.name)
    return tmp.name


def _detect_echoes(paths):
    """Return echo count: reads 4D shape for a single NIfTI, else len(paths)."""
    if not paths:
        return None
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
        return 1
    return len(paths)


def _format_order(paths):
    paths = [p for p in paths if p]
    if not paths:
        return ""
    sorted_paths = sorted(paths, key=_natural_key)
    entries = [f"{i+1}. {Path(p).name}" for i, p in enumerate(sorted_paths)]
    n_cols = 4 if len(entries) > 5 else 1
    col_width = max(len(e) for e in entries) + 3
    rows = []
    for i in range(0, len(entries), n_cols):
        chunk = entries[i:i + n_cols]
        rows.append("".join(e.ljust(col_width) for e in chunk).rstrip())
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Slice viewer helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _volume_array(nii_path):
    return nib.load(str(nii_path)).get_fdata().astype(np.float32)


def _make_slice_image(nii_path, slice_idx=None, vmin=-0.2, vmax=0.2):
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
# Run configuration log block
# ---------------------------------------------------------------------------

def _print_run_config(work_dir, mode, phase_paths, te_list_ms, mag_path, mask_path,
                      voxel_size, voxel_user_set, b0, b0_dir, eroded_rad, phase_sign,
                      orig_phase_paths=None, orig_mag_path=None, orig_mask_path=None):
    print("============================")
    print("RUN CONFIGURATION")
    print("============================")
    if mode == "4d":
        print(f"Input mode      : Single 4D volume ({len(te_list_ms)} echoes)")
        print(f"Phase file      : {Path(phase_paths[0]).name}")
    elif len(phase_paths) == 1:
        print(f"Input mode      : Single 3D phase")
        print(f"Phase file      : {Path(phase_paths[0]).name}")
    else:
        print(f"Input mode      : Multiple 3D echoes ({len(phase_paths)} files)")
        print("Phase files (in processing order):")
        for i, (p, te) in enumerate(zip(phase_paths, te_list_ms), 1):
            print(f"  {i}. {Path(p).name}    TE = {te} ms")
    print(f"TE values (ms)  : {', '.join(str(t) for t in te_list_ms)}")
    print(f"Magnitude       : {Path(mag_path).name if mag_path else '(none)'}")
    print(f"Brain mask      : {Path(mask_path).name if mask_path else '(none)'}")
    if voxel_size and voxel_user_set:
        print(f"Voxel size (mm) : {' '.join(f'{v:.4g}' for v in voxel_size)}  (user-specified)")
    elif voxel_size:
        print(f"Voxel size (mm) : {' '.join(f'{v:.4g}' for v in voxel_size)}  (default for .mat input)")
    else:
        print(f"Voxel size (mm) : (from NIfTI header)")
    print(f"B0 (T)          : {b0}")
    print(f"B0 direction    : {b0_dir if b0_dir else '[0 0 1] (default)'}")
    print(f"Mask erosion    : {eroded_rad} voxels")
    print(f"Reverse phase   : {'yes' if phase_sign == 1 else 'no (default)'}")
    print(f"Staging dir     : {work_dir}")
    if mode in ("4d", "single"):
        te_sec = [t / 1000.0 for t in te_list_ms]
        cmd = ["python run.py"]
        phase_cli = orig_phase_paths[0] if orig_phase_paths else str(phase_paths[0])
        cmd.append(f"--phase {phase_cli}")
        cmd.append("--te " + " ".join(f"{t:.6g}" for t in te_sec))
        if mag_path:
            mag_cli = orig_mag_path if orig_mag_path else str(mag_path)
            cmd.append(f"--mag {mag_cli}")
        if mask_path:
            mask_cli = orig_mask_path if orig_mask_path else str(mask_path)
            cmd.append(f"--mask {mask_cli}")
        if voxel_user_set and voxel_size:
            cmd.append("--voxel-size " + " ".join(f"{v:.4g}" for v in voxel_size))
        if b0_dir:
            cmd.append("--b0-dir " + " ".join(f"{v:.4g}" for v in b0_dir))
        if b0 != 3.0:
            cmd.append(f"--b0 {b0}")
        if eroded_rad != 3:
            cmd.append(f"--eroded-rad {eroded_rad}")
        if phase_sign != -1:
            cmd.append("--reverse-phase-sign 1")
        _sep = "-" * 56
        print()
        print(_sep)
        print("  Equivalent command-line invocation:")
        print(_sep)
        print("  " + " \\\n      ".join(cmd))
        print(_sep)
    print()


# ---------------------------------------------------------------------------
# Background inference thread
# ---------------------------------------------------------------------------

def _run_thread(job, work_dir, mode, phase_paths, te_list_ms, mag_path, mask_path,
                voxel_size, voxel_user_set, b0, b0_dir, eroded_rad, phase_sign, vmin, vmax,
                orig_phase_paths=None, orig_mag_path=None, orig_mask_path=None):
    log_q = job["log_queue"]
    orig = sys.stdout
    sys.stdout = _QueueWriter(log_q, orig)
    try:
        with _pipeline_lock:
            job["status"] = "running"
            te_list_s = [t / 1000.0 for t in te_list_ms]

            _print_run_config(work_dir, mode, phase_paths, te_list_ms, mag_path, mask_path,
                              voxel_size, voxel_user_set, b0, b0_dir, eroded_rad, phase_sign,
                              orig_phase_paths=orig_phase_paths, orig_mag_path=orig_mag_path,
                              orig_mask_path=orig_mask_path)

            out_dir = work_dir / "iqsmplus_output"
            out_dir.mkdir(exist_ok=True)

            # iQSM+ handles multi-echo natively; combine multiple 3D files into a 4D NIfTI
            if mode == "multi" and len(phase_paths) > 1:
                print("============================")
                print(f"COMBINING {len(phase_paths)} 3D ECHOES → 4D VOLUME")
                print("============================")
                imgs = [nib.load(str(p)) for p in phase_paths]
                affine = imgs[0].affine
                data_4d = np.stack([img.get_fdata(dtype=np.float32) for img in imgs], axis=-1)
                combined_path = work_dir / "phase_4d.nii.gz"
                nib.save(nib.Nifti1Image(data_4d, affine), str(combined_path))
                phase_input_path = str(combined_path)
                print(f"  Combined shape: {data_4d.shape}")
            else:
                phase_input_path = str(phase_paths[0])

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

    except Exception as exc:
        import traceback
        print(f"\n❌ Error: {exc}")
        print(traceback.format_exc())
        job["status"] = "error"
    finally:
        sys.stdout = orig
        log_q.put(None)


def _result_files(job):
    files = [p for p in (job.get("qsm_path"),) if p]
    return files or None


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


def _stream_job(job):
    log = ""
    while True:
        msg = job["log_queue"].get()
        if msg is None:
            break
        log += msg + "\n"
        state, slider = _state_and_slider_update(job)
        yield (log, _result_files(job), job.get("qsm_image"), state, slider)
    state, slider = _state_and_slider_update(job)
    yield (log, _result_files(job), job.get("qsm_image"), state, slider)


# ---------------------------------------------------------------------------
# Main pipeline callback
# ---------------------------------------------------------------------------

def run_pipeline(phase_files, te_ms_str, mag_file, mask_file, voxel_str,
                 b0_val, b0dir_str, eroded_rad, negate_phase, vmin, vmax,
                 mag_orig_path=None, mask_orig_path=None):
    _noop = (None, None, None, gr.update())

    te_list_ms = [float(t.strip()) for t in te_ms_str.replace(",", " ").split() if t.strip()]
    if not te_list_ms:
        yield ("❌ Enter echo times (ms)", *_noop)
        return

    if not phase_files:
        yield ("❌ Upload phase file(s) — a single 3D/4D NIfTI/MAT or multiple 3D echoes", *_noop)
        return

    files = sorted(phase_files if isinstance(phase_files, list) else [phase_files],
                   key=_natural_key)
    missing = [str(_to_path(f)) for f in files
               if not (_to_path(f) and _to_path(f).exists())]
    if missing:
        yield ("❌ Some uploaded files no longer exist. Click 'Clear All' and re-upload:\n  "
               + "\n  ".join(missing), *_noop)
        return

    # Determine mode
    paths_str = [str(_to_path(f)) for f in files]
    n_echoes_detected = _detect_echoes(paths_str)
    if len(files) == 1 and n_echoes_detected and n_echoes_detected > 1:
        mode = "4d"
        n_phase_echoes = n_echoes_detected
    else:
        mode = "multi" if len(files) > 1 else "single"
        n_phase_echoes = len(files)

    if n_phase_echoes != len(te_list_ms):
        yield (f"❌ {n_phase_echoes} echo(es) but {len(te_list_ms)} TE value(s) — counts must match",
               *_noop)
        return

    work_dir = Path(tempfile.mkdtemp(prefix="iqsmplus_"))

    orig_phase_paths = [str(_to_path(f)) for f in files]
    _omag = _to_path(mag_file)
    orig_mag_path = mag_orig_path or (str(_omag) if _omag and _omag.exists() else None)
    _omask = _to_path(mask_file)
    orig_mask_path = mask_orig_path or (str(_omask) if _omask and _omask.exists() else None)

    # Stage phase files (convert .mat → NIfTI if needed)
    phase_paths = []
    for f in files:
        src = _to_path(f)
        if src.suffix.lower() == ".mat":
            nii = _mat_to_nii(str(src))
            dst = work_dir / (src.stem + ".nii.gz")
            shutil.move(nii, dst)
        else:
            dst = work_dir / src.name
            shutil.copy(src, dst)
        phase_paths.append(dst)

    # Stage magnitude (convert .mat → NIfTI if needed)
    mag_path = None
    if mag_file:
        src = _to_path(mag_file)
        if src and src.exists():
            if src.suffix.lower() == ".mat":
                nii = _mat_to_nii(str(src))
                dst = work_dir / (src.stem + "_mag.nii.gz")
                shutil.move(nii, dst)
            else:
                dst = work_dir / src.name
                shutil.copy(src, dst)
            mag_path = str(dst)

    # Stage mask (convert .mat → NIfTI if needed)
    mask_path = None
    if mask_file:
        src = _to_path(mask_file)
        if src and src.exists():
            if src.suffix.lower() == ".mat":
                nii = _mat_to_nii(str(src))
                dst = work_dir / (src.stem + "_mask.nii.gz")
                shutil.move(nii, dst)
            else:
                dst = work_dir / src.name
                shutil.copy(src, dst)
            mask_path = str(dst)

    # Voxel size
    voxel_size = None
    voxel_user_set = False
    if voxel_str and voxel_str.strip():
        try:
            voxel_size = [float(v) for v in voxel_str.replace(",", " ").split()]
        except ValueError:
            yield ("❌ Invalid voxel size — enter three numbers, e.g. 1 1 2", *_noop)
            return
        if len(voxel_size) != 3:
            yield ("❌ Voxel size must have exactly 3 values (x y z)", *_noop)
            return
        voxel_user_set = True
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
              voxel_size, voxel_user_set, float(b0_val), b0_dir, int(eroded_rad), phase_sign,
              float(vmin), float(vmax),
              orig_phase_paths, orig_mag_path, orig_mask_path),
        daemon=True,
    ).start()

    yield from _stream_job(job)


# ---------------------------------------------------------------------------
# Demo loader
# ---------------------------------------------------------------------------

def load_demo():
    ph_path = _DEMO_DIR / "ph_multi_echo.nii.gz"
    if not ph_path.exists():
        return ([], "❌ Demo data not found.\nRun: python run.py --download-demo",
                None, None, None, "", None, None,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), None, None)

    paths = [str(ph_path)]
    mag = _DEMO_DIR / "mag_multi_echo.nii.gz"
    mask = _DEMO_DIR / "mask_multi_echo.nii.gz"
    mag_path  = str(mag)  if mag.exists()  else None
    mask_path = str(mask) if mask.exists() else None

    te_ms_str     = ""
    n_ech         = _detect_echoes(paths)
    voxel_update  = gr.update()
    b0_update     = gr.update()
    b0dir_update  = gr.update()
    eroded_update = gr.update()
    negate_update = gr.update()

    try:
        import json
        with open(_DEMO_DIR / "params.json") as fh:
            params = json.load(fh)
        te_values = params.get("TE_seconds", [])
        if isinstance(te_values, list) and te_values:
            te_ms_str = ", ".join(f"{t * 1000:.4g}" for t in te_values)
            n_ech = len(te_values)
        elif isinstance(te_values, (int, float)):
            te_ms_str = f"{te_values * 1000:.4g}"
            n_ech = 1
        vox = params.get("voxel_size_mm")
        if vox:
            voxel_update = gr.update(value=" ".join(f"{v:.4g}" for v in vox))
        b0 = params.get("B0_Tesla")
        if b0 is not None:
            b0_update = gr.update(value=float(b0))
        b0dir = params.get("B0_dir")
        if b0dir:
            b0dir_update = gr.update(value=" ".join(f"{v:.4g}" for v in b0dir))
        er = params.get("eroded_rad")
        if er is not None:
            eroded_update = gr.update(value=int(er))
        pc = params.get("phase_sign_convention")
        if pc is not None:
            negate_update = gr.update(value=(pc == 1))
    except Exception:
        pass

    return (paths, _format_order(paths), n_ech, gr.update(), gr.update(),
            te_ms_str, mag_path, mask_path,
            voxel_update, b0_update, b0dir_update, eroded_update, negate_update,
            mag_path, mask_path)


# ---------------------------------------------------------------------------
# CSS / JS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Section titles — coloured for quick scanning */
.gradio-container h3 {
    padding-left: 12px !important;
    color: #1d4ed8 !important;
    border-left: 4px solid #1d4ed8 !important;
    margin-left: 4px !important;
}
.dark .gradio-container h3 {
    color: #60a5fa !important;
    border-left-color: #60a5fa !important;
}

/* Section panels — thicker, more visible borders */
.dr-section {
    margin-bottom: 16px !important;
    border: 2px solid #4b5563 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08) !important;
    overflow: hidden !important;
}
.dark .dr-section {
    border-color: #9ca3af !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.30) !important;
}

/* Secondary buttons */
button.secondary {
    background: #f3f4f6 !important;
    border: 1px solid #9ca3af !important;
    color: #111827 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08) !important;
    transition: background 0.12s ease-in-out, border-color 0.12s ease-in-out !important;
}
button.secondary:hover {
    background: #e5e7eb !important;
    border-color: #6b7280 !important;
}
button.secondary:active {
    background: #d1d5db !important;
}
.dark button.secondary {
    background: #374151 !important;
    border: 1px solid #6b7280 !important;
    color: #f3f4f6 !important;
}
.dark button.secondary:hover {
    background: #4b5563 !important;
    border-color: #9ca3af !important;
    color: #ffffff !important;
}
.dark button.secondary:active {
    background: #1f2937 !important;
}
"""

FORCE_DARK_JS = """
<script>
  (function() {
    try {
      var url = new URL(window.location.href);
      if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.replace(url.toString());
      }
    } catch (e) {}
  })();
</script>
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="iQSM+") as app:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>{FORCE_DARK_JS}", padding=False)
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

    accumulated = gr.State([])
    mag_orig    = gr.State(None)
    mask_orig   = gr.State(None)

    with gr.Row(equal_height=False):

        # ── Left column: inputs ──────────────────────────────────────
        with gr.Column(scale=4):

            # ── 1. Phase ──────────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Phase  *(multiple 3D echoes OR a single 4D volume; .nii / .nii.gz / .mat)*")
                phase_input = gr.File(
                    file_count="multiple",
                    file_types=[".nii", ".gz", ".mat"],
                    show_label=False,
                    height=180,
                )
                sorted_order = gr.Textbox(
                    label="For multiple 3D echo files, confirm ascending TE order; rename files if not sorted correctly",
                    interactive=False,
                    placeholder="Upload files to see sorted order",
                    lines=5,
                    max_lines=15,
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear All", variant="stop")
                    demo_btn  = gr.Button("Load Demo Data", variant="secondary")

            # ── 2. Magnitude ──────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Magnitude  *(optional — improves multi-echo fitting; .nii / .nii.gz / .mat)*")
                mag_file = gr.File(
                    file_count="single",
                    file_types=[".nii", ".gz", ".mat"],
                    show_label=False,
                )

            # ── 3. Echo Times ──────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Echo Times")
                with gr.Row():
                    first_te     = gr.Number(label="First TE (ms)", precision=3)
                    echo_spacing = gr.Number(label="Echo Spacing (ms)", precision=3)
                    n_echoes     = gr.Number(label="Number of Echoes", precision=0, interactive=True)
                fill_te_btn = gr.Button("Compute full train of echo times ↓", variant="secondary")
                te_ms = gr.Textbox(
                    label="Echo Times (ms)",
                    placeholder="Single echo: 20    Multi-echo: 3.2, 6.5, 9.8, 13.0",
                    info="One value for single-echo; comma-separated for multi-echo. Use fields above for evenly spaced echoes.",
                )

            # ── 4. Brain Mask ──────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Brain Mask  *(optional — if omitted, all voxels are processed; .nii / .nii.gz / .mat)*")
                mask_file = gr.File(
                    file_count="single",
                    file_types=[".nii", ".gz", ".mat"],
                    show_label=False,
                )

            # ── 5. Parameters ─────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Parameters")
                with gr.Row():
                    b0_val     = gr.Number(value=3.0, label="B0 (Tesla)", minimum=0.1, maximum=14.0, step=0.5)
                    eroded_rad = gr.Slider(label="Mask erosion (voxels)", minimum=0, maximum=10, step=1, value=3)
                with gr.Row():
                    voxel_str  = gr.Textbox(
                        label="Voxel size — x y z (mm)",
                        placeholder="e.g.  1 1 2   (leave blank to read from NIfTI header)",
                    )
                    b0dir_str  = gr.Textbox(
                        label="B0 direction — x y z (unit vector)",
                        placeholder="e.g.  0.1 0.0 0.995   (leave blank for axial [0 0 1])",
                        info="For non-axial acquisitions, enter the B0 field direction unit vector.",
                    )
                negate_phase = gr.Checkbox(
                    label="Reverse phase sign",
                    value=False,
                    info="Enable if iron-rich deep grey matter appears dark (rather than bright) in the QSM output.",
                )

            run_btn = gr.Button("Run Pipeline", variant="primary")

        # ── Right column: log + results + vis ───────────────────────
        with gr.Column(scale=5):

            # ── 6. Log ─────────────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Log")
                log_out = gr.Textbox(
                    show_label=False, lines=8, max_lines=20,
                    interactive=False, autoscroll=True,
                )

            # ── 7. Results ─────────────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Results  *(click the file size on the right to download)*")
                result_file = gr.File(show_label=False, file_count="multiple")

            # ── 8. Visualisation ───────────────────────────────────
            with gr.Group(elem_classes="dr-section"):
                gr.Markdown("### Visualisation")
                img_qsm = gr.Image(
                    label="QSM",
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=400,
                )
                with gr.Row(equal_height=True):
                    prev_btn = gr.Button("◀ Prev", scale=1)
                    slice_slider = gr.Slider(
                        minimum=0, maximum=0, value=0, step=1,
                        show_label=False, container=False,
                        interactive=True, scale=8,
                    )
                    next_btn = gr.Button("Next ▶", scale=1)
                with gr.Row():
                    vmin_input = gr.Number(value=_DISPLAY_VMIN, label="QSM min (ppm)", precision=3)
                    vmax_input = gr.Number(value=_DISPLAY_VMAX, label="QSM max (ppm)", precision=3)
            output_state = gr.State((None,))

    # ── Callbacks ───────────────────────────────────────────────────────

    def add_files(new_files, current):
        if not new_files:
            return current, _format_order(current), _detect_echoes(current), None, gr.update()
        files = new_files if isinstance(new_files, list) else [new_files]
        new_paths = [str(_to_path(f)) for f in files]
        new_names = {Path(p).name for p in new_paths}
        kept = [p for p in current if Path(p).name not in new_names]
        updated = kept + new_paths
        voxel_update = gr.update()
        first_nii = next(
            (p for p in new_paths
             if Path(p).name.lower().endswith(".nii") or Path(p).name.lower().endswith(".nii.gz")),
            None,
        )
        if first_nii:
            try:
                zooms = nib.load(first_nii).header.get_zooms()
                voxel_update = gr.update(value=f"{zooms[0]:.4g} {zooms[1]:.4g} {zooms[2]:.4g}")
            except Exception:
                pass
        return updated, _format_order(updated), _detect_echoes(updated), None, voxel_update

    def clear_files():
        return [], "", None, None, gr.update(value="")

    def compute_te_list(first, spacing, n):
        if not first or not spacing or not n:
            return gr.update()
        tes = [round(first + i * spacing, 4) for i in range(int(n))]
        return ", ".join(f"{t:g}" for t in tes)

    phase_input.upload(
        add_files,
        inputs=[phase_input, accumulated],
        outputs=[accumulated, sorted_order, n_echoes, phase_input, voxel_str],
    )
    clear_btn.click(clear_files, outputs=[accumulated, sorted_order, n_echoes, phase_input, voxel_str])
    fill_te_btn.click(compute_te_list, inputs=[first_te, echo_spacing, n_echoes], outputs=te_ms)
    demo_btn.click(
        load_demo,
        outputs=[accumulated, sorted_order, n_echoes, first_te, echo_spacing, te_ms,
                 mag_file, mask_file, voxel_str, b0_val, b0dir_str, eroded_rad, negate_phase,
                 mag_orig, mask_orig],
    )
    mag_file.upload(lambda f: None, inputs=mag_file, outputs=mag_orig)
    mask_file.upload(lambda f: None, inputs=mask_file, outputs=mask_orig)

    run_btn.click(
        run_pipeline,
        inputs=[accumulated, te_ms, mag_file, mask_file, voxel_str,
                b0_val, b0dir_str, eroded_rad, negate_phase, vmin_input, vmax_input,
                mag_orig, mask_orig],
        outputs=[log_out, result_file, img_qsm, output_state, slice_slider],
    )

    def render_slice(state, idx, vmin, vmax):
        qsm_path = state[0] if state else None
        return _make_slice_image(qsm_path, idx, vmin, vmax) if qsm_path else None

    def step_slice(current, state, delta):
        if state is None or state[0] is None:
            return gr.update()
        depth = _volume_array(str(state[0])).shape[2]
        return max(0, min(int(current) + delta, depth - 1))

    _render_inputs  = [output_state, slice_slider, vmin_input, vmax_input]
    _render_outputs = [img_qsm]

    prev_btn.click(lambda c, s: step_slice(c, s, -1), inputs=[slice_slider, output_state], outputs=slice_slider)
    next_btn.click(lambda c, s: step_slice(c, s, +1), inputs=[slice_slider, output_state], outputs=slice_slider)
    slice_slider.change(render_slice, inputs=_render_inputs, outputs=_render_outputs)
    vmin_input.change(render_slice, inputs=_render_inputs, outputs=_render_outputs)
    vmax_input.change(render_slice, inputs=_render_inputs, outputs=_render_outputs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="iQSM+ Gradio server")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    args = parser.parse_args()
    app.launch(server_name=args.server_name, server_port=args.server_port)
