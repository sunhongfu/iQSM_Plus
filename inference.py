"""
Pure Python inference pipeline for iQSM+.

Replaces the MATLAB preprocessing/postprocessing steps with numpy/scipy
so the tool runs without any MATLAB dependency.
"""

import os
import tempfile
import logging
import time

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from scipy.ndimage import zoom, binary_erosion

# ---------------------------------------------------------------------------
# Locate model code and checkpoints
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))  # iQSM_Plus/
_CKPT_DIR = os.path.join(_HERE, "checkpoints")
_HF_REPO = "sunhongfu/iQSM_Plus"
_CKPT_FILENAMES = [
    "iQSM_plus.pth",
    "LoTLayer_chi.pth",
]

from models.lot_unet import LoT_Unet, LoTLayer  # noqa: E402
from models.unet import Unet  # noqa: E402


class CheckpointNotFoundError(Exception):
    """Raised when model checkpoint files have not been downloaded yet."""


_CKPT_NOT_FOUND_MSG = """\
Model weights not found in checkpoints/.

Run this command on the host machine (outside Docker) before starting the app:

    python run.py --download-checkpoints

This downloads the weights into the checkpoints/ folder that Docker mounts.
Once done, click Run again — no restart needed.\
"""


def _ckpt(filename: str) -> str:
    """Return local path to a checkpoint, raising CheckpointNotFoundError if absent."""
    local = os.path.join(_CKPT_DIR, filename)
    if os.path.exists(local):
        return local
    raise CheckpointNotFoundError(_CKPT_NOT_FOUND_MSG)


# ---------------------------------------------------------------------------
# Laplacian kernel (matches the hardcoded kernel in Inference_iQSMSeries.py)
# ---------------------------------------------------------------------------
_CONV_OP = np.array(
    [
        [[1 / 13, 3 / 26, 1 / 13], [3 / 26, 3 / 13, 3 / 26], [1 / 13, 3 / 26, 1 / 13]],
        [[3 / 26, 3 / 13, 3 / 26], [3 / 13, -44 / 13, 3 / 13], [3 / 26, 3 / 13, 3 / 26]],
        [[1 / 13, 3 / 26, 1 / 13], [3 / 26, 3 / 13, 3 / 26], [1 / 13, 3 / 26, 1 / 13]],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Model loading (cached globally so the Gradio app doesn't reload per call)
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def get_model(device: torch.device) -> nn.Module:
    """Load (or return cached) iQSM+ model."""
    key = str(device)
    if key in _model_cache:
        return _model_cache[key]

    conv_op = torch.from_numpy(_CONV_OP).unsqueeze(0).unsqueeze(0)

    lot_layer = LoTLayer(conv_op)
    lot_layer = nn.DataParallel(lot_layer)
    lot_layer.load_state_dict(
        torch.load(_ckpt("LoTLayer_chi.pth"), map_location=device, weights_only=True)
    )
    lot_layer = lot_layer.module
    lot_layer.eval()

    unet = Unet(4, 16, 1)
    unet = nn.DataParallel(unet)
    unet.load_state_dict(
        torch.load(_ckpt("iQSM_plus.pth"), map_location=device, weights_only=True)
    )
    unet = unet.module
    unet.eval()

    model = LoT_Unet(lot_layer, unet)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    _model_cache[key] = model
    return model


# ---------------------------------------------------------------------------
# Preprocessing helpers  (pure Python equivalents of MATLAB steps)
# ---------------------------------------------------------------------------

def _make_sphere(radius: int) -> np.ndarray:
    """Spherical binary structuring element of given radius."""
    c = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid(c, c, c, indexing='ij')
    return (x**2 + y**2 + z**2) <= radius**2


def _zero_pad(arr: np.ndarray, multiple: int = 16):
    """
    Pad the first three spatial dimensions so each is a multiple of `multiple`.
    Returns (padded_array, positions) where positions = [(x1,x2), (y1,y2), (z1,z2)].
    """
    shape = arr.shape
    pad_spec = []
    positions = []
    for s in shape[:3]:
        total = (multiple - s % multiple) % multiple
        before = total // 2
        after = total - before
        pad_spec.append((before, after))
        positions.append((before, before + s))
    if arr.ndim == 4:
        pad_spec.append((0, 0))
    return np.pad(arr, pad_spec), positions


def _zero_remove(arr: np.ndarray, positions: list) -> np.ndarray:
    """Undo _zero_pad using the saved positions."""
    (x1, x2), (y1, y2), (z1, z2) = positions
    if arr.ndim == 3:
        return arr[x1:x2, y1:y2, z1:z2]
    return arr[x1:x2, y1:y2, z1:z2, :]


def _brain_bbox(mask: np.ndarray, pad: int = 16) -> tuple:
    """
    Bounding box of nonzero mask region, expanded by `pad` voxels on each side
    and clamped to valid array indices.  Returns a tuple of slices (one per dim).
    Falls back to the full volume if the mask is empty.
    """
    nonzero = np.argwhere(mask > 0.5)
    if len(nonzero) == 0:
        return tuple(slice(0, s) for s in mask.shape)
    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0)
    return tuple(
        slice(max(0, int(lo) - pad), min(int(s), int(hi) + 1 + pad))
        for lo, hi, s in zip(mins, maxs, mask.shape)
    )


def _interpolate_phase_to_isotropic(phase: np.ndarray, vox: np.ndarray) -> np.ndarray:
    """
    Interpolate phase data to isotropic resolution.
    Uses complex-domain interpolation to preserve phase wraps
    (equivalent to MATLAB: angle(imresize3(exp(1j*phase), new_size))).
    """
    min_vox = vox.min()
    factors = (vox / min_vox).tolist()  # zoom factors per spatial dim
    if phase.ndim == 4:
        factors = factors + [1.0]  # don't zoom along echo dim

    cplx = np.exp(1j * phase)
    real_z = zoom(cplx.real, factors, order=1)
    imag_z = zoom(cplx.imag, factors, order=1)
    return np.angle(real_z + 1j * imag_z).astype(np.float32)


def _interpolate_volume(vol: np.ndarray, vox: np.ndarray) -> np.ndarray:
    """Linear interpolation of a non-phase volume (magnitude, mask)."""
    min_vox = vox.min()
    factors = (vox / min_vox).tolist()
    if vol.ndim == 4:
        factors = factors + [1.0]
    return zoom(vol.astype(np.float32), factors, order=1)


# ---------------------------------------------------------------------------
# Diagnostics: device selection and memory usage.
#
# Logged via the `logging` module (not print()) specifically because a
# `print()` to a non-TTY stdout (i.e. whenever this runs inside a Docker
# container, as it always does in the OpenRecon deployment) is block-buffered
# by default -- messages can sit in the buffer and never reach the captured
# container log if the process is killed before the buffer flushes. A
# `logging.StreamHandler` calls `flush()` after every single record, so
# switching progress messages over to `logging` means each one is guaranteed
# to reach the log immediately, even if the process is hard-killed a moment
# later (e.g. by the kernel OOM killer, which gives no chance to run any
# cleanup/finally code).
# ---------------------------------------------------------------------------

def _log_device_info(device: torch.device) -> None:
    logging.info("iQSM+ inference device: %s", device)
    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else 0
            props = torch.cuda.get_device_properties(idx)
            logging.info("  GPU %d: %s, %.1f GB total memory, capability %d.%d",
                         idx, props.name, props.total_memory / (1024 ** 3), props.major, props.minor)
        except Exception:
            logging.warning("Could not query CUDA device properties", exc_info=True)
    else:
        logging.warning("torch.cuda.is_available() is False -- running iQSM+ on CPU. This is "
                         "far slower and uses substantially more host RAM for a volume this "
                         "size than the equivalent GPU run.")


def _cgroup_memory_usage_bytes():
    """(usage, limit) in bytes as enforced by Docker's cgroup, or (None, None)
    if unreadable (e.g. not running inside a Linux container)."""
    try:  # cgroup v2
        with open("/sys/fs/cgroup/memory.current") as f:
            usage = int(f.read().strip())
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
            limit = None if raw == "max" else int(raw)
        return usage, limit
    except OSError:
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
            usage = int(f.read().strip())
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            limit = int(f.read().strip())
        return usage, limit
    except OSError:
        return None, None


def _log_mem(tag: str) -> None:
    """Log host RSS, cgroup usage/limit, and GPU memory. Cheap -- call
    liberally around every stage so a future OOM kill leaves a trail of
    breadcrumbs showing how memory was trending, instead of the log just
    going silent with no explanation (as happened on 2026-07-03)."""
    try:
        with open("/proc/self/status") as f:
            status = f.read()
        vmrss_kb = next((int(line.split()[1]) for line in status.splitlines()
                         if line.startswith("VmRSS:")), None)
    except OSError:
        vmrss_kb = None
    rss_str = "%.1f MB" % (vmrss_kb / 1024.0) if vmrss_kb is not None else "unknown"

    usage, limit = _cgroup_memory_usage_bytes()
    if usage is not None and limit:
        cgroup_str = "%.1f/%.1f MB (%.0f%%)" % (usage / (1024 ** 2), limit / (1024 ** 2), 100.0 * usage / limit)
    elif usage is not None:
        cgroup_str = "%.1f MB (no limit found)" % (usage / (1024 ** 2))
    else:
        cgroup_str = "unknown"

    gpu_str = "n/a"
    if torch.cuda.is_available():
        gpu_str = "allocated=%.1f MB reserved=%.1f MB" % (
            torch.cuda.memory_allocated() / (1024 ** 2), torch.cuda.memory_reserved() / (1024 ** 2))

    logging.info("[mem] %-32s host RSS=%s | cgroup=%s | GPU %s", tag, rss_str, cgroup_str, gpu_str)


def _array_mb(arr: np.ndarray) -> float:
    return arr.nbytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Main reconstruction function
# ---------------------------------------------------------------------------

def run_iqsm_plus(
    phase_nii_path: str,
    te: float,
    *,
    mag_nii_path: str | None = None,
    mask_nii_path: str | None = None,
    voxel_size: list | None = None,
    b0_dir: list | None = None,
    b0: float = 3.0,
    eroded_rad: int = 3,
    phase_sign: int = -1,
    output_dir: str | None = None,
    progress_fn=None,
) -> str:
    """
    Run iQSM+ QSM reconstruction in pure Python — single-echo.

    Multi-echo combination (magnitude × TE² weighted averaging) is handled
    *externally* by the caller (see run.py / app.py), exactly as in iQSM:
    one call to this function per echo, then combine the per-echo χ
    volumes. Keeping the combiner outside the model also enables the web
    app's "Echo Selection" panel to recombine subsets without re-running
    inference.

    Parameters
    ----------
    phase_nii_path : str
        Path to wrapped-phase NIfTI file (3D, single-echo).
        Phase convention: phase = -delta_B * gamma * TE.
        If your data uses the opposite sign, negate before calling.
    te : float
        Echo time in **seconds**, e.g. 0.020.
    mag_nii_path : str, optional
        Unused by iQSM+ inference — kept for signature parity with iQSM.
        Magnitude is consumed only by the external multi-echo combiner.
    mask_nii_path : str, optional
        Path to brain-mask NIfTI (3D). If not provided, whole volume is used.
    voxel_size : [x, y, z] mm, optional
        Overrides the voxel size from the NIfTI header.
    b0_dir : [x, y, z], optional
        B0 field direction unit vector. Default: [0, 0, 1] (axial).
    b0 : float
        B0 field strength in Tesla. Default: 3.0.
    eroded_rad : int
        Radius (voxels) for brain-mask erosion. Default: 3.
    phase_sign : int (+1 or -1)
        Multiplier applied to the raw phase before passing it to the model.
        Default −1 matches the original MATLAB pipeline (scanner convention
        phase = +ΔB·γ·TE).  Use +1 if your scanner already stores phase as
        −ΔB·γ·TE (i.e., the sign is already inverted).
    output_dir : str, optional
        Directory where output NIfTI is written. Defaults to a temp dir.
    progress_fn : callable(float, str), optional
        Called with (fraction_done, message) to report progress.

    Returns
    -------
    str
        Absolute path to the output QSM NIfTI file.
    """

    def _log(frac, msg):
        # logging (not just print): a StreamHandler flushes after every record,
        # so this message is guaranteed to reach the captured container log
        # immediately -- print() alone can sit in stdout's block-buffer and be
        # lost if the process is killed (e.g. OOM) before the buffer flushes.
        logging.info("[%3.0f%%] %s", frac * 100, msg)
        print(f"[{frac:.0%}] {msg}", flush=True)
        if progress_fn is not None:
            progress_fn(frac, msg)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="iqsm_plus_")
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(0.0, f"Using device: {device}")
    _log_device_info(device)
    _log_mem("run_iqsm_plus start")

    # ------------------------------------------------------------------
    # 1. Load phase data
    # ------------------------------------------------------------------
    _log(0.05, "Loading phase NIfTI …")
    phase_img = nib.load(phase_nii_path)
    phase = phase_img.get_fdata(dtype=np.float32)
    affine = phase_img.affine

    # Voxel size from header (or user override)
    if voxel_size is not None:
        vox = np.array(voxel_size, dtype=np.float64)
    else:
        zooms = phase_img.header.get_zooms()
        vox = np.array(zooms[:3], dtype=np.float64)
    logging.info("voxel_size (mm) = %s (source: %s)", vox.tolist(),
                 "caller override" if voxel_size is not None else "NIfTI header")

    # B0 direction default
    if b0_dir is None:
        b0_dir = [0.0, 0.0, 1.0]
    b0_dir = np.array(b0_dir, dtype=np.float64)
    b0_dir = b0_dir / np.linalg.norm(b0_dir)

    # TE as a 1-element numpy vector (model expects a tensor)
    te_arr = np.array([float(te)], dtype=np.float32)

    # Ensure phase is single-precision and 3D
    phase = phase.astype(np.float32)
    if phase.ndim != 3:
        raise ValueError(
            f"run_iqsm_plus expects a 3D phase volume, got shape {phase.shape}. "
            "Multi-echo runs should call this function once per echo and "
            "combine outputs externally (see run.py / app.py)."
        )

    imsize_orig = np.array(phase.shape[:3], dtype=int)  # (H, W, D)

    # ------------------------------------------------------------------
    # 2. Load brain mask (optional). Magnitude is unused by iQSM+
    #    inference — it's consumed only by the external multi-echo combiner.
    # ------------------------------------------------------------------
    if mask_nii_path is not None:
        _log(0.10, "Loading brain mask NIfTI …")
        mask = nib.load(mask_nii_path).get_fdata(dtype=np.float32)
    else:
        mask = np.ones(imsize_orig, dtype=np.float32)
        eroded_rad = 0  # no erosion when using whole-head mask

    logging.info("Loaded phase %s (%.1f MB), mag %s (%.1f MB), mask %s (%.1f MB)",
                 phase.shape, _array_mb(phase), mag.shape, _array_mb(mag), mask.shape, _array_mb(mask))
    _log_mem("after loading inputs")

    # ------------------------------------------------------------------
    # 3. Preprocessing (mirrors iQSM_plus.m steps)
    # ------------------------------------------------------------------

    # 3a. Phase sign convention flip (matches MATLAB sf = -1 by default)
    phase = float(phase_sign) * phase

    # 3b. Isotropic interpolation
    interp_flag = not np.allclose(vox, vox.min())
    if interp_flag:
        zoom_factors = (vox / vox.min()).tolist()
        projected_voxels = phase.size * float(np.prod(zoom_factors))
        projected_mb = projected_voxels * 4 / (1024 ** 2)  # float32
        # A near-isotropic protocol should have zoom factors close to 1x on every
        # axis. Factors this large almost always mean voxel_size was computed
        # wrong upstream (e.g. wrong matrixSize field, or a partition/slice count
        # mismatch) rather than a genuinely anisotropic acquisition -- and zoom()
        # will happily try to allocate an output array many GB (or TB) in size for
        # it, which is exactly the kind of silent, un-loggable memory blowup that
        # gets a process OOM-killed with no other trace in the log. Surface it
        # loudly *before* attempting the allocation, not after.
        if max(zoom_factors) > 8.0 or projected_mb > 2048:
            logging.warning(
                "Isotropic interpolation zoom factors are extreme: vox=%s -> factors=%s. "
                "Projected output size for phase alone: ~%.0f MB (%.2f GB). This usually "
                "indicates voxel_size_mm was computed incorrectly upstream (e.g. wrong "
                "matrixSize/partition-count field), not a real anisotropic acquisition. "
                "Proceeding anyway, but this is the most likely cause of an OOM if the "
                "process dies during this step.",
                vox.tolist(), ["%.2f" % f for f in zoom_factors], projected_mb, projected_mb / 1024.0)
        _log(0.15, "Interpolating to isotropic resolution …")
        logging.info("Interpolation: vox=%s -> factors=%s, phase %s -> projected ~%s (%.1f MB)",
                     vox.tolist(), ["%.2f" % f for f in zoom_factors], phase.shape,
                     tuple(int(round(s * f)) for s, f in zip(phase.shape, zoom_factors + [1.0])), projected_mb)
        t0 = time.perf_counter()
        phase = _interpolate_phase_to_isotropic(phase, vox)
        mask = _interpolate_volume(mask, vox)
        vox_iso = np.full(3, vox.min())
        imsize_iso = np.array(phase.shape[:3], dtype=int)
        logging.info("Interpolation done in %.1f s -> phase %s (%.1f MB)",
                     time.perf_counter() - t0, phase.shape, _array_mb(phase))
        _log_mem("after isotropic interpolation")
    else:
        vox_iso = vox.copy()
        imsize_iso = imsize_orig.copy()

    # 3c. Brain-mask erosion
    if eroded_rad > 0:
        _log(0.18, f"Eroding brain mask (radius={eroded_rad}) …")
        struct = _make_sphere(eroded_rad)
        mask_bin = mask > 0.5
        mask = binary_erosion(mask_bin, structure=struct).astype(np.float32)

    # 3d. Dimension permutation so B0 is closest to z-axis
    permute_flag = abs(b0_dir[1]) > abs(b0_dir[2])
    if permute_flag:
        b0_dir[[1, 2]] = b0_dir[[2, 1]]
        phase = np.transpose(phase, (0, 2, 1))
        mask  = np.transpose(mask,  (1, 0, 2))

    # 3e. Crop to brain bounding box (+ 16-voxel context padding)
    bbox = _brain_bbox(mask, pad=16)
    phase_crop = phase[bbox]                    # (H_c, W_c, D_c)
    mask_crop  = mask[bbox]                     # (H_c, W_c, D_c)
    _log(0.22, f"Cropped volume: {phase.shape[:3]} → {phase_crop.shape[:3]}")
    _log_mem("after crop")

    # 3f. Zero-pad cropped volume to multiples of 16
    phase_pad, positions = _zero_pad(phase_crop, 16)
    mask_pad, _          = _zero_pad(mask_crop,  16)
    logging.info("Padded volume: phase_pad %s (%.1f MB)", phase_pad.shape, _array_mb(phase_pad))

    # ------------------------------------------------------------------
    # 4. Deep learning inference
    # ------------------------------------------------------------------
    _log(0.25, "Loading iQSM+ model …")
    t0 = time.perf_counter()
    model = get_model(device)
    logging.info("Model loaded in %.1f s", time.perf_counter() - t0)
    _log_mem("after model load")

    _log(0.30, "Running reconstruction …")

    te_t = torch.from_numpy(te_arr).float().to(device)      # shape (1,)
    b0_t = torch.tensor([b0], dtype=torch.float32).to(device)
    z_prjs_t = torch.from_numpy(b0_dir.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3)

    # phase_pad shape: (H, W, D)
    phase_t = torch.from_numpy(phase_pad).float()           # (H, W, D)
    phase_t = phase_t.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W, D)

    mask_t = torch.from_numpy(mask_pad).float()
    mask_t = mask_t.unsqueeze(0).unsqueeze(0).to(device)    # (1, 1, H, W, D)

    with torch.inference_mode():
        pred_chi = model(phase_t, mask_t, te_t, b0_t, z_prjs_t) * mask_t  # (1, 1, H, W, D)

    pred_chi = pred_chi.squeeze().cpu().numpy().astype(np.float32)        # (H, W, D)

    # ------------------------------------------------------------------
    # 5. Post-processing
    # ------------------------------------------------------------------
    _log(0.82, "Post-processing …")

    # 5a. Remove zero-padding from cropped result, paste back into full volume
    chi_crop = _zero_remove(pred_chi, positions)            # (H_c, W_c, D_c)
    chi_fitted = np.zeros(phase.shape[:3], dtype=np.float32)
    chi_fitted[bbox] = chi_crop                             # (H, W, D)

    # 5b. Undo dimension permutation
    if permute_flag:
        chi_fitted = np.transpose(chi_fitted, (0, 2, 1))

    # 5c. Undo isotropic interpolation (back to original resolution)
    if interp_flag:
        factors = (imsize_orig / imsize_iso).tolist()
        logging.info("Undoing isotropic interpolation: %s -> factors=%s -> projected %s",
                     chi_fitted.shape, ["%.3f" % f for f in factors],
                     tuple(int(round(s * f)) for s, f in zip(chi_fitted.shape, factors)))
        t0 = time.perf_counter()
        chi_fitted = zoom(chi_fitted, factors, order=1)
        logging.info("Undo-interpolation done in %.1f s -> %s", time.perf_counter() - t0, chi_fitted.shape)
        _log_mem("after undo-interpolation")

    # ------------------------------------------------------------------
    # 6. Save output NIfTI
    # ------------------------------------------------------------------
    _log(0.95, "Saving output NIfTI …")
    out_path = os.path.join(output_dir, "iQSM_plus.nii.gz")
    out_img = nib.Nifti1Image(chi_fitted, affine)
    # Store voxel size from original header
    out_img.header.set_zooms(tuple(vox_iso if not interp_flag else vox))
    nib.save(out_img, out_path)

    _log_mem("run_iqsm_plus end")
    _log(1.0, f"Done! Saved to {out_path} (total {time.perf_counter() - t_start:.1f} s)")
    return out_path
