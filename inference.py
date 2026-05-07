"""
Pure Python inference pipeline for iQSM+.

Replaces the MATLAB preprocessing/postprocessing steps with numpy/scipy
so the tool runs without any MATLAB dependency.
"""

import os
import tempfile

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
        print(f"[{frac:.0%}] {msg}")
        if progress_fn is not None:
            progress_fn(frac, msg)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="iqsm_plus_")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _log(0.0, f"Using device: {device}")

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

    # ------------------------------------------------------------------
    # 3. Preprocessing (mirrors iQSM_plus.m steps)
    # ------------------------------------------------------------------

    # 3a. Phase sign convention flip (matches MATLAB sf = -1 by default)
    phase = float(phase_sign) * phase

    # 3b. Isotropic interpolation
    interp_flag = not np.allclose(vox, vox.min())
    if interp_flag:
        _log(0.15, "Interpolating to isotropic resolution …")
        phase = _interpolate_phase_to_isotropic(phase, vox)
        mask = _interpolate_volume(mask, vox)
        vox_iso = np.full(3, vox.min())
        imsize_iso = np.array(phase.shape[:3], dtype=int)
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

    # 3f. Zero-pad cropped volume to multiples of 16
    phase_pad, positions = _zero_pad(phase_crop, 16)
    mask_pad, _          = _zero_pad(mask_crop,  16)

    # ------------------------------------------------------------------
    # 4. Deep learning inference
    # ------------------------------------------------------------------
    _log(0.25, "Loading iQSM+ model …")
    model = get_model(device)

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
        chi_fitted = zoom(chi_fitted, factors, order=1)

    # ------------------------------------------------------------------
    # 6. Save output NIfTI
    # ------------------------------------------------------------------
    _log(0.95, "Saving output NIfTI …")
    out_path = os.path.join(output_dir, "iQSM_plus.nii.gz")
    out_img = nib.Nifti1Image(chi_fitted, affine)
    # Store voxel size from original header
    out_img.header.set_zooms(tuple(vox_iso if not interp_flag else vox))
    nib.save(out_img, out_path)

    _log(1.0, f"Done! Saved to {out_path}")
    return out_path
