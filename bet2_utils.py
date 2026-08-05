"""
Automatic brain-mask generation via FSL's bet2, vendored at vendor/bet2/ (bin + its ~15
runtime shared libraries -- see vendor/bet2/README.md for provenance/license). Not a full
FSL install.

Used by run.py / app.py when no --mask / uploaded mask is supplied: bet2 runs on the
magnitude volume to generate one automatically, so a brain mask is available by default
without requiring a separate manual step. A missing/failing bet2 never aborts
reconstruction -- callers fall back to whole-head (no mask) instead.

Uses print() rather than the `logging` module so messages appear in app.py's live Log
panel too (its background inference thread redirects sys.stdout, not the logging module).
"""

import os
import subprocess
import traceback
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

_BET2_CANDIDATES = [
    os.environ.get("BET2_DIR"),
    "/opt/bet2",                                    # Docker image, if built that way
    str(REPO_ROOT / "vendor" / "bet2"),             # local checkout (this repo's own copy)
]
BET2_DIR = next((p for p in _BET2_CANDIDATES
                 if p and os.path.isfile(os.path.join(p, "bin", "bet2"))), None)


def first_3d_volume(nii_path, output_dir, name="mag_for_bet2"):
    """bet2 needs a plain 3D volume. Returns nii_path unchanged if it's already 3D,
    otherwise saves and returns the path to its first volume along the last axis."""
    img = nib.load(str(nii_path))
    if len(img.shape) == 3:
        return str(nii_path)
    data = np.asarray(img.dataobj)[..., 0].astype(np.float32)
    out_path = os.path.join(output_dir, f"{name}.nii.gz")
    nib.save(nib.Nifti1Image(data, img.affine), out_path)
    return out_path


def run_bet2(mag_nii_path, output_dir, fractional_intensity=0.5):
    """Run bet2 on a (3D) magnitude volume, returning the path to the binary brain
    mask NIfTI, or None if bet2 isn't available or the run fails."""
    if BET2_DIR is None:
        print("bet2 not found (checked $BET2_DIR, /opt/bet2, ./vendor/bet2) -- "
              "skipping automatic brain extraction")
        return None

    bet2Bin = os.path.join(BET2_DIR, "bin", "bet2")
    outPrefix = os.path.join(output_dir, "bet2_out")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.join(BET2_DIR, "lib")
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")

    tic = perf_counter()
    try:
        result = subprocess.run(
            [bet2Bin, mag_nii_path, outPrefix, "-m", "-f", str(fractional_intensity)],
            env=env, capture_output=True, text=True, timeout=120,
        )
    except Exception:
        print(f"bet2 failed to run -- skipping brain extraction:\n{traceback.format_exc()}")
        return None

    if result.returncode != 0:
        print(f"bet2 exited with code {result.returncode} -- skipping brain extraction. "
              f"stdout={result.stdout} stderr={result.stderr}")
        return None

    maskPath = outPrefix + "_mask.nii.gz"
    if not os.path.exists(maskPath):
        print(f"bet2 completed but mask file not found at {maskPath} -- skipping brain extraction")
        return None

    mask = nib.load(maskPath).get_fdata()
    print(f"bet2 brain extraction completed in {perf_counter() - tic:.1f}s -> "
          f"{Path(maskPath).name} ({100.0 * mask.sum() / mask.size:.1f}% of voxels)")
    return maskPath
