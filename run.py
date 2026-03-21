"""
iQSM+ – Command-line interface
Usage:
    python run.py --demo                                              # download & run demo
    python run.py --config config.yaml                               # from config file
    python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz
    python run.py --phase ph.nii.gz --te 0.0032 0.0065 0.0098 --mag mag.nii.gz
    python run.py --config config.yaml --output ./other/             # CLI overrides config
    python run.py --help
"""

import argparse
import os
import tempfile
import urllib.request

import yaml

from inference import run_iqsm_plus


# ---------------------------------------------------------------------------
# Demo data (mirrors app.py)
# ---------------------------------------------------------------------------
_DEMO_BASE      = "https://github.com/sunhongfu/iQSM_Plus/releases/download/v1.0-demo"
_DEMO_CACHE_DIR = os.path.join(tempfile.gettempdir(), "iqsm_plus_demo")
_DEMO_TE        = [0.0032, 0.0065, 0.0098, 0.0131, 0.0164, 0.0197, 0.0231, 0.0264]


def _download_demo() -> tuple[str, str, str]:
    os.makedirs(_DEMO_CACHE_DIR, exist_ok=True)
    files = {
        "ph_multi_echo.nii.gz":   f"{_DEMO_BASE}/ph_multi_echo.nii.gz",
        "mag_multi_echo.nii.gz":  f"{_DEMO_BASE}/mag_multi_echo.nii.gz",
        "mask_multi_echo.nii.gz": f"{_DEMO_BASE}/mask_multi_echo.nii.gz",
    }
    for name, url in files.items():
        dest = os.path.join(_DEMO_CACHE_DIR, name)
        if not os.path.exists(dest):
            print(f"Downloading {name} …")
            urllib.request.urlretrieve(url, dest)
    return (
        os.path.join(_DEMO_CACHE_DIR, "ph_multi_echo.nii.gz"),
        os.path.join(_DEMO_CACHE_DIR, "mag_multi_echo.nii.gz"),
        os.path.join(_DEMO_CACHE_DIR, "mask_multi_echo.nii.gz"),
    )


def run_demo(output_dir: str):
    te_str = ", ".join(f"{te:.4g}" for te in _DEMO_TE)
    print("── iQSM+ demo ─────────────────────────────────────────────────")
    print(f"  Data cached at: {_DEMO_CACHE_DIR}")
    print(f"  Parameters: 64×64×32 crop, 1×1×1 mm, 8 echoes, B0=3T")
    print(f"  TE: {te_str} s")
    print("────────────────────────────────────────────────────────────────")
    phase_path, mag_path, mask_path = _download_demo()
    qsm_path = run_iqsm_plus(
        phase_nii_path=phase_path,
        te_values=_DEMO_TE,
        mag_nii_path=mag_path,
        mask_nii_path=mask_path,
        voxel_size=[1, 1, 1],
        b0=3.0,
        eroded_rad=0,
        phase_sign=-1,
        output_dir=output_dir,
    )
    print(f"\nOutputs:")
    print(f"  QSM (susceptibility): {qsm_path}")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--demo",   action="store_true")
    pre.add_argument("--config", metavar="FILE")
    known, _ = pre.parse_known_args()

    if known.demo:
        pre2 = argparse.ArgumentParser(add_help=False)
        pre2.add_argument("--output", default="./iqsm_plus_demo_output")
        pre2.add_argument("--demo", action="store_true")
        a, _ = pre2.parse_known_args()
        run_demo(a.output)
        return

    config = _load_config(known.config) if known.config else {}

    parser = argparse.ArgumentParser(
        description="iQSM+: QSM reconstruction from single- or multi-echo MRI phase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demo",       action="store_true",
                        help="Download and run the built-in demo dataset.")
    parser.add_argument("--config",     metavar="FILE",
                        help="YAML config file. CLI arguments override config values.")
    parser.add_argument("--phase",      metavar="FILE",
                        help="Wrapped phase NIfTI, 3D (single-echo) or 4D (multi-echo).")
    parser.add_argument("--te",         nargs="+", type=float, metavar="SEC",
                        help="Echo time(s) in seconds (e.g. --te 0.020 or --te 0.0032 0.0065).")
    parser.add_argument("--mag",        metavar="FILE", default=None,
                        help="Magnitude NIfTI (optional).")
    parser.add_argument("--mask",       metavar="FILE", default=None,
                        help="Brain mask NIfTI (optional; ones if omitted).")
    parser.add_argument("--output",     metavar="DIR",  default="./iqsm_plus_output",
                        help="Output directory.")
    parser.add_argument("--b0",         type=float, default=3.0,
                        help="B0 field strength in Tesla.")
    parser.add_argument("--b0-dir",     nargs=3, type=float, metavar=("X","Y","Z"),
                        default=None,
                        help="B0 direction unit vector (default: read from header or [0,0,1]).")
    parser.add_argument("--voxel-size", nargs=3, type=float, metavar=("X","Y","Z"),
                        default=None,
                        help="Voxel size in mm. Reads from NIfTI header if omitted.")
    parser.add_argument("--eroded-rad", type=int, default=3, metavar="N",
                        help="Mask erosion radius in voxels.")
    parser.add_argument("--phase-sign", type=int, choices=[-1, 1], default=-1,
                        help="Phase sign convention: -1 (default) or +1.")
    parser.set_defaults(**config)
    args = parser.parse_args()

    if not args.phase:
        parser.error("--phase is required (or use --demo, or set 'phase' in config.yaml).")
    if not args.te:
        parser.error("--te is required (or set 'te' in config.yaml).")
    if any(te <= 0 for te in args.te):
        parser.error("All --te values must be positive (in seconds).")

    qsm_path = run_iqsm_plus(
        phase_nii_path=args.phase,
        te_values=args.te,
        mag_nii_path=args.mag,
        mask_nii_path=args.mask,
        voxel_size=args.voxel_size,
        b0_dir=args.b0_dir,
        b0=args.b0,
        eroded_rad=args.eroded_rad,
        phase_sign=args.phase_sign,
        output_dir=args.output,
    )

    print(f"\nOutputs:")
    print(f"  QSM (susceptibility): {qsm_path}")


if __name__ == "__main__":
    main()
