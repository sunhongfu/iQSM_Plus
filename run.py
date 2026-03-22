"""
iQSM+ – Command-line interface

Setup (first time — downloads demo NIfTIs into demo/):
    python run.py --download-demo

Run:
    python run.py --config config.yaml
    python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz
    python run.py --phase ph.nii.gz --te 0.0032 0.0065 0.0098 --mag mag.nii.gz
    python run.py --config config.yaml --output ./other/   # CLI overrides config
    python run.py --help

Checkpoints are stored in checkpoints/ (downloaded from Hugging Face on first use).
Demo data is stored in demo/ (downloaded on first use or via --download-demo).
"""

import argparse
import os

import yaml

_HF_REPO  = "sunhongfu/iQSM_Plus"
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DEMO_DIR = os.path.join(_HERE, "demo")

_DEMO_FILENAMES = [
    "ph_multi_echo.nii.gz",
    "mag_multi_echo.nii.gz",
    "mask_multi_echo.nii.gz",
    "params.json",
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _ensure_local(filename: str, hf_path: str, local_dir: str) -> str:
    """Return local path to file, downloading from HF Hub into local_dir if absent."""
    local = os.path.join(local_dir, filename)
    if not os.path.exists(local):
        import shutil
        from huggingface_hub import hf_hub_download
        print(f"  {filename} …", end=" ", flush=True)
        cached = hf_hub_download(repo_id=_HF_REPO, filename=hf_path)
        os.makedirs(local_dir, exist_ok=True)
        shutil.copy(cached, local)
        print(f"ok  →  {local}")
    return local


def cmd_download_demo():
    import json
    print(f"Fetching demo data from huggingface.co/{_HF_REPO} → {_DEMO_DIR}/")
    os.makedirs(_DEMO_DIR, exist_ok=True)
    for name in _DEMO_FILENAMES:
        _ensure_local(name, f"demo/{name}", _DEMO_DIR)
    phase = os.path.join(_DEMO_DIR, "ph_multi_echo.nii.gz")
    mag   = os.path.join(_DEMO_DIR, "mag_multi_echo.nii.gz")
    mask  = os.path.join(_DEMO_DIR, "mask_multi_echo.nii.gz")
    with open(os.path.join(_DEMO_DIR, "params.json")) as f:
        p = json.load(f)
    te     = p["TE_seconds"]
    vox    = p["voxel_size_mm"]
    b0     = p["B0_Tesla"]
    sign   = p["phase_sign_convention"]
    eroded = p.get("eroded_rad", 0)
    mat    = "×".join(str(x) for x in p.get("matrix_size", []))
    te_str  = str(te) if isinstance(te, (int, float)) else " ".join(f"{v:.4g}" for v in te)
    vox_str = " ".join(str(v) for v in vox)
    print(f"""
Demo dataset: {p.get("description", "")}
  Matrix:  {mat}
  Voxel:   {vox_str} mm
  TE:      {te_str} s
  B0:      {b0} T

To run reconstruction on this data:

    python run.py \\
        --phase  {phase} \\
        --mag    {mag} \\
        --mask   {mask} \\
        --te     {te_str} \\
        --b0     {b0} \\
        --voxel-size {vox_str} \\
        --eroded-rad {eroded} \\
        --phase-sign {sign} \\
        --output ./iqsm_plus_demo_output/

Or copy config.yaml, fill in the paths above, and run:

    python run.py --config config.yaml
""")


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
    # Pre-parse action flags before building the full parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--download-demo", action="store_true")
    pre.add_argument("--config", metavar="FILE")
    known, _ = pre.parse_known_args()

    if known.download_demo:
        cmd_download_demo()
        return

    config = _load_config(known.config) if known.config else {}

    parser = argparse.ArgumentParser(
        description="iQSM+: QSM reconstruction from single- or multi-echo MRI phase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--download-demo", action="store_true",
                        help="Pre-warm HF cache with demo NIfTIs and show how to run them.")
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
    parser.add_argument("--b0-dir",     nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=None,
                        help="B0 direction unit vector (default: read from header or [0,0,1]).")
    parser.add_argument("--voxel-size", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=None,
                        help="Voxel size in mm. Reads from NIfTI header if omitted.")
    parser.add_argument("--eroded-rad", type=int, default=3, metavar="N",
                        help="Mask erosion radius in voxels.")
    parser.add_argument("--phase-sign", type=int, choices=[-1, 1], default=-1,
                        help="Phase sign convention: -1 (default) or +1.")
    parser.set_defaults(**config)
    args = parser.parse_args()

    if not args.phase:
        parser.error("--phase is required (or use --download-demo, or set 'phase' in config.yaml).")
    if not args.te:
        parser.error("--te is required (or set 'te' in config.yaml).")
    if any(te <= 0 for te in args.te):
        parser.error("All --te values must be positive (in seconds).")

    from inference import run_iqsm_plus
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
