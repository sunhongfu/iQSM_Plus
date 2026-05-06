#!/usr/bin/env python3
"""
dicom_to_nifti — convert GRE DICOM(s) into NIfTI files + a params.json.

Works for both single-echo and multi-echo gradient-echo acquisitions, and for
any of the modality combinations a scanner may export:

  • phase (P/PHASE) + magnitude (M/MAGNITUDE), or
  • real (R/REAL) + imaginary (I/IMAGINARY)
    → phase and magnitude are derived from the complex signal:
        phase     = angle(R + 1j·I)
        magnitude = |R + 1j·I|
  • any combination of the four in a single folder
    → real + imaginary is preferred when both are present.

DICOMs are walked recursively from one or more folders, split by `ImageType`,
grouped by `EchoTime`, and slices sorted by `ImagePositionPatient`. The
output is a NIfTI volume per modality (3D for single-echo, 4D for multi-echo)
plus a `params.json` containing TE(s), voxel size, B0 strength, and B0
direction (in image coordinates) — everything a downstream QSM / R2* pipeline
typically needs.

Run `python dicom_to_nifti.py --help` for usage examples.
"""

import argparse
import json
import sys
from pathlib import Path

# data_utils.py lives next to this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_dicom_qsm_folder


_EXAMPLES = """\
examples:
  # A single folder of DICOMs (auto-split by ImageType). The folder may
  # contain phase + magnitude, or real + imaginary, or all four mixed
  # together. When both pairs are present, real + imaginary is preferred:
  python dicom_to_nifti.py --dicom_dir /path/to/dicoms

  # Phase and magnitude exported as two separate folders:
  python dicom_to_nifti.py --phase_dir /path/to/phase \\
                           --mag_dir   /path/to/magnitude

  # Real and imaginary exported as two separate folders:
  python dicom_to_nifti.py --real_dir /path/to/real \\
                           --imag_dir /path/to/imaginary

  # Specify the output folder (default: ./dicom_converted):
  python dicom_to_nifti.py --dicom_dir /path/to/dicoms --out_dir ./converted

  # Force the slice-direction sign-flip correction on / off when forming
  # phase + magnitude from real + imaginary (default 'auto' = on for GE,
  # off otherwise):
  python dicom_to_nifti.py --dicom_dir /path/to/dicoms --chopper on
  python dicom_to_nifti.py --dicom_dir /path/to/dicoms --chopper off

outputs (in --out_dir):
  phase NIfTI       — 3D for single-echo, 4D (echoes in last dim) for multi-echo
  magnitude NIfTI   — same shape as phase (always written when phase comes from
                      real + imaginary; otherwise only when magnitude DICOMs
                      are found)
  params.json       — TE(s) in ms, voxel size (mm), B0 (T), B0 direction
"""


def _walk_files(folder, parser):
    p = Path(folder)
    if not p.is_dir():
        parser.error(f"Not a directory: {folder}")
    files = [str(c) for c in p.rglob("*") if c.is_file()]
    if not files:
        parser.error(f"No files found in: {folder}")
    print(f"Found {len(files)} files in {folder}")
    return files


def main():
    parser = argparse.ArgumentParser(
        prog="dicom_to_nifti.py",
        description=__doc__.strip().split("\n\n", 1)[0],
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dicom_dir", metavar="DIR",
        help="A single folder of mixed DICOMs (any combination of phase, "
             "magnitude, real, imaginary). Modalities are auto-split by "
             "ImageType.",
    )
    parser.add_argument(
        "--phase_dir", metavar="DIR",
        help="Folder containing only phase DICOMs "
             "(must be supplied together with --mag_dir).",
    )
    parser.add_argument(
        "--mag_dir", metavar="DIR",
        help="Folder containing only magnitude DICOMs "
             "(must be supplied together with --phase_dir).",
    )
    parser.add_argument(
        "--real_dir", metavar="DIR",
        help="Folder containing only real-part DICOMs "
             "(must be supplied together with --imag_dir; phase + magnitude "
             "are derived from real + imaginary).",
    )
    parser.add_argument(
        "--imag_dir", metavar="DIR",
        help="Folder containing only imaginary-part DICOMs "
             "(must be supplied together with --real_dir).",
    )
    parser.add_argument(
        "--out_dir", default="./dicom_converted", metavar="DIR",
        help="Where to write the converted NIfTI files and params.json "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--chopper", choices=["auto", "on", "off"], default="auto",
        help="GE slice-direction sign-flip correction applied when forming "
             "phase + magnitude from real + imaginary. 'auto' (default): "
             "apply only when the scanner is GE. 'on': always apply. "
             "'off': never apply. Has no effect when phase + magnitude DICOMs "
             "are used directly.",
    )
    args = parser.parse_args()

    # ── Validate input mode ────────────────────────────────────────────
    has_combined = args.dicom_dir is not None
    has_pm = args.phase_dir is not None or args.mag_dir is not None
    has_ri = args.real_dir is not None or args.imag_dir is not None

    modes_used = sum(int(b) for b in (has_combined, has_pm, has_ri))
    if modes_used == 0:
        parser.error(
            "No input folder. Provide --dicom_dir, "
            "or --phase_dir + --mag_dir, or --real_dir + --imag_dir."
        )
    if modes_used > 1:
        parser.error(
            "Pick one input mode only: --dicom_dir (single combined folder) "
            "OR --phase_dir + --mag_dir OR --real_dir + --imag_dir. "
            "Don't combine modes."
        )
    if has_pm and (args.phase_dir is None or args.mag_dir is None):
        parser.error("--phase_dir and --mag_dir must be supplied together.")
    if has_ri and (args.real_dir is None or args.imag_dir is None):
        parser.error("--real_dir and --imag_dir must be supplied together.")

    # ── Walk folder(s) and collect file paths ──────────────────────────
    file_paths = []
    if has_combined:
        file_paths.extend(_walk_files(args.dicom_dir, parser))
    elif has_pm:
        file_paths.extend(_walk_files(args.phase_dir, parser))
        file_paths.extend(_walk_files(args.mag_dir,   parser))
    elif has_ri:
        file_paths.extend(_walk_files(args.real_dir, parser))
        file_paths.extend(_walk_files(args.imag_dir, parser))

    print(f"\nParsing {len(file_paths)} DICOM file(s)…")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = load_dicom_qsm_folder(file_paths, out_dir, chopper=args.chopper)
    except Exception as exc:
        parser.error(f"DICOM parsing failed: {exc}")

    phase_path = result["phase_path"]
    mag_path   = result["mag_path"]
    te_s       = result["te_values_s"]
    voxel      = result["voxel_size"]
    b0         = result["b0"]
    b0_dir     = result["b0_dir"]

    te_ms          = [float(t) * 1000 for t in te_s]
    te_ms_string   = ", ".join(f"{t:g}" for t in te_ms)
    voxel_string   = " ".join(f"{v:.4g}" for v in voxel) if voxel else ""
    b0_dir_string  = " ".join(f"{v:.4g}" for v in b0_dir) if b0_dir else ""

    params = {
        "te_ms":             te_ms,
        "te_ms_string":      te_ms_string,
        "voxel_size_mm":     list(voxel) if voxel else None,
        "voxel_size_string": voxel_string,
        "b0_T":              float(b0) if b0 is not None else None,
        "b0_dir":            list(b0_dir) if b0_dir else None,
        "b0_dir_string":     b0_dir_string,
        "n_echoes":          len(te_s),
        "phase_nifti":       Path(phase_path).name,
        "magnitude_nifti":   Path(mag_path).name if mag_path else None,
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))

    print()
    print("=" * 60)
    print("✅ Conversion complete")
    print("=" * 60)
    print(f"Output dir   : {out_dir}")
    print(f"Phase NIfTI  : {Path(phase_path).name}")
    if mag_path:
        print(f"Magnitude    : {Path(mag_path).name}")
    else:
        print("Magnitude    : ⚠️  not detected. If your magnitude DICOMs are in a "
              "separate folder, re-run with --phase_dir + --mag_dir or "
              "--real_dir + --imag_dir.")
    print(f"Parameters   : params.json")
    print()
    print("─── Acquisition values ───")
    print(f"  Echo Times (ms)  : {te_ms_string}")
    print(f"  Voxel size (mm)  : {voxel_string}")
    if b0 is not None:
        print(f"  B0 (Tesla)       : {b0}")
    if b0_dir:
        print(f"  B0 direction     : {b0_dir_string}")
    print("──────────────────────────")


if __name__ == "__main__":
    main()
