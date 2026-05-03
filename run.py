"""
iQSM+ — Command-line interface

First-time setup (download checkpoints + demo data):
    python run.py --download-checkpoints
    python run.py --download-demo

Run from raw DICOMs (recommended — phase + magnitude auto-separated, TEs and
B0 direction read from headers):
    python run.py --dicom_dir /path/to/dicoms

Run from pre-converted NIfTI / MAT files:
    python run.py --echo_files ph1.nii ph2.nii ph3.nii --te_ms 4 8 12
    python run.py --echo_4d phase_4d.nii.gz --te_ms 4 8 12 --mag mag_4d.nii.gz
    python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz

Non-axial acquisition — provide B0 direction in image coordinates:
    python run.py --echo_4d ph.nii.gz --te_ms 4 8 12 --b0-dir 0.1 0.0 0.995

YAML config (any CLI arg can be set instead in the config):
    python run.py --config config.yaml
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from huggingface_hub import hf_hub_download

from data_utils import (
    load_array_with_affine,
    load_dicom_qsm_folder,
)

REPO_ROOT = Path(__file__).resolve().parent

HF_REPO_ID = "sunhongfu/iQSM_Plus"
CHECKPOINT_FILES = [
    "iQSM_plus.pth",
    "LoTLayer_chi.pth",
]
DEMO_FILES = [
    "demo/ph_multi_echo.nii.gz",
    "demo/mag_multi_echo.nii.gz",
    "demo/mask_multi_echo.nii.gz",
    "demo/params.json",
]


# ---------------------------------------------------------------------------
# HuggingFace download helpers
# ---------------------------------------------------------------------------

def _download_files(file_list, local_subdir):
    target_dir = REPO_ROOT / local_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    for remote_path in file_list:
        local_name = Path(remote_path).name
        destination = target_dir / local_name
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_path,
            repo_type="model",
        )
        shutil.copyfile(downloaded_path, destination)
        print(f"Downloaded: {destination}")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_path(base_dir, value):
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else Path(base_dir) / p


def _stage_input(path, work_dir, suffix=""):
    """Copy NIfTI as-is or convert MAT → NIfTI. Returns staged path."""
    src = Path(path)
    if src.suffix.lower() == ".mat":
        arr, _ = load_array_with_affine(src)
        dst = Path(work_dir) / (src.stem + suffix + ".nii.gz")
        nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(dst))
        print(f"Converted MAT → NIfTI: {src.name} → {dst.name}")
        return dst
    return src


def _combine_3d_to_4d(paths, out_path):
    """Stack a list of 3D NIfTIs into a single 4D NIfTI."""
    imgs = [nib.load(str(p)) for p in paths]
    affine = imgs[0].affine
    data = np.stack([img.get_fdata(dtype=np.float32) for img in imgs], axis=-1)
    nib.save(nib.Nifti1Image(data, affine), str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description="iQSM+: Orientation-Adaptive QSM reconstruction from raw MRI phase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", metavar="FILE",
                        help="YAML config file. Any flag below can be set there.")

    parser.add_argument("--data_dir", metavar="DIR",
                        help="Resolve relative input paths against this folder. "
                             "Defaults to the current working directory.")

    parser.add_argument(
        "--dicom_dir", metavar="DIR",
        help="Folder of multi-echo GRE phase (and, optionally, magnitude) DICOMs. "
             "Files are walked recursively, split into phase vs. magnitude via "
             "ImageType, grouped by EchoTime, sorted by ImagePositionPatient, "
             "and saved as one NIfTI per modality in <output>/dicom_converted_nii/. "
             "TE values, voxel size, B0 strength **and B0 direction** are auto-detected.",
    )
    parser.add_argument("--echo_files", nargs="+", metavar="FILE",
                        help="Multiple 3D phase NIfTI / MAT files (one per echo). "
                             "They will be stacked into a 4D volume internally.")
    parser.add_argument("--echo_4d", metavar="FILE",
                        help="Single 4D phase NIfTI / MAT (echoes in last dim).")
    parser.add_argument("--phase", metavar="FILE",
                        help="Single 3D phase NIfTI / MAT (legacy single-echo input).")

    parser.add_argument("--te", nargs="+", type=float, metavar="SEC",
                        help="Echo time(s) in **seconds**, e.g. --te 0.020 0.040.")
    parser.add_argument("--te_ms", nargs="+", type=float, metavar="MS",
                        help="Echo time(s) in **milliseconds**, e.g. --te_ms 4 8 12.")

    parser.add_argument("--mag", metavar="FILE",
                        help="Magnitude NIfTI / MAT (3D or 4D). Used internally by "
                             "iQSM+ for magnitude × TE² weighted multi-echo fitting.")
    parser.add_argument("--mask", metavar="FILE",
                        help="Brain mask NIfTI / MAT (optional; ones if omitted).")
    parser.add_argument("--bet_mask", metavar="FILE",
                        help="Alias for --mask.")

    parser.add_argument("--output", metavar="DIR", default="./iqsm_plus_output",
                        help="Output directory.")
    parser.add_argument("--b0", type=float, default=3.0,
                        help="B0 field strength in Tesla (default: 3.0).")
    parser.add_argument("--b0-dir", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=None, dest="b0_dir",
                        help="B0 direction unit vector. Default: read from DICOM "
                             "headers (when --dicom_dir is used) or [0, 0, 1] (axial).")
    parser.add_argument("--voxel-size", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=None,
                        help="Voxel size in mm. Reads from NIfTI header if omitted.")
    parser.add_argument("--eroded-rad", type=int, default=3, metavar="N",
                        help="Mask erosion radius in voxels (default: 3).")
    parser.add_argument("--reverse-phase-sign", type=int, choices=[0, 1], default=0,
                        help="0 = no (default), 1 = yes. Set to 1 if iron-rich "
                             "deep grey matter appears dark in the QSM output.")

    parser.add_argument("--download-checkpoints", action="store_true",
                        help="Download model weights from HuggingFace and exit.")
    parser.add_argument("--download-demo", action="store_true",
                        help="Download demo NIfTIs from HuggingFace and exit.")
    return parser


def _apply_config_defaults(args, parser):
    if not args.config:
        return args, None
    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    config_dir = config_path.parent
    user_provided = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            user_provided.add(tok.lstrip("-").replace("-", "_"))
    for key, value in cfg.items():
        attr = key.replace("-", "_")
        if attr in user_provided:
            continue
        if hasattr(args, attr):
            setattr(args, attr, value)
    return args, config_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.download_checkpoints:
        _download_files(CHECKPOINT_FILES, "checkpoints")
        return
    if args.download_demo:
        _download_files(DEMO_FILES, "demo")
        return

    args, config_dir = _apply_config_defaults(args, parser)

    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    elif config_dir is not None:
        data_dir = config_dir.resolve()
    else:
        data_dir = Path.cwd()

    given = sum(x is not None for x in [args.dicom_dir, args.echo_files,
                                        args.echo_4d, args.phase])
    if given == 0:
        parser.error(
            "No phase input. Use one of: --dicom_dir, --echo_files, --echo_4d, --phase "
            "(or set 'phase' / 'echoes' / 'echo_4d' / 'dicom_dir' in --config)."
        )
    if given > 1:
        parser.error(
            "Provide exactly one of: --dicom_dir, --echo_files, --echo_4d, --phase."
        )

    if config_dir is not None and not Path(args.output).is_absolute():
        output_dir = (config_dir / args.output).resolve()
    else:
        output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix="iqsmplus_run_"))

    # ── Resolve phase input ─────────────────────────────────────────────
    phase_input_path = None      # path passed to run_iqsm_plus (3D or 4D)
    te_values_s = None
    mag_path_resolved = None     # may be set by DICOM parsing
    b0_dir_value = list(args.b0_dir) if args.b0_dir is not None else None

    if args.dicom_dir:
        dicom_path = _resolve_path(data_dir, args.dicom_dir)
        if not dicom_path.is_dir():
            parser.error(f"--dicom_dir is not a directory: {dicom_path}")
        file_list = [str(p) for p in dicom_path.rglob("*") if p.is_file()]
        if not file_list:
            parser.error(f"--dicom_dir contains no files: {dicom_path}")
        nii_out = output_dir / "dicom_converted_nii"
        print(f"Parsing DICOMs from {dicom_path}")
        print(f"Writing converted NIfTI files to {nii_out}")
        result = load_dicom_qsm_folder(file_list, nii_out)
        phase_input_path = str(result["phase_path"])
        mag_path_resolved = str(result["mag_path"]) if result["mag_path"] else None
        te_values_s = list(result["te_values_s"])
        if args.voxel_size is None and result["voxel_size"]:
            args.voxel_size = result["voxel_size"]
            print(f"Voxel size from DICOM: {args.voxel_size}")
        if args.b0 == 3.0 and result["b0"] is not None:
            args.b0 = float(result["b0"])
            print(f"B0 from DICOM: {args.b0} T")
        if b0_dir_value is None and result["b0_dir"] is not None:
            b0_dir_value = list(result["b0_dir"])
            print(f"B0 direction from DICOM: {b0_dir_value}")
        if args.te_ms is not None:
            if len(args.te_ms) != len(te_values_s):
                parser.error(
                    f"--te_ms count ({len(args.te_ms)}) doesn't match parsed echoes "
                    f"({len(te_values_s)})."
                )
            te_values_s = [t / 1000.0 for t in args.te_ms]
            print(f"Using user-supplied TEs (ms): {args.te_ms}")
        elif args.te is not None:
            if len(args.te) != len(te_values_s):
                parser.error(
                    f"--te count ({len(args.te)}) doesn't match parsed echoes "
                    f"({len(te_values_s)})."
                )
            te_values_s = list(args.te)
            print(f"Using user-supplied TEs (s): {args.te}")

    elif args.echo_4d:
        if args.te_ms is None and args.te is None:
            parser.error("when using --echo_4d, provide --te_ms or --te")
        src = _resolve_path(data_dir, args.echo_4d)
        staged = _stage_input(src, work_dir)
        phase_input_path = str(staged)
        te_values_s = ([t / 1000.0 for t in args.te_ms] if args.te_ms
                       else list(args.te))

    elif args.echo_files:
        if args.te_ms is None and args.te is None:
            parser.error("when using --echo_files, provide --te_ms or --te")
        te_values_s = ([t / 1000.0 for t in args.te_ms] if args.te_ms
                       else list(args.te))
        if len(args.echo_files) != len(te_values_s):
            parser.error("--echo_files and TE counts must match.")
        staged = []
        for f in args.echo_files:
            src = _resolve_path(data_dir, f)
            staged.append(_stage_input(src, work_dir))
        # iQSM+ wants a 4D phase volume — combine the 3D echoes
        combined = work_dir / "phase_4d.nii.gz"
        _combine_3d_to_4d(staged, combined)
        phase_input_path = str(combined)
        print(f"Combined {len(staged)} 3D echoes into {combined.name}")

    else:  # --phase (legacy)
        if args.te_ms is None and args.te is None:
            parser.error("when using --phase, provide --te_ms or --te")
        src = _resolve_path(data_dir, args.phase)
        staged = _stage_input(src, work_dir)
        phase_input_path = str(staged)
        te_values_s = ([t / 1000.0 for t in args.te_ms] if args.te_ms
                       else list(args.te))

    # ── Stage magnitude / mask ──────────────────────────────────────────
    mag_arg = args.mag or mag_path_resolved
    if mag_arg:
        mag_src = _resolve_path(data_dir, mag_arg)
        mag_path = _stage_input(mag_src, work_dir, suffix="_mag")
    else:
        mag_path = None

    mask_value = args.mask or args.bet_mask
    if mask_value:
        mask_src = _resolve_path(data_dir, mask_value)
        mask_path = _stage_input(mask_src, work_dir, suffix="_mask")
    else:
        mask_path = None

    # Default voxel size for MAT-only input
    if args.voxel_size is None:
        mat_inputs = []
        if args.echo_4d:
            mat_inputs.append(args.echo_4d)
        if args.echo_files:
            mat_inputs.extend(args.echo_files)
        if args.phase:
            mat_inputs.append(args.phase)
        if mat_inputs and all(Path(p).suffix.lower() == ".mat" for p in mat_inputs):
            args.voxel_size = [1.0, 1.0, 1.0]
            print("Note: no --voxel-size given for MAT input — defaulting to 1×1×1 mm")

    if any(t <= 0 for t in te_values_s):
        parser.error("All echo times must be positive.")

    phase_sign = 1 if args.reverse_phase_sign else -1

    # ── Print summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("iQSM+ RUN CONFIGURATION")
    print("=" * 60)
    print(f"Phase file      : {Path(phase_input_path).name}")
    print(f"Echoes          : {len(te_values_s)}")
    print(f"TE (s)          : {', '.join(f'{t:g}' for t in te_values_s)}")
    print(f"Magnitude       : {Path(mag_path).name if mag_path else '(none)'}")
    print(f"Brain mask      : {Path(mask_path).name if mask_path else '(none)'}")
    if args.voxel_size:
        print(f"Voxel size (mm) : {' '.join(f'{v:.4g}' for v in args.voxel_size)}")
    else:
        print(f"Voxel size (mm) : (from NIfTI header)")
    print(f"B0 (T)          : {args.b0}")
    print(f"B0 direction    : {b0_dir_value if b0_dir_value else '[0 0 1] (default)'}")
    print(f"Mask erosion    : {args.eroded_rad} voxels")
    print(f"Reverse phase   : {'yes' if phase_sign == 1 else 'no'}")
    print(f"Output dir      : {output_dir}")
    print("=" * 60)
    print()

    # ── Run reconstruction ──────────────────────────────────────────────
    from inference import run_iqsm_plus, CheckpointNotFoundError
    try:
        qsm_path = run_iqsm_plus(
            phase_nii_path=phase_input_path,
            te_values=te_values_s,
            mag_nii_path=str(mag_path) if mag_path else None,
            mask_nii_path=str(mask_path) if mask_path else None,
            voxel_size=args.voxel_size,
            b0_dir=b0_dir_value,
            b0=args.b0,
            eroded_rad=args.eroded_rad,
            phase_sign=phase_sign,
            output_dir=str(output_dir),
        )
    except CheckpointNotFoundError as exc:
        print(f"\nError: {exc}\n", flush=True)
        raise SystemExit(1)

    print()
    print(f"Output:")
    print(f"  QSM (susceptibility): {qsm_path}")


if __name__ == "__main__":
    main()
