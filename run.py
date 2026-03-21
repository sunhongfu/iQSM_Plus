"""
iQSM+ – Command-line interface
Usage:
    python run.py --phase ph.nii.gz --te 0.020 --mask mask.nii.gz --output ./results/
    python run.py --phase ph.nii.gz --te 0.0032 0.0065 0.0098 --mag mag.nii.gz --output ./results/
    python run.py --help
"""

import argparse

from inference import run_iqsm_plus


def main():
    parser = argparse.ArgumentParser(
        description="iQSM+: QSM reconstruction from single- or multi-echo MRI phase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase",   required=True, metavar="FILE",
                        help="Wrapped phase NIfTI (.nii / .nii.gz), 3D (single-echo) or 4D (multi-echo).")
    parser.add_argument("--te",      required=True, nargs="+", type=float, metavar="SEC",
                        help="Echo time(s) in seconds. One value for single-echo, "
                             "multiple for multi-echo (e.g. --te 0.0032 0.0065 0.0098).")
    parser.add_argument("--mag",     metavar="FILE", default=None,
                        help="Magnitude NIfTI (optional; used for echo fitting).")
    parser.add_argument("--mask",    metavar="FILE", default=None,
                        help="Brain mask NIfTI (optional; ones if omitted).")
    parser.add_argument("--output",  metavar="DIR",  default="./iqsm_plus_output",
                        help="Output directory.")
    parser.add_argument("--b0",      type=float, default=3.0,
                        help="B0 field strength in Tesla.")
    parser.add_argument("--b0-dir",  nargs=3, type=float, metavar=("X","Y","Z"),
                        default=None,
                        help="B0 direction unit vector (default: read from NIfTI or [0,0,1]).")
    parser.add_argument("--voxel-size", nargs=3, type=float, metavar=("X","Y","Z"),
                        default=None,
                        help="Voxel size in mm. Reads from NIfTI header if omitted.")
    parser.add_argument("--eroded-rad", type=int, default=3, metavar="N",
                        help="Mask erosion radius in voxels.")
    parser.add_argument("--phase-sign", type=int, choices=[-1, 1], default=-1,
                        help="Phase sign convention: -1 (default) or +1.")

    args = parser.parse_args()

    if any(te <= 0 for te in args.te):
        parser.error("All --te values must be positive (in seconds, e.g. 0.020).")

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
