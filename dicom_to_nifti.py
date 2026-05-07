#!/usr/bin/env python3
"""
dicom_to_nifti — convert GRE DICOM(s) into NIfTI files + a params.json.

This script is intentionally **independent of the downstream pipeline**: the
exact same file ships with iQSM, iQSM+, and DeepRelaxo, and produces a
generic output bundle (phase + magnitude NIfTIs + a `params.json`) that any
of the three pipelines — or any other QSM / R2* tool — can consume.

Works for both single-echo and multi-echo gradient-echo acquisitions, and for
whatever subset of the four modalities a scanner exports:

  • phase only          → phase NIfTI written (fine for iQSM / iQSM+; the
                           multi-echo combiner falls back to TE²-only
                           weighting when no magnitude is available).
  • magnitude only      → magnitude NIfTI written (fine for DeepRelaxo).
  • phase + magnitude   → both written.
  • real + imaginary    → phase and magnitude derived from the complex
                           signal:  phase = angle(R + 1j·I),
                                    magnitude = |R + 1j·I|.
  • any mixture in one folder
                        → modalities are auto-split by ImageType /
                           ComplexImageComponent / GE private tag.
                           Real + imaginary, when complete, is preferred
                           over explicit phase / magnitude DICOMs.

DICOMs are walked recursively from one or more folders, split by modality
(via DICOM `ImageType`, `ComplexImageComponent`, and the GE private tag
`(0043, 102f)`), grouped by `EchoTime`, and slices sorted by
`ImagePositionPatient`. The output is a NIfTI volume per modality (3D for
single-echo, 4D for multi-echo) plus a `params.json` containing TE(s),
voxel size, B0 strength, and B0 direction (in image coordinates).

`params.json` also contains *copy-paste strings* (`te_ms_string`,
`voxel_size_string`, `b0_dir_string`) formatted exactly the way the three
web apps expect their input fields, so users can paste them straight in.

Run `python dicom_to_nifti.py --help` for usage examples.
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


_EXAMPLES = """\
examples:
  # A single folder of DICOMs (auto-split by ImageType). Any subset of the
  # four modalities is accepted — phase only, magnitude only, phase +
  # magnitude, real + imaginary, all four mixed, etc. When real + imaginary
  # form a complete pair, they're preferred over explicit phase / magnitude
  # DICOMs (phase = angle(R+jI), magnitude = |R+jI|):
  python dicom_to_nifti.py --dicom_dir /path/to/dicoms

  # Phase and magnitude exported as two separate folders:
  python dicom_to_nifti.py --phase_dir /path/to/phase \\
                           --mag_dir   /path/to/magnitude

  # Phase only (fine for iQSM / iQSM+ when magnitude isn't available —
  # multi-echo combination falls back to TE²-only weighting):
  python dicom_to_nifti.py --phase_dir /path/to/phase

  # Magnitude only (fine for DeepRelaxo, which uses magnitude only):
  python dicom_to_nifti.py --mag_dir /path/to/magnitude

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
  params.json       — TE(s) in ms, voxel size (mm), B0 (T), B0 direction,
                      plus copy-paste strings for the web app input fields
"""


# ---------------------------------------------------------------------------
# Modality classification (ImageType + ComplexImageComponent + GE private tag)
# ---------------------------------------------------------------------------

# GE Medical Systems private tag, named in the dictionary
# "Image Type (real, imaginary, phase, magnitude)". Predates DICOM's standard
# ComplexImageComponent and GE often uses it instead of marking ImageType.
# Enumeration: 0=magnitude, 1=phase, 2=real, 3=imaginary.
_GE_IMAGE_TYPE_TAG = (0x0043, 0x102F)
_GE_IMAGE_TYPE_MAP = {0: "M", 1: "P", 2: "R", 3: "I"}


def _image_type_markers(ds):
    """Collect modality markers from every channel a vendor might use:
    ImageType, ComplexImageComponent (DICOM standard), and the GE private tag."""
    markers = {str(t).upper() for t in list(getattr(ds, "ImageType", []))}

    cic = getattr(ds, "ComplexImageComponent", None)
    if cic:
        s = str(cic).upper()
        if s.startswith("MAG"):
            markers.add("M")
        elif s.startswith("PH"):
            markers.add("P")
        elif s.startswith("REAL"):
            markers.add("R")
        elif s.startswith("IMAG"):
            markers.add("I")

    try:
        elem = ds.get(_GE_IMAGE_TYPE_TAG)
        if elem is not None:
            v = elem.value
            if isinstance(v, (list, tuple)):
                v = v[0] if v else None
            if v is not None:
                ge_marker = _GE_IMAGE_TYPE_MAP.get(int(v))
                if ge_marker:
                    markers.add(ge_marker)
    except Exception:
        pass

    return markers


def _is_phase_dicom(ds):
    markers = _image_type_markers(ds)
    if "P" in markers or "PHASE" in markers:
        return True
    if {"M", "MAGNITUDE", "I", "IMAGINARY", "R", "REAL"} & markers:
        return False
    for m in markers:
        if m.startswith("P_") or m.endswith("_P"):
            return True
        if m.startswith(("M_", "R_", "I_")) or m.endswith(("_M", "_R", "_I")):
            return False
    return False


def _is_magnitude_dicom(ds):
    markers = _image_type_markers(ds)
    if "M" in markers or "MAGNITUDE" in markers:
        return True
    if {"P", "PHASE", "I", "IMAGINARY", "R", "REAL"} & markers:
        return False
    for m in markers:
        if m.startswith("M_") or m.endswith("_M"):
            return True
        if m.startswith(("P_", "R_", "I_")) or m.endswith(("_P", "_R", "_I")):
            return False
    # No marker — historical default: treat as magnitude.
    return True


def _is_real_dicom(ds):
    markers = _image_type_markers(ds)
    if "R" in markers or "REAL" in markers:
        return True
    if {"M", "MAGNITUDE", "P", "PHASE", "I", "IMAGINARY"} & markers:
        return False
    for m in markers:
        if m.startswith("R_") or m.endswith("_R"):
            return True
    return False


def _is_imag_dicom(ds):
    markers = _image_type_markers(ds)
    if "I" in markers or "IMAGINARY" in markers:
        return True
    if {"M", "MAGNITUDE", "P", "PHASE", "R", "REAL"} & markers:
        return False
    for m in markers:
        if m.startswith("I_") or m.endswith("_I"):
            return True
    return False


# ---------------------------------------------------------------------------
# Slice-level helpers
# ---------------------------------------------------------------------------

def _slice_position(ds):
    """Signed distance along slice-normal direction. Falls back to SliceLocation."""
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is not None and ipp is not None:
        try:
            row = np.array([float(v) for v in iop[0:3]])
            col = np.array([float(v) for v in iop[3:6]])
            normal = np.cross(row, col)
            return float(np.dot([float(v) for v in ipp], normal))
        except Exception:
            pass
    sl = getattr(ds, "SliceLocation", None)
    if sl is not None:
        try:
            return float(sl)
        except Exception:
            pass
    return float(getattr(ds, "InstanceNumber", 0))


def _build_affine(sorted_slices):
    """Build a 4×4 RAS NIfTI affine from sorted DICOM slices (LPS→RAS)."""
    ds0 = sorted_slices[0]
    iop = [float(v) for v in ds0.ImageOrientationPatient]
    ipp0 = np.array([float(v) for v in ds0.ImagePositionPatient])
    ps = [float(v) for v in ds0.PixelSpacing]
    row_dir = np.array(iop[0:3])
    col_dir = np.array(iop[3:6])
    col_spacing = ps[1]
    row_spacing = ps[0]
    if len(sorted_slices) > 1:
        ipp1 = np.array([float(v) for v in sorted_slices[1].ImagePositionPatient])
        slice_vec = ipp1 - ipp0
        slice_spc = float(np.linalg.norm(slice_vec))
        slice_dir = slice_vec / slice_spc if slice_spc > 0 else np.cross(row_dir, col_dir)
    else:
        slice_dir = np.cross(row_dir, col_dir)
        slice_spc = float(getattr(ds0, "SliceThickness", 1.0))
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = row_dir * col_spacing
    affine_lps[:3, 1] = col_dir * row_spacing
    affine_lps[:3, 2] = slice_dir * slice_spc
    affine_lps[:3, 3] = ipp0
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    return flip @ affine_lps


def _rescale_pixel(ds):
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept
    return arr


def _stack_echo(slices):
    """Sort slices by position and stack into a 3D volume + affine."""
    slices = sorted(slices, key=_slice_position)
    volumes = [_rescale_pixel(ds) for ds in slices]
    if len({v.shape for v in volumes}) > 1:
        raise ValueError("Inconsistent slice shapes within a single echo group")
    # pixel_array is (rows, cols); transpose → (cols, rows) so the array
    # matches IOP's row_dir/col_dir convention used by _build_affine.
    vol = np.stack([v.T for v in volumes], axis=2).astype(np.float32)
    return vol, _build_affine(slices)


def _normalise_phase_to_radians(vol):
    """Best-effort normalisation of phase values to the range [-pi, pi]."""
    pmin, pmax = float(vol.min()), float(vol.max())
    pi = float(np.pi)
    if -pi - 0.1 <= pmin and pmax <= pi + 0.1:
        return vol
    rng = pmax - pmin
    if rng <= 0:
        return vol
    if pmin >= -1.0 and pmax > 6.5:
        return (vol - pmin) / rng * (2 * pi) - pi
    if abs(pmin + pmax) < 0.1 * rng:
        return vol / max(abs(pmin), abs(pmax)) * pi
    return (vol - pmin) / rng * (2 * pi) - pi


# ---------------------------------------------------------------------------
# Main DICOM-folder parser
# ---------------------------------------------------------------------------

def _walk_files(folder, parser):
    p = Path(folder)
    if not p.is_dir():
        parser.error(f"Not a directory: {folder}")
    files = [str(c) for c in p.rglob("*") if c.is_file()]
    if not files:
        parser.error(f"No files found in: {folder}")
    print(f"Found {len(files)} files in {folder}")
    return files


def _convert(file_paths, output_dir, chopper="auto"):
    """Parse GRE DICOMs into phase + magnitude NIfTI volumes.

    Returns a dict with phase_path, mag_path, te_values_s, voxel_size, b0,
    b0_dir, and phase_shape. Raises ValueError on missing phase / real-imag
    pair, mixed studies, mismatched echo slice counts, or empty input.
    """
    try:
        import pydicom
        from pydicom.errors import InvalidDicomError
    except ImportError as exc:
        raise ImportError(
            "pydicom is required for DICOM input. Install with: pip install pydicom"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for f in file_paths:
        f = Path(f)
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), force=True)
        except (InvalidDicomError, OSError, EOFError, AttributeError, ValueError):
            continue
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        candidates.append((f, ds))

    if not candidates:
        raise ValueError(
            "No readable DICOM files with pixel data were found. Make sure "
            "the folder contains DICOM images (.dcm, .ima, .dicom, or no "
            "extension)."
        )

    studies = {getattr(ds, "StudyInstanceUID", None) for _, ds in candidates}
    studies.discard(None)
    if len(studies) > 1:
        raise ValueError(
            f"DICOMs from {len(studies)} different studies/exams detected. "
            "Please provide DICOMs from a single GRE acquisition only."
        )

    phase_files = [(f, ds) for f, ds in candidates if _is_phase_dicom(ds)]
    real_files  = [(f, ds) for f, ds in candidates if _is_real_dicom(ds)]
    imag_files  = [(f, ds) for f, ds in candidates if _is_imag_dicom(ds)]
    mag_files   = [(f, ds) for f, ds in candidates if _is_magnitude_dicom(ds)]

    def _group_by_te(items):
        groups = {}
        for f, ds in items:
            te = getattr(ds, "EchoTime", None)
            if te is None:
                continue
            key = round(float(te), 4)
            groups.setdefault(key, []).append((f, ds))
        return groups

    phase_groups = _group_by_te(phase_files)
    real_groups  = _group_by_te(real_files)  if real_files  else {}
    imag_groups  = _group_by_te(imag_files)  if imag_files  else {}
    mag_groups   = _group_by_te(mag_files)   if mag_files   else {}

    use_complex = (
        bool(real_groups) and bool(imag_groups)
        and set(real_groups) == set(imag_groups)
    )

    if use_complex:
        te_keys_ms = sorted(real_groups.keys())
        if not te_keys_ms:
            raise ValueError(
                "Real/imaginary DICOMs detected but they have no EchoTime tag — "
                "cannot determine echoes."
            )
        for te in te_keys_ms:
            n_r = len(real_groups[te])
            n_i = len(imag_groups[te])
            if n_r != n_i:
                raise ValueError(
                    f"Real/imag echo TE = {te:g} ms has mismatched slice counts "
                    f"({n_r} real vs {n_i} imag)."
                )
        counts = {te: len(real_groups[te]) for te in te_keys_ms}
        if len(set(counts.values())) > 1:
            details = "\n".join(f"    TE = {te:g} ms : {n} slices"
                                for te, n in sorted(counts.items()))
            raise ValueError(
                "Real/imag echo groups have mismatched slice counts:\n" + details
            )

        sample_ds = real_groups[te_keys_ms[0]][0][1]
        manufacturer = str(getattr(sample_ds, "Manufacturer", "")).upper()
        is_ge = "GE MEDICAL" in manufacturer or manufacturer == "GE"
        mode = (chopper or "auto").lower()
        if mode not in ("auto", "on", "off"):
            raise ValueError(f"chopper must be 'auto', 'on', or 'off' (got {chopper!r}).")
        apply_chopper = (mode == "on") or (mode == "auto" and is_ge)

        phase_vols, mag_vols = [], []
        affine = None
        for te in te_keys_ms:
            real_vol, aff = _stack_echo([ds for _, ds in real_groups[te]])
            imag_vol, _   = _stack_echo([ds for _, ds in imag_groups[te]])
            if affine is None:
                affine = aff
            if apply_chopper:
                z = real_vol.shape[2]
                chopper_vec = ((-1.0) ** np.arange(z, dtype=np.float32))[None, None, :]
                real_vol = real_vol * chopper_vec
                imag_vol = imag_vol * chopper_vec
            cplx = real_vol.astype(np.float32) + 1j * imag_vol.astype(np.float32)
            phase_vols.append(np.angle(cplx).astype(np.float32))
            mag_vols.append(np.abs(cplx).astype(np.float32))
        if apply_chopper:
            why = ("manufacturer is GE" if is_ge and mode == "auto"
                   else "user requested --chopper on")
            print(f"  Applied slice-direction chopper ((-1)^z) to real + imag "
                  f"before forming complex signal ({why}).")
        elif mode == "off" and is_ge:
            print("  ⚠️  GE scanner detected but --chopper off was requested — "
                  "phase may show alternating π flips along the slice direction.")
        meta_groups = real_groups
    else:
        if not phase_files and not mag_files:
            raise ValueError(
                "No phase or magnitude DICOMs detected. Need ImageType containing "
                "'P' / 'PHASE' or 'M' / 'MAGNITUDE', or both 'R' / 'REAL' and "
                "'I' / 'IMAGINARY' so phase can be derived. If your scanner "
                "doesn't tag the modality explicitly, export only the needed "
                "series and try again."
            )

        phase_vols = []
        affine = None
        if phase_groups:
            counts = {te: len(items) for te, items in phase_groups.items()}
            if len(set(counts.values())) > 1:
                details = "\n".join(f"    TE = {te:g} ms : {n} slices"
                                    for te, n in sorted(counts.items()))
                raise ValueError(
                    "Phase echo groups have mismatched slice counts:\n" + details
                )
            te_keys_ms = sorted(phase_groups.keys())
            for te in te_keys_ms:
                slices = [ds for _, ds in phase_groups[te]]
                vol, aff = _stack_echo(slices)
                vol = _normalise_phase_to_radians(vol)
                phase_vols.append(vol)
                if affine is None:
                    affine = aff
            meta_groups = phase_groups
        else:
            # Magnitude-only run (e.g. for DeepRelaxo). Use mag DICOMs as the
            # canonical metadata source; no phase output.
            counts = {te: len(items) for te, items in mag_groups.items()}
            if len(set(counts.values())) > 1:
                details = "\n".join(f"    TE = {te:g} ms : {n} slices"
                                    for te, n in sorted(counts.items()))
                raise ValueError(
                    "Magnitude echo groups have mismatched slice counts:\n" + details
                )
            te_keys_ms = sorted(mag_groups.keys())
            meta_groups = mag_groups

        mag_vols = []
        if mag_groups:
            common = [te for te in te_keys_ms if te in mag_groups]
            if common == te_keys_ms:
                for te in te_keys_ms:
                    slices = [ds for _, ds in mag_groups[te]]
                    vol, aff = _stack_echo(slices)
                    mag_vols.append(vol)
                    if affine is None:
                        affine = aff

    te_values_s = [te / 1000.0 for te in te_keys_ms]

    if phase_vols:
        if len(phase_vols) == 1:
            phase_array = phase_vols[0]
        else:
            if len({v.shape for v in phase_vols}) > 1:
                raise ValueError("Phase echoes have inconsistent volume shapes")
            phase_array = np.stack(phase_vols, axis=-1).astype(np.float32)
    else:
        phase_array = None

    mag_array = None
    if mag_vols:
        if len(mag_vols) == 1:
            mag_array = mag_vols[0]
        else:
            mag_array = np.stack(mag_vols, axis=-1).astype(np.float32)

    phase_path = None
    if phase_array is not None:
        if len(te_values_s) == 1:
            phase_path = output_dir / "dcm_converted_phase.nii.gz"
        else:
            phase_path = output_dir / "dcm_converted_phase_4d.nii.gz"
        nib.save(nib.Nifti1Image(phase_array, affine), str(phase_path))

    mag_path = None
    if mag_array is not None:
        if len(te_values_s) == 1:
            mag_path = output_dir / "dcm_converted_magnitude.nii.gz"
        else:
            mag_path = output_dir / "dcm_converted_magnitude_4d.nii.gz"
        nib.save(nib.Nifti1Image(mag_array, affine), str(mag_path))

    ds0 = meta_groups[te_keys_ms[0]][0][1]
    ps = list(getattr(ds0, "PixelSpacing", [1.0, 1.0]))
    sorted_slices0 = sorted([ds for _, ds in meta_groups[te_keys_ms[0]]],
                            key=_slice_position)
    if len(sorted_slices0) > 1:
        ipp0 = np.array([float(v) for v in sorted_slices0[0].ImagePositionPatient])
        ipp1 = np.array([float(v) for v in sorted_slices0[1].ImagePositionPatient])
        slice_spc = float(np.linalg.norm(ipp1 - ipp0))
    else:
        slice_spc = float(getattr(ds0, "SliceThickness", 1.0))
    voxel_size = [float(ps[0]), float(ps[1]), slice_spc]

    b0 = None
    fs = getattr(ds0, "MagneticFieldStrength", None)
    if fs is not None:
        try:
            b0 = float(fs)
        except Exception:
            b0 = None

    b0_dir = None
    iop = getattr(ds0, "ImageOrientationPatient", None)
    if iop is not None:
        try:
            row = np.array([float(v) for v in iop[0:3]])
            col = np.array([float(v) for v in iop[3:6]])
            slice_n = np.cross(row, col)
            b0_lps = np.array([0., 0., 1.])
            b0_dir_arr = np.array([
                float(np.dot(row, b0_lps)),
                float(np.dot(col, b0_lps)),
                float(np.dot(slice_n, b0_lps)),
            ])
            n = np.linalg.norm(b0_dir_arr)
            if n > 0:
                b0_dir = (b0_dir_arr / n).tolist()
        except Exception:
            b0_dir = None

    return {
        "phase_path": phase_path,
        "mag_path": mag_path,
        "te_values_s": te_values_s,
        "voxel_size": voxel_size,
        "b0": b0,
        "b0_dir": b0_dir,
        "phase_shape": tuple(phase_array.shape) if phase_array is not None else None,
        "mag_shape":   tuple(mag_array.shape)   if mag_array   is not None else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="dicom_to_nifti.py",
        description=__doc__.strip().split("\n\n", 1)[0],
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dicom_dir", metavar="DIR",
        help="A single folder of DICOMs. Any subset of the four modalities "
             "is accepted — phase only (iQSM / iQSM+ without magnitude), "
             "magnitude only (DeepRelaxo), phase + magnitude, real + "
             "imaginary, all four mixed, etc. Modalities are auto-split by "
             "ImageType / ComplexImageComponent / GE private tag. When "
             "real + imaginary form a complete pair, they're preferred "
             "over explicit phase / magnitude DICOMs.",
    )
    parser.add_argument(
        "--phase_dir", metavar="DIR",
        help="Folder containing only phase DICOMs. Can be used alone "
             "(phase-only output, fine for iQSM / iQSM+ when magnitude "
             "isn't available — multi-echo combination then falls back to "
             "TE²-only weighting) or paired with --mag_dir.",
    )
    parser.add_argument(
        "--mag_dir", metavar="DIR",
        help="Folder containing only magnitude DICOMs. Can be used alone "
             "(magnitude-only output, fine for DeepRelaxo) or paired with "
             "--phase_dir.",
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

    has_combined = args.dicom_dir is not None
    has_pm = args.phase_dir is not None or args.mag_dir is not None
    has_ri = args.real_dir is not None or args.imag_dir is not None

    modes_used = sum(int(b) for b in (has_combined, has_pm, has_ri))
    if modes_used == 0:
        parser.error(
            "No input folder. Provide --dicom_dir, "
            "or --phase_dir and/or --mag_dir, or --real_dir + --imag_dir."
        )
    if modes_used > 1:
        parser.error(
            "Pick one input mode only: --dicom_dir (single combined folder) "
            "OR --phase_dir / --mag_dir (either or both) OR --real_dir + "
            "--imag_dir. Don't combine modes."
        )
    # phase / magnitude are independently usable — only one of the two is
    # strictly required (phase-only is fine for iQSM / iQSM+; magnitude-only
    # is fine for DeepRelaxo). Real + imaginary, in contrast, must always be
    # paired since the complex signal can't be formed from one alone.
    if has_ri and (args.real_dir is None or args.imag_dir is None):
        parser.error("--real_dir and --imag_dir must be supplied together.")

    file_paths = []
    if has_combined:
        file_paths.extend(_walk_files(args.dicom_dir, parser))
    elif has_pm:
        if args.phase_dir is not None:
            file_paths.extend(_walk_files(args.phase_dir, parser))
        if args.mag_dir is not None:
            file_paths.extend(_walk_files(args.mag_dir, parser))
    elif has_ri:
        file_paths.extend(_walk_files(args.real_dir, parser))
        file_paths.extend(_walk_files(args.imag_dir, parser))

    print(f"\nParsing {len(file_paths)} DICOM file(s)…")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = _convert(file_paths, out_dir, chopper=args.chopper)
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
        # Machine-readable
        "te_ms":             te_ms,
        "voxel_size_mm":     list(voxel) if voxel else None,
        "b0_T":              float(b0) if b0 is not None else None,
        "b0_dir":            list(b0_dir) if b0_dir else None,
        "n_echoes":          len(te_s),
        "phase_nifti":       Path(phase_path).name if phase_path else None,
        "magnitude_nifti":   Path(mag_path).name if mag_path else None,
        # Copy-paste strings — formatted exactly the way the three web apps'
        # input fields expect them, so users can paste straight in.
        "te_ms_string":      te_ms_string,
        "voxel_size_string": voxel_string,
        "b0_dir_string":     b0_dir_string,
    }
    (out_dir / "params.json").write_text(json.dumps(params, indent=2))

    print()
    print("=" * 60)
    print("✅ Conversion complete")
    print("=" * 60)
    print(f"Output dir   : {out_dir}")
    if phase_path:
        print(f"Phase NIfTI  : {Path(phase_path).name}")
    else:
        print("Phase NIfTI  : (none — magnitude-only run)")
    if mag_path:
        print(f"Magnitude    : {Path(mag_path).name}")
    else:
        print("Magnitude    : ⚠️  not detected. If your magnitude DICOMs are in a "
              "separate folder, re-run with --phase_dir + --mag_dir or "
              "--real_dir + --imag_dir.")
    print(f"Parameters   : params.json")
    print()
    print("─── Acquisition values (paste these into the web app) ───")
    print(f"  Echo Times (ms)  : {te_ms_string}")
    print(f"  Voxel size (mm)  : {voxel_string}")
    if b0 is not None:
        print(f"  B0 (Tesla)       : {b0}")
    if b0_dir:
        print(f"  B0 direction     : {b0_dir_string}")
    print("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
