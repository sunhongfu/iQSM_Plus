"""
Data utilities for iQSM — NIfTI / MAT loaders and DICOM → NIfTI conversion
for QSM-style inputs (wrapped phase + optional magnitude).

Mirrors DeepRelaxo's data_utils.py design so the two codebases stay
consistent. Used by both the web app and the command-line interface.
"""

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.io


# ---------------------------------------------------------------------------
# MAT loader (handles both v5 and v7.3)
# ---------------------------------------------------------------------------

def _load_mat_array(path):
    """Load the single 3D/4D numeric array from a MAT file.

    v5 files go through scipy.io.loadmat; v7.3 (HDF5) files fall through to
    h5py and are transposed (HDF5 stores MATLAB v7.3 data column-major,
    h5py reads row-major — transposing restores the logical MATLAB shape,
    matching scipy's auto-transpose on v5 files).
    """
    try:
        mat_data = scipy.io.loadmat(path)
        candidates = [
            (name, np.asarray(value))
            for name, value in mat_data.items()
            if not name.startswith("__") and np.issubdtype(np.asarray(value).dtype, np.number)
        ]
    except (NotImplementedError, ValueError):
        # NotImplementedError → MATLAB v7.3 (HDF5-backed)
        # ValueError → other unrecognised format. Fall through to h5py.
        candidates = []

    if not candidates:
        with h5py.File(path, "r") as handle:
            candidates = [
                (name, np.asarray(dataset).T)
                for name, dataset in handle.items()
                if isinstance(dataset, h5py.Dataset)
                and np.issubdtype(np.asarray(dataset).dtype, np.number)
            ]

    candidates = [
        (name, np.squeeze(array))
        for name, array in candidates
        if np.asarray(array).size > 1
    ]
    candidates = [
        (name, array)
        for name, array in candidates
        if array.ndim >= 3
    ]

    if not candidates:
        raise ValueError(
            f"Could not find a usable 3D/4D numeric array in MAT file {path}. "
            "Each .mat file must contain a single 3D (mask / single echo) or "
            "4D (multi-echo) numeric variable."
        )

    if len(candidates) > 1:
        # Pick the largest array (typical convention for phase/magnitude)
        candidates.sort(key=lambda kv: kv[1].size, reverse=True)

    return candidates[0][1]


def load_array_with_affine(path):
    """Load NIfTI or MAT file → (numpy array float32, affine or None)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return np.asarray(_load_mat_array(path), dtype=np.float32), None
    image = nib.load(str(path))
    return np.asarray(image.dataobj, dtype=np.float32), image.affine


def load_mask_array(mask_path, reference_shape):
    """Boolean mask aligned to reference_shape; all-True if mask_path is None."""
    if mask_path is None:
        return np.ones(reference_shape, dtype=bool)
    mask, _ = load_array_with_affine(mask_path)
    mask = mask > 0
    if mask.shape != tuple(reference_shape):
        raise ValueError(
            f"Mask shape {mask.shape} does not match reference shape {tuple(reference_shape)}"
        )
    return mask


# ---------------------------------------------------------------------------
# Shape helpers (header-only when possible, for fast UI rendering)
# ---------------------------------------------------------------------------

def file_shape(path_str):
    """Return the volume shape of a NIfTI/MAT file, or None on failure.

    NIfTI shape is read from the header (lazy). MAT requires loading the array.
    """
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        name = p.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            return tuple(nib.load(str(p)).shape)
        arr, _ = load_array_with_affine(p)
        return tuple(arr.shape)
    except Exception:
        return None


def shape_summary(paths):
    """Markdown summary that always shows per-file shapes for transparency.

    - 0 files → empty
    - 1 file → "Shape: A × B × C"
    - many files → header (matched / mismatched / unreadable) + per-file list
    """
    if not paths:
        return ""
    paths = list(paths)
    items = [(Path(p).name, file_shape(p)) for p in paths]
    n = len(paths)
    unreadable = [name for name, s in items if s is None]
    unique_shapes = {s for _, s in items if s is not None}

    def _fmt(s):
        return " × ".join(str(d) for d in s)

    if n == 1:
        name, shape = items[0]
        if shape:
            return f"&nbsp;&nbsp;**Shape:** {_fmt(shape)}"
        return f"&nbsp;&nbsp;`{name}` *(shape could not be read)*"

    if not unreadable and len(unique_shapes) == 1:
        header = (
            f"**{n} files** &nbsp;·&nbsp; "
            f"all matching shape **{_fmt(next(iter(unique_shapes)))}** ✓"
        )
    elif len(unique_shapes) > 1:
        header = (
            f"**{n} files** &nbsp;·&nbsp; "
            f"⚠️ **mismatched shapes** — all files must share the same volume dimensions"
        )
    else:
        header = (
            f"**{n} files** &nbsp;·&nbsp; "
            f"⚠️ {len(unreadable)} could not be read"
        )
    lines = [header, ""]
    for name, s in items:
        lines.append(f"- `{name}` — {_fmt(s) if s else '*(could not read)*'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DICOM loader for QSM input (phase + optional magnitude)
# ---------------------------------------------------------------------------

# GE Medical Systems private tag, named in the dictionary
# "Image Type (real, imaginary, phase, magnitude)". Predates DICOM's
# standard ComplexImageComponent and GE often uses it instead of marking
# ImageType. Enumeration: 0=magnitude, 1=phase, 2=real, 3=imaginary.
_GE_IMAGE_TYPE_TAG = (0x0043, 0x102F)
_GE_IMAGE_TYPE_MAP = {0: "M", 1: "P", 2: "R", 3: "I"}


def _image_type_markers(ds):
    """Collect modality markers from every channel a vendor might use:
    ImageType, ComplexImageComponent (DICOM standard), and the GE private tag."""
    markers = {str(t).upper() for t in list(getattr(ds, "ImageType", []))}

    # DICOM standard, Enhanced MR: 'MAGNITUDE' / 'PHASE' / 'REAL' / 'IMAGINARY' / 'MIXED'
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

    # GE private tag (often the only reliable source on GE scanners).
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
    """True if the DICOM is a phase image (excludes magnitude/real/imaginary)."""
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
    """True if the DICOM is a magnitude image (excludes phase/real/imaginary)."""
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
    """True if the DICOM is the real part of a complex acquisition."""
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
    """True if the DICOM is the imaginary part of a complex acquisition."""
    markers = _image_type_markers(ds)
    if "I" in markers or "IMAGINARY" in markers:
        return True
    if {"M", "MAGNITUDE", "P", "PHASE", "R", "REAL"} & markers:
        return False
    for m in markers:
        if m.startswith("I_") or m.endswith("_I"):
            return True
    return False


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
    """Apply RescaleSlope / RescaleIntercept to ds.pixel_array."""
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
    """Best-effort normalisation of phase values to the range [-pi, pi].

    Many scanners store phase as 16-bit integers in the range [0, 4095]
    (or similar) representing -π to π. RescaleSlope/Intercept usually doesn't
    capture this. If the array's range extends well beyond [-π, π] but
    looks like 0..max or -max..max, we rescale.
    """
    pmin, pmax = float(vol.min()), float(vol.max())
    pi = float(np.pi)
    if -pi - 0.1 <= pmin and pmax <= pi + 0.1:
        return vol  # already radians
    rng = pmax - pmin
    if rng <= 0:
        return vol
    # Common case: 0..4095 → -pi..pi
    if pmin >= -1.0 and pmax > 6.5:
        # 0..max → -pi..pi
        return (vol - pmin) / rng * (2 * pi) - pi
    # Symmetric case: -max..max
    if abs(pmin + pmax) < 0.1 * rng:
        return vol / max(abs(pmin), abs(pmax)) * pi
    # Fall back: linear remap
    return (vol - pmin) / rng * (2 * pi) - pi


def load_dicom_qsm_folder(file_paths, output_dir, chopper="auto"):
    """Parse GRE DICOMs into phase + magnitude NIfTI volumes.

    Recognises four ImageType buckets — phase (P/PHASE), magnitude
    (M/MAGNITUDE), real (R/REAL), and imaginary (I/IMAGINARY). Decision rule:

      • If real **and** imaginary DICOMs are both present and cover the same
        echo set, derive phase = ``angle(R + 1j·I)`` and magnitude =
        ``|R + 1j·I|``. This branch is preferred even when explicit P/M
        DICOMs are also present in the same folder.
      • Otherwise, use the explicit phase DICOMs (and magnitude DICOMs, if
        any) directly.

    Each modality is grouped by ``EchoTime`` and slices sorted by
    ``ImagePositionPatient`` before stacking.

    The ``chopper`` parameter controls the GE slice-direction sign-flip
    correction (a missing fftshift in GE's 3D recon — applies only when
    forming the complex signal from real + imaginary):

      • ``"auto"`` (default) — apply when ``Manufacturer`` looks like GE.
      • ``"on"``  — always apply, regardless of manufacturer.
      • ``"off"`` — never apply.

    Returns a dict:
        {
            "phase_path"  : Path         – 3D (single-echo) or 4D NIfTI of phase
            "mag_path"    : Path | None  – 3D / 4D NIfTI of magnitude (if found
                                            or derived from real/imag)
            "te_values_s" : list[float]  – echo times in **seconds**, sorted
            "voxel_size"  : list[float]  – [x, y, z] in mm
            "b0"          : float | None – field strength in Tesla
            "b0_dir"      : list[float] | None  – B0 direction in image space
            "phase_shape" : tuple        – shape of the saved phase NIfTI
        }

    Raises ``ValueError`` on missing phase / real-imag pair, mixed studies,
    mismatched echo slice counts, or empty input.
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

    # Reject mixed studies / acquisitions
    studies = {getattr(ds, "StudyInstanceUID", None) for _, ds in candidates}
    studies.discard(None)
    if len(studies) > 1:
        raise ValueError(
            f"DICOMs from {len(studies)} different studies/exams detected. "
            "Please provide DICOMs from a single GRE acquisition only."
        )

    # Sort each candidate into one of four ImageType buckets. Use a strict
    # priority order so each DICOM ends up in at most one bucket.
    phase_files = [(f, ds) for f, ds in candidates if _is_phase_dicom(ds)]
    real_files  = [(f, ds) for f, ds in candidates if _is_real_dicom(ds)]
    imag_files  = [(f, ds) for f, ds in candidates if _is_imag_dicom(ds)]
    mag_files   = [(f, ds) for f, ds in candidates if _is_magnitude_dicom(ds)]

    # Group by EchoTime
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

    # Pick a source for phase + magnitude.
    # If real+imag form a complete pair (same TE set, both populated),
    # prefer them and *derive* phase + magnitude — even when explicit
    # P/M DICOMs are also present (the user explicitly asked for this).
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
        # Validate slice counts (per-echo, real vs imag must match)
        for te in te_keys_ms:
            n_r = len(real_groups[te])
            n_i = len(imag_groups[te])
            if n_r != n_i:
                raise ValueError(
                    f"Real/imag echo TE = {te:g} ms has mismatched slice counts "
                    f"({n_r} real vs {n_i} imag) — likely DICOMs from different "
                    "series are mixed."
                )
        counts = {te: len(real_groups[te]) for te in te_keys_ms}
        if len(set(counts.values())) > 1:
            details = "\n".join(f"    TE = {te:g} ms : {n} slices"
                                for te, n in sorted(counts.items()))
            raise ValueError(
                "Real/imag echo groups have mismatched slice counts:\n" + details
            )

        # GE's reconstruction inserts an alternating ±1 along the slice
        # direction in image space (FFT centering quirk). Without correction,
        # real/imag look fine in magnitude but phase has π flips on every
        # other slice, looking like massive wrapping. Multiply each z-slice
        # by (-1)^z before forming the complex signal — equivalent to the
        # ``chopper`` array used in many GE-DICOM MATLAB pipelines.
        sample_ds = real_groups[te_keys_ms[0]][0][1]
        manufacturer = str(getattr(sample_ds, "Manufacturer", "")).upper()
        is_ge = "GE MEDICAL" in manufacturer or manufacturer == "GE"
        mode = (chopper or "auto").lower()
        if mode not in ("auto", "on", "off"):
            raise ValueError(f"chopper must be 'auto', 'on', or 'off' (got {chopper!r}).")
        apply_chopper = (mode == "on") or (mode == "auto" and is_ge)

        # _stack_echo already sorts each list by IPP, so paired real/imag
        # slices end up at matching positions inside each 3D volume.
        phase_vols, mag_vols = [], []
        affine = None
        for te in te_keys_ms:
            real_vol, aff = _stack_echo([ds for _, ds in real_groups[te]])
            imag_vol, _   = _stack_echo([ds for _, ds in imag_groups[te]])
            if affine is None:
                affine = aff
            if apply_chopper:
                # chopper shape (1, 1, Z) so it broadcasts over the volume
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
        # Use the real series as the canonical metadata source.
        meta_groups = real_groups
    else:
        if not phase_files:
            raise ValueError(
                "No phase DICOMs detected (need ImageType containing 'P' / 'PHASE', "
                "or both 'R' / 'REAL' and 'I' / 'IMAGINARY' so phase can be derived). "
                "If your scanner doesn't tag the modality explicitly, export only the "
                "needed series and try again."
            )
        if not phase_groups:
            raise ValueError(
                "No EchoTime tag found in phase DICOMs — cannot determine echoes."
            )

        counts = {te: len(items) for te, items in phase_groups.items()}
        if len(set(counts.values())) > 1:
            details = "\n".join(f"    TE = {te:g} ms : {n} slices"
                                for te, n in sorted(counts.items()))
            raise ValueError(
                "Phase echo groups have mismatched slice counts — likely DICOMs from "
                "multiple scans are mixed. Detected:\n" + details
            )

        te_keys_ms = sorted(phase_groups.keys())

        phase_vols = []
        affine = None
        for te in te_keys_ms:
            slices = [ds for _, ds in phase_groups[te]]
            vol, aff = _stack_echo(slices)
            vol = _normalise_phase_to_radians(vol)
            phase_vols.append(vol)
            if affine is None:
                affine = aff

        mag_vols = []
        if mag_groups:
            common = [te for te in te_keys_ms if te in mag_groups]
            if common == te_keys_ms:
                for te in te_keys_ms:
                    slices = [ds for _, ds in mag_groups[te]]
                    vol, _ = _stack_echo(slices)
                    mag_vols.append(vol)
            # If mag echoes don't match phase echoes, drop magnitude rather than
            # risk silently misaligning.
        meta_groups = phase_groups

    te_values_s = [te / 1000.0 for te in te_keys_ms]

    # Stack into 4D if multi-echo
    if len(phase_vols) == 1:
        phase_array = phase_vols[0]
    else:
        if len({v.shape for v in phase_vols}) > 1:
            raise ValueError("Phase echoes have inconsistent volume shapes")
        phase_array = np.stack(phase_vols, axis=-1).astype(np.float32)

    mag_array = None
    if mag_vols:
        if len(mag_vols) == 1:
            mag_array = mag_vols[0]
        else:
            mag_array = np.stack(mag_vols, axis=-1).astype(np.float32)

    # Save NIfTI files
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

    # Metadata: voxel size from PixelSpacing + slice spacing
    # (taken from the modality whose DICOMs we actually used)
    ds0 = meta_groups[te_keys_ms[0]][0][1]
    ps = list(getattr(ds0, "PixelSpacing", [1.0, 1.0]))
    # Slice spacing from two consecutive slices, or SliceThickness
    sorted_slices0 = sorted([ds for _, ds in meta_groups[te_keys_ms[0]]],
                            key=_slice_position)
    if len(sorted_slices0) > 1:
        ipp0 = np.array([float(v) for v in sorted_slices0[0].ImagePositionPatient])
        ipp1 = np.array([float(v) for v in sorted_slices0[1].ImagePositionPatient])
        slice_spc = float(np.linalg.norm(ipp1 - ipp0))
    else:
        slice_spc = float(getattr(ds0, "SliceThickness", 1.0))
    voxel_size = [float(ps[0]), float(ps[1]), slice_spc]

    # B0 field strength
    b0 = None
    fs = getattr(ds0, "MagneticFieldStrength", None)
    if fs is not None:
        try:
            b0 = float(fs)
        except Exception:
            b0 = None

    # B0 direction in image coordinates: scanner LPS z-axis projected onto
    # (row_dir, col_dir, slice_dir).
    b0_dir = None
    iop = getattr(ds0, "ImageOrientationPatient", None)
    if iop is not None:
        try:
            row = np.array([float(v) for v in iop[0:3]])
            col = np.array([float(v) for v in iop[3:6]])
            slice_n = np.cross(row, col)
            # Scanner B0 unit vector in LPS
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
        "phase_shape": tuple(phase_array.shape),
    }
