# bet2 (FSL Brain Extraction Tool), vendored

`bin/bet2` plus the ~15 shared libraries under `lib/` it needs at runtime. Extracted from the
official `brainlife/fsl:6.0.7.22` Docker image (`docker.io/brainlife/fsl`) via `ldd`-resolved
dependency copying -- not a full FSL install, which is 5+ GB (atlases, GUI tools, pipelines for
other modalities this project never uses).

Vendored directly into the repo (rather than re-extracted from `brainlife/fsl` at build time)
so `docker/qsm.dockerfile` has **zero dependency on that image continuing to exist** -- Docker
Hub tags are not guaranteed permanent, `brainlife/fsl` is a community-maintained (not official)
image, and Docker Hub has historically had inactivity-based garbage collection for free-tier
images. Extraction commands, for reference/re-extraction if ever needed:

```bash
docker run --rm brainlife/fsl:6.0.7.22 sh -c '
mkdir -p /tmp/out/bin /tmp/out/lib
cp /usr/local/fsl/pkgs/fsl-bet2-*/bin/bet2 /tmp/out/bin/
for lib in libfsl-meshclass.so libfsl-newimage.so libfsl-utils.so liblapack.so.3 \
           libstdc++.so.6 libgcc_s.so.1 libfsl-miscmaths.so libfsl-NewNifti.so \
           libgfortran.so.5 libfsl-cprob.so libfsl-znz.so libquadmath.so.0 \
           libzstd.so.1 libbz2.so.1.0 libz.so.1; do
    cp -L "/usr/local/fsl/lib/$lib" "/tmp/out/lib/$lib"
done
cp /tmp/out/lib/liblapack.so.3 /tmp/out/lib/libblas.so.3
tar -czf - -C /tmp/out .
' > bet2_extracted.tar.gz
```

## License

Governed by the FSL Software License Agreement (Part B, "Downloading Agreement") --
royalty-free, sublicensable, explicitly permits redistribution and incorporation into other
software, scoped to research use (matches this project's own "Research use only. Not for
diagnostic use." qualification). Per the license's attribution requirement:

> All or portions of this licensed product (such portions are the "Software") have been
> obtained under license from The General Hospital Corporation ("MGH") and are subject to the
> following terms and conditions: see https://fsl.fmrib.ox.ac.uk/fsl/docs/#/license

Citation: Smith SM. Fast robust automated brain extraction. Human Brain Mapping,
17(3):143-155, 2002.
