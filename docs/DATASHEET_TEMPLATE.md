# Dataset datasheet (per dataset)

The supervisor explicitly requires the processing history of every dataset so we can argue
that the detector learns generator traces, not preprocessing artifacts. Fill one block per
dataset. `scripts/make_datasheets.py` auto-fills the measurable fields (count, resolution,
format) from `master_metadata.csv`; the provenance fields below are manual and must be
confirmed with the data source / the supervisor.

---

## <dataset_name>

- Role: real | fake
- Generator / source: <e.g. StyleGAN3-FFHQ, FLUX.1-schnell, Face Research Lab London>
- Kind: photo | diffusion | gan
- Count: <auto>
- Native resolution(s): <auto: min/median/max>
- On-disk format: <auto: png/jpg>  (JPEG quality if known: <manual>)
- Color space: <RGB/sRGB; ICC profile if any>

### Provenance / processing history (manual - the part the supervisor cares about)
- Sensor-to-image pipeline (real) OR generation pipeline (fake): <describe>
- Known resize/crop applied by the source: <yes/no, method>
- Known compression applied by the source: <none/JPEG q?/other>
- Any watermark removal, alignment, or face-crop step: <describe>
- License / access route: <link or "via the supervisor on <date>">

### Risk notes
- Distribution narrowness (real): <e.g. single camera, heavy alignment>
- Confounds vs other datasets (resolution, compression mismatch): <describe>
- Mitigation in our pipeline: scaling/cropping variants + PNG-only derived images.
