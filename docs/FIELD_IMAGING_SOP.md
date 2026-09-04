# Field specimen imaging SOP

Use one protocol at every site. A model can tolerate some photographic variation, but quantitative
body-size and colour comparisons cannot be repaired after inconsistent capture.

## Before fieldwork

- Assign stable `site_id` codes and globally unique `specimen_id` values.
- Fix one camera body, lens, focal length, copy-stand height, background, and light arrangement.
- Fix dorsal/ventral orientation and wing-position rules.
- Validate the LEPY scale-bar template and physical ruler with at least ten manually measured pilot
  specimens.
- Photograph a neutral/colour reference at the start and end of each imaging session. LEPY reports
  image-channel traits; cross-session colour inference still depends on a controlled capture chain.
- Record the camera, lens, light, operator, protocol version, and LEPY configuration version in the
  project log.

## Per specimen

1. Assign the specimen ID before imaging.
2. Place one spread individual on the standardized background without labels touching the body.
3. Include the calibrated scale bar in its fixed image region.
4. Use manual exposure, fixed white balance, fixed focus, and lossless TIFF when possible.
5. Capture the required RGB view. If UV is used, do not move the specimen or camera before the
   registered UV exposure.
6. Name files `<specimen_id>_rgb.tif` and `<specimen_id>_uv.tif`.
7. Record `site_id`, country, WGS84 latitude/longitude, ISO collection date, and observed
   temperature in `field_metadata.csv`.
8. Keep original image files immutable; export edits as new derivatives.

## Daily quality control

- Confirm sharpness at wing scales and that no wing edge is cropped.
- Confirm the scale bar is fully visible and successfully detected.
- Check for folded, overlapping, damaged, or strongly tilted wings.
- Review LEPY masks, points of interest, and obvious left/right asymmetry warnings.
- Re-image failures before specimens are released or stored.
- Back up images and metadata together; never infer specimen IDs from folder order.

## Validation before analysis

- Compare automated body length, forewing length, area, and wing span against blinded manual
  measurements on a held-out subset from every site and operator.
- Quantify repeatability by re-imaging a subset on different days.
- Inspect colour-reference drift and include imaging batch/device effects in downstream models when
  needed.
- Freeze the accepted LEPY checkout and configuration for the final analysis, and retain the engine
  fingerprint and config hash exported by EuroLepi Studio.
