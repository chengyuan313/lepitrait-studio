# Standardized specimen imaging SOP

## Purpose

Produce comparable images for morphology, pigmentation and specimen-photo species identification. The SOP applies to museum specimens and newly collected specimens photographed in the same standardized configuration.

## Station

1. Mount the camera perpendicular to the specimen plane on a rigid copy stand.
2. Fix camera body, macro lens, working distance and focus method for an imaging batch.
3. Use two diffuse lights at symmetric angles. Record lamp model, nominal colour temperature and diffuser configuration.
4. Use a matte, neutral, texture-free background that contrasts with the specimen without clipping pale scales.
5. Include a millimetre ruler and a colour target in every frame. Keep both in the specimen plane.
6. Assign stable IDs to camera, lens, light configuration, colour profile and imaging batch.

## Capture

1. Place the specimen centrally with its longitudinal axis aligned to the frame.
2. Ensure wings are visible and labels do not overlap the specimen, scale or colour target.
3. Record the view as `dorsal`, `ventral` or `unknown`. When safe and feasible, photograph both sides of new specimens.
4. Use manual exposure, manual white balance and fixed ISO. Avoid automatic scene-dependent adjustments.
5. Check focus across both forewings. Re-capture images with motion blur, glare, deep shadow or clipped pale scales.
6. Save the camera RAW file. Generate a calibrated, non-destructive 16-bit TIFF derivative and a smaller review JPEG.

## Colour processing

1. Build or select the colour profile from the target under the same lighting configuration.
2. Apply lens correction, white balance and the colour profile without local contrast or saturation enhancement.
3. Record the profile identifier and processing software version.
4. Treat images without a target/profile as uncalibrated. Their colour measurements are relative and must not be compared across batches.
5. Iridescent and structural colours require fixed illumination and viewing geometry; visible-light RGB is not a direct reflectance measurement.

## File naming

Use `<institution>_<catalogNumber>_<view>_<batch>.<ext>`, for example:

```text
MZH_GV12345_dorsal_2027A.tif
MZH_GV12345_ventral_2027A.tif
```

Never encode the taxon name in the classifier input filename. Store taxonomy in the manifest.

## Capture acceptance gate

- ruler and colour target present;
- specimen and target are coplanar;
- no overlap with label or pinning tools;
- sufficient resolution and focus;
- no highlight/shadow clipping on the wings;
- view, batch and equipment IDs present;
- raw file preserved;
- taxonomic label stored separately from classifier pixels.

