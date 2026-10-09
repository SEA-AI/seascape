# Outputs

`labels.json` is the ground truth, in [COCO's detection format](https://cocodataset.org/#format-data): per frame, a box around each target with its range and bearing from the camera, its true heading, its `dims_m` (length, beam and height above the waterline, as built), the horizon, and what rendered it. A box's category is its asset's type, under its kind as supercategory, with a fixed id, so every `labels.json` agrees. FiftyOne reads the boxes and their fields as they are; the per-frame keys (`horizon_px`, `camera`, `band`, `hfov_deg`, `time_s`) stay in the JSON:

```python
import fiftyone as fo

fo.Dataset.from_dir("out/", fo.types.COCODetectionDataset, data_path=".")
```

Each box's `contrast` is how visible its target is in the frame as written: Weber's, unsigned and averaged over the target's index-pass pixels, `mean(|L - L_background|) / L_background`, against a thin ring of background around them, `null` with none. 0 is invisible; a dark hull under a bright superstructure does not cancel, and a uniform target reads its plain Weber contrast. EO takes luminance, decoded from sRGB in a jpg or png; IR takes the values written, grey in a jpg, centikelvin in a png, radiance in an exr. Koschmieder's limit of visibility is a luminance contrast of 2%, so a loader can mark an EO box with `contrast < 0.02` as `iscrowd`, or ignore it.

`render --variants` writes one `labels.json` per seed folder and merges them into one beside the folders: ids renumbered, each `file_name` from there, and each seed's scenario, every drawn value in it, under `info.scenarios` by its folder.

An LWIR jpg is what a thermal camera shows: 8-bit grey, its contrast span damped so a clip does not flicker. An LWIR png is what it measures: 16-bit centikelvin, the unit radiometric thermal cameras write, so `cv2.imread(path, cv2.IMREAD_UNCHANGED) / 100` is kelvin. A viewer shows that as flat grey; `montage`, `panorama` and `video` tone it through the same AGC as the jpg.

`seascape montage` lays a render out for review, one row per band, each frame captioned with its camera. It reads the images already written, so it needs no Blender and a layout can be redone without re-rendering:

```bash
uv run seascape render scenarios/twin-pod.toml -o out/
uv run seascape montage scenarios/twin-pod.toml -o out/   # out/montage.png
```

`seascape panorama` stitches each rig's frames, per band, from `calibration.json`:

```bash
uv run seascape panorama out/ --projection rectilinear --frame rig --ruler
```
