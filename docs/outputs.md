# Outputs

`labels.json` is the ground truth, in [COCO's detection format](https://cocodataset.org/#format-data): per frame, a box around each target with its range and bearing from the camera, its true heading, its `dims_m` (length, beam and height above the waterline, as built), the horizon, and what rendered it. A box's category is its asset's type, under its kind as supercategory, with a fixed id, so every `labels.json` agrees. FiftyOne reads the boxes and their fields as they are; the per-frame keys (`horizon_px`, `camera`, `band`, `hfov_deg`, `time_s`) stay in the JSON:

```python
import fiftyone as fo

fo.Dataset.from_dir("out/", fo.types.COCODetectionDataset, data_path=".")
```

An EO target whose contrast against its surroundings is under 5 %, the threshold of the WMO's meteorological optical range (WMO-No. 8, ch. 9), gets no box, as an occluded one gets none. `visibility_km` takes Koschmieder's 2 %, so a target loses its box nearer than the scenario's visibility. Its contrast is O'Kane et al.'s (1995) RSS Weber contrast in luminance, against a thin ring of background around its index-pass pixels.

An IR target whose RSS temperature contrast against the same ring is under 5 × its camera's `netd_k` (Rose 1948) gets no box; without a `netd_k` none is dropped. It is measured before the noise, and an exr carries none.

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
