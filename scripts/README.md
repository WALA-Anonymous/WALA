# Data Utilities

## Depth Anything V2 Cache

`precompute_depth_anything_cache.py` precomputes dense depth maps for one
LeRobot dataset. Run it from the repository root after installing WALA and the
Depth Anything V2 dependencies described in the main README.

```bash
python scripts/precompute_depth_anything_cache.py \
    --dataset-path /path/to/lerobot_dataset \
    --cache-dir ./cache/depth_anything \
    --ckpt ./checkpoints/depth_anything/depth_anything_v2_vitl.pth \
    --batch-size 32
```

By default, video keys are read from `meta/modality.json`. A key can also be
selected explicitly; repeat `--video-key` to process multiple views:

```bash
python scripts/precompute_depth_anything_cache.py \
    --dataset-path /path/to/lerobot_dataset \
    --video-key video.head \
    --video-key video.left_wrist \
    --cache-dir ./cache/depth_anything
```

Each depth map is min-max quantized to a 16-bit PNG and stored with lossless PNG
compression. A JSON sidecar records the per-frame minimum and maximum depth
values used for reconstruction. Existing PNG and JSON pairs are skipped, so an
interrupted run can be resumed with the same command. Use `--overwrite` to
recompute existing entries.

Useful options:

- `--frame-stride N` processes every `N`th frame.
- `--max-episodes N` limits processing to the first `N` episodes.
- `--image-size HEIGHT WIDTH` sets the saved depth-map resolution.
- `--input-size N` sets the Depth Anything V2 inference size.
- `--batch-size N` controls the inference batch size on one device.
- `--device cuda` or `--device cpu` selects the inference device.
- `--video-backend pyav` selects the video decoder used by the LeRobot loader.

The cache key includes the resolved dataset path, dataset name, video key,
episode, and frame index. Set the training configuration's `depth_cache_dir` to
the same directory passed through `--cache-dir`.
