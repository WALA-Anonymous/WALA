#!/usr/bin/env python3
"""Precompute Depth Anything maps for LeRobot datasets used by WALA."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPTH_ANYTHING_ROOT = REPO_ROOT / "wala" / "model" / "third_party" / "Depth-Anything-V2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DEPTH_ANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2
from wala.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from wala.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from wala.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform


def safe_cache_component(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def depth_cache_key(dataset_path: Path, dataset_name: str, video_key: str, trajectory_id: int, frame_index: int) -> str:
    dataset_id = hashlib.md5(str(dataset_path.resolve()).encode()).hexdigest()[:12]
    return str(
        Path(safe_cache_component(dataset_name))
        / dataset_id
        / safe_cache_component(video_key)
        / f"episode_{safe_cache_component(trajectory_id)}"
        / f"frame_{int(frame_index):08d}.png"
    )


def infer_video_keys(dataset_path: Path) -> list[str]:
    modality_path = dataset_path / "meta" / "modality.json"
    with open(modality_path, "r") as f:
        modality = json.load(f)

    video_meta = modality.get("video", {})
    keys = []
    for key in video_meta.keys():
        keys.append(key if str(key).startswith("video.") else f"video.{key}")
    return keys


def build_depth_model(args) -> torch.nn.Module:
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    if args.encoder not in model_configs:
        raise ValueError(f"Unsupported encoder={args.encoder!r}")

    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    if torch.cuda.is_available() and args.device != "cpu":
        model = model.to(args.device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def resize_image(img: Image.Image, image_size: tuple[int, int] | None) -> Image.Image:
    if image_size is None:
        return img
    height, width = image_size
    if img.size == (width, height):
        return img
    return img.resize((width, height), Image.BICUBIC)


@torch.no_grad()
def infer_depth_batch(model, images: list[Image.Image], input_size: int, device: torch.device) -> torch.Tensor:
    tensors = []
    sizes = []
    for img in images:
        rgb = np.array(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tensor, (height, width) = model.image2tensor(bgr, input_size)
        tensors.append(tensor)
        sizes.append((height, width))

    image_batch = torch.cat(tensors, dim=0).to(device=device, dtype=torch.float32)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
    with ctx:
        depth = model.forward(image_batch)

    depth = depth[:, None].float()
    if all(size == sizes[0] for size in sizes):
        depth = F.interpolate(depth, sizes[0], mode="bilinear", align_corners=True)[:, 0]
    else:
        depth = torch.stack(
            [
                F.interpolate(depth[i : i + 1], size, mode="bilinear", align_corners=True)[0, 0]
                for i, size in enumerate(sizes)
            ],
            dim=0,
        )
    return depth.detach().cpu()


def depth_meta_path(path: Path) -> Path:
    return path.with_suffix(".json")


def has_depth_cache(path: Path) -> bool:
    if path.suffix.lower() == ".png":
        return path.exists() and depth_meta_path(path).exists()
    if path.exists():
        return True
    legacy_png = path.with_suffix(".png")
    return (
        path.with_suffix(".npy").exists()
        or path.with_suffix(".npz").exists()
        or (legacy_png.exists() and depth_meta_path(legacy_png).exists())
    )


def save_depth(path: Path, depth: torch.Tensor, overwrite: bool) -> bool:
    meta_path = depth_meta_path(path)
    if path.exists() and meta_path.exists() and not overwrite:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)

    depth_array = depth.float().numpy()
    finite_mask = np.isfinite(depth_array)
    if not np.any(finite_mask):
        depth_array = np.zeros_like(depth_array, dtype=np.float32)
        d_min, d_max = 0.0, 1.0
    else:
        valid = depth_array[finite_mask]
        d_min = float(valid.min())
        d_max = float(valid.max())
        depth_array = np.nan_to_num(depth_array, nan=d_min, posinf=d_max, neginf=d_min)

    if d_max <= d_min + 1e-12:
        depth_u16 = np.zeros_like(depth_array, dtype=np.uint16)
    else:
        depth_norm = (depth_array - d_min) / (d_max - d_min)
        depth_u16 = np.clip(np.rint(depth_norm * 65535.0), 0, 65535).astype(np.uint16)

    tmp_png_path = Path(f"{path}.tmp.{os.getpid()}.png")
    tmp_meta_path = Path(f"{meta_path}.tmp.{os.getpid()}.json")
    ok = cv2.imwrite(str(tmp_png_path), depth_u16, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError(f"Failed to write depth cache image: {tmp_png_path}")
    with open(tmp_meta_path, "w") as f:
        json.dump({"min": d_min, "max": d_max, "format": "uint16_png_minmax"}, f)
    os.replace(tmp_png_path, path)
    os.replace(tmp_meta_path, meta_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--video-key", action="append", default=None)
    parser.add_argument("--cache-dir", default="./cache/depth_anything", type=Path)
    parser.add_argument("--ckpt", default="./checkpoints/depth_anything/depth_anything_v2_vitl.pth")
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--image-size", nargs=2, type=int, default=[224, 224], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--frame-stride", default=1, type=int)
    parser.add_argument("--max-episodes", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.dataset_path = args.dataset_path.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.ckpt = str(Path(args.ckpt).resolve())
    image_size = tuple(args.image_size) if args.image_size else None

    video_keys = args.video_key or infer_video_keys(args.dataset_path)
    print(f"Dataset: {args.dataset_path}")
    print(f"Video keys: {video_keys}")
    print(f"Cache dir: {args.cache_dir}")

    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs={
            "video": ModalityConfig(delta_indices=[0], modality_keys=video_keys),
        },
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT_NO_ACTION,
        video_backend=args.video_backend,
        transforms=ComposedModalityTransform(transforms=[]),
        data_cfg={
            "validate_language": False,
            "video_backend": args.video_backend,
            "dataset_cache_dir": str(REPO_ROOT / "cache" / "datasets"),
        },
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = build_depth_model(args)
    model = model.to(device)

    trajectories = list(dataset.trajectory_ids)
    if args.max_episodes is not None:
        trajectories = trajectories[: args.max_episodes]

    total_saved = 0
    total_skipped = 0
    old_delta_indices = {key: dataset.delta_indices[key].copy() for key in video_keys}

    for trajectory_id in tqdm(trajectories, desc="Episodes"):
        trajectory_index = dataset.get_trajectory_index(trajectory_id)
        trajectory_length = int(dataset.trajectory_lengths[trajectory_index])
        frame_indices = np.arange(0, trajectory_length, max(args.frame_stride, 1), dtype=np.int64)
        dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
        dataset.curr_traj_id = trajectory_id

        for video_key in video_keys:
            pending_indices = []
            pending_paths = []
            for frame_index in frame_indices:
                key = depth_cache_key(
                    dataset.dataset_path,
                    dataset.dataset_name,
                    video_key,
                    trajectory_id,
                    int(frame_index),
                )
                path = args.cache_dir / key
                if has_depth_cache(path) and not args.overwrite:
                    total_skipped += 1
                    continue
                pending_indices.append(int(frame_index))
                pending_paths.append(path)

            for start in range(0, len(pending_indices), args.batch_size):
                batch_indices = np.asarray(pending_indices[start : start + args.batch_size], dtype=np.int64)
                batch_paths = pending_paths[start : start + args.batch_size]
                if len(batch_indices) == 0:
                    continue

                dataset.delta_indices[video_key] = batch_indices - int(batch_indices[0])
                frames = dataset.get_video(trajectory_id, video_key, int(batch_indices[0]))
                images = [
                    resize_image(Image.fromarray(frame), image_size)
                    for frame in frames
                ]
                depths = infer_depth_batch(model, images, args.input_size, device)
                for path, depth in zip(batch_paths, depths):
                    if save_depth(path, depth, overwrite=args.overwrite):
                        total_saved += 1

            dataset.delta_indices[video_key] = old_delta_indices[video_key]

    print(f"Saved depth maps: {total_saved}")
    print(f"Skipped existing depth maps: {total_skipped}")


if __name__ == "__main__":
    main()
