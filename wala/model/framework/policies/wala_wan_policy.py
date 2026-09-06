# Copyright 2026 Anonymous Authors. All rights reserved.
# Licensed under the MIT License.

"""WALA policy variant using Wan2.2 as the visual backbone."""

import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint

from deployment.model_server.tools.image_tools import to_pil_preserve
from wala.model.framework.base_framework import baseframework
from wala.model.framework.policies.wala_policy import WALAPolicy
from wala.model.framework.share_tools import merge_framework_config
from wala.model.modules.action_model import get_action_model
from wala.model.modules.latent_action_model import (
    ContinuousTransitionBottleneckTokenizer,
    LatentActionToTransitionTokens,
)
from wala.model.modules.vision_encoder import get_dino_model
from wala.model.tools import FRAMEWORK_REGISTRY
from wala.training.trainer_utils import initialize_overwatch
from wala.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class WALAWanPolicyDefaultConfig:
    """Default parameters for the Wan2.2-backed WALA policy."""

    name: str = "WALA_WAN"

    wan: dict = field(default_factory=lambda: {
        "base_wm": "./checkpoints/wan/Wan2.2-TI2V-5B-Diffusers",
        "num_frames": 5,
        "video_size": [224, 224],
        "concat_multi_camera": None,
        "freeze_backbone": False,
        "freeze_text_encoder": True,
        "freeze_vae": True,
        "extract_layers": [-1],
        "text_max_length": 256,
        "torch_dtype": "bfloat16",
        "resampler_layers": 2,
        "resampler_num_heads": 24,
        "state_injection": "text",
        "state_num_tokens": 8,
        "state_encoder_hidden_dim": 512,
        "state_cross_attention_gate_init": 0.1,
    })

    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "MLP",
        "action_dim": 14,
        "state_dim": 14,
        "action_hidden_dim": 3072,
        "condition_dim": 1024,
        "future_action_window_size": 49,
        "past_action_window_size": 0,
    })

    dino: dict = field(default_factory=lambda: {
        "dino_backbone": "./checkpoints/dinov3/dinov3-vitb16-pretrain-lvd1689m",
    })

    transition_bottleneck: dict = field(default_factory=lambda: {
        "training_mode": "vla",
        "dino_dim": 768,
        "num_transition_tokens": 32,
        "hidden_dim": 1024,
        "encoder_layers": 8,
        "decoder_layers": 4,
        "latent_resampler_layers": 1,
        "num_heads": 8,
        "max_future_steps": 64,
        "max_patches": 1024,
        "max_action_tokens": 256,

        "use_depth": False,
        "depth_loss_weight": 1.0,
        "depth_fusion_scale": 1.0,
        "depth_anything_encoder": "vitl",
        "depth_anything_ckpt": "./checkpoints/depth_anything/depth_anything_v2_vitl.pth",
        "depth_anything_input_size": 224,
        "depth_anything_batch_size": 16,
        "depth_source": "auto",
        "depth_cache_dir": "./cache/depth_anything",
        "depth_cache_write_missing": False,
        "depth_patch_size": 16,

        "smooth_l1_weight": 1.0,
        "cosine_weight": 0.1,
        "composition_weight": 0,
        "reencode_weight": 0,
        "content_invariance_weight": 0,
        "content_aug_scale_std": 0.05,
        "content_aug_bias_std": 0.02,
        "min_composition_steps": 2,

        "use_transition_alignment": True,
        "transition_alignment_loss_weight": 0.005,
        "transition_alignment_l1_weight": 0.1,

        "use_transition_decode": True,
        "transition_decode_loss_weight": 0.5,
        "transition_decode_cosine_weight": 0.1,
        "transition_decode_use_all_future": True,
        "depth_decoder_loss_weight": 1.0,
        "depth_gradient_loss_weight": 0.2,
        "depth_decoder_use_all_future": True,

        "freeze_encoder_in_vla": True,
        "train_decoder_in_vla": True,
    })

    obs_image_size: Optional[list] = None
    vla_state_dropout_prob: float = 0.0


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = str(dtype_name).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    if dtype_name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported Wan dtype: {dtype_name!r}.")


def _resize_pil(image: Image.Image, size_hw: Tuple[int, int]) -> Image.Image:
    h, w = int(size_hw[0]), int(size_hw[1])
    return image.convert("RGB").resize((w, h), Image.BICUBIC)


def _concat_pil_horizontal(images: List[Image.Image]) -> Image.Image:
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def _concat_pil_vertical(images: List[Image.Image]) -> Image.Image:
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    canvas = Image.new("RGB", (max(widths), sum(heights)))
    y = 0
    for img in images:
        canvas.paste(img, (0, y))
        y += img.height
    return canvas


_SDPA_ENABLE_GQA_PATCHED = False


def _patch_sdpa_enable_gqa_compat() -> None:
    global _SDPA_ENABLE_GQA_PATCHED
    if _SDPA_ENABLE_GQA_PATCHED:
        return

    original_sdpa = F.scaled_dot_product_attention

    def sdpa_compat(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if enable_gqa and key is not None and value is not None:
            q_heads = query.size(-3)
            k_heads = key.size(-3)
            v_heads = value.size(-3)
            if q_heads != k_heads:
                if q_heads % k_heads != 0:
                    raise ValueError(f"Cannot emulate GQA: query heads={q_heads}, key heads={k_heads}.")
                key = key.repeat_interleave(q_heads // k_heads, dim=-3)
            if q_heads != v_heads:
                if q_heads % v_heads != 0:
                    raise ValueError(f"Cannot emulate GQA: query heads={q_heads}, value heads={v_heads}.")
                value = value.repeat_interleave(q_heads // v_heads, dim=-3)

        kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        }
        if scale is not None:
            kwargs["scale"] = scale
        return original_sdpa(query, key, value, **kwargs)

    try:
        original_sdpa(
            torch.empty(1, 1, 1, 1),
            torch.empty(1, 1, 1, 1),
            torch.empty(1, 1, 1, 1),
            enable_gqa=False,
        )
    except TypeError as exc:
        if "enable_gqa" not in str(exc):
            raise
        F.scaled_dot_product_attention = sdpa_compat

    _SDPA_ENABLE_GQA_PATCHED = True


@contextmanager
def _diffusers_wan_import_compat():
    torch_library = getattr(torch, "library", None)
    if torch_library is None:
        yield
        return

    original_custom_op = getattr(torch_library, "custom_op", None)
    original_register_fake = getattr(torch_library, "register_fake", None)
    if original_custom_op is None or original_register_fake is None:
        yield
        return

    target_prefix = "_diffusers_flash_attn_3::"

    def _identity_decorator(fn=None):
        def wrap(func):
            return func

        return wrap if fn is None else fn

    def custom_op_compat(name, fn=None, /, **kwargs):
        if isinstance(name, str) and name.startswith(target_prefix):
            return _identity_decorator(fn)
        return original_custom_op(name, fn, **kwargs)

    def register_fake_compat(op, fn=None, /, **kwargs):
        if isinstance(op, str) and op.startswith(target_prefix):
            return _identity_decorator(fn)
        return original_register_fake(op, fn, **kwargs)

    torch_library.custom_op = custom_op_compat
    torch_library.register_fake = register_fake_compat
    try:
        yield
    finally:
        torch_library.custom_op = original_custom_op
        torch_library.register_fake = original_register_fake


class Wan2DiffusersBackbone(nn.Module):
    """Wan2.2-TI2V feature extractor in Diffusers format."""

    def __init__(self, cfg) -> None:
        super().__init__()
        wan_cfg = cfg.framework.wan
        model_name = getattr(wan_cfg, "base_wm", "./checkpoints/wan/Wan2.2-TI2V-5B-Diffusers")
        dtype = _resolve_dtype(getattr(wan_cfg, "torch_dtype", "bfloat16"))

        try:
            with _diffusers_wan_import_compat():
                from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
                from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
                from diffusers.video_processor import VideoProcessor
                from transformers import T5TokenizerFast, UMT5EncoderModel
        except (ImportError, RuntimeError) as exc:
            raise ImportError(
                "WALA_WAN requires Diffusers with Wan2.2 support. "
                "Install it with `pip install diffusers==0.38.0`."
            ) from exc

        logger.info(f"Loading Wan2.2-TI2V Diffusers backbone from {model_name}")
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            torch_dtype=dtype,
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=dtype,
        )
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            torch_dtype=dtype,
        )

        self.num_frames = int(getattr(wan_cfg, "num_frames", 5))
        if self.num_frames < 1:
            raise ValueError("wan.num_frames must be positive.")
        self.video_size = tuple(int(x) for x in getattr(wan_cfg, "video_size", [224, 224]))
        if len(self.video_size) != 2:
            raise ValueError(f"wan.video_size should be [height, width], got {self.video_size}.")
        self.concat_multi_camera = getattr(wan_cfg, "concat_multi_camera", None)
        self.text_max_length = int(getattr(wan_cfg, "text_max_length", 256))
        self.freeze_backbone = bool(getattr(wan_cfg, "freeze_backbone", True))
        self.freeze_text_encoder = bool(getattr(wan_cfg, "freeze_text_encoder", True))
        self.freeze_vae = bool(getattr(wan_cfg, "freeze_vae", True))

        self.video_processor = VideoProcessor(vae_scale_factor=2 ** len(self.vae.temperal_downsample))

        hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

        class _ConfigShim:
            pass

        class _ModelShim:
            pass

        self._model_config = _ConfigShim()
        self._model_config.hidden_size = hidden_size
        self._model_shim = _ModelShim()
        self._model_shim.config = self._model_config

        self._intermediate_features = []
        self._hooks = []
        self._extract_layers = list(getattr(wan_cfg, "extract_layers", [-1]))
        self._register_hooks()

        if self.freeze_text_encoder:
            self.text_encoder.requires_grad_(False)
        if self.freeze_vae:
            self.vae.requires_grad_(False)
        if self.freeze_backbone:
            self.transformer.requires_grad_(False)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        if enabled and hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()
        elif not enabled and hasattr(
            self.transformer,
            "disable_gradient_checkpointing",
        ):
            self.transformer.disable_gradient_checkpointing()

    @property
    def model(self):
        return self._model_shim

    def _register_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        num_blocks = len(self.transformer.blocks)
        for layer_idx in self._extract_layers:
            actual_idx = int(layer_idx) if int(layer_idx) >= 0 else num_blocks + int(layer_idx)
            if 0 <= actual_idx < num_blocks:
                self._hooks.append(self.transformer.blocks[actual_idx].register_forward_hook(self._capture_hook))

    def _capture_hook(self, module, inputs, output) -> None:
        feat = output[0] if isinstance(output, tuple) else output
        self._intermediate_features.append(feat)

    def _make_mosaic(self, sample_images) -> Image.Image:
        if isinstance(sample_images, Image.Image):
            views = [sample_images]
        elif isinstance(sample_images, (list, tuple)):
            views = [img for img in sample_images if isinstance(img, Image.Image)]
        else:
            views = [to_pil_preserve(sample_images)]

        if not views:
            views = [Image.new("RGB", (self.video_size[1], self.video_size[0]), color=0)]

        h, w = self.video_size
        mode = None if self.concat_multi_camera in {None, "none", "None"} else str(self.concat_multi_camera).lower()

        if mode == "robotwin" and len(views) >= 3:
            top_h = int(round(h * 2.0 / 3.0))
            bottom_h = h - top_h
            left_w = w // 2
            right_w = w - left_w
            top = _resize_pil(views[0], (top_h, w))
            left = _resize_pil(views[1], (bottom_h, left_w))
            right = _resize_pil(views[2], (bottom_h, right_w))
            return _concat_pil_vertical([top, _concat_pil_horizontal([left, right])])

        if mode == "horizontal" and len(views) > 1:
            widths = [w // len(views)] * len(views)
            widths[-1] += w - sum(widths)
            return _concat_pil_horizontal([
                _resize_pil(img, (h, view_w)) for img, view_w in zip(views, widths)
            ])

        if mode == "vertical" and len(views) > 1:
            heights = [h // len(views)] * len(views)
            heights[-1] += h - sum(heights)
            return _concat_pil_vertical([
                _resize_pil(img, (view_h, w)) for img, view_h in zip(views, heights)
            ])

        return _resize_pil(views[0], (h, w))

    def _encode_text(self, instructions: List[str]) -> torch.Tensor:
        device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            instructions,
            padding="max_length",
            max_length=self.text_max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)
        ctx = torch.no_grad() if self.freeze_text_encoder else nullcontext()
        with ctx:
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state
        return text_embeds.to(dtype=next(self.text_encoder.parameters()).dtype)

    def _encode_images_vae(self, images) -> torch.Tensor:
        device = next(self.vae.parameters()).device
        dtype = next(self.vae.parameters()).dtype
        height, width = self.video_size

        batch_videos = []
        for sample_images in images:
            mosaic = self._make_mosaic(sample_images)
            video_tensor = self.video_processor.preprocess_video(
                [mosaic],
                height=height,
                width=width,
            ).to(device=device, dtype=dtype)

            if video_tensor.shape[2] < self.num_frames:
                pad = video_tensor[:, :, -1:].repeat(1, 1, self.num_frames - video_tensor.shape[2], 1, 1)
                video_tensor = torch.cat([video_tensor, pad], dim=2)
            elif video_tensor.shape[2] > self.num_frames:
                video_tensor = video_tensor[:, :, : self.num_frames]

            batch_videos.append(video_tensor.squeeze(0))

        video = torch.stack(batch_videos, dim=0)
        ctx = torch.no_grad() if self.freeze_vae else nullcontext()
        with ctx:
            latents = self.vae.encode(video).latent_dist.sample()

        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        return (latents - latents_mean) / latents_std

    def build_inputs(self, images, instructions) -> dict:
        if len(images) != len(instructions):
            raise ValueError(f"Image/instruction batch mismatch: {len(images)} vs {len(instructions)}.")

        text_embeds = self._encode_text(instructions)
        latents = self._encode_images_vae(images)

        p_t, p_h, p_w = self.transformer.config.patch_size
        _, _, t, h, w = latents.shape
        seq_len = (t // p_t) * (h // p_h) * (w // p_w)
        timestep = torch.zeros(latents.shape[0], seq_len, device=latents.device, dtype=torch.long)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
        }

    def forward(self, **kwargs):
        self._intermediate_features.clear()
        _patch_sdpa_enable_gqa_compat()

        run_ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
        with run_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.transformer(
                hidden_states=kwargs["hidden_states"],
                timestep=kwargs["timestep"],
                encoder_hidden_states=kwargs["encoder_hidden_states"],
            )

        extracted = []
        for feat in self._intermediate_features:
            if feat.dim() == 5:
                b, c, t, h, w = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
            extracted.append(feat)

        if not extracted:
            feat = dit_output.sample if hasattr(dit_output, "sample") else dit_output
            if isinstance(feat, tuple):
                feat = feat[0]
            if feat.dim() == 5:
                b, c, t, h, w = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
            extracted.append(feat)

        class _WanOutput:
            def __init__(self, hidden_states):
                self.hidden_states = tuple(hidden_states)

        return _WanOutput(extracted)


class WanStateEncoder(nn.Module):
    """Encode normalized robot state vectors as tokens in the Wan hidden space."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_tokens: int,
        encoder_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, encoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(encoder_hidden_dim, num_tokens * hidden_dim),
        )
        self.token_embedding = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(states).reshape(
            states.shape[0],
            self.num_tokens,
            self.hidden_dim,
        )
        return self.out_norm(tokens + self.token_embedding)


class WanActionResamplerLayer(nn.Module):
    """Cross-attention layer that maps Wan tokens to action-query tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        use_state_cross_attention: bool = False,
        state_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_state_cross_attention = use_state_cross_attention
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        if self.use_state_cross_attention:
            self.state_query_norm = nn.LayerNorm(hidden_dim)
            self.state_memory_norm = nn.LayerNorm(hidden_dim)
            self.state_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            self.state_gate = nn.Parameter(torch.tensor(float(state_gate_init)))
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        state_memory: Optional[torch.Tensor] = None,
        state_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.query_norm(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        q = self.cross_query_norm(queries)
        memory = self.memory_norm(memory)
        queries = queries + self.cross_attn(q, memory, memory, need_weights=False)[0]

        if self.use_state_cross_attention:
            if state_memory is None:
                raise ValueError("state_memory is required for cross-attention state injection.")
            state_q = self.state_query_norm(queries)
            normalized_state = self.state_memory_norm(state_memory)
            state_update = self.state_cross_attn(
                state_q,
                normalized_state,
                normalized_state,
                need_weights=False,
            )[0]
            if state_valid is not None:
                state_update = state_update * state_valid[:, None, None].to(state_update.dtype)
            queries = queries + self.state_gate * state_update

        return queries + self.ffn(queries)


class WanActionResampler(nn.Module):
    """Learnable action queries over Wan hidden tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_action_tokens: int,
        num_layers: int = 2,
        num_heads: int = 24,
        use_state_cross_attention: bool = False,
        state_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_state_cross_attention = use_state_cross_attention
        self.gradient_checkpointing = False
        self.action_queries = nn.Parameter(torch.randn(1, num_action_tokens, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            WanActionResamplerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                use_state_cross_attention=use_state_cross_attention,
                state_gate_init=state_gate_init,
            )
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(
        self,
        wan_tokens: torch.Tensor,
        state_tokens: Optional[torch.Tensor] = None,
        state_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        queries = self.action_queries.expand(wan_tokens.shape[0], -1, -1)
        for layer in self.layers:
            should_checkpoint = (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and any(param.requires_grad for param in layer.parameters())
            )
            if should_checkpoint and state_tokens is not None:
                queries = checkpoint(
                    lambda query, memory, state_memory, current_layer=layer: current_layer(
                        query,
                        memory,
                        state_memory=state_memory,
                        state_valid=state_valid,
                    ),
                    queries,
                    wan_tokens,
                    state_tokens,
                    use_reentrant=False,
                )
            elif should_checkpoint:
                queries = checkpoint(
                    lambda query, memory, current_layer=layer: current_layer(
                        query,
                        memory,
                    ),
                    queries,
                    wan_tokens,
                    use_reentrant=False,
                )
            else:
                queries = layer(
                    queries,
                    wan_tokens,
                    state_memory=state_tokens,
                    state_valid=state_valid,
                )
        return self.out_norm(queries)


@FRAMEWORK_REGISTRY.register("WALA_WAN")
class WALAWanPolicy(WALAPolicy):
    """WALA policy with Wan2.2 visual features and latent action supervision."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(WALAWanPolicyDefaultConfig, config)

        self.wan_backbone = Wan2DiffusersBackbone(self.config)
        self.config.framework.action_model.action_hidden_dim = self.wan_backbone.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.action_model_type = str(
            self.config.framework.action_model.action_model_type
        ).strip()
        if self.action_model_type != "MLP":
            raise ValueError(
                f"Unsupported action_model_type={self.action_model_type!r}; expected 'MLP'."
            )

        act_cfg = self.config.framework.action_model
        dino_cfg = self.config.framework.dino
        ft_cfg = self.config.framework.transition_bottleneck
        wan_cfg = self.config.framework.wan

        self.future_action_window_size = act_cfg.future_action_window_size
        self.past_action_window_size = act_cfg.past_action_window_size
        self.chunk_len = 1 + self.future_action_window_size

        self.num_transition_tokens = getattr(
            ft_cfg,
            "num_transition_tokens",
            getattr(ft_cfg, "num_future_tokens", self.chunk_len),
        )
        self.dino_feature_dim = getattr(ft_cfg, "dino_dim", 768)
        self.transition_hidden_dim = getattr(ft_cfg, "hidden_dim", self.dino_feature_dim)
        self.l1_loss = nn.L1Loss()

        self.state_injection = str(getattr(wan_cfg, "state_injection", "text")).strip().lower()
        self.state_injection = self.state_injection.replace("-", "_")
        if self.state_injection not in {"text", "cross_attention"}:
            raise ValueError(
                "wan.state_injection must be either 'text' or 'cross_attention', "
                f"got {self.state_injection!r}."
            )

        use_state_cross_attention = self.state_injection == "cross_attention"
        self.wan_action_resampler = WanActionResampler(
            hidden_dim=act_cfg.action_hidden_dim,
            num_action_tokens=self.chunk_len,
            num_layers=int(getattr(wan_cfg, "resampler_layers", 2)),
            num_heads=int(getattr(wan_cfg, "resampler_num_heads", 24)),
            use_state_cross_attention=use_state_cross_attention,
            state_gate_init=float(getattr(wan_cfg, "state_cross_attention_gate_init", 0.1)),
        )
        self.state_dim = int(act_cfg.state_dim)
        if use_state_cross_attention:
            self.state_encoder = WanStateEncoder(
                state_dim=self.state_dim,
                hidden_dim=act_cfg.action_hidden_dim,
                num_tokens=int(getattr(wan_cfg, "state_num_tokens", 8)),
                encoder_hidden_dim=int(getattr(wan_cfg, "state_encoder_hidden_dim", 512)),
            )
        else:
            self.state_encoder = None

        self.dino_encoder = get_dino_model(backone_name=dino_cfg.dino_backbone)
        for param in self.dino_encoder.parameters():
            param.requires_grad = False

        self.use_depth_lam = getattr(ft_cfg, "use_depth", False)
        self.depth_source = str(getattr(ft_cfg, "depth_source", "auto")).lower()
        self.depth_anything = None
        if self.use_depth_lam and self.depth_source != "offline":
            depth_anything_root = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "third_party",
                    "Depth-Anything-V2",
                )
            )

            if depth_anything_root not in sys.path:
                sys.path.insert(0, depth_anything_root)

            from depth_anything_v2.dpt import DepthAnythingV2

            da_encoder = getattr(ft_cfg, "depth_anything_encoder", "vitl")
            model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
                "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
            }
            if da_encoder not in model_configs:
                raise ValueError(f"Unsupported depth_anything_encoder={da_encoder!r}.")

            self.depth_anything = DepthAnythingV2(**model_configs[da_encoder])
            ckpt = getattr(ft_cfg, "depth_anything_ckpt", None)
            if ckpt is None or ckpt == "":
                ckpt = os.path.join(
                    depth_anything_root,
                    "checkpoints",
                    f"depth_anything_v2_{da_encoder}.pth",
                )
            elif not os.path.isabs(ckpt):
                ckpt = os.path.abspath(ckpt)
            self.depth_anything.load_state_dict(torch.load(ckpt, map_location="cpu"))
            self.depth_anything.eval()
            for param in self.depth_anything.parameters():
                param.requires_grad = False

        self.transition_bottleneck = ContinuousTransitionBottleneckTokenizer(
            dino_dim=self.dino_feature_dim,
            hidden_dim=self.transition_hidden_dim,
            num_transition_tokens=self.num_transition_tokens,
            encoder_layers=getattr(ft_cfg, "encoder_layers", 8),
            decoder_layers=getattr(ft_cfg, "decoder_layers", 4),
            num_heads=ft_cfg.num_heads,
            max_future_steps=ft_cfg.max_future_steps,
            max_patches=ft_cfg.max_patches,
            smooth_l1_weight=ft_cfg.smooth_l1_weight,
            cosine_weight=ft_cfg.cosine_weight,
            composition_weight=getattr(ft_cfg, "composition_weight", 0),
            reencode_weight=getattr(ft_cfg, "reencode_weight", 0),
            content_invariance_weight=getattr(ft_cfg, "content_invariance_weight", 0),
            content_aug_scale_std=getattr(ft_cfg, "content_aug_scale_std", 0.05),
            content_aug_bias_std=getattr(ft_cfg, "content_aug_bias_std", 0.02),
            min_composition_steps=getattr(ft_cfg, "min_composition_steps", 2),
            use_depth=getattr(ft_cfg, "use_depth", False),
            depth_loss_weight=getattr(ft_cfg, "depth_loss_weight", 1.0),
            depth_gradient_loss_weight=getattr(ft_cfg, "depth_gradient_loss_weight", 0.2),
            depth_fusion_scale=getattr(ft_cfg, "depth_fusion_scale", 1.0),
            depth_patch_size=getattr(ft_cfg, "depth_patch_size", 16),
        )

        self.latent_action_to_transition_tokens_for_encoder = LatentActionToTransitionTokens(
            action_dim=act_cfg.action_hidden_dim,
            transition_dim=self.transition_hidden_dim,
            num_transition_tokens=self.num_transition_tokens,
            num_layers=getattr(ft_cfg, "latent_resampler_layers", 1),
            num_heads=ft_cfg.num_heads,
            max_action_tokens=getattr(ft_cfg, "max_action_tokens", 256),
        )

        self.latent_action_to_transition_tokens_for_decoder = LatentActionToTransitionTokens(
            action_dim=act_cfg.action_hidden_dim,
            transition_dim=self.transition_hidden_dim,
            num_transition_tokens=self.num_transition_tokens,
            num_layers=getattr(ft_cfg, "latent_resampler_layers", 1),
            num_heads=ft_cfg.num_heads,
            max_action_tokens=getattr(ft_cfg, "max_action_tokens", 256),
        )

        ft_mode = getattr(ft_cfg, "training_mode", "vla")
        if ft_mode in {"pretrain", "transition_bottleneck_pretrain", "transition_pretrain"}:
            frozen_modules = [
                self.wan_backbone,
                self.wan_action_resampler,
                self.action_model,
                self.latent_action_to_transition_tokens_for_encoder,
                self.latent_action_to_transition_tokens_for_decoder,
            ]
            if self.state_encoder is not None:
                frozen_modules.append(self.state_encoder)
            for module in frozen_modules:
                for param in module.parameters():
                    param.requires_grad = False
        else:
            if getattr(ft_cfg, "freeze_encoder_in_vla", True):
                self._set_transition_encoder_requires_grad(False)

            if not getattr(ft_cfg, "train_decoder_in_vla", True):
                self._set_transition_decoder_requires_grad(False)
                self._set_transition_depth_decoder_requires_grad(False)

        self._configure_gradient_checkpointing()
        self._profile_times = None
        self._profile_cuda_sync_enabled = False

    def _build_wan_instructions(self, instructions: List[str], states: List[object]) -> List[str]:
        prompt_suffix = f" Predict the next {self.chunk_len} robot actions."
        if self.state_injection == "cross_attention":
            return [instruction + prompt_suffix for instruction in instructions]

        state_dropout_prob = float(getattr(self.config.framework, "vla_state_dropout_prob", 0.0) or 0.0)
        if not 0.0 <= state_dropout_prob <= 1.0:
            raise ValueError(f"vla_state_dropout_prob must be in [0, 1], got {state_dropout_prob}")
        if self.training and state_dropout_prob > 0.0:
            states_for_prompt = [
                None if state is not None and torch.rand(()).item() < state_dropout_prob else state
                for state in states
            ]
        else:
            states_for_prompt = states

        state_texts = [self._format_state_as_text(state) for state in states_for_prompt]
        return [
            instruction + state_text + prompt_suffix
            for instruction, state_text in zip(instructions, state_texts)
        ]

    def _encode_state_tokens(
        self,
        states: List[object],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state_batch = torch.zeros(len(states), self.state_dim, dtype=torch.float32)
        state_valid = torch.zeros(len(states), dtype=torch.bool)

        for index, state in enumerate(states):
            if state is None:
                continue
            if torch.is_tensor(state):
                values = state.detach().to(device="cpu", dtype=torch.float32)
            else:
                values = torch.as_tensor(np.asarray(state), dtype=torch.float32)
            values = values.squeeze()
            if values.ndim == 0:
                values = values.reshape(1)
            elif values.ndim > 1:
                values = values[-1].reshape(-1)
            else:
                values = values.reshape(-1)
            if values.numel() > self.state_dim:
                raise ValueError(
                    f"State dimension {values.numel()} exceeds configured state_dim={self.state_dim}."
                )
            state_batch[index, : values.numel()] = values.clamp(-1.0, 1.0)
            state_valid[index] = True

        state_dropout_prob = float(getattr(self.config.framework, "vla_state_dropout_prob", 0.0) or 0.0)
        if not 0.0 <= state_dropout_prob <= 1.0:
            raise ValueError(f"vla_state_dropout_prob must be in [0, 1], got {state_dropout_prob}")
        if self.training and state_dropout_prob > 0.0:
            keep_mask = torch.rand(len(states)) >= state_dropout_prob
            state_valid &= keep_mask

        state_batch = state_batch.to(device=device)
        state_valid = state_valid.to(device=device)
        state_tokens = self.state_encoder(state_batch)
        return state_tokens, state_valid

    def _encode_wan_latent_actions(self, batch_images, instructions, states) -> torch.Tensor:
        wan_inputs = self.wan_backbone.build_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wan_outputs = self.wan_backbone(**wan_inputs)
            wan_tokens = wan_outputs.hidden_states[-1]
            if self.state_injection == "cross_attention":
                state_tokens, state_valid = self._encode_state_tokens(states, wan_tokens.device)
                return self.wan_action_resampler(
                    wan_tokens,
                    state_tokens=state_tokens,
                    state_valid=state_valid,
                )
            return self.wan_action_resampler(wan_tokens)

    def forward_vla(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        (
            batch_images,
            instructions,
            actions,
            head_past_and_current_images,
            head_future_images,
            head_past_and_current_depth_keys,
            head_future_depth_keys,
        ) = self._build_training_visual_inputs(examples)

        states = [example.get("state", None) for example in examples]
        instructions = self._build_wan_instructions(instructions, states)
        valid_indices = [i for i, action in enumerate(actions) if action is not None]

        dino_feature_current, dino_feature_future = self._encode_dino_current_and_future(
            head_past_and_current_images=head_past_and_current_images,
            head_future_images=head_future_images,
        )

        latent_action = self._encode_wan_latent_actions(batch_images, instructions, states)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            transition_alignment_loss = 0.0 * latent_action.sum()
            transition_alignment_cosine_loss = 0.0 * latent_action.sum()
            transition_alignment_l1_loss = 0.0 * latent_action.sum()
            transition_depth_decode_loss = 0.0 * latent_action.sum()
            transition_depth_decode_pixel_loss = 0.0 * latent_action.sum()
            transition_depth_decode_grad_loss = 0.0 * latent_action.sum()

            current_depth = None
            future_depth = None
            if getattr(self.config.framework.transition_bottleneck, "use_depth", False):
                current_depth, future_depth = self._encode_depth_current_and_future(
                    head_past_and_current_images=head_past_and_current_images,
                    head_future_images=head_future_images,
                    head_past_and_current_depth_keys=head_past_and_current_depth_keys,
                    head_future_depth_keys=head_future_depth_keys,
                )

            if getattr(self.config.framework.transition_bottleneck, "use_transition_alignment", True):
                with torch.no_grad():
                    target_transition_tokens = self.transition_bottleneck.encode(
                        dino_feature_current.float(),
                        dino_feature_future.float(),
                        current_depth=current_depth,
                        future_depth=future_depth,
                    )
                pred_transition_tokens_for_encoder = self.latent_action_to_transition_tokens_for_encoder(latent_action)
                target_transition_tokens = target_transition_tokens.detach().float()

                pred_norm = F.normalize(pred_transition_tokens_for_encoder.float(), dim=-1)
                target_norm = F.normalize(target_transition_tokens, dim=-1)
                transition_alignment_cosine_loss = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()

                pred_ln = F.layer_norm(
                    pred_transition_tokens_for_encoder.float(),
                    pred_transition_tokens_for_encoder.shape[-1:],
                )
                target_ln = F.layer_norm(target_transition_tokens, target_transition_tokens.shape[-1:])
                transition_alignment_l1_loss = F.smooth_l1_loss(pred_ln, target_ln)

                alignment_l1_w = getattr(
                    self.config.framework.transition_bottleneck,
                    "transition_alignment_l1_weight",
                    0.1,
                )
                transition_alignment_loss = transition_alignment_cosine_loss + alignment_l1_w * transition_alignment_l1_loss

            transition_decode_loss = 0.0 * latent_action.sum()
            transition_decode_l1_loss = 0.0 * latent_action.sum()
            transition_decode_cosine_loss = 0.0 * latent_action.sum()

            if getattr(self.config.framework.transition_bottleneck, "use_transition_decode", True):
                target_delta = (
                    dino_feature_future.float()
                    - dino_feature_current[:, None, :, :].float()
                ).detach()

                pred_transition_tokens_for_decoder = self.latent_action_to_transition_tokens_for_decoder(latent_action)

                pred_delta = self.transition_bottleneck.decode_delta(
                    dino_feature_current.float(),
                    pred_transition_tokens_for_decoder.float(),
                    num_future_steps=dino_feature_future.shape[1],
                )

                if getattr(self.config.framework.transition_bottleneck, "transition_decode_use_all_future", True):
                    pred_delta_for_loss = pred_delta.float()
                    target_delta_for_loss = target_delta.float()
                else:
                    pred_delta_for_loss = pred_delta[:, -1, :, :].float()
                    target_delta_for_loss = target_delta[:, -1, :, :].float()

                transition_decode_l1_loss = F.l1_loss(pred_delta_for_loss, target_delta_for_loss)

                pred_delta_norm = F.normalize(pred_delta_for_loss.float(), dim=-1)
                target_delta_norm = F.normalize(target_delta_for_loss.float(), dim=-1)
                transition_decode_cosine_loss = 1.0 - (pred_delta_norm * target_delta_norm).sum(dim=-1).mean()

                decode_cos_w = getattr(
                    self.config.framework.transition_bottleneck,
                    "transition_decode_cosine_weight",
                    0.1,
                )

                if getattr(self.config.framework.transition_bottleneck, "use_depth", False):
                    bsz, steps, patches, _ = dino_feature_future.shape
                    depth_patch_hw = self.transition_bottleneck._infer_square_patch_hw(patches)

                    current_depth_norm, future_depth_norm = self.transition_bottleneck._normalize_depth_clip(
                        current_depth=current_depth.to(device=dino_feature_current.device),
                        future_depth=future_depth.to(device=dino_feature_current.device),
                    )

                    target_depth_delta = (
                        future_depth_norm - current_depth_norm[:, None, :, :]
                    ).detach()

                    pred_depth_delta = self.transition_bottleneck._decode_dense_depth_delta(
                        current_depth=current_depth_norm,
                        transition_tokens=pred_transition_tokens_for_decoder.float(),
                        num_future_steps=steps,
                        patch_hw=depth_patch_hw,
                    )

                    if getattr(self.config.framework.transition_bottleneck, "depth_decoder_use_all_future", True):
                        pred_depth_delta_for_loss = pred_depth_delta.float()
                        target_depth_delta_for_loss = target_depth_delta.float()
                    else:
                        pred_depth_delta_for_loss = pred_depth_delta[:, -1, :, :].float()
                        target_depth_delta_for_loss = target_depth_delta[:, -1, :, :].float()

                    transition_depth_decode_pixel_loss = F.smooth_l1_loss(
                        pred_depth_delta_for_loss,
                        target_depth_delta_for_loss,
                    )
                    transition_depth_decode_grad_loss = self.transition_bottleneck._depth_gradient_l1_loss(
                        pred_depth_delta_for_loss,
                        target_depth_delta_for_loss,
                    )
                    depth_grad_w = getattr(
                        self.config.framework.transition_bottleneck,
                        "depth_gradient_loss_weight",
                        0.2,
                    )
                    transition_depth_decode_loss = (
                        transition_depth_decode_pixel_loss
                        + depth_grad_w * transition_depth_decode_grad_loss
                    )

                depth_decode_w = getattr(self.config.framework.transition_bottleneck, "depth_decoder_loss_weight", 0.5)

                transition_decode_loss = (
                    transition_decode_l1_loss
                    + decode_cos_w * transition_decode_cosine_loss
                    + depth_decode_w * transition_depth_decode_loss
                )

            transition_alignment_w = getattr(
                self.config.framework.transition_bottleneck,
                "transition_alignment_loss_weight",
                0.01,
            )
            transition_decode_w = getattr(
                self.config.framework.transition_bottleneck,
                "transition_decode_loss_weight",
                0.1,
            )

        with torch.autocast("cuda", dtype=torch.float32):
            if len(valid_indices) > 0:
                latent_action = latent_action.float()
                valid_actions = [actions[i] for i in valid_indices]
                valid_latent_action = latent_action[valid_indices]

                actions_tensor = torch.tensor(
                    np.array(valid_actions), device=latent_action.device, dtype=latent_action.dtype
                )
                if actions_tensor.shape[1] < self.chunk_len:
                    raise AssertionError(
                        f"Action sequence length ({actions_tensor.shape[1]}) is shorter than "
                        f"chunk_len ({self.chunk_len}). Please check your config and data alignment."
                    )
                actions_target = actions_tensor[:, -self.chunk_len:, :]
                action_loss = self.action_model(valid_latent_action, actions_target)
            else:
                action_loss = 0.0 * latent_action.mean()

            total_loss = (
                action_loss.float()
                + transition_alignment_w * transition_alignment_loss.float()
                + transition_decode_w * transition_decode_loss.float()
            )

        return {
            "action_loss": action_loss,
            "transition_alignment_loss": transition_alignment_loss,
            "transition_alignment_cosine_loss": transition_alignment_cosine_loss,
            "transition_alignment_l1_loss": transition_alignment_l1_loss,
            "transition_decode_loss": transition_decode_loss,
            "transition_decode_l1_loss": transition_decode_l1_loss,
            "transition_decode_cosine_loss": transition_decode_cosine_loss,
            "transition_depth_decode_loss": transition_depth_decode_loss,
            "transition_depth_decode_pixel_loss": transition_depth_decode_pixel_loss,
            "transition_depth_decode_grad_loss": transition_depth_decode_grad_loss,
            "total_loss": total_loss,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> dict:
        if type(examples) is not list:
            examples = [examples]

        if len(examples) > 1:
            valid_examples = [example for example in examples if example.get("action", None) is not None]
            if not valid_examples:
                return {"normalized_actions": np.array([])}
            examples = valid_examples

        raw_batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        states = [example.get("state", None) for example in examples]

        def safe_get(images, idx):
            try:
                return images[idx][0][-1]
            except Exception:
                return None

        batch_images = []
        for images in raw_batch_images:
            head = safe_get(images, 0)
            left = safe_get(images, 1)
            right = safe_get(images, 2)
            ref_img = head or left or right
            if ref_img is not None:
                blank_img = Image.new(ref_img.mode, ref_img.size, color=0)
            else:
                blank_img = Image.new("RGB", (224, 224), color=0)
            batch_images.append([head or blank_img, left or blank_img, right or blank_img])

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        instructions = self._build_wan_instructions(instructions, states)
        latent_action = self._encode_wan_latent_actions(batch_images, instructions, states).float()

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(latent_action)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
