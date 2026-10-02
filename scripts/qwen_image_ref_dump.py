#!/usr/bin/env python3
"""Golden-reference dump for Qwen-Image-2.1 (diffusers, CPU fp32) — the
input of the native sidecar's golden test (Todo 3.2):

  cargo run --release -p spark-model --features cuda,gpu-examples \\
      --example qwen_image_golden -- <golden_dir>

Runs each pipeline stage separately (only one big component is in RAM at a
time, ~35 GB peak) and writes the intermediate tensors as safetensors:

  stage text : input_ids, prompt_embeds (pre-norm last hidden, sys prefix dropped)
  stage dit  : noise, sigmas, per-step noise_pred, selected block activations,
               final latents
  stage vae  : decoder block outputs + decoded image (RGBA, [-1, 1]) + ref.png

The defaults (256x256, 4 steps) keep the CPU run to a few minutes; the
golden files (~240 MB) are not checked in.

Reference environment (CPU-only, no CUDA wheel needed):
  uv venv refenv && uv pip install --python refenv/bin/python \\
      torch torchvision --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python refenv/bin/python "transformers>=5.17" accelerate \\
      pillow safetensors numpy git+https://github.com/huggingface/diffusers

Usage:
  for s in text dit vae; do
      refenv/bin/python scripts/qwen_image_ref_dump.py $s --out golden; done
  [--prompt P] [--height 256] [--width 256] [--steps 4] [--seed 42]
"""

import argparse
import glob
import os

import numpy as np
import torch
from safetensors.torch import load_file, save_file

SNAP = sorted(
    glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/*"))
)[-1]


def stage_text(args):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(os.path.join(SNAP, "processor"))
    sys_prompt = "Comprehend and analyze the provided prompt."
    template = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    sys_message = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}]
    sys_tokens = processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
    drop_idx = len(sys_tokens[0])
    text = template.format(args.prompt)
    inputs = processor(text=[text], padding=True, padding_side="left", return_tensors="pt")
    print("drop_idx", drop_idx, "input_ids", inputs.input_ids.shape, inputs.input_ids[0, :12].tolist())

    enc = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(SNAP, "text_encoder"), dtype=torch.bfloat16 if args.bf16 else torch.float32
    ).eval()
    text_model = getattr(enc.model, "language_model", enc.model)
    handle = text_model.norm.register_forward_hook(lambda module, a, output: a[0])
    with torch.no_grad():
        out = enc(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
    handle.remove()
    hidden = out.hidden_states[-1][0].float()
    embeds = hidden[drop_idx:]
    print("prompt_embeds", embeds.shape, "absmean", embeds.abs().mean().item())
    save_file(
        {
            "input_ids": inputs.input_ids[0].to(torch.int64),
            "drop_idx": torch.tensor([drop_idx]),
            "hidden_all": hidden.clone(),
            "prompt_embeds": embeds.clone(),
            "layer0_in": out.hidden_states[0][0].float().clone(),
            "layer1_out": out.hidden_states[1][0].float().clone(),
        },
        os.path.join(args.out, "text.safetensors"),
    )


def calculate_shift(image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def stage_dit(args):
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    text = load_file(os.path.join(args.out, "text.safetensors"))
    prompt_embeds = text["prompt_embeds"][None].to(dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(SNAP, "scheduler"))
    dit = QwenImage21Transformer2DModel.from_pretrained(os.path.join(SNAP, "transformer"), dtype=dtype).eval()

    lat_h = 2 * (args.height // 32)
    lat_w = 2 * (args.width // 32)
    gen = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn((1, 1, 64, lat_h, lat_w), generator=gen, dtype=torch.float32)
    latents = noise.view(1, 64, lat_h * lat_w).transpose(1, 2).contiguous().to(dtype)

    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    mu = calculate_shift(
        latents.shape[1],
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(sigmas=sigmas, mu=mu)
    timesteps = scheduler.timesteps
    print("mu", mu, "sigmas", scheduler.sigmas.tolist())

    image_pad_mask = torch.zeros(prompt_embeds.shape[:2], dtype=torch.bool)
    image_pad_mask = torch.cat([image_pad_mask, image_pad_mask.new_ones(1, latents.shape[1] // 4)], dim=1)
    img_shapes = [[(1, lat_h, lat_w)]]

    dump = {
        "noise_packed": latents.float().clone(),
        "sigmas": scheduler.sigmas.float().clone(),
        "timesteps": timesteps.float().clone(),
        "mu": torch.tensor([mu], dtype=torch.float64),
    }

    # Capture a few intermediate activations on steps 0 (prefill) and 1 (cached decode).
    captures = {}
    step_tag = {"i": 0}

    def hook(name):
        def fn(module, inputs, output):
            if step_tag["i"] < 2:
                o = output[0] if isinstance(output, tuple) else output
                captures[f"step{step_tag['i']}.{name}"] = o.detach().float().contiguous().clone()

        return fn

    handles = [
        dit.txt_in.register_forward_hook(hook("txt_in")),
        dit.img_in.register_forward_hook(hook("img_in")),
        dit.modulation.register_forward_hook(hook("modulation")),
        dit.transformer_blocks[0].attn.register_forward_hook(hook("block0.attn")),
        dit.transformer_blocks[0].register_forward_hook(hook("block0")),
        dit.transformer_blocks[1].register_forward_hook(hook("block1")),
        dit.transformer_blocks[15].register_forward_hook(hook("block15")),
        dit.transformer_blocks[31].register_forward_hook(hook("block31")),
        dit.norm_out.register_forward_hook(hook("norm_out")),
    ]

    cache = QwenImage21KVCache(len(dit.transformer_blocks)) if not args.no_kv_cache else None
    scheduler.set_begin_index(0)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            step_tag["i"] = i
            mode = None if cache is None else ("extract" if i == 0 else "cached")
            timestep = t.expand(1).to(latents.dtype)
            pred = dit(
                hidden_states=latents,
                timestep=timestep / 1000,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=None,
                img_shapes=img_shapes,
                img_mask=image_pad_mask,
                kv_cache=cache,
                kv_cache_mode=mode,
                return_dict=False,
            )[0]
            pred = pred[:, -latents.size(1) :]
            dump[f"noise_pred.{i}"] = pred.float().clone()
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
            dump[f"latents.{i}"] = latents.float().clone()
            print(f"step {i} t={t.item():.3f} pred absmean {pred.float().abs().mean().item():.5f}", flush=True)
    for h in handles:
        h.remove()
    dump.update(captures)
    dump["final_latents"] = latents.float().clone()
    dump["lat_hw"] = torch.tensor([lat_h, lat_w])
    save_file({k: v.clone().contiguous() for k, v in dump.items()}, os.path.join(args.out, "dit.safetensors"))


def stage_vae(args):
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(SNAP, "vae"), dtype=torch.float32).eval()
    d = load_file(os.path.join(args.out, "dit.safetensors"))
    lat_h, lat_w = d["lat_hw"].tolist()
    latents = d["final_latents"]
    latents = latents.transpose(1, 2).reshape(1, 64, 1, lat_h, lat_w)
    mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    z = latents * std + mean

    captures = {}

    def hook(name):
        def fn(module, inputs, output):
            captures[name] = output.detach().float().contiguous().clone()

        return fn

    dec = vae.decoder
    handles = [
        vae.post_quant_conv.register_forward_hook(hook("post_quant_conv")),
        dec.conv_in.register_forward_hook(hook("conv_in")),
        dec.mid_block.resnets[0].register_forward_hook(hook("mid.res0")),
        dec.mid_block.attentions[0].register_forward_hook(hook("mid.attn0")),
        dec.mid_block.register_forward_hook(hook("mid")),
    ]
    for i, blk in enumerate(dec.up_blocks):
        handles.append(blk.register_forward_hook(hook(f"up{i}")))
    with torch.no_grad():
        image = vae.decode(z, return_dict=False)[0][:, :, 0]
    for h in handles:
        h.remove()
    print("image", image.shape, "min", image.min().item(), "max", image.max().item())
    captures["z"] = z
    captures["image"] = image
    save_file({k: v.clone().contiguous() for k, v in captures.items()}, os.path.join(args.out, "vae.safetensors"))

    from PIL import Image

    arr = ((image[0].permute(1, 2, 0) / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(arr).save(os.path.join(args.out, "ref.png"))




# ── edit (image-conditioned) stages — Todo 3.2a golden follow-up ──────────
#
# A deterministic 256x256 condition image plus `output_resolution=256`
# keeps every CPU fp32 forward small: the ViT sees 256 patches (64 merged
# tokens), the VAE encodes a 16x16 latent grid and the DiT prefix adds one
# 256-token condition block. The Rust harness reads edit_src.png and runs
# its own preprocessing, so resize + normalization are covered too.

EDIT_RES = 256


def make_condition_image(path):
    from PIL import Image

    img = Image.new("RGB", (EDIT_RES, EDIT_RES))
    px = img.load()
    for y in range(EDIT_RES):
        for x in range(EDIT_RES):
            # Smooth gradients + a diagonal stripe: structured, not flat.
            stripe = 255 if (x + y) % 64 < 12 else 0
            px[x, y] = (x % 256, y % 256, stripe)
    img.save(path)
    return img


def stage_edit_text(args):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(os.path.join(SNAP, "processor"))
    img = make_condition_image(os.path.join(args.out, "edit_src.png"))

    sys_prompt = "Comprehend and analyze the provided prompt."
    template = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    sys_message = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}]
    drop_idx = len(processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])
    text = template.format(args.prompt)
    inputs = processor(text=[text], images=[img], padding=True, padding_side="left", return_tensors="pt")
    img_token_id = processor.tokenizer.encode("<|image_pad|>")[0]
    print("drop_idx", drop_idx, "input_ids", inputs.input_ids.shape, "grid", inputs.image_grid_thw.tolist())

    enc = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(SNAP, "text_encoder"), dtype=torch.bfloat16 if args.bf16 else torch.float32
    ).eval()
    text_model = getattr(enc.model, "language_model", enc.model)
    handle = text_model.norm.register_forward_hook(lambda module, a, output: a[0])
    forward_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "pixel_values": inputs.pixel_values.to(enc.dtype),
        "image_grid_thw": inputs.image_grid_thw,
        "output_hidden_states": True,
    }
    if hasattr(inputs, "mm_token_type_ids"):
        forward_kwargs["mm_token_type_ids"] = inputs.mm_token_type_ids
    with torch.no_grad():
        out = enc(**forward_kwargs)
    handle.remove()
    hidden = out.hidden_states[-1][0].float()
    embeds = hidden[drop_idx:]
    pad_mask = (inputs.input_ids[0] == img_token_id)[drop_idx:]
    with torch.no_grad():
        feats = enc.model.get_image_features(
            inputs.pixel_values.to(enc.dtype), inputs.image_grid_thw, return_dict=True
        )
    image_embeds = feats.pooler_output
    if isinstance(image_embeds, (list, tuple)):
        image_embeds = torch.cat(list(image_embeds), dim=0)
    image_embeds = image_embeds.float().reshape(-1, image_embeds.shape[-1])
    def flat(d):
        if isinstance(d, (list, tuple)):
            d = torch.cat([x.reshape(-1, x.shape[-1]) for x in d], dim=0)
        return d.float().reshape(-1, d.shape[-1]).contiguous()

    deepstack = [flat(d) for d in feats.deepstack_features]
    print("image_embeds", tuple(image_embeds.shape), "deepstack", [tuple(d.shape) for d in deepstack])
    print("prompt_embeds", embeds.shape, "absmean", embeds.abs().mean().item(), "pad tokens", int(pad_mask.sum()))
    save_file(
        {
            "input_ids": inputs.input_ids[0].to(torch.int64),
            "drop_idx": torch.tensor([drop_idx]),
            "img_token_id": torch.tensor([img_token_id]),
            "hidden_all": hidden.clone(),
            "prompt_embeds": embeds.clone(),
            "image_pad_mask": pad_mask.to(torch.int64).clone(),
            "image_embeds": image_embeds.contiguous().clone(),
            "deepstack_0": deepstack[0].clone(),
            "deepstack_1": deepstack[1].clone(),
            "deepstack_2": deepstack[2].clone(),
        },
        os.path.join(args.out, "edit_text.safetensors"),
    )


def stage_edit_vae(args):
    from PIL import Image

    from diffusers import AutoencoderKLQwenImage21
    from diffusers.image_processor import VaeImageProcessor

    vae = AutoencoderKLQwenImage21.from_pretrained(os.path.join(SNAP, "vae"), dtype=torch.float32).eval()
    ip = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=vae.config.z_dim)
    img = Image.open(os.path.join(args.out, "edit_src.png")).convert("RGBA")
    x = ip.preprocess(img, width=EDIT_RES, height=EDIT_RES).unsqueeze(2)
    with torch.no_grad():
        enc = vae.encode(x, return_dict=False)[0]
        lat = enc.mode()
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    lat = (lat - mean) / std
    lh, lw = lat.shape[3], lat.shape[4]
    packed = lat.view(1, vae.config.z_dim, lh * lw).transpose(1, 2).contiguous()
    print("cond_latents", packed.shape, "absmean", packed.abs().mean().item())
    save_file(
        {
            "vae_in": x[:, :, 0].float().contiguous().clone(),
            "cond_latents": packed.float().clone(),
            "lat_hw": torch.tensor([lh, lw]),
        },
        os.path.join(args.out, "edit_vae.safetensors"),
    )


def stage_edit_dit(args):
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    text = load_file(os.path.join(args.out, "edit_text.safetensors"))
    cond = load_file(os.path.join(args.out, "edit_vae.safetensors"))
    prompt_embeds = text["prompt_embeds"][None].to(dtype)
    pad_mask = text["image_pad_mask"].bool()[None]
    cond_latents = cond["cond_latents"].to(dtype)
    clh, clw = cond["lat_hw"].tolist()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(SNAP, "scheduler"))
    dit = QwenImage21Transformer2DModel.from_pretrained(os.path.join(SNAP, "transformer"), dtype=dtype).eval()

    lat_h = 2 * (EDIT_RES // 32)
    lat_w = 2 * (EDIT_RES // 32)
    gen = torch.Generator("cpu").manual_seed(args.seed)
    noise = torch.randn((1, 1, 64, lat_h, lat_w), generator=gen, dtype=torch.float32)
    latents = noise.view(1, 64, lat_h * lat_w).transpose(1, 2).contiguous().to(dtype)

    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    mu = calculate_shift(
        latents.shape[1],
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(sigmas=sigmas, mu=mu)
    timesteps = scheduler.timesteps

    img_mask = torch.cat([pad_mask, pad_mask.new_ones(1, latents.shape[1] // 4)], dim=1)
    img_shapes = [[(1, clh, clw), (1, lat_h, lat_w)]]

    dump = {
        "noise_packed": latents.float().clone(),
        "sigmas": scheduler.sigmas.float().clone(),
        "lat_hw": torch.tensor([lat_h, lat_w]),
    }
    captures = {}
    step_tag = {"i": 0}

    def hook(name):
        def fn(module, inputs, output):
            if step_tag["i"] < 2:
                o = output[0] if isinstance(output, tuple) else output
                captures[f"step{step_tag['i']}.{name}"] = o.detach().float().contiguous().clone()

        return fn

    handles = [
        dit.transformer_blocks[0].register_forward_hook(hook("block0")),
        dit.transformer_blocks[15].register_forward_hook(hook("block15")),
        dit.transformer_blocks[31].register_forward_hook(hook("block31")),
    ]
    cache = QwenImage21KVCache(len(dit.transformer_blocks))
    scheduler.set_begin_index(0)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            step_tag["i"] = i
            latent_model_input = torch.cat([cond_latents, latents], dim=1)
            pred = dit(
                hidden_states=latent_model_input,
                timestep=t.expand(1).to(latents.dtype) / 1000,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=None,
                img_shapes=img_shapes,
                img_mask=img_mask,
                kv_cache=cache,
                kv_cache_mode="extract" if i == 0 else "cached",
                return_dict=False,
            )[0]
            pred = pred[:, -latents.size(1) :]
            dump[f"noise_pred.{i}"] = pred.float().clone()
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
            print(f"step {i} t={t.item():.3f} pred absmean {pred.float().abs().mean().item():.5f}", flush=True)
    for h in handles:
        h.remove()
    dump.update(captures)
    dump["final_latents"] = latents.float().clone()
    save_file({k: v.clone().contiguous() for k, v in dump.items()}, os.path.join(args.out, "edit_dit.safetensors"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["text", "dit", "vae", "edit_text", "edit_vae", "edit_dit"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default='A neon shop sign that reads "ATLAS", rainy night')
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no-kv-cache", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)
    {
        "text": stage_text,
        "dit": stage_dit,
        "vae": stage_vae,
        "edit_text": stage_edit_text,
        "edit_vae": stage_edit_vae,
        "edit_dit": stage_edit_dit,
    }[args.stage](args)


if __name__ == "__main__":
    main()
