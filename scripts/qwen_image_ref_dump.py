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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["text", "dit", "vae"])
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
    {"text": stage_text, "dit": stage_dit, "vae": stage_vae}[args.stage](args)


if __name__ == "__main__":
    main()
