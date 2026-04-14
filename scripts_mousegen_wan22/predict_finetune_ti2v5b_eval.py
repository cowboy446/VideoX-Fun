import argparse
import json
import hashlib
import os
import sys
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    AutoTokenizer,
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2TI2VPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import calculate_dimensions, filter_kwargs, get_image_to_video_latent, save_videos_grid
CUDA_VISIBLE_DEVICES = "7"

# 将CUDA_VISIBLE_DEVICES设置为环境变量，以便在脚本中使用
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

DEFAULT_DATASET_ROOT = "datasets"
DEFAULT_IMAGE_ROOT = "datasets/validset/f9/poor_dataset_01"
DEFAULT_META_JSON = "datasets/poor_dataset_01/meta_text_video_pairs_qwen_8b.json"
DEFAULT_OUTPUT_ROOT = "datasets/validset/finetune_eval_0_original_f49_iter25"
DEFAULT_FINETUNE_CHECKPOINT_DIR = None
DEFAULT_CONFIG_PATH = "config/wan2.2/wan_civitai_5b.yaml"
DEFAULT_MODEL_NAME = "models/wan22/Wan2.2-TI2V-5B"


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a finetuned Wan2.2 TI2V 5B model on validset/f0 samples.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--meta-json", default=DEFAULT_META_JSON)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--finetune-checkpoint-dir", default=DEFAULT_FINETUNE_CHECKPOINT_DIR, help="Optional finetune checkpoint directory. Omit to use the original ti2v5b weights.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--transformer-path", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--negative-prompt", default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    parser.add_argument("--sample-width", type=int, default=704)
    parser.add_argument("--sample-height", type=int, default=512)
    parser.add_argument("--video-length", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--sampler-name", default="Flow_Unipc", choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    parser.add_argument("--shift", type=int, default=5)
    parser.add_argument("--weight-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--gpu-memory-mode", default="model_full_load", choices=["model_full_load", "model_full_load_and_qfloat8", "model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"])
    parser.add_argument("--enable-teacache", action="store_true", default=True)
    parser.add_argument("--disable-teacache", action="store_false", dest="enable_teacache")
    parser.add_argument("--teacache-threshold", type=float, default=0.10)
    parser.add_argument("--num-skip-start-steps", type=int, default=5)
    parser.add_argument("--teacache-offload", action="store_true", default=False)
    parser.add_argument("--cfg-skip-ratio", type=float, default=0)
    parser.add_argument("--enable-riflex", action="store_true", default=False)
    parser.add_argument("--riflex-k", type=int, default=6)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--fsdp-dit", action="store_true", default=False)
    parser.add_argument("--fsdp-text-encoder", action="store_true", default=True)
    parser.add_argument("--compile-dit", action="store_true", default=False)
    parser.add_argument("--lora-weight", type=float, default=0.55)
    parser.add_argument("--lora-high-weight", type=float, default=0.55)
    return parser


def load_prompt_map(meta_json_path):
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta_items = json.load(f)

    prompt_map = {}
    for item in meta_items:
        file_path = item.get("file_path")
        prompt = item.get("text")
        if file_path and prompt:
            prompt_map[file_path] = prompt
    return prompt_map


def resolve_prompt(prompt_map, rel_mp4_path):
    candidate = rel_mp4_path.as_posix()
    if candidate in prompt_map:
        return prompt_map[candidate]

    parts = candidate.split("/", 1)
    if len(parts) == 2 and parts[1] in prompt_map:
        return prompt_map[parts[1]]

    if candidate.startswith("poor_dataset_01/"):
        stripped = candidate[len("poor_dataset_01/"):]
        if stripped in prompt_map:
            return prompt_map[stripped]

    return None


def prompt_to_filename(prompt, max_length=60):
    filename = prompt.strip().replace("/", "_").replace("\\", "_")
    for character in [":", "*", "?", '"', "<", ">", "|", "\n", "\r", "\t"]:
        filename = filename.replace(character, "_")
    filename = " ".join(filename.split())
    filename = filename.strip(" ._")
    if not filename:
        filename = "prompt"
    if len(filename) > max_length:
        filename = filename[:max_length].rstrip(" ._")
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{filename}__{digest}"


def load_state_dict(path):
    if path is None:
        return None
    if path.endswith("safetensors"):
        from safetensors.torch import load_file

        return load_file(path)

    state_dict = torch.load(path, map_location="cpu")
    return state_dict["state_dict"] if isinstance(state_dict, dict) and "state_dict" in state_dict else state_dict


def create_pipeline(args):
    device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    weight_dtype = torch.bfloat16 if args.weight_dtype == "bfloat16" else torch.float16

    transformer_path = args.transformer_path
    if transformer_path is None and args.finetune_checkpoint_dir:
        finetune_checkpoint_dir = Path(args.finetune_checkpoint_dir)
        if not finetune_checkpoint_dir.exists():
            raise FileNotFoundError(f"Finetune checkpoint directory does not exist: {finetune_checkpoint_dir}")

        candidates = [
            finetune_checkpoint_dir / "diffusion_pytorch_model.safetensors",
            finetune_checkpoint_dir / "adapter_model.safetensors",
            finetune_checkpoint_dir / "pytorch_model.bin",
        ]
        for candidate in candidates:
            if candidate.exists():
                transformer_path = str(candidate)
                break

    config = OmegaConf.load(args.config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(args.model_name, config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    if config["transformer_additional_kwargs"].get("transformer_combination_type", "single") == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.model_name, config["transformer_additional_kwargs"].get("transformer_high_noise_model_subpath", "transformer")),
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None

    transformer_state_dict = load_state_dict(transformer_path)
    if transformer_state_dict is not None:
        print(f"From checkpoint: {transformer_path}")
        missing, unexpected = transformer.load_state_dict(transformer_state_dict, strict=False)
        print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    chosen_autoencoder = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = chosen_autoencoder.from_pretrained(
        os.path.join(args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    vae_state_dict = load_state_dict(args.vae_path)
    if vae_state_dict is not None:
        print(f"From checkpoint: {args.vae_path}")
        missing, unexpected = vae.load_state_dict(vae_state_dict, strict=False)
        print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    chosen_scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    if args.sampler_name in {"Flow_Unipc", "Flow_DPM++"}:
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = chosen_scheduler(**filter_kwargs(chosen_scheduler, OmegaConf.to_container(config["scheduler_kwargs"])))

    pipeline = Wan2_2TI2VPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    if args.ulysses_degree > 1 or args.ring_degree > 1:
        from functools import partial

        transformer.enable_multi_gpus_inference()
        if transformer_2 is not None:
            transformer_2.enable_multi_gpus_inference()
        if args.fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            if transformer_2 is not None:
                pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
            print("Add FSDP DIT")
        if args.fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print("Add FSDP TEXT ENCODER")

    if args.compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        if transformer_2 is not None:
            for i in range(len(pipeline.transformer_2.blocks)):
                pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
        print("Add Compile")

    if args.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation"], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    coefficients = get_teacache_coefficients(args.model_name) if args.enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {args.teacache_threshold} and skip the first {args.num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients,
            args.num_inference_steps,
            args.teacache_threshold,
            num_skip_start_steps=args.num_skip_start_steps,
            offload=args.teacache_offload,
        )
        if transformer_2 is not None:
            pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

    if args.cfg_skip_ratio is not None:
        print(f"Enable cfg_skip_ratio {args.cfg_skip_ratio}.")
        pipeline.transformer.enable_cfg_skip(args.cfg_skip_ratio, args.num_inference_steps)
        if transformer_2 is not None:
            pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

    if args.lora_path is not None:
        pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=weight_dtype)
        if transformer_2 is not None:
            pipeline = merge_lora(
                pipeline,
                args.lora_path,
                args.lora_high_weight,
                device=device,
                dtype=weight_dtype,
                sub_transformer_name="transformer_2",
            )

    return pipeline, boundary, weight_dtype, device


def build_generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def infer_t2v(pipeline, args, boundary, prompt, device, seed):
    generator = build_generator(device, seed)
    with torch.no_grad():
        sample = pipeline(
            prompt,
            num_frames=args.video_length,
            negative_prompt=args.negative_prompt,
            height=args.sample_height,
            width=args.sample_width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            boundary=boundary,
            video=None,
            mask_video=None,
            shift=args.shift,
        ).videos
    return sample

def infer_ti2v(pipeline, args, boundary, prompt, image_path, device, seed):
    with Image.open(image_path) as img:
        image_width, image_height = img.size
    sample_width, sample_height = calculate_dimensions(image_width * image_height, image_width / image_height)
    generator = build_generator(device, seed)
    with torch.no_grad():
        input_video, input_video_mask, _ = get_image_to_video_latent(
            str(image_path),
            None,
            video_length=args.video_length,
            sample_size=[sample_height, sample_width],
        )
        sample = pipeline(
            prompt,
            num_frames=args.video_length,
            negative_prompt=args.negative_prompt,
            height=sample_height,
            width=sample_width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            boundary=boundary,
            video=input_video,
            mask_video=input_video_mask,
            shift=args.shift,
        ).videos
    return sample


def save_sample(sample, output_path, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_videos_grid(sample, str(output_path), fps=fps)


def main():
    args = build_parser().parse_args()
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    prompt_map = load_prompt_map(args.meta_json)

    pipeline, boundary, _, device = create_pipeline(args)

    image_paths = sorted(image_root.rglob("*.jpg"))
    print(f"Found {len(image_paths)} jpg files under {image_root}")

    t2v_dir = output_root / "t2v"
    ti2v_dir = output_root / "ti2v"
    t2v_dir.mkdir(parents=True, exist_ok=True)
    ti2v_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for index, image_path in enumerate(image_paths):
        rel_image_path = image_path.relative_to(image_root)
        rel_mp4_path = rel_image_path.with_suffix(".mp4")
        prompt = resolve_prompt(prompt_map, rel_mp4_path)
        if prompt is None:
            print(f"[SKIP] prompt not found for {rel_mp4_path.as_posix()}")
            skipped += 1
            continue

        seed = args.seed + index
        prompt_name = prompt_to_filename(prompt)
        t2v_output_path = t2v_dir / f"{prompt_name}.mp4"
        ti2v_output_path = ti2v_dir / f"{prompt_name}.mp4"

        print(f"[{index + 1}/{len(image_paths)}] {rel_mp4_path.as_posix()}")
        # t2v_sample = infer_t2v(pipeline, args, boundary, prompt, device, seed)
        # save_sample(t2v_sample, t2v_output_path, args.fps)

        ti2v_sample = infer_ti2v(pipeline, args, boundary, prompt, image_path, device, seed)
        save_sample(ti2v_sample, ti2v_output_path, args.fps)

        processed += 1

    print(f"processed={processed}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()