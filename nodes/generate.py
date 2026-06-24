"""FL PRXPixel text-to-image node."""

import logging

import torch
from PIL import Image
from comfy.utils import ProgressBar

from ..modules.image_utils import pil_images_to_comfy
from ..modules.resolution import format_patch_info, get_patch_info

logger = logging.getLogger("FL_PRXPixel")


class FL_PRXPixel_Generate:
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "patch_count", "effective_patch_tokens", "patch_info")
    FUNCTION = "generate"
    CATEGORY = "FL/PRXPixel"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PRXPIXEL_PIPELINE",),
                "prompt": ("STRING", {"default": "A product photo of a red cube on a white background", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "show_preview": ("BOOLEAN", {"default": True}),
                "preview_every": ("INT", {"default": 1, "min": 1, "max": 100}),
                "use_resolution_binning": ("BOOLEAN", {"default": True}),
                "use_patch_grid": ("BOOLEAN", {"default": False}),
                "patch_columns": ("INT", {"default": 64, "min": 16, "max": 256, "step": 1}),
                "patch_rows": ("INT", {"default": 64, "min": 16, "max": 256, "step": 1}),
            },
        }

    def generate(
        self,
        pipeline,
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        guidance_scale,
        seed,
        batch_size,
        use_resolution_binning=True,
        use_patch_grid=False,
        patch_columns=64,
        patch_rows=64,
        show_preview=True,
        preview_every=1,
    ):
        pipe = pipeline["pipeline"]
        device = pipeline["device"]
        call_parameters = pipeline.get("call_parameters", set())

        transformer = getattr(pipe, "transformer", None)
        transformer_config = getattr(transformer, "config", None)
        patch_size = int(getattr(transformer_config, "patch_size", 16) or 16)
        if use_patch_grid:
            width = int(patch_columns) * patch_size
            height = int(patch_rows) * patch_size
            use_resolution_binning = False

        patch_info = get_patch_info(
            pipe,
            width=width,
            height=height,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            use_resolution_binning=use_resolution_binning,
        )
        patch_info_text = format_patch_info(patch_info)

        kwargs = {
            "prompt": [prompt] * batch_size if batch_size > 1 else prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        }
        if negative_prompt and "negative_prompt" in call_parameters:
            kwargs["negative_prompt"] = [negative_prompt] * batch_size if batch_size > 1 else negative_prompt
        if "width" in call_parameters:
            kwargs["width"] = width
        if "height" in call_parameters:
            kwargs["height"] = height
        if "use_resolution_binning" in call_parameters:
            kwargs["use_resolution_binning"] = use_resolution_binning

        generator_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        if "generator" in call_parameters:
            kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(seed)

        pbar = ProgressBar(steps)

        def latents_to_preview(latents):
            preview = latents[:1].detach().float().clamp(-1.0, 1.0)
            preview = ((preview + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
            preview = preview[0].permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(preview, mode="RGB")
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            return image

        def legacy_callback(*args, **_kwargs):
            try:
                step = args[0] if args else _kwargs.get("step", 0)
                pbar.update_absolute(int(step) + 1, steps)
            except Exception:
                pass

        def step_end_callback(_pipe, step, _timestep, callback_kwargs):
            preview_tuple = None
            if show_preview and (int(step) % int(preview_every) == 0 or int(step) == steps - 1):
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    try:
                        preview_tuple = ("JPEG", latents_to_preview(latents), 512)
                    except Exception as exc:
                        logger.warning("Could not create PRXPixel preview for step %s: %s", step, exc)
            pbar.update_absolute(int(step) + 1, steps, preview_tuple)
            return callback_kwargs

        if "callback" in call_parameters:
            kwargs["callback"] = legacy_callback
        elif "callback_on_step_end" in call_parameters:
            kwargs["callback_on_step_end"] = step_end_callback

        logger.info(
            "Generating PRXPixel image: steps=%s guidance=%s seed=%s %s",
            steps,
            guidance_scale,
            seed,
            patch_info_text,
        )
        with torch.inference_mode():
            output = pipe(**kwargs)

        images = pil_images_to_comfy(output.images)
        pbar.update_absolute(steps, steps)
        return (
            images,
            patch_info["patch_count"],
            patch_info["effective_patch_tokens"],
            patch_info_text,
        )
