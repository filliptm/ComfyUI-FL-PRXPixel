"""Standalone PRXPixel smoke test.

Run from the ComfyUI root:
    venv\\Scripts\\python.exe custom_nodes\\ComfyUI-FL-PRXPixel\\tests\\prxpixel_standalone_smoke.py
"""

from pathlib import Path

import torch
from diffusers import PRXPixelPipeline
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "output" / "prxpixel_standalone_smoke.png"


def main():
    pipe = PRXPixelPipeline.from_pretrained(
        "Photoroom/prxpixel-t2i",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(12345)
    image = pipe(
        "A simple blue glass sphere on a matte white table, studio lighting",
        num_inference_steps=2,
        guidance_scale=3.0,
        generator=generator,
    ).images[0]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)

    checked = Image.open(OUTPUT).convert("RGB")
    stat = ImageStat.Stat(checked)
    extrema = checked.getextrema()
    if checked.width <= 0 or checked.height <= 0:
        raise RuntimeError("Generated image has invalid dimensions.")
    if all(lo == hi for lo, hi in extrema):
        raise RuntimeError("Generated image is flat; all channels have a single value.")

    print(f"saved={OUTPUT}")
    print(f"size={checked.width}x{checked.height}")
    print(f"mean={[round(v, 2) for v in stat.mean]}")


if __name__ == "__main__":
    main()

