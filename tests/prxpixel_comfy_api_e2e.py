"""ComfyUI API end-to-end test for FL PRXPixel nodes.

Run while ComfyUI is serving on http://127.0.0.1:8188:
    venv\\Scripts\\python.exe custom_nodes\\ComfyUI-FL-PRXPixel\\tests\\prxpixel_comfy_api_e2e.py
"""

import json
import time
import urllib.request
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
SERVER = "http://127.0.0.1:8188"
CLIENT_ID = "fl-prxpixel-e2e"


def request_json(path, payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{SERVER}{path}", data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_nodes():
    object_info = request_json("/object_info", timeout=60)
    required = {"FL_PRXPixel_ModelLoader", "FL_PRXPixel_Generate", "SaveImage"}
    missing = sorted(required.difference(object_info))
    if missing:
        raise RuntimeError(f"Missing ComfyUI nodes: {missing}")


def queue_workflow():
    prompt = {
        "1": {
            "class_type": "FL_PRXPixel_ModelLoader",
            "inputs": {
                "model_name": "Photoroom/prxpixel-t2i",
                "device": "cuda",
                "dtype": "bfloat16",
                "offload": False,
                "force_reload": False,
            },
        },
        "2": {
            "class_type": "FL_PRXPixel_Generate",
            "inputs": {
                "pipeline": ["1", 0],
                "prompt": "A small blue glass sphere on a matte white table, studio lighting",
                "negative_prompt": "",
                "width": 1024,
                "height": 1024,
                "steps": 2,
                "guidance_scale": 3.0,
                "seed": 24680,
                "batch_size": 1,
                "show_preview": True,
                "preview_every": 1,
                "use_resolution_binning": True,
                "use_patch_grid": False,
                "patch_columns": 64,
                "patch_rows": 64,
            },
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["2", 0],
                "filename_prefix": "prxpixel_comfy_e2e",
            },
        },
    }
    return request_json("/prompt", {"prompt": prompt, "client_id": CLIENT_ID}, timeout=30)["prompt_id"]


def wait_for_completion(prompt_id, timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = request_json(f"/history/{prompt_id}", timeout=30)
        entry = history.get(prompt_id)
        if entry and "outputs" in entry:
            return entry
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def validate_output(entry):
    outputs = entry.get("outputs", {})
    image_infos = []
    for node_output in outputs.values():
        image_infos.extend(node_output.get("images", []))
    if not image_infos:
        raise RuntimeError("No output images reported by ComfyUI history.")

    info = image_infos[0]
    path = ROOT / "output" / info["filename"]
    if info.get("subfolder"):
        path = ROOT / "output" / info["subfolder"] / info["filename"]
    image = Image.open(path).convert("RGB")
    extrema = image.getextrema()
    if image.width <= 0 or image.height <= 0:
        raise RuntimeError(f"Invalid image dimensions: {image.size}")
    if all(lo == hi for lo, hi in extrema):
        raise RuntimeError("Output image is flat; all channels have a single value.")
    print(f"prompt_id={entry.get('prompt_id', 'unknown')}")
    print(f"output={path}")
    print(f"size={image.width}x{image.height}")


def main():
    validate_nodes()
    prompt_id = queue_workflow()
    entry = wait_for_completion(prompt_id)
    entry["prompt_id"] = prompt_id
    validate_output(entry)


if __name__ == "__main__":
    main()
