"""Model registry and runtime options for PRXPixel."""

MODEL_CONFIGS = {
    "Photoroom/prxpixel-t2i": {
        "repo_id": "Photoroom/prxpixel-t2i",
        "default_steps": 28,
        "default_guidance": 5.0,
        "native_resolution": 1024,
    },
}


def get_available_devices():
    try:
        import torch

        if torch.cuda.is_available():
            return ["cuda", "cpu"]
    except Exception:
        pass
    return ["cpu"]


def get_dtype_options():
    return ["bfloat16", "float16", "float32"]

