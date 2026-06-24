"""ComfyUI-FL-PRXPixel: PRXPixel text-to-image nodes for ComfyUI."""

import logging
import os
import sys

logger = logging.getLogger("FL_PRXPixel")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[ComfyUI-FL-PRXPixel] %(message)s"))
    logger.addHandler(handler)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import folder_paths

    models_dir = os.path.join(folder_paths.models_dir, "prxpixel")
    os.makedirs(models_dir, exist_ok=True)
    folder_paths.add_model_folder_path("prxpixel", models_dir)
except Exception as exc:
    logger.warning("Could not register PRXPixel model folder: %s", exc)

try:
    from .nodes import FL_PRXPixel_Generate, FL_PRXPixel_ModelLoader

    NODE_CLASS_MAPPINGS = {
        "FL_PRXPixel_ModelLoader": FL_PRXPixel_ModelLoader,
        "FL_PRXPixel_Generate": FL_PRXPixel_Generate,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "FL_PRXPixel_ModelLoader": "FL PRXPixel Model Loader",
        "FL_PRXPixel_Generate": "FL PRXPixel Text To Image",
    }

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    logger.info("Loaded successfully. Nodes available: Model Loader, Text To Image")
except Exception as exc:
    logger.exception("Failed to load nodes: %s", exc)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

