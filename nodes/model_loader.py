"""FL PRXPixel Model Loader node."""

import logging

from comfy.utils import ProgressBar

from ..modules.loader import PRXPixelLoader
from ..modules.model_info import MODEL_CONFIGS, get_available_devices, get_dtype_options

logger = logging.getLogger("FL_PRXPixel")


class FL_PRXPixel_ModelLoader:
    RETURN_TYPES = ("PRXPIXEL_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "FL/PRXPixel"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(MODEL_CONFIGS.keys()), {"default": "Photoroom/prxpixel-t2i"}),
                "device": (get_available_devices(), {"default": get_available_devices()[0]}),
                "dtype": (get_dtype_options(), {"default": "bfloat16"}),
                "offload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "force_reload": ("BOOLEAN", {"default": False}),
            },
        }

    def load_model(self, model_name, device, dtype, offload, force_reload=False):
        pbar = ProgressBar(2)
        pbar.update_absolute(0, 2)
        pipeline = PRXPixelLoader.load_pipeline(
            model_name=model_name,
            device=device,
            dtype=dtype,
            offload=offload,
            force_reload=force_reload,
        )
        pbar.update_absolute(2, 2)
        logger.info("Loaded %s on %s (%s)", model_name, device, dtype)
        return (pipeline,)

