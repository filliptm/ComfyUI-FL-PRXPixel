"""PRXPixel pipeline loading and caching."""

import gc
import inspect
import logging
import os

import torch

from .model_info import MODEL_CONFIGS

logger = logging.getLogger("FL_PRXPixel")

LOADED_PIPELINES = {}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class PRXPixelLoader:
    @staticmethod
    def load_pipeline(
        model_name,
        device="cuda",
        dtype="bfloat16",
        offload=False,
        force_reload=False,
    ):
        cache_key = f"{model_name}_{device}_{dtype}_{offload}"
        if not force_reload and cache_key in LOADED_PIPELINES:
            logger.info("Using cached pipeline: %s", model_name)
            return LOADED_PIPELINES[cache_key]

        if force_reload:
            PRXPixelLoader.clear_cache()

        try:
            from diffusers import PRXPixelPipeline as PipelineClass
        except ImportError:
            from diffusers import DiffusionPipeline as PipelineClass

        config = MODEL_CONFIGS.get(model_name, {"repo_id": model_name})
        repo_id = config["repo_id"]
        torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)

        kwargs = {"torch_dtype": torch_dtype}
        local_path = PRXPixelLoader._local_model_path(model_name)
        source = local_path or repo_id

        logger.info("Loading PRXPixel pipeline from %s", source)
        pipe = PipelineClass.from_pretrained(source, **kwargs)

        if offload and device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            actual_device = "cuda"
        else:
            pipe.to(device)
            actual_device = device

        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)

        result = {
            "pipeline": pipe,
            "device": actual_device,
            "dtype": torch_dtype,
            "dtype_name": dtype,
            "model_name": model_name,
            "repo_id": repo_id,
            "call_parameters": set(inspect.signature(pipe.__call__).parameters.keys()),
        }
        LOADED_PIPELINES[cache_key] = result
        return result

    @staticmethod
    def clear_cache():
        LOADED_PIPELINES.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _local_model_path(model_name):
        try:
            import folder_paths

            local = folder_paths.get_full_path("prxpixel", model_name)
            if local and os.path.isdir(local):
                return local
        except Exception:
            pass
        return None
