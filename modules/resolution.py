"""Resolution and patch-count helpers for PRXPixel."""


def get_internal_resolution(pipe, width, height, use_resolution_binning=True):
    """Return the internal height and width used by the PRXPixel denoising loop."""
    internal_height = int(height)
    internal_width = int(width)

    if not use_resolution_binning:
        return internal_height, internal_width

    image_processor = getattr(pipe, "image_processor", None)
    default_sample_size = getattr(pipe, "default_sample_size", None)
    if image_processor is None or default_sample_size is None:
        return internal_height, internal_width

    try:
        from diffusers.pipelines.prx.pipeline_prx_pixel import ASPECT_RATIO_BINS

        bins = ASPECT_RATIO_BINS.get(default_sample_size)
        if bins is None:
            return internal_height, internal_width
        return image_processor.classify_height_width_bin(internal_height, internal_width, ratios=bins)
    except Exception:
        return internal_height, internal_width


def get_patch_info(pipe, width, height, batch_size=1, guidance_scale=1.0, use_resolution_binning=True):
    """Calculate internal resolution, patch count, and CFG-expanded patch tokens."""
    internal_height, internal_width = get_internal_resolution(
        pipe,
        width=width,
        height=height,
        use_resolution_binning=use_resolution_binning,
    )
    transformer = getattr(pipe, "transformer", None)
    config = getattr(transformer, "config", None)
    patch_size = int(getattr(config, "patch_size", 1) or 1)
    patch_rows = internal_height // patch_size
    patch_cols = internal_width // patch_size
    patch_count = patch_rows * patch_cols
    cfg_multiplier = 2 if guidance_scale > 1.0 else 1
    effective_patch_tokens = patch_count * int(batch_size) * cfg_multiplier
    return {
        "requested_width": int(width),
        "requested_height": int(height),
        "internal_width": int(internal_width),
        "internal_height": int(internal_height),
        "patch_size": patch_size,
        "patch_rows": patch_rows,
        "patch_cols": patch_cols,
        "patch_count": patch_count,
        "batch_size": int(batch_size),
        "cfg_multiplier": cfg_multiplier,
        "effective_patch_tokens": effective_patch_tokens,
        "use_resolution_binning": bool(use_resolution_binning),
    }


def format_patch_info(info):
    return (
        f"requested={info['requested_width']}x{info['requested_height']}; "
        f"internal={info['internal_width']}x{info['internal_height']}; "
        f"patch_size={info['patch_size']}; "
        f"patch_grid={info['patch_cols']}x{info['patch_rows']}; "
        f"patch_count={info['patch_count']}; "
        f"effective_patch_tokens={info['effective_patch_tokens']} "
        f"(batch={info['batch_size']}, cfg_multiplier={info['cfg_multiplier']}); "
        f"resolution_binning={info['use_resolution_binning']}"
    )

