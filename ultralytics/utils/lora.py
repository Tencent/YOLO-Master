from ultralytics.utils import LOGGER

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None


def apply_lora(model, args):
    """
    Apply LoRA to the model.

    Args:
        model (nn.Module): The model to apply LoRA to.
        args (SimpleNamespace): The arguments containing LoRA configuration.

    Returns:
        (nn.Module): The model with LoRA applied.
    """
    if not hasattr(args, 'lora_r') or args.lora_r <= 0:
        return model

    if LoraConfig is None:
        LOGGER.warning("peft not found. Cannot apply LoRA. Please install peft: pip install peft")
        return model

    LOGGER.info(f"Applying LoRA with r={args.lora_r}, alpha={getattr(args, 'lora_alpha', 16)}, dropout={getattr(args, 'lora_dropout', 0.1)}")

    target_modules = getattr(args, 'lora_target_modules', None)
    if target_modules is None:
        # Default to 'conv' for YOLO models which typically name their Conv2d layers 'conv' inside Conv blocks.
        # This is a heuristic and might need adjustment.
        target_modules = ['conv']
        LOGGER.info(f"lora_target_modules not specified, defaulting to {target_modules}")

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=getattr(args, 'lora_alpha', 16),
        lora_dropout=getattr(args, 'lora_dropout', 0.1),
        target_modules=target_modules,
        bias="none",
        task_type=None  # Generic task
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def save_lora_adapters(model, save_dir):
    """
    Save LoRA adapters.

    Args:
        model (nn.Module): The model with LoRA adapters.
        save_dir (str | Path): The directory to save the adapters.
    """
    if PeftModel is not None and isinstance(model, PeftModel):
        model.save_pretrained(save_dir)
    elif hasattr(model, 'save_pretrained'):
        model.save_pretrained(save_dir)
    else:
        # If model is wrapped (e.g. DDP), try to access the underlying model
        if hasattr(model, 'module') and (isinstance(model.module, PeftModel) or hasattr(model.module, 'save_pretrained')):
            model.module.save_pretrained(save_dir)
        else:
            LOGGER.warning("Model is not a PeftModel, skipping LoRA adapter save.")
