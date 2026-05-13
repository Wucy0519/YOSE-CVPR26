import os
import copy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import logging as begin_logging
import math
import shutil
import gc
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from einops import rearrange
from PIL import Image
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.nn.functional as F

# Hugging Face imports
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from deepspeed import DeepSpeedEngine

# Diffusers imports
import diffusers
from diffusers import WanImageToVideoPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, cast_training_params, free_memory
from diffusers.utils import export_to_video, convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.models import AutoencoderKLWan

# dataset and model
from models.dataset import TrainingDataset as VideoPaintDataset
from models.wcy_kit import *
from models.transformer_yose import *
from models.pipeline import *

logger = get_logger(__name__)

def get_gradient_norm(parameters):
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm**0.5
    return norm

def concatenate_images_horizontally(images1, images2, images3, output_type="np"):
    '''
    Concatenate three lists of images horizontally.
    Args:
        images1: List[Image.Image] or List[np.ndarray]
        images2: List[Image.Image] or List[np.ndarray]
        images3: List[Image.Image] or List[np.ndarray]
    Returns:
        List[Image.Image] or List[np.ndarray]
    '''
    concatenated_images = []
    for img1, img2, img3 in zip(images1, images2, images3):
        # Convert images to numpy arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        arr3 = np.array(img3)

        # Concatenate arrays horizontally
        concatenated_img = np.concatenate((arr1, arr2, arr3), axis=1)

        # Convert back to PIL Image
        if output_type == "pil":
            concatenated_img = Image.fromarray(concatenated_img)
        elif output_type == "np":
            pass
        else:
            raise NotImplementedError
        concatenated_images.append(concatenated_img)
    return concatenated_images

def log_validation(
    ori_video,
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    validating_step=0,
):
    logger.info(f"Running validation...")

    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    pipeline_args["prompt"] = pipeline_args["prompt"][0]
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, num_inference_steps=12, generator=generator, output_type="np").frames[0]
        # Concatenate images horizontally
        original_video = (ori_video[0].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
        masked_video = (pipeline_args['masked_video'][0].permute(1, 2, 3, 0).cpu().numpy() + 1) / 2
        video_ = concatenate_images_horizontally(
            original_video, 
            masked_video, 
            video
        )
        videos.append(video_)

    if accelerator.is_main_process:
        phase_name = f"validation_{epoch + 1:03d}_{validating_step}"
        for i, video in enumerate(videos):
            prompt = (
                pipeline_args["prompt"][:25]
                .replace(" ", "_")
                .replace(" ", "_")
                .replace("'", "_")
                .replace('"', "_")
                .replace("/", "_")
            )
            filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
            export_to_video(video, filename, fps=8)

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    return videos


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def _get_t5_prompt_embeds(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if text_input_ids is None:
        batch_size = len(prompt)
    else:
        batch_size = text_input_ids.shape[0]


    if tokenizer is not None and text_input_ids is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='/path/to/your/train/dataset',  
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames used in one video clip.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Interval between sampled frames.",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=81,
        help="Maximum number of frames to load from each video.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of input frames.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Width of input frames.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=6,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )


    # Model path parameters
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='/path/to/your/pretrain_model',
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Optional path to pretrained LoRA weights.",
    )

    # Training parameters
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        help="Training architecture, e.g., lora.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4, # old:4
        help=("The scaling factor to scale LoRA weight update. The actual scaling factor is `lora_alpha / rank`"),
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Target modules to apply LoRA to.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        help="LoRA initialization method.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,  # todo
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validating_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--use_gradient_checkpointing", # todo
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",  # todo
        default=False,
        action='store_true',
        help="Enable offload for checkpointing.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Validation
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="One or more prompt(s) that is used during validation to verify that the model is learning. Multiple validation prompts should be separated by the '--validation_prompt_seperator' string.",
    )
    parser.add_argument(
        "--validation_prompt_separator",
        type=str,
        default=":::",
        help="String that separates multiple validation prompts",
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Logging and output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train/",
        help="Output directory to save checkpoints/logs.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs/",
        help="Output directory to save checkpoints/logs.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--runs_name", type=str, default=None, help="Runs name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--meta_file_path",
        type=str,
        default=None,
        help="The path to meta data.",
    )
    parser.add_argument(
        "--val_meta_file_path",
        type=str,
        default=None,
        help="The path to meta data.",
    )
    parser.add_argument(
        "--corrupt_file_path",
        type=str,
        default=None,
        help="The path to corrupt data.",
    )
    parser.add_argument(
        "--random_mask",
        action="store_true",
        help=(
            "Training with random mask"
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_text_seq_length",
        type=int,
        default=226,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Whether or not to use the pinned memory setting in pytorch dataloader.",
    )
    return parser.parse_args()



def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    begin_logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    scheduler_copy = copy.deepcopy(scheduler)
    transformer = WanTransformer3DBlockModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer"
    )

    yose_model = YOSE()

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    yose_model.requires_grad_(True)

    weight_dtype = torch.bfloat16  # todo
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    yose_model.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            transformer.enable_xformers_memory_efficient_attention()
            yose_model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.use_gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        yose_model.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, DeepSpeedEngine):
                    model = model.module
                unwrapped = unwrap_model(model)
                class_name = unwrapped.__class__.__name__
                if hasattr(unwrapped, "yose_model"):
                    yose_model = unwrapped.yose_model
                    yose_model.save_pretrained(os.path.join(output_dir, "yose_model"))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                if weights:    
                    weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None
        yose_model_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(unwrap_model(model), type(unwrap_model(yose_model))):
                yose_model_ = unwrap_model(model).from_pretrained(input_dir, subfolder="yose_model")
                unwrap_model(model).load_state_dict(yose_model_.state_dict())

        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    # False False
    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    logger.info(f"use_deepspeed_optimizer: {use_deepspeed_optimizer}, use_deepspeed_scheduler: {use_deepspeed_scheduler}")

    # Optimization parameters
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer_lora_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    transformer_parameters = list(filter(lambda p: p.requires_grad, yose_model.parameters()))
    transformer_parameters_with_lr = {"params": transformer_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_lora_parameters_with_lr, transformer_parameters_with_lr]
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    train_dataset = VideoPaintDataset(
        video_path=args.dataset_path,
        max_num_frames=81,
        frame_interval=1,
        num_frames=17,
        height=args.height,
        width=args.width
    )
    validation_dataset = VideoPaintDataset(
        video_path=args.dataset_path,
        max_num_frames=81,
        frame_interval=1,
        num_frames=17,
        height=args.height,
        width=args.width
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.dataloader_num_workers,
        persistent_workers=False
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=args.pin_memory,
        num_workers=args.dataloader_num_workers,
        persistent_workers=False
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler
        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    class CombinedModel(nn.Module):
        def __init__(self, yose_model, transformer):
            super().__init__()
            self.yose_model = yose_model
            self.transformer = transformer

        def begin(self, x):
            mask_condition, scale_shift = self.yose_model(x)
            return mask_condition, scale_shift

        def forward(self, latent_model_input, timesteps, prompt_embeds, mask_condition, masks):
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=None,
                return_dict=False,
                add_condition=None,
                masks=masks,
            )[0]
            return noise_pred

    combined_model = CombinedModel(yose_model, transformer)

    combined_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        combined_model, optimizer, train_dataloader, lr_scheduler
    )  # todo 

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        def sanitize_config(config):
            for key, value in list(config.items()): 
                if isinstance(value, (list, dict, set)): 
                    config[key] = str(value)
                elif value is None:
                    config[key] = "None"
            return config
        tracker_cfg = sanitize_config(dict(vars(args)))
        print(tracker_cfg)
        tracker_name = args.tracker_name or "VideoInpainting"
        accelerator.init_trackers(
            project_name=tracker_name, 
            config=tracker_cfg
        )

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(accelerator.device, weight_dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        accelerator.device, weight_dtype
    )
    vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample) # 4
    vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample) # 8

    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, args.num_train_epochs):
        combined_model.train()
        for step, batch in enumerate(train_dataloader):
            skip_batch = torch.tensor([0], device=accelerator.device)
            models_to_accumulate = [combined_model]

            batch_size, channel, num_frames, height, width = batch["video"].shape
            text, video, mask, masked_video = batch["text"], batch["video"].to(accelerator.device), batch["mask"].to(accelerator.device), batch["masked_video"].to(accelerator.device) 

            try:
                with accelerator.accumulate(models_to_accumulate):

                    with torch.no_grad():
                        video_latent = vae.encode(video.to(dtype=weight_dtype)).latent_dist.mode()
                        video_latent = (video_latent - latents_mean) * latents_std
                        video_latent = video_latent.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                        big_mask, small_mask = get_mask_index_wcy(mask) 
                        masked_video, mask = cv2_inpaint_inf(masked_video, mask[:, :1], small_mask)

                        # masked video [2, 16, 5, 60, 104]
                        masked_video = vae.encode(masked_video.to(dtype=weight_dtype)).latent_dist.mode()
                        masked_video = (masked_video - latents_mean) * latents_std
                        masked_video = masked_video.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                        masks_latents = vae.encode(mask.to(dtype=weight_dtype)*2.-1.).latent_dist.mode()
                        masks_latents = (masks_latents - latents_mean) * latents_std
                        masks_latents = masks_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                    noise = torch.randn_like(video_latent).to(device=accelerator.device, dtype=weight_dtype)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="logit_normal",
                        batch_size=batch_size,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29,
                    )
                    indices = (u * scheduler_copy.config.num_train_timesteps).long()
                    timesteps = scheduler_copy.timesteps[indices].to(device=accelerator.device)
                    sigmas = get_sigmas(timesteps, n_dim=video_latent.ndim, dtype=weight_dtype)
                    noisy_video_latents = (1.0 - sigmas) * video_latent + sigmas * noise
                    latent_model_input = noisy_video_latents

                    mask_condition, scale_shift = combined_model.begin(None)
                    wcy_cond = None

                    def reve2big(mask):
                        b, c, n, h, w = mask.shape
                        mask = F.interpolate(mask.reshape(b*c*n, 1, h, w), size=(h*2, w*2), mode='nearest')
                        return mask.reshape(b, c, n, h*2, w*2)

                    sm_d1, sm_d0 = dilation_dev01(small_mask, 3)
                    blend_sm = (sm_d1 + small_mask)/2. 
                    loss_mask = reve2big(sm_d1)

                    b, c, n, h, w = big_mask.shape
                    wcy_cond = [[sm_d1, blend_sm], small_mask, torch.cat([masked_video, masked_video, masks_latents], dim=1), scale_shift]

                    #big_mask     :[2,  1, 5, 60, 104]
                    #small_mask   :[2,  1, 5, 30, 52 ]
                    #masked_video :[2, 16, 5, 60, 104]
                    latent_model_input = torch.cat([latent_model_input, masked_video, masks_latents], dim=1)

                    noise_pred = combined_model(
                        latent_model_input=latent_model_input,
                        timesteps=timesteps,
                        prompt_embeds=None,
                        mask_condition=mask_condition,
                        masks=wcy_cond)
                    
                    
                    gt_latent = noise-video_latent #  noise_pred_gt

                    weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)
                    loss_mask_main = loss_mask

                    loss_main = ((noise_pred*loss_mask_main).float()-(gt_latent*loss_mask_main).float())**2 # [2,  16, 5, 60, 104]
                    loss_main = torch.mean(loss_main, dim=1, keepdim=True) #[2, 1, 5, 60, 104]
                    loss_main = loss_main.mean()/loss_mask_main.mean()
                    loss_main = loss_main * weighting.float()
                    loss = loss_main.mean()

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(combined_model.parameters(), args.max_grad_norm)
                        
                    if accelerator.state.deepspeed_plugin is None:
                        optimizer.step()
                        optimizer.zero_grad()
                    lr_scheduler.step()

            except Exception as e:
                print(">>>>>>>>>>>> Error data index : ", batch["index"])
                print("Error data path : ", batch["wcy"])
                print(f"Error on GPU {accelerator.process_index}: {str(e)}")
                traceback.print_exc()
                skip_batch.fill_(1)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # accelerator.reduce will block all GPU for collective op
            all_skip = accelerator.reduce(skip_batch, reduction="max").item()
            if all_skip:
                print(f"[Step {step}] Skipping batch due to error.")
                accelerator.wait_for_everyone()
                continue

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                torch.cuda.empty_cache()
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    try:
                                        shutil.rmtree(removing_checkpoint)
                                    except:
                                        pass

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process and ((global_step % args.validating_steps == 0) or (global_step == 1)):
                    unwrapped_model = accelerator.unwrap_model(combined_model)
                    pipe = WanPaintBlockPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=unwrapped_model.transformer,
                        YOSE=unwrapped_model.yose_model,
                        vae=unwrap_model(vae),
                        scheduler=scheduler,
                    )
                    vali_num = 0
                    for step, batch in enumerate(validation_dataloader):
                        pipeline_args = {
                            "guidance_scale": args.guidance_scale,
                            "height": args.height,
                            "width": args.width,
                            "prompt": batch["text"],
                            "mask": batch["mask"],
                            "masked_video": batch["masked_video"],
                            "num_frames": np.array(batch['masked_video'][0]).shape[0],
                            "use_masks": False,
                        }
                        validation_outputs = log_validation(
                            ori_video=batch["video"],
                            pipe=pipe,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args,
                            epoch=epoch,
                            validating_step=global_step,
                        )
                        del batch
                        torch.cuda.empty_cache()
                        gc.collect()
                        vali_num += 1
                        if vali_num >= 2:
                            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)