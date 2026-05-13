import os
import imageio
import ffmpeg
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, UMT5EncoderModel
from peft import PeftModel

from diffusers.utils import export_to_video, convert_unet_state_dict_to_peft
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
import torch.nn.functional as F

from models.dataset import VideoPaintDatasetTest_wcy
from models.transformer_yose import *
from models.pipeline import WanPaintBlockPipeline

wan_path = r"/home/wcy/checkpoint/MiniMax-Remover"
data_path = r"/home/wcy/datas/dataset_yose/video_dataset/davis"
save_path = r"/home/wcy/datas/save/result-yose"

height = 480
width = 832
device = torch.device("cuda")
weight_dtype = torch.bfloat16

num_inference_steps = 12
max_num_frames = 100

random_seed = 2025
vae = AutoencoderKLWan.from_pretrained(
    wan_path, subfolder="vae"
)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    wan_path, subfolder="scheduler"
)
transformer = WanTransformer3DBlockModel.from_pretrained(
    wan_path, subfolder="transformer"
)

yose = YOSE()
yose_param = load_file('.//ckpt//minimax-YOSE.safetensors')
yose.load_state_dict(yose_param)

pipe = WanPaintBlockPipeline(
    transformer=transformer,
    YOSE=yose,
    vae=vae,
    scheduler=scheduler
    ).to(device=device, dtype=weight_dtype)

dataset = VideoPaintDatasetTest_wcy(
    base_path=data_path,
    max_num_frames=81,
    frame_interval=1,
    num_frames=100, # todo
    height=height,
    width=width,
    dilate=True
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

import os
folder_path = save_path
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

print(f"\nTotal samples in dataset: {len(dataset)}")
for i, batch in enumerate(dataloader):
    print(f"\nBatch {i}:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: list of length = {len(value)}, sample = {value[0]}")
        else:
            print(f"  {key}: type = {type(value)}")

    name = os.path.splitext(os.path.basename(batch["path"][0]))[0]

    num_frames = batch['masked_video'].shape[2]
    input_mask = batch['mask']
    input_video = batch['masked_video']

    num_frames = min(max_num_frames, num_frames)
    if num_frames % 4 > 0:
        num_frames = num_frames - num_frames%4 +1
    else:
        num_frames = num_frames - 3
    input_mask = input_mask[:, :, :num_frames]
    input_video = input_video[:, :, :num_frames]

    video = pipe(
        mask=input_mask,
        masked_video=input_video,
        prompt=batch['text'],
        height=height,
        width=width,
        num_frames=input_video.shape[2],
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
        generator=torch.Generator(device=device).manual_seed(random_seed),
    ).frames[0]

    print("save to:", save_path+"//{}.mp4".format(name))
    export_to_video(video, save_path+"//{}.mp4".format(name))