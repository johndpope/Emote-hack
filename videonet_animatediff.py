import time
from os.path import join

import torch
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from diffusers.schedulers import PNDMScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPVisionModel, CLIPImageProcessor
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import StableDiffusionImageVariationPipeline, StableDiffusionPipeline
import copy
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import UNet2DConditionModel, Transformer2DModel
from einops import rearrange
from xformers.ops import memory_efficient_attention
from models.motionmodule import get_motion_module
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from models.videonet import VideoNet

torch.manual_seed(17)

import pkg_resources

for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
    print("tensorboard_plugins:",entry_point.dist)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load_mm loads a motion module into video net
def load_mm(video_net: VideoNet, mm_state_dict):
    refactored_mm_state_dict = {}
    for key in mm_state_dict:
        key_split = key.split('.')
        
        # modify the key split to have the correct arguments (except first unet)
        key_split[2] = 'attentions'
        key_split.insert(4, 'tam')
        new_key = '.'.join(key_split)
        refactored_mm_state_dict[new_key] = mm_state_dict[key]

    # load the modified weights into video_net
    _, unexpected = video_net.unet.load_state_dict(refactored_mm_state_dict, strict=False)

    return


        
if __name__ == '__main__':
    num_frames = 8

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    # construct pipe from imag evariation diffuser
    pipe = StableDiffusionImageVariationPipeline.from_pretrained('/media/2TB/ani/animate-anyone/pretrained_models/sd-image-variations-diffusers', revision="v2.0", vae=vae).to(device)
    
    video_net = VideoNet(pipe.unet, num_frames=num_frames).to("cuda")

  
    # load mm pretrained weights from animatediff
    load_mm(video_net, torch.load('/media/2TB/stable-diffusion-webui/extensions/sd-webui-animatediff/model/v3_sd15_mm.ckpt'))

    # Step 2: Initialize the TensorBoard SummaryWriter
    writer = SummaryWriter('runs/videonet_experiment')

    # Assuming you have already loaded your model as `video_net`
    # Step 3: Add model graph to TensorBoard
    # Note: You may need to pass a sample input to `add_graph` depending on your model structure
    # Here, `initial_noise` is a sample input tensor
    # Get the correct number of latent dimensions from the model's configuration
    # Get the correct number of latent dimensions from the model's configuration
    num_channels_latent = 4  # This should be verified from the model's configuration

    # Initial noise tensor should match the latent dimensions and the model's expected input size
    initial_noise = torch.randn(1, num_channels_latent, 512, 512).to(device)

    # Timestep tensor; the value might need to be adjusted based on how the diffusion model processes it
    timesteps = torch.tensor([1]).to("cuda")

    # Assuming reference_embeddings need to match the number of attention blocks in your VideoNet model
    n = len(video_net.ref_cond_attn_blocks)
    reference_embeddings = torch.randn(1, n, num_channels_latent, 512, 512).to(device)

    # Clip condition embeddings should match the latent dimensions and expected size
    clip_condition_embeddings = torch.randn(1, num_channels_latent, 512, 512).to(device)

    # You need to ensure the shapes and types match what your VideoNet model expects
    with torch.no_grad():
        writer.add_graph(video_net, (initial_noise, timesteps, reference_embeddings, clip_condition_embeddings))