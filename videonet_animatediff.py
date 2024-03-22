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
from magicanimate.models.unet_controlnet import UNet3DConditionModel
torch.manual_seed(17)

import pkg_resources
from omegaconf import OmegaConf
for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
    print("tensorboard_plugins:",entry_point.dist)

import zmq
import time
import json





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


""" Why sd-image-variations-diffusers? 
The concept of sd-image-variations-diffusers appears to differ from normal Stable Diffusion (SD) in the focus on generating variations of an existing image or theme. Here’s how it stands out:

Purpose of Variations: While normal SD primarily generates images from textual descriptions starting from scratch, sd-image-variations-diffusers seems to specialize in creating different versions or slight modifications of an existing image. This can be particularly useful for exploring alternative possibilities, fine-tuning details, or generating multiple iterations of a concept.

Control and Consistency: Generating variations likely involves maintaining certain aspects of the original image constant, such as the overall theme, composition, or key elements, while altering others. This differs from the usual SD process, where each new generation can result in widely different images even with similar text prompts.

Technique and Process: The use of the term “diffusers” suggests a specific approach or technique within the diffusion model framework, perhaps focusing on controlled manipulation of the image generation process. This could involve sophisticated methods to ensure that the variations are coherent and aligned with the original image’s characteristics.

Targeted Creativity: sd-image-variations-diffusers may provide tools for more targeted creativity, allowing artists and users to iterate on a concept or visual idea more precisely. This could be useful in scenarios where the initial concept is clear, but the execution requires experimentation with variations to find the ideal manifestation.

In summary, the difference lies in the specific application and functionality of generating nuanced variations of an image, as opposed to generating entirely new images from text descriptions.
"""






        
if __name__ == '__main__':
    num_frames = 24

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    # construct pipe from imag evariation diffuser
    pipe = StableDiffusionImageVariationPipeline.from_pretrained('/media/2TB/ani/animate-anyone/pretrained_models/sd-image-variations-diffusers', revision="v2.0", vae=vae).to(device)
    
    video_net = VideoNet(pipe.unet, num_frames=num_frames).to("cuda")

  
    # load mm pretrained weights from animatediff
    load_mm(video_net, torch.load('/media/2TB/stable-diffusion-webui/extensions/sd-webui-animatediff/model/v3_sd15_mm.ckpt'))


    # Setup ZMQ context and socket
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)  # PUB socket for publishing messages
    socket.bind('tcp://127.0.0.1:5555')
    # Wait a moment to allow subscribers to connect
    time.sleep(1)

    def notify_forward_hook(module_identifier):
        def hook(module, input, output):
            # Dispatch message only when forward pass occurs
            msg = f"Forward pass through: {module_identifier}"
            socket.send_string(msg)
            print(msg)  # Also print to console for verification
        return hook

    # Register the hook for all modules and pass the module identifier
    for name, module in video_net.named_modules():
        hook = notify_forward_hook(f"{name} ({module.__class__.__name__})")
        module.register_forward_hook(hook)

    # video_net.eval()
    socket.send_string("hello")
        # Dummy dimensions
    N = 4  # batch size
    L = 256  # latent dimension size
    T = 1  # time steps, assuming a single time step for simplicity
    E = 512  # embedding size

    # Sample data generation
    input_latents = torch.rand(N, L)
    t = torch.rand(N, T)
    reference_frame_embeddings = torch.rand(N, E)
    clip_raw_frame_embeddings = torch.rand(N, E)

    # Assuming video_net is your VideoNet model instance
    # Perform a forward pass
    noise_pred = video_net(input_latents, t, reference_frame_embeddings, clip_raw_frame_embeddings, skip_temporal_attn=True)