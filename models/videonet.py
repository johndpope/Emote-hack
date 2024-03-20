import copy
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import UNet2DConditionModel, Transformer2DModel
from einops import rearrange
from xformers.ops import memory_efficient_attention

from .motionmodule import get_motion_module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SpatialAttentionModule is a spatial attention module between reference and input
class SpatialAttentionModule(nn.Module):
    def __init__(self, num_inp_channels: int, embed_dim: int = 40, num_heads: int = 8) -> None:
        super(SpatialAttentionModule, self).__init__()

        self.num_inp_channels = num_inp_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # create input projection layers
        self.norm_in = nn.GroupNorm(num_groups=32, num_channels=num_inp_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(num_inp_channels, num_inp_channels, kernel_size=1, stride=1, padding=0)

        # create multiheaded attention module
        self.to_q = nn.Linear(num_inp_channels, embed_dim)
        self.to_k = nn.Linear(num_inp_channels, embed_dim)
        self.to_v = nn.Linear(num_inp_channels, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # create output projection layer
        self.proj_out = nn.Conv2d(num_inp_channels, num_inp_channels, kernel_size=1, stride=1, padding=0)

    # forward passes the activation through a spatial attention module
    def forward(self, x, reference_tensor):
        # expand and concat x with reference embedding where x is [b*t,c,h,w]
        orig_w = x.shape[3]
        concat = torch.cat((x, reference_tensor), axis=3)
        h, w = concat.shape[2], concat.shape[3]

        # pass data through input projections
        proj_x = self.norm_in(concat)
        proj_x = self.proj_in(proj_x)

        # re-arrange data from (b*t,c,h,w) to correct groupings to [b*t,w*h,c]
        grouped_x = rearrange(proj_x, 'bt c h w -> bt (h w) c')
        reshaped_x = rearrange(x, 'bt c h w -> bt (h w) c')

        # compute self-attention on the concatenated data along w dimension
        q, k, v = self.to_q(reshaped_x), self.to_k(grouped_x), self.to_v(grouped_x)

        # split embeddings for multi-headed attention
        q = rearrange(q, 'bt (h w) (n d) -> bt (h w) n d', h=x.shape[2], w=x.shape[3], n=self.num_heads)
        k = rearrange(k, 'bt (h w) (n d) -> bt (h w) n d', h=h, w=w, n=self.num_heads)
        v = rearrange(v, 'bt (h w) (n d) -> bt (h w) n d', h=h, w=w, n=self.num_heads)

        # run attention calculation
        attn_out = memory_efficient_attention(q, k, v)
        # reshape from multihead
        attn_out = rearrange(attn_out, 'bt (h w) n d -> bt (h w) (n d)', h=x.shape[2], w=x.shape[3], n=self.num_heads)
        
        norm1_out = self.norm1(attn_out + reshaped_x)
        ffn_out = self.ffn(norm1_out)
        attn_out = self.norm2(norm1_out + ffn_out)

        # re-arrange data from (b*t,w*h,c) to (b*t,c,h,w)
        attn_out = rearrange(attn_out, 'bt (h w) c -> bt c h w', h=x.shape[2], w=x.shape[3])

        # pass output through out projection
        out = self.proj_out(attn_out)

        # return sliced out with x as adding residual before reshape would be the same as adding x
        return out + x


# TemporalAttentionModule is a temporal attention module
class TemporalAttentionModule(nn.Module):
    def __init__(self, num_inp_channels: int, num_frames: int, embed_dim: int = 40, num_heads: int = 8) -> None:
        super(TemporalAttentionModule, self).__init__()

        self.num_inp_channels = num_inp_channels
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # create input projection layers
        self.norm_in = nn.GroupNorm(num_groups=32, num_channels=num_inp_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(num_inp_channels, num_inp_channels, kernel_size=1, stride=1, padding=0)

        # create multiheaded attention module
        self.to_q = nn.Linear(num_inp_channels, embed_dim)
        self.to_k = nn.Linear(num_inp_channels, embed_dim)
        self.to_v = nn.Linear(num_inp_channels, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # create output projection layer
        self.proj_out = nn.Conv2d(num_inp_channels, num_inp_channels, kernel_size=1, stride=1, padding=0)

    # forward performs temporal attention on the input (b*t,c,h,w)
    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        # pass data through input projections
        proj_x = self.norm_in(x)
        proj_x = self.proj_in(proj_x)

        # re-arrange data from (b*t,c,h,w) to correct groupings to (b*t,w*h,c)
        grouped_x = rearrange(x, '(b t) c h w -> (b h w) t c', t=self.num_frames)

        # perform self-attention on the grouped_x
        q, k, v = self.to_q(grouped_x), self.to_k(grouped_x), self.to_v(grouped_x)
        attn_out = memory_efficient_attention(q, k, v)
        norm1_out = self.norm1(attn_out + grouped_x)
        ffn_out = self.ffn(norm1_out)
        attn_out = self.norm2(norm1_out + ffn_out)

        # rearrange out to be back into the grouped batch and timestep format
        attn_out = rearrange(attn_out, '(b h w) t c -> (b t) c h w', t=self.num_frames, h=h, w=w)

        # pass attention output through out projection
        attn_out = self.proj_out(attn_out)

        return attn_out + x


# ReferenceConditionedAttentionBlock is an attention block which performs spatial and temporal attention
class ReferenceConditionedAttentionBlock(nn.Module):
    def __init__(self, cross_attn: Transformer2DModel, num_frames: int, skip_temporal_attn: bool = False):
        super(ReferenceConditionedAttentionBlock, self).__init__()

        # store configurations and submodules
        self.skip_temporal_attn = skip_temporal_attn
        self.num_frames = num_frames
        self.cross_attn = cross_attn

        # extract channel dimension from provided cross_attn and 
        num_channels = cross_attn.config.in_channels
        embed_dim = cross_attn.config.in_channels
        self.sam = SpatialAttentionModule(num_channels, embed_dim=embed_dim)
        self.tam = get_motion_module(num_channels,
                    motion_module_type='Vanilla', 
                    motion_module_kwargs={})

        # store the reference tensor used by this module (this must be updated before the forward pass)
        self.reference_tensor = None

    # update_reference_tensor updates the reference tensor for the module
    def update_reference_tensor(self, reference_tensor: torch.FloatTensor):
        self.reference_tensor = reference_tensor

    # update_num_frames updates the number of frames the temporal attention module is configured for
    def update_num_frames(self, num_frames: int):
        self.num_frames = num_frames

    # forward performs spatial attention, cross attention, and temporal attention
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # begin spatial attention

        # pass concat tensor through spatial attention module along w axis [bt,c,h,w]
        out = self.sam(hidden_states, self.reference_tensor)

        # begin cross attention
        out = self.cross_attn(out, encoder_hidden_states, timestep, added_cond_kwargs, class_labels,
                            cross_attention_kwargs, attention_mask, encoder_attention_mask, return_dict)[0]

        # begin temporal attention
        if self.skip_temporal_attn:
            return (out,)
        
        # reshape data from [bt c h w] to be [b c t h w]
        temporal_input = rearrange(out, '(b t) c h w -> b c t h w', t=self.num_frames)
        
        # pass the data through the temporal attention module
        temporal_output = self.tam(temporal_input, None, None)

        # reshape temporal output back from [b c t h w] to [bt c h w]
        temporal_output = rearrange(temporal_output, 'b c t h w -> (b t) c h w')

        return (temporal_output,)


# VideoNet is a unet initialized from stable diffusion used to denoise video frames
class VideoNet(nn.Module):
    def __init__(self, sd_unet: UNet2DConditionModel, num_frames: int = 24, batch_size: int = 2):
        super(VideoNet, self).__init__()
        self.batch_size = batch_size

        # create a deep copy of the sd_unet
        self.unet = copy.deepcopy(sd_unet)

        # maintain a list of all the new ReferenceConditionedResNets and TemporalAttentionBlocks
        self.ref_cond_attn_blocks: List[ReferenceConditionedAttentionBlock] = []

        # replace attention blocks with ReferenceConditionedAttentionBlock
        down_blocks = self.unet.down_blocks
        mid_block = self.unet.mid_block
        up_blocks = self.unet.up_blocks

        for i in range(len(down_blocks)):
            if hasattr(down_blocks[i], "attentions"):
                attentions = down_blocks[i].attentions
                for j in range(len(attentions)):
                    attentions[j] = ReferenceConditionedAttentionBlock(attentions[j], num_frames)
                    self.ref_cond_attn_blocks.append(attentions[j])

        for i in range(len(mid_block.attentions)):
            mid_block.attentions[i] = ReferenceConditionedAttentionBlock(mid_block.attentions[i], num_frames)
            self.ref_cond_attn_blocks.append(mid_block.attentions[i])
        
        for i in range(len(up_blocks)):
            if hasattr(up_blocks[i], "attentions"):
                attentions = up_blocks[i].attentions
                for j in range(len(attentions)):
                    attentions[j] = ReferenceConditionedAttentionBlock(attentions[j], num_frames)
                    self.ref_cond_attn_blocks.append(attentions[j])

    # update_reference_embeddings updates all the reference embeddings in the unet
    def update_reference_embeddings(self, reference_embeddings):
        if len(reference_embeddings) != len(self.ref_cond_attn_blocks):
            print("[!] WARNING - amount of input reference embeddings does not match number of modules in VideoNet")

        for i in range(len(self.ref_cond_attn_blocks)):
            # update the reference conditioned blocks embedding
            self.ref_cond_attn_blocks[i].update_reference_tensor(reference_embeddings[i])

    # update_num_frames updates all temporal attention block frame number
    def update_num_frames(self, num_frames):
        for i in range(len(self.ref_cond_attn_blocks)):
            # update the number of frames
            self.ref_cond_attn_blocks[i].update_num_frames(num_frames)

    # update_skip_temporal_attn updates all the skip temporal attention attributes
    def update_skip_temporal_attn(self, skip_temporal_attn):
        for i in range(len(self.ref_cond_attn_blocks)):
            # update the skip_temporal_attn attribute
            self.ref_cond_attn_blocks[i].skip_temporal_attn = skip_temporal_attn

    # forward pass just passes pose + conditioning embeddings to unet and returns activations
    def forward(self, intial_noise, timesteps, reference_embeddings, clip_condition_embeddings, skip_temporal_attn=False):
        # update the reference tensors for the ReferenceConditionedResNet modules
        self.update_reference_embeddings(reference_embeddings)

        # update the skip temporal attention attribute
        self.update_skip_temporal_attn(skip_temporal_attn)

        # forward pass the pose + conditioning embeddings through the unet
        return self.unet(
            intial_noise,
            timesteps,
            encoder_hidden_states=clip_condition_embeddings,
        )[0]

