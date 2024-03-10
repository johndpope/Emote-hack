import argparse
import logging
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from omegaconf import OmegaConf
import numpy as np
from EMOModel import EMOModel
from EMOAnimationPipeline import EMOAnimationPipeline


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def main(cfg):
    accelerator = Accelerator(log_with="mlflow", project_dir="./mlruns")
    logging.basicConfig(level=logging.INFO)

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    save_dir = f"{cfg.output_dir}/{cfg.exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to("cuda")
    image_enc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to("cuda")
    

    reference_unet_config = {
        "sample_size": 256,                # The size of the input samples
        "in_channels": 3,                  # The number of input channels (e.g., for RGB images this is 3)
        "out_channels": 3,                 # The number of output channels
        "down_block_types": ("DownBlock2D",) * 4,   # A tuple defining the types of blocks in the downsampling path
        "up_block_types": ("UpBlock2D",) * 4,       # A tuple defining the types of blocks in the upsampling path
        # ... Additional configurations
    }

    denoising_unet_config = {
        "sample_size": 256,                # The size of the input samples
        "in_channels": 3,                  # The number of input channels (e.g., for RGB images this is 3)
        "out_channels": 3,                 # The number of output channels
        "down_block_types": ("DownBlock2D", "AttnDownBlock2D") * 2,   # A tuple defining the types of blocks, including attention blocks
        "up_block_types": ("UpBlock2D", "AttnUpBlock2D") * 2,         # A tuple defining the types of blocks, including attention blocks
        # ... Additional configurations
    }

    # Configuration for the EMOModel
    emo_config = {
        "num_speed_buckets": 10,
        "speed_embedding_dim": 64,
        "reference_unet_config": reference_unet_config,
        "denoising_unet_config": denoising_unet_config,
        # ... Additional model configurations
    }
    # emo_config = {
    #     "reference_unet_config": cfg.reference_unet_config,
    #     "denoising_unet_config": cfg.denoising_unet_config,
    #     "num_speed_buckets": cfg.num_speed_buckets,
    #     "speed_embedding_dim": cfg.speed_embedding_dim,
    # }
    
    emo_model = EMOModel(vae, image_enc, emo_config).to("cuda")
    optimizer = torch.optim.AdamW(emo_model.parameters(), lr=cfg.solver.learning_rate)

    # Accelerator preparation
    emo_model, optimizer = accelerator.prepare(emo_model, optimizer)

    # Initialize EMOAnimationPipeline
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    emo_pipeline = EMOAnimationPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=emo_model.reference_unet,
        denoising_unet=emo_model.denoising_unet,
        face_locator=emo_model.face_locator,
        speed_encoder=emo_model.speed_embedding,
        scheduler=scheduler,
    ).to("cuda")

    # Training loop (simplified)
    for epoch in range(cfg.num_epochs):
        for batch in cfg.train_dataloader:
            # Simplified training step
            optimizer.zero_grad()
            output = emo_model(batch['noisy_latents'], batch['timesteps'], batch['ref_image'],
                               batch['motion_frames'], batch['audio_features'], batch['head_rotation_speeds'])
           
            loss = F.mse_loss(output, batch['target'])
            accelerator.backward(loss)
            optimizer.step()

            # Calculate signal-to-noise ratio using EMOAnimationPipeline
            with torch.no_grad():
                generated_video = emo_pipeline(
                    prompt=batch['prompt'],
                    source_image=batch['ref_image'],
                    audio=batch['audio_path'],
                    head_rotation_speeds=batch['head_rotation_speeds'],
                    num_inference_steps=50,
                    output_type="numpy",
                ).videos
                
                # Calculate signal-to-noise ratio
                signal = np.mean(generated_video)
                noise = np.std(generated_video)
                snr = signal / noise
                
                # Log the signal-to-noise ratio
                accelerator.log({"snr": snr}, step=epoch)

    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/stage1.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)  # assuming YAML configuration
    main(config)