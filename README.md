

# EMO: Emote Portrait Alive - 
using chatgpt to reverse engineer code from HumanAIGC/EMO white paper. Work in progress - WIP



![Image](https://github.com/johndpope/Emote-hack/assets/289994/0d758a3a-841f-4849-b58c-439dda05c9a7)


Just copy and paste the html from here in chatgpt (custom chat)
https://chat.openai.com/g/g-UzGVIbBpB-diffuser-wizard
https://arxiv.org/html/2402.17485v1


The Moore-AnimateAnyone seems very close to this implementation - it was ripped off from magicanimate
it has training code train_stage_1.py
https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/train_stage_1.py



✅  Training data 36,000 videos / facial videos - 40gb
https://academictorrents.com/details/843b5adb0358124d388c4e9836654c246b988ff4

```shell
magnet:?xt=urn:btih:843b5adb0358124d388c4e9836654c246b988ff4&dn=CelebV-HQ&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=https%3A%2F%2Fipv6.academictorrents.com%2Fannounce.php
```








## UPDATE
rather than splitting up classes- I've collapsed into a single Net.py file to aid in copy and paste into LLM. 
you can  also paste in the architecture diagram from pdf into chatgpt / claude and ask it to fix stuff.





## Chat here about specific components (everyone has read/write access) - understandably there's flaws in this code - together we can fix.
https://docs.google.com/spreadsheets/d/1meRHgcFZ8mxWplvJweAd-5P_tkH0UeD2eT2Ot_k46jA/edit#gid=865829673





## Stage 1: Training the VAE (FramesEncodingVAE) with the Backbone Network and FaceLocator

ORIGINAL PAPER
The first stage is the image pretraining, where the Backbone Network, the ReferenceNet, and the Face Lo-
cator are token into training, in this stage, the Backbone takes a single frame as
input, while ReferenceNet handles a distinct, randomly chosen frame from the
same video clip. Both the Backbone and the ReferenceNet initialize weights from the original SD. 


CLAUDE 3 - OPUS
In this stage, the focus is on training the Variational Autoencoder (VAE) to encode and decode video frames.
The VAE consists of the FramesEncodingVAE class, which combines the encoding of reference and motion frames with additional components like ReferenceNet and SpeedEncoder.
The Backbone Network (Denoising UNet) and FaceLocator are also trained during this stage.
The goal is to learn a compressed representation of the video frames and reconstruct them accurately.
The training process involves feeding reference images, motion frames, and speed values to the VAE and minimizing the reconstruction loss.



## Stage 2: Training the Temporal Modules and Audio Layers
In this stage, the temporal modules and audio layers are introduced into the training process.
The temporal modules are responsible for ensuring smooth transitions and coherence between generated video frames.
The audio layers integrate audio features into the generation process, allowing the model to synchronize the character's movements and expressions with the audio.
During this stage, the model learns to generate video frames that are temporally consistent and aligned with the audio input.
The training data includes video frames, audio features, and corresponding ground truth frames.

## Stage 3: Training the Speed Layers

The final stage focuses on training the speed layers of the model.
The speed layers control the speed and stability of the generated character's motion across video clips.
By training the speed layers, the model learns to generate videos with consistent and controllable character motion.
The training data in this stage includes video frames, audio features, and speed embeddings.
The model is trained to generate video frames that (match the specified speed ?? - this is WRONG - it should just do automatically) and maintain coherence across different clips.


It's important to note that the training stages are progressive, meaning that each stage builds upon the previous one. The weights and learned representations from the previous stages are often used as initializations for the subsequent stages.




https://github.com/johndpope/Emote-hack/blob/main/Net.py

```javascript

-✅ FramesEncodingVAE
  - __init__(input_channels, latent_dim, img_size, reference_net)
  - reparameterize(mu, logvar)
  - forward(reference_image, motion_frames, speed_value)
  - vae_loss(recon_frames, reference_image, motion_frames, reference_mu, reference_logvar, motion_mu, motion_logvar)

- DownsampleBlock
  - __init__(in_channels, out_channels)
  - forward(x)

- UpsampleBlock
  - __init__(in_channels, out_channels)
  - forward(x1, x2)

- ✅ ReferenceNet
  - __init__(vae_model, speed_encoder, config)
  - forward(reference_image, motion_frames, head_rotation_speed)

- ✅ SpeedEncoder
  - __init__(num_speed_buckets, speed_embedding_dim)
  - get_bucket_centers()
  - get_bucket_radii()
  - encode_speed(head_rotation_speed)
  - forward(head_rotation_speeds)

- CrossAttentionLayer
  - __init__(feature_dim)
  - forward(latent_code, audio_features)

- AudioAttentionLayers
  - __init__(feature_dim, num_layers)
  - forward(latent_code, audio_features)

-✅ EMOModel
  - __init__(vae, image_encoder, config)
  - forward(noisy_latents, timesteps, ref_image, motion_frames, audio_features, head_rotation_speeds)

-✅ Wav2VecFeatureExtractor
  - __init__(model_name, device)
  - extract_features_from_mp4(video_path, m, n)
  - extract_features_for_frame(video_path, frame_index, m)

- AudioFeatureModel
  - __init__(input_size, output_size)
  - forward(x)

-✅ FaceLocator
  - __init__()
  - forward(images)

-✅ FaceHelper
  - __init__()
  - __del__()
  - generate_face_region_mask(frame_image, video_id, frame_idx)
  - generate_face_region_mask_np_image(frame_np, video_id, frame_idx, padding)
  - generate_face_region_mask_pil_image(frame_image, video_id, frame_idx)
  - calculate_pose(face2d)
  - draw_axis(img, yaw, pitch, roll, tdx, tdy, size)
  - get_head_pose(image_path)
  - get_head_pose_velocities_at_frame(video_reader, frame_index, n_previous_frames)

- EmoVideoReader
  - __init__(pixel_transform, cond_transform, state)
  - augmentedImageAtFrame(index)
  - augmentation(images, transform, state)

-✅ EMODataset
  - __init__(use_gpu, data_dir, sample_rate, n_sample_frames, width, height, img_scale, img_ratio, video_dir, drop_ratio, json_file, stage, transform)
  - __len__()
  - augmentation(images, transform, state)
  - __getitem__(index)

  ```


```javascript
- EMOAnimationPipeline (copied from magicanimate)
  - has some training code
```

 
