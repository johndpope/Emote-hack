

# EMO: Emote Portrait Alive - 
using chatgpt to reverse engineer code from HumanAIGC/EMO white paper. Work in progress - WIP


![Image](https://github.com/johndpope/Emote-hack/assets/289994/0d758a3a-841f-4849-b58c-439dda05c9a7)


https://arxiv.org/html/2402.17485v1


## WARNING -  the repo is work in progress. If you're here to train the model - come back later. Classes here are more like placeholders / building blocks.
The heavy lifting now is implementing the denoise of unet/ integrating attentions.



## Background papers to research / study 
- **AnimateDiff** (no training code?)
- **MagicAnimate** (no training code?)
- **AnimateAnyone** (no code)
- **Moore-AnimateAnyone** (training code)
  There's training code for 3 stages
  https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/train_stage_1.py
- **AnimateAnyone** - https://github.com/jimmyl02/animate/tree/main/animate-anyone
  3 training stages here
  https://github.com/jimmyl02/animate/tree/main/animate-anyone
 - **DiffusedHeads** - (no training code)  https://github.com/MStypulkowski/diffused-heads

While this is using poseguider - it's not hard to see a dwpose / facial driving the animation. https://www.reddit.com/r/StableDiffusion/comments/1281iva/new_controlnet_face_model/?rdt=50313&onetap_auto=true


These papers build on previous code. 




Claude3 has been the best to use to understand the paper.
It's possible to upload the text of paper / the diagram / and throw all the code at it. It has 200k context size. 

# Model Architecture:
Almost all the models are here
https://github.com/johndpope/Emote-hack/blob/main/Net.py

I'm exploring audio attention in junk folder.
There's a synthesize class that will generate both sounds / images.
this paper is supposed to train the audio attention so if it gets 
a specific sound - it would correspond to facial movements.
This needs further exploring / testing.
./junk/AudioAttention/synthesize.py 
ideally the network would take a sound (wav2vec stuff) - and show an facial expression. Right? Facelocator is drafted - could use extra eyes - the paper is saying the face region is a M mask for all the video frames.


## Face Locator:
The face locator is a separate module that learns to detect and localize the face region in a single input image.It takes a reference image as input and outputs the corresponding face region mask.(DRAFTED - train_stage_0.py)
UPDATE - I think we can substitute this work for Alibaba's existing trained model (6.8gb) to drop in replace and provide mask conditioning https://github.com/johndpope/Emote-hack/issues/28


## Speed Encoder:
The speed encoder takes the audio waveform as input and extracts speed embeddings.
The speed embeddings encode the velocity and motion information derived from the audio.

## Backbone Network (Audio-Driven Generator):
The backbone network is an audio-driven generator that takes the following inputs:
The face region image extracted by the face locator from the reference image.
The speed embeddings obtained from the speed encoder.
Noisy latents generated from the face region image.
The backbone network generates the output video frames conditioned on the audio and the reference image.
It incorporates the speed embeddings to guide the motion and velocity of the generated frames.


# Inference Process:

  ## Reference Image:
  During inference, the user provides a single reference image of the desired character.
  ## Face Locator:
  The face locator is applied to the reference image to detect and extract the face region.
  The face region mask is obtained from the face locator.
  ## Audio Waveform:
  The user provides an audio waveform as input, which can be a speech or any other audio signal.
  ## Speed Encoder:
  The audio waveform is passed through the speed encoder to obtain the speed embeddings.
  The speed embeddings encode the velocity and motion information derived from the audio.

  ## Backbone Network (Audio-Driven Generator):
  The extracted face region image, speed embeddings, and noisy latents are fed into the backbone network.
  The backbone network generates the output video frames conditioned on the audio and the reference image.
  The speed embeddings guide the motion and velocity of the generated frames, ensuring synchronization with the audio.

# Training Process:

  ## Face Locator:
  The face locator is trained separately using a dataset of images with corresponding face region annotations or masks.

  ## Speed Encoder:
  The speed encoder is trained using a dataset of audio waveforms and corresponding velocity or motion annotations.
  ## Backbone Network (Audio-Driven Generator):
  The backbone network is trained using a dataset consisting of reference images, audio waveforms, and corresponding ground truth video frames.
  During training, the face locator extracts the face regions from the reference images, and the speed encoder provides the speed embeddings from the audio waveforms.
  The backbone network learns to generate video frames that match the ground truth frames while being conditioned on the audio and reference image.

In this rearchitected model, the inference process takes a single reference image and an audio waveform as input, and the model generates the output video frames conditioned on the audio and the reference image. The face locator and speed encoder are used to extract the necessary information from the inputs, which is then fed into the backbone network for generating the output video.



## Training Data (☢️ dont need this yet.)

- **Total Videos:** 36,000 facial videos
- **Total Size:** 40GB


### Training Strategy
for now - to simplify problem - we can use a single video the ./data/M2Ohb0FAaJU_1.mp4. We don't need the 40gb of videos. 
Once all stages are trained on this single video (by overfitting this single use case)  we should be able to give EMO the first frame + audio and it should produce a video with head moving.



### Torrent Download

You can download the dataset via the provided magnet link or by visiting [Academic Torrents](https://academictorrents.com/details/843b5adb0358124d388c4e9836654c246b988ff4).

```plaintext
magnet:?xt=urn:btih:843b5adb0358124d388c4e9836654c246b988ff4&dn=CelebV-HQ&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=https%3A%2F%2Fipv6.academictorrents.com%2Fannounce.php
```



### Sample Video (Cropped & Trimmed)

Note: The sample includes rich tagging. For more details, see `./data/test.json`.

[![Watch the Sample Video](./junk/frame_0094_debug.jpg)](./junk/M2Ohb0FAaJU_1.mp4)



### Models / architecture
(flux)



```javascript
- ✅ ReferenceNet
  - __init__(self, config, reference_unet, denoising_unet, vae, dtype)
  - forward(self, reference_image, motion_features, timesteps)

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

magicanimate code - it has custom blocks for unet - maybe very useful when wiring up the attentions in unet.
```javascript
- EMOAnimationPipeline (copied from magicanimate)
  - has some training code / this should not need text encoder / clip to aling with EMO paper. 
```

