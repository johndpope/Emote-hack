# Emote-hack
using chatgpt to reverse engineer code from HumanAIGC/EMO white paper. WIP


Just copy and paste the html from here in chatgpt chat
https://arxiv.org/html/2402.17485v1


This code is 70% there (based off previous work from HumanAIGC)
see the train_stage_1.py
https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/train_stage_1.py

✅  Training data 

https://academictorrents.com/details/843b5adb0358124d388c4e9836


use extractframes.py
```python
video_path = 'M2Ohb0FAaJU_1.mp4'
extract_and_save_frames(video_path,'.')
```

Mask Generation - cherry picked from video retalking. Is it too old?
https://github.com/OpenTalker/video-retalking

https://raw.githubusercontent.com/OpenTalker/video-retalking/

#### Pretrained Models for Video-retalking 
Please download our [pre-trained models](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0?usp=share_link) and put them in `./checkpoints`.
```shell
mkdir ./checkpoints  
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth -O ./checkpoints/30_net_gen.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip -O ./checkpoints/BFM.zip
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt -O ./checkpoints/DNet.pt
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth -O ./checkpoints/ENet.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/expression.mat -O ./checkpoints/expression.mat
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth -O ./checkpoints/face3d_pretrain_epoch_20.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth -O ./checkpoints/GFPGANv1.3.pth
wget https://carimage-1253226081.cos.ap-beijing.myqcloud.com/gpen/GPEN-BFR-1024.pth -O ./checkpoints/GPEN-BFR-1024.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/LNet.pth -O ./checkpoints/LNet.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth -O ./checkpoints/ParseNet-latest.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth -O ./checkpoints/RetinaFace-R50.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat -O ./checkpoints/shape_predictor_68_face_landmarks.dat
unzip -d ./checkpoints/BFM ./checkpoints/BFM.zip
```



mkdir images_folder