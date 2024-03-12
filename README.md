# Emote-hack
using chatgpt to reverse engineer code from HumanAIGC/EMO white paper. WIP


Just copy and paste the html from here in chatgpt (custom chat) https://chat.openai.com/g/g-UzGVIbBpB-diffuser-wizard
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


custom chatgpt (using diffusers models / pipelines)
https://chat.openai.com/g/g-UzGVIbBpB-diffuser-wizard
prompt - https://gist.github.com/johndpope/04879444d0979f244fb88c4929b989e9



decord + ubuntu cuda12 / 3090 support (for gpu acceleration in video reading)
https://github.com/johndpope/decord-cuda12

because all the videos are not the same size - they need reshapping - so abandoning gpu optimization for now.