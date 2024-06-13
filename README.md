# Music Dream Visualizer (MDV)
## Introduction
Welcome to music dream visualizer! This is a LLM course project built by three undergraduate students, which explores the potential of LLM-based song generating. In this project, we tried to compose a song by just offering one prompt to the pipeline.

The whole pipeline:


## Get Started
To get started, you need to set the environment:
```sh
mkdir /root/project_output
sudo apt-get update
sudo apt-get install -y libsndfile1
conda create -n your_env_name python=3.10.13
pip install -r requirements.txt
```
You should make your directory structure look like:
```
|--root
    |--requirements.txt
    |--chat.py
    |--chat_gradio.py
    |--project_output
        |--gen_images
            |--0.jpg
            |--1.jpg
            |--2.jpg
        |--text_images
            |--0.jpg
            |--1.jpg
            |--2.jpg
        |--output.txt
        |--lyric_output.txt
        |--pitch_output.txt
        |--time_output.txt
        |--output.mp4
        |--final.mp4
```
Files in the project_output directory are generated during the inference.

To run the pipeline, you also need an openai api-key, SongComposer model from Mar2Ding/songcomposer_sft on huggingface, and DiffSinger model from https://github.com/MoonInTheRiver/DiffSinger/tree/master (SVS version, Opencpop dataset, link B). If you run this on the server of professor Xu, these two models are already loaded in /ssdshare/.

