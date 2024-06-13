import gradio as gr
import json
from openai import OpenAI
import os
import re
import requests
from PIL import Image, ImageFont, ImageDraw
from io import BytesIO
import time
import numpy as np
import imageio
import concurrent.futures
from transformers import AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList
import torch
import shutil
from moviepy.editor import VideoFileClip, AudioFileClip
import math
import copy
from tqdm import tqdm
from threading import Thread

## get the 歌词 from gpt-4o
def get_response(client, prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def withoutRoman(item):
    return not bool(re.search(r'[A-Za-z0-9\(\)]', item))

def get_lyric_list(result):
    lyrics = [item.strip() for item in result.split("<H>")]
    new_lyrics = []
    for item in lyrics:
        new_lyrics.extend(item.split())
    lyrics = [item for item in new_lyrics if len(item) > 0]
    lyrics = list(filter(withoutRoman, lyrics))
    return lyrics

def get_lyrics(client, prompt, model="gpt-4o"):
    return get_lyric_list(get_response(client, prompt, model))


## 2. Get the images according to the 歌词

def get_image_dalle(client, prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1):
  response = client.images.generate(
    model=model,
    prompt=prompt,
    size=size,
    quality=quality,
    n=n,
  )

  image_url = response.data[0].url
  response = requests.get(image_url, timeout=5000)
  img = Image.open(BytesIO(response.content))
  img_np = np.array(img)
  img = Image.fromarray(img_np)
  
  return img

### get_image_response和threading_query函数为多线程版本的get_image_dalle
def get_image_response(client, prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1, timeout=5000):
    # client = OpenAI(OPENAI_API_KEY)
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )

    image_url = response.data[0].url
    response = requests.get(image_url, timeout=timeout)
    img = Image.open(BytesIO(response.content))
    img_np = np.array(img)
    img = Image.fromarray(img_np)
    
    return img

def threading_query(client, datasets, folder_name, BATCH_SIZE = 8, SLEEP_TIME = 1, query_function = get_image_response, 
                    model="dall-e-3", size="1024x1024", quality="standard", timeout=5000):
    # The target function for multi-threading
    def target(prompt, num, base_num):
        time.sleep(num * SLEEP_TIME)
        img = query_function(client, prompt, model=model, size=size, quality=quality, timeout=timeout)
        image_path = os.path.join(folder_name, f"{num+base_num}.jpg")
        img.save(image_path)

    datasets = [datasets[i : i + BATCH_SIZE] for i in range(0, len(datasets), BATCH_SIZE)]

    base_num = 0
    for row_idx in tqdm(range(len(datasets))):
        row = datasets[row_idx]
        len_row = len(row)
        input_list = row
        thread_list = [Thread(target=target, daemon=True, name=str(i), kwargs={"prompt":item, "num":i, "base_num":base_num}) for i, item in enumerate(input_list)]
        [thread.start() for thread in thread_list]
        [thread.join() for thread in thread_list]
        base_num += len_row
        
## 加上字幕
def get_average_brightness(img):
    img_np = np.array(img)
    average_brightness = img_np.sum() / img_np.size / 255
    return average_brightness

# 智能判断应该用黑字幕或者白字幕
def should_use_white_or_black_text(img):
    average_brightness = get_average_brightness(img)
    # print(average_brightness)
    # 假设亮度大于0.5（即128）时背景较亮，使用黑色字幕
    if average_brightness > 0.5:
        return (0, 0, 0)
    else:
        return (255, 255, 255)
    
def write_text_to_image(img, text, color=(255, 255, 255), color_func=None, font_size=40, 
                        font_path="/ssdshare/MSZ_font/MaShanZheng-Regular.ttf", clone=True):
    # 以隶书为例，需要上传字体文件
    # clone = False 则在img本身上绘制，否则会创建一个新的副本
    # color_func: 判断字幕颜色的函数
    if clone:
        img = img.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    size_x, size_y = bbox[2], bbox[3]
    # 默认下侧距离边缘1个字高的正中地方绘制歌词
    img_x, img_y = img.size[0], img.size[1]
    if color_func:
        color = color_func(np.array(img)[img_y - font_size - size_y: img_y - font_size, (img_x - size_x)// 2:(img_x + size_x)// 2, :])
    draw.text(((img_x - size_x)// 2, img_y - font_size - size_y), text, color, font=font)
    return img

## generate video
def process_image(file_name, postfix=".jpg"):
    if file_name.endswith(postfix):
        image = Image.open(file_name)
        frame = image.convert("RGB")
        frame = np.array(frame.getdata()).reshape(frame.size[0], frame.size[1], 3)
    return frame
def generate_video(file_path="/root/project_output/output.mp4", image_files_dir=".", postfix=".jpg", fps=2):
    # fps: frame/second
    # read images in a directory
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 寻找所有 png 文件
        image_files = [os.path.join(image_files_dir, file) for file in os.listdir(image_files_dir) if file.endswith(postfix)]
        # 利用线程池并行处理图像
        images = list(executor.map(process_image, image_files))
    # 将图片转换为视频文件
    with imageio.get_writer(file_path, fps=fps) as video:
        for image in images:
            video.append_data(image)
    return 

## get the 歌曲 of each line of 歌词

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    
def inference(model, tokenizer, question, device="cuda:1"):
    print(question)
    question = f'[UNUSED_TOKEN_146]user\n{question}[UNUSED_TOKEN_145]\n'
    stop_words_ids = [ 
                    torch.tensor([2]).to('cuda:1'), #'</s>'
                    torch.tensor([92542]).to('cuda:1'), #'[UNUSED_TOKEN_145]'
                    ]
    stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])

    d = f"{question}"
    input_ids = tokenizer(d, return_tensors="pt")["input_ids"]
    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["[UNUSED_TOKEN_145]"])[0]]
    with torch.no_grad():
        generate = model.generate(input_ids.to(device), 
                                    do_sample=True,
                                    temperature=1.0,
                                    repetition_penalty=1.005, 
                                    max_new_tokens=1000, 
                                    top_p=0.8, 
                                    top_k=50, 
                                    eos_token_id=eos_token_id,
                                    stopping_criteria=stopping_criteria,)
    response = tokenizer.decode(generate[0].tolist(), skip_special_tokens=True)
    
    # return response[len('[UNUSED_TOKEN_146]assistant\n'):-len('[UNUSED_TOKEN_145]\n')]
    resp = response.split("[UNUSED_TOKEN_145]", 1)[-1].strip()
    if resp.startswith("[UNUSED_TOKEN_146]"):
        resp = resp[len("[UNUSED_TOKEN_146]"):].strip()
    if "<bop>" in resp:
        resp = resp.split("<bop>", 1)[-1]
    if "<eop>" in resp:
        resp = resp.split("<eop>", 1)[0]
    return resp.strip()

def filter_lyrics(lyrics):
    # 去除每句歌词末尾逗号和句号（如果有的话）
    new_lyrics = []
    for i in range(len(lyrics)):
        if lyrics[i][-1] in ["，", "。"]:
            new_lyrics.append(lyrics[i][:-1])
        else:
            new_lyrics.append(lyrics[i])
            
    return new_lyrics

def get_commas(lyrics):
    cnt = 0
    commas = []
    tmp = ''.join(lyrics)
    tmp = tmp.replace('，','|,|').replace('。','|,|')
    for item in tmp.split('|'):
        if(item == ','):
            commas.append(cnt)
            cnt = 0
            continue
        cnt += len(item)
    return commas

def get_direct_comma_position(lyrics):
    # 记录逗号在每句话中的位置，用于视频生成中的图片时长控制
    direct_comma_position = []
    for item in lyrics:
        item_comma = []
        for i in range(len(item)):
            if item[i] == '，' or item[i] == '。':
                item_comma.append(i)
        direct_comma_position.append(item_comma)
    return direct_comma_position

index_list = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]

def get_prompt(lyric_list):
    # no more than 10 sentences
    # example prompt
    # prompt = 'Compose a tune in harmony with the accompanying lyrics. <bol> Total 6 lines.\
    # The first line:在|那|玉|兰|花|开|的|地|方\n\
    # The second line:有|一|所|赫|赫|有|名|的|学|府\n\
    # The third line:那|是|清|华|我|的|清|华\n\
    # The fourth line:在|你|怀|抱|中|我|学|会|的|飞|翔\n\
    # The fifth line:清|华|大|学|你|的|名|字\n\
    # The sixth line:散|发|着|光\n<eol>'
    prompt = f'Compose a tune in harmony with the accompanying lyrics. <bol> Total {len(lyric_list)} lines.'
    for item in lyric_list:
        prompt += f'The {index_list[lyric_list.index(item)]} line:{"|".join(item)}\n'
    prompt += '<eol>'
    return prompt

def get_geci(lyrics, geci_model, geci_tokenizer):
    prompting = []
    cnt = 0
    results = []
    results_str = ''
    for item in lyrics:
        prompting.append(item)
        cnt += 1
        if cnt == 2:
            result = inference(geci_model, geci_tokenizer, get_prompt(prompting))
            cnt = 0
            prompting = []
            results.append(result)
            results_str += result
    results = ''.join(results)
    return results, results_str

def post_process_geci(results_str):
    result_list = results_str.split("line:")[1:]
    result_list = [item.split("The", 1)[0].strip().split("|") for item in result_list]
    result_list = [[jtem.split(",") for jtem in item] for item in result_list]
    result_list = [[[ktem.strip() for ktem in jtem] for jtem in item] for item in result_list]
    return result_list

def insert_commas_to_geci(result_list, commas):
    # 在result__list的相应位置加逗号，以便控制视频中的图片时长
    result_list_copy = copy.deepcopy(result_list)
    for i in range(len(result_list_copy)):
        for j in commas[i]:
            result_list_copy[i].insert(j, [])
    return result_list_copy

def extract_key_info(s):
    lines = s.split('line:')[1:]
    L = [[],[],[],[]]
    for line in lines:
        line += ' '
        line_p = re.findall('^(.+>)[^>]+$',line)
        sentences = line_p[0].split(' |')
        for sentence in sentences:
            results = sentence.split(',')
            if len(results)>0:
                for i in range(4):
                    L[i].append(results[i])
            else:
                    print('!!! not handled: ',sentence)
        for i in L:
            i.append(',')        
    return L

def get_rests(commas, L):
    res_pitch = []
    res_time = []
    res_lyric = []
    now_rest_time = 0.0
    cnt = 0
    flag = 0
    count = 0
    comma_used_in_pitch = commas.copy()
    if commas:
        comma_used_in_pitch = [item + 1 for item in comma_used_in_pitch]
        comma_used_in_pitch[0] -= 1
    for i in range(len(L[1])):
        if(count<len(comma_used_in_pitch)):
            if(flag == comma_used_in_pitch[count]):
                res_pitch.append('rest')
                res_time.append(str(now_rest_time / (cnt + 1) * 2))
                count += 1
                flag = 0
        flag += 1
        if(L[1][i] == ','):
            res_pitch.append('rest')
            res_time.append(str(now_rest_time / cnt * 3))
            now_rest_time = 0
            cnt = 0
            continue
        characters = re.findall(r'<(.*?)>', L[1][i])
        fake_pitch = []
        for j in range(len(characters)):
            fake_pitch.append(characters[j])
        real_pitch = ' '.join(fake_pitch)
        times = re.findall(r'<(.*?)>', L[2][i])
        fake_time = []
        for j in range(len(times)):
            fake_time.append(str(float(times[j]) * 3 / 1000))
        real_time = ' '.join(fake_time)
        rest_time = float(L[3][i].replace('<','').replace('>','')) / 1000
        res_pitch.append(real_pitch)
        res_time.append(real_time)
        now_rest_time += rest_time
        cnt += 1
    flag = 0
    count = 0
    for item in L[0]:
        if(count < len(commas)):
            if(flag == commas[count]):
                res_lyric.append('AP')
                count += 1
                flag = -1
        flag += 1
        if(item == ','):
            res_lyric.append('AP')
            continue
        res_lyric.append(item)
    return res_pitch, res_time, res_lyric

def get_final_lyrics(result_list):
    # 准备最终歌词
    final_lyrics = []
    for item in result_list:
        lyric = ""
        for i in range(len(item)):
            lyric += item[i][0]
        final_lyrics.append(lyric)
    return final_lyrics
   
def generate_images(client, final_lyrics, folder_name="/root/project_output/gen_images", model="dall-e-3", size="1024x1024", quality="standard", sleep_time=1):
    # 图片生成
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)

    i = 1
    # 单线程版本
    # for sentence in final_lyrics:
    #     img = get_image_dalle(f'{sentence}，图片上不要有文字。', model=model, size=size, quality=quality)
    #     image_path = os.path.join(folder_name, f"{i}.jpg")
    #     img.save(image_path)
    #     i += 1
    #     time.sleep(sleep_time)
    # 多线程版本
    threading_query(client, final_lyrics, folder_name, query_function=get_image_response, model=model, size=size, quality=quality, SLEEP_TIME=sleep_time)
        

def add_zimu(final_lyrics, folder_open = "/root/project_output/gen_images", folder_save = "/root/project_output/text_images"):
    # 加上字幕，字幕是中国特色，真没什么合适的英文
    if os.path.exists(folder_save):
        shutil.rmtree(folder_save)
    os.makedirs(folder_save, exist_ok=True)
    for i in range(len(final_lyrics)):
        img_path = os.path.join(folder_open, f"{i}.jpg")
        img = Image.open(img_path)
        imgnew = write_text_to_image(img, final_lyrics[i], color_func=should_use_white_or_black_text, 
                                    font_size=min(math.ceil(900 / len(final_lyrics[i])), 80))
        save_path = os.path.join(folder_save, f"{i}.jpg")
        imgnew.save(save_path)    
    
def prepare_times(result_list_copy, res_time):
    # 准备视频中的图片显示时长
    wordcount = []
    for item in result_list_copy:
        wordcount.append(len(item) + 1)
    # wordcount[-1] -= 1 # songcomposer不稳定。。。
    duration = []
    delta_time = sum(wordcount) - len(res_time)
    if delta_time > 0:
        wordcount[-1] -= delta_time

    j = 0
    lyric_time = 0
    for num in wordcount:
        for _ in range(num):
            time_per_word = [float(part) for part in res_time[j].split()]
            lyric_time += sum(time_per_word)
            j += 1
        duration.append(lyric_time)
        lyric_time = 0

    total_time = sum(duration)
    return duration, total_time

def generate_video_new(file_path="/root/project_output/output.mp4", image_files_dir=".", postfix=".jpg", fps=10, durations=None):
    # 控制每张图片的显示时间
    with concurrent.futures.ThreadPoolExecutor() as executor:
        image_files = [os.path.join(image_files_dir, file) for file in os.listdir(image_files_dir) if file.endswith(postfix)]
        images = list(executor.map(process_image, image_files))
    with imageio.get_writer(file_path, fps=fps) as video:
        for i, image in enumerate(images):
            for _ in range(durations[i]):
                video.append_data(image)   
 
def get_final_video(total_time, duration, file_path="/root/project_output/output.mp4", image_files_dir="/root/project_output/text_images", fps=10):
    # 生成视频
    # fps 越大生成越慢，但图片显示时长越精确
    top5_time = sum(duration[:5])
    top5_frame = math.floor(top5_time * fps)
    durations = []
    for i in range(len(duration)):
        if i < 4:
            if i % 2 == 0:
                durations.append(math.ceil(duration[i] * fps))
            else:
                durations.append(math.floor(duration[i] * fps))
        elif i == 4:
            durations.append(top5_frame - sum(durations))
        else:
            if i % 2 == 0:
                durations.append(math.ceil(duration[i] * fps))
            else:
                durations.append(math.floor(duration[i] * fps))
    generate_video_new(file_path=file_path, image_files_dir=image_files_dir, fps=fps, durations=durations)

def save_lyric_pitch_time(res_lyric, res_pitch, res_time, file_path_lyric, file_path_pitch, file_path_time):
    res_lyric_ = []
    res_pitch_ = []
    res_time_ = []
    flag = 0
    for i in range(len(res_lyric)):
        if(flag == 1):
            flag = 0
            continue
        if(i == len(res_lyric) - 1): 
            res_lyric_.append(res_lyric[i])
            res_pitch_.append(res_pitch[i])
            res_time_.append(res_time[i])
            continue
        if(res_lyric[i] == 'AP' and res_lyric[i+1] == 'AP'):
            res_lyric_.append('AP')
            res_pitch_.append('rest')
            real_time = float(res_time[i])+float(res_time[i+1])
            res_time_.append(str(real_time))
            flag = 1
        else:
            res_lyric_.append(res_lyric[i])
            res_pitch_.append(res_pitch[i])
            res_time_.append(res_time[i])
    print(res_lyric_)
    print(res_pitch_)
    print(res_time_)
    print(len(res_lyric_))
    print(len(res_pitch_))
    print(len(res_time_))
    res_lyric_ = ''.join(res_lyric_)
    res_pitch_ = '|'.join(res_pitch_)
    res_time_ = '|'.join(res_time_)
    with open(file_path_lyric, 'w') as f:
        f.write(res_lyric_)
    with open(file_path_pitch, 'w') as f:
        f.write(res_pitch_)
    with open(file_path_time, 'w') as f:
        f.write(res_time_)
    
### call
geci_ckpt_path = "/ssdshare/SongComposer" # your path
geci_tokenizer = AutoTokenizer.from_pretrained(geci_ckpt_path, trust_remote_code=True)
# 先half再cuda
geci_model = AutoModel.from_pretrained(geci_ckpt_path, trust_remote_code=True).half().to('cuda:1')

def predict(message):
#### Your Task ####
# Insert code here to perform the inference
    message = message + "。只显示歌词。只用逗号或句号断句。使用<H>分割行，不超过10句歌词。"
    print(message)
    os.environ['HTTP_PROXY']="http://Clash:QOAF8Rmd@10.1.0.213:7890"
    os.environ['HTTPS_PROXY']="http://Clash:QOAF8Rmd@10.1.0.213:7890"
    os.environ['ALL_PROXY']="socks5://Clash:QOAF8Rmd@10.1.0.213:7893"
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_API_KEY)

    lyrics = get_lyrics(client, message, model="gpt-4o")
    lyrics = filter_lyrics(lyrics)
    commas = get_commas(lyrics)
    direct_comma_position = get_direct_comma_position(lyrics)

    s, results_str = get_geci(lyrics, geci_model, geci_tokenizer)

    result_list = post_process_geci(results_str)
    result_list_copy = insert_commas_to_geci(result_list, direct_comma_position)

    L = extract_key_info(s)
    res_pitch, res_time, res_lyric = get_rests(commas, L)

    file_path_lyric = '/root/project_output/lyric_output.txt'
    file_path_pitch = '/root/project_output/pitch_output.txt'
    file_path_time = '/root/project_output/time_output.txt'
    save_lyric_pitch_time(res_lyric, res_pitch, res_time, file_path_lyric, file_path_pitch, file_path_time)

    # branch 1
    final_lyrics = get_final_lyrics(result_list)
    generate_images(client, final_lyrics, model="dall-e-3")
    add_zimu(final_lyrics)
    duration, total_time = prepare_times(result_list_copy, res_time=res_time)

    file_path="/root/project_output/output.mp4"
    image_files_dir="/root/project_output/text_images"
    wav_file_dir = "/ssdshare/DiffSinger/infer_out/example_out.wav"
    mp4_dir = "/root/project_output/final.mp4"
    get_final_video(total_time, duration, file_path=file_path, image_files_dir=image_files_dir, fps=10)

    # branch 2
    os.environ['PYTHONPATH'] = '/ssdshare/DiffSinger'
    os.environ['MY_DS_EXP_NAME'] = '0228_opencpop_ds100_rel'
    os.system("python /ssdshare/DiffSinger/inference/svs/my_infer.py --config /ssdshare/DiffSinger/usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name 0228_opencpop_ds100_rel")
    # merge
    video = VideoFileClip(file_path)
    audio = AudioFileClip(wav_file_dir)

    video = video.set_audio(audio)
    video.write_videofile(mp4_dir, codec='libx264')

    return gr.Video(mp4_dir)


input_interface = gr.Textbox(label="Please input the lyrics you want to generate the video.")
iface = gr.Interface(fn=predict, inputs=input_interface, outputs="video")
iface.launch(share=True)