"""
srun -p INTERN2 --job-name='husky_multi_test' --gres=gpu:1 --cpus-per-task=8 --quotatype="auto" python -u demo/inference_new.py
"""

import abc
from typing import Optional

import os
import requests
from PIL import Image
from io import BytesIO

import torch
import torchvision.transforms as T
from peft import PeftModel
from torchvision.transforms.functional import InterpolationMode

from transformers import (
    LlamaTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from husky.model.modeling_husky_embody2 import HuskyForConditionalGeneration

from husky.conversation import (
    conv_templates,
    get_conv_template,
)

from husky.video_transformers import (
    GroupNormalize,
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
    get_index,
)

from husky.compression import compress_module
from decord import VideoReader, cpu

# import deepspeed

IGNORE_INDEX = -100
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMG_START_TOKEN = "<img>"
DEFAULT_IMG_END_TOKEN = "</img>"

DEFAULT_VIDEO_START_TOKEN = "<vid>"
DEFAULT_VIDEO_END_TOKEN = "</vid>"

def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024 ** 3)
            allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory

def load_model(
        model_path, device, num_gpus, max_gpu_memory=None, load_8bit=False, lora_weights=None
):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = LlamaTokenizer.from_pretrained(
        model_path, use_fast=False)

    if lora_weights is None:
        model = HuskyForConditionalGeneration.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    else:
        kwargs["device_map"] = "auto"
        model = HuskyForConditionalGeneration.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            lora_weights,
            **kwargs
        )

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    model = model.eval()
    return model, tokenizer

def load_image(image_file, input_size=224):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    crop_pct = 224 / 256
    size = int(input_size / crop_pct)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(size, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image = transform(image)
    return image

def load_video(video_path, num_segments=8):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = 224
    scale_size = 224
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std)
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    video = transform(images_group)
    return video

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops, encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

@torch.inference_mode()
def generate_stream(
        model, tokenizer, image_processor, params, device
):
    prompt = params["prompt"]
    images = params.get("images", None)
    videos = params.get("videos", None)
    temperature = float(params.get("temperature", 0.7))
    max_new_tokens = int(params.get("max_new_tokens", 1024))

    num_queries = model.config.num_query_tokens

    stop_words = ["Human: ", "Assistant: ", "###", "\n\n"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    generation_config = GenerationConfig(
        bos_token_id=1,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria
    )

    pixel_values = None
    if images is not None:
        pixel_values = load_image(images).to(device)  # only support one image
        image_query = DEFAULT_IMG_START_TOKEN + DEFAULT_IMG_END_TOKEN
        prompt = prompt.replace("<image>", image_query)

    elif videos is not None:
        pixel_values = load_video(videos).to(device)
        video_query = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_END_TOKEN
        prompt = prompt.replace("<video>", video_query)

    model_inputs = tokenizer([prompt], return_tensors="pt")
    model_inputs.pop("token_type_ids", None)

    if pixel_values is not None:
        model_inputs["pixel_values"] = pixel_values

        generation_output = model.generate(
            **model_inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )
    else:
        generation_output = model.language_model.generate(
            **model_inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )

    preds = generation_output.sequences
    outputs = tokenizer.batch_decode(preds, skip_special_tokens=True)
    return outputs

class Chat:
    def __init__(
            self,
            model_path,
            device,
            num_gpus=1,
            load_8bit=False,
            temperature=0.7,
            max_new_tokens=512,
            lora_path=None,
    ):
        model, tokenizer = load_model(
            model_path, device, num_gpus, load_8bit=load_8bit, lora_weights=lora_path
        )

        self.model = model
        # self.model.language_model = deepspeed.init_inference(
        #     self.model.language_model, mp_size=1, dtype=torch.float16, checkpoint=None, replace_with_kernel_inject=True)
        self.tokenizer = tokenizer
        num_queries = model.config.num_query_tokens

        self.device = device
        self.dtype = model.dtype

        stop_words = ["Human: ", "Assistant: ", "###", "\n\n"]
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.conv = get_conv_template("husky")

        self.image_query = DEFAULT_IMG_START_TOKEN + DEFAULT_IMG_END_TOKEN
        self.video_query = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_END_TOKEN

        self.generation_config = GenerationConfig(
            bos_token_id=1,
            do_sample=True,
            top_k=20,
            top_p=0.9,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria
        )

    def ask(self, text, conv, modal_type="image"):
        assert modal_type in ["text", "image", "video"]
        conversations = []

        if len(conv.messages) > 0 or modal_type == "text":
            conv.append_message(conv.roles[0], text)
        elif modal_type == "image":
            conv.append_message(conv.roles[0], self.image_query + "\n" + text)
        else:
            conv.append_message(conv.roles[0], self.video_query + "\n" + text)

        conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        return conversations

    @torch.no_grad()
    def get_image_embedding(self, image_file):
        pixel_values = load_image(image_file)
        pixel_values = pixel_values.unsqueeze(0).to(self.device, dtype=self.dtype)
        language_model_inputs = self.model.extract_feature(pixel_values)
        return language_model_inputs

    @torch.no_grad()
    def get_video_embedding(self, video_file):
        pixel_values = load_video(video_file)
        TC, H, W = pixel_values.shape
        pixel_values = pixel_values.reshape(TC // 3, 3, H, W).transpose(0, 1)  # [C, T, H, W]
        pixel_values = pixel_values.unsqueeze(0).to(self.device, dtype=self.dtype)
        assert len(pixel_values.shape) == 5
        language_model_inputs = self.model.extract_feature(pixel_values)
        return language_model_inputs

    @torch.no_grad()
    def answer(self, conversations, language_model_inputs, modal_type="image"):
        model_inputs = self.tokenizer(
            conversations,
            return_tensors="pt",
        )
        model_inputs.pop("token_type_ids", None)

        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        if modal_type == "text":
            generation_output = self.model.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        else:
            pixel_values = model_inputs.pop("pixel_values", None)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)

            generation_output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                language_model_inputs=language_model_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )

        preds = generation_output.sequences
        outputs = self.tokenizer.batch_decode(preds, skip_special_tokens=True)[0]

        if modal_type == "text":
            skip_echo_len = len(conversations[0]) - conversations[0].count("</s>") * 3
            outputs = outputs[skip_echo_len:].strip()

        return outputs 

if __name__ == '__main__':
    # model_path = "/mnt/petrelfs/zhangqinglong/Documents/Husky/work_dirs/husky_v3/EmbodiedGPT/pretrain_0727"
    model_path = "/mnt/petrelfs/share_data/gvembodied/workdirs/align_new_myyf"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chat = Chat(model_path, device=device, num_gpus=1, max_new_tokens=1024, load_8bit=False)

    vision_feature = None
    image_state = False
    video_state = False

    while True:
        query = input("\n")
        if query.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            if os.path.exists(query):
                print("received.")
                vision_feature = chat.get_image_embedding(query)
                chat.conv = get_conv_template("husky").copy()
                image_state = True
                continue
        if query.lower().endswith(('.mp4', '.mkv', '.avi', '.wmv', '.iso', ".webm")):
            if os.path.exists(query):
                print("received.")
                vision_feature = chat.get_video_embedding(query)
                chat.conv = get_conv_template("husky").copy()
                video_state = True
                continue

        if query == "stop":
            break
        if query == "clear" or query == "" or query == "\n":
            chat.conv = get_conv_template("husky").copy()
            image_state = False
            video_state = False
            os.system("clear")
            print("欢迎使用 husky-13b-zh 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue

        if image_state:
            modal_type = "image"
        elif video_state:
            modal_type = "video"
        else:
            modal_type = "text"

        # image_test = "assets/husky.jpg"
        # image_test = "assets/yoga.mp4"
        # video_test = "assets/pretty_girl.mp4"
        # video_test = "assets/stock-footage-billiards-concentrated-young-woman-playing-in-club.webm"
        # video_test = "assets/stock-footage-kherson-ukraine-may-open-free-rock-music-festival-crowd-partying-at-a-rock-concert.webm"
        conversations = chat.ask(text=query, conv=chat.conv, modal_type=modal_type)
        outputs = chat.answer(conversations, vision_feature, modal_type=modal_type)
        # NOTE: strip is important to align with the training data.
        chat.conv.messages[-1][1] = outputs.strip()

        print(f"Husky: \n{outputs}")
