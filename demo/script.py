"""
srun -p INTERN2 --job-name='husky_multi_test' --gres=gpu:1 --cpus-per-task=8 --quotatype="auto" python -u demo/test.py
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
from collections import defaultdict

from transformers import (
    LlamaTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from robohusky.model.modeling_husky_embody2 import HuskyForConditionalGeneration

from robohusky.conversation import (
    conv_templates,
    get_conv_template,
)

from robohusky.video_transformers import (
    GroupNormalize,
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
    get_index,
)

from robohusky.compression import compress_module
from decord import VideoReader, cpu

import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

import cv2
import matplotlib.pyplot as plt

from maskrcnn_benchmark.config import cfg
from glip.predictor_glip import GLIPDemo
#from segment_anything import build_sam, SamPredictor

# import deepspeed

IGNORE_INDEX = -100
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMG_START_TOKEN = "<img>"
DEFAULT_IMG_END_TOKEN = "</img>"

DEFAULT_VIDEO_START_TOKEN = "<vid>"
DEFAULT_VIDEO_END_TOKEN = "</vid>"

def iou(box1, box2):
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union != 0 else 0


def show_predictions(scores, boxes, classes):
    num_obj = len(scores)
    if num_obj == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_obj))

    for obj_ind in range(num_obj):
        box = boxes[obj_ind]
        score = scores[obj_ind]
        name = classes[obj_ind]

        # color_mask = np.random.random((1, 3)).tolist()[0]
        color_mask = colors[obj_ind]

        # m = masks[obj_ind][0]
        # img = np.ones((m.shape[0], m.shape[1], 3))
        # for i in range(3):
        #     img[:,:,i] = color_mask[i]
        # ax.imshow(np.dstack((img, m*0.45)))

        x0, y0, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color_mask, facecolor=(0, 0, 0, 0), lw=2))

        label = name + ': {:.2}'.format(score)
        ax.text(x0, y0, label, color=color_mask, fontsize='large', fontfamily='sans-serif')

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

def read_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    
    return image

def load_image(image, input_size=224):
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
            pad_token_id=0,
            do_sample=True,
            top_k=20,
            top_p=0.9,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria
        )

    def reduce_boxes(self,scores,boxes,names,iou_threshold = 0.5):  
        # print("debug!!!!$$$$$$$$$$$$$$$$$$$")
        # print(scores)
        # print(boxes)
        # print(names)
        keep_boxes = torch.ones(len(boxes), dtype=torch.bool)
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                if iou(boxes[i], boxes[j]) > iou_threshold and keep_boxes[i] and keep_boxes[j]:
                    if scores[i] > scores[j]:
                        keep_boxes[j] = False
                    else:
                        keep_boxes[i] = False
        # print("keep_boxes",keep_boxes)
        scores = torch.tensor(scores) if not isinstance(scores, torch.Tensor) else scores
        filtered_scores = scores[keep_boxes]
        filtered_boxes = boxes[keep_boxes]
        filtered_names = [name for i, name in enumerate(names) if keep_boxes[i]]

        return filtered_scores,filtered_boxes,filtered_names


    def refine_bbox_dict(self,boxes, names):
        counter_dict = {}
        result_dict = {}

        for i, name in enumerate(names):
            if name in counter_dict:
                counter_dict[name] += 1
            else:
                counter_dict[name] = 0  

            unique_name = f"{name}_{counter_dict[name]}"

            result_dict[unique_name] = boxes[i].tolist()

        for name, box in result_dict.items():
            print(f"{name}: {box}")
        
        return result_dict



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

    def merge_box(self,dict1,dict2,dict3):
        combined_dict = defaultdict(list)
        for key, value in dict1.items():
            combined_dict[key].append(value)
        for key, value in dict2.items():
            combined_dict[key].append(value)

        for key, value in dict3.items():
            combined_dict[key].append(value)

        combined_dict = dict(combined_dict)

        # for key, value in combined_dict.items():
        #     print(f'{key}: {value}')
        
        return combined_dict


    def glip_show(self,detect_model,image_glip, object_list,view="left"):
        image_glip=cv2.cvtColor(image_glip, cv2.COLOR_BGR2RGB)
        scores, boxes, names = detect_model.inference_on_image(image_glip, object_list)
        print(scores, boxes, names)
        #scores, boxes, names = self.reduce_boxes(scores, boxes, names)
        #box_dict=self.refine_bbox_dict(boxes, names)
        # draw output image
        plt.figure(figsize=(10, 10))
        # image_rbg = cv2.cvtColor(image_glip, cv2.COLOR_BGR2RGB)
        plt.imshow(cv2.cvtColor(image_glip, cv2.COLOR_BGR2RGB))
        show_predictions(scores, boxes, names)
        plt.axis('off')
        plt.savefig("./test_glip_"+view+".png",bbox_inches="tight", dpi=300, pad_inches=0.0)
        # return box_dict
        return None

    @torch.no_grad()
    def ask_question(self,input_imgs,detect_model):
        left_view = input_imgs[0]
        right_view = input_imgs[1]
        top_view = input_imgs[2]
        # print("#############")
        vision_feature = self.get_image_embedding(left_view)
        left_glip = np.array(left_view)
        right_glip = np.array(right_view)
        top_glip = np.array(top_view)
        self.conv = get_conv_template("husky").copy()
        modal_type = "image"
        conversations = self.ask(text="Please describe this image in detail and focus on the parts are interactive.", conv=self.conv, modal_type=modal_type)
        caption = self.answer(conversations, vision_feature, modal_type=modal_type)
        print("caption:",caption)
        self.conv.messages[-1][1] = caption.strip()
        conversations = self.ask(text="List all visible objects or object-parts in a single line with brief labels only and separate them by commas. Pithy!!!", conv=self.conv, modal_type=modal_type)
        object_list = self.answer(conversations, vision_feature, modal_type=modal_type)
        object_list = "red apple, green pear"
        print("object_list",object_list)
        left_dict = self.glip_show(detect_model,left_glip, object_list,view="left")
        right_dict = self.glip_show(detect_model,right_glip, object_list,view="right")
        top_dict = self.glip_show(detect_model,top_glip, object_list,view="top")
        # final_dict = self.merge_box(left_dict,right_dict,top_dict)
        # return final_dict

        
 
    @torch.no_grad()
    def post_question(self,detect_model,mod="detect"):
        image_state = False
        video_state = False
        print("#############")
        print("please input the query!")
        query = input("\n")  
        if query.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            if os.path.exists(query):
                print("received.")
                image=read_image(query)
                vision_feature = self.get_image_embedding(image)
                image_glip = np.array(image)
                self.conv = get_conv_template("husky").copy()
                image_state = True
        if query.lower().endswith(('.mp4', '.mkv', '.avi', '.wmv', '.iso', ".webm")):
            if os.path.exists(query):
                print("received.")
                vision_feature = self.get_video_embedding(query)
                self.conv = get_conv_template("husky").copy()
                video_state = True
        if image_state:
            modal_type = "image"
        elif video_state:
            modal_type = "video"
        else:
            modal_type = "text"
        
        # if mod=="caption":
        #     conversations = self.ask(text="please describe this image in detail", conv=self.conv, modal_type=modal_type)
        #     caption = self.answer(conversations, vision_feature, modal_type=modal_type)
        #     print("mod is caption")List all visible objects and visible object-parts in a single line only and separate them by semicolons. Pithy!!!
        #     print(caption)

        if mod=="detect":
            conversations = self.ask(text="Please describe this image in detail and focus on the parts are interactive.", conv=self.conv, modal_type=modal_type)
            caption = self.answer(conversations, vision_feature, modal_type=modal_type)
            self.conv.messages[-1][1] = caption.strip()
            conversations = self.ask(text="List all visible objects or object-parts in a single line with brief labels only and separate them by commas. Pithy!!!", conv=self.conv, modal_type=modal_type)
            #object_list = self.answer(conversations, vision_feature, modal_type=modal_type)
            
            scores, boxes, names = detect_model.inference_on_image(image_glip, object_list)
            scores, boxes, names = self.reduce_boxes(scores, boxes, names)
            box_dict=self.refine_bbox_dict(boxes, names)
            # draw output image
            plt.figure(figsize=(10, 10))
            # image_rbg = cv2.cvtColor(image_glip, cv2.COLOR_BGR2RGB)
            plt.imshow(image_glip)
            show_predictions(scores, boxes, names)
            plt.axis('off')
            plt.savefig("./test_glip.png",bbox_inches="tight", dpi=300, pad_inches=0.0)
            print("mod is detection and caption")
            print(caption)
            print(object_list)
            print("box:",boxes)
            print("names:",names)






if __name__ == '__main__':
    parser = argparse.ArgumentParser("Segment-Anything-and-Name-It Demo", add_help=True)
    parser.add_argument(
        "--glip_checkpoint", type=str, default="/mnt/petrelfs/share_data/gvembodied/detmodels/glip_large.pth", help="path to checkpoint file"
    )
    parser.add_argument(
        "--glip_config_file", type=str, default="/mnt/petrelfs/share_data/gvembodied/embodiedgptatS/demo/glip/configs/glip_Swin_L.yaml", help="path to configuration file"
    )
    #parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    #parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", type=str, default="outputs", help="output directory")
    parser.add_argument("--model_path", type=str, default="/mnt/petrelfs/share_data/gvembodied/workdirs/align_gpt4v_part2010", help="model path")
    parser.add_argument("--device", type=str, default="cuda", help="running on cuda")
    args = parser.parse_args()
    chat = Chat(args.model_path, device=args.device, num_gpus=1, max_new_tokens=1024, load_8bit=False)
    vision_feature = None
    image_state = False
    video_state = False

    # cfg
    glip_checkpoint = args.glip_checkpoint
    glip_config_file = args.glip_config_file

    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(glip_config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", glip_checkpoint])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    # initialize glip
    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.6,
        show_mask_heatmaps=False
    )
    imgs_path="/mnt/petrelfs/share_data/gvembodied/embodiedgptatS/CLIP/"
    test_imgs=[read_image((imgs_path+"left.png")),read_image((imgs_path+"right.png")),read_image((imgs_path+"top.png"))]
    while True:
        chat.ask_question(test_imgs,glip_demo)
        if input():
            continue


    
    # chat.post_question(detect_model=glip_demo,mod="caption")
    #chat.post_question(detect_model=glip_demo,mod="detect")


    

