import os
import random

from typing import Dict, Optional, Sequence
from PIL import PngImagePlugin, Image, ImageFile

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from robohusky.train.tcsloader import TCSLoader
from robohusky.conversation import get_conv_template

IGNORE_INDEX = -100

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

DEFAULT_IMG_START_TOKEN = "<img>"
DEFAULT_IMG_END_TOKEN = "</img>"

DEFAULT_VIDEO_START_TOKEN = "<vid>"
DEFAULT_VIDEO_END_TOKEN = "</vid>"

def is_image(image_file):
    if image_file.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
        return True
    else:
        return False

def is_video(image_file):
    if image_file.lower().endswith(('.mp4', '.mkv', '.avi', '.wmv', '.iso', ".webm")):
        return True
    else:
        return False

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform

def format_inputs(sources):
    # Apply prompt templates
    conv = get_conv_template("husky").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            # vision is only supported for the human input
            if role == conv.roles[0]:
                value = sentence["value"]
                if "<image>" in value:
                    if value.endswith("\n<image>"):
                        value = "<image>\n" + value.replace("\n<image>", "")
                    image_query = DEFAULT_IMG_START_TOKEN + DEFAULT_IMG_END_TOKEN
                    sentence["value"] = value.replace("<image>", image_query)

                elif "<video>" in value:
                    if value.endswith("\n<video>"):
                        value = "<video>\n" + value.replace("\n<video>", "")
                    video_query = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_END_TOKEN
                    sentence["value"] = value.replace("<video>", video_query)

            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    return conversations, conv

def process_func(examples, tokenizer, max_seq_length):
    conversations, conv = format_inputs(examples['conversations'])
    model_inputs = tokenizer(
        conversations,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    model_inputs.pop("token_type_ids", None)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    targets = model_inputs["input_ids"].clone()

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX

    model_inputs["labels"] = targets
    return model_inputs

class BaseDataset(Dataset):
    def __init__(self, dataset, processor, image_path="", input_size=224):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.image_path = image_path

        self.transform = build_transform(input_size)
        self.husky_processor = processor

        self.cached_data_dict = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        data = self.dataset[i]
        image_file = data.pop("image", None)

        if self.image_path != "":
            image_file = os.path.join(self.image_path, image_file)
            if not os.path.exists(image_file):
                return self.__getitem__((i + 1) % len(self.dataset))
            image = Image.open(image_file)
        else:
            image = Image.open(image_file)

        for k, v in data.items():
            data[k] = [v]
        ret = self.husky_processor(data)
        for k, v in ret.items():
            ret[k] = v[0]

        pixel_values = self.transform(image)
        ret["pixel_values"] = pixel_values

        self.cached_data_dict[i] = ret
        return ret

class CephDataset(Dataset):
    def __init__(self, dataset, processor, input_size=224):
        super(CephDataset, self).__init__()
        self.dataset = dataset

        self.transform = build_transform(input_size)
        self.husky_processor = processor

        conf_path = "./petrelf.conf"
        self.conf_path = os.path.abspath(conf_path)

        self.initialized = False
        self._init_memcached()

    def _init_memcached(self):
        if not self.initialized:
            assert self.conf_path is not None
            self.mt_loader = TCSLoader(self.conf_path)
            self.initialized = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data = self.dataset[i]
        image_file = data.pop("image", None)

        try:
            image = self.mt_loader(image_file).convert('RGB')
        except (AttributeError, OSError):
            with open("error.txt", 'a') as f:
                f.write(image_file + '\n')
            i = random.randint(0, len(self.dataset))
            return self.__getitem__(i % len(self.dataset))

        for k, v in data.items():
            data[k] = [v]

        ret = self.husky_processor(data)
        for k, v in ret.items():
            ret[k] = v[0]
        pixel_values = self.transform(image)
        ret["pixel_values"] = pixel_values
        return ret
