import os
import random

from typing import Dict, Optional, Sequence, Iterator, List, Iterable, Union
from PIL import PngImagePlugin, Image, ImageFile, ImageOps

import numpy as np

import torch
from torch.utils.data import (
    Dataset,
    ConcatDataset,
    Sampler,
    WeightedRandomSampler
)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from robohusky.train.tcsloader import TCSLoader

from decord import VideoReader, cpu
from robohusky.video_transformers import (
    GroupNormalize,
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
    get_index,
)

from robohusky.conversation import get_conv_template

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

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

DEFAULT_EMBED_TOKEN = "<quad>"

conf_path = "/your path to/petrelf.conf"

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

def is_numpy(image_file):
    if image_file.endswith(".npy"):
        return True
    else:
        return False

def get_media_type(image_file):
    if is_image(image_file):
        return "image"
    elif is_video(image_file):
        return "video"
    elif is_numpy(image_file):
        return "numpy"
    else:
        return "text"

def build_transform(input_size, norm_type="openai", media_type="image"):
    if norm_type == "openai":
        mean = OPENAI_CLIP_MEAN
        std = OPENAI_CLIP_STD
    elif norm_type == "imagenet":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    else:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

    if media_type == "image":
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    elif media_type == "video":
        transform = T.Compose([
            GroupScale(int(input_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(input_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=mean, std=std)
        ])
    else:
        transform = None
    return transform

def check_format(data):
    if not ('id' in data and 'image' in data and 'conversations' in data and len(data['conversations']) % 2 == 0):
        print(f"Lake field: {data}")
        return False
    for i, message in enumerate(data['conversations']):
        if i == 0:
            if not (message['value'].startswith("<image>\n") or message['value'].endswith("\n<image>")):
                print(f"No <image>: {data}")
                return False
        if i % 2 == 0:
            if not (message['from'] == 'human'):
                print(f"Not from human: {data}")
                return False
        else:
            if not (message['from'] == 'gpt'):
                print(f"Not from gpt: {data}")
                return False
        if message['value'] is None or (len(message['value']) == 0):
            print(f"No Message: {data}")
            return False
    return True

def format_inputs(sources, conv_tempt="husky", num_query_tokens=256):
    # Apply prompt templates
    conv = get_conv_template(conv_tempt).copy()
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

                    image_query = DEFAULT_IMG_START_TOKEN + num_query_tokens * DEFAULT_EMBED_TOKEN + DEFAULT_IMG_END_TOKEN
                    sentence["value"] = value.replace("<image>", image_query)

                elif "<video>" in value:
                    if value.endswith("\n<video>"):
                        value = "<video>\n" + value.replace("\n<video>", "")

                    video_query = DEFAULT_VIDEO_START_TOKEN + num_query_tokens * DEFAULT_EMBED_TOKEN + DEFAULT_VIDEO_END_TOKEN
                    sentence["value"] = value.replace("<video>", video_query)

            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    return conversations, conv

def process_func(examples, tokenizer, max_seq_length=-1, conv_tempt="husky", num_query_tokens=256):
    conversations, conv = format_inputs(examples['conversations'], conv_tempt, num_query_tokens)
    if max_seq_length < 0:
        model_inputs = tokenizer(
            conversations,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    else:
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
    def __init__(
            self,
            dataset,
            processor,
            image_path="",
            input_size=224,
            num_segments=8,
            norm_type="openai",
            media_type="image"
    ):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.image_path = image_path
        self.input_size = input_size
        self.num_segments = num_segments

        self.media_type = media_type
        self.transform = build_transform(input_size, norm_type, media_type)
        self.husky_processor = processor
        self.tcs_loader = TCSLoader(os.path.abspath(conf_path), media_type=media_type)

        self.cached_data_dict = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        data = self.dataset[i]
        image_file = data["image"] if "image" in data else data["video"]

        if self.media_type == "llm" or image_file == "":
            # Pseudo pixel_values
            # pixel_values = torch.zeros(size=(3, self.input_size, self.input_size))
            pixel_values = None
        else:
            if self.image_path != "":
                image_file = os.path.join(self.image_path, image_file)
            if "s3://" not in image_file and not os.path.exists(image_file):
                i = random.randint(0, len(self.dataset))
                return self.__getitem__(i % len(self.dataset))

            try:
                if self.media_type == "image":
                    # load from ceph
                    if "s3://" in image_file:
                        image = self.tcs_loader(image_file)
                    else:
                        image = Image.open(image_file).convert('RGB')

                    # process image with extreme aspect ratios
                    height, width = image.size
                    if height / width >= 1.8:
                        delta = height - width
                        padding = (0, delta // 2, 0, delta - delta // 2)
                        image = ImageOps.expand(image, padding)
                    elif height / width <= 0.56:
                        delta = width - height
                        padding = (delta // 2, 0, delta - delta // 2, 0)
                        image = ImageOps.expand(image, padding)
                    pixel_values = self.transform(image)
                elif self.media_type == "video":
                    if "s3://" in image_file:
                        vr = self.tcs_loader(image_file)
                    else:
                        vr = VideoReader(image_file, ctx=cpu(0))

                    num_frames = len(vr)
                    frame_indices = get_index(num_frames, self.num_segments)
                    images_group = list()
                    for frame_index in frame_indices:
                        img = Image.fromarray(vr[frame_index].asnumpy())
                        images_group.append(img)
                    pixel_values = self.transform(images_group)
                    TC, H, W = pixel_values.shape
                    pixel_values = pixel_values.reshape(TC // 3, 3, H, W).transpose(0, 1)  # [C, T, H, W]
                else:
                    # load numpy
                    if "s3://" in image_file:
                        pixel_values = self.tcs_loader(image_file)
                    else:
                        pixel_values = np.load(image_file)
                    pixel_values = torch.tensor(pixel_values).transpose(0, 1)
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

        if pixel_values is not None:
            ret["pixel_values"] = pixel_values

        self.cached_data_dict[i] = ret
        return ret

class WeightedConcatDataset(ConcatDataset):
    def __init__(
            self,
            datasets: List[Dataset],
            weights: Sequence[float] = None,
            replacement: bool = True,
            batch_size: int = -1,
            generator=None
    ) -> None:
        super().__init__(datasets)
        if weights is None:
            weights = [1.0] * len(self.datasets)
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             "weights have shape {}".format(tuple(weights_tensor.shape)))
        self.weights = weights_tensor
        self.batch_size = batch_size

        self.replacement = replacement
        self.generator = generator

        if self.batch_size <= 0:
            self.num_samples = sum([len(d) for d in datasets])
            self.sampler = WeightedRandomSampler(
                weights=self.weights,
                num_samples=self.num_samples,
                replacement=self.replacement
            )
        else:
            self.task_batches = [len(d) // batch_size for d in datasets]
            self.num_samples = sum(self.task_batches) * batch_size
            self.sampler = WeightedBatchSampler(
                weights=self.weights,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                replacement=self.replacement
            )

    def __iter__(self) -> Iterator[int]:
        return iter(self.sampler)

    def __len__(self) -> int:
        return self.num_samples

class WeightedBatchSampler(Sampler[int]):
    weights: torch.Tensor
    num_samples: int
    batch_size: int
    replacement: bool

    def __init__(
            self,
            weights: Sequence[float],
            num_samples: int,
            batch_size: int,
            replacement: bool = True,
            generator=None
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             "weights have shape {}".format(tuple(weights_tensor.shape)))

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_batches = num_samples // batch_size
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_batches, self.replacement, generator=self.generator)
        rand_tensor = rand_tensor.repeat_interleave(self.batch_size)

        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples
