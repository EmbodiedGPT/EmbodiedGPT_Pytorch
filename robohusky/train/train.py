#!/usr/bin/env python
# coding=utf-8
# Copyright Qing-Long Zhang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
import json
import logging
import os
import sys
import warnings
from functools import partial

from multiprocessing import cpu_count

from typing import Optional
from dataclasses import dataclass, field

from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset, load_from_disk

from robohusky.dist_utils import init_dist
from robohusky.model.modeling_husky_embody2 import HuskyForConditionalGeneration

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    LlamaTokenizer,
    Trainer,
    set_seed,
    default_data_collator,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from robohusky.base_dataset import (
    process_func,
    BaseDataset,
    CephDataset,
    build_transform
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from transformers.utils.logging import (
    set_verbosity_info,
    set_verbosity,
    enable_default_handler,
    enable_explicit_format,
)
from robohusky.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn
)

from robohusky.train.llama_rmsnorm_monkey_patch import (
    replace_llama_rmsnorm_with_fused_rmsnorm
)

replace_llama_attn_with_flash_attn()
replace_llama_rmsnorm_with_fused_rmsnorm()

IGNORE_INDEX = -100
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMG_START_TOKEN = "<img>"
DEFAULT_IMG_END_TOKEN = "</img>"

DEFAULT_VIDEO_START_TOKEN = "<vid>"
DEFAULT_VIDEO_END_TOKEN = "</vid>"

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.32.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_model: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    freeze_vision_model: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained vision model whose head dimensions are different."},
    )
    freeze_vision_adapter: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained vision adapter whose head dimensions are different."},
    )
    freeze_text_model: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained text model whose head dimensions are different."},
    )
    freeze_qformer: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained qformer model whose head dimensions are different."},
    )
    un_freeze_vision_embedding: bool = field(
        default=False,
        metadata={"help": "Will enable to tuning image patch_embedding when vision_model are frozen"},
    )
    un_freeze_video_embedding: bool = field(
        default=False,
        metadata={"help": "Will enable to tuning video patch_embedding when vision_model are frozen"},
    )
    un_freeze_llm_head: bool = field(
        default=False,
        metadata={"help": "Will enable to tuning video patch_embedding when vision_model are frozen"},
    )
    use_lora: bool = field(
        default=False, metadata={"help": "add the LoRA adapters to the base model"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data directory containing input files."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacrebleu) on a jsonlines file."
        },
    )
    train_val_split: Optional[float] = field(
        default=0.0, metadata={"help": "Percent to split off of train for validation."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file."},
    )
    image_path: Optional[str] = field(
        default=None,
        metadata={"help": "An optional image path"},
    )
    video_path: Optional[str] = field(
        default=None,
        metadata={"help": "An optional video path"},
    )
    input_size: Optional[int] = field(
        default=224,
        metadata={"help": "The input size of images."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    val_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    conv_style: Optional[str] = field(
        default=None, metadata={"help": "prompt style for a conversation."}
    )
    save_data_path: Optional[str] = field(
        default=None, metadata={"help": "prompt style for a conversation."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl", "parquet"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "jsonl",
                                     "parquet"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension == "json", "`test_file` should be a json file."

def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    init_dist(launcher='slurm', backend='nccl', port=29598)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Get the datasets
    # you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        ds = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            data_dir=data_args.data_dir,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]

        # ds = load_dataset(
        #     "json" if extension == "jsonl" else extension,
        #     data_files=data_files,
        #     split="train"
        # )
        ds = json.load(open(data_args.train_file, "r"))

    # 5. Load pretrained model, tokenizer, and image processor
    #
    # Distributed training: The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        legacy=True,
    )
    # add special token
    tokenizer.pad_token_id = 0
    if tokenizer.unk_token is None:
        tokenizer.add_special_tokens({"unk_token": DEFAULT_UNK_TOKEN})

    tokens_list = [
        DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_END_TOKEN,
        DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN
    ]
    tokenizer.add_tokens(tokens_list, special_tokens=True)

    model = HuskyForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, ignore_mismatched_sizes=True
    )
    embedding_size = model.language_model.get_input_embeddings().weight.shape[0]

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        model.language_model.resize_token_embeddings(len(tokenizer))
    model.config.text_config.vocab_size = len(tokenizer)

    model.config.use_cache = False

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_model:
        _freeze_params(model)
        # only update language projection
        model.language_projection.weight.requires_grad = True

    if model_args.freeze_vision_model:
        model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_vision_adapter:
        _freeze_params(model.vision_adapter)

    if model_args.freeze_qformer:
        model.qformer = model.qformer.eval()
        _freeze_params(model.qformer)
        model.query_tokens.requires_grad = False

    if model_args.freeze_text_model:
        _freeze_params(model.language_model)

    if model_args.use_lora:
        training_args.ddp_find_unused_parameters = False
        _freeze_params(model)
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        model.language_model.print_trainable_parameters()

    if model_args.un_freeze_video_embedding:
        _freeze_params(model)
        model.vision_model.video_embeddings.patch_embedding.weight.requires_grad = True
        model.vision_model.video_embeddings.class_embedding.requires_grad = True
        model.vision_model.video_embeddings.position_embedding.requires_grad = True

    if model_args.un_freeze_llm_head:
        model.language_model.lm_head.weight.requires_grad = True

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # 7. Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.

    # set padding.
    padding = "max_length" if data_args.pad_to_max_length else False

    def husky_processor(examples):
        processor = partial(
            process_func,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
        )
        model_inputs = processor(examples)
        return model_inputs

    # Data collator
    label_pad_token_id = IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    concat_dataset = []
    for data in ds:
        data_file = data["text_file"]
        extension = data_file.split(".")[-1]
        extension = "json" if extension == "jsonl" else extension
        logger.info(f"Loading dataset: {data['data_name']}")

        raw_dataset = load_dataset(extension, data_files=data_file, num_proc=cpu_count(), split="train")
        if data["data_type"] == "base":
            temp = BaseDataset(
                raw_dataset,
                processor=husky_processor,
                image_path=data["image_path"],
                input_size=data_args.input_size
            )
        else:
            temp = CephDataset(
                raw_dataset,
                processor=husky_processor,
                input_size=data_args.input_size
            )
        concat_dataset.append(temp)

    logger.info(f"All datasets have been loaded!")

    if len(concat_dataset) > 1:
        train_dataset = ConcatDataset(concat_dataset)
        # train_dataset = train_dataset.shuffle(seed=42)
    else:
        train_dataset = concat_dataset[0]

    # 8. Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if model_args.use_lora:
            model.language_model.save_pretrained(training_args.output_dir)
        else:
            trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()
