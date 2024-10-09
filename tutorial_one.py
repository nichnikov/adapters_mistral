"""
https://www.mlexpert.io/blog/fine-tuning-llm-on-custom-dataset-with-qlora
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
"""

import json
import os
from pprint import pprint
from datasets import load_dataset
import torch
import pandas as pd
import torch.nn as nn
import transformers
from huggingface_hub import notebook_login
import bitsandbytes as bnb

from mistral_inference.transformer import Transformer

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "tiiuae/falcon-7b"
# MODEL_NAME = "openchat/openchat_3.5"
 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# mistral_path = "/root/mistral_models/Nemo-Instruct"
#model = Transformer.from_folder(mistral_path)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
 
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters(model)