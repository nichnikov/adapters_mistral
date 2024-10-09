
"""
https://www.e2enetworks.com/blog/a-step-by-step-guide-to-fine-tuning-the-mistral-7b-llm
"""

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from huggingface_hub import login


fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False) # , optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

from datasets import load_dataset


train_dataset = load_dataset('gem/viggo', split='train')
eval_dataset = load_dataset('gem/viggo', split='validation')
test_dataset = load_dataset('gem/viggo', split='test')

print(train_dataset)
print(eval_dataset)
print(test_dataset)
print(train_dataset["gem_id"][:5])
print(train_dataset["meaning_representation"][:5])
print(train_dataset["target"][:5])
print(train_dataset["references"][:5])

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


base_model_path = "/root/mistral_models/Nemo-Instruct"
# base_model_id = "Mistral-Nemo-Instruct-2407"
base_model_id = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

login(token="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")
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
