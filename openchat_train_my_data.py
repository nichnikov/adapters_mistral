import os
import pandas as pd
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from huggingface_hub import login
import datasets
from datasets import DatasetDict


wandb_API_token = "c3db32f9452b8387ca3468644bffd7f7de99bf02"

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False) # , optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

df = pd.read_csv(os.path.join("data", "queries2024_ltn15_with_answr_ltn200.csv"), sep="\t")
ds = datasets.Dataset.from_pandas(df[["QueryText", "Answer"]])

train_testvalid = ds.train_test_split(test_size=0.2)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']})

print(train_test_valid_dataset)

train_dataset = train_test_valid_dataset['train']
eval_dataset = train_test_valid_dataset['test']
test_dataset = train_test_valid_dataset['validation']

print(train_dataset)
print(eval_dataset)
print(test_dataset)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# base_model_path = "/root/mistral_models/Nemo-Instruct"
# base_model_id = "Mistral-Nemo-Instruct-2407"
# base_model_id = "mistralai/Mistral-7B-v0.1"
# base_model_id = "openchat/openchat_3.5"
base_model_id = "Vikhrmodels/Vikhr-7b-0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

login(token="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")
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)


tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Ты квалифицированный бухгалтер, тебе задают профессиональный вопрос. Используй свои знания и как можно точнее отвечай на вопрос.


### Вопрос:
{data_point["QueryText"]}


### Ответ:
{data_point["Answer"]}
"""
    return tokenize(full_prompt)



tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

print(tokenized_train_dataset[4])

print(tokenized_train_dataset[4]['input_ids'])

print(len(tokenized_train_dataset[4]['input_ids']))

print("Текст Вопроса: " + test_dataset[1]['QueryText'])
print("Текст ответа: " + test_dataset[1]['Answer'] + "\n")

eval_prompt = """Ты квалифицированный бухгалтер, тебе задают профессиональный вопрос. Используй свои знания и как можно точнее отвечай на вопрос.


### Вопрос:
Добрый день Как в отчёте о движении капитала отразить погашение убытка текущего года за счёт уставного капитала?


### Ответ:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True))

from peft import prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
print_trainable_parameters(model)
# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

print(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

import transformers
from datetime import datetime


project = "write-support-bss"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name


tokenizer.pad_token = tokenizer.eos_token


trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        # per_device_train_batch_size=10, # было 2
        gradient_accumulation_steps=4,
        max_steps=1000,
        learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
        logging_steps=50,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()