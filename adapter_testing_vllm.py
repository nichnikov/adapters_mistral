""""
Тестирование валидации на данных экспертной поддержки
с использованием ускорителей от фреймворка vllm
https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_with_profiler.py
https://github.com/vllm-project/vllm/issues/1129

тут чувак предлагает смержить сначала модель с ПЕФТ и потом применить vllm
https://github.com/Reasoning-Lab/Elementary-Math-Solving-Zalo-AI-2023/blob/79ed4742d91755b4a00fadd0079279394222928b/merge_peft_adapter.py
"""
import os
import re
import pandas as pd
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# base_model_id = "Vikhrmodels/Vikhr-7b-0.1"
base_model_id = "openchat/openchat_3.5"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

test_data_df = pd.read_csv(os.path.join("data", "validate_QueriesAnswers.csv"), sep="\t")
test_data_dict = test_data_df.to_dict(orient="records")

patterns = re.compile(r"\n|\¶|(?P<url>https?://[^\s]+|\s+)")
for d in test_data_dict[205:215]:
    promt = """Нужно сопоставить Вопрос и Ответ в Тексте и сделать Суждение: Правда или Ложь
    
    ### Текст:
    Вопрос: {} Ответ: {}
    
    
    ### Суждение:
    """.format(str(d["Query"]), patterns.sub(" ", str(d["answer"])))

    from peft import PeftModel
    ft_model = PeftModel.from_pretrained(base_model, "mistral-write-support-bss/checkpoint-1000")

    model_input = tokenizer(promt, return_tensors="pt").to("cuda")
    ft_model.eval()

    with torch.no_grad():
        open_chat_opinion = tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=300, pad_token_id=2)[0], skip_special_tokens=True)
        print(re.sub(r'\s+', " ", open_chat_opinion))
        if re.findall(r"### Суждение:.*\n.*Правда", open_chat_opinion):
            print("Result:", "Правда")
        if re.findall(r"### Суждение:.*\n.*Ложь", open_chat_opinion):
            print("Result:", "Ложь")
