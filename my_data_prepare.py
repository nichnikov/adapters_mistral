import os
import pandas as pd
from datasets import DatasetDict
import datasets

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Ты квалифицированный бухгалтер, тебе задают профессиональный вопрос. Используй свои знания и как можно точнее отвечай на вопрос.


### Вопрос:
{data_point["QueryText"]}


### Ответ:
{data_point["Answer"]}
"""
    return full_prompt # tokenize(full_prompt)


df = pd.read_csv(os.path.join("data", "queries2024_short_exmpl1000.csv"), sep="\t")
print(df)
print(df.info())
"QueryText"
"Answer"
df_dicts = df.to_dict(orient="records")
print(df_dicts[0])

for d in df_dicts[:10]:
    print(generate_and_tokenize_prompt(d)) 

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

