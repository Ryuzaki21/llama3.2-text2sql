# spider dataset is already on hf 
# trained on kaggle t4 

import os
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset, load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

login(token="YOUR_HF_TOKEN")

# spider has train and validation split built in
train_dataset = load_dataset("spider", split="train")
eval_dataset = load_dataset("spider", split="validation")
print(f"train: {len(train_dataset)}, eval: {len(eval_dataset)}")

# llama needs this exact format to know when to stop
def format_sample(sample):
    user = f"Convert to SQL: {sample['question']}"
    assistant = sample['query']
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{assistant}\n<|eot_id|>"

train_data = []
for sample in train_dataset:
    train_data.append({"text": format_sample(sample)})

eval_data = []
for sample in eval_dataset:
    eval_data.append({"text": format_sample(sample)})

hf_train = Dataset.from_list(train_data)
hf_eval = Dataset.from_list(eval_data)
print(f"done formatting")

# 4bit to fit on t4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

# r=8 worked well, didn't bother trying higher
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# batch 2 is max on t4 without oom
training_args = TrainingArguments(
    output_dir="./text2sql",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    bf16=False,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_eval
)

print("starting training...")
trainer.train()

model.save_pretrained("./text2sql-model")
tokenizer.save_pretrained("./text2sql-model")
print("saved!")

model.push_to_hub("Ryuzaki21/llama3.2-text2sql")
tokenizer.push_to_hub("Ryuzaki21/llama3.2-text2sql")
print("pushed!")
