from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# load from huggingface 
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "Ryuzaki21/llama3.2-text2sql")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

def test(question):
    inputs = tokenizer(
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nConvert to SQL: {question}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("assistant")[-1].strip()

print(test("How many employees are older than 30?"))
