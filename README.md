# llama3.2-text2sql — Fine-tuned Llama 3.2 3B on Spider Dataset

Fine-tuned Llama 3.2 3B to convert natural language questions to SQL queries.

## Overview

Trained on the Spider dataset — 7000 samples for training and 1034 for evaluation. The model takes a natural language question as input and outputs a SQL query.

## Example

Input:
```
How many employees are older than 30?
```

Output:
```sql
SELECT count(*) FROM Employee WHERE age > 30
```

## Tech Stack

| Component | Details |
|-----------|---------|
| Base model | meta-llama/Llama-3.2-3B |
| Training | SFT + QLoRA, 4-bit quantization, r=8 |
| Dataset | Spider (7000 train, 1034 eval) |
| Tools | transformers, peft, trl, bitsandbytes |

## Project Structure
```
text2sql/
├── train.py       # fine-tuning on spider dataset
├── test.py        # test the model
└── requirements.txt
```

## Getting Started
```bash
git clone https://github.com/Ryuzaki21/text2sql.git
cd text2sql
pip install -r requirements.txt
```

## How it works

1. Loaded Spider dataset directly from HuggingFace
2. Fine-tuned Llama 3.2 3B on Kaggle T4 GPU using QLoRA
3. Only 0.07% of parameters trained

## Links

[Model](https://huggingface.co/Ryuzaki21/llama3.2-text2sql)

## License

MIT
