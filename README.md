---
base_model: meta-llama/Llama-3.2-3B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:meta-llama/Llama-3.2-3B
- lora
- sft
- transformers
- trl
datasets:
- anubhutiv1/kanye_west_lyrics
metrics:
- accuracy
---

# Fine-tuned Llama 3.2 3B for Kanye west style bars generation

This model is a fine-tuned version of `meta-llama/Llama-3.2-3B` using QLoRA. It has been adapted for kanye west style bars generation. give a bar it will generate kanye west style follow up bar. 
**Source:** [[llama_kanye_best]](https://huggingface.co/anubhutiv1/llama_kanye_best)


## Model Overview

This model builds upon the capabilities of the original Llama 3.2 3B model, enhancing its performance on a specific domain/task through fine-tuning. It utilizes 4-bit NormalFloat (NF4) quantization for memory efficiency, making it suitable for environments with limited VRAM.

## Training Details

* **Base Model:** `meta-llama/Llama-3.2-3B`
* **Fine-tuning Method:** QLoRA (Quantized Low-Rank Adaptation)
* **Quantization:** 4-bit NF4 with double quantization enabled.
* **Compute DType:** `torch.float16` for computation.
* **Training Hardware:** NVIDIA GeForce RTX 2060 (6GB VRAM)
* **Frameworks:**
    * `transformers` ([link to docs](https://huggingface.co/docs/transformers/))
    * `peft` ([link to docs](https://huggingface.co/docs/peft/))
    * `trl` ([link to docs](https://huggingface.co/docs/trl/))
    * `bitsandbytes` ([link to docs](https://github.com/TimDettmers/bitsandbytes))
* **Dataset:**
    * **Name/Description:** "A collection of rap bars from kanye west songs"
    * **Source:** [[kanye_west_lyrics]](https://huggingface.co/datasets/anubhutiv1/kanye_west_lyrics)
    * **Format:** "JSONL with 'prompt' and 'completion' fields,"
    * **Size:** "3,000 instruction-response pairs."
* **Training Parameters:**
    * `num_train_epochs`: 3
    * `per_device_train_batch_size`: 1
    * `gradient_accumulation_steps`: 8
    * `learning_rate`: 2e-4
    * `max_seq_length`: 256
    * `lora_r`: 8
    * `lora_alpha`: 16
    * `lora_dropout`: 0.05
    * Optimizer: Paged AdamW 8-bit
    * Scheduler: Cosine learning rate scheduler with warmup
    * Seed: 42

## How to Use

This model is a PEFT (LoRA) adapter. To use it, you need to load the original `meta-llama/Llama-3.2-3B` base model and then attach this adapter.

First, ensure you have the necessary libraries installed:
```bash
pip install transformers peft bitsandbytes accelerate trl torch
```

Then, you can load and use the model for bars generation:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, pipeline
import torch
import os

model_name = "meta-llama/Llama-3.2-3B"
OFFLOAD_DIRECTORY = "./model_offload_cache"
os.makedirs(OFFLOAD_DIRECTORY, exist_ok=True)

print(f"Loading base model: {model_name}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)
print("Base model loaded.")

tokenizer_inference = AutoTokenizer.from_pretrained(model_name)
if tokenizer_inference.pad_token is None:
    tokenizer_inference.pad_token = tokenizer_inference.eos_token
tokenizer_inference.padding_side = "right"

print("Loading PEFT adapter and attaching to base model...")
model_inference = PeftModel.from_pretrained(base_model, "./fine_tuned_llama_adapter")
print("PEFT adapter loaded and attached.")

model_inference.eval()

generator = pipeline(
    "text-generation",
    model=model_inference,
    tokenizer=tokenizer_inference,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Bougie girl, grab my hand "

outputs = generator(
    prompt,
    max_new_tokens=100,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    pad_token_id=tokenizer_inference.pad_token_id
)

print(outputs[0]["generated_text"])
```