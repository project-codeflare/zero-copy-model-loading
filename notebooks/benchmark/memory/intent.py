import transformers
import torch

# Simple benchmark to measure peak memory footprint of a PyTorch
# model.
text = ("I came here to eat chips and beat you up, "
        "and I'm all out of chips.")
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            'mrm8488/t5-base-finetuned-e2m-intent')
tokenizer = transformers.AutoTokenizer.from_pretrained('t5-base')

with torch.inference_mode():
    model_input = tokenizer(text, return_tensors='pt')
    result = model.generate(**model_input)
    print(result)

