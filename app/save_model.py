import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# load and save bert model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

torch.save(model, "model/gpt2.pt")
torch.save(tokenizer, "model/gpt2-tokenizer.pt")