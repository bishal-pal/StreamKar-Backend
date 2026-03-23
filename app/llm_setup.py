import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .settings import DEVICE


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=None,
    dtype=torch.float16 if DEVICE == "mps" else torch.float32,
)
model.to(DEVICE)

model.generation_config.max_new_tokens = 150
model.generation_config.temperature = 0.1
model.generation_config.do_sample = False
model.generation_config.repetition_penalty = 1.1
model.generation_config.top_k = 40
model.generation_config.top_p = 0.7

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)
