
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline
from config import MODEL_NAME

load_dotenv()

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# HuggingFace pipeline
llm_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=4096,
    max_new_tokens=4096,
    do_sample=False,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
