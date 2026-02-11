"""
Docstring for L2.l2_doc_processing
"""

import os
from PIL import Image
import pytesseract
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

def llm():
    llm = ChatOpenAI(
        openai_api_base="https://router.huggingface.co/v1",
        model="HuggingFaceTB/SmolLM3-3B:hf-inference",
        openai_api_key=os.environ["HF_TOKEN"],  # Important: add this
        temperature=1,
    )
    return llm





if __name__ == "__main__":
    

