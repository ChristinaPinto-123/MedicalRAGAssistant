import os
from dotenv import load_dotenv
load_dotenv()

from typing import List, Tuple
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.schemas import SourceCitation

class ClinicalRAGEngine:
    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable is missing from .env")

        # Route OpenAI provider directly to the Azure/GitHub Models endpoint
        Settings.llm = OpenAI(
            model="gpt-4o",
            api_key=token,
            api_base="https://models.inference.ai.azure.com",
            temperature=0.0
        )
        
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

        self.evidence_weights = {
            "Systematic Review": 1.0,
            "Guideline": 0.95,
            "RCT": 0.85,
            "Hospital Protocol": 0.70
        }
        self.index = self._initialize_knowledge_base()