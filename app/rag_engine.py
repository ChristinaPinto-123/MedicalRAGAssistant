import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.schemas import SourceCitation

class ClinicalRAGEngine:
    def __init__(self):
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN is missing from your .env file!")

        # Configure OpenAI provider for GitHub Models
        Settings.llm = OpenAI(
            model="gpt-4o",
            api_base="https://models.inference.ai.azure.com",
            api_key=token,
            temperature=0.0,
            max_retries=3,
            timeout=60.0
        )
        
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        
        self.evidence_weights = {
            "Systematic Review": 1.0,
            "Guideline": 0.95,
            "RCT": 0.85,
            "Hospital Protocol": 0.70
        }
        self.index = self._initialize_mock_kb()

    def _initialize_mock_kb(self) -> VectorStoreIndex:
        docs = [
            Document(
                text="NICE NG80 Section 1.3: In adults with severe eosinophilic asthma uncontrolled on high-dose inhaled corticosteroids (ICS) plus long-acting beta-agonists (LABA), add-on biologic therapies targeting IL-5 (mepolizumab, reslizumab) or IL-5R (benralizumab) are recommended if blood eosinophil count is >= 300 cells/uL.",
                metadata={"source_id": "NICE-NG80", "doc_type": "Guideline", "title": "NICE Asthma Guideline NG80", "url": "https://www.nice.org.uk/guidance/ng80"}
            ),
            Document(
                text="Cochrane Review 2022: Anti-IL-5 monoclonal antibodies reduce severe asthma exacerbation rates by 50% in patients with severe refractory eosinophilic asthma compared with placebo.",
                metadata={"source_id": "Cochrane-CD010834", "doc_type": "Systematic Review", "title": "Anti-IL-5 therapies for asthma", "url": "https://doi.org/10.1002/14651858.CD010834.pub4"}
            )
        ]
        return VectorStoreIndex.from_documents(docs)

    def query(self, query_str: str, top_k: int = 5):
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query_str)

        sources = []
        context_blocks = []
        for n in nodes:
            meta = n.node.metadata
            doc_type = meta.get("doc_type", "Hospital Protocol")
            sources.append(
                SourceCitation(
                    source_id=meta.get("source_id", "DOC-001"),
                    doc_type=doc_type,
                    title=meta.get("title", "Clinical Evidence"),
                    doi_or_url=meta.get("url"),
                    evidence_level_weight=self.evidence_weights.get(doc_type, 0.70),
                    snippet=n.node.get_content()
                )
            )
            context_blocks.append(f"[{meta.get('source_id')}] {n.node.get_content()}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"You are a clinical decision support assistant. Synthesize a concise answer to the question using ONLY the provided evidence.\n"
            f"Cite source IDs in square brackets (e.g. [NICE-NG80]) for each claim.\n\n"
            f"Evidence:\n{context_str}\n\n"
            f"Question: {query_str}\n\nAnswer:"
        )

        response = Settings.llm.complete(prompt)
        return str(response), sources