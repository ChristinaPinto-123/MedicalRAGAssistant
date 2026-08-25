import os
from typing import List, Tuple
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.schemas import SourceCitation

class ClinicalRAGEngine:
    def __init__(self):
        Settings.llm = OpenAI(
        model="gpt-4o",
        api_base="https://models.inference.ai.azure.com",
        api_key=os.getenv("GITHUB_TOKEN"),
        temperature=0.0
    )
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        
        self.index = self._initialize_mock_kb()
        self.evidence_weights = {
            "Systematic Review": 1.0,
            "Guideline": 0.95,
            "RCT": 0.85,
            "Hospital Protocol": 0.70
        }

    def _initialize_mock_kb(self) -> VectorStoreIndex:
        """Seed index with reference medical evidence."""
        docs = [
            Document(
                text="NICE NG80 Section 1.3: In adults with severe eosinophilic asthma uncontrolled on high-dose ICS plus LABA, consider add-on biologic therapies such as mepolizumab, reslizumab, or benralizumab if blood eosinophils are >= 300 cells/uL.",
                metadata={"source_id": "NICE-NG80", "type": "Guideline", "title": "NICE Asthma Guideline NG80", "doi": "https://www.nice.org.uk/guidance/ng80"}
            ),
            Document(
                text="Cochrane Review CD010834: Anti-IL-5 biologics reduce the rate of severe asthma exacerbations by approximately 50% in patients with severe refractory eosinophilic asthma compared to placebo (N=3,400, High certainty).",
                metadata={"source_id": "Cochrane-CD010834", "type": "Systematic Review", "title": "Anti-IL-5 for severe asthma", "doi": "10.1002/14651858.CD010834"}
            )
        ]
        return VectorStoreIndex.from_documents(docs)

    def retrieve_and_synthesize(self, query: str, top_k: int) -> Tuple[str, List[SourceCitation], float]:
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(query)

        sources: List[SourceCitation] = []
        context_str = ""
        avg_retrieval_score = 0.0

        for node_with_score in retrieved_nodes:
            node = node_with_score.node
            score = node_with_score.score or 0.85
            avg_retrieval_score += score
            
            doc_type = node.metadata.get("type", "Guideline")
            weight = self.evidence_weights.get(doc_type, 0.6)

            src = SourceCitation(
                source_id=node.metadata.get("source_id", "DOC"),
                doc_type=doc_type,
                title=node.metadata.get("title", "Clinical Document"),
                doi_or_url=node.metadata.get("doi"),
                evidence_level_weight=weight,
                snippet=node.text
            )
            sources.append(src)
            context_str += f"[{src.source_id}] {src.snippet}\n\n"

        avg_retrieval_score /= max(len(retrieved_nodes), 1)

        # Enforce strict medical grounding in synthesis prompt
        prompt = (
            "You are a clinical decision assistant. Provide a concise, evidence-based answer "
            "synthesizing ONLY the provided clinical context. Attribute all factual statements "
            "with bracketed source IDs (e.g., [NICE-NG80]). Do NOT hallucinate off-context guidelines.\n\n"
            f"Context:\n{context_str}\n"
            f"Clinical Query: {query}\n"
            "Response:"
        )

        response = Settings.llm.complete(prompt)
        return response.text, sources, avg_retrieval_score