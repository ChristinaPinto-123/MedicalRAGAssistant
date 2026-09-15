import os
from typing import List, Tuple
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.schemas import SourceCitation

class ClinicalRAGEngine:
    def __init__(self):
        # OpenAI or GitHub Models compatible endpoint setup
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN")
        api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            raise ValueError("No API token provided. Set OPENAI_API_KEY or GITHUB_TOKEN in your .env.")

        Settings.llm = OpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o"),
            api_key=api_key,
            api_base=api_base,
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

    def _initialize_knowledge_base(self) -> VectorStoreIndex:
        """Seeds standard clinical guidelines for decision support."""
        documents = [
            Document(
                text=(
                    "NICE NG80 Section 1.3: In adults with severe eosinophilic asthma uncontrolled "
                    "on high-dose inhaled corticosteroids (ICS) plus long-acting beta2-agonists (LABA), "
                    "add-on biologic therapies targeting IL-5 (mepolizumab, reslizumab) or IL-5R "
                    "(benralizumab) are recommended if blood eosinophil counts exceed 300 cells/uL."
                ),
                metadata={
                    "source_id": "NICE-NG80",
                    "doc_type": "Guideline",
                    "title": "NICE Asthma: Diagnosis, Monitoring and Chronic Management (NG80)",
                    "url": "https://www.nice.org.uk/guidance/ng80"
                }
            ),
            Document(
                text=(
                    "Cochrane Review 2022: Anti-IL-5 monoclonal antibodies reduce severe exacerbation "
                    "rates by approximately 50% in patients with severe refractory eosinophilic asthma "
                    "compared to standard care placebo."
                ),
                metadata={
                    "source_id": "Cochrane-CD010834",
                    "doc_type": "Systematic Review",
                    "title": "Anti-IL-5 therapies for severe asthma",
                    "url": "https://doi.org/10.1002/14651858.CD010834.pub4"
                }
            )
        ]
        return VectorStoreIndex.from_documents(documents)

    def retrieve_and_synthesize(self, query_str: str, top_k: int = 3) -> Tuple[str, List[SourceCitation], float]:
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query_str)

        sources: List[SourceCitation] = []
        context_lines: List[str] = []
        similarity_scores: List[float] = []

        for node_with_score in nodes:
            meta = node_with_score.node.metadata
            doc_type = meta.get("doc_type", "Hospital Protocol")
            score = node_with_score.score if node_with_score.score is not None else 0.80
            similarity_scores.append(score)

            sources.append(
                SourceCitation(
                    source_id=meta.get("source_id", "GEN-01"),
                    doc_type=doc_type,
                    title=meta.get("title", "Clinical Evidence"),
                    doi_or_url=meta.get("url"),
                    evidence_level_weight=self.evidence_weights.get(doc_type, 0.70),
                    snippet=node_with_score.node.get_content().strip()
                )
            )
            context_lines.append(f"[{meta.get('source_id')}] {node_with_score.node.get_content().strip()}")

        context_str = "\n\n".join(context_lines)
        prompt = (
            "You are an evidence-based clinical decision-support AI. Answer the medical question "
            "strictly using the retrieved evidence below. You MUST cite the source ID in square brackets "
            "(e.g., [NICE-NG80]) for every factual claim made. If the context does not contain the answer, "
            "state that there is insufficient evidence.\n\n"
            f"Evidence:\n{context_str}\n\n"
            f"Query: {query_str}\n\n"
            "Clinical Synthesis:"
        )

        response = Settings.llm.complete(prompt)
        mean_retrieval_similarity = sum(similarity_scores) / max(len(similarity_scores), 1)

        return str(response), sources, mean_retrieval_similarity