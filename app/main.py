from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from app.schemas import MedicalQueryRequest, MedicalQueryResponse
from app.rag_engine import ClinicalRAGEngine
from app.nli_validator import MedicalNLIVerifier

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load embedding, LLM settings, and NLI models on startup
    state["rag_engine"] = ClinicalRAGEngine()
    state["nli_verifier"] = MedicalNLIVerifier()
    yield
    state.clear()

app = FastAPI(
    title="Clinical Evidence RAG Engine",
    description="Evidence-grounded medical question answering with NLI-based claim attribution and confidence scoring",
    version="1.0.0",
    lifespan=lifespan
)

def compute_confidence(
    retrieval_score: float,
    grounding_ratio: float,
    sources: list
) -> tuple[float, str]:
    """Calculates weighted composite confidence score."""
    avg_evidence_weight = (
        sum(s.evidence_level_weight for s in sources) / len(sources) if sources else 0.5
    )
    
    # Weights: 30% retrieval similarity, 50% NLI claim grounding, 20% evidence hierarchy
    composite = (0.30 * retrieval_score) + (0.50 * grounding_ratio) + (0.20 * avg_evidence_weight)
    composite = round(min(max(composite, 0.0), 1.0), 3)

    if composite >= 0.85:
        tier = "High"
    elif composite >= 0.65:
        tier = "Moderate"
    else:
        tier = "Low / Needs Review"

    return composite, tier

@app.post(
    "/api/v1/query",
    response_model=MedicalQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Medical Knowledge Base"
)
async def query_medical_rag(payload: MedicalQueryRequest):
    try:
        rag: ClinicalRAGEngine = state["rag_engine"]
        nli: MedicalNLIVerifier = state["nli_verifier"]

        # 1. Retrieve & Synthesize
        raw_answer, sources, retrieval_score = rag.retrieve_and_synthesize(
            query=payload.query,
            top_k=payload.max_evidence_nodes
        )

        # 2. Deconstruct and Validate Claims via NLI
        verified_claims, grounding_ratio = nli.validate_generation(
            synthesized_text=raw_answer,
            sources=sources,
            threshold=payload.entailment_threshold
        )

        # 3. Formulate Composite Score
        confidence_score, confidence_tier = compute_confidence(
            retrieval_score=retrieval_score,
            grounding_ratio=grounding_ratio,
            sources=sources
        )

        return MedicalQueryResponse(
            synthesized_answer=raw_answer,
            verified_claims=verified_claims,
            sources=sources,
            confidence_score=confidence_score,
            confidence_tier=confidence_tier
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Medical RAG pipeline failure: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)