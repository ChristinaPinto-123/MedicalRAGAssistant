from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from dotenv import load_dotenv

from app.schemas import MedicalQueryRequest, MedicalQueryResponse
from app.rag_engine import ClinicalRAGEngine
from app.nli_validator import MedicalNLIVerifier

load_dotenv()

pipeline_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize heavy models on startup into memory
    pipeline_state["rag_engine"] = ClinicalRAGEngine()
    pipeline_state["nli_verifier"] = MedicalNLIVerifier()
    yield
    pipeline_state.clear()

app = FastAPI(
    title="Clinical Evidence RAG API",
    description="Asynchronous RAG with automated NLI-based claim validation and evidence ranking.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post(
    "/api/v1/query",
    response_model=MedicalQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute grounded medical query with citation verification"
)
async def query_clinical_rag(request: MedicalQueryRequest):
    rag_engine: ClinicalRAGEngine = pipeline_state.get("rag_engine")
    nli_verifier: MedicalNLIVerifier = pipeline_state.get("nli_verifier")

    if not rag_engine or not nli_verifier:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline engines are not initialized."
        )

    try:
        # Step 1: Retrieval & LLM Generation
        answer, sources, retrieval_sim = rag_engine.retrieve_and_synthesize(
            query_str=request.query,
            top_k=request.max_evidence_nodes
        )

        # Step 2: NLI Verification across claims
        verified_claims, grounding_ratio = nli_verifier.validate_generation(
            synthesized_text=answer,
            sources=sources,
            threshold=request.entailment_threshold
        )

        # Step 3: Composite Confidence Scoring
        # 40% NLI Grounding + 35% Retrieval Relevance + 25% Hierarchy Weight
        mean_evidence_hierarchy = (
            sum(s.evidence_level_weight for s in sources) / max(len(sources), 1)
        )
        composite_confidence = (
            (0.40 * grounding_ratio) +
            (0.35 * retrieval_sim) +
            (0.25 * mean_evidence_hierarchy)
        )

        if composite_confidence >= 0.80:
            tier = "High"
        elif composite_confidence >= 0.60:
            tier = "Moderate"
        else:
            tier = "Low / Needs Review"

        return MedicalQueryResponse(
            synthesized_answer=answer,
            verified_claims=verified_claims,
            sources=sources,
            confidence_score=round(composite_confidence, 4),
            confidence_tier=tier
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical RAG processing error: {str(exc)}"
        )