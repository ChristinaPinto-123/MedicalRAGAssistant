from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class MedicalQueryRequest(BaseModel):
    query: str = Field(..., example="What is the first-line therapy for severe eosinophilic asthma?")
    max_evidence_nodes: int = Field(default=5, ge=1, le=10)
    entailment_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

class SourceCitation(BaseModel):
    source_id: str
    doc_type: Literal["Guideline", "RCT", "Systematic Review", "Hospital Protocol"]
    title: str
    doi_or_url: Optional[str] = None
    evidence_level_weight: float = Field(..., description="Weight from 0.4 to 1.0 based on evidence hierarchy")
    snippet: str

class VerifiedClaim(BaseModel):
    claim_text: str
    cited_source_id: str
    nli_status: Literal["Entailed", "Contradicted", "Neutral / Unverified"]
    entailment_score: float

class MedicalQueryResponse(BaseModel):
    synthesized_answer: str
    verified_claims: List[VerifiedClaim]
    sources: List[SourceCitation]
    confidence_score: float = Field(..., description="Calculated composite confidence (0.0 - 1.0)")
    confidence_tier: Literal["High", "Moderate", "Low / Needs Review"]