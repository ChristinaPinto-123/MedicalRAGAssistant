from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class MedicalQueryRequest(BaseModel):
    query: str = Field(
        ...,
        examples=["What is the recommended add-on biologic for severe eosinophilic asthma?"]
    )
    max_evidence_nodes: int = Field(default=3, ge=1, le=10)
    entailment_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

class SourceCitation(BaseModel):
    source_id: str
    doc_type: Literal["Guideline", "RCT", "Systematic Review", "Hospital Protocol"]
    title: str
    doi_or_url: Optional[str] = None
    evidence_level_weight: float = Field(..., ge=0.0, le=1.0)
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
    confidence_score: float = Field(..., description="Weighted composite confidence (0.0 to 1.0)")
    confidence_tier: Literal["High", "Moderate", "Low / Needs Review"]