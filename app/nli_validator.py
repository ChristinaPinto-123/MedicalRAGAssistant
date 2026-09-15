import re
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.schemas import VerifiedClaim, SourceCitation

class MedicalNLIVerifier:
    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def decompose_claims(self, text: str) -> List[str]:
        """Splits synthesized clinical text into distinct atomic sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def verify_claim(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Classifies relation between retrieved clinical context and generated claim."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Class indexing for MoritzLaurer MNLI models:
        # Index 0: Entailment, Index 1: Neutral, Index 2: Contradiction
        entailment = probs[0].item()
        neutral = probs[1].item()
        contradiction = probs[2].item()

        if entailment >= 0.70:
            return "Entailed", float(entailment)
        elif contradiction >= 0.40:
            return "Contradicted", float(contradiction)
        else:
            return "Neutral / Unverified", float(neutral)

    def validate_generation(
        self,
        synthesized_text: str,
        sources: List[SourceCitation],
        threshold: float
    ) -> Tuple[List[VerifiedClaim], float]:
        """Validates all synthesized claims against retrieved citation snippets."""
        claims = self.decompose_claims(synthesized_text)
        combined_premise = " ".join([f"[{s.source_id}] {s.snippet}" for s in sources])

        verified_claims: List[VerifiedClaim] = []
        entailed_count = 0

        for claim in claims:
            status, score = self.verify_claim(combined_premise, claim)

            # Match explicitly referenced source IDs if present
            matched_id = "General Context"
            for src in sources:
                if src.source_id in claim:
                    matched_id = src.source_id
                    break

            if status == "Entailed" and score >= threshold:
                entailed_count += 1

            verified_claims.append(
                VerifiedClaim(
                    claim_text=claim,
                    cited_source_id=matched_id,
                    nli_status=status,
                    entailment_score=round(score, 4)
                )
            )

        grounding_ratio = entailed_count / max(len(claims), 1)
        return verified_claims, grounding_ratio