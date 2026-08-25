import torch
import spacy
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.schemas import VerifiedClaim, SourceCitation

class MedicalNLIVerifier:
    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # SpaCy for sentence/claim extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def decompose_claims(self, text: str) -> List[str]:
        """Splits synthesized clinical text into atomic sentences/claims."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

    def verify_claim(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Runs NLI inference: returns label and entailment probability."""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        # DeBERTa MNLI class mapping: 0 -> Entailment, 1 -> Neutral, 2 -> Contradiction
        entailment_prob = probs[0].item()
        contradiction_prob = probs[2].item()

        if entailment_prob >= 0.70:
            return "Entailed", float(entailment_prob)
        elif contradiction_prob >= 0.50:
            return "Contradicted", float(contradiction_prob)
        else:
            return "Neutral / Unverified", float(probs[1].item())

    def validate_generation(
        self,
        synthesized_text: str,
        sources: List[SourceCitation],
        threshold: float
    ) -> Tuple[List[VerifiedClaim], float]:
        """Validates all claims against aggregated source evidence."""
        claims = self.decompose_claims(synthesized_text)
        combined_premise = " ".join([f"[{s.source_id}] {s.snippet}" for s in sources])
        
        verified_claims: List[VerifiedClaim] = []
        entailed_count = 0

        for claim in claims:
            # Map claim to best source or overall retrieved context
            status, score = self.verify_claim(combined_premise, claim)
            
            # Simple heuristic: find if a specific citation ID is mentioned
            cited_id = "General Context"
            for src in sources:
                if src.source_id in claim:
                    cited_id = src.source_id
                    break

            if status == "Entailed" and score >= threshold:
                entailed_count += 1

            verified_claims.append(
                VerifiedClaim(
                    claim_text=claim,
                    cited_source_id=cited_id,
                    nli_status=status,
                    entailment_score=round(score, 4)
                )
            )

        grounding_ratio = entailed_count / max(len(claims), 1)
        return verified_claims, grounding_ratio