import json
import logging

from ai.llm.gemini import GeminiLLM
from ai.schemas.evidence import Evidence
from ai.schemas.source import Source
from ai.schemas.validation import ValidationResult

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Validation Configuration
# -------------------------------------------------------

VALIDATION_BATCH_SIZE = 15


class ValidationAgent:

    def __init__(self, llm=None):
        self.llm = llm or GeminiLLM()

    # -------------------------------------------------------
    # Public Validation Method
    # -------------------------------------------------------

    def validate(
        self,
        evidences: list[Evidence],
        sources: list[Source],
    ) -> list[ValidationResult]:
        """
        Validate evidence in batches to reduce Gemini token usage.
        """

        if not evidences:
            return []

        source_map = {
            source.source_id: source
            for source in sources
        }

        results = []

        for start in range(0, len(evidences), VALIDATION_BATCH_SIZE):

            batch = evidences[start:start + VALIDATION_BATCH_SIZE]

            logger.info(
                "Validating evidence batch %d-%d of %d",
                start + 1,
                start + len(batch),
                len(evidences),
            )

            batch_results = self._validate_batch(batch, source_map)
            results.extend(batch_results)

        logger.info(
            "Validation completed: %d validation results.",
            len(results),
        )

        return results

    # -------------------------------------------------------
    # Batch Validation
    # -------------------------------------------------------

    def _validate_batch(
        self,
        evidences: list[Evidence],
        source_map: dict,
    ) -> list[ValidationResult]:

        evidence_payload = []

        for evidence in evidences:

            source = source_map.get(evidence.source_id)

            evidence_payload.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "claim": evidence.claim,
                    "excerpt": evidence.excerpt,
                    "entity": evidence.entity,
                    "topic": evidence.topic,
                    "source_title": source.title if source else "",
                    "publisher": source.publisher if source else "",
                    "published_date": (
                        source.published_date if source else ""
                    ),
                }
            )

        prompt = f"""
You are a research evidence validator.

Evaluate each evidence item using ONLY the supplied excerpt.

Return ONLY valid JSON.

Evidence:
{json.dumps(evidence_payload, indent=2)}

Return:

[
  {{
    "evidence_id": "...",
    "is_valid": true,
    "credibility_score": 0.85,
    "recency_score": 0.90,
    "is_duplicate": false,
    "has_conflict": false,
    "reason": "One short sentence."
  }}
]

Rules:
- One result per evidence item.
- Use only provided information.
- credibility_score and recency_score between 0 and 1.
- Keep reason under 20 words.
- Return JSON only.
"""

        response = self.llm.generate(prompt)

        cleaned_response = self._clean_response(response)

        try:
            data = json.loads(cleaned_response)

        except json.JSONDecodeError as e:
            logger.error("Gemini returned invalid validation JSON.")
            logger.error(cleaned_response)

            raise ValueError(
                "Validation agent returned invalid JSON."
            ) from e

        if not isinstance(data, list):
            raise ValueError(
                "Validation agent expected a JSON array."
            )

        results = []

        for item in data:

            if not isinstance(item, dict):
                continue

            required_fields = [
                "evidence_id",
                "is_valid",
                "credibility_score",
                "recency_score",
                "is_duplicate",
                "has_conflict",
                "reason",
            ]

            if any(field not in item for field in required_fields):
                logger.warning(
                    "Skipping malformed validation item: %s",
                    item,
                )
                continue

            try:
                credibility = max(
                    0.0,
                    min(float(item["credibility_score"]), 1.0),
                )

                recency = max(
                    0.0,
                    min(float(item["recency_score"]), 1.0),
                )

            except Exception:
                continue

            results.append(
                ValidationResult(
                    evidence_id=str(item["evidence_id"]).strip(),
                    is_valid=bool(item["is_valid"]),
                    credibility_score=credibility,
                    recency_score=recency,
                    is_duplicate=bool(item["is_duplicate"]),
                    has_conflict=bool(item["has_conflict"]),
                    reason=str(item["reason"]).strip()[:120],
                )
            )

        if not results:
            raise ValueError(
                "Validation agent returned no valid validation results."
            )

        return results

    # -------------------------------------------------------
    # Response Cleaner
    # -------------------------------------------------------

    @staticmethod
    def _clean_response(response: str) -> str:
        """
        Remove Markdown fences and extract JSON array.
        """

        cleaned = response.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        start = cleaned.find("[")
        end = cleaned.rfind("]")

        if start != -1 and end != -1 and start < end:
            cleaned = cleaned[start:end + 1]

        return cleaned
