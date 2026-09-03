import json
import logging

from ai.browser.tavily_search import TavilySearchEngine
from ai.llm.gemini import GeminiLLM
from ai.schemas.evidence import Evidence
from ai.schemas.source import Source

logger = logging.getLogger(__name__)

# Limit content sent to Gemini to reduce token usage
MAX_CONTENT_LENGTH = 4000


class ExtractionAgent:

    def __init__(self, llm=None, browser=None):
        self.llm = llm or GeminiLLM()
        self.browser = browser or TavilySearchEngine()

    # -------------------------------------------------------
    # Extract evidence from a single source
    # -------------------------------------------------------

    def extract(self, source: Source) -> list[Evidence]:

        extracted = self.browser.extract([source.url])

        if not extracted:
            return []

        content = extracted[0].get("raw_content", "")

        if not content:
            return []

        # Reduce token usage
        content = content[:MAX_CONTENT_LENGTH]

        prompt = f"""
You are an evidence extraction agent.

Analyze the following source.

SOURCE ID:
{source.source_id}

SOURCE TITLE:
{source.title}

SOURCE URL:
{source.url}

CONTENT:
{content}

Extract ONLY the 3 most important factual claims supported by this content.

Return ONLY valid JSON.

[
  {{
    "claim": "Short factual claim",
    "excerpt": "Short supporting excerpt",
    "entity": "Main entity",
    "topic": "Research topic",
    "relevance_score": 0.95
  }}
]

Rules:
- Maximum 3 evidence items.
- Excerpts must come directly from the content.
- Do not invent facts.
- relevance_score between 0 and 1.
- Return only JSON.
"""

        response = self.llm.generate(prompt)

        if not response:
            raise ValueError("Extraction agent received an empty response.")

        cleaned_response = self._clean_response(response)

        try:
            data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error("Invalid Gemini JSON response.")
            logger.error(cleaned_response)
            raise ValueError("Extraction agent returned invalid JSON.") from e

        if not isinstance(data, list):
            raise ValueError("Extraction agent expected a JSON array.")

        evidences = []

        required_fields = [
            "claim",
            "excerpt",
            "entity",
            "topic",
            "relevance_score",
        ]

        for index, item in enumerate(data, start=1):

            if not isinstance(item, dict):
                continue

            if any(field not in item for field in required_fields):
                continue

            try:
                score = float(item["relevance_score"])
            except Exception:
                continue

            score = max(0.0, min(score, 1.0))

            evidences.append(
                Evidence(
                    evidence_id=f"{source.source_id}_evidence_{index:03d}",
                    claim=item["claim"].strip(),
                    excerpt=item["excerpt"].strip(),
                    entity=item["entity"].strip(),
                    topic=item["topic"].strip(),
                    relevance_score=score,
                    source_id=source.source_id,
                )
            )

        if not evidences:
            raise ValueError("Extraction agent returned no valid evidence.")

        logger.info(
            "Extraction completed: %d evidence items from source %s.",
            len(evidences),
            source.source_id,
        )

        return evidences

    # -------------------------------------------------------
    # Clean Gemini response
    # -------------------------------------------------------

    @staticmethod
    def _clean_response(response: str) -> str:

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

        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

        return cleaned
