import json
import logging

from ai.llm.gemini import GeminiLLM
from ai.schemas.evidence import Evidence
from ai.schemas.validation import ValidationResult
from ai.schemas.research_task import ResearchTask
from ai.schemas.report import Report

logger = logging.getLogger(__name__)

# Maximum evidence items sent to Gemini
MAX_REPORT_EVIDENCE = 15


class ReportAgent:

    def __init__(self, llm=None):
        self.llm = llm or GeminiLLM()

    # -------------------------------------------------------
    # Generate Final Research Report
    # -------------------------------------------------------

    def generate_report(
        self,
        tasks: list[ResearchTask],
        evidences: list[Evidence],
        validations: list[ValidationResult],
        citations,
    ) -> Report:

        if not evidences:
            raise ValueError(
                "No evidence available for report generation."
            )

        validation_map = {
            validation.evidence_id: validation
            for validation in validations
        }

        report_evidence = []

        # Keep only validated evidence and trim fields
        for evidence in evidences:

            validation = validation_map.get(evidence.evidence_id)

            if validation and not validation.is_valid:
                continue

            report_evidence.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "claim": evidence.claim,
                    "excerpt": evidence.excerpt,
                    "source_title": evidence.source_id,
                    "credibility_score": (
                        validation.credibility_score
                        if validation
                        else evidence.relevance_score
                    ),
                    "recency_score": (
                        validation.recency_score
                        if validation
                        else 0.5
                    ),
                }
            )

        if not report_evidence:
            raise ValueError(
                "No validated evidence available for report generation."
            )

        # Sort by credibility and keep only top evidence
        report_evidence.sort(
            key=lambda x: (
                x["credibility_score"],
                x["recency_score"],
            ),
            reverse=True,
        )

        report_evidence = report_evidence[:MAX_REPORT_EVIDENCE]

        logger.info(
            f"Generating report using {len(report_evidence)} evidence items."
        )

        prompt = f"""
You are a senior McKinsey strategy research analyst.

Create a professional research report using ONLY the supplied research tasks and validated evidence.

RESEARCH TASKS:
{json.dumps([task.model_dump() for task in tasks], indent=2)}

VALIDATED EVIDENCE:
{json.dumps(report_evidence, indent=2)}

Return ONLY valid JSON.

Required structure:

{{
  "title":"Research report title",
  "executive_summary":"Concise executive summary.",
  "key_findings":[
    {{
      "text":"Finding",
      "evidence_ids":["evidence_001"]
    }}
  ],
  "market_signals":[
    {{
      "text":"Market signal",
      "evidence_ids":["evidence_001"]
    }}
  ],
  "competitor_observations":[
    {{
      "text":"Observation",
      "evidence_ids":["evidence_001"]
    }}
  ],
  "implications":[
    {{
      "text":"Implication",
      "evidence_ids":["evidence_001"]
    }}
  ],
  "recommendations":[
    {{
      "text":"Recommendation",
      "evidence_ids":["evidence_001"]
    }}
  ],
  "evidence_appendix":[
    "evidence_001"
  ]
}}

Rules:
- Use only supplied evidence.
- Do not invent statistics or facts.
- Every finding/recommendation must reference evidence_ids.
- Return JSON only.
"""

        response = self.llm.generate(prompt)

        cleaned_response = self._clean_response(response)

        try:
            data = json.loads(cleaned_response)

            report = Report.model_validate(data)

            report.citations = citations

            logger.info("Research report generated successfully.")

            return report

        except Exception as e:
            logger.error("Invalid report returned by Gemini.")
            logger.error(cleaned_response)

            raise ValueError(
                "Report agent returned invalid report."
            ) from e

    # -------------------------------------------------------
    # Clean Gemini Response
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

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

        return cleaned
