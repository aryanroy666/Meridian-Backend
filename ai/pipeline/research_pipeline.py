import time
import logging

from google.genai.errors import ServerError, ClientError

from ai.planner.planner_agent import PlannerAgent
from ai.research.research_agent import ResearchAgent
from ai.extraction.extraction_agent import ExtractionAgent
from ai.validation.validation_agent import ValidationAgent
from ai.report.report_agent import ReportAgent
from ai.report.citation_builder import CitationBuilder
from ai.report.report_linker import ReportLinker

from ai.schemas.research_result import ResearchResult
from ai.schemas.research_task import ResearchTask
from ai.schemas.source import Source
from ai.schemas.evidence import Evidence
from ai.schemas.validation import ValidationResult

logger = logging.getLogger(__name__)


class ResearchPipeline:

    def __init__(
        self,
        planner=None,
        researcher=None,
        extractor=None,
        validator=None,
        reporter=None,
        citation_builder=None,
        report_linker=None,
    ):
        self.planner = planner or PlannerAgent()
        self.researcher = researcher or ResearchAgent()
        self.extractor = extractor or ExtractionAgent()
        self.validator = validator or ValidationAgent()
        self.reporter = reporter or ReportAgent()
        self.citation_builder = citation_builder or CitationBuilder()
        self.report_linker = report_linker or ReportLinker()

    def run(self, query: str) -> ResearchResult:
        """
        Complete research pipeline.

        Planner
            ↓
        Research
            ↓
        Extraction
            ↓
        Validation
            ↓
        Citation Builder
            ↓
        Report Generator
            ↓
        Report Linker
        """

        pipeline_start = time.time()

        try:
            # -------------------------------------------------------
            # STEP 1 — Planner
            # -------------------------------------------------------
            logger.info("========== PLANNER STAGE ==========")

            start = time.time()

            tasks: list[ResearchTask] = self.planner.create_plan(query)

            if not tasks:
                raise ValueError("Planner returned no research tasks")

            logger.info(
                f"Planner generated {len(tasks)} research tasks "
                f"in {time.time() - start:.2f}s."
            )

            # -------------------------------------------------------
            # STEP 2 — Research
            # -------------------------------------------------------
            logger.info("========== RESEARCH STAGE ==========")

            start = time.time()

            sources: list[Source] = []

            for task in tasks:
                task_sources = self.researcher.research(task)
                sources.extend(task_sources)

            if not sources:
                raise ValueError("Research agent returned no sources")

            logger.info(
                f"Research found {len(sources)} sources "
                f"in {time.time() - start:.2f}s."
            )

            # -------------------------------------------------------
            # STEP 3 — Extraction
            # -------------------------------------------------------
            logger.info("========== EXTRACTION STAGE ==========")

            start = time.time()

            evidences: list[Evidence] = []

            for source in sources:
                source_evidence = self.extractor.extract(source)
                evidences.extend(source_evidence)

            if not evidences:
                raise ValueError("Extraction agent returned no evidence")

            logger.info(
                f"Extraction produced {len(evidences)} evidence items "
                f"in {time.time() - start:.2f}s."
            )

            # -------------------------------------------------------
            # STEP 4 — Validation
            # -------------------------------------------------------
            logger.info("========== VALIDATION STAGE ==========")

            start = time.time()

            validations: list[ValidationResult] = self.validator.validate(
                evidences=evidences,
                sources=sources,
            )

            if not validations:
                raise ValueError(
                    "Validation agent returned no validation results"
                )

            logger.info(
                f"Validation produced {len(validations)} results "
                f"in {time.time() - start:.2f}s."
            )

            # -------------------------------------------------------
            # STEP 5 — Citation Builder
            # -------------------------------------------------------
            logger.info("========== CITATION STAGE ==========")

            start = time.time()

            citations = self.citation_builder.build(sources)

            if not citations:
                raise ValueError("Citation builder returned no citations")

            logger.info(
                f"Generated {len(citations)} citations "
                f"in {time.time() - start:.2f}s."
            )

            # -------------------------------------------------------
            # STEP 6 — Report Generation
            # -------------------------------------------------------
            logger.info("========== REPORT STAGE ==========")

            start = time.time()

            # Reduce prompt size by selecting the most relevant evidence
            if len(evidences) > 20:
                try:
                    top_evidence = sorted(
                        evidences,
                        key=lambda x: getattr(x, "relevance_score", 0),
                        reverse=True,
                    )[:20]
                except Exception:
                    top_evidence = evidences[:20]
            else:
                top_evidence = evidences

            report = self.reporter.generate_report(
                tasks=tasks,
                evidences=top_evidence,
                validations=validations,
                citations=citations,
            )

            if not report:
                raise ValueError("Report agent returned no report")

            logger.info(
                f"Report generated in {time.time() - start:.2f}s."
            )

            # -------------------------------------------------------
            # STEP 7 — Report Linking
            # -------------------------------------------------------
            logger.info("========== REPORT LINKER STAGE ==========")

            start = time.time()

            linked_report = self.report_linker.link_report(
                report=report,
                evidences=evidences,
                citations=citations,
            )

            if not linked_report:
                raise ValueError(
                    "Report linker returned no linked report"
                )

            logger.info(
                f"Linked {len(linked_report.key_findings)} key findings "
                f"in {time.time() - start:.2f}s."
            )

            logger.info(
                f"Pipeline completed successfully in "
                f"{time.time() - pipeline_start:.2f}s."
            )

            return ResearchResult(
                report=report,
                linked_report=linked_report,
                tasks=tasks,
                sources=sources,
                evidences=evidences,
                validations=validations,
            )

        # Gemini temporary failures
        except (ServerError, ClientError) as e:
            logger.error(f"Gemini API Error: {e}")
            raise RuntimeError(
                "Gemini is temporarily unavailable. Please retry."
            ) from e

        # Any other pipeline failure
        except Exception as e:
            logger.exception(
                f"Research pipeline failed after "
                f"{time.time() - pipeline_start:.2f}s."
            )
            raise
