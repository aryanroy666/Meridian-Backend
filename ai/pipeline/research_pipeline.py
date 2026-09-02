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

# -------------------------------------------------------
# Gemini Retry Configuration
# -------------------------------------------------------

MAX_GEMINI_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds
REQUEST_DELAY = 0.5  # seconds between Gemini requests


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

    # -------------------------------------------------------
    # Generic Gemini Retry Wrapper
    # -------------------------------------------------------

    def _run_with_retry(self, func, stage_name, item_name):
        """
        Executes a Gemini-dependent function with retry logic.

        Retries only on retryable HTTP status codes:
        429, 500, 502, 503, 504.
        """

        retryable_codes = {429, 500, 502, 503, 504}

        for attempt in range(1, MAX_GEMINI_RETRIES + 1):
            try:
                logger.info(
                    f"{stage_name}: Processing {item_name} "
                    f"(Attempt {attempt}/{MAX_GEMINI_RETRIES})"
                )
                return func()

            except (ServerError, ClientError) as e:
                error_code = getattr(e, "code", None)

                if error_code not in retryable_codes:
                    logger.error(
                        f"{stage_name}: Non-retryable Gemini error "
                        f"({error_code}) while processing {item_name}: {e}"
                    )
                    raise

                if attempt == MAX_GEMINI_RETRIES:
                    logger.error(
                        f"{stage_name}: Failed after "
                        f"{MAX_GEMINI_RETRIES} retries for {item_name}"
                    )
                    raise RuntimeError(
                        "Gemini unavailable after multiple retries."
                    ) from e

                wait_time = INITIAL_BACKOFF * (2 ** (attempt - 1))

                logger.warning(
                    f"{stage_name}: Gemini returned HTTP {error_code}. "
                    f"Retrying {item_name} in {wait_time}s "
                    f"({attempt}/{MAX_GEMINI_RETRIES})"
                )

                time.sleep(wait_time)

    # -------------------------------------------------------
    # Main Pipeline
    # -------------------------------------------------------

    def run(self, query: str) -> ResearchResult:
        pipeline_start = time.time()

        try:

            # =====================================================
            # STEP 1 — Planner
            # =====================================================

            logger.info("========== PLANNER STAGE ==========")
            start = time.time()

            tasks: list[ResearchTask] = self.planner.create_plan(query)

            if not tasks:
                raise ValueError("Planner returned no research tasks.")

            logger.info(
                f"Planner generated {len(tasks)} research tasks "
                f"in {time.time() - start:.2f}s."
            )

            # =====================================================
            # STEP 2 — Research
            # =====================================================

            logger.info("========== RESEARCH STAGE ==========")
            start = time.time()

            sources: list[Source] = []

            for task in tasks:
                task_sources = self.researcher.research(task)
                sources.extend(task_sources)

            if not sources:
                raise ValueError("Research agent returned no sources.")

            logger.info(
                f"Research found {len(sources)} sources "
                f"in {time.time() - start:.2f}s."
            )

            # =====================================================
            # STEP 3 — Extraction
            # =====================================================

            logger.info("========== EXTRACTION STAGE ==========")
            start = time.time()

            evidences: list[Evidence] = []

            for index, source in enumerate(sources, start=1):

                source_evidence = self._run_with_retry(
                    lambda s=source: self.extractor.extract(s),
                    stage_name="Extraction",
                    item_name=f"Source {index}/{len(sources)}",
                )

                evidences.extend(source_evidence)

                # Prevent burst requests to Gemini.
                if index < len(sources):
                    time.sleep(REQUEST_DELAY)

            if not evidences:
                raise ValueError("Extraction agent returned no evidence.")

            logger.info(
                f"Extraction produced {len(evidences)} evidence items "
                f"in {time.time() - start:.2f}s."
            )

            # =====================================================
            # STEP 4 — Validation
            # =====================================================

            logger.info("========== VALIDATION STAGE ==========")
            start = time.time()

            # Small delay before sending validation request.
            time.sleep(REQUEST_DELAY)

            validations: list[ValidationResult] = self._run_with_retry(
                lambda: self.validator.validate(
                    evidences=evidences,
                    sources=sources,
                ),
                stage_name="Validation",
                item_name="Evidence Batch",
            )

            if not validations:
                raise ValueError(
                    "Validation agent returned no validation results."
                )

            logger.info(
                f"Validation produced {len(validations)} results "
                f"in {time.time() - start:.2f}s."
            )

            # =====================================================
            # STEP 5 — Citation Builder
            # =====================================================

            logger.info("========== CITATION STAGE ==========")
            start = time.time()

            citations = self.citation_builder.build(sources)

            if not citations:
                raise ValueError("Citation builder returned no citations.")

            logger.info(
                f"Generated {len(citations)} citations "
                f"in {time.time() - start:.2f}s."
            )

            # =====================================================
            # STEP 6 — Report Generation
            # =====================================================

            logger.info("========== REPORT STAGE ==========")
            start = time.time()

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

            report = self._run_with_retry(
                lambda: self.reporter.generate_report(
                    tasks=tasks,
                    evidences=top_evidence,
                    validations=validations,
                    citations=citations,
                ),
                stage_name="Report",
                item_name="Final Report",
            )

            if not report:
                raise ValueError("Report agent returned no report.")

            logger.info(
                f"Report generated in {time.time() - start:.2f}s."
            )

            # =====================================================
            # STEP 7 — Report Linking
            # =====================================================

            logger.info("========== REPORT LINKER STAGE ==========")
            start = time.time()

            linked_report = self.report_linker.link_report(
                report=report,
                evidences=evidences,
                citations=citations,
            )

            if not linked_report:
                raise ValueError(
                    "Report linker returned no linked report."
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

        # -------------------------------------------------------
        # Gemini Retry Failure
        # -------------------------------------------------------

        except RuntimeError:
            raise

        # -------------------------------------------------------
        # Gemini API Errors
        # -------------------------------------------------------

        except (ServerError, ClientError) as e:
            logger.exception("Unhandled Gemini API error.")

            raise RuntimeError(
                "Gemini is temporarily unavailable. Please retry."
            ) from e

        # -------------------------------------------------------
        # Other Pipeline Errors
        # -------------------------------------------------------

        except Exception:
            logger.exception(
                f"Research pipeline failed after "
                f"{time.time() - pipeline_start:.2f}s."
            )
            raise
