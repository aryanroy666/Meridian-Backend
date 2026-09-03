import time
import logging

from ai.pipeline.research_pipeline import ResearchPipeline
from ai.schemas.research_result import ResearchResult

from backend.repositories.planner_task_repository import PlannerTaskRepository
from backend.repositories.source_repository import SourceRepository
from backend.repositories.evidence_repository import EvidenceRepository
from backend.repositories.validation_repository import ValidationRepository
from backend.repositories.report_repository import ReportRepository

logger = logging.getLogger(__name__)


class ResearchService:
    def __init__(
        self,
        pipeline=None,
        planner_task_repository=None,
        source_repository=None,
        evidence_repository=None,
        validation_repository=None,
        report_repository=None,
    ):
        self.pipeline = pipeline or ResearchPipeline()

        self.planner_task_repository = (
            planner_task_repository or PlannerTaskRepository()
        )

        self.source_repository = (
            source_repository or SourceRepository()
        )

        self.evidence_repository = (
            evidence_repository or EvidenceRepository()
        )

        self.validation_repository = (
            validation_repository or ValidationRepository()
        )

        self.report_repository = (
            report_repository or ReportRepository()
        )

    # ==========================================================
    # Run Complete Research Pipeline
    # ==========================================================

    def run_research(
        self,
        query: str,
        job_id: str,
    ) -> ResearchResult:

        start_time = time.time()

        # ------------------------------------------------------
        # Validate Input
        # ------------------------------------------------------

        if not query or not query.strip():
            raise ValueError("Research query cannot be empty.")

        query = query.strip()

        logger.info("=" * 60)
        logger.info("Starting research job: %s", job_id)
        logger.info("Research query: %s", query)
        logger.info("=" * 60)

        # ------------------------------------------------------
        # Execute AI Pipeline
        # ------------------------------------------------------

        try:
            result = self.pipeline.run(query)

        # Gemini unavailable after retries
        except RuntimeError as e:

            elapsed = time.time() - start_time

            logger.error(
                "Research job %s failed after %.2fs.",
                job_id,
                elapsed,
            )

            logger.error("Gemini is temporarily unavailable.")
            logger.error(str(e))

            # Return a clean error to the API layer.
            raise RuntimeError(
                "Gemini is temporarily unavailable after multiple retries. Please retry in a few minutes."
            ) from e

        # Any unexpected pipeline failure
        except Exception as e:

            elapsed = time.time() - start_time

            logger.exception(
                "Unexpected pipeline error for research job %s after %.2fs.",
                job_id,
                elapsed,
            )

            raise RuntimeError(
                f"Research pipeline failed: {str(e)}"
            ) from e

        # ------------------------------------------------------
        # Save Planner Tasks
        # ------------------------------------------------------

        task_count = 0

        for task in result.tasks:
            self.planner_task_repository.create_task(
                job_id=job_id,
                task_type=task.purpose,
                query=task.query,
            )
            task_count += 1

        logger.info("Saved %s planner tasks.", task_count)

        # ------------------------------------------------------
        # Save Sources
        # ------------------------------------------------------

        source_map = {}

        for source in result.sources:
            db_source = self.source_repository.create_source(
                job_id=job_id,
                url=source.url,
                title=source.title,
            )

            source_map[source.source_id] = db_source

        logger.info("Saved %s sources.", len(source_map))

        # ------------------------------------------------------
        # Save Evidence
        # ------------------------------------------------------

        evidence_map = {}

        for evidence in result.evidences:

            db_source = source_map.get(evidence.source_id)

            if not db_source:
                logger.warning(
                    "Skipping evidence %s because source was not found.",
                    evidence.evidence_id,
                )
                continue

            db_evidence = self.evidence_repository.create_evidence(
                job_id=job_id,
                source_id=db_source["id"],
                claim=evidence.claim,
                quote=evidence.excerpt,
                confidence=evidence.relevance_score,
            )

            evidence_map[evidence.evidence_id] = db_evidence

        logger.info("Saved %s evidence items.", len(evidence_map))

        # ------------------------------------------------------
        # Save Validation Results
        # ------------------------------------------------------

        validation_count = 0

        for validation in result.validations:

            db_evidence = evidence_map.get(validation.evidence_id)

            if not db_evidence:
                logger.warning(
                    "Skipping validation for %s because evidence was not found.",
                    validation.evidence_id,
                )
                continue

            self.validation_repository.create_validation(
                evidence_id=db_evidence["id"],
                is_valid=validation.is_valid,
                credibility_score=validation.credibility_score,
                recency_score=validation.recency_score,
                is_duplicate=validation.is_duplicate,
                has_conflict=validation.has_conflict,
                reason=validation.reason,
            )

            validation_count += 1

        logger.info("Saved %s validation results.", validation_count)

        # ------------------------------------------------------
        # Save Final Report
        # ------------------------------------------------------

        report_data = result.report.model_dump()

        self.report_repository.create_report(
            job_id=job_id,
            report=report_data,
        )

        logger.info("Saved research report.")

        # ------------------------------------------------------
        # Success Summary
        # ------------------------------------------------------

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("Research job %s completed successfully.", job_id)
        logger.info("Tasks: %s", len(result.tasks))
        logger.info("Sources: %s", len(result.sources))
        logger.info("Evidence: %s", len(result.evidences))
        logger.info("Validations: %s", len(result.validations))
        logger.info("Completed in %.2f seconds.", elapsed)
        logger.info("=" * 60)

        return result
