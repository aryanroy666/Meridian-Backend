import os
import re
import time
import logging

from dotenv import load_dotenv
from google import genai

from ai.llm.base import LLM

# =======================================================
# Load Environment Variables
# =======================================================

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment.")

# =======================================================
# Retry Configuration
# =======================================================

MAX_MODEL_RETRIES = 3
DEFAULT_BACKOFF = 2  # Seconds

# =======================================================
# Gemini LLM Wrapper
# =======================================================


class GeminiLLM(LLM):
    """
    Shared Gemini client used across all AI agents.

    Features
    --------
    - Uses Gemini Chat API.
    - Primary model + multiple fallback models.
    - Automatic retry for temporary Gemini failures.
    - Handles 429 quota errors using Gemini RetryInfo delay.
    - Handles 503 unavailable errors with exponential backoff.
    """

    def __init__(self):
        self.client = genai.Client(api_key=api_key)

        # -------------------------------------------------------
        # Model Configuration (Best → Weakest)
        # -------------------------------------------------------

        self.primary_model = "gemini-3.1-flash-lite"

        self.fallback_models = [
            "gemini-3.5-flash",
            "gemini-2.5-flash-lite",
        ]

        logger.info(
            "Primary Gemini model: %s | Fallback models: %s",
            self.primary_model,
            ", ".join(self.fallback_models),
        )

    # =======================================================
    # Internal Chat API Call
    # =======================================================

    def _chat_generate(self, model: str, prompt: str) -> str:
        """Send prompt using Gemini Chat API."""

        chat = self.client.chats.create(model=model)

        response = chat.send_message(prompt)

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text.strip()

    # =======================================================
    # Retry Wrapper
    # =======================================================

    def _generate_with_retry(self, model: str, prompt: str) -> str:
        """Retry Gemini request on temporary failures."""

        last_error = None

        for attempt in range(1, MAX_MODEL_RETRIES + 1):

            try:
                logger.info(
                    "Using Gemini model: %s (Attempt %d/%d)",
                    model,
                    attempt,
                    MAX_MODEL_RETRIES,
                )

                return self._chat_generate(model=model, prompt=prompt)

            except Exception as e:

                last_error = e
                error_text = str(e)

                retryable = any(
                    keyword in error_text
                    for keyword in (
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "503",
                        "UNAVAILABLE",
                        "500",
                        "INTERNAL",
                    )
                )

                if not retryable:
                    logger.error("Non-retryable Gemini error (%s): %s", model, e)
                    raise

                # Read Gemini RetryInfo (429 errors)
                retry_match = re.search(
                    r"retry in ([0-9.]+)s",
                    error_text,
                    re.IGNORECASE,
                )

                if retry_match:
                    wait_time = float(retry_match.group(1))
                else:
                    wait_time = DEFAULT_BACKOFF * (2 ** (attempt - 1))

                if attempt < MAX_MODEL_RETRIES:
                    logger.warning(
                        "%s unavailable (Attempt %d/%d). Retrying in %.1fs...",
                        model,
                        attempt,
                        MAX_MODEL_RETRIES,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "%s failed after %d attempts.",
                        model,
                        MAX_MODEL_RETRIES,
                    )

        raise last_error

    # =======================================================
    # Public Generate Method
    # =======================================================

    def generate(self, prompt: str) -> str:
        """
        Generate text using primary model and fallback models.

        Order:
        1. gemini-3.5-flash
        2. gemini-2.5-flash
        3. gemini-2.5-flash-lite
        """

        models = [self.primary_model] + self.fallback_models

        last_error = None

        for model in models:
            try:
                return self._generate_with_retry(model, prompt)

            except Exception as e:
                last_error = e

                logger.warning(
                    "Model %s failed. Trying next fallback model...",
                    model,
                )

        error_text = str(last_error)

        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            raise RuntimeError(
                "Gemini free-tier quota exceeded. Please retry after a few seconds."
            ) from last_error

        if "503" in error_text or "UNAVAILABLE" in error_text:
            raise RuntimeError(
                "Gemini service is temporarily unavailable. Please retry in a few minutes."
            ) from last_error

        raise RuntimeError(
            "Gemini request failed after all retry attempts."
        ) from last_error
