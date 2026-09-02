import os
import logging

from dotenv import load_dotenv
from google import genai

from ai.llm.base import LLM

# -------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment.")

# -------------------------------------------------------
# Gemini LLM Wrapper
# -------------------------------------------------------


class GeminiLLM(LLM):
    """
    Shared Gemini client used across all AI agents.

    - Uses Gemini Chat API (avoids AFC warning).
    - Uses Flash Lite as the primary model.
    - Falls back to Flash if Lite is unavailable.
    - Retry logic is handled by ResearchPipeline.
    """

    def __init__(self):
        self.client = genai.Client(api_key=api_key)

        # Primary model (fast and inexpensive)
        self.primary_model = "gemini-3.5-flash"

        # Fallback model
        self.fallback_model = "gemini-2.5-flash-lite"

    # ---------------------------------------------------
    # Internal Chat Generator
    # ---------------------------------------------------

    def _chat_generate(self, model: str, prompt: str) -> str:
        """
        Send a prompt using Gemini Chat API.
        This removes the Automatic Function Calling (AFC) warning.
        """

        chat = self.client.chats.create(model=model)

        response = chat.send_message(prompt)

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return response.text.strip()

    # ---------------------------------------------------
    # Public Generate Method
    # ---------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Generate text using Gemini.

        Flow:
        1. Try primary model once.
        2. If primary fails, try fallback model once.
        3. Raise RuntimeError if both fail.

        NOTE:
        ResearchPipeline is responsible for retrying Gemini
        requests up to 5 times with exponential backoff.
        """

        # Try primary model
        try:
            logger.info(f"Using primary Gemini model: {self.primary_model}")

            return self._chat_generate(
                model=self.primary_model,
                prompt=prompt,
            )

        except Exception as primary_error:
            logger.warning(
                f"Primary Gemini model failed: {primary_error}. "
                "Trying fallback model..."
            )

        # Try fallback model
        try:
            logger.info(f"Using fallback Gemini model: {self.fallback_model}")

            return self._chat_generate(
                model=self.fallback_model,
                prompt=prompt,
            )

        except Exception as fallback_error:
            logger.error(f"Fallback Gemini model failed: {fallback_error}")

            raise RuntimeError(
                "Gemini is temporarily unavailable."
            ) from fallback_error
