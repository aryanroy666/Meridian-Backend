import os

from dotenv import load_dotenv
from google import genai

from ai.llm.base import LLM


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
import os
import time

from dotenv import load_dotenv
from google import genai

from ai.llm.base import LLM


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")


class GeminiLLM(LLM):

    def __init__(self):
        self.client = genai.Client(api_key=api_key)

        # Primary model
        self.primary_model = "gemini-3.5-flash-lite"

        # Fallback model
        self.fallback_model = "gemini-3.5-flash"

    def generate(self, prompt: str) -> str:

        # --------------------------------------------------
        # 1. Try primary model
        # --------------------------------------------------
        for attempt in range(2):
            try:
                print(
                    f"Trying primary model "
                    f"{self.primary_model} (attempt {attempt + 1}/2)"
                )

                response = self.client.models.generate_content(
                    model=self.primary_model,
                    contents=prompt
                )

                print("Primary Gemini model succeeded.")
                return response.text

            except Exception as e:
                print(
                    f"Primary Gemini model failed "
                    f"(attempt {attempt + 1}/2): {e}"
                )

                if attempt == 0:
                    print("Waiting 5 seconds before retry...")
                    time.sleep(5)

        # --------------------------------------------------
        # 2. Primary failed twice → fallback model
        # --------------------------------------------------
        print(
            f"Primary model unavailable. "
            f"Trying fallback model {self.fallback_model}..."
        )

        try:
            response = self.client.models.generate_content(
                model=self.fallback_model,
                contents=prompt
            )

            print("Fallback Gemini model succeeded.")
            return response.text

        except Exception as e:
            print(f"Fallback Gemini model failed: {e}")

            raise RuntimeError(
                "Gemini is temporarily unavailable. "
                "Please try the research again later."
            )
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")


class GeminiLLM(LLM):

    def __init__(self):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.5-flash-lite"

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response.text
