import os
import json
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

load_dotenv() 

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")

EXTRACTION_INSTRUCTION = """
You are an AI assistant that extracts structured data from pharmaceutical tender documents.

Extract the following and return ONLY this JSON:

{
  "tender_id": string or null,
  "mode": string or null,
  "open_date": string or null,
  "cnpj": string or null,
  "institution": string or null,
  "city": string or null,
  "state": string or null,
  "tender_time": string or null,
  "items": [
    {
      "item_lot_number": string or null,
      "product_name": string or null,
      "quantity": string or null
    }
  ]
}

Rules:
- Output must be valid JSON ONLY.
- No markdown, no explanations.
- Missing fields = null.
"""

class PDFReaderService:
    """Reads text from a PDF file."""

    def read_pdf_as_text(self, pdf_path: Path | str) -> str:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text


class GeminiTenderExtractor:
    """Wraps Gemini client and knows how to extract JSON from tender text."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or GEMINI_API_KEY
        self.client = genai.Client(api_key=self.api_key)

    def extract_json_from_tender_text(self, tender_text: str) -> dict:
        prompt = EXTRACTION_INSTRUCTION + "\n\nTender text:\n\n" + tender_text

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        raw = response.text.strip()

        # Remove ```json fences if they appear
        if raw.startswith("```"):
            parts = raw.split("```")
            # parts[1] is typically: "json\n{...}"
            raw = parts[1].lstrip("json").strip()

        return json.loads(raw)

class TenderExtractionWorkflow:
    """
    Coordinates reading the PDF and calling Gemini.
    This is what the rest of the app (web / CLI) should use.
    """

    def __init__(
        self,
        pdf_reader: Optional[PDFReaderService] = None,
        extractor: Optional[GeminiTenderExtractor] = None,
    ) -> None:
        self.pdf_reader = pdf_reader or PDFReaderService()
        self.extractor = extractor or GeminiTenderExtractor()

    def extract_from_pdf(self, pdf_path: Path | str) -> dict:
        """Read PDF, send to Gemini, return parsed JSON dict."""
        tender_text = self.pdf_reader.read_pdf_as_text(pdf_path)

        if not tender_text.strip():
            raise ValueError("No text found in PDF. Is the PDF a scanned image?")

        return self.extractor.extract_json_from_tender_text(tender_text)
