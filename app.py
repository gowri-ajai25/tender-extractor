import json
import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory, jsonify

from tender_extractor import TenderExtractionWorkflow, EXTRACTION_INSTRUCTION


UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")

# Will store the last uploaded PDF path for both panels
CURRENT_PDF_PATH: Path | None = None

# Short, casual chat instruction (keeps extraction behavior when PDF is provided)
CHAT_INSTRUCTION = """
You are a friendly, casual assistant.

• If the user is just chatting ("hi", "how are you", etc.), reply briefly and naturally.
• Do NOT mention tenders or PDFs unless the user directly asks about them.
• If the user asks for specific tender fields (e.g., "extract tender_id and items")
  AND tender text is available, extract those fields from the tender and reply ONLY
  with valid JSON.
• If tender text is NOT available and the user asks for fields, reply briefly asking
  the user to upload the PDF first.
Keep replies short and conversational unless the user requests structured JSON.
"""


def create_app(workflow: TenderExtractionWorkflow | None = None) -> Flask:
    """
    Flask application factory.
    Does NOT run the server; just returns the app instance.
    """
    app = Flask(__name__, template_folder="templates")

    # Ensure folders exist
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    extraction_workflow = workflow or TenderExtractionWorkflow()

    @app.route("/", methods=["GET"])
    def index():
        # Left side starts with the default extraction prompt from your code
        return render_template("index.html", default_prompt=EXTRACTION_INSTRUCTION)

    @app.route("/prompt_extract", methods=["POST"])
    def prompt_extract():
        """
        Runs a user-edited extraction prompt against the CURRENT_PDF_PATH.
        Returns JSON (as text) and saves it as a downloadable file.
        """
        global CURRENT_PDF_PATH

        data = request.get_json() or {}
        prompt = (data.get("prompt") or "").strip()

        if not prompt:
            return jsonify({"error": "Prompt is empty."}), 400

        if CURRENT_PDF_PATH is None or not CURRENT_PDF_PATH.exists():
            return jsonify({
                "error": "Please upload a PDF in the JSON Extractor chat first."
            }), 400

        try:
            tender_text = extraction_workflow.pdf_reader.read_pdf_as_text(
                CURRENT_PDF_PATH
            )

            full_prompt = prompt + "\n\nTender text:\n\n" + tender_text

            response = extraction_workflow.extractor.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )

            raw = (response.text or "").strip()

            if raw.startswith("```"):
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1].lstrip("json").strip()

            try:
                parsed = json.loads(raw)
                result_json_str = json.dumps(parsed, indent=2, ensure_ascii=False)
            except Exception:
                result_json_str = raw

            json_id = uuid.uuid4().hex
            output_filename = f"{json_id}_prompt_output.json"
            output_path = OUTPUT_FOLDER / output_filename
            with output_path.open("w", encoding="utf-8") as f:
                f.write(result_json_str)

            return jsonify({"json": result_json_str, "filename": output_filename})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/chat_extract", methods=["POST"])
    def chat_extract():
        """
        Chat endpoint for the JSON extractor (right-hand chat box).

        Behavior:
        - If a file is attached:
            * Save it, set CURRENT_PDF_PATH, run the default extractor (so user gets the JSON file).
            * If a message is provided too, run a chat prompt that includes the tender text + user message
              and return the chat reply as well as the JSON filename.
        - If no file is attached:
            * If a message is provided and CURRENT_PDF_PATH exists, answer using that tender text as context.
            * If a message is provided and no PDF exists, answer as a general-purpose assistant.
        """
        global CURRENT_PDF_PATH

        pdf_file = request.files.get("pdf")
        message = (request.form.get("message") or "").strip()

        if not pdf_file and not message:
            return jsonify({"error": "Please attach a PDF or enter a message."}), 400

        if pdf_file and pdf_file.filename != "":
            pdf_id = uuid.uuid4().hex
            safe_name = f"{pdf_id}_{pdf_file.filename}"
            pdf_path = UPLOAD_FOLDER / safe_name
            pdf_file.save(pdf_path)

            CURRENT_PDF_PATH = pdf_path

            try:
                result = extraction_workflow.extract_from_pdf(pdf_path)
                result_json_str = json.dumps(result, indent=2, ensure_ascii=False)

                json_id = uuid.uuid4().hex
                output_filename = f"{json_id}_output.json"
                output_path = OUTPUT_FOLDER / output_filename
                with output_path.open("w", encoding="utf-8") as f:
                    f.write(result_json_str)

            except Exception as e:
                return jsonify({"error": f"Extraction failed: {e}"}), 500

            if message:
                try:
                    tender_text = extraction_workflow.pdf_reader.read_pdf_as_text(CURRENT_PDF_PATH)

                    prompt = (
                        CHAT_INSTRUCTION
                        + "\n\nTender text:\n\n"
                        + tender_text
                        + "\n\nUser:\n"
                        + message
                    )

                    resp = extraction_workflow.extractor.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )

                    reply_text = (resp.text or "").strip()
                    if reply_text.startswith("```"):
                        parts = reply_text.split("```")
                        if len(parts) >= 2:
                            reply_text = parts[1].lstrip("json").strip()

                    return jsonify({"reply": reply_text, "filename": output_filename})

                except Exception as e:
                    return jsonify({"reply": result_json_str, "filename": output_filename, "chat_error": str(e)}), 200

            return jsonify({"reply": result_json_str, "filename": output_filename})

        try:
            if CURRENT_PDF_PATH and CURRENT_PDF_PATH.exists():
                tender_text = extraction_workflow.pdf_reader.read_pdf_as_text(CURRENT_PDF_PATH)
                prompt = (
                    CHAT_INSTRUCTION
                    + "\n\nTender text:\n\n"
                    + tender_text
                    + "\n\nUser:\n"
                    + message
                )
            else:
                # No tender available — answer as a general assistant (no tender mention unless asked)
                prompt = (
                    CHAT_INSTRUCTION
                    + "\n\nUser:\n"
                    + message
                )

            response = extraction_workflow.extractor.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            reply_text = (response.text or "").strip()
            if reply_text.startswith("```"):
                parts = reply_text.split("```")
                if len(parts) >= 2:
                    reply_text = parts[1].lstrip("json").strip()

            return jsonify({"reply": reply_text})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/download/<filename>")
    def download_file(filename):
        return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

    return app