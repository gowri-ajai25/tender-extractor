from tender_extractor import TenderExtractionWorkflow
from app import create_app  # your app.py


def main() -> None:
    workflow = TenderExtractionWorkflow()
    app = create_app(workflow)

    # Run the web server
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
