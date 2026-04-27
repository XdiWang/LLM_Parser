# LLM Parser



LLM Parser  is a comprehensive toolkit designed to evaluate, test, and visualize the performance of various document parsers (especially for PDFs) in the context of Large Language Model (LLM) applications. It includes specific tools to test parser resilience against adversarial PDF techniques like "Implicit Injection."

## Interactive Parsing Lab (Streamlit Web App)

The web interface allows for real-time testing and side-by-side comparison of different LangChain-based parsers.

Key Features: Supports PDF, DOCX, CSV, JSON, HTML, and images. Provides metadata inspection and advanced parsing options (OCR, extraction strategies).

How to Start:

# Run from the project root directory
streamlit run app.py


Usage:

Upload a file in the "Parsing Console" on the left.

Select the parsers you wish to test (e.g., Docling, Unstructured, PyMuPDF).

Click "Start Parsing".

View the extracted text and metadata in the right panel.

## LangChain Automation Pipeline

Use this module to batch-process PDF files through multiple parsers automatically and save the results as text files.

How to Start:

# Navigate to the loader directory
cd LangChain/File_Loader

# Run the automation script
python run.py


Configuration:

Open run.py to modify the INPUT_FILE variable to point to your specific PDF.

The script will execute 10 different parsing logics (including Docling, PDFPlumber, PyPDFium2, etc.) and save results to the Output/ directory.

Windows users can also use run.bat.

## Evaluation & Visualization

This module quantifies how effectively parsers "expose" or "recover" hidden content from adversarial documents.

A. Run Automated Evaluation

Calculates metrics such as "Injection Parsing Success Rate" and "Recovery Completeness."

How to Start:

# Run from the project root
python Evaluation/evaluate_injection_recovery.py --dataset-root ./Dataset --parse-results-root ./Parser/BatchParseResult


Output: Results are saved in Evaluation/results/all_batches/, including CSV metric files and a summary JSON.

B. Generate Visualization Heatmaps

Converts evaluation CSVs into intuitive heatmaps.

How to Start:

# Generate heatmaps based on evaluation results
python Figures/plot_parser_heatmaps.py --csv-dir Evaluation/results/batch1 --output-dir Figures/output/batch1


Parameters:

--csv-dir: Path to the directory containing the evaluation CSV files.

--output-dir: Path where the .png heatmaps will be saved.

Output: Generates specific heatmaps (e.g., parser_category_success_rate.png) and an overview file.

Dataset

Dataset constructued：[`Dean2Wang/Adversarial-PDF-Parsing-Taxonomy-8Ops`](https://huggingface.co/datasets/Dean2Wang/Adversarial-PDF-Parsing-Taxonomy-8Ops)

Core Dependencies

Ensure you have the following installed to run all modules:

Web/UI: streamlit, streamlit-pdf-viewer

Frameworks: langchain, langchain-community, langchain-docling, langchain-unstructured

Parsers: pdfminer.six, pdfplumber, pymupdf, pymupdf4llm, pypdfium2, beautifulsoup4

Math/Plotting: matplotlib, torch (optional, for Docling acceleration)

System: Java 11+ (required for OpenDataLoader)

