import base64
import csv
import html
import json
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_pdf_viewer import pdf_viewer
except ImportError:
    pdf_viewer = None
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    JSONLoader,
    PDFMinerLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    PyPDFium2Loader,
    PyPDFLoader,
)

try:
    from langchain_docling import DoclingLoader
except ImportError:
    DoclingLoader = None

try:
    from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
except ImportError:
    OpenDataLoaderPDFLoader = None

try:
    from langchain_pymupdf4llm import PyMuPDF4LLMLoader
except ImportError:
    PyMuPDF4LLMLoader = None

try:
    from langchain_unstructured import UnstructuredLoader
except ImportError:
    UnstructuredLoader = None

try:
    import torch
except ImportError:
    torch = None

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
        RapidOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
except ImportError:
    DocumentConverter = None
    PdfFormatOption = None
    PdfPipelineOptions = None
    AcceleratorOptions = None
    AcceleratorDevice = None
    RapidOcrOptions = None
    InputFormat = None


st.set_page_config(
    page_title="LLM Parser Lab",
    page_icon="📚",
    layout="wide",
)

PARSER_INFO = {
    "Docling": "Supports PDF, DOCX, PPTX, XLSX, HTML, images, and more. Ideal as a unified entry point.",
    "Unstructured": "Multi-format parser suitable for unified extraction from unstructured documents.",
    "PyPDFLoader": "LangChain's basic PDF parser, suitable for general text extraction.",
    "PyMuPDF": "Fast and metadata-rich, suitable for standard PDFs.",
    "PyMuPDF4LLM": "Converts PDFs into Markdown output better suited for LLMs.",
    "PDFPlumber": "Usually more friendly for multi-column layouts and complex formatting.",
    "PyPDFium2": "Built on PDFium with strong parsing performance.",
    "PDFMiner": "Pure Python implementation with a focus on layout analysis.",
    "OpenDataLoader": "Parses PDFs into Markdown and supports content safety filtering.",
    "PyPDFDirectory": "Batch directory loader that can parse multiple PDFs from one directory.",
    "CSVLoader": "Used for parsing CSV files, suitable for structured tabular data.",
    "JSONLoader": "Used for parsing JSON files and can extract content with a jq schema.",
    "BSHTMLLoader": "Used for parsing HTML files and extracting text via BeautifulSoup.",
}

PARSER_SUPPORT = {
    "Docling": {"pdf", "docx", "pptx", "xlsx", "html", "htm", "md", "txt", "png", "jpg", "jpeg", "tiff", "bmp"},
    "Unstructured": {"pdf", "docx", "pptx", "xlsx", "html", "htm", "md", "txt", "csv", "json"},
    "PyPDFLoader": {"pdf"},
    "PyMuPDF": {"pdf"},
    "PyMuPDF4LLM": {"pdf"},
    "PDFPlumber": {"pdf"},
    "PyPDFium2": {"pdf"},
    "PDFMiner": {"pdf"},
    "OpenDataLoader": {"pdf"},
    "PyPDFDirectory": {"pdf"},
    "CSVLoader": {"csv"},
    "JSONLoader": {"json"},
    "BSHTMLLoader": {"html", "htm"},
}

UPLOAD_TYPES = sorted({ext for exts in PARSER_SUPPORT.values() for ext in exts})
PREFERRED_ORDER = [
    "Docling",
    "Unstructured",
    "PyPDFLoader",
    "PyMuPDF",
    "PyMuPDF4LLM",
    "CSVLoader",
    "JSONLoader",
    "BSHTMLLoader",
    "PDFPlumber",
    "PyPDFium2",
    "PDFMiner",
    "OpenDataLoader",
    "PyPDFDirectory",
]
MARKDOWN_PARSERS = {"PyMuPDF4LLM", "OpenDataLoader"}


def get_file_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower().lstrip(".")


def get_available_parsers(file_extension: str) -> list[str]:
    if not file_extension:
        return list(PARSER_INFO.keys())
    return [
        parser_name
        for parser_name in PREFERRED_ORDER
        if file_extension in PARSER_SUPPORT.get(parser_name, set())
    ]


def get_default_parsers(file_extension: str) -> list[str]:
    available = get_available_parsers(file_extension)
    return available[:3]


def decode_uploaded_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def format_metadata(metadata: dict[str, Any]) -> str:
    if not metadata:
        return "{}"
    return json.dumps(metadata, ensure_ascii=False, indent=2, default=str)


def docs_to_display_text(docs: list[Any], parser_name: str, include_metadata: bool = True) -> str:
    sections: list[str] = [f"Parser: {parser_name}", f"Document segments: {len(docs)}", "=" * 40, ""]

    for index, doc in enumerate(docs, start=1):
        page_content = getattr(doc, "page_content", "") or ""
        metadata = getattr(doc, "metadata", {}) or {}
        sections.append(f"--- Result {index} ---")
        if include_metadata:
            sections.append("[Metadata]")
            sections.append(format_metadata(metadata))
            sections.append("")
        sections.append("[Content]")
        sections.append(page_content)
        sections.append("")

    return "\n".join(sections)


def docs_to_serializable(docs: list[Any], include_metadata: bool = True) -> list[dict[str, Any]]:
    serialized_docs: list[dict[str, Any]] = []
    for doc in docs:
        serialized_docs.append(
            {
                "page_content": getattr(doc, "page_content", "") or "",
                "metadata": (getattr(doc, "metadata", {}) or {}) if include_metadata else {},
            }
        )
    return serialized_docs


def parse_csv_preview(file_bytes: bytes) -> list[dict[str, Any]]:
    text = decode_uploaded_text(file_bytes)
    rows = list(csv.DictReader(StringIO(text)))
    if rows:
        return rows

    reader = csv.reader(StringIO(text))
    raw_rows = list(reader)
    if not raw_rows:
        return []

    headers = [f"column_{index + 1}" for index in range(len(raw_rows[0]))]
    return [dict(zip(headers, row)) for row in raw_rows]


def parse_json_preview(file_bytes: bytes) -> Any:
    return json.loads(decode_uploaded_text(file_bytes))


def summarize_docs(serialized_docs: list[dict[str, Any]]) -> dict[str, Any]:
    total_chars = sum(len(item.get("page_content", "")) for item in serialized_docs)
    non_empty = sum(1 for item in serialized_docs if item.get("page_content", "").strip())
    return {
        "doc_count": len(serialized_docs),
        "non_empty_count": non_empty,
        "total_chars": total_chars,
    }


def render_metadata_panel(serialized_docs: list[dict[str, Any]]) -> None:
    has_metadata = any(item.get("metadata") for item in serialized_docs)
    if not has_metadata:
        st.info("No metadata was extracted, or this parser did not return any displayable metadata.")
        return

    for index, item in enumerate(serialized_docs[:8], start=1):
        metadata = item.get("metadata", {})
        if not metadata:
            continue
        with st.expander(f"Segment {index} Metadata", expanded=index == 1):
            st.json(metadata)


def render_parsed_text_panel(parser_name: str, serialized_docs: list[dict[str, Any]]) -> None:
    st.markdown("#### Parsed Text")
    st.caption("The right side stays fixed on text output for easier side-by-side comparison with the original file.")

    if not serialized_docs:
        st.info("This parser did not return any displayable text content.")
        return

    for index, item in enumerate(serialized_docs, start=1):
        metadata = item.get("metadata", {})
        page_label = metadata.get("page")
        if isinstance(page_label, int):
            page_label += 1

        title = f"Segment {index}"
        if page_label is not None:
            title += f" · Page {page_label}"

        with st.expander(title, expanded=index == 1):
            content = item.get("page_content", "")
            if parser_name in MARKDOWN_PARSERS:
                st.markdown(content or "_Empty content_")
            else:
                st.text_area(
                    f"parsed_text_{parser_name}_{index}",
                    value=content,
                    height=220,
                    label_visibility="collapsed",
                )


def build_pages_sidebar(serialized_docs: list[dict[str, Any]]) -> str:
    items: list[str] = []
    for index, item in enumerate(serialized_docs, start=1):
        metadata = item.get("metadata", {}) or {}
        page_label = metadata.get("page")
        if isinstance(page_label, int):
            page_label += 1
        title = f"Segment {index}"
        if page_label is not None:
            title = f"Page {page_label}"
        anchor = f"segment-{index}"
        items.append(f'<a href="#{anchor}" class="compare-nav-item">{html.escape(title)}</a>')

    if not items:
        items.append('<div class="compare-nav-empty">No page information available</div>')

    return "".join(items)


def build_image_gallery(file_bytes: bytes, file_name: str) -> str:
    image_base64 = base64.b64encode(file_bytes).decode("utf-8")
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    extension = get_file_extension(file_name)
    mime = mime_map.get(extension, "image/png")
    safe_name = html.escape(file_name)
    return f"""
    <div class=\"compare-media-single\">
        <img src=\"data:{mime};base64,{image_base64}\" alt=\"{safe_name}\" />
        <div class=\"compare-media-label\">{safe_name}</div>
    </div>
    """


def build_csv_table_html(file_bytes: bytes) -> str:
    rows = parse_csv_preview(file_bytes)
    if not rows:
        return '<div class="compare-empty">This CSV file has no data to display.</div>'

    headers = list(rows[0].keys())
    head_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_rows: list[str] = []
    for row in rows[:100]:
        cells = "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
        body_rows.append(f"<tr>{cells}</tr>")

    return f"""
    <div class=\"compare-table-wrap\">
        <table class=\"compare-table\">
            <thead><tr>{head_html}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """


def build_json_html(file_bytes: bytes) -> str:
    try:
        json_text = json.dumps(parse_json_preview(file_bytes), ensure_ascii=False, indent=2)
    except Exception as exc:
        json_text = f"Failed to preview JSON file: {exc}"
    return f'<pre class="compare-code">{html.escape(json_text[:40000])}</pre>'


def build_text_html(file_bytes: bytes) -> str:
    text = decode_uploaded_text(file_bytes)
    return f'<pre class="compare-code">{html.escape(text[:40000])}</pre>'


def build_html_code_html(file_bytes: bytes) -> str:
    html_text = decode_uploaded_text(file_bytes)
    return f'<pre class="compare-code">{html.escape(html_text[:40000])}</pre>'


def render_pdf_file_view(
    file_bytes: bytes,
    serialized_docs: list[dict[str, Any]] | None = None,
    viewer_key: str = "pdf",
) -> None:
    if pdf_viewer is None:
        st.warning("`streamlit-pdf-viewer` is not installed. Please install it and restart the app.")
        return

    page_count = len(serialized_docs or [])
    nav_col, viewer_col = st.columns([0.2, 0.8], gap="small")

    with nav_col:
        st.markdown("##### Page Navigation")
        if page_count:
            selected_page = st.radio(
                "PDF page selector",
                options=list(range(1, page_count + 1)),
                format_func=lambda page: f"Page {page}",
                index=0,
                label_visibility="collapsed",
                key=f"pdf_page_selector_{viewer_key}",
            )
        else:
            selected_page = None
            st.caption("No page information available")

    with viewer_col:
        pdf_viewer(
            input=file_bytes,
            width="100%",
            height=780,
            key=f"pdf_preview_viewer_{viewer_key}",
            render_text=True,
            resolution_boost=2,
            zoom_level="auto",
            viewer_align="center",
            show_page_separator=True,
            scroll_to_page=selected_page,
        )


def render_compare_file_view(
    file_name: str,
    file_extension: str,
    file_bytes: bytes,
    serialized_docs: list[dict[str, Any]],
    viewer_key: str,
) -> None:
    st.markdown("#### Original File Preview")
    st.caption("The left side stays fixed on the original file view with independent scrolling for easier manual comparison.")

    image_extensions = {"png", "jpg", "jpeg", "bmp", "tiff", "webp", "gif"}

    if file_extension == "pdf":
        render_pdf_file_view(file_bytes, serialized_docs, viewer_key=viewer_key)
        return
    if file_extension in image_extensions:
        components.html(build_image_gallery(file_bytes, file_name), height=780, scrolling=True)
        return
    if file_extension == "csv":
        components.html(build_csv_table_html(file_bytes), height=780, scrolling=True)
        return
    if file_extension == "json":
        components.html(build_json_html(file_bytes), height=780, scrolling=True)
        return
    if file_extension in {"html", "htm"}:
        preview_tabs = st.tabs(["Rendered View", "Source Code"])
        with preview_tabs[0]:
            components.html(decode_uploaded_text(file_bytes), height=760, scrolling=True)
        with preview_tabs[1]:
            components.html(build_html_code_html(file_bytes), height=760, scrolling=True)
        return
    if file_extension in {"md", "txt"}:
        components.html(build_text_html(file_bytes), height=780, scrolling=True)
        return

    st.info(f"Custom preview for `.{file_extension}` is not available yet. The raw content is shown below.")
    components.html(build_text_html(file_bytes), height=780, scrolling=True)


def render_structured_preview(
    file_extension: str,
    parser_name: str,
    serialized_docs: list[dict[str, Any]],
    file_bytes: bytes,
    file_name: str,
    include_metadata: bool,
) -> None:
    compare_col, result_col = st.columns([1.05, 0.95], gap="large")

    with compare_col:
        render_compare_file_view(
            file_name=file_name,
            file_extension=file_extension,
            file_bytes=file_bytes,
            serialized_docs=serialized_docs,
            viewer_key=parser_name,
        )

    with result_col:
        result_tabs = ["Parsed Text"]
        if include_metadata:
            result_tabs.append("Metadata")
        tabs = st.tabs(result_tabs)

        with tabs[0]:
            render_parsed_text_panel(parser_name, serialized_docs)

        if include_metadata:
            with tabs[1]:
                render_metadata_panel(serialized_docs)


def load_with_docling(file_path: str, force_ocr: bool) -> list[Any]:
    if DoclingLoader is None:
        raise RuntimeError("`langchain-docling` is not installed. Please install the required dependency first.")

    file_extension = Path(file_path).suffix.lower()

    if file_extension != ".pdf":
        return DoclingLoader(file_path=file_path).load()

    if not all([
        DocumentConverter,
        PdfFormatOption,
        PdfPipelineOptions,
        AcceleratorOptions,
        AcceleratorDevice,
        InputFormat,
    ]):
        raise RuntimeError("Docling PDF dependencies are not fully installed.")

    pipeline_options = PdfPipelineOptions()
    if force_ocr and RapidOcrOptions:
        pipeline_options.ocr_options = RapidOcrOptions(force_full_page_ocr=True)

    if torch is not None and torch.cuda.is_available():
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA,
        )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    loader = DoclingLoader(file_path=file_path, converter=converter)

    try:
        return loader.load()
    except Exception as exc:
        error_message = str(exc)
        lowered = error_message.lower()
        if "huggingface.co" in lowered or "connecttimeout" in lowered or "maxretryerror" in lowered:
            raise RuntimeError(
                "Docling parsing requires downloading models from Hugging Face, but the current network request timed out."
                "Please make sure `huggingface.co` is accessible, or pre-download the required models into the local cache before running again."
            ) from exc
        raise


def load_with_parser(parser_name: str, file_path: str, options: dict[str, Any]) -> list[Any]:
    if parser_name == "Docling":
        return load_with_docling(file_path, options.get("docling_force_ocr", False))
    if parser_name == "Unstructured":
        if UnstructuredLoader is None:
            raise RuntimeError("`langchain-unstructured` is not installed. Please install the required dependency first.")
        strategy = options.get("unstructured_strategy", "fast")
        return UnstructuredLoader(file_path, strategy=strategy).load()
    if parser_name == "PyPDFLoader":
        return PyPDFLoader(file_path).load()
    if parser_name == "PyMuPDF":
        return PyMuPDFLoader(file_path).load()
    if parser_name == "PyMuPDF4LLM":
        if PyMuPDF4LLMLoader is None:
            raise RuntimeError("`langchain-pymupdf4llm` is not installed. Please install the required dependency first.")
        return PyMuPDF4LLMLoader(file_path).load()
    if parser_name == "PDFPlumber":
        return PDFPlumberLoader(file_path).load()
    if parser_name == "PyPDFium2":
        return PyPDFium2Loader(file_path).load()
    if parser_name == "PDFMiner":
        return PDFMinerLoader(file_path).load()
    if parser_name == "OpenDataLoader":
        if OpenDataLoaderPDFLoader is None:
            raise RuntimeError("`langchain-opendataloader-pdf` is not installed. Please install the required dependency first.")
        loader_params: dict[str, Any] = {
            "file_path": [file_path],
            "format": "markdown",
        }
        safety_mode = options.get("open_data_safety", "default")
        if safety_mode != "default":
            loader_params["content_safety_off"] = [safety_mode]
        return OpenDataLoaderPDFLoader(**loader_params).load()
    if parser_name == "CSVLoader":
        return CSVLoader(file_path=file_path, encoding="utf-8").load()
    if parser_name == "JSONLoader":
        jq_schema = options.get("json_jq_schema", ".")
        text_content = options.get("json_text_content", False)
        return JSONLoader(file_path=file_path, jq_schema=jq_schema, text_content=text_content).load()
    if parser_name == "BSHTMLLoader":
        return BSHTMLLoader(file_path=file_path, open_encoding="utf-8").load()
    raise ValueError(f"Unsupported parser: {parser_name}")


def load_with_directory_parser(directory_path: str) -> list[Any]:
    return PyPDFDirectoryLoader(directory_path).load()


def run_selected_parsers(
    file_bytes: bytes,
    file_name: str,
    parsers: tuple[str, ...],
    options: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        file_path = temp_path / file_name
        file_path.write_bytes(file_bytes)

        directory_results: list[Any] | None = None
        if "PyPDFDirectory" in parsers:
            directory_results = load_with_directory_parser(str(temp_path))

        for parser_name in parsers:
            try:
                if parser_name == "PyPDFDirectory":
                    docs = directory_results or []
                else:
                    docs = load_with_parser(parser_name, str(file_path), options)

                results[parser_name] = {
                    "success": True,
                    "count": len(docs),
                    "text": docs_to_display_text(docs, parser_name, options.get("extract_metadata", True)),
                    "docs": docs_to_serializable(docs, options.get("extract_metadata", True)),
                }
            except Exception as exc:
                results[parser_name] = {
                    "success": False,
                    "error": str(exc),
                }

    return results


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0b1020;
        --panel: rgba(16, 24, 48, 0.82);
        --line: rgba(147, 197, 253, 0.18);
        --text: #e6eef8;
        --muted: #9fb3c8;
        --accent: #7dd3fc;
        --accent-2: #f9a8d4;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(125, 211, 252, 0.16), transparent 30%),
            radial-gradient(circle at top right, rgba(249, 168, 212, 0.14), transparent 28%),
            linear-gradient(180deg, #09101c 0%, #0b1020 45%, #060a14 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1220px;
    }

    h1, h2, h3 {
        font-family: 'Noto Serif SC', serif;
        color: var(--text);
        letter-spacing: 0.02em;
    }

    p, li, label, .stMarkdown, .stCaption {
        color: var(--muted);
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 28px 30px;
        background: linear-gradient(135deg, rgba(16,24,48,0.94), rgba(10,16,32,0.76));
        box-shadow: 0 20px 60px rgba(0,0,0,0.28);
        margin-bottom: 18px;
    }

    .hero::after {
        content: "";
        position: absolute;
        inset: auto -80px -80px auto;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(125, 211, 252, 0.24), transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.1rem;
        margin-bottom: 0.4rem;
        color: #f8fbff;
    }

    .hero-subtitle {
        font-size: 1rem;
        line-height: 1.8;
        max-width: 820px;
    }

    .info-card {
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px 18px;
        background: var(--panel);
        backdrop-filter: blur(10px);
        min-height: 120px;
    }

    .metric-chip {
        display: inline-block;
        margin: 6px 8px 0 0;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        color: #06111f;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
    }

    .stTextArea textarea, .stCodeBlock code {
        font-family: 'IBM Plex Mono', monospace;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: 0;
    }

    .compare-media-single {
        height: 760px;
        display: flex;
        flex-direction: column;
        gap: 14px;
        padding: 12px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(8, 12, 24, 0.88);
    }

    .compare-media-single img {
        width: 100%;
        flex: 1;
        object-fit: contain;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02);
    }

    .compare-media-label {
        font-size: 13px;
        color: var(--muted);
        text-align: center;
    }

    .compare-table-wrap {
        height: 760px;
        overflow: auto;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(8, 12, 24, 0.88);
    }

    .compare-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }

    .compare-table thead th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #101830;
        color: #f8fbff;
    }

    .compare-table th,
    .compare-table td {
        border-bottom: 1px solid rgba(147, 197, 253, 0.12);
        padding: 10px 12px;
        text-align: left;
        vertical-align: top;
        white-space: nowrap;
    }

    .compare-code {
        height: 760px;
        overflow: auto;
        margin: 0;
        padding: 18px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(8, 12, 24, 0.88);
        color: #d7e3f4;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        line-height: 1.7;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .compare-empty {
        height: 760px;
        display: grid;
        place-items: center;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(8, 12, 24, 0.88);
        color: var(--muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">LLM Parser Lab</div>
        <div class="hero-subtitle">
            Upload files in multiple formats, dynamically switch available LangChain parsers based on file type, and directly inspect extracted content and metadata.
        </div>
        <div>
            <span class="metric-chip">Multi Format</span>
            <span class="metric-chip">Docling</span>
            <span class="metric-chip">Unstructured</span>
            <span class="metric-chip">CSV / JSON / HTML</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.1, 1.4], gap="large")

with left_col:
    st.markdown("### Parsing Console")
    uploaded_file = st.file_uploader("Upload File", type=UPLOAD_TYPES)

    file_extension = get_file_extension(uploaded_file.name) if uploaded_file else ""
    parser_options = get_available_parsers(file_extension)
    default_parsers = get_default_parsers(file_extension)

    if uploaded_file:
        st.caption(f"Current file: `{uploaded_file.name}` · Type: `.{file_extension}`")
    else:
        st.caption("Supported formats: PDF / DOCX / PPTX / XLSX / HTML / CSV / JSON / MD / TXT / common image formats")

    selected_parsers = st.multiselect(
        "Select parsers to run",
        options=parser_options,
        default=default_parsers,
    )

    with st.expander("Advanced Options", expanded=False):
        open_data_safety = st.selectbox(
            "OpenDataLoader content safety filter (PDF)",
            options=["default", "all", "hidden-text", "off-page", "tiny", "hidden-ocg"],
            index=0,
        )
        unstructured_strategy = st.selectbox(
            "Unstructured parsing strategy",
            options=["fast", "hi_res", "ocr_only"],
            index=0,
        )
        docling_force_ocr = st.checkbox("Force full-page OCR in Docling (PDF only)", value=False)
        json_jq_schema = st.text_input("JSONLoader jq_schema", value=".")
        json_text_content = st.checkbox("Output JSONLoader content as plain text", value=False)
        extract_metadata = st.checkbox("Extract and display metadata", value=True)

    st.markdown("### Available Parser Notes")
    if parser_options:
        for parser_name in parser_options:
            st.markdown(f"- `{parser_name}`: {PARSER_INFO[parser_name]}")
    else:
        st.info("No parser matches the current file type.")

    run_clicked = st.button("Start Parsing", type="primary", use_container_width=True)

with right_col:
    st.markdown("### Usage Guide")
    tips_col1, tips_col2 = st.columns(2, gap="medium")
    with tips_col1:
        st.markdown(
            """
            <div class="info-card">
                <strong>Multi-format entry</strong><br><br>
                <code>Docling</code> and <code>Unstructured</code> are now recommended first as general-purpose entry points, then dedicated loaders can be added based on the specific file type.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tips_col2:
        st.markdown(
            """
            <div class="info-card">
                <strong>Dedicated loaders</strong><br><br>
                <code>CSVLoader</code> is for CSV, <code>JSONLoader</code> for JSON, <code>BSHTMLLoader</code> for HTML, while the existing dedicated PDF parser chain is still preserved.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Dependency Notes")
    st.markdown(
        """
- `OpenDataLoader` requires Java 11+ installed locally
- `Unstructured` usually also needs Poppler / Tesseract for OCR scenarios
- `JSONLoader` depends on `jq`
- `BSHTMLLoader` depends on `beautifulsoup4`
        """
    )

if run_clicked:
    if uploaded_file is None:
        st.warning("Please upload a file first.")
    elif not selected_parsers:
        st.warning("Please select at least one parser.")
    else:
        options = {
            "open_data_safety": open_data_safety,
            "unstructured_strategy": unstructured_strategy,
            "docling_force_ocr": docling_force_ocr,
            "json_jq_schema": json_jq_schema,
            "json_text_content": json_text_content,
            "extract_metadata": extract_metadata,
        }

        file_bytes = uploaded_file.getvalue()

        with st.spinner("Running the selected parsers, please wait..."):
            results = run_selected_parsers(
                file_bytes,
                uploaded_file.name,
                tuple(selected_parsers),
                options,
            )

        success_count = sum(1 for item in results.values() if item.get("success"))
        st.markdown("## Parsing Results")
        st.caption(f"Succeeded: {success_count} / {len(results)} parsers")

        tabs = st.tabs(list(results.keys()))
        for tab, parser_name in zip(tabs, results.keys()):
            result = results[parser_name]
            with tab:
                if result.get("success"):
                    summary = summarize_docs(result.get("docs", []))
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Segments", result["count"])
                    metric_cols[1].metric("Non-empty Segments", summary["non_empty_count"])
                    metric_cols[2].metric("Total Characters", summary["total_chars"])
                    metric_cols[3].metric("Metadata", "On" if extract_metadata else "Off")
                    st.success(f"`{parser_name}` parsed successfully and returned {result['count']} result(s).")
                    render_structured_preview(
                        file_extension=file_extension,
                        parser_name=parser_name,
                        serialized_docs=result.get("docs", []),
                        file_bytes=file_bytes,
                        file_name=uploaded_file.name,
                        include_metadata=extract_metadata,
                    )
                    with st.expander("Full Text Output", expanded=False):
                        st.download_button(
                            label=f"Download {parser_name} Result",
                            data=result["text"],
                            file_name=f"{Path(uploaded_file.name).stem}_{parser_name}.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                        st.text_area(
                            f"{parser_name} Parsed Content",
                            value=result["text"],
                            height=520,
                        )
                else:
                    st.error(f"`{parser_name}` parsing failed")
                    st.code(result.get("error", "Unknown error"))
else:
    st.markdown("## Waiting to Start")
    st.caption("Upload a file and select parsers, then click \"Start Parsing\" to see the comparison results here.")
