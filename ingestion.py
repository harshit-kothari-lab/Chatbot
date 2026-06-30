# import os
# import re
# import fitz  # PyMuPDF
# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# load_dotenv()

# PDF_DIR = "data/pdfs"
# PERSIST_DIR = "db/chroma_db"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# # ---------------------------
# # Text cleaning + page filters
# # ---------------------------
# def clean_text(text: str) -> str:
#     text = text.replace("\u00ad", "")  # soft hyphen
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def is_low_value_page(text: str) -> bool:
#     lower = text.lower().strip()

#     if len(lower) < 80:
#         return True

#     # likely table of contents
#     if "contents" in lower and ("introduction" in lower or "methodology" in lower):
#         return True

#     # acknowledgements / references pages
#     if lower.startswith("acknowledgement") or lower.startswith("acknowledgements"):
#         return True
#     if lower.startswith("references"):
#         return True

#     return False


# # ---------------------------
# # Section title detection
# # ---------------------------
# SECTION_PATTERNS = [
#     "abstract",
#     "introduction",
#     "research objective",
#     "objective",
#     "background",
#     "related work",
#     "literature review",
#     "methodology",
#     "methods",
#     "dataset",
#     "data",
#     "sentiment extraction",
#     "feature engineering",
#     "prediction task",
#     "classifiers",
#     "experiments",
#     "results",
#     "discussion",
#     "conclusion",
#     "limitations",
#     "future work"
# ]


# def detect_section(text: str) -> str:
#     """
#     Try to infer the section name from the page text.
#     Returns a section label if found, else 'unknown'.
#     """
#     lower = text.lower()

#     # first ~1000 chars are enough for section detection
#     head = lower[:1000]

#     for sec in SECTION_PATTERNS:
#         if sec in head:
#             return sec

#     return "unknown"


# # ---------------------------
# # PDF extraction
# # ---------------------------
# def extract_pdf_pages(pdf_path: str):
#     docs = []
#     pdf_name = os.path.basename(pdf_path)

#     with fitz.open(pdf_path) as pdf:
#         for page_num, page in enumerate(pdf, start=1):
#             raw_text = page.get_text("text")
#             text = clean_text(raw_text)

#             if not text:
#                 continue

#             if is_low_value_page(text):
#                 continue

#             section = detect_section(text)

#             docs.append(
#                 Document(
#                     page_content=text,
#                     metadata={
#                         "source": pdf_name,
#                         "page": page_num,
#                         "section": section
#                     }
#                 )
#             )

#     return docs


# def load_all_pdfs(pdf_dir=PDF_DIR):
#     if not os.path.exists(pdf_dir):
#         raise FileNotFoundError(f"{pdf_dir} does not exist.")

#     pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
#     if not pdf_files:
#         raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

#     all_docs = []

#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(pdf_dir, pdf_file)
#         print(f"Loading {pdf_file}...")
#         docs = extract_pdf_pages(pdf_path)
#         print(f"  -> kept {len(docs)} useful pages")
#         all_docs.extend(docs)

#     print(f"\nLoaded {len(all_docs)} useful page-level documents from {len(pdf_files)} PDFs.")
#     return all_docs


# # ---------------------------
# # Chunking
# # ---------------------------
# def split_documents(documents, chunk_size=700, chunk_overlap=120):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )

#     chunks = splitter.split_documents(documents)

#     # add chunk_id
#     for i, chunk in enumerate(chunks):
#         chunk.metadata["chunk_id"] = i

#     print(f"Created {len(chunks)} chunks.")
#     return chunks


# # ---------------------------
# # Vector store
# # ---------------------------
# def build_vectorstore(chunks, persist_dir=PERSIST_DIR):
#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embedding_model,
#         persist_directory=persist_dir
#     )
#     print(f"Vector store built and saved to {persist_dir}")
#     return vectorstore


# def main():
#     if os.path.exists(PERSIST_DIR):
#         print("Vector DB already exists. Delete db/chroma_db if you want to rebuild.")
#         return

#     docs = load_all_pdfs()
#     chunks = split_documents(docs)
#     build_vectorstore(chunks)
#     print("Ingestion complete.")


# if __name__ == "__main__":

#     main()




"""
===============================================================================
INGESTION PIPELINE
-------------------------------------------------------------------------------

This script performs the complete ingestion pipeline for the RAG chatbot.

Workflow:

PDF
 │
 ├── Extract text (PyMuPDF)
 │
 ├── OCR fallback (EasyOCR) if page contains little/no text
 │
 ├── Remove repeated headers and footers
 │
 ├── Detect document sections
 │
 ├── Extract tables
 │
 ├── Create LangChain Documents
 │
 ├── Semantic chunking
 │
 ├── Remove duplicate chunks
 │
 ├── Generate embeddings
 │
 └── Store inside ChromaDB

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import re
import hashlib
import logging
from pathlib import Path
from collections import Counter

import fitz                     # PyMuPDF
import pdfplumber
import easyocr
import numpy as np

from PIL import Image

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

PDF_DIR = "data/pdfs"

PERSIST_DIR = "db/chroma_db"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

CHUNK_SIZE = 700

CHUNK_OVERLAP = 120

OCR_TEXT_THRESHOLD = 60

MIN_PAGE_LENGTH = 80


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)


# =============================================================================
# EMBEDDING MODEL
# =============================================================================

logger.info("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(

    model_name=EMBED_MODEL,

    model_kwargs={"device": "cpu"},

    encode_kwargs={"normalize_embeddings": True}

)

logger.info("Embedding model loaded.")


# =============================================================================
# OCR READER
# =============================================================================

"""
EasyOCR is only used if a page contains almost no text.

This allows the chatbot to work with:

✓ scanned PDFs
✓ image PDFs
✓ photographs
✓ invoices
✓ handwritten notes (basic)
"""

logger.info("Loading OCR model...")

ocr_reader = easyocr.Reader(

    ['en'],

    gpu=False

)

logger.info("OCR model loaded.")


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Cleans extracted text.

    Removes:

    - extra spaces
    - soft hyphens
    - repeated newlines
    - tabs

    Returns cleaned text.
    """

    if not text:
        return ""

    text = text.replace("\u00ad", "")

    text = text.replace("\n", " ")

    text = text.replace("\t", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =============================================================================
# HASHING
# =============================================================================

def file_hash(filepath: str):
    """
    Creates a SHA256 hash of the PDF.

    Used later for incremental indexing.

    If a PDF hasn't changed,
    we don't need to process it again.
    """

    sha = hashlib.sha256()

    with open(filepath, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# =============================================================================
# LOW VALUE PAGE FILTER
# =============================================================================

LOW_VALUE_WORDS = [

    "references",

    "bibliography",

    "acknowledgement",

    "acknowledgements",

    "copyright",

    "table of contents",

    "contents"

]


def is_low_value_page(text: str):
    """
    Determines whether a page is useful.

    Returns False for

    - references

    - acknowledgements

    - copyright pages

    - very short pages

    """

    if len(text) < MIN_PAGE_LENGTH:

        return True

    lower = text.lower()

    for word in LOW_VALUE_WORDS:

        if lower.startswith(word):

            return True

    return False

# =============================================================================
# OCR UTILITIES
# =============================================================================

"""
Some PDFs contain no selectable text.

Example:
- Scanned books
- Scanned invoices
- Photographs saved as PDF

PyMuPDF returns almost nothing for these pages.

In such cases we convert the page into an image and
run OCR only on that page.

This keeps the ingestion pipeline fast while still
supporting scanned PDFs.
"""

def page_to_image(page):
    """
    Converts a PDF page into a PIL image.

    We render at 300 DPI because OCR quality improves
    significantly compared to the default resolution.
    """

    zoom = 300 / 72

    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=matrix)

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    return img


def perform_ocr(page):
    """
    Runs EasyOCR on a page.

    Returns extracted text.
    """

    image = page_to_image(page)

    image_np = np.array(image)

    results = ocr_reader.readtext(
        image_np,
        detail=0,
        paragraph=True
    )

    return clean_text(" ".join(results))


# =============================================================================
# HEADER / FOOTER DETECTION
# =============================================================================

"""
Most PDFs repeat the same header/footer.

Example:

Company Confidential
Page 5

Embedding these repeatedly hurts retrieval.

We'll detect repeated first/last lines across pages
and remove them.
"""


def detect_headers_and_footers(pdf):

    header_counter = Counter()

    footer_counter = Counter()

    pages = []

    for page in pdf:

        text = clean_text(page.get_text("text"))

        if not text:
            continue

        lines = text.split(". ")

        if not lines:
            continue

        pages.append(lines)

        header_counter.update(lines[:2])

        footer_counter.update(lines[-2:])

    headers = {

        line

        for line, count in header_counter.items()

        if count >= 3

    }

    footers = {

        line

        for line, count in footer_counter.items()

        if count >= 3

    }

    return headers, footers


def remove_headers_footers(text, headers, footers):

    if not text:
        return text

    for h in headers:
        text = text.replace(h, "")

    for f in footers:
        text = text.replace(f, "")

    return clean_text(text)


# =============================================================================
# SECTION DETECTION
# =============================================================================

"""
Instead of hardcoding only a few headings,
we support many common research-paper sections.
"""

SECTION_PATTERNS = [

    "abstract",

    "introduction",

    "background",

    "related work",

    "literature review",

    "problem statement",

    "objective",

    "dataset",

    "data collection",

    "methodology",

    "method",

    "implementation",

    "architecture",

    "experimental setup",

    "results",

    "discussion",

    "analysis",

    "evaluation",

    "limitations",

    "future work",

    "conclusion"

]


def detect_section(text):

    if not text:
        return "unknown"

    head = text[:1500].lower()

    for section in SECTION_PATTERNS:

        if section in head:

            return section.title()

    return "Unknown"


# =============================================================================
# TABLE EXTRACTION
# =============================================================================

"""
Tables are often flattened into meaningless text.

Instead we extract them separately and convert
them into Markdown.

Markdown tables embed surprisingly well.
"""


def extract_tables(pdf_path, page_number):

    markdown_tables = []

    try:

        with pdfplumber.open(pdf_path) as pdf:

            page = pdf.pages[page_number - 1]

            tables = page.extract_tables()

            for table in tables:

                if not table:

                    continue

                md = []

                for row in table:

                    row = [

                        "" if cell is None else str(cell)

                        for cell in row

                    ]

                    md.append("| " + " | ".join(row) + " |")

                markdown_tables.append("\n".join(md))

    except Exception as e:

        logger.warning(f"Table extraction failed: {e}")

    return markdown_tables


# =============================================================================
# IMAGE EXTRACTION
# =============================================================================

"""
Right now we only save image metadata.

Later we'll optionally connect this
to an image captioning model like BLIP.

This keeps the pipeline future-proof.
"""


def extract_images(page):

    images = []

    try:

        image_list = page.get_images(full=True)

        for img in image_list:

            images.append(

                {

                    "xref": img[0],

                    "width": img[2],

                    "height": img[3]

                }

            )

    except Exception:

        pass

    return images

# =============================================================================
# PDF EXTRACTION ENGINE
# =============================================================================

"""
This is the core of the ingestion pipeline.

For every page we:

1. Extract native text
2. If insufficient text -> OCR
3. Remove repeated headers & footers
4. Skip useless pages
5. Detect section
6. Extract tables
7. Detect images
8. Create a rich LangChain Document

Each page can generate multiple Documents:
- Main text
- Tables (if present)

This greatly improves retrieval quality.
"""


def extract_pdf_pages(pdf_path):

    docs = []

    pdf_name = os.path.basename(pdf_path)

    logger.info(f"Processing {pdf_name}")

    try:

        pdf = fitz.open(pdf_path)

    except Exception as e:

        logger.error(f"Could not open {pdf_name}: {e}")

        return docs

    headers, footers = detect_headers_and_footers(pdf)

    logger.info(f"Detected {len(headers)} repeated headers")

    logger.info(f"Detected {len(footers)} repeated footers")

    for page_number, page in enumerate(pdf, start=1):

        try:

            ############################################################
            # STEP 1 : Native Text Extraction
            ############################################################

            raw_text = page.get_text("text")

            raw_text = clean_text(raw_text)

            ############################################################
            # STEP 2 : OCR Fallback
            ############################################################

            if len(raw_text) < OCR_TEXT_THRESHOLD:

                logger.info(
                    f"{pdf_name} | Page {page_number}: OCR triggered"
                )

                raw_text = perform_ocr(page)

            ############################################################
            # STEP 3 : Remove Headers & Footers
            ############################################################

            raw_text = remove_headers_footers(

                raw_text,

                headers,

                footers

            )

            ############################################################
            # STEP 4 : Skip Empty Pages
            ############################################################

            if not raw_text:

                continue

            ############################################################
            # STEP 5 : Skip Low Value Pages
            ############################################################

            if is_low_value_page(raw_text):

                continue

            ############################################################
            # STEP 6 : Section Detection
            ############################################################

            section = detect_section(raw_text)

            ############################################################
            # STEP 7 : Tables
            ############################################################

            tables = extract_tables(

                pdf_path,

                page_number

            )

            ############################################################
            # STEP 8 : Images
            ############################################################

            images = extract_images(page)

            ############################################################
            # STEP 9 : Word Count
            ############################################################

            word_count = len(raw_text.split())

            ############################################################
            # STEP 10 : Main Page Document
            ############################################################

            docs.append(

                Document(

                    page_content=raw_text,

                    metadata={

                        "source": pdf_name,

                        "page": page_number,

                        "section": section,

                        "chunk_type": "text",

                        "word_count": word_count,

                        "num_tables": len(tables),

                        "num_images": len(images)

                    }

                )

            )

            ############################################################
            # STEP 11 : Table Documents
            ############################################################

            for table_number, table in enumerate(tables):

                docs.append(

                    Document(

                        page_content=table,

                        metadata={

                            "source": pdf_name,

                            "page": page_number,

                            "section": section,

                            "chunk_type": "table",

                            "table_number": table_number + 1

                        }

                    )

                )

        except Exception as e:

            logger.warning(

                f"{pdf_name} | Page {page_number} failed: {e}"

            )

            continue

    pdf.close()

    logger.info(

        f"{pdf_name}: Extracted {len(docs)} documents"

    )

    return docs


# =============================================================================
# LOAD ALL PDFs
# =============================================================================

"""
Scans data/pdfs/

Processes every PDF independently.

If one PDF fails,
the remaining PDFs still continue.
"""


def load_all_pdfs(pdf_dir=PDF_DIR):

    if not os.path.exists(pdf_dir):

        raise FileNotFoundError(

            f"{pdf_dir} does not exist."

        )

    pdf_files = sorted(

        [

            f

            for f in os.listdir(pdf_dir)

            if f.lower().endswith(".pdf")

        ]

    )

    if not pdf_files:

        raise FileNotFoundError(

            "No PDFs found."

        )

    all_docs = []

    logger.info(

        f"Found {len(pdf_files)} PDFs."

    )

    for pdf in pdf_files:

        path = os.path.join(

            pdf_dir,

            pdf

        )

        docs = extract_pdf_pages(path)

        all_docs.extend(docs)

    logger.info(

        f"Created {len(all_docs)} page-level documents."

    )

    return all_docs

# =============================================================================
# DOCUMENT QUALITY UTILITIES
# =============================================================================

"""
These functions improve chunk quality before embedding.

Instead of embedding every chunk generated by LangChain,
we first clean, filter and deduplicate them.

Better chunks
=
Better retrieval
=
Better chatbot answers.
"""


def is_good_chunk(text: str) -> bool:
    """
    Filters out low-quality chunks.

    Examples rejected:

    - very short chunks
    - only numbers
    - page numbers
    - repeated punctuation
    """

    if not text:
        return False

    text = clean_text(text)

    if len(text) < 80:
        return False

    words = text.split()

    if len(words) < 15:
        return False

    letters = sum(c.isalpha() for c in text)

    if letters < 40:
        return False

    return True


def chunk_hash(text: str):
    """
    Creates a unique fingerprint
    for every chunk.

    Used for duplicate removal.
    """

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()


# =============================================================================
# SEMANTIC CHUNKING
# =============================================================================

"""
LangChain still performs the actual splitting,
but we configure it carefully and then
post-process every chunk.
"""


def split_documents(documents):

    logger.info("Creating semantic chunks...")

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[

            "\n\n",

            "\n",

            ". ",

            "? ",

            "! ",

            "; ",

            ", ",

            " "

        ]

    )

    raw_chunks = splitter.split_documents(documents)

    logger.info(

        f"Initial chunks: {len(raw_chunks)}"

    )

    cleaned_chunks = []

    seen = set()

    chunk_counter = 1

    ############################################################
    # Clean every chunk
    ############################################################

    for chunk in raw_chunks:

        text = clean_text(

            chunk.page_content

        )

        ########################################################
        # Skip low-quality chunks
        ########################################################

        if not is_good_chunk(text):

            continue

        ########################################################
        # Remove duplicates
        ########################################################

        fingerprint = chunk_hash(text)

        if fingerprint in seen:

            continue

        seen.add(fingerprint)

        ########################################################
        # Preserve metadata
        ########################################################

        metadata = dict(

            chunk.metadata

        )

        metadata["chunk_id"] = chunk_counter

        metadata["chunk_hash"] = fingerprint

        metadata["character_count"] = len(text)

        metadata["token_estimate"] = len(text.split())

        ########################################################
        # Create final chunk
        ########################################################

        cleaned_chunks.append(

            Document(

                page_content=text,

                metadata=metadata

            )

        )

        chunk_counter += 1

    logger.info(

        f"Final chunks: {len(cleaned_chunks)}"

    )

    return cleaned_chunks


# =============================================================================
# CHUNK STATISTICS
# =============================================================================

"""
Purely for logging.

Helps understand whether chunking
worked as expected.
"""


def print_chunk_statistics(chunks):

    if not chunks:

        logger.warning("No chunks created.")

        return

    lengths = [

        len(chunk.page_content)

        for chunk in chunks

    ]

    avg = sum(lengths) / len(lengths)

    logger.info("=" * 60)

    logger.info("Chunk Statistics")

    logger.info(f"Total Chunks : {len(chunks)}")

    logger.info(f"Average Size : {avg:.1f}")

    logger.info(f"Smallest     : {min(lengths)}")

    logger.info(f"Largest      : {max(lengths)}")

    logger.info("=" * 60)


# =============================================================================
# VECTOR DATABASE
# =============================================================================

def build_vectorstore(chunks):

    logger.info("=" * 60)
    logger.info("Building Chroma Vector Database")

    ids = []

    for chunk in chunks:

        meta = chunk.metadata

        ids.append(

            f"{meta['source']}_"
            f"{meta['page']}_"
            f"{meta['chunk_id']}"

        )

    Chroma.from_documents(

        documents=chunks,

        embedding=embedding_model,

        ids=ids,

        persist_directory=PERSIST_DIR

    )

    logger.info("Vector database created successfully.")


# =============================================================================
# DATABASE SUMMARY
# =============================================================================

def print_database_summary(chunks):

    logger.info("=" * 60)

    logger.info("DATABASE SUMMARY")

    logger.info("=" * 60)

    pdfs = {

        chunk.metadata["source"]

        for chunk in chunks

    }

    pages = {

        (

            chunk.metadata["source"],

            chunk.metadata["page"]

        )

        for chunk in chunks

    }

    logger.info(f"PDFs           : {len(pdfs)}")

    logger.info(f"Pages          : {len(pages)}")

    logger.info(f"Total Chunks   : {len(chunks)}")

    logger.info("=" * 60)

# =============================================================================
# MAIN
# =============================================================================

def main():

    logger.info("=" * 60)

    logger.info("Enterprise RAG Ingestion")

    logger.info("=" * 60)

    if os.path.exists(PERSIST_DIR):

        logger.warning(

            f"{PERSIST_DIR} already exists."

        )

        logger.warning(

            "Delete it first if you want to rebuild."

        )

        return

    ###########################################################

    docs = load_all_pdfs()

    logger.info(f"Loaded {len(docs)} documents.")

    ###########################################################

    chunks = split_documents(docs)

    ###########################################################

    print_chunk_statistics(chunks)

    ###########################################################

    build_vectorstore(chunks)

    ###########################################################

    print_database_summary(chunks)

    ###########################################################

    logger.info("")

    logger.info("INGESTION COMPLETED SUCCESSFULLY")

    logger.info("")


if __name__ == "__main__":

    main() 