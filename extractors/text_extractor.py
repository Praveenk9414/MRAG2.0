import os
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_and_tables(pdf_path):
    """
    Extracts text and tables from a PDF and returns a list of Document objects.
    Each Document includes metadata for page number, type, and source_file.
    """
    docs = []
    pdf_name = os.path.basename(pdf_path)  # get file name for metadata

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):  # start=1 for page numbering
            text = page.extract_text()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "page_no": i,
                        "type": "text",
                        "source_file": pdf_name  # added source_file
                    }
                ))

            # Extract tables
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables or []):
                # Replace None with empty string
                safe_table = [[cell if cell is not None else "" for cell in row] for row in table]
                table_text = "\n".join([" | ".join(row) for row in safe_table])
                docs.append(Document(
                    page_content=table_text,
                    metadata={
                        "page_no": i,
                        "type": "table",
                        "table_index": t_idx + 1,
                        "source_file": pdf_name  # added source_file
                    }
                ))
    return docs


def chunk_documents(docs):
    """
    Splits large text chunks into smaller Document objects while preserving metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunked_docs = []

    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            # Keep original metadata and just update page_content
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    return chunked_docs
