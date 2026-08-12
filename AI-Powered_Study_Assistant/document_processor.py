import tempfile

from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP


def process_uploaded_pdf(uploaded_file) -> List[Document]:

    # Create temporary PDF file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.getvalue()
        )

        tmp_file_path = tmp_file.name


    # Load PDF
    loader = PyPDFLoader(
        tmp_file_path
    )

    documents = loader.load()


    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )


    # Split documents into chunks
    chunks = text_splitter.split_documents(
        documents
    )


    return chunks