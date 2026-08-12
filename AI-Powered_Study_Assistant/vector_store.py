from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import EMBEDDING_MODEL


def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_vector_store(
    docs: List[Document]
) -> FAISS:
  
    embeddings = get_embedding_function()

    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )

    return vector_store