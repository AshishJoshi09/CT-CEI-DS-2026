"""
Query Processing -> Context Retrieval -> Answer Generation

"""

from groq import Groq

from embedder import Embedder
from vector_store import VectorStore
import config


SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context from the user's documents. If the answer isn't in the context,
say so clearly instead of guessing. Keep answers concise and grounded in the context.
When useful, mention which source/page the information came from."""


class RAGPipeline:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def retrieve(self, query: str, top_k: int = None):
        top_k = top_k or config.TOP_K
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, top_k)

    def _build_context(self, results) -> str:
        blocks = []
        for chunk, score in results:
            page_info = f", page {chunk.page}" if chunk.page else ""
            blocks.append(f"[Source: {chunk.source}{page_info}]\n{chunk.text}")
        return "\n\n---\n\n".join(blocks)

    def generate_answer(self, query: str, results) -> str:
        context = self._build_context(results)
        user_message = f"""Context from documents:

{context}

Question: {query}

Answer the question using only the context above."""

        response = self.client.chat.completions.create(
            model=config.GENERATION_MODEL,
            max_tokens=config.MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def answer(self, query: str, top_k: int = None, verbose: bool = False) -> str:
        results = self.retrieve(query, top_k)

        if verbose:
            print("\nRetrieved chunks:")
            for chunk, score in results:
                page_info = f" (page {chunk.page})" if chunk.page else ""
                print(f"  - {chunk.source}{page_info} | score={score:.3f}")
            print()

        return self.generate_answer(query, results)
