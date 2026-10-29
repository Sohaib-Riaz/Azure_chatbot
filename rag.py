
"""Restaurant menu RAG: TF-IDF + lightweight multilingual embeddings."""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from ingest import parse_pdf

# Small multilingual embedding model
DENSE_MODEL = os.getenv(
    "RAG_EMBED_MODEL",
    "intfloat/multilingual-e5-small"
)

class MenuRAG:
    def __init__(self, pdf_path: str, use_dense: bool = True):
        self.chunks = parse_pdf(pdf_path)
        self.texts = [chunk.text for chunk in self.chunks]

        # Keyword search: works even if embeddings are unavailable
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True
        )
        self.tfidf_mat = self.tfidf.fit_transform(self.texts)

        self.embedder = None
        self.dense_mat = None

        if use_dense:
            try:
                from fastembed import TextEmbedding

                print("Loading embedding model...")
                self.embedder = TextEmbedding(model_name=DENSE_MODEL)

                # E5 models use "passage:" for documents
                passages = [
                    "passage: " + text for text in self.texts
                ]
                self.dense_mat = np.array(
                    list(self.embedder.embed(passages))
                )

                # Normalize vectors
                norms = np.linalg.norm(
                    self.dense_mat, axis=1, keepdims=True
                )
                self.dense_mat = self.dense_mat / np.maximum(norms, 1e-12)

                print("Embedding model loaded successfully.")

            except Exception as e:
                print(
                    f"Embeddings unavailable: {e}\n"
                    "Using TF-IDF keyword search instead."
                )
                self.embedder = None
                self.dense_mat = None

    def search(
        self,
        query: str,
        k: int = 5,
        veg_only: bool = False,
        max_price: int | None = None,
        section: str | None = None
    ):
        # Start with keyword search
        q = self.tfidf.transform([query])
        scores = (self.tfidf_mat @ q.T).toarray().ravel()

        # Add semantic search scores if embeddings are available
        if self.embedder is not None and self.dense_mat is not None:
            query_vector = np.array(
                list(self.embedder.query_embed(["query: " + query]))[0]
            )
            query_vector = query_vector / max(
                np.linalg.norm(query_vector), 1e-12
            )

            semantic_scores = self.dense_mat @ query_vector
            scores = 0.4 * scores + 0.6 * semantic_scores

        # Apply filters
        for i, chunk in enumerate(self.chunks):
            if chunk.kind == "dish":
                if veg_only and not chunk.vegetarian:
                    scores[i] = -1
                if max_price is not None:
                    if chunk.price is None or chunk.price > max_price:
                        scores[i] = -1
                if section and section.lower() not in chunk.section.lower():
                    scores[i] = -1
            elif veg_only or max_price is not None or section:
                scores[i] = -1

        # Return the highest-scoring relevant chunks
        top_indices = np.argsort(-scores)[:k]

        return [
            (self.chunks[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    @staticmethod
    def build_context(hits) -> str:
        return "\n\n".join(
            f"[{chunk.id}] {chunk.text}"
            for chunk, score in hits
        )


if __name__ == "__main__":
    pdf_path = "Desi_Restaurant_Menu.pdf"
    rag = MenuRAG(pdf_path)

    print("\nMenu search ready. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            break

        results = rag.search(question)

        if not results:
            print("Bot: Menu mein is bare mein relevant information nahi mili.")
        else:
            print("\nRelevant menu items:")
            for chunk, score in results:
                print(f"- {chunk.name}: {chunk.text}")
