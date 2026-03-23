from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .settings import COLLECTION_NAME, DEVICE


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
)

vector_size = len(embeddings.embed_query("sample text"))
print(vector_size)

client = QdrantClient(url="http://localhost:6333")
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

qdrant = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

def retrieve_scored_nodes(query: str, k: int, score_threshold: float):
    return qdrant.similarity_search_with_relevance_scores(
        query,
        k=k,
        score_threshold=score_threshold,
    )
