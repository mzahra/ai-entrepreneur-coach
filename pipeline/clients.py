import os

import cohere
from pinecone import Pinecone

COHERE_EMBED_MODEL = "embed-v4.0"
PINECONE_INDEX_NAME = "entrepreneur-coach-ideas"

_cohere_client = None
_pinecone_index = None


def get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    return _cohere_client


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index
