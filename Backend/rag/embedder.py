"""
Embedder
--------
Wraps Amazon Titan Text Embeddings V2 (amazon.titan-embed-text-v2:0)
via AWS Bedrock to produce 1024-dimensional vectors.

Key design choices:
  - Single boto3 session reused across calls (thread-safe for read ops).
  - Exponential back-off on throttling (Bedrock rate limits per second).
  - embed_texts() batches input so callers don't need to think about limits.
  - Normalised=True so cosine similarity == dot product in Pinecone.
"""

import json
import logging
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    EMBEDDING_MODEL_ID,
    EMBEDDING_DIMENSIONS,
)

logger = logging.getLogger(__name__)

# Titan V2 hard limit per request
_TITAN_MAX_CHARS = 8_000   # ~2 000 tokens; well under 8 192-token hard limit

# Back-off settings for throttle retries
_MAX_RETRIES = 5
_BASE_BACKOFF = 1.0   # seconds


class TitanEmbedder:
    """
    Singleton-style embedder for Amazon Titan Text Embeddings V2.

    Usage:
        embedder = TitanEmbedder()
        vectors = embedder.embed_texts(["text one", "text two"])
    """

    def __init__(self):
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION,
        )
        self._client = session.client("bedrock-runtime")
        self.model_id = EMBEDDING_MODEL_ID
        self.dimensions = EMBEDDING_DIMENSIONS
        logger.info(f"TitanEmbedder ready — model={self.model_id}, dims={self.dimensions}")

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings.  Returns a parallel list of float vectors.
        Handles truncation and retries internally.
        """
        vectors = []
        for text in texts:
            vec = self._embed_single(text)
            vectors.append(vec)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (convenience wrapper)."""
        return self._embed_single(text)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _embed_single(self, text: str) -> list[float]:
        """Call Bedrock for one text with exponential back-off on throttling."""
        # Titan silently truncates; we pre-truncate to avoid surprises
        truncated = text[:_TITAN_MAX_CHARS]

        payload = json.dumps({
            "inputText": truncated,
            "dimensions": self.dimensions,
            "normalize": True,
        })

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=payload,
                )
                body = json.loads(response["body"].read())
                return body["embedding"]

            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code in ("ThrottlingException", "ServiceUnavailableException"):
                    wait = _BASE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        f"Bedrock throttled (attempt {attempt + 1}/{_MAX_RETRIES}). "
                        f"Retrying in {wait:.1f}s…"
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(
            f"Bedrock embedding failed after {_MAX_RETRIES} retries for text: {truncated[:80]!r}"
        )


# Module-level singleton — import and reuse everywhere
_embedder: Optional[TitanEmbedder] = None


def get_embedder() -> TitanEmbedder:
    """Return the module-level TitanEmbedder singleton."""
    global _embedder
    if _embedder is None:
        _embedder = TitanEmbedder()
    return _embedder
