"""Persistent vector storage and cosine-similarity search with pgvector."""
import json
import math
from typing import Any

import psycopg

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url.strip():
            raise ValueError("DATABASE_URL must not be empty")
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.database_url = database_url
        self.dimensions = dimensions

    @staticmethod
    def prepare_database(database_url: str) -> dict[str, str]:
        if not database_url.strip():
            raise ValueError("DATABASE_URL must not be empty")
        with psycopg.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ingestion_chunks (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        embedding vector NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS ingestion_chunks_strategy_idx "
                    "ON ingestion_chunks (strategy)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS ingestion_chunks_metadata_idx "
                    "ON ingestion_chunks USING GIN (metadata)"
                )
                cursor.execute(
                    """
                    SELECT current_setting('server_version'), extversion
                    FROM pg_extension
                    WHERE extname = 'vector'
                    """
                )
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("pgvector extension is not installed in the database")
        return {"postgres_version": row[0], "pgvector_version": row[1]}

    def server_info(self) -> dict[str, str]:
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT current_setting('server_version'),
                           extversion
                    FROM pg_extension
                    WHERE extname = 'vector'
                    """
                )
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("pgvector extension is not installed in the database")
        return {"postgres_version": row[0], "pgvector_version": row[1]}

    def create_schema(self) -> None:
        self.prepare_database(self.database_url)

    def _vector_literal(self, vector: list[float]) -> str:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected vector dimension {self.dimensions}, got {len(vector)}"
            )
        values = [float(value) for value in vector]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("vectors must contain only finite numbers")
        if not any(value != 0 for value in values):
            raise ValueError("cosine search does not accept zero vectors")
        return "[" + ",".join(repr(value) for value in values) + "]"

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Atomically replace a strategy snapshot, removing obsolete chunks."""
        if not strategy.strip():
            raise ValueError("strategy must not be empty")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("chunk IDs must be unique within a strategy")

        rows = []
        for chunk, vector in zip(chunks, vectors):
            if not chunk.document_id or not chunk.source or chunk.page < 1:
                raise ValueError(f"chunk {chunk.chunk_id} has invalid source metadata")
            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "page": chunk.page,
                }
            )
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    json.dumps(metadata, ensure_ascii=False),
                    self._vector_literal(vector),
                )
            )
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM ingestion_chunks WHERE strategy = %s", (strategy,)
                )
                if rows:
                    cursor.executemany(
                        """
                        INSERT INTO ingestion_chunks
                            (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            source = EXCLUDED.source,
                            page = EXCLUDED.page,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        rows,
                    )

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not strategy.strip():
            raise ValueError("strategy must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        vector = self._vector_literal(query_vector)
        if filters is None:
            filters = {}
        if not isinstance(filters, dict):
            raise ValueError("filters must be a metadata object")
        filter_json = json.dumps(filters, ensure_ascii=False)

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    WITH query_vector AS (SELECT %s::vector AS embedding)
                    SELECT chunks.chunk_id, chunks.document_id, chunks.source, chunks.page,
                           chunks.text, chunks.metadata,
                           1 - (chunks.embedding <=> query_vector.embedding) AS score
                    FROM ingestion_chunks AS chunks
                    CROSS JOIN query_vector
                    WHERE chunks.strategy = %s
                      AND chunks.metadata @> %s::jsonb
                    ORDER BY chunks.embedding <=> query_vector.embedding
                    LIMIT %s
                    """,
                    (vector, strategy, filter_json, top_k),
                )
                rows = cursor.fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]
