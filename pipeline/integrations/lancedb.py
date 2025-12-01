"""Integration with LanceDB for vector storage and similarity search."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from pipeline.exceptions import DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time

logger = logging.getLogger(__name__)

# Constants
_DEFAULT_TABLE_NAME = "embeddings"
_DEFAULT_EMBEDDING_DIM = 768
_DEFAULT_LIMIT = 10
_DEFAULT_MAX_DISTANCE = 0.3
_MAX_EMBEDDINGS_BATCH = 10000


class LanceDBStorage:
    """LanceDB integration for embedding storage and similarity search."""

    def __init__(
        self,
        db_path: Union[str, Path],
        table_name: str = _DEFAULT_TABLE_NAME,
        embedding_dim: int = _DEFAULT_EMBEDDING_DIM,
    ):
        """Initialize LanceDB storage.

        Args:
            db_path: Path to LanceDB database directory
            table_name: Name of the table to use
            embedding_dim: Dimension of embeddings

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate and convert path
        if isinstance(db_path, str):
            if not db_path or not db_path.strip():
                raise ValueError("db_path cannot be empty")
            self.db_path = Path(db_path)
        elif isinstance(db_path, Path):
            self.db_path = db_path
        else:
            raise ValueError(f"db_path must be str or Path, got {type(db_path)}")
        
        # Validate parameters
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError(f"table_name must be non-empty str, got {type(table_name)}")
        
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive int, got {embedding_dim}")
        
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self._db = None
        self._table = None

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def store_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadata: Optional[Sequence[dict[str, Any]]] = None,
    ) -> None:
        """Store embeddings in LanceDB.

        Args:
            embeddings: Sequence of embedding vectors
            texts: Sequence of text strings
            metadata: Optional sequence of metadata dictionaries

        Raises:
            ValueError: If parameters are invalid
            DataSourceError: If storage fails
        """
        # Validate parameters
        if not isinstance(embeddings, Sequence):
            raise ValueError(f"embeddings must be Sequence, got {type(embeddings)}")
        
        if not isinstance(texts, Sequence):
            raise ValueError(f"texts must be Sequence, got {type(texts)}")
        
        if metadata is not None and not isinstance(metadata, Sequence):
            raise ValueError(f"metadata must be Sequence or None, got {type(metadata)}")
        
        self._ensure_connection()
        self._validate_lengths(embeddings, texts, metadata)
        
        # Process in batches to avoid memory issues
        embeddings_list = list(embeddings)
        texts_list = list(texts)
        metadata_list = list(metadata) if metadata else None
        
        total_items = len(embeddings_list)
        for batch_start in range(0, total_items, _MAX_EMBEDDINGS_BATCH):
            batch_end = min(batch_start + _MAX_EMBEDDINGS_BATCH, total_items)
            
            batch_embeddings = embeddings_list[batch_start:batch_end]
            batch_texts = texts_list[batch_start:batch_end]
            batch_metadata = metadata_list[batch_start:batch_end] if metadata_list else None
            
            payload = self._prepare_records(batch_embeddings, batch_texts, batch_metadata)
            
            try:
                self._table.add(payload)
                logger.info(f"Stored {len(payload)} embeddings in LanceDB (batch {batch_start}-{batch_end})")
            except Exception as e:
                logger.error(f"Failed to store embeddings batch: {e}")
                raise DataSourceError(f"Failed to store embeddings: {e}") from e

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def search_similar(
        self,
        query_embedding: Sequence[float],
        limit: int = _DEFAULT_LIMIT,
        max_distance: float = _DEFAULT_MAX_DISTANCE,
    ) -> list[dict[str, Any]]:
        """Search for similar embeddings (lower distance = closer).

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            max_distance: Maximum distance threshold

        Returns:
            List of similar embeddings with metadata

        Raises:
            ValueError: If parameters are invalid
            DataSourceError: If search fails
        """
        # Validate parameters
        if not isinstance(query_embedding, Sequence):
            raise ValueError(f"query_embedding must be Sequence, got {type(query_embedding)}")
        
        query_list = list(query_embedding)
        if len(query_list) != self.embedding_dim:
            raise ValueError(
                f"query_embedding dimension {len(query_list)} != expected {self.embedding_dim}"
            )
        
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be positive int, got {limit}")
        
        if not isinstance(max_distance, (int, float)) or max_distance <= 0:
            raise ValueError(f"max_distance must be positive number, got {max_distance}")

        self._ensure_connection()
        
        try:
            results = (
                self._table.search(query_list)
                .limit(limit)
                .to_pandas()
            )
            
            if results.empty:
                return []
            
            # Filter by max_distance
            filtered = results[results["_distance"] <= max_distance]
            
            # Convert to list of dicts
            return filtered.rename(columns={"_distance": "distance"}).to_dict("records")
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            raise DataSourceError(f"Failed to search similar embeddings: {e}") from e

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def create_index(self) -> None:
        """Trigger LanceDB index creation (no-op if already built).

        Raises:
            DataSourceError: If index creation fails
        """
        self._ensure_connection()
        logger.info("Vector index is managed automatically by LanceDB")

    def _ensure_connection(self) -> None:
        """Ensure LanceDB connection is established.

        Raises:
            DataSourceError: If connection fails
        """
        if self._db is not None:
            return
        
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as exc:
            raise DataSourceError(
                "LanceDB support requires `lancedb` and `pyarrow` packages"
            ) from exc

        try:
            # Ensure database directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            self._db = lancedb.connect(str(self.db_path))
            
            if self.table_name not in self._db.table_names():
                schema = pa.schema(
                    [
                        pa.field("id", pa.string()),
                        pa.field(
                            "embedding",
                            pa.list_(pa.float32(), self.embedding_dim),
                        ),
                        pa.field("text", pa.string()),
                        pa.field("metadata", pa.string()),
                    ]
                )
                self._table = self._db.create_table(self.table_name, schema=schema)
                logger.info(f"Created LanceDB table: {self.table_name}")
            else:
                self._table = self._db.open_table(self.table_name)
                logger.debug(f"Opened existing LanceDB table: {self.table_name}")
        except Exception as e:
            raise DataSourceError(f"Failed to connect to LanceDB: {e}") from e

    def _validate_lengths(
        self,
        embeddings: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadata: Optional[Sequence[dict[str, Any]]],
    ) -> None:
        """Validate that sequences have matching lengths.

        Args:
            embeddings: Sequence of embeddings
            texts: Sequence of texts
            metadata: Optional sequence of metadata

        Raises:
            ValueError: If lengths don't match
        """
        num_embeddings = len(embeddings)
        if num_embeddings == 0:
            raise ValueError("embeddings cannot be empty")
        
        if num_embeddings != len(texts):
            raise ValueError(f"embeddings ({num_embeddings}) and texts ({len(texts)}) must have the same length")
        
        if metadata is not None and len(metadata) != num_embeddings:
            raise ValueError(f"metadata length ({len(metadata)}) must match embeddings length ({num_embeddings})")

    def _prepare_records(
        self,
        embeddings: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadata: Optional[Sequence[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Prepare records for LanceDB insertion.

        Args:
            embeddings: Sequence of embeddings
            texts: Sequence of texts
            metadata: Optional sequence of metadata

        Returns:
            List of record dictionaries

        Raises:
            ValueError: If embeddings have wrong dimension
        """
        records: list[dict[str, Any]] = []
        
        for idx, (embedding, text) in enumerate(zip(embeddings, texts)):
            if not isinstance(text, str):
                raise ValueError(f"texts[{idx}] must be str, got {type(text)}")
            
            vector = list(embedding)
            if len(vector) != self.embedding_dim:
                raise ValueError(
                    f"Embedding {idx} has dimension {len(vector)}, "
                    f"expected {self.embedding_dim}"
                )
            
            # Validate embedding values are numeric
            try:
                vector = [float(v) for v in vector]
            except (TypeError, ValueError) as e:
                raise ValueError(f"Embedding {idx} contains non-numeric values: {e}") from e
            
            record = {
                "id": self._generate_record_id(),
                "embedding": vector,
                "text": text,
                "metadata": json.dumps(metadata[idx]) if metadata else "{}",
            }
            records.append(record)
        
        return records

    @staticmethod
    def _generate_record_id() -> str:
        """Generate a unique record ID.

        Returns:
            Unique ID string
        """
        return uuid.uuid4().hex
