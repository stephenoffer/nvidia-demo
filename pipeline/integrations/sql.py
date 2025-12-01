"""SQL data access for multimodal datasets.

Provides SQL-based querying and data access for structured data sources.
Supports PostgreSQL, MySQL, and SQLite for flexible data integration.

Uses Ray Data for distributed SQL queries.
See: https://docs.ray.io/en/latest/data/data.html
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import ray  # https://docs.ray.io/
from ray.data import Dataset  # https://docs.ray.io/en/latest/data/data.html

from pipeline.exceptions import DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time
from pipeline.utils.constants import _GPU_BATCH_SIZE

logger = logging.getLogger(__name__)

# Constants
_DEFAULT_CONNECTION_TIMEOUT = 30
_DEFAULT_POOL_RECYCLE = 3600
_MAX_QUERY_RESULT_SIZE = 10 * 1024 * 1024 * 1024  # 10GB


class SQLDataLoader:
    """SQL data loader for structured data sources.

    Supports reading from SQL databases and converting to Ray Datasets
    for integration with the multimodal pipeline.
    """

    def __init__(
        self,
        connection_string: str,
        query: str,
        database_type: str = "postgresql",
        max_result_size: Optional[int] = None,
    ):
        """Initialize SQL data loader.

        Args:
            connection_string: Database connection string
            query: SQL query to execute
            database_type: Type of database ('postgresql', 'mysql', 'sqlite')
            max_result_size: Maximum result size in bytes (None = use default)

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not isinstance(connection_string, str) or not connection_string.strip():
            raise ValueError(f"connection_string must be non-empty str, got {type(connection_string)}")
        
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"query must be non-empty str, got {type(query)}")
        
        valid_db_types = {"postgresql", "mysql", "sqlite"}
        if database_type not in valid_db_types:
            raise ValueError(f"database_type must be one of {valid_db_types}, got {database_type}")
        
        if max_result_size is not None and max_result_size <= 0:
            raise ValueError(f"max_result_size must be positive, got {max_result_size}")
        
        self.connection_string = connection_string
        self.query = query
        self.database_type = database_type
        self.max_result_size = max_result_size if max_result_size is not None else _MAX_QUERY_RESULT_SIZE

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def load(self) -> Dataset:
        """Load data from SQL database.

        Uses pandas/SQLAlchemy to read data, then converts to Ray Dataset.
        Ensures proper connection cleanup to prevent resource leaks.

        Returns:
            Ray Dataset containing query results

        Raises:
            DataSourceError: If loading fails
        """
        logger.info(f"Loading data from {self.database_type} database")

        try:
            import pandas as pd  # https://pandas.pydata.org/
            from sqlalchemy import create_engine  # https://www.sqlalchemy.org/
        except ImportError as e:
            raise DataSourceError(f"Required dependencies not installed: {e}") from e

        # Create database engine with connection pooling and timeout
        # Use pool_pre_ping to detect stale connections
        engine = None
        try:
            engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=_DEFAULT_POOL_RECYCLE,  # Recycle connections after 1 hour
                connect_args={"connect_timeout": _DEFAULT_CONNECTION_TIMEOUT},  # Connection timeout
            )

            # Read data using pandas with timeout
            try:
                df = pd.read_sql(self.query, engine, chunksize=None)
                
                # Check result size
                if hasattr(df, 'memory_usage'):
                    memory_usage = df.memory_usage(deep=True).sum()
                    if memory_usage > self.max_result_size:
                        raise DataSourceError(
                            f"Query result size {memory_usage} bytes exceeds "
                            f"maximum {self.max_result_size} bytes"
                        )
                
                # CRITICAL: Do NOT use cuDF here - SQL loader reads entire dataset into memory
                # Converting to cuDF and back provides no benefit and wastes GPU memory
                # Ray Data will handle GPU acceleration during map_batches operations
                # Keep pandas DataFrame for Ray Data compatibility
                logger.info(f"Loaded {len(df)} rows from SQL database")

                # Convert pandas DataFrame to Ray Dataset
                dataset = ray.data.from_pandas([df])

                return dataset
            except Exception as e:
                raise DataSourceError(f"Failed to execute SQL query: {e}") from e
            finally:
                # Ensure engine is properly closed
                if engine is not None:
                    engine.dispose()
                    logger.debug("SQL engine disposed")

        except Exception as e:
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass
            raise DataSourceError(f"Failed to load data from SQL database: {e}") from e

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def write_results(self, dataset: Dataset, table_name: str) -> None:
        """Write dataset results back to SQL database.

        Converts Ray Dataset to pandas and writes using SQLAlchemy.
        Ensures proper connection cleanup and handles large datasets efficiently.

        Args:
            dataset: Ray Dataset to write
            table_name: Target table name

        Raises:
            ValueError: If parameters are invalid
            DataSourceError: If writing fails
        """
        # Validate parameters
        if not isinstance(table_name, str) or not table_name.strip():
            raise ValueError(f"table_name must be non-empty str, got {type(table_name)}")
        
        logger.info(f"Writing dataset to {table_name} (using streaming batches)")

        try:
            import pandas as pd  # https://pandas.pydata.org/
            from sqlalchemy import create_engine  # https://www.sqlalchemy.org/
        except ImportError as e:
            raise DataSourceError(f"Required dependencies not installed: {e}") from e

        # Create database engine with connection pooling
        engine = None
        try:
            engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_recycle=_DEFAULT_POOL_RECYCLE,
                connect_args={"connect_timeout": _DEFAULT_CONNECTION_TIMEOUT},
            )

            # Convert Ray Dataset to pandas DataFrame in batches
            # Use iter_batches for large datasets to avoid memory issues
            batch_size = _GPU_BATCH_SIZE
            first_batch = True

            for batch in dataset.iter_batches(batch_size=batch_size):
                try:
                    df_batch = batch if isinstance(batch, pd.DataFrame) else pd.DataFrame(batch)

                    # Use GPU-accelerated operations for large batches
                    try:
                        from pipeline.utils.gpu.etl import gpu_drop_duplicates
                        
                        if len(df_batch) > 1000:
                            # Use GPU-accelerated duplicate removal before writing
                            df_batch = gpu_drop_duplicates(df_batch, num_gpus=1)
                    except ImportError:
                        pass  # Fallback to CPU operations

                    # Write batch to database
                    df_batch.to_sql(
                        table_name,
                        engine,
                        if_exists="append" if not first_batch else "replace",
                        index=False,
                        method="multi",  # Use multi-row insert for efficiency
                        chunksize=1000,  # Insert in chunks
                    )
                    first_batch = False
                    logger.debug(f"Wrote batch of {len(df_batch)} rows to {table_name}")
                except Exception as e:
                    logger.error(f"Failed to write batch to {table_name}: {e}")
                    raise DataSourceError(f"Failed to write batch: {e}") from e

            logger.info(f"Successfully wrote data to {table_name}")
        except Exception as e:
            raise DataSourceError(f"Failed to write data to SQL database: {e}") from e
        finally:
            # Ensure engine is properly closed
            if engine is not None:
                try:
                    engine.dispose()
                    logger.debug("SQL engine disposed")
                except Exception:
                    pass
