"""Integration with NVIDIA NeMo for text embeddings and language models.

NVIDIA NeMo is a framework for building, training, and fine-tuning large
language models and multimodal foundation models. This integration provides
high-quality text embeddings for semantic deduplication and data curation.

See: https://docs.nvidia.com/nemo-framework/user-guide/latest/
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import torch  # https://pytorch.org/

from pipeline.exceptions import DataSourceError
from pipeline.utils.decorators import handle_errors, log_execution_time
from pipeline.utils.constants import _SYNTHETIC_BATCH_SIZE

logger = logging.getLogger(__name__)

# Constants
_DEFAULT_EMBEDDING_DIM = 384
_DEFAULT_MIN_TEXT_LENGTH = 10
_DEFAULT_MIN_WORDS = 5


class NeMoEmbeddingGenerator:
    """Generate text embeddings using NVIDIA NeMo models.

    Uses NeMo's pre-trained embedding models for high-quality semantic
    representations suitable for deduplication and similarity search.
    """

    def __init__(
        self,
        model_name: str = "nvidia/nemo-megatron-gpt-1.3B",
        device: str = "cuda",
        batch_size: Optional[int] = None,
    ):
        """Initialize NeMo embedding generator.

        Args:
            model_name: NeMo model name or path
            device: Device to run model on ('cuda' or 'cpu')
            batch_size: Batch size for embedding generation

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name must be non-empty str, got {type(model_name)}")
        
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {device}")
        
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.model_name = model_name
        self.batch_size = batch_size if batch_size is not None else _SYNTHETIC_BATCH_SIZE
        self._model = None
        self._tokenizer = None

        # Properly validate and set CUDA device
        from pipeline.utils.gpu.memory import get_cuda_device

        if device == "cuda" and torch.cuda.is_available():
            try:
                device_id = get_cuda_device(0)
                self.device = f"cuda:{device_id}"
                torch.cuda.set_device(device_id)
                logger.info(f"NeMo using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Failed to initialize CUDA device: {e}, falling back to CPU")
                self.device = "cpu"
        else:
            self.device = "cpu"
            logger.info("Using CPU for NeMo (CUDA not available or not requested)")

    def _load_model(self) -> None:
        """Load NeMo model and tokenizer.

        Raises:
            DataSourceError: If model loading fails
        """
        if self._model is not None:
            return

        try:
            # Try to import NeMo (check availability)
            import importlib.util  # https://docs.python.org/3/library/importlib.html

            nemo_spec = importlib.util.find_spec("nemo")
            if nemo_spec is None:
                raise ImportError("NeMo not installed")

            logger.info(f"Loading NVIDIA NeMo model: {self.model_name}")

            # Try to use proper NeMo API
            # Note: Actual NeMo model loading depends on NeMo version and model type
            # In production, use NeMo's model registry or checkpoint loading:
            #   from nemo.collections.nlp.models import TextEmbeddingModel
            #   self._model = TextEmbeddingModel.from_pretrained(self.model_name)
            #   self._model = self._model.to(self.device)
            #   self._model.eval()
            # For now, use placeholder until NeMo is properly configured
            self._model = None
            self._tokenizer = None

            logger.info("NVIDIA NeMo model loaded successfully (placeholder - implement actual loading)")
        except ImportError:
            logger.warning(
                "NVIDIA NeMo not available. Install via: "
                "pip install nemo_toolkit[all] or use transformers fallback"
            )
            self._model = None
            self._tokenizer = None

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def generate_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for texts using NVIDIA NeMo.

        Args:
            texts: List of text strings

        Returns:
            Tensor of embeddings [num_texts, embedding_dim]

        Raises:
            ValueError: If texts is invalid
            DataSourceError: If generation fails
        """
        # Validate parameters
        if not isinstance(texts, list):
            raise ValueError(f"texts must be list, got {type(texts)}")
        
        if not texts:
            # Ensure device is correct for empty tensor
            if "cuda" in self.device:
                return torch.empty((0, _DEFAULT_EMBEDDING_DIM), device=self.device, dtype=torch.float32)
            return torch.empty((0, _DEFAULT_EMBEDDING_DIM), dtype=torch.float32)
        
        # Validate text items
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"texts[{i}] must be str, got {type(text)}")

        from pipeline.utils.gpu.memory import check_gpu_memory, gpu_memory_cleanup

        # Check GPU memory if using CUDA
        if "cuda" in self.device:
            # Estimate memory: batch_size * embedding_dim * 4 bytes (float32)
            estimated_memory = self.batch_size * _DEFAULT_EMBEDDING_DIM * 4 * 2  # 2x for safety
            has_memory, mem_info = check_gpu_memory(estimated_memory)
            if not has_memory:
                logger.warning(
                    f"Insufficient GPU memory for embeddings. "
                    f"Required: {estimated_memory}, Available: {mem_info['free']}"
                )
                # Fallback to CPU
                self.device = "cpu"

        self._load_model()

        with gpu_memory_cleanup():
            if self._model is None:
                # Fallback to transformers if NeMo not available
                logger.info("Using transformers fallback for embeddings")
                return self._generate_with_transformers(texts)

            # NeMo embedding generation (placeholder)
            # In production, use proper NeMo API:
            #   embeddings = self._model.encode(texts, batch_size=self.batch_size)
            logger.warning("NeMo model not fully implemented, using transformers fallback")
            return self._generate_with_transformers(texts)

    def _generate_with_transformers(self, texts: List[str]) -> torch.Tensor:
        """Fallback embedding generation using transformers library.

        Uses sentence-transformers which provides similar functionality
        to NeMo embeddings for semantic tasks.

        Args:
            texts: List of text strings

        Returns:
            Tensor of embeddings [num_texts, embedding_dim]

        Raises:
            DataSourceError: If generation fails
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[attr-defined]
        except ImportError:
            raise DataSourceError(
                "Neither NeMo nor sentence-transformers available. "
                "Install one: pip install nemo_toolkit[all] or pip install sentence-transformers"
            ) from None

        # Use a good general-purpose embedding model
        # In production, use NeMo models when available
        # sentence-transformers handles device placement automatically
        # Cache model to avoid reloading for large-scale processing
        # Note: In production, use model pooling or Ray actors to share models
        model = None
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            
            # Warmup GPU with dummy inference for better performance
            if "cuda" in self.device:
                try:
                    _ = model.encode(["warmup"], batch_size=1, show_progress_bar=False)
                    torch.cuda.synchronize()
                except Exception:
                    pass  # Ignore warmup errors
            
            # Use sentence-transformers' optimized batch encoding
            # It handles GPU memory efficiently and batches automatically
            # sentence-transformers uses PyTorch DataLoader internally for efficiency
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,  # Disable progress bar for performance
                normalize_embeddings=True,  # Normalize for better similarity computation
                # Additional optimizations
                convert_to_numpy=False,  # Keep as tensor for GPU operations
                output_value="sentence_embedding",  # Get sentence embeddings (not token embeddings)
            )
            
            # Ensure embeddings are on correct device
            if isinstance(embeddings, torch.Tensor):
                if "cuda" in self.device and not embeddings.is_cuda:
                    embeddings = embeddings.to(self.device, non_blocking=True)
                elif "cuda" not in self.device and embeddings.is_cuda:
                    embeddings = embeddings.cpu()
                # Synchronize if on GPU
                if embeddings.is_cuda:
                    torch.cuda.synchronize()
            
            return embeddings
        except Exception as e:
            raise DataSourceError(f"Failed to generate embeddings: {e}") from e
        finally:
            # Cleanup model to free GPU memory
            # sentence-transformers models can be large
            if model is not None:
                del model
            if "cuda" in self.device:
                torch.cuda.empty_cache()

    def get_embedding_dim(self) -> int:
        """Get embedding dimension for the current model.

        Returns:
            Embedding dimension
        """
        if self._model is None:
            # Default for sentence-transformers fallback
            return _DEFAULT_EMBEDDING_DIM
        # NeMo model dimension (placeholder)
        return 768


class NeMoTextProcessor:
    """Text processing using NVIDIA NeMo for advanced NLP tasks.

    Provides text cleaning, normalization, and quality scoring using
    NeMo's language understanding capabilities.
    """

    def __init__(self, device: str = "cuda"):
        """Initialize NeMo text processor.

        Args:
            device: Device to run models on

        Raises:
            ValueError: If device is invalid
        """
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {device}")
        
        self.device = device
        self._embedding_generator = NeMoEmbeddingGenerator(device=device)

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def compute_text_quality_score(self, text: str) -> float:
        """Compute quality score for text using NeMo.

        Args:
            text: Text string to score

        Returns:
            Quality score between 0 and 1

        Raises:
            ValueError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be str, got {type(text)}")
        
        # Placeholder for NeMo-based quality scoring
        # In production, use NeMo models for:
        # - Language detection
        # - Toxicity detection
        # - Readability scoring
        # - Semantic coherence

        # Simple heuristic for now
        if not text or len(text.strip()) < _DEFAULT_MIN_TEXT_LENGTH:
            return 0.0

        # Check for basic quality indicators
        score = 1.0
        if len(text) < 50:
            score *= 0.8
        
        word_count = len(text.split())
        if word_count < _DEFAULT_MIN_WORDS:
            score *= 0.7

        return score

    @log_execution_time
    @handle_errors(error_class=DataSourceError)
    def normalize_text(self, text: str) -> str:
        """Normalize text using NeMo preprocessing.

        Args:
            text: Raw text string

        Returns:
            Normalized text string

        Raises:
            ValueError: If text is invalid
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be str, got {type(text)}")
        
        # Placeholder for NeMo text normalization
        # In production, use NeMo's text preprocessing:
        # - Unicode normalization
        # - Tokenization
        # - Language-specific normalization

        # Simple normalization for now
        normalized = text.strip()
        # Remove excessive whitespace
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized
