"""Semantic deduplication using embedding-based clustering on GPU.

Uses PyTorch for GPU acceleration, NVIDIA cuML for GPU-accelerated clustering,
and Ray for distributed processing.

NVIDIA Libraries:
- cuML: https://docs.rapids.ai/api/cuml/stable/
- PyTorch: https://pytorch.org/
- Ray: https://docs.ray.io/
"""

import logging
from typing import List, Set, Union


import ray
import torch  # https://pytorch.org/

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1, memory=8 * 1024 * 1024 * 1024)  # 8GB memory limit
class SemanticGPUWorker:
    """Ray actor for GPU-accelerated semantic deduplication using NVIDIA cuML.

    Uses cuML KMeans for GPU-accelerated clustering, which provides significant
    speedups over CPU sklearn for large-scale deduplication workloads.
    """

    def __init__(self, embedding_dim: int = 768):
        """Initialize semantic deduplication worker.

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim

        # Properly initialize CUDA device with validation and pinning
        from pipeline.utils.gpu.memory import get_cuda_device

        if torch.cuda.is_available():
            try:
                # Ray manages GPU assignment via num_gpus in @ray.remote decorator
                # When Ray sets CUDA_VISIBLE_DEVICES="0", device 0 in the actor's view is actually
                # the GPU assigned by Ray. We should use device 0 (the first visible device).
                # Do NOT parse CUDA_VISIBLE_DEVICES - Ray handles device assignment.
                device_id = 0  # Always use device 0 - Ray sets CUDA_VISIBLE_DEVICES to map it correctly
                device_id = get_cuda_device(device_id)  # Validate device is available
                
                self.device = torch.device(f"cuda:{device_id}")
                torch.cuda.set_device(device_id)
                # Device is already set by get_cuda_device() above
                if torch.cuda.current_device() != device_id:
                    torch.cuda.set_device(device_id)
                logger.info(f"Semantic worker using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Failed to initialize CUDA device: {e}, falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")

        # Try to import cuML for GPU-accelerated clustering
        try:
            from cuml.cluster import KMeans as cuMLKMeans  # type: ignore[attr-defined]

            # Check cuML compatibility
            from pipeline.utils.gpu.rapids import check_cuml_compatibility

            cuml_info = check_cuml_compatibility()
            if not cuml_info.get("available"):
                raise ImportError("cuML not available or incompatible version")
            if cuml_info.get("issues"):
                for issue in cuml_info["issues"]:
                    logger.warning(f"cuML compatibility issue: {issue}")

            self._cuml_available = True
            self._KMeans = cuMLKMeans
            logger.info(
                f"Semantic worker initialized on {self.device} with cuML support"
            )

            # Ensure RMM is initialized for optimal cuML performance
            # RMM pool should be initialized once at application startup
            # This is a safety check - RMM should already be initialized in core.py
            from pipeline.utils.gpu.rapids import initialize_rmm_pool

            rmm_initialized = initialize_rmm_pool()
            if not rmm_initialized:
                logger.warning("RMM pool not initialized, cuML performance may be suboptimal")
            else:
                logger.debug("RMM pool verified/initialized for semantic deduplication")
        except ImportError:
            # Fallback to sklearn if cuML not available
            try:
                from sklearn.cluster import KMeans  # https://scikit-learn.org/

                self._cuml_available = False
                self._KMeans = KMeans
                logger.warning(
                    "cuML not available, falling back to CPU sklearn. "
                    "Install cuML for GPU acceleration: pip install cuml-cu12"
                )
            except ImportError as err:
                raise ImportError(
                    "Neither cuML nor sklearn available. Install at least one: "
                    "pip install cuml-cu12 or pip install scikit-learn"
                ) from err

        logger.info(f"Semantic worker initialized on {self.device}")

    def __del__(self):
        """Cleanup GPU resources on worker destruction."""
        try:
            if self.device.type == "cuda":
                import torch  # https://pytorch.org/
                torch.cuda.empty_cache()
                # Clear cuPy memory pool if used
                try:
                    import cupy as cp  # https://docs.cupy.dev/
                    cp.get_default_memory_pool().free_all_blocks()
                except ImportError:
                    pass
        except (RuntimeError, AttributeError):
            pass

    def generate_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for texts on GPU.

        Args:
            texts: List of text strings

        Returns:
            Tensor of embeddings [num_texts, embedding_dim]
        """
        from pipeline.utils.gpu.memory import check_gpu_memory, gpu_memory_cleanup

        if not texts:
            # Return empty tensor on correct device with proper dtype
            return torch.empty((0, self.embedding_dim), device=self.device, dtype=torch.float32)

        # Estimate required memory (rough: num_texts * embedding_dim * 4 bytes per float32)
        estimated_memory = len(texts) * self.embedding_dim * 4
        if self.device.type == "cuda":
            has_memory, mem_info = check_gpu_memory(estimated_memory)
            if not has_memory:
                raise RuntimeError(
                    f"Insufficient GPU memory for {len(texts)} embeddings. "
                    f"Required: {estimated_memory}, Available: {mem_info['free']}"
                )

        # Use GPU memory cleanup context to ensure cleanup
        with gpu_memory_cleanup():
            try:
                # Simplified embedding generation
                # In production, use NVIDIA NeMo or sentence-transformers
                embeddings = []
                for text in texts:
                    # Simple hash-based embedding (replace with actual model)
                    embedding = self._simple_embedding(text)
                    embeddings.append(embedding)

                result = torch.stack(embeddings).to(self.device, non_blocking=True)
                # Synchronize after transfer to ensure completion
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                    # Check for CUDA errors after synchronization
                    from pipeline.utils.gpu.memory import check_cuda_errors

                    check_cuda_errors()
                return result
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    logger.error(f"GPU memory error during embedding generation: {e}")
                    # Clear cache and retry once
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(f"GPU OOM during embedding generation: {e}") from e
                raise

    def _simple_embedding(self, text: str) -> torch.Tensor:
        """Generate simple embedding (placeholder for real model).

        Args:
            text: Text string

        Returns:
            Embedding vector
        """
        # This is a placeholder - in production use NVIDIA NeMo or sentence-transformers
        import hashlib

        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        # Use cuPy for GPU random number generation if available
        try:
            import cupy as cp  # https://docs.cupy.dev/

            # Set cuPy device context - operations must be within device context
            device_id = self.device.index if self.device.index is not None else 0
            cp.cuda.Device(device_id).use()
            with cp.cuda.Device(device_id):
                cp.random.seed(hash_val % (2**32))
                # Use cuPy's optimized random number generation
                embedding_array = cp.random.randn(self.embedding_dim, dtype=cp.float32)
                # Convert to torch tensor (zero-copy if possible)
                # torch.as_tensor can do zero-copy conversion from cuPy arrays
                result = torch.as_tensor(embedding_array, device=self.device)
                # Free cuPy array immediately after conversion
                del embedding_array
                # Don't free all blocks here - that's too aggressive and hurts performance
                # Only free unused blocks if memory pressure is high
            return result
        except ImportError:
            # Fallback to NumPy if cuPy not available
            import numpy as np  # https://numpy.org/

            np.random.seed(hash_val % (2**32))
            # Create tensor directly on target device to avoid unnecessary transfers
            return torch.tensor(
                np.random.randn(self.embedding_dim), dtype=torch.float32, device=self.device
            )

    def cluster_embeddings(self, embeddings: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """Perform K-means clustering on embeddings using NVIDIA cuML.

        Args:
            embeddings: Embedding tensor [num_texts, embedding_dim] on GPU
            num_clusters: Number of clusters

        Returns:
            Cluster assignments tensor on GPU
        """
        from pipeline.utils.gpu.memory import gpu_memory_cleanup

        # Ensure embeddings are on correct device
        if not embeddings.is_cuda and self.device.type == "cuda":
            embeddings = embeddings.to(self.device, non_blocking=True)
            torch.cuda.synchronize()  # Ensure transfer completes before operations

        with gpu_memory_cleanup():
            if self._cuml_available:
                # Use cuML KMeans for GPU-accelerated clustering
                # cuML expects numpy/cupy arrays, but we can work with torch tensors
                try:
                    import cupy as cp  # https://docs.cupy.dev/

                    # Convert torch tensor to cupy array (zero-copy if possible)
                # Set cuPy device context before operations
                    device_id = self.device.index if self.device.index is not None else 0
                    # Ensure cuPy is using the correct device
                    cp.cuda.Device(device_id).use()
                    with cp.cuda.Device(device_id):
                        if embeddings.is_cuda:
                            # Zero-copy conversion from torch to cuPy (same GPU)
                            # torch and cuPy share CUDA memory when on same device
                            embeddings_cp = cp.asarray(embeddings)
                        else:
                            # Need to move to GPU first, then convert
                            embeddings_gpu = embeddings.cuda(device=device_id)
                            embeddings_cp = cp.asarray(embeddings_gpu)
                            del embeddings_gpu  # Free intermediate tensor

                    # cuML KMeans on GPU with optimal parameters
                    # cuML KMeans supports additional parameters for better performance
                    # See: https://docs.rapids.ai/api/cuml/stable/api.html#kmeans
                    kmeans = self._KMeans(
                        n_clusters=num_clusters,
                        random_state=42,
                        max_iter=300,
                        n_init=10,
                        # cuML-specific optimizations
                        init="k-means||",  # Use scalable k-means++ initialization (faster)
                        tol=1e-4,  # Convergence tolerance (balance speed vs accuracy)
                        verbose=False,  # Disable verbose output for performance
                        # Additional cuML optimizations
                        oversampling_factor=2.0,  # For k-means|| initialization
                        output_type="cupy",  # Return cuPy array (faster than NumPy)
                    )
                    # cuML fit_predict is GPU-accelerated and returns cuPy array
                    clusters_cp = kmeans.fit_predict(embeddings_cp)

                    # Cleanup cuML model immediately after use
                    del kmeans

                    # Convert back to torch tensor on GPU (zero-copy if possible)
                    # cuPy array -> PyTorch tensor conversion is efficient when on same device
                    clusters = torch.as_tensor(clusters_cp, device=self.device, dtype=torch.long)

                    # Cleanup cuPy arrays immediately
                    # cuPy arrays hold GPU memory, must be freed explicitly
                    del embeddings_cp, clusters_cp
                    # Don't free all blocks - too aggressive, hurts performance
                    # cuPy memory pool manages memory efficiently on its own

                    logger.info(f"Clustered {len(embeddings)} embeddings using cuML on GPU")
                    # Check for CUDA errors
                    if self.device.type == "cuda":
                        from pipeline.utils.gpu.memory import check_cuda_errors

                        check_cuda_errors()
                    return clusters
                except ImportError:
                    # Fallback if cuPy not available
                    logger.warning("cuPy not available, using CPU conversion for cuML")
                    embeddings_np = embeddings.cpu().numpy()
                    kmeans = self._KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                    clusters_np = kmeans.fit_predict(embeddings_np)
                    result = torch.tensor(clusters_np, device=self.device, dtype=torch.long)
                    del embeddings_np, clusters_np
                    return result
                except RuntimeError as e:
                    if "CUDA" in str(e) or "out of memory" in str(e).lower():
                        logger.error(f"GPU memory error during clustering: {e}")
                        # Clear cache and fallback to CPU
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                            try:
                                import cupy as cp
                                cp.get_default_memory_pool().free_all_blocks()
                            except ImportError:
                                pass
                        # Fallback to CPU
                        logger.warning("Falling back to CPU clustering due to GPU error")
                        embeddings_np = embeddings.cpu().numpy()
                        kmeans = self._KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                        clusters_np = kmeans.fit_predict(embeddings_np)
                        return torch.tensor(clusters_np, device=torch.device("cpu"), dtype=torch.long)
                    raise
            else:
                # Fallback to sklearn on CPU
                embeddings_np = embeddings.cpu().numpy()
                kmeans = self._KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                clusters_np = kmeans.fit_predict(embeddings_np)
                result = torch.tensor(clusters_np, device=self.device, dtype=torch.long)
                del embeddings_np, clusters_np
                return result

    def find_semantic_duplicates(
        self, embeddings: Union[torch.Tensor, ray.ObjectRef], clusters: Union[torch.Tensor, ray.ObjectRef], similarity_threshold: float = 0.95
    ) -> Set[int]:
        """Find semantic duplicates within clusters using GPU-accelerated operations.

        Uses PyTorch for efficient GPU matrix operations and cuPy for advanced
        GPU array operations when available.

        Args:
            embeddings: Embedding tensor [num_texts, embedding_dim] on GPU or ObjectRef
            clusters: Cluster assignments tensor on GPU or ObjectRef
            similarity_threshold: Cosine similarity threshold

        Returns:
            Set of indices to remove (duplicates)
        """
        from pipeline.utils.gpu.memory import gpu_memory_cleanup

        # Handle ObjectRef if passed with timeout
        from pipeline.utils.constants import _DEFAULT_RAY_TASK_TIMEOUT
        if isinstance(embeddings, ray.ObjectRef):
            try:
                embeddings = ray.get(embeddings, timeout=_DEFAULT_RAY_TASK_TIMEOUT)
            except ray.exceptions.GetTimeoutError:
                raise RuntimeError("Timeout getting embeddings") from None
        if isinstance(clusters, ray.ObjectRef):
            try:
                clusters = ray.get(clusters, timeout=_DEFAULT_RAY_TASK_TIMEOUT)
            except ray.exceptions.GetTimeoutError:
                raise RuntimeError("Timeout getting clusters") from None

        # Ensure tensors are on correct device
        if not embeddings.is_cuda and self.device.type == "cuda":
            embeddings = embeddings.to(self.device, non_blocking=True)
        elif embeddings.device != self.device:
            embeddings = embeddings.to(self.device, non_blocking=True)

        if not clusters.is_cuda and self.device.type == "cuda":
            clusters = clusters.to(self.device, non_blocking=True)
        elif clusters.device != self.device:
            clusters = clusters.to(self.device, non_blocking=True)

        # Synchronize after transfers to ensure completion
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            # Check for CUDA errors after synchronization
            from pipeline.utils.gpu.memory import check_cuda_errors
            check_cuda_errors()

        duplicates = set()

        with gpu_memory_cleanup():
            # Normalize embeddings for cosine similarity using PyTorch
            embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Get unique clusters using PyTorch (more efficient than NumPy for GPU tensors)
            unique_clusters = torch.unique(clusters).cpu().tolist()

            for cluster_id in unique_clusters:
                # Use PyTorch boolean indexing for GPU efficiency
                cluster_mask = clusters == cluster_id
                cluster_indices = torch.where(cluster_mask)[0].cpu().tolist()

                if len(cluster_indices) < 2:
                    continue

                # Compute pairwise similarities within cluster using GPU matrix multiplication
                cluster_embeddings = embeddings_norm[cluster_mask]
                similarities = torch.mm(cluster_embeddings, cluster_embeddings.t())

                # Find duplicates (keep first occurrence)
                # Move to CPU for iteration to avoid GPU-CPU sync in loop
                # Free GPU memory immediately after CPU transfer
                similarities_cpu = similarities.cpu()
                del similarities, cluster_embeddings  # Free GPU memory immediately

                # Batch process similarities on CPU to avoid repeated .item() calls
                # Convert to numpy array once for efficient CPU-side comparison
                similarities_np = similarities_cpu.numpy()
                for i in range(len(cluster_indices)):
                    if cluster_indices[i] in duplicates:
                        continue

                    for j in range(i + 1, len(cluster_indices)):
                        # Use numpy array indexing - much faster than repeated .item() calls
                        if similarities_np[i, j] >= similarity_threshold:
                            duplicates.add(cluster_indices[j])

                # Cleanup CPU tensor
                del similarities_cpu

        # Final cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return duplicates


class SemanticDeduplicator:
    """Semantic deduplication using embedding-based clustering with NVIDIA cuML.

    Uses GPU-accelerated K-means clustering via cuML for efficient large-scale
    deduplication of multimodal datasets.
    """

    def __init__(self, similarity_threshold: float = 0.95, num_clusters: int = 1000):
        """Initialize semantic deduplicator.

        Args:
            similarity_threshold: Cosine similarity threshold for duplicates
            num_clusters: Number of clusters for K-means
        """
        self.similarity_threshold = similarity_threshold
        self.num_clusters = num_clusters

    def deduplicate(self, texts: List[str], num_workers: int = 1) -> List[bool]:
        """Deduplicate texts using semantic clustering with NVIDIA cuML.

        Args:
            texts: List of text strings
            num_workers: Number of GPU workers

        Returns:
            Boolean mask indicating which texts to keep
        """
        logger.info(
            f"Deduplicating {len(texts)} texts using semantic clustering "
            f"with NVIDIA cuML (num_clusters={self.num_clusters})"
        )

        # Create GPU worker
        worker = SemanticGPUWorker.remote()

        try:
            # Generate embeddings with timeout and retry
            logger.info("Generating embeddings")
            from pipeline.utils.constants import _DEFAULT_RAY_TASK_TIMEOUT
            
            embedding_future = worker.generate_embeddings.remote(texts)
            try:
                embeddings = ray.get(embedding_future, timeout=_DEFAULT_RAY_TASK_TIMEOUT)
            except ray.exceptions.GetTimeoutError:
                logger.error("Timeout generating embeddings, killing worker")
                ray.kill(worker, no_restart=True)
                raise RuntimeError("Timeout generating embeddings") from None

            # Cluster embeddings using cuML on GPU with timeout
            logger.info(f"Clustering into {self.num_clusters} clusters using cuML")
            cluster_future = worker.cluster_embeddings.remote(embeddings, self.num_clusters)
            try:
                clusters = ray.get(cluster_future, timeout=_DEFAULT_RAY_TASK_TIMEOUT)
            except ray.exceptions.GetTimeoutError:
                logger.error("Timeout clustering embeddings, killing worker")
                ray.kill(worker, no_restart=True)
                raise RuntimeError("Timeout clustering embeddings") from None

            # Find duplicates within clusters
            # Note: This depends on embeddings and clusters, so must wait for them first
            # For large tensors, use ray.put() to avoid multiple serializations
            logger.info("Finding semantic duplicates")
            embeddings_ref = ray.put(embeddings)
            clusters_ref = ray.put(clusters)
            duplicates_future = worker.find_semantic_duplicates.remote(
                embeddings_ref, clusters_ref, self.similarity_threshold
            )
            # Get duplicates with timeout
            try:
                duplicates = ray.get(duplicates_future, timeout=_DEFAULT_RAY_TASK_TIMEOUT)
            except ray.exceptions.GetTimeoutError:
                logger.error("Timeout finding duplicates, killing worker")
                ray.kill(worker, no_restart=True)
                raise RuntimeError("Timeout finding duplicates") from None
            # Cleanup references
            del embeddings_ref, clusters_ref

            # Create keep mask
            keep_mask = [i not in duplicates for i in range(len(texts))]

            logger.info(
                f"Found {len(duplicates)} duplicates ({len(duplicates) / len(texts) * 100:.1f}%)"
            )

            return keep_mask
        finally:
            # Cleanup GPU worker and resources
            from pipeline.utils.gpu.memory import clear_gpu_cache

            try:
                ray.kill(worker, no_restart=True)
            except (ValueError, ray.exceptions.RayActorError):
                pass
            clear_gpu_cache()
