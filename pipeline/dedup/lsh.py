"""Locality-sensitive hashing (LSH) based deduplication on GPU.

Uses PyTorch for GPU acceleration, NVIDIA NCCL for multi-GPU communication,
and Ray for distributed processing.

NVIDIA Libraries:
- NCCL: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- PyTorch: https://pytorch.org/
- Ray: https://docs.ray.io/
"""

import hashlib
import logging
from typing import List, Set, Union

import numpy as np  # https://numpy.org/
import ray
import torch  # https://pytorch.org/

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1, memory=8 * 1024 * 1024 * 1024)  # 8GB memory limit
class LSHGPUWorker:
    """Ray actor for GPU-accelerated LSH deduplication with NCCL support.

    Supports multi-GPU communication using NVIDIA NCCL for efficient
    signature aggregation and duplicate detection across GPU workers.
    """

    def __init__(
        self,
        num_bands: int = 20,
        band_width: int = 5,
        use_nccl: bool = True,
    ):
        """Initialize LSH worker.

        Args:
            num_bands: Number of bands for LSH
            band_width: Width of each band
            use_nccl: Whether to use NCCL for multi-GPU communication
        """
        self.num_bands = num_bands
        self.band_width = band_width

        # Properly initialize CUDA device with validation and pinning
        from pipeline.utils.gpu.memory import get_cuda_device

        if torch.cuda.is_available():
            try:
                # CRITICAL: Ray manages GPU assignment via num_gpus in @ray.remote decorator
                # CUDA_VISIBLE_DEVICES is set by Ray to limit visible devices per actor
                # When Ray sets CUDA_VISIBLE_DEVICES="0", device 0 in the actor's view is actually
                # the GPU assigned by Ray. We should use device 0 (the first visible device).
                # Do NOT parse CUDA_VISIBLE_DEVICES - Ray handles device assignment.
                device_id = 0  # Always use device 0 - Ray sets CUDA_VISIBLE_DEVICES to map it correctly
                device_id = get_cuda_device(device_id)  # Validate device is available
                
                self.device = torch.device(f"cuda:{device_id}")
                # CRITICAL: Device is already set by get_cuda_device() above
                # Verify device is correct
                if torch.cuda.current_device() != device_id:
                    torch.cuda.set_device(device_id)
                logger.info(f"LSH worker using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Failed to initialize CUDA device: {e}, falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")

        self.use_nccl = use_nccl and torch.cuda.is_available()

        # Initialize NCCL group if multi-GPU communication needed
        self._nccl_group = None
        if self.use_nccl:
            try:
                from pipeline.utils.nccl import initialize_nccl_for_ray_actors

                self._nccl_group = initialize_nccl_for_ray_actors()
                if self._nccl_group:
                    logger.info(f"LSH worker initialized with NCCL support on {self.device}")
                else:
                    logger.info(f"LSH worker initialized without NCCL on {self.device}")
            except ImportError:
                logger.warning("NCCL utilities not available, using single-GPU mode")
                self.use_nccl = False

        if not self._nccl_group:
            logger.info(f"LSH worker initialized on {self.device}")

    def __del__(self):
        """Cleanup GPU resources on worker destruction."""
        try:
            if self._nccl_group:
                self._nccl_group.cleanup()
            if self.device.type == "cuda":
                import torch  # https://pytorch.org/
                torch.cuda.empty_cache()
        except (RuntimeError, AttributeError):
            pass

    def compute_minhashes(self, texts: List[str]) -> torch.Tensor:
        """Compute MinHash signatures for texts on GPU.

        Args:
            texts: List of text strings

        Returns:
            Tensor of MinHash signatures [num_texts, num_hashes]
        """
        from pipeline.utils.gpu.memory import gpu_memory_cleanup

        if not texts:
            # Return empty tensor on correct device
            return torch.empty((0, self.num_bands * self.band_width), device=self.device, dtype=torch.int64)

        with gpu_memory_cleanup():
            # Generate hash signatures
            signatures = []
            for text in texts:
                minhash = self._compute_minhash(text)
                signatures.append(minhash)

            # Create tensor on GPU directly
            # For integer data, use blocking transfer (more reliable)
            # Non-blocking transfers are better for float data
            result = torch.tensor(signatures, device=self.device, dtype=torch.int64, non_blocking=False)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return result

    def _compute_minhash(self, text: str) -> List[int]:
        """Compute MinHash signature for a single text.

        Args:
            text: Text string

        Returns:
            List of hash values
        """
        # Simple MinHash implementation
        # In production, use proper shingling and multiple hash functions
        num_hashes = self.num_bands * self.band_width
        hashes = []

        # Create shingles (character n-grams)
        shingles = set()
        n = 5  # n-gram size
        for i in range(len(text) - n + 1):
            shingles.add(text[i : i + n])

        # Compute hash for each shingle
        for i in range(num_hashes):
            min_hash = float("inf")
            for shingle in shingles:
                # Use different hash seeds for each hash function
                hash_val = int(hashlib.md5(f"{shingle}_{i}".encode()).hexdigest(), 16)
                min_hash = min(min_hash, hash_val)
            hashes.append(min_hash)

        return hashes

    def find_duplicates(self, signatures: Union[torch.Tensor, ray.ObjectRef], threshold: float = 0.8) -> Set[int]:
        """Find duplicate pairs using LSH bands.

        Args:
            signatures: MinHash signatures tensor [num_texts, num_hashes] or ObjectRef
            threshold: Similarity threshold

        Returns:
            Set of indices to remove (duplicates)
        """
        from pipeline.utils.gpu.memory import gpu_memory_cleanup

        # Handle ObjectRef if passed
        if isinstance(signatures, ray.ObjectRef):
            # Get signatures with timeout
            from pipeline.utils.constants import _DEFAULT_RAY_TASK_TIMEOUT
            try:
                signatures = ray.get(signatures, timeout=_DEFAULT_RAY_TASK_TIMEOUT)
            except ray.exceptions.GetTimeoutError:
                raise TimeoutError("Timeout getting signatures") from None
        
        if len(signatures) == 0:
            return set()

        # CRITICAL: Ensure signatures are on correct device before operations
        # Use non-blocking transfer for performance, but synchronize after
        if not signatures.is_cuda and self.device.type == "cuda":
            signatures = signatures.to(self.device, non_blocking=True)
        elif signatures.device != self.device:
            signatures = signatures.to(self.device, non_blocking=True)

        # CRITICAL: Synchronize after non-blocking transfer to ensure completion
        # Then check for CUDA errors before proceeding
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            # Check for CUDA errors after synchronization
            from pipeline.utils.gpu.memory import check_cuda_errors
            check_cuda_errors()

        num_texts = len(signatures)
        duplicates = set()

        with gpu_memory_cleanup():
            # Group signatures into bands
            for band_idx in range(self.num_bands):
                start_idx = band_idx * self.band_width
                end_idx = start_idx + self.band_width

                # Extract band signatures
                band_sigs = signatures[:, start_idx:end_idx]

                # Find texts with identical band signatures (candidate pairs)
                # Use GPU-accelerated comparison
                for i in range(num_texts):
                    if i in duplicates:
                        continue

                    # Compare with all other texts in this band
                    band_i = band_sigs[i]
                    matches = torch.all(band_sigs == band_i, dim=1)
                    # Move to CPU once for all matches
                    match_indices = torch.where(matches)[0].cpu().numpy()

                    # CRITICAL: Batch CPU operations to avoid repeated GPU-CPU sync
                    # Move all needed signatures to CPU at once, not per-iteration
                    if len(match_indices) > 0:
                        # Move signatures[i] to CPU once
                        sig_i_cpu = signatures[i].cpu().numpy()
                        # Batch move all candidate signatures to CPU
                        candidate_indices = [j for j in match_indices if i < j and j not in duplicates]
                        if candidate_indices:
                            # Move all candidate signatures to CPU in one batch
                            candidate_sigs_cpu = signatures[candidate_indices].cpu().numpy()
                            
                            for idx, j in enumerate(candidate_indices):
                                sig_j_cpu = candidate_sigs_cpu[idx]
                                similarity = self._compute_similarity(sig_i_cpu, sig_j_cpu)

                                if similarity >= threshold:
                                    # Keep the first occurrence, mark others as duplicates
                                    duplicates.add(j)
                            
                            # Cleanup CPU arrays
                            del candidate_sigs_cpu
                        # Cleanup CPU arrays
                        del sig_i_cpu

        # Final cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return duplicates

    def _compute_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Compute Jaccard similarity from MinHash signatures.

        Args:
            sig1: First signature
            sig2: Second signature

        Returns:
            Similarity score between 0 and 1
        """
        matches = np.sum(sig1 == sig2)
        return matches / len(sig1)


class LSHDeduplicator:
    """LSH-based fuzzy deduplication using GPU acceleration with NCCL support.

    Uses NVIDIA NCCL for efficient multi-GPU communication when multiple
    GPU workers are used for distributed deduplication.
    """

    def __init__(
        self,
        num_bands: int = 20,
        band_width: int = 5,
        similarity_threshold: float = 0.8,
        use_nccl: bool = True,
    ):
        """Initialize LSH deduplicator.

        Args:
            num_bands: Number of LSH bands
            band_width: Width of each band
            similarity_threshold: Similarity threshold for duplicates
            use_nccl: Whether to use NCCL for multi-GPU communication
        """
        self.num_bands = num_bands
        self.band_width = band_width
        self.similarity_threshold = similarity_threshold
        self.use_nccl = use_nccl

    def deduplicate(self, texts: List[str], num_workers: int = 1) -> List[bool]:
        """Deduplicate texts using LSH.

        Args:
            texts: List of text strings
            num_workers: Number of GPU workers

        Returns:
            Boolean mask indicating which texts to keep
        """
        logger.info(f"Deduplicating {len(texts)} texts using LSH")

        # Create GPU workers with NCCL support
        # Limit workers to available GPUs to avoid resource contention
        import torch
        if torch.cuda.is_available():
            max_workers = min(num_workers, torch.cuda.device_count())
        else:
            max_workers = num_workers
        
        workers = [
            LSHGPUWorker.remote(self.num_bands, self.band_width, use_nccl=self.use_nccl)
            for _ in range(max_workers)
        ]

        # Split texts across workers (use max_workers, not num_workers)
        chunk_size = len(texts) // max_workers + 1 if max_workers > 0 else len(texts)
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

        # Process chunks in parallel
        futures = []
        chunk_map = {}  # Map future to chunk for later reference
        for worker, chunk in zip(workers[: len(chunks)], chunks):
            future = worker.compute_minhashes.remote(chunk)
            futures.append(future)
            chunk_map[future] = chunk

        # Collect signatures efficiently - batch ray.get() with timeout
        all_signatures = []
        all_texts = []
        # Get all results at once (parallel collection) with timeout
        from pipeline.utils.constants import _DEFAULT_RAY_TASK_TIMEOUT
        try:
            signatures_list = ray.get(futures, timeout=_DEFAULT_RAY_TASK_TIMEOUT)
        except ray.exceptions.GetTimeoutError:
            logger.error("Timeout computing minhashes, cleaning up workers")
            # Use Ray's built-in actor cleanup
            for actor in workers:
                try:
                    ray.kill(actor, no_restart=True)
                except (ValueError, ray.exceptions.RayActorError):
                    pass
            raise TimeoutError("Timeout computing minhashes") from None
        for signatures, future in zip(signatures_list, futures):
            all_signatures.append(signatures)
            all_texts.extend(chunk_map[future])

        # Concatenate all signatures
        # Note: Ray handles inter-worker communication efficiently.
        # NCCL is more beneficial for in-process multi-GPU operations.
        # For Ray actors, torch.cat after ray.get() is optimal.
        from pipeline.utils.gpu.memory import gpu_memory_cleanup

        with gpu_memory_cleanup():
            if all_signatures:
                # CRITICAL: Concatenate on GPU if first signature is on GPU
                # Determine target device from first signature
                device = all_signatures[0].device if all_signatures else torch.device("cpu")
                # Concatenate all signatures (may involve device transfers)
                combined_signatures = torch.cat(all_signatures, dim=0)
                # Move to target device if needed (non-blocking for performance)
                if combined_signatures.device != device:
                    combined_signatures = combined_signatures.to(device, non_blocking=True)
                # CRITICAL: Synchronize after concatenation and transfer
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    # Check for CUDA errors after synchronization
                    from pipeline.utils.gpu.memory import check_cuda_errors
                    check_cuda_errors()
            else:
                combined_signatures = torch.empty((0, self.num_bands * self.band_width))

        # Find duplicates using first worker
        # For large tensors, use ray.put() to avoid multiple serializations
        try:
            if len(combined_signatures) > 0:
                # Put large tensor in object store once, reuse reference
                signature_ref = ray.put(combined_signatures)
                # Get duplicates with timeout
                from pipeline.utils.constants import _DEFAULT_RAY_TASK_TIMEOUT
                try:
                    duplicates = ray.get(
                        workers[0].find_duplicates.remote(signature_ref, self.similarity_threshold),
                        timeout=_DEFAULT_RAY_TASK_TIMEOUT,
                    )
                except ray.exceptions.GetTimeoutError:
                    logger.error("Timeout finding duplicates, cleaning up workers")
                    from pipeline.utils.actor_utils import cleanup_actor_pool
                    cleanup_actor_pool(workers)
                    raise TimeoutError("Timeout finding duplicates") from None
                finally:
                    # Cleanup reference
                    del signature_ref
            else:
                duplicates = set()
        finally:
            # Cleanup GPU resources
            from pipeline.utils.gpu.memory import clear_gpu_cache

            clear_gpu_cache()
            # Cleanup workers properly - use Ray's built-in cleanup
            for actor in workers:
                try:
                    ray.kill(actor, no_restart=True)
                except (ValueError, ray.exceptions.RayActorError):
                    pass

        # Create keep mask
        keep_mask = [i not in duplicates for i in range(len(texts))]

        logger.info(
            f"Found {len(duplicates)} duplicates ({len(duplicates) / len(texts) * 100:.1f}%)"
        )

        return keep_mask
