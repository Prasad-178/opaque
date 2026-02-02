"""
Real embeddings support using sentence-transformers.

Provides utilities for:
- Loading pre-trained embedding models
- Embedding text documents
- Loading sample datasets for testing
"""
import numpy as np
from typing import List, Optional, Union, Tuple
from pathlib import Path


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding models.

    Supports various pre-trained models like:
    - all-MiniLM-L6-v2 (384 dim, fast)
    - all-mpnet-base-v2 (768 dim, better quality)
    - paraphrase-MiniLM-L3-v2 (384 dim, fastest)
    """

    # Model presets with their dimensions
    PRESETS = {
        "fast": "paraphrase-MiniLM-L3-v2",       # 384 dim, fastest
        "balanced": "all-MiniLM-L6-v2",           # 384 dim, good balance
        "quality": "all-mpnet-base-v2",           # 768 dim, best quality
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name or preset ("fast", "balanced", "quality")
            device: Device to use ("cpu", "cuda", "mps", or None for auto)
            normalize: Whether to L2-normalize embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        # Check for preset name
        if model_name in self.PRESETS:
            model_name = self.PRESETS[model_name]

        self.model_name = model_name
        self.normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed text(s) into vectors.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Embeddings as numpy array of shape (n, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text."""
        return self.embed([query])[0]


def load_sample_dataset(
    name: str = "quora",
    max_samples: int = 1000,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Load a sample dataset for testing.

    Args:
        name: Dataset name ("quora", "stsb", "msmarco")
        max_samples: Maximum number of samples
        seed: Random seed for sampling

    Returns:
        Tuple of (texts, ids)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets is required. "
            "Install with: pip install datasets"
        )

    rng = np.random.default_rng(seed)

    if name == "quora":
        # Quora Question Pairs - good for semantic search
        dataset = load_dataset("quora", split="train", trust_remote_code=True)
        # Extract unique questions
        questions = set()
        for item in dataset:
            q = item["questions"]["text"]
            questions.add(q[0])
            questions.add(q[1])
        texts = list(questions)

    elif name == "stsb":
        # STS Benchmark - semantic similarity
        dataset = load_dataset("sentence-transformers/stsb", split="train")
        texts = []
        for item in dataset:
            texts.append(item["sentence1"])
            texts.append(item["sentence2"])
        texts = list(set(texts))

    elif name == "msmarco":
        # MS MARCO - passage retrieval
        dataset = load_dataset("ms_marco", "v1.1", split="train", trust_remote_code=True)
        texts = []
        for item in dataset:
            for passage in item["passages"]["passage_text"]:
                texts.append(passage)
                if len(texts) >= max_samples * 2:
                    break
            if len(texts) >= max_samples * 2:
                break
        texts = list(set(texts))

    elif name == "simple":
        # Simple built-in dataset for quick testing (no download needed)
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast auburn fox leaps above a sleepy canine.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing helps computers understand text.",
            "Python is a popular programming language for data science.",
            "Vector databases store high-dimensional embeddings efficiently.",
            "Homomorphic encryption allows computation on encrypted data.",
            "Privacy-preserving search protects user query information.",
            "Locality sensitive hashing enables approximate nearest neighbor search.",
            "The weather today is sunny and warm.",
            "It's raining cats and dogs outside.",
            "The stock market crashed yesterday.",
            "Bitcoin prices are highly volatile.",
            "Quantum computers use qubits instead of classical bits.",
            "The Eiffel Tower is located in Paris, France.",
            "Tokyo is the capital city of Japan.",
            "The Great Wall of China is visible from space.",
            "Coffee contains caffeine which is a stimulant.",
            "Green tea has many health benefits.",
        ]
        # Expand by creating variations
        expanded = texts.copy()
        for t in texts:
            expanded.append(f"Question: {t}")
            expanded.append(f"Answer: {t}")
        texts = expanded

    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'quora', 'stsb', 'msmarco', or 'simple'")

    # Sample if needed
    if len(texts) > max_samples:
        indices = rng.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]

    # Generate IDs
    ids = [f"doc_{i:06d}" for i in range(len(texts))]

    return texts, ids


def create_embedded_database(
    texts: List[str],
    ids: Optional[List[str]] = None,
    model: Optional[EmbeddingModel] = None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = True,
    positive_shift: bool = True,
) -> Tuple[np.ndarray, List[str], int]:
    """
    Create a database of embedded texts.

    Args:
        texts: List of texts to embed
        ids: Optional list of IDs (auto-generated if None)
        model: Pre-initialized EmbeddingModel (creates new if None)
        model_name: Model name if creating new model
        batch_size: Batch size for encoding
        show_progress: Show progress bar
        positive_shift: Shift embeddings to positive range (required for PHE)

    Returns:
        Tuple of (embeddings, ids, dimension)
    """
    if model is None:
        model = EmbeddingModel(model_name)

    if ids is None:
        ids = [f"doc_{i:06d}" for i in range(len(texts))]

    print(f"Embedding {len(texts)} texts with {model.model_name}...")
    embeddings = model.embed(texts, batch_size=batch_size, show_progress=show_progress)

    if positive_shift:
        # Shift to positive range for PHE compatibility
        # LightPHE requires all values > 0 for element-wise multiplication
        embeddings = embeddings - embeddings.min() + 0.01

    return embeddings, ids, model.dimension
