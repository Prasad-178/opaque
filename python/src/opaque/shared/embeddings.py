"""
Real embeddings support using HuggingFace transformers with ONNX Runtime.

Uses ONNX Runtime backend instead of PyTorch for:
- Better stability on macOS (no multiprocessing segfaults)
- Faster inference
- Smaller dependency footprint

Provides utilities for:
- Loading pre-trained embedding models
- Embedding text documents
- Loading sample datasets for testing
"""
import numpy as np
from typing import List, Optional, Union, Tuple
from pathlib import Path
import os


class EmbeddingModel:
    """
    Wrapper for HuggingFace embedding models using ONNX Runtime.

    Supports various pre-trained models like:
    - all-MiniLM-L6-v2 (384 dim, fast)
    - all-mpnet-base-v2 (768 dim, better quality)
    - paraphrase-MiniLM-L3-v2 (384 dim, fastest)
    """

    # Model presets with their dimensions
    PRESETS = {
        "fast": "sentence-transformers/paraphrase-MiniLM-L3-v2",       # 384 dim, fastest
        "balanced": "sentence-transformers/all-MiniLM-L6-v2",           # 384 dim, good balance
        "quality": "sentence-transformers/all-mpnet-base-v2",           # 768 dim, best quality
    }
    
    # Known model dimensions
    MODEL_DIMS = {
        "sentence-transformers/paraphrase-MiniLM-L3-v2": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L3-v2": 384,
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,  # Kept for API compatibility, ONNX handles this
        normalize: bool = True,
        use_onnx: bool = True,
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name or preset ("fast", "balanced", "quality")
            device: Ignored (ONNX Runtime handles device selection automatically)
            normalize: Whether to L2-normalize embeddings
            use_onnx: Use ONNX Runtime backend (default True, more stable)
        """
        # Check for preset name
        if model_name in self.PRESETS:
            model_name = self.PRESETS[model_name]
        
        # Add sentence-transformers prefix if not present
        if "/" not in model_name:
            model_name = f"sentence-transformers/{model_name}"

        self.model_name = model_name
        self.normalize = normalize
        self.use_onnx = use_onnx
        self._session = None
        self._tokenizer = None
        self._dimension = None
        
        self._load_model()

    def _load_model(self):
        """Load the model using ONNX Runtime or fallback to transformers."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. "
                "Install with: pip install transformers"
            )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.use_onnx:
            self._load_onnx_model()
        else:
            self._load_transformers_model()

    def _load_onnx_model(self):
        """Load model using ONNX Runtime."""
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "onnxruntime and huggingface-hub are required. "
                "Install with: pip install onnxruntime huggingface-hub"
            )
        
        # Try to download ONNX model from HuggingFace Hub
        try:
            onnx_path = hf_hub_download(
                repo_id=self.model_name,
                filename="onnx/model.onnx",
            )
        except Exception:
            # Fallback: try model.onnx in root
            try:
                onnx_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename="model.onnx",
                )
            except Exception:
                # No ONNX model available, convert on the fly
                print(f"  No pre-built ONNX model for {self.model_name}, using transformers backend...")
                self.use_onnx = False
                self._load_transformers_model()
                return
        
        # Create ONNX Runtime session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        # Use available execution providers
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]
        
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )
        
        # Get dimension from model output
        if self.model_name in self.MODEL_DIMS:
            self._dimension = self.MODEL_DIMS[self.model_name]
        else:
            # Infer from a test run
            test_inputs = self._tokenizer(["test"], return_tensors="np", padding=True, truncation=True)
            test_output = self._run_onnx_inference(test_inputs)
            self._dimension = test_output.shape[-1]

    def _load_transformers_model(self):
        """Load model using pure transformers (no torch dependency for inference)."""
        try:
            from transformers import AutoModel, AutoConfig
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for non-ONNX mode. "
                "Install with: pip install transformers torch"
            )
        
        self._config = AutoConfig.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._dimension = self._config.hidden_size

    def _run_onnx_inference(self, inputs: dict) -> np.ndarray:
        """Run inference using ONNX Runtime."""
        # Prepare inputs for ONNX
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        
        # Add token_type_ids if required by the model
        input_names = [inp.name for inp in self._session.get_inputs()]
        if "token_type_ids" in input_names:
            if "token_type_ids" in inputs:
                onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            else:
                onnx_inputs["token_type_ids"] = np.zeros_like(onnx_inputs["input_ids"])
        
        # Run inference
        outputs = self._session.run(None, onnx_inputs)
        
        # Get the last hidden state (first output)
        last_hidden_state = outputs[0]
        
        # Mean pooling with attention mask
        attention_mask = inputs["attention_mask"]
        mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        
        return embeddings

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
        
        all_embeddings = []
        
        # Process in batches
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(texts), batch_size), desc="Batches")
            except ImportError:
                iterator = range(0, len(texts), batch_size)
        else:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self._tokenizer(
                batch_texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            if self.use_onnx and self._session is not None:
                embeddings = self._run_onnx_inference(inputs)
            else:
                embeddings = self._run_transformers_inference(inputs)
            
            all_embeddings.append(embeddings)
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        # Normalize if requested
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-9, a_max=None)
            embeddings = embeddings / norms
        
        return embeddings.astype(np.float32)

    def _run_transformers_inference(self, inputs: dict) -> np.ndarray:
        """Run inference using transformers (requires torch)."""
        import torch
        
        # Convert to torch tensors
        torch_inputs = {
            k: torch.tensor(v) for k, v in inputs.items()
        }
        
        with torch.no_grad():
            outputs = self._model(**torch_inputs)
            last_hidden_state = outputs.last_hidden_state
            
            # Mean pooling
            attention_mask = torch_inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        
        return embeddings.numpy()

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
        name: Dataset name ("quora", "stsb", "msmarco", "simple")
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
