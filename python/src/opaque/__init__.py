"""
Project Opaque: Privacy-preserving vector search engine.

Uses a two-stage funnel approach:
1. Coarse Stage: LSH for fast approximate filtering
2. Fine Stage: PHE (Paillier) for secure exact scoring

The server NEVER sees the raw query vector.
The client NEVER downloads the full database.
"""

__version__ = "0.1.0"
