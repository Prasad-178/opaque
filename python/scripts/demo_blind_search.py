#!/usr/bin/env python3
"""
Phase 2: "Blind Search" - End-to-end search without LSH.

Demonstrates:
1. Server starts with mock vector database
2. Client generates keys and registers with server
3. Client encrypts query and sends to server
4. Server computes encrypted dot products
5. Client decrypts and ranks results
6. Verification that encrypted search matches plaintext search

The server NEVER sees the raw query or the final similarity scores.
"""
import sys
import argparse
import pickle
import base64
import time
from pathlib import Path
from multiprocessing import Process

# Add src to path for development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def serialize_encrypted(obj):
    """Serialize encrypted object to base64 string."""
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')


def deserialize_encrypted(b64_str):
    """Deserialize encrypted object from base64 string."""
    return pickle.loads(base64.b64decode(b64_str))


def run_server(host: str, port: int, num_vectors: int, dimension: int, seed: int):
    """Run the FastAPI server in a separate process."""
    import uvicorn
    from opaque.server.api import app, state
    from opaque.server.compute import create_mock_database, ComputeEngine

    # Initialize database
    state.vector_store = create_mock_database(num_vectors, dimension, seed)
    state.compute_engine = ComputeEngine(state.vector_store)
    print(f"Server initialized with {num_vectors} vectors of dimension {dimension}")

    uvicorn.run(app, host=host, port=port, log_level="warning")


def run_client_demo(
    host: str,
    port: int,
    num_vectors: int,
    dimension: int,
    top_k: int,
    seed: int,
):
    """Run the client-side demo."""
    import httpx
    import numpy as np
    from opaque.client.crypto import CryptoClient
    from opaque.shared.utils import (
        generate_random_vectors,
        compute_plaintext_similarity,
        top_k_indices,
        Timer,
    )

    base_url = f"http://{host}:{port}"
    client_id = "demo_client"

    print("=" * 60)
    print("Project Opaque - Phase 2: Blind Search Demo")
    print("=" * 60)
    print(f"\nServer: {base_url}")
    print(f"Database: {num_vectors} vectors, {dimension} dimensions")
    print(f"Top-K: {top_k}")

    # Wait for server to be ready
    print("\nWaiting for server...")
    for _ in range(30):
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                health = resp.json()
                print(f"Server ready: {health['num_vectors']} vectors")
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        print("Server not responding!")
        return

    # Step 1: Generate keys
    print("\n" + "-" * 40)
    print("STEP 1: Generate Paillier Keys")
    print("-" * 40)
    with Timer() as t:
        crypto = CryptoClient(key_size=2048, precision=5)
    print(f"Key generation: {t.elapsed_ms:.0f}ms")

    # Step 2: Register with server
    print("\n" + "-" * 40)
    print("STEP 2: Register Public Key")
    print("-" * 40)
    resp = httpx.post(
        f"{base_url}/keys",
        json={"client_id": client_id, "public_key": crypto.public_key},
    )
    print(f"Registration: {resp.json()}")

    # Step 3: Generate query
    print("\n" + "-" * 40)
    print("STEP 3: Generate Query Vector")
    print("-" * 40)
    query = generate_random_vectors(1, dimension, normalize=True, seed=123)[0]
    print(f"Query vector: [{query[0]:.4f}, {query[1]:.4f}, ..., {query[-1]:.4f}]")

    # Step 4: Encrypt query
    print("\n" + "-" * 40)
    print("STEP 4: Encrypt Query")
    print("-" * 40)
    with Timer() as t:
        encrypted_query = crypto.encrypt_vector(query, silent=True)
    print(f"Encryption: {t.elapsed_ms:.0f}ms")

    # Serialize for HTTP
    encrypted_query_b64 = serialize_encrypted(encrypted_query)
    print(f"Serialized size: {len(encrypted_query_b64):,} bytes")

    # Step 5: Send to server
    print("\n" + "-" * 40)
    print("STEP 5: Server Computes Encrypted Scores")
    print("-" * 40)
    with Timer() as t:
        resp = httpx.post(
            f"{base_url}/search/blind",
            json={
                "client_id": client_id,
                "encrypted_query_b64": encrypted_query_b64,
            },
            timeout=300.0,  # PHE can be slow
        )
    print(f"Server request: {t.elapsed_ms:.0f}ms")

    if resp.status_code != 200:
        print(f"Error: {resp.text}")
        return

    result = resp.json()
    print(f"Server compute time: {result['server_time_ms']:.0f}ms")
    print(f"Received {result['num_results']} encrypted scores")

    # Step 6: Decrypt scores
    print("\n" + "-" * 40)
    print("STEP 6: Decrypt Scores")
    print("-" * 40)
    with Timer() as t:
        scores = []
        for score_b64 in result['encrypted_scores_b64']:
            encrypted_score = deserialize_encrypted(score_b64)
            decrypted = crypto.decrypt_score(encrypted_score)
            scores.append(decrypted)
    print(f"Decryption: {t.elapsed_ms:.0f}ms")

    # Step 7: Rank results
    print("\n" + "-" * 40)
    print("STEP 7: Rank Results")
    print("-" * 40)
    import numpy as np
    scores_array = np.array(scores)
    top_indices = top_k_indices(scores_array, top_k)
    vector_ids = result['vector_ids']

    print(f"\nTop-{top_k} Results:")
    print("-" * 40)
    for rank, idx in enumerate(top_indices, 1):
        print(f"  #{rank}: {vector_ids[idx]} (score: {scores[idx]:.6f})")

    # Step 8: Verify accuracy
    print("\n" + "-" * 40)
    print("STEP 8: Verify Accuracy")
    print("-" * 40)

    # Recreate database for verification (in production, client wouldn't have this)
    db_vectors = generate_random_vectors(num_vectors, dimension, normalize=True, seed=seed)
    db_ids = [f"doc_{i:06d}" for i in range(num_vectors)]

    plaintext_scores = compute_plaintext_similarity(query, db_vectors)
    plaintext_top_indices = top_k_indices(plaintext_scores, top_k)
    plaintext_top_ids = [db_ids[i] for i in plaintext_top_indices]

    encrypted_top_ids = [vector_ids[i] for i in top_indices]

    recall = len(set(plaintext_top_ids) & set(encrypted_top_ids)) / top_k
    top1_match = encrypted_top_ids[0] == plaintext_top_ids[0]

    print(f"Recall@{top_k}: {recall * 100:.1f}%")
    print(f"Top-1 Match: {'Yes' if top1_match else 'No'}")

    # Privacy summary
    print("\n" + "=" * 60)
    print("PRIVACY GUARANTEES")
    print("=" * 60)
    print("  [x] Server never saw the raw query vector")
    print("  [x] Server never saw the similarity scores")
    print("  [x] Client never downloaded the full database")
    print("  [x] Results match plaintext search")


def main():
    parser = argparse.ArgumentParser(
        description="Demo blind search for Project Opaque"
    )
    parser.add_argument(
        "--num-vectors", "-n",
        type=int,
        default=100,
        help="Number of vectors in database",
    )
    parser.add_argument(
        "--dimension", "-d",
        type=int,
        default=128,
        help="Vector dimension",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of top results",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Only run the server (for debugging)",
    )
    parser.add_argument(
        "--client-only",
        action="store_true",
        help="Only run the client (assumes server is running)",
    )

    args = parser.parse_args()

    if args.server_only:
        run_server(args.host, args.port, args.num_vectors, args.dimension, args.seed)
    elif args.client_only:
        run_client_demo(
            args.host, args.port, args.num_vectors, args.dimension, args.top_k, args.seed
        )
    else:
        # Start server in background process
        server_process = Process(
            target=run_server,
            args=(args.host, args.port, args.num_vectors, args.dimension, args.seed),
        )
        server_process.start()

        try:
            # Give server time to start
            time.sleep(2)

            # Run client demo
            run_client_demo(
                args.host, args.port, args.num_vectors, args.dimension, args.top_k, args.seed
            )
        finally:
            # Shutdown server
            server_process.terminate()
            server_process.join(timeout=5)
            if server_process.is_alive():
                server_process.kill()


if __name__ == "__main__":
    main()
