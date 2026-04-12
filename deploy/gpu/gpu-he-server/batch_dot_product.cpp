// batch_dot_product.cpp — End-to-end test of GPU batch dot product.
//
// Tests the full pipeline:
// 1. Load Lattigo eval keys (already HEonGPU NTT domain)
// 2. Load Lattigo secret key (for encryption/decryption in test)
// 3. Encrypt a packed query vector
// 4. Encode packed centroids as plaintext
// 5. GPU: MulNew + Rescale + RotateNew loop
// 6. Decrypt and verify scores match expected dot products
//
// Usage: ./batch_dot_product <galois_keys.bin> <secret_key.bin>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <heongpu/heongpu.hpp>

using namespace heongpu;
constexpr auto S = Scheme::CKKS;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./batch_dot_product <galois_keys.bin> <secret_key.bin>" << std::endl;
        return 1;
    }

    auto ctx = GenHEContext<S>(sec_level_type::none);
    ctx->set_poly_modulus_degree(16384);
    std::vector<uint64_t> q = {1152921504606748673ULL,35184372121601ULL,35184372744193ULL,
        35184373006337ULL,35184371138561ULL,35184370941953ULL,35184370352129ULL,35184373989377ULL};
    std::vector<uint64_t> p = {2305843009211662337ULL};
    ctx->set_coeff_modulus_values(q, p);
    ctx->generate();

    int N = 16384;
    int QP = q.size() + p.size();
    int Q_size = q.size();
    double scale = pow(2.0, 45);
    int slot_count = N / 2; // 8192 usable CKKS slots
    int dim = 128; // Vector dimension
    int centroids_per_pack = slot_count / dim; // 64 centroids per pack

    std::cout << "=== GPU Batch Dot Product Test ===" << std::endl;
    std::cout << "N=" << N << " slots=" << slot_count << " dim=" << dim
              << " centroids/pack=" << centroids_per_pack << std::endl;

    // Load Galois keys
    std::cout << "\nLoading Galois keys..." << std::endl;
    std::ifstream gk_file(argv[1], std::ios::binary);
    Galoiskey<S> gk;
    gk.set_context(ctx);
    gk.load(gk_file);
    gk_file.close();

    // Load and reconstruct secret key
    std::cout << "Loading secret key..." << std::endl;
    std::ifstream sk_file(argv[2], std::ios::binary);
    std::vector<uint64_t> sk_data(QP * N);
    sk_file.read(reinterpret_cast<char*>(sk_data.data()), QP * N * sizeof(uint64_t));
    sk_file.close();

    HEKeyGenerator<S> keygen(ctx);
    Secretkey<S> sk(ctx);
    keygen.generate_secret_key(sk);
    sk.store_in_device();
    cudaMemcpy(sk.data(), sk_data.data(), QP * N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    Publickey<S> pk(ctx);
    keygen.generate_public_key(pk, sk);

    HEEncoder<S> encoder(ctx);
    HEEncryptor<S> encryptor(ctx, pk);
    HEDecryptor<S> decryptor(ctx, sk);
    HEArithmeticOperator<S> ops(ctx, encoder);

    // Create a query vector: [0.1, 0.2, 0.3, ..., 0.128] repeated across slots
    std::vector<double> packed_query(slot_count, 0.0);
    for (int c = 0; c < centroids_per_pack; c++) {
        for (int d = 0; d < dim; d++) {
            packed_query[c * dim + d] = (d + 1) * 0.01; // [0.01, 0.02, ..., 1.28]
        }
    }

    // Create centroid vectors: centroid[i] = constant (i+1)*0.01
    std::vector<double> packed_centroids(slot_count, 0.0);
    for (int c = 0; c < centroids_per_pack && c < 4; c++) {
        for (int d = 0; d < dim; d++) {
            packed_centroids[c * dim + d] = (c + 1) * 0.01; // centroid 0: all 0.01, centroid 1: all 0.02, etc.
        }
    }

    // Compute expected dot products
    std::cout << "\nExpected dot products:" << std::endl;
    for (int c = 0; c < 4; c++) {
        double dot = 0;
        for (int d = 0; d < dim; d++) {
            dot += packed_query[c * dim + d] * packed_centroids[c * dim + d];
        }
        std::cout << "  centroid " << c << ": " << std::fixed << std::setprecision(6) << dot << std::endl;
    }

    // Encrypt query
    Plaintext<S> pt_query(ctx);
    encoder.encode(pt_query, packed_query, scale);
    Ciphertext<S> ct_query(ctx);
    encryptor.encrypt(ct_query, pt_query);

    // Encode centroids as plaintext
    Plaintext<S> pt_centroids(ctx);
    encoder.encode(pt_centroids, packed_centroids, scale);

    // === GPU Batch Dot Product ===
    std::cout << "\n=== Performing GPU Batch Dot Product ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: Multiply encrypted query × plaintext centroids
    // multiply_plain(input_ct, input_pt, output_ct)
    Ciphertext<S> ct_result(ctx);
    ops.multiply_plain(ct_query, pt_centroids, ct_result);

    auto after_mul = std::chrono::high_resolution_clock::now();

    // Step 2: Rescale
    ops.rescale_inplace(ct_result);

    auto after_rescale = std::chrono::high_resolution_clock::now();

    // Step 3: Rotation sum within each dim-sized segment
    // After multiply, slot[c*dim + d] = query[d] * centroid[c][d]
    // We need to sum within each segment: sum_{d=0}^{dim-1} slot[c*dim + d]
    for (int stride = 1; stride < dim; stride *= 2) {
        Ciphertext<S> ct_rotated(ct_result);
        ops.rotate_rows_inplace(ct_rotated, gk, stride);
        ops.add_inplace(ct_result, ct_rotated);
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto mul_us = std::chrono::duration_cast<std::chrono::microseconds>(after_mul - start).count();
    auto rescale_us = std::chrono::duration_cast<std::chrono::microseconds>(after_rescale - after_mul).count();
    auto rotate_us = std::chrono::duration_cast<std::chrono::microseconds>(end - after_rescale).count();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "  Multiply: " << mul_us << " us" << std::endl;
    std::cout << "  Rescale:  " << rescale_us << " us" << std::endl;
    std::cout << "  Rotations (" << (int)log2(dim) << " steps): " << rotate_us << " us" << std::endl;
    std::cout << "  TOTAL:    " << total_us << " us (" << total_us/1000.0 << " ms)" << std::endl;

    // Decrypt and extract scores at positions [0, dim, 2*dim, ...]
    Plaintext<S> pt_result(ctx);
    decryptor.decrypt(pt_result, ct_result);
    std::vector<double> decoded(slot_count);
    encoder.decode(decoded, pt_result);

    std::cout << "\n=== Results ===" << std::endl;
    bool all_correct = true;
    for (int c = 0; c < 4; c++) {
        double score = decoded[c * dim];
        double expected = 0;
        for (int d = 0; d < dim; d++) {
            expected += packed_query[c * dim + d] * packed_centroids[c * dim + d];
        }
        double error = std::abs(score - expected);
        bool ok = error < 0.1; // CKKS tolerance
        std::cout << "  Centroid " << c << ": score=" << std::fixed << std::setprecision(6)
                  << score << " expected=" << expected << " error=" << error
                  << (ok ? " ✓" : " ✗") << std::endl;
        if (!ok) all_correct = false;
    }

    if (all_correct) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "  GPU BATCH DOT PRODUCT WORKS!" << std::endl;
        std::cout << "  Total time: " << total_us/1000.0 << " ms" << std::endl;
        std::cout << "========================================" << std::endl;

        // Benchmark: run 10 iterations
        std::cout << "\n=== Benchmark (10 iterations) ===" << std::endl;
        auto bench_start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 10; iter++) {
            Ciphertext<S> ct_bench(ctx);
            ops.multiply_plain(ct_query, pt_centroids, ct_bench);
            ops.rescale_inplace(ct_bench);
            for (int stride = 1; stride < dim; stride *= 2) {
                Ciphertext<S> ct_rot(ct_bench);
                ops.rotate_rows_inplace(ct_rot, gk, stride);
                ops.add_inplace(ct_bench, ct_rot);
            }
        }
        auto bench_end = std::chrono::high_resolution_clock::now();
        auto avg_us = std::chrono::duration_cast<std::chrono::microseconds>(bench_end - bench_start).count() / 10;
        std::cout << "  Average: " << avg_us << " us (" << avg_us/1000.0 << " ms) per batch dot product" << std::endl;
    } else {
        std::cout << "\n✗ BATCH DOT PRODUCT FAILED" << std::endl;
        return 1;
    }

    return 0;
}
