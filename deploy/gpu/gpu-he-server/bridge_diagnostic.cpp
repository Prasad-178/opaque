// bridge_diagnostic.cpp — Comprehensive HEonGPU bridge diagnostic.
//
// Tests EVERY assumption about the Lattigo ↔ HEonGPU data bridge:
//   1. Ciphertext memory layout (poly-major vs level-major)
//   2. get_data() return format after operations
//   3. Scale propagation through multiply_plain + rescale
//   4. Depth tracking through the compute chain
//   5. Data round-trip: raw load → compute → raw extract
//
// Build:  (from HEonGPU/build)
//   cmake .. -DCMAKE_CUDA_ARCHITECTURES=75 -DHEonGPU_BUILD_EXAMPLES=ON && make bridge_diagnostic
//
// Run:    ./bridge_diagnostic
//
// Output: Detailed report of every internal state, suitable for comparing
//         with Go client expectations.

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <heongpu/heongpu.hpp>

using namespace heongpu;
constexpr auto S = Scheme::CKKS;

void print_separator(const char* title) {
    std::cout << "\n========================================\n"
              << "  " << title << "\n"
              << "========================================\n";
}

int main() {
    // Use the EXACT same parameters as Opaque (GPU-compatible: LogP=[61])
    auto ctx = GenHEContext<S>(sec_level_type::none);
    ctx->set_poly_modulus_degree(16384);
    std::vector<uint64_t> q = {
        1152921504606748673ULL, 35184372121601ULL, 35184372744193ULL,
        35184373006337ULL, 35184371138561ULL, 35184370941953ULL,
        35184370352129ULL, 35184373989377ULL
    };
    std::vector<uint64_t> p = {2305843009211662337ULL};
    ctx->set_coeff_modulus_values(q, p);
    ctx->generate();

    int N = 16384;
    int Q_size = q.size();    // 8
    int QP_size = q.size() + p.size(); // 9
    double scale = pow(2.0, 45);
    int slot_count = N / 2;   // 8192
    int dim = 128;

    // Generate keys
    HEKeyGenerator<S> keygen(ctx);
    Secretkey<S> sk(ctx);
    keygen.generate_secret_key(sk);
    Publickey<S> pk(ctx);
    keygen.generate_public_key(pk, sk);

    HEEncoder<S> encoder(ctx);
    HEEncryptor<S> encryptor(ctx, pk);
    HEDecryptor<S> decryptor(ctx, sk);
    HEArithmeticOperator<S> ops(ctx, encoder);

    // ============================================================
    print_separator("TEST 1: Ciphertext Internal State After Encryption");
    // ============================================================

    std::vector<double> query(slot_count, 0.0);
    for (int i = 0; i < dim; i++) query[i] = (i + 1) * 0.01;

    Plaintext<S> pt_q(ctx);
    encoder.encode(pt_q, query, scale);

    Ciphertext<S> ct_q(ctx);
    encryptor.encrypt(ct_q, pt_q);

    std::cout << "After encryption:\n"
              << "  coeff_modulus_count = " << ct_q.coeff_modulus_count() << " (expect " << Q_size << ")\n"
              << "  size (cipher_size)  = " << ct_q.size() << " (expect 2)\n"
              << "  depth               = " << ct_q.depth() << " (expect 0)\n"
              << "  scale               = " << std::fixed << std::setprecision(1) << ct_q.scale() << " (expect " << scale << ")\n"
              << "  in_ntt_domain       = " << ct_q.in_ntt_domain() << " (expect 1)\n";

    // Get raw data and check size
    ct_q.store_in_host();
    std::vector<uint64_t> ct_raw;
    ct_q.get_data(ct_raw);
    int expected_size = ct_q.size() * (ct_q.coeff_modulus_count() - ct_q.depth()) * N;
    std::cout << "  raw data size       = " << ct_raw.size() << " (expect " << expected_size << ")\n";
    std::cout << "  first 5 coeffs      = [";
    for (int i = 0; i < 5; i++) std::cout << ct_raw[i] << (i < 4 ? "," : "");
    std::cout << "]\n";

    // ============================================================
    print_separator("TEST 2: Plaintext Internal State After Encoding");
    // ============================================================

    std::vector<double> centroids(slot_count, 0.0);
    for (int c = 0; c < 4; c++)
        for (int d = 0; d < dim; d++)
            centroids[c * dim + d] = (c + 1) * 0.01;

    Plaintext<S> pt_c(ctx);
    encoder.encode(pt_c, centroids, scale);

    std::cout << "After encoding:\n"
              << "  depth               = " << pt_c.depth() << " (expect 0)\n"
              << "  scale               = " << std::fixed << std::setprecision(1) << pt_c.scale() << " (expect " << scale << ")\n"
              << "  in_ntt_domain       = " << pt_c.in_ntt_domain() << " (expect 1)\n";

    // ============================================================
    print_separator("TEST 3: Memory Layout — get_data() format");
    // ============================================================

    // Write known pattern to ciphertext, get_data, check layout
    Ciphertext<S> ct_pattern(ctx);
    encryptor.encrypt(ct_pattern, pt_q);
    ct_pattern.store_in_device();

    // Write pattern: poly0_level_i coefficient_j = i*1000000 + j
    int active_levels = ct_pattern.coeff_modulus_count() - ct_pattern.depth();
    int total_size = ct_pattern.size() * active_levels * N;
    std::vector<uint64_t> pattern(total_size);
    for (int poly = 0; poly < ct_pattern.size(); poly++) {
        for (int lvl = 0; lvl < active_levels; lvl++) {
            for (int j = 0; j < N; j++) {
                // Index assuming POLY-MAJOR layout: [poly0_all_levels, poly1_all_levels]
                int idx_poly_major = (poly * active_levels + lvl) * N + j;
                pattern[idx_poly_major] = poly * 100000000ULL + lvl * 1000000ULL + j;
            }
        }
    }
    cudaMemcpy(ct_pattern.data(), pattern.data(), total_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Read back via get_data
    ct_pattern.store_in_host();
    std::vector<uint64_t> readback;
    ct_pattern.get_data(readback);

    std::cout << "Layout test (write poly-major, read via get_data):\n"
              << "  Total size written = " << total_size << "\n"
              << "  Total size read    = " << readback.size() << "\n";

    // Check if readback matches poly-major pattern
    bool poly_major_match = true;
    int mismatches = 0;
    for (int i = 0; i < std::min((int)readback.size(), total_size); i++) {
        if (readback[i] != pattern[i]) {
            poly_major_match = false;
            mismatches++;
            if (mismatches <= 5) {
                std::cout << "  Mismatch at [" << i << "]: wrote=" << pattern[i] << " read=" << readback[i] << "\n";
            }
        }
    }
    if (poly_major_match) {
        std::cout << "  RESULT: get_data() returns POLY-MAJOR layout ✓\n";
    } else {
        std::cout << "  RESULT: get_data() does NOT return poly-major! " << mismatches << " mismatches\n";
        // Try to detect actual layout
        // Check if it's level-major: [lvl0_poly0, lvl0_poly1, lvl1_poly0, lvl1_poly1, ...]
        std::cout << "  Checking level-major layout...\n";
        bool level_major = true;
        for (int lvl = 0; lvl < active_levels && level_major; lvl++) {
            for (int poly = 0; poly < 2 && level_major; poly++) {
                int src_idx = (lvl * 2 + poly) * N;
                if (src_idx < (int)readback.size()) {
                    uint64_t expected = poly * 100000000ULL + lvl * 1000000ULL + 0;
                    if (readback[src_idx] != expected) level_major = false;
                }
            }
        }
        if (level_major) {
            std::cout << "  RESULT: get_data() returns LEVEL-MAJOR layout!\n";
        } else {
            std::cout << "  RESULT: Unknown layout. First 10 values:\n";
            for (int i = 0; i < 10; i++) {
                std::cout << "    [" << i << "] = " << readback[i]
                          << " (poly=" << readback[i]/100000000
                          << " lvl=" << (readback[i]%100000000)/1000000
                          << " j=" << readback[i]%1000000 << ")\n";
            }
        }
    }

    // ============================================================
    print_separator("TEST 4: Compute Chain — Scale/Depth Tracking");
    // ============================================================

    // Re-encrypt fresh
    Ciphertext<S> ct_fresh(ctx);
    encryptor.encrypt(ct_fresh, pt_q);

    std::cout << "Before compute:\n"
              << "  ct depth=" << ct_fresh.depth() << " scale=" << ct_fresh.scale() << "\n"
              << "  pt depth=" << pt_c.depth() << " scale=" << pt_c.scale() << "\n";

    // Step 1: multiply_plain
    Ciphertext<S> ct_mul(ctx);
    ops.multiply_plain(ct_fresh, pt_c, ct_mul);
    std::cout << "After multiply_plain:\n"
              << "  depth=" << ct_mul.depth() << " scale=" << ct_mul.scale() << "\n"
              << "  coeff_modulus_count=" << ct_mul.coeff_modulus_count() << "\n";

    // Step 2: rescale
    ops.rescale_inplace(ct_mul);
    std::cout << "After rescale:\n"
              << "  depth=" << ct_mul.depth() << " scale=" << ct_mul.scale() << "\n"
              << "  coeff_modulus_count=" << ct_mul.coeff_modulus_count() << "\n";

    // Get result data
    ct_mul.store_in_host();
    std::vector<uint64_t> result_data;
    ct_mul.get_data(result_data);
    int result_levels = ct_mul.coeff_modulus_count() - ct_mul.depth();
    std::cout << "  result data size    = " << result_data.size() << "\n"
              << "  expected size       = " << ct_mul.size() << " * " << result_levels << " * " << N
              << " = " << (ct_mul.size() * result_levels * N) << "\n";

    // ============================================================
    print_separator("TEST 5: Raw Load → Compute → Verify (Simulates Bridge)");
    // ============================================================

    // This simulates what the GPU server does:
    // 1. Create default Ciphertext(ctx)
    // 2. Overwrite data via cudaMemcpy (from Go client data)
    // 3. Set scale and depth
    // 4. Run multiply_plain + rescale + rotate
    // 5. Extract result and compare with native computation

    // Native computation (ground truth)
    Ciphertext<S> ct_native(ctx);
    encryptor.encrypt(ct_native, pt_q);

    Ciphertext<S> result_native(ctx);
    ops.multiply_plain(ct_native, pt_c, result_native);
    ops.rescale_inplace(result_native);

    Plaintext<S> pt_native(ctx);
    decryptor.decrypt(pt_native, result_native);
    std::vector<double> decoded_native(slot_count);
    encoder.decode(decoded_native, pt_native);

    // Simulated bridge computation
    // Step A: Extract raw data from native ct
    ct_native.store_in_host();
    std::vector<uint64_t> native_raw;
    ct_native.get_data(native_raw);

    // Step B: Create new ct and load raw data (simulate cudaMemcpy)
    Ciphertext<S> ct_bridge(ctx);
    ct_bridge.store_in_device();
    cudaMemcpy(ct_bridge.data(), native_raw.data(), native_raw.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    ct_bridge.set_scale(ct_native.scale());
    ct_bridge.set_depth(ct_native.depth());

    // Step C: Compute with bridge ct
    Ciphertext<S> result_bridge(ctx);
    ops.multiply_plain(ct_bridge, pt_c, result_bridge);
    ops.rescale_inplace(result_bridge);

    // Step D: Decrypt bridge result
    Plaintext<S> pt_bridge(ctx);
    decryptor.decrypt(pt_bridge, result_bridge);
    std::vector<double> decoded_bridge(slot_count);
    encoder.decode(decoded_bridge, pt_bridge);

    // Compare
    double max_err = 0;
    for (int i = 0; i < slot_count; i++) {
        double err = std::abs(decoded_native[i] - decoded_bridge[i]);
        if (err > max_err) max_err = err;
    }

    std::cout << "Native vs Bridge (same data, cudaMemcpy reconstruction):\n";
    for (int c = 0; c < 4; c++) {
        std::cout << "  Slot " << c * dim << ": native=" << std::setprecision(6) << decoded_native[c * dim]
                  << " bridge=" << decoded_bridge[c * dim]
                  << " err=" << std::scientific << std::abs(decoded_native[c * dim] - decoded_bridge[c * dim]) << "\n";
    }
    std::cout << "  Max error: " << std::scientific << max_err << "\n";
    if (max_err < 0.001) {
        std::cout << "  RESULT: cudaMemcpy bridge WORKS ✓\n";
    } else {
        std::cout << "  RESULT: cudaMemcpy bridge FAILS ✗\n";
    }

    // ============================================================
    print_separator("TEST 6: Result Extraction Layout");
    // ============================================================

    // After compute, extract result and verify the layout we use to return to Go
    result_bridge.store_in_host();
    std::vector<uint64_t> bridge_result_raw;
    result_bridge.get_data(bridge_result_raw);

    int r_levels = result_bridge.coeff_modulus_count() - result_bridge.depth();
    int r_polys = result_bridge.size();

    std::cout << "Result ciphertext state:\n"
              << "  size=" << r_polys << " depth=" << result_bridge.depth()
              << " active_levels=" << r_levels << " scale=" << result_bridge.scale() << "\n"
              << "  raw data total     = " << bridge_result_raw.size() << " uint64\n"
              << "  expected (poly*lvl*N) = " << r_polys * r_levels * N << " uint64\n"
              << "  in_ntt_domain      = " << result_bridge.in_ntt_domain() << "\n";

    // Verify the server's extraction code logic
    std::cout << "\nServer extraction simulation:\n";
    for (int p_idx = 0; p_idx < r_polys; p_idx++) {
        int offset = p_idx * r_levels * N;
        std::cout << "  Poly " << p_idx << ": offset=" << offset
                  << " size=" << r_levels * N
                  << " end=" << offset + r_levels * N << "\n";
        if (offset + r_levels * N > (int)bridge_result_raw.size()) {
            std::cout << "  ERROR: extraction goes out of bounds!\n";
        }
    }

    std::cout << "\n========================================\n"
              << "  DIAGNOSTIC COMPLETE\n"
              << "========================================\n";

    return 0;
}
