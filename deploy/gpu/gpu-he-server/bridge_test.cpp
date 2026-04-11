// bridge_test.cpp — End-to-end test of Lattigo → HEonGPU key bridge.
//
// This program:
// 1. Loads Galois keys from a file in HEonGPU format (produced by our Go converter)
// 2. Creates an HEonGPU context with matching params
// 3. Generates a secret key + encrypts a test vector
// 4. Performs a rotation using the loaded Galois keys
// 5. Decrypts and verifies the result
//
// If the rotation produces correct results, the key format bridge works.
//
// Usage:
//   ./bridge_test <galois_keys.bin>
//
// Build as HEonGPU example (add to example/basic/CMakeLists.txt):
//   add_executable(bridge_test bridge_test.cpp)
//   target_link_libraries(bridge_test PRIVATE heongpu)

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <sstream>
#include <chrono>

#include <heongpu/heongpu.hpp>

using namespace heongpu;
constexpr auto S = Scheme::CKKS;

int main(int argc, char** argv) {
    std::cout << "=== Lattigo → HEonGPU Key Bridge Test ===" << std::endl;

    // Create context matching Lattigo GPU params
    auto context = GenHEContext<S>(sec_level_type::none);
    context->set_poly_modulus_degree(16384);

    std::vector<uint64_t> q_primes = {
        1152921504606748673ULL, 35184372121601ULL, 35184372744193ULL,
        35184373006337ULL, 35184371138561ULL, 35184370941953ULL,
        35184370352129ULL, 35184373989377ULL
    };
    std::vector<uint64_t> p_primes = { 2305843009211662337ULL };

    context->set_coeff_modulus_values(q_primes, p_primes);
    context->generate();

    int ring_size = 16384;
    int Q_size = q_primes.size();

    std::cout << "Context: N=" << ring_size << ", Q=" << Q_size
              << ", P=" << p_primes.size() << std::endl;

    // Generate a local secret key for testing
    HEKeyGenerator<S> keygen(context);
    Secretkey<S> sk(context);
    keygen.generate_secret_key(sk);
    Publickey<S> pk(context);
    keygen.generate_public_key(pk, sk);

    // Load Galois keys from file if provided
    if (argc > 1) {
        std::string gk_path = argv[1];
        std::cout << "\nLoading Galois keys from: " << gk_path << std::endl;

        std::ifstream gk_file(gk_path, std::ios::binary);
        if (!gk_file.is_open()) {
            std::cerr << "ERROR: Cannot open " << gk_path << std::endl;
            return 1;
        }

        // Get file size
        gk_file.seekg(0, std::ios::end);
        size_t file_size = gk_file.tellg();
        gk_file.seekg(0, std::ios::beg);
        std::cout << "File size: " << file_size << " bytes ("
                  << file_size / (1024*1024) << " MB)" << std::endl;

        try {
            Galoiskey<S> gk;
            gk.set_context(context);
            gk.load(gk_file);
            gk_file.close();

            std::cout << "Galois keys loaded successfully!" << std::endl;

            // Test: encrypt a vector, rotate it, decrypt
            HEEncoder<S> encoder(context);
            HEEncryptor<S> encryptor(context, pk);
            HEDecryptor<S> decryptor(context, sk);
            HEArithmeticOperator<S> ops(context, encoder);

            double scale = pow(2.0, 45);
            int slot_count = ring_size / 2;

            std::vector<double> msg(slot_count, 0.0);
            msg[0] = 10.0; msg[1] = 20.0; msg[2] = 30.0; msg[3] = 40.0;

            Plaintext<S> pt(context);
            encoder.encode(pt, msg, scale);
            Ciphertext<S> ct(context);
            encryptor.encrypt(ct, pt);

            std::cout << "\nOriginal: [10, 20, 30, 40, ...]" << std::endl;

            // Try rotation by 1 using the loaded Galois keys
            auto start = std::chrono::high_resolution_clock::now();
            Ciphertext<S> ct_rot(ct);
            ops.rotate_rows_inplace(ct_rot, gk, 1);
            auto end = std::chrono::high_resolution_clock::now();
            auto rotate_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            Plaintext<S> pt_rot(context);
            decryptor.decrypt(pt_rot, ct_rot);
            std::vector<double> decoded(slot_count);
            encoder.decode(decoded, pt_rot);

            std::cout << "After rot(1): [";
            for (int i = 0; i < 5; i++) {
                std::cout << std::fixed << std::setprecision(2) << decoded[i];
                if (i < 4) std::cout << ", ";
            }
            std::cout << ", ...]" << std::endl;
            std::cout << "Rotation time: " << rotate_us << " us" << std::endl;

            // Verify correctness
            bool correct = (std::abs(decoded[0] - 20.0) < 0.1 &&
                           std::abs(decoded[1] - 30.0) < 0.1 &&
                           std::abs(decoded[2] - 40.0) < 0.1);

            if (correct) {
                std::cout << "\n✓ BRIDGE WORKS! Rotation with Lattigo-generated keys is correct." << std::endl;
            } else {
                std::cout << "\n✗ BRIDGE FAILED! Rotation produced incorrect results." << std::endl;
                std::cout << "Expected: [20, 30, 40, ...] but got different values." << std::endl;
                std::cout << "This means the coefficient ordering differs between libraries." << std::endl;
                return 1;
            }

            // Benchmark multiple rotations
            std::cout << "\n=== Rotation Benchmarks ===" << std::endl;
            for (int shift : {1, 2, 4, 8}) {
                auto bstart = std::chrono::high_resolution_clock::now();
                int n_iters = 10;
                for (int i = 0; i < n_iters; i++) {
                    Ciphertext<S> ct_bench(ct);
                    ops.rotate_rows_inplace(ct_bench, gk, shift);
                }
                auto bend = std::chrono::high_resolution_clock::now();
                auto avg_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    bend - bstart).count() / n_iters;
                std::cout << "  Rotate by " << shift << ": " << avg_us << " us ("
                          << avg_us / 1000.0 << " ms)" << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "ERROR loading Galois keys: " << e.what() << std::endl;
            std::cerr << "This likely means the format doesn't match." << std::endl;
            return 1;
        }
    } else {
        // No file provided — generate locally and show expected sizes
        std::cout << "\nNo Galois key file provided. Generating locally for reference..." << std::endl;

        std::vector<int> shifts;
        for (int i = 0; i < 14; i++) shifts.push_back(1 << i);
        Galoiskey<S> gk(context, shifts);
        keygen.generate_galois_key(gk, sk);

        gk.store_in_host();
        std::stringstream ss;
        gk.save(ss);
        std::cout << "Native HEonGPU Galois key size: " << ss.str().size()
                  << " bytes (" << ss.str().size() / (1024*1024) << " MB)" << std::endl;

        std::cout << "\nTo test the bridge:" << std::endl;
        std::cout << "1. Generate Galois keys in Lattigo with NewParametersGPU()" << std::endl;
        std::cout << "2. Serialize with SerializeGaloisKeysHEonGPU()" << std::endl;
        std::cout << "3. Save to file and run: ./bridge_test <file>" << std::endl;
    }

    return 0;
}
