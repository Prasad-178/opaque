// bridge_with_roots.cpp — Extracts NTT roots AND tests key loading.
// 1. Creates HEonGPU context → prints ψ values
// 2. If a galois key file is provided, loads and tests rotation
//
// Build as HEonGPU example.
// Usage: ./bridge_with_roots [galois_keys.bin]

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
    std::cout << "=== Bridge With NTT Root Exchange ===" << std::endl;

    auto context = GenHEContext<S>(sec_level_type::none);
    context->set_poly_modulus_degree(16384);

    std::vector<uint64_t> q = {
        1152921504606748673ULL, 35184372121601ULL, 35184372744193ULL,
        35184373006337ULL, 35184371138561ULL, 35184370941953ULL,
        35184370352129ULL, 35184373989377ULL
    };
    std::vector<uint64_t> p = { 2305843009211662337ULL };

    context->set_coeff_modulus_values(q, p);
    context->generate();

    // Extract NTT roots using the same function the context uses internally
    std::vector<Modulus64> prime_vector;
    for (auto qi : q) prime_vector.push_back(Modulus64(qi));
    for (auto pi : p) prime_vector.push_back(Modulus64(pi));

    std::vector<uint64_t> ntt_roots =
        generate_primitive_root_of_unity(16384, prime_vector);

    std::cout << "\n=== NTT ROOTS (paste these into Go client) ===" << std::endl;
    std::cout << "serverRoots := []uint64{" << std::endl;
    for (size_t i = 0; i < ntt_roots.size(); i++) {
        std::cout << "    " << ntt_roots[i] << "ULL,";
        if (i < q.size()) std::cout << " // Q[" << i << "]";
        else std::cout << " // P[" << (i - q.size()) << "]";
        std::cout << std::endl;
    }
    std::cout << "}" << std::endl;

    // If file provided, load and test
    if (argc > 1) {
        std::string path = argv[1];
        std::cout << "\n=== Loading Galois keys from: " << path << " ===" << std::endl;

        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Cannot open " << path << std::endl;
            return 1;
        }

        f.seekg(0, std::ios::end);
        std::cout << "File size: " << f.tellg() << " bytes" << std::endl;
        f.seekg(0, std::ios::beg);

        try {
            Galoiskey<S> gk;
            gk.set_context(context);
            gk.load(f);
            f.close();
            std::cout << "Keys loaded successfully!" << std::endl;

            // Generate local keys for testing
            HEKeyGenerator<S> keygen(context);
            Secretkey<S> sk(context);
            keygen.generate_secret_key(sk);
            Publickey<S> pk(context);
            keygen.generate_public_key(pk, sk);

            HEEncoder<S> encoder(context);
            HEEncryptor<S> encryptor(context, pk);
            HEDecryptor<S> decryptor(context, sk);
            HEArithmeticOperator<S> ops(context, encoder);

            double scale = pow(2.0, 45);
            int slot_count = 16384 / 2;

            std::vector<double> msg(slot_count, 0.0);
            msg[0] = 10.0; msg[1] = 20.0; msg[2] = 30.0; msg[3] = 40.0;

            Plaintext<S> pt(context);
            encoder.encode(pt, msg, scale);
            Ciphertext<S> ct(context);
            encryptor.encrypt(ct, pt);

            // Test rotation with loaded keys
            Ciphertext<S> ct_rot(ct);
            ops.rotate_rows_inplace(ct_rot, gk, 1);

            Plaintext<S> pt_dec(context);
            decryptor.decrypt(pt_dec, ct_rot);
            std::vector<double> decoded(slot_count);
            encoder.decode(decoded, pt_dec);

            std::cout << "\nOriginal:  [10, 20, 30, 40, ...]" << std::endl;
            std::cout << "After rot: [";
            for (int i = 0; i < 5; i++) {
                std::cout << std::fixed << std::setprecision(2) << decoded[i];
                if (i < 4) std::cout << ", ";
            }
            std::cout << ", ...]" << std::endl;

            bool correct = (std::abs(decoded[0] - 20.0) < 0.5 &&
                           std::abs(decoded[1] - 30.0) < 0.5 &&
                           std::abs(decoded[2] - 40.0) < 0.5);

            if (correct) {
                std::cout << "\n✓ BRIDGE WORKS! NTT root exchange successful!" << std::endl;

                // Benchmark
                std::cout << "\n=== Rotation Benchmarks ===" << std::endl;
                for (int shift : {1, 2, 4, 8, 16, 32, 64, 128}) {
                    auto start = std::chrono::high_resolution_clock::now();
                    int iters = 20;
                    for (int i = 0; i < iters; i++) {
                        Ciphertext<S> tmp(ct);
                        ops.rotate_rows_inplace(tmp, gk, shift);
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    auto avg_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start).count() / iters;
                    std::cout << "  Rotate by " << shift << ": " << avg_us << " us ("
                              << avg_us / 1000.0 << " ms)" << std::endl;
                }
            } else {
                std::cout << "\n✗ BRIDGE FAILED!" << std::endl;
                return 1;
            }

        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            return 1;
        }
    } else {
        std::cout << "\nRun with a galois key file to test:" << std::endl;
        std::cout << "  1. Copy the roots above" << std::endl;
        std::cout << "  2. Run Go export tool with those roots" << std::endl;
        std::cout << "  3. ./bridge_with_roots galois_keys.bin" << std::endl;
    }

    return 0;
}
