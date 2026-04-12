// bridge_final.cpp — Final bridge test. Keys are already in HEonGPU's NTT domain
// (converted in Go). Just load and test rotation — NO NTT needed on GPU side.
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
    if (argc < 2) {
        std::cout << "Usage: ./bridge_final <galois_keys.bin>" << std::endl;
        return 1;
    }

    auto ctx = GenHEContext<S>(sec_level_type::none);
    ctx->set_poly_modulus_degree(16384);
    std::vector<uint64_t> q = {1152921504606748673ULL,35184372121601ULL,35184372744193ULL,
        35184373006337ULL,35184371138561ULL,35184370941953ULL,35184370352129ULL,35184373989377ULL};
    std::vector<uint64_t> p = {2305843009211662337ULL};
    ctx->set_coeff_modulus_values(q, p);
    ctx->generate();

    // Load keys — already in HEonGPU's NTT domain
    std::ifstream f(argv[1], std::ios::binary);
    f.seekg(0, std::ios::end);
    std::cout << "File: " << f.tellg() << " bytes" << std::endl;
    f.seekg(0, std::ios::beg);

    Galoiskey<S> gk;
    gk.set_context(ctx);
    gk.load(f);
    f.close();
    std::cout << "Loaded (already NTT domain — no GPU NTT needed)" << std::endl;

    // Generate local keys for encryption/decryption
    HEKeyGenerator<S> keygen(ctx);
    Secretkey<S> sk(ctx);
    keygen.generate_secret_key(sk);
    Publickey<S> pk(ctx);
    keygen.generate_public_key(pk, sk);

    HEEncoder<S> encoder(ctx);
    HEEncryptor<S> encryptor(ctx, pk);
    HEDecryptor<S> decryptor(ctx, sk);
    HEArithmeticOperator<S> ops(ctx, encoder);

    double scale = pow(2.0, 45);
    int slot_count = 16384 / 2;
    std::vector<double> msg(slot_count, 0.0);
    msg[0] = 10.0; msg[1] = 20.0; msg[2] = 30.0; msg[3] = 40.0;

    Plaintext<S> pt(ctx);
    encoder.encode(pt, msg, scale);
    Ciphertext<S> ct(ctx);
    encryptor.encrypt(ct, pt);

    // Test rotation
    Ciphertext<S> ct_rot(ct);
    auto start = std::chrono::high_resolution_clock::now();
    ops.rotate_rows_inplace(ct_rot, gk, 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    Plaintext<S> pd(ctx);
    decryptor.decrypt(pd, ct_rot);
    std::vector<double> dec(slot_count);
    encoder.decode(dec, pd);

    std::cout << "\nOriginal: [10, 20, 30, 40, ...]" << std::endl;
    std::cout << "Rotated:  [" << std::fixed << std::setprecision(2)
              << dec[0] << ", " << dec[1] << ", " << dec[2] << ", " << dec[3] << ", " << dec[4] << "]"
              << std::endl;
    std::cout << "Time: " << us << " us" << std::endl;

    bool correct = (std::abs(dec[0] - 20.0) < 0.5 && std::abs(dec[1] - 30.0) < 0.5 && std::abs(dec[2] - 40.0) < 0.5);

    if (correct) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "  BRIDGE WORKS! LATTIGO -> HEONGPU OK!" << std::endl;
        std::cout << "========================================" << std::endl;

        // Benchmark rotations
        std::cout << "\n=== Rotation Benchmarks ===" << std::endl;
        for (int shift : {1, 2, 4, 8, 16, 32, 64, 128}) {
            auto s = std::chrono::high_resolution_clock::now();
            int iters = 20;
            for (int i = 0; i < iters; i++) {
                Ciphertext<S> tmp(ct);
                ops.rotate_rows_inplace(tmp, gk, shift);
            }
            auto e = std::chrono::high_resolution_clock::now();
            auto avg = std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / iters;
            std::cout << "  Rotate by " << shift << ": " << avg << " us (" << avg/1000.0 << " ms)" << std::endl;
        }
    } else {
        std::cout << "\n✗ BRIDGE FAILED" << std::endl;
        return 1;
    }

    return 0;
}
