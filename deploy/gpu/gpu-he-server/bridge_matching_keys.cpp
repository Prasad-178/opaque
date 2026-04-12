// bridge_matching_keys.cpp — Bridge test with MATCHING secret key.
// Loads both Galois keys (from Lattigo) and secret key (from Lattigo).
// Uses the SAME secret key for encryption, rotation, and decryption.
//
// Usage: ./bridge_matching_keys <galois_keys.bin> <secret_key.bin>

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
        std::cout << "Usage: ./bridge_matching_keys <galois_keys.bin> <secret_key.bin>" << std::endl;
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

    // Load Galois keys (already in HEonGPU NTT domain)
    std::cout << "Loading Galois keys..." << std::endl;
    std::ifstream gk_file(argv[1], std::ios::binary);
    Galoiskey<S> gk;
    gk.set_context(ctx);
    gk.load(gk_file);
    gk_file.close();
    std::cout << "Galois keys loaded" << std::endl;

    // Load secret key raw coefficients (already NTT-domain converted)
    std::cout << "Loading secret key..." << std::endl;
    std::ifstream sk_file(argv[2], std::ios::binary);
    std::vector<uint64_t> sk_data(QP * N);
    sk_file.read(reinterpret_cast<char*>(sk_data.data()), QP * N * sizeof(uint64_t));
    sk_file.close();
    std::cout << "Secret key loaded (" << sk_data.size() << " uint64)" << std::endl;

    // Construct HEonGPU Secretkey from raw data
    Secretkey<S> sk(ctx);
    // Upload raw data to GPU via the secret key's internal storage
    // The Secretkey stores data on GPU as QP*N uint64 values in NTT domain
    DeviceVector<uint64_t> d_sk(sk_data);
    // Need to copy to the secret key's internal buffer
    // Since Secretkey doesn't have a public set_data(), we need to use
    // the keygen to generate a key and then overwrite its data
    HEKeyGenerator<S> keygen(ctx);
    keygen.generate_secret_key(sk);
    sk.store_in_device();

    // Overwrite the secret key data with our Lattigo-converted data
    cudaMemcpy(sk.data(), d_sk.data(), QP * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    std::cout << "Secret key data overwritten with Lattigo key" << std::endl;

    // Generate public key from the Lattigo secret key
    Publickey<S> pk(ctx);
    keygen.generate_public_key(pk, sk);
    std::cout << "Public key generated from Lattigo SK" << std::endl;

    // Setup operators
    HEEncoder<S> encoder(ctx);
    HEEncryptor<S> encryptor(ctx, pk);
    HEDecryptor<S> decryptor(ctx, sk);
    HEArithmeticOperator<S> ops(ctx, encoder);

    // Encrypt test vector
    double scale = pow(2.0, 45);
    int slot_count = N / 2;
    std::vector<double> msg(slot_count, 0.0);
    msg[0] = 10.0; msg[1] = 20.0; msg[2] = 30.0; msg[3] = 40.0;

    Plaintext<S> pt(ctx);
    encoder.encode(pt, msg, scale);
    Ciphertext<S> ct(ctx);
    encryptor.encrypt(ct, pt);

    // Verify encryption works (decrypt without rotation)
    {
        Plaintext<S> pd(ctx);
        decryptor.decrypt(pd, ct);
        std::vector<double> dec(slot_count);
        encoder.decode(dec, pd);
        std::cout << "\nDecrypt (no rotation): ["
                  << std::fixed << std::setprecision(2)
                  << dec[0] << ", " << dec[1] << ", " << dec[2] << ", " << dec[3] << "]" << std::endl;
        bool dec_ok = (std::abs(dec[0] - 10.0) < 0.5);
        std::cout << "Decryption: " << (dec_ok ? "OK" : "FAIL") << std::endl;
        if (!dec_ok) {
            std::cout << "Secret key reconstruction failed!" << std::endl;
            return 1;
        }
    }

    // Test rotation with Lattigo Galois keys
    Ciphertext<S> ct_rot(ct);
    auto start = std::chrono::high_resolution_clock::now();
    ops.rotate_rows_inplace(ct_rot, gk, 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    Plaintext<S> pd(ctx);
    decryptor.decrypt(pd, ct_rot);
    std::vector<double> dec(slot_count);
    encoder.decode(dec, pd);

    std::cout << "\nRotated: [" << std::fixed << std::setprecision(2)
              << dec[0] << ", " << dec[1] << ", " << dec[2] << ", " << dec[3] << ", " << dec[4] << "]"
              << std::endl;
    std::cout << "Time: " << us << " us" << std::endl;

    bool correct = (std::abs(dec[0] - 20.0) < 0.5 && std::abs(dec[1] - 30.0) < 0.5 && std::abs(dec[2] - 40.0) < 0.5);

    if (correct) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "  BRIDGE WORKS! LATTIGO -> HEONGPU OK!" << std::endl;
        std::cout << "========================================" << std::endl;

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
        std::cout << "\nBRIDGE FAILED" << std::endl;
        return 1;
    }

    return 0;
}
