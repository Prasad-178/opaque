// verify_bridge.cpp — Verify coefficient compatibility between Lattigo and HEonGPU.
//
// This program:
// 1. Creates an HEonGPU CKKS context with the exact same parameters as Lattigo GPU path
// 2. Generates a secret key
// 3. Exports the raw secret key coefficients (for use in Lattigo)
// 4. Generates evaluation keys (Galois + relin)
// 5. Exports the raw eval key layout (decomposition structure, sizes)
// 6. Encrypts a known plaintext and exports raw ciphertext coefficients
// 7. Prints everything for comparison with Lattigo's output
//
// Build (on GPU instance):
//   cd /home/ubuntu/HEonGPU/build
//   g++ -std=c++17 -I../src/include -I_deps/gpu-ntt-src/src/include \
//       -I_deps/rmm-src/include -I_deps/spdlog-src/include \
//       -I_deps/rapids_logger-src/include \
//       -L./src -lheongpu -lcudart -lgmp -lntl \
//       -o verify_bridge /home/ubuntu/opaque/deploy/gpu/gpu-he-server/verify_bridge.cpp
//
// Run: ./verify_bridge

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <sstream>

#include <heongpu/heongpu.cuh>

using namespace heongpu;

int main() {
    std::cout << "=== HEonGPU Format Bridge Verification ===" << std::endl;

    // Match Lattigo GPU path parameters exactly:
    // LogN=14, LogQ=[60,45,45,45,45,45,45,45], LogP=[61]
    constexpr auto Scheme = heongpu::Scheme::CKKS;

    auto context = GenHEContext<Scheme>(sec_level_type::none);
    context->set_poly_modulus_degree(16384); // 2^14

    // Use exact same Q primes as Lattigo generates
    std::vector<uint64_t> q_primes = {
        1152921504606748673ULL,  // Q[0]: 60-bit
        35184372121601ULL,       // Q[1]: 46-bit
        35184372744193ULL,       // Q[2]: 46-bit
        35184373006337ULL,       // Q[3]: 46-bit
        35184371138561ULL,       // Q[4]: 45-bit
        35184370941953ULL,       // Q[5]: 45-bit
        35184370352129ULL,       // Q[6]: 45-bit
        35184373989377ULL        // Q[7]: 46-bit
    };

    // Use exact same P prime as Lattigo GPU path
    std::vector<uint64_t> p_primes = {
        2305843009211662337ULL   // P[0]: 61-bit
    };

    context->set_coeff_modulus_values(q_primes, p_primes);
    context->generate();

    int ring_size = context->get_poly_modulus_degree();
    int Q_size = q_primes.size();
    int Q_prime_size = Q_size + p_primes.size();

    std::cout << "\n=== Context ===" << std::endl;
    std::cout << "Ring size:      " << ring_size << std::endl;
    std::cout << "Q primes:       " << Q_size << std::endl;
    std::cout << "P primes:       " << p_primes.size() << std::endl;
    std::cout << "Q+P total:      " << Q_prime_size << std::endl;

    // Generate keys
    HEKeyGenerator<Scheme> keygen(context);
    Secretkey<Scheme> sk(context);
    keygen.generate_secret_key(sk);

    Publickey<Scheme> pk(context);
    keygen.generate_public_key(pk, sk);

    Relinkey<Scheme> rlk(context);
    keygen.generate_relin_key(rlk, sk);

    // Generate Galois keys for rotations by powers of 2
    // Match Lattigo's galoisElements: rotation by 1,2,4,...,8192
    std::vector<int> shifts;
    for (int i = 0; i < 14; i++) {
        shifts.push_back(1 << i);
    }
    Galoiskey<Scheme> gk(context, shifts);
    keygen.generate_galois_key(gk, sk);

    std::cout << "\n=== Key Sizes ===" << std::endl;

    // Export secret key to host and print first few coefficients
    sk.store_in_host();
    std::vector<uint64_t> sk_data;
    // Secret key is a single polynomial in the ring
    // Size should be Q_prime_size * ring_size
    sk.get_data(sk_data);
    std::cout << "Secret key size: " << sk_data.size() << " uint64" << std::endl;
    std::cout << "SK first 8 coeffs: ";
    for (int i = 0; i < 8 && i < (int)sk_data.size(); i++) {
        std::cout << sk_data[i] << " ";
    }
    std::cout << std::endl;

    // Export relin key info
    rlk.store_in_host();
    std::cout << "\nRelin key:" << std::endl;
    // Save to stringstream to measure size
    std::stringstream rlk_stream;
    rlk.save(rlk_stream);
    std::string rlk_data = rlk_stream.str();
    std::cout << "  Serialized size: " << rlk_data.size() << " bytes = "
              << rlk_data.size() / (1024*1024) << " MB" << std::endl;

    // Export Galois key info
    gk.store_in_host();
    std::stringstream gk_stream;
    gk.save(gk_stream);
    std::string gk_data = gk_stream.str();
    std::cout << "\nGalois key (all rotations):" << std::endl;
    std::cout << "  Serialized size: " << gk_data.size() << " bytes = "
              << gk_data.size() / (1024*1024) << " MB" << std::endl;

    // Test encryption + dot product
    HEEncoder<Scheme> encoder(context);
    HEEncryptor<Scheme> encryptor(context);
    HEDecryptor<Scheme> decryptor(context);
    HEOperator<Scheme> op(context);

    double scale = pow(2.0, 45); // Match Lattigo's LogDefaultScale=45

    // Create a simple test vector
    int slot_count = ring_size / 2;
    std::vector<double> message(slot_count, 0.0);
    message[0] = 1.0;
    message[1] = 2.0;
    message[2] = 3.0;
    message[3] = 4.0;

    Plaintext<Scheme> pt(context);
    encoder.encode(pt, message, scale);

    Ciphertext<Scheme> ct(context);
    encryptor.encrypt(ct, pt, pk);

    // Export ciphertext to host and print structure
    ct.store_in_host();
    std::vector<uint64_t> ct_data;
    ct.get_data(ct_data);
    std::cout << "\n=== Ciphertext ===" << std::endl;
    std::cout << "Size: " << ct_data.size() << " uint64" << std::endl;
    std::cout << "Expected: " << 2 * Q_size * ring_size << " uint64 "
              << "(2 polys × " << Q_size << " levels × " << ring_size << ")" << std::endl;
    std::cout << "Ring size: " << ct.ring_size() << std::endl;
    std::cout << "Coeff modulus count: " << ct.coeff_modulus_count() << std::endl;
    std::cout << "Cipher size: " << ct.size() << std::endl;
    std::cout << "Depth: " << ct.depth() << std::endl;
    std::cout << "Scale: " << ct.scale() << std::endl;
    std::cout << "In NTT: " << ct.in_ntt_domain() << std::endl;

    // Print first few coefficients of c0 and c1
    std::cout << "c0 first 4 coeffs: ";
    for (int i = 0; i < 4; i++) {
        std::cout << ct_data[i] << " ";
    }
    std::cout << std::endl;

    int c1_offset = Q_size * ring_size;
    if (c1_offset < (int)ct_data.size()) {
        std::cout << "c1 first 4 coeffs: ";
        for (int i = 0; i < 4; i++) {
            std::cout << ct_data[c1_offset + i] << " ";
        }
        std::cout << std::endl;
    }

    // Verify decryption works
    Plaintext<Scheme> pt_dec(context);
    decryptor.decrypt(pt_dec, ct, sk);
    std::vector<double> decoded(slot_count);
    encoder.decode(decoded, pt_dec);

    std::cout << "\n=== Decrypt verification ===" << std::endl;
    std::cout << "Decoded[0..3]: ";
    for (int i = 0; i < 4; i++) {
        std::cout << std::fixed << std::setprecision(4) << decoded[i] << " ";
    }
    std::cout << std::endl;

    // Test a rotation
    Ciphertext<Scheme> ct_rot(context);
    op.rotate_rows(ct_rot, ct, gk, 1);
    Plaintext<Scheme> pt_rot(context);
    decryptor.decrypt(pt_rot, ct_rot, sk);
    std::vector<double> decoded_rot(slot_count);
    encoder.decode(decoded_rot, pt_rot);

    std::cout << "After rotate by 1: ";
    for (int i = 0; i < 4; i++) {
        std::cout << std::fixed << std::setprecision(4) << decoded_rot[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << "Ciphertext layout: " << ct_data.size() << " uint64 = "
              << ct.size() << " polys × " << Q_size << " levels × " << ring_size
              << " coeffs" << std::endl;
    std::cout << "Data ordering: [c0_level0[N] | c0_level1[N] | ... | c1_level0[N] | ...]"
              << std::endl;
    std::cout << "All data in NTT domain: " << ct.in_ntt_domain() << std::endl;
    std::cout << "\nIf Lattigo's raw coefficient extraction produces the same"
              << "\nordering and values for the SAME secret key, the bridge works."
              << std::endl;

    return 0;
}
