// verify_bridge.cpp — Verify coefficient layout between Lattigo and HEonGPU.
// Based on HEonGPU's own 2_basic_ckks.cpp example API.

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <sstream>

#include <heongpu/heongpu.hpp>

using namespace heongpu;
constexpr auto S = Scheme::CKKS;

int main() {
    std::cout << "=== HEonGPU Format Bridge Verification ===" << std::endl;

    // Match Lattigo GPU path: LogN=14, LogQ=[60,45,45,45,45,45,45,45], LogP=[61]
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
    int Q_size = q_primes.size(); // 8
    int Q_prime_size = Q_size + p_primes.size(); // 9

    std::cout << "Ring: " << ring_size << ", Q: " << Q_size
              << " primes, P: " << p_primes.size() << " primes" << std::endl;

    // Generate keys
    HEKeyGenerator<S> keygen(context);
    Secretkey<S> sk(context);
    keygen.generate_secret_key(sk);
    Publickey<S> pk(context);
    keygen.generate_public_key(pk, sk);
    Relinkey<S> rlk(context);
    keygen.generate_relin_key(rlk, sk);

    std::vector<int> shifts;
    for (int i = 0; i < 14; i++) shifts.push_back(1 << i);
    Galoiskey<S> gk(context, shifts);
    keygen.generate_galois_key(gk, sk);

    // Init HE operators
    HEEncoder<S> encoder(context);
    HEEncryptor<S> encryptor(context, pk);
    HEDecryptor<S> decryptor(context, sk);
    HEArithmeticOperator<S> ops(context, encoder);

    double scale = pow(2.0, 45);

    // Encode + encrypt a simple test vector
    int slot_count = ring_size / 2;
    std::vector<double> msg(slot_count, 0.0);
    msg[0] = 1.0; msg[1] = 2.0; msg[2] = 3.0; msg[3] = 4.0;

    Plaintext<S> pt(context);
    encoder.encode(pt, msg, scale);

    Ciphertext<S> ct(context);
    encryptor.encrypt(ct, pt);

    // Export ciphertext to host
    ct.store_in_host();
    std::vector<uint64_t> ct_data;
    ct.get_data(ct_data);

    std::cout << "\n=== Ciphertext Layout ===" << std::endl;
    std::cout << "Total uint64: " << ct_data.size() << std::endl;
    std::cout << "Expected:     " << 2 * Q_size * ring_size
              << " (2 polys × " << Q_size << " levels × " << ring_size << ")" << std::endl;
    std::cout << "Ring size:    " << ct.ring_size() << std::endl;
    std::cout << "Coeff mod cnt:" << ct.coeff_modulus_count() << std::endl;
    std::cout << "Cipher size:  " << ct.size() << std::endl;
    std::cout << "Depth:        " << ct.depth() << std::endl;
    std::cout << "Scale:        " << ct.scale() << std::endl;
    std::cout << "In NTT:       " << ct.in_ntt_domain() << std::endl;

    std::cout << "\nc0[0..3]: ";
    for (int i = 0; i < 4 && i < (int)ct_data.size(); i++)
        std::cout << ct_data[i] << " ";
    std::cout << std::endl;

    int c1_off = (ct_data.size() / 2);
    std::cout << "c1[0..3]: ";
    for (int i = 0; i < 4; i++)
        std::cout << ct_data[c1_off + i] << " ";
    std::cout << std::endl;

    // Verify decryption
    Plaintext<S> pt_dec(context);
    decryptor.decrypt(pt_dec, ct);
    std::vector<double> decoded(slot_count);
    encoder.decode(decoded, pt_dec);

    std::cout << "\n=== Decryption Check ===" << std::endl;
    std::cout << "Decoded[0..3]: ";
    for (int i = 0; i < 4; i++)
        std::cout << std::fixed << std::setprecision(6) << decoded[i] << " ";
    std::cout << std::endl;

    // Verify rotation works
    Ciphertext<S> ct2(ct);
    ops.rotate_rows_inplace(ct2, gk, 1);
    Plaintext<S> pt_rot(context);
    decryptor.decrypt(pt_rot, ct2);
    std::vector<double> dec_rot(slot_count);
    encoder.decode(dec_rot, pt_rot);

    std::cout << "After rot 1:   ";
    for (int i = 0; i < 4; i++)
        std::cout << std::fixed << std::setprecision(6) << dec_rot[i] << " ";
    std::cout << std::endl;

    // Galois key serialization info
    gk.store_in_host();
    std::stringstream gk_ss;
    gk.save(gk_ss);
    std::cout << "\n=== Galois Key ===" << std::endl;
    std::cout << "Serialized: " << gk_ss.str().size() << " bytes = "
              << gk_ss.str().size() / (1024*1024) << " MB" << std::endl;

    // Relin key info
    rlk.store_in_host();
    std::stringstream rlk_ss;
    rlk.save(rlk_ss);
    std::cout << "Relin key:  " << rlk_ss.str().size() << " bytes = "
              << rlk_ss.str().size() / (1024*1024) << " MB" << std::endl;

    // Secret key size (via serialization)
    sk.store_in_host();
    std::stringstream sk_ss;
    sk.save(sk_ss);
    std::cout << "\n=== Secret Key ===" << std::endl;
    std::cout << "Serialized: " << sk_ss.str().size() << " bytes" << std::endl;

    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << "Ciphertext: " << ct_data.size() << " uint64, layout: "
              << "[c0_levels | c1_levels], each level " << ring_size << " coeffs" << std::endl;
    std::cout << "If Lattigo GPU path (LogP=[61]) produces same ct_data.size()=" << ct_data.size()
              << " and same coefficient ordering, the bridge works." << std::endl;

    return 0;
}
