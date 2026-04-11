// ntt_key_roundtrip.cpp — Definitive test: can we INTT+NTT a native Galois key
// and still get correct rotation results?
//
// 1. Generate a Galois key natively (correct by construction)
// 2. Test rotation works → should pass
// 3. Apply INTT to key data (convert to coefficient domain)
// 4. Apply NTT back (convert back to NTT domain)
// 5. Test rotation again → if this passes, we know the exact NTT params
//
// This isolates whether the NTT/INTT round-trip preserves key correctness.

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <heongpu/heongpu.hpp>

using namespace heongpu;
constexpr auto S = Scheme::CKKS;

int main() {
    std::cout << "=== Definitive NTT Key Round-Trip Test ===" << std::endl;

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

    int N = 16384;
    int n_power = 14;
    int Q_size = q.size();
    int QP_size = Q_size + p.size();

    HEKeyGenerator<S> keygen(context);
    Secretkey<S> sk(context);
    keygen.generate_secret_key(sk);
    Publickey<S> pk(context);
    keygen.generate_public_key(pk, sk);

    // Generate Galois key for rotation by 1
    std::vector<int> shifts = {1};
    Galoiskey<S> gk(context, shifts);
    keygen.generate_galois_key(gk, sk);

    HEEncoder<S> encoder(context);
    HEEncryptor<S> encryptor(context, pk);
    HEDecryptor<S> decryptor(context, sk);
    HEArithmeticOperator<S> ops(context, encoder);

    // Encrypt test vector
    double scale = pow(2.0, 45);
    int slot_count = N / 2;
    std::vector<double> msg(slot_count, 0.0);
    msg[0] = 10.0; msg[1] = 20.0; msg[2] = 30.0; msg[3] = 40.0;

    Plaintext<S> pt(context);
    encoder.encode(pt, msg, scale);
    Ciphertext<S> ct(context);
    encryptor.encrypt(ct, pt);

    // Test 1: Rotation with original (native) key
    {
        Ciphertext<S> ct_rot(ct);
        ops.rotate_rows_inplace(ct_rot, gk, 1);
        Plaintext<S> pt_dec(context);
        decryptor.decrypt(pt_dec, ct_rot);
        std::vector<double> decoded(slot_count);
        encoder.decode(decoded, pt_dec);
        std::cout << "\n1) NATIVE key rotation: ["
                  << std::fixed << std::setprecision(2)
                  << decoded[0] << ", " << decoded[1] << ", " << decoded[2] << ", " << decoded[3] << "]"
                  << std::endl;
        bool ok = (std::abs(decoded[0] - 20.0) < 0.5);
        std::cout << "   " << (ok ? "PASS" : "FAIL") << std::endl;
    }

    // Get the galois element for rotation by 1
    // From the galois_elt map: step 1 → element 5
    int galois_elt = 5;

    // Get key data pointer (on GPU)
    Data64* key_data = gk.data(galois_elt);
    int galoiskey_size = 2 * Q_size * QP_size * N;
    std::cout << "\n   Key data size: " << galoiskey_size << " uint64" << std::endl;

    // Now try different NTT parameter combinations for INTT + NTT round-trip
    auto* ntt_table = context->get_ntt_table().data();
    auto* intt_table = context->get_intt_table().data();
    auto* moduli = context->get_modulus().data();

    // Try various batch_size / mod_count combinations
    struct TestCase {
        int batch_size;
        int mod_count;
        const char* desc;
    };

    std::vector<TestCase> tests = {
        {QP_size, QP_size, "QP x QP (like encryptor)"},
        {2 * Q_size, QP_size, "2*Q x QP"},
        {2 * Q_size * QP_size, QP_size, "2*Q*QP x QP"},
        {galoiskey_size / N, 1, "total_polys x 1"},
        {Q_size, QP_size, "Q x QP"},
    };

    for (auto& tc : tests) {
        // Make a copy of the key data to restore after test
        DeviceVector<Data64> backup(galoiskey_size);
        cudaMemcpy(backup.data(), key_data, galoiskey_size * sizeof(Data64), cudaMemcpyDeviceToDevice);

        // Apply INTT
        gpuntt::ntt_rns_configuration<uint64_t> cfg_inv = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context->get_n_inverse().data(),
            .stream = cudaStreamDefault
        };

        gpuntt::GPU_INTT_Inplace(key_data, intt_table, moduli, cfg_inv, tc.batch_size, tc.mod_count);
        cudaDeviceSynchronize();

        // Apply NTT
        gpuntt::ntt_rns_configuration<uint64_t> cfg_fwd = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = cudaStreamDefault
        };

        gpuntt::GPU_NTT_Inplace(key_data, ntt_table, moduli, cfg_fwd, tc.batch_size, tc.mod_count);
        cudaDeviceSynchronize();

        // Test rotation
        Ciphertext<S> ct_rot(ct);
        ops.rotate_rows_inplace(ct_rot, gk, 1);
        Plaintext<S> pt_dec(context);
        decryptor.decrypt(pt_dec, ct_rot);
        std::vector<double> decoded(slot_count);
        encoder.decode(decoded, pt_dec);

        bool ok = (std::abs(decoded[0] - 20.0) < 0.5 && std::abs(decoded[1] - 30.0) < 0.5);
        std::cout << "\n2) INTT+NTT batch=" << tc.batch_size << " mod=" << tc.mod_count
                  << " (" << tc.desc << "): ["
                  << std::fixed << std::setprecision(2)
                  << decoded[0] << ", " << decoded[1] << ", " << decoded[2] << ", " << decoded[3] << "]"
                  << "  " << (ok ? "PASS ✓" : "FAIL ✗") << std::endl;

        // Restore original key data
        cudaMemcpy(key_data, backup.data(), galoiskey_size * sizeof(Data64), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }

    return 0;
}
