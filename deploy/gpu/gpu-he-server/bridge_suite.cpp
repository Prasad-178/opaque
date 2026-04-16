// bridge_suite.cpp — layered GPU bridge diagnostic. Uses Lattigo-exported
// state (secret key, Galois keys, plaintext + ciphertext coefficient dumps,
// expected dot products) to localise the exact layer at which the Lattigo ↔
// HEonGPU ciphertext bridge starts producing wrong results.
//
// The suite walks the search pipeline one operation at a time and reports
// max slot-error and worst-slot at each layer:
//   T0  native encrypt + decrypt          — tests loaded sk.
//   T1  bridge ct → NTT → decrypt         — tests ciphertext bridge alone.
//   T2  bridge pt → NTT → decode          — tests plaintext bridge alone.
//   T3  native ct × bridge pt             — tests multiply_plain with bridged pt.
//   T4  bridge ct × native pt             — tests multiply_plain with bridged ct.
//   T5  bridge ct × bridge pt             — full bridge at the multiply step.
//   T6  T5 + rescale_inplace              — tests rescale on bridged ct.
//   T7  T6 + rotate_rows by 1             — tests Galois rotation on bridged ct.
//   T8  T6 + full rotation loop + add     — matches the server's batched dot product.
//
// Build from the gpu-he-server CMakeLists.txt target. Run:
//   ./bridge_suite /tmp/bridge_sk.bin /tmp/bridge_galois.bin /tmp/bridge_test.bin
//
// Privacy note: the diagnostic loads a test-only secret key exported via
// cmd/bridge-export. The same file is not used in production.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <heongpu/heongpu.hpp>
#include <gpuntt/ntt_merge/ntt.cuh>

using namespace heongpu;
constexpr auto S = Scheme::CKKS;

namespace {

constexpr uint32_t kMagic = 0x42524745;

struct TestCase {
    std::vector<double> q;
    std::vector<double> c;
    std::vector<double> expected; // batched dot product per centroid
    double ct_scale;
    double pt_scale;
    // Lattigo coefficient-domain data. Layout: poly_major, [Qsize × N] each.
    std::vector<uint64_t> c0_coefs;
    std::vector<uint64_t> c1_coefs;
    std::vector<uint64_t> pt_coefs;
};

uint32_t read_u32(std::ifstream& is) {
    uint32_t v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
uint64_t read_u64(std::ifstream& is) {
    uint64_t v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
double read_f64(std::ifstream& is) {
    double v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

double max_abs_error(const std::vector<double>& a, const std::vector<double>& b,
                     int& worst_slot) {
    double err = 0.0;
    worst_slot = 0;
    int n = std::min(a.size(), b.size());
    for (int i = 0; i < n; i++) {
        double d = std::abs(a[i] - b[i]);
        if (d > err) {
            err = d;
            worst_slot = i;
        }
    }
    return err;
}

// Report helper: "PASS" when err < 1e-3, else "FAIL".
void report(const std::string& name, double err, int worst,
            double got, double want) {
    const bool pass = err < 1e-3;
    std::cout << "  " << (pass ? "PASS " : "FAIL ") << name
              << "  max_err=" << std::scientific << std::setprecision(3) << err
              << "  worst_slot=" << worst
              << "  (got=" << std::fixed << std::setprecision(6) << got
              << " want=" << want << ")\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " bridge_sk.bin bridge_galois.bin bridge_test.bin\n";
        return 1;
    }

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

    const int N = 16384;
    const int log_n = 14;
    const int slot_count = N / 2;
    const int Q_size = (int)q.size();
    const int P_size = (int)p.size();

    // --- Load SK -----------------------------------------------------------
    std::ifstream sk_is(argv[1], std::ios::binary);
    if (!sk_is) { std::cerr << "cannot open sk\n"; return 2; }
    size_t sk_size = (size_t)(Q_size + P_size) * N;
    std::vector<uint64_t> sk_data(sk_size);
    sk_is.read(reinterpret_cast<char*>(sk_data.data()), sk_size * sizeof(uint64_t));

    HEKeyGenerator<S> keygen(ctx);
    Secretkey<S> sk(ctx);
    keygen.generate_secret_key(sk); // allocate; we overwrite data below.
    sk.store_in_device();
    cudaMemcpy(sk.data(), sk_data.data(), sk_size * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    Publickey<S> pk(ctx);
    keygen.generate_public_key(pk, sk);

    HEEncoder<S> encoder(ctx);
    HEEncryptor<S> encryptor(ctx, pk);
    HEDecryptor<S> decryptor(ctx, sk);
    HEArithmeticOperator<S> ops(ctx, encoder);

    // --- Load Galois keys --------------------------------------------------
    std::ifstream gk_is(argv[2], std::ios::binary);
    if (!gk_is) { std::cerr << "cannot open gk\n"; return 3; }
    Galoiskey<S> gk;
    gk.set_context(ctx);
    gk.load(gk_is);
    std::cout << "Loaded sk + galois keys\n";

    // --- Load tests --------------------------------------------------------
    std::ifstream ts(argv[3], std::ios::binary);
    if (!ts) { std::cerr << "cannot open tests\n"; return 4; }
    if (read_u32(ts) != kMagic) { std::cerr << "bad magic\n"; return 5; }
    (void)read_u32(ts); // version
    uint32_t logN_got = read_u32(ts);
    uint32_t Q_got = read_u32(ts);
    uint32_t sc_got = read_u32(ts);
    uint32_t dim_got = read_u32(ts);
    uint32_t cent_pack = read_u32(ts);
    uint32_t n_tests = read_u32(ts);
    if ((int)logN_got != log_n || (int)Q_got != Q_size || (int)sc_got != slot_count) {
        std::cerr << "header mismatch\n"; return 6;
    }
    const int dim = dim_got;
    const int packs = cent_pack;

    std::vector<TestCase> tests(n_tests);
    for (auto& t : tests) {
        t.q.resize(slot_count);
        t.c.resize(slot_count);
        t.expected.resize(packs);
        ts.read(reinterpret_cast<char*>(t.q.data()), slot_count * 8);
        ts.read(reinterpret_cast<char*>(t.c.data()), slot_count * 8);
        ts.read(reinterpret_cast<char*>(t.expected.data()), packs * 8);
        t.ct_scale = std::bit_cast<double>(read_u64(ts));
        t.pt_scale = std::bit_cast<double>(read_u64(ts));
        t.c0_coefs.resize((size_t)Q_size * N);
        t.c1_coefs.resize((size_t)Q_size * N);
        t.pt_coefs.resize((size_t)Q_size * N);
        ts.read(reinterpret_cast<char*>(t.c0_coefs.data()), t.c0_coefs.size() * 8);
        ts.read(reinterpret_cast<char*>(t.c1_coefs.data()), t.c1_coefs.size() * 8);
        ts.read(reinterpret_cast<char*>(t.pt_coefs.data()), t.pt_coefs.size() * 8);
    }
    std::cout << "Loaded " << n_tests << " test cases (dim=" << dim
              << ", packs=" << packs << ")\n";

    // Shared NTT configs (mirrored from operator.cu).
    gpuntt::ntt_rns_configuration<uint64_t> cfg_ntt = {
        .n_power = log_n,
        .ntt_type = gpuntt::FORWARD,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = cudaStreamDefault,
    };
    gpuntt::ntt_rns_configuration<uint64_t> cfg_intt = {
        .n_power = log_n,
        .ntt_type = gpuntt::INVERSE,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = ctx->get_n_inverse().data(),
        .stream = cudaStreamDefault,
    };

    // Small helper: load Lattigo c0/c1 coefs into a fresh Ciphertext and run
    // forward NTT with the (cipher_size * Q_size, Q_size) shape that matches
    // HEonGPU's own ciphertext NTT calls (the fix landed in main.cpp).
    auto bridge_ct = [&](const TestCase& t) {
        Ciphertext<S> ct(ctx);
        ct.store_in_device();
        std::vector<uint64_t> ct_flat((size_t)2 * Q_size * N);
        std::memcpy(ct_flat.data(), t.c0_coefs.data(),
                    (size_t)Q_size * N * sizeof(uint64_t));
        std::memcpy(ct_flat.data() + (size_t)Q_size * N, t.c1_coefs.data(),
                    (size_t)Q_size * N * sizeof(uint64_t));
        cudaMemcpy(ct.data(), ct_flat.data(),
                   ct_flat.size() * sizeof(uint64_t),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        gpuntt::GPU_NTT_Inplace(
            ct.data(), ctx->get_ntt_table().data(),
            ctx->get_modulus().data(), cfg_ntt,
            2 * Q_size, Q_size);
        cudaDeviceSynchronize();
        ct.set_scale(t.ct_scale);
        ct.set_depth(0);
        return ct;
    };

    // Load Lattigo plaintext coefs into a fresh Plaintext + apply HEonGPU NTT.
    auto bridge_pt = [&](const TestCase& t) {
        Plaintext<S> pt(ctx);
        encoder.encode(pt, std::vector<double>(slot_count, 0.0), t.pt_scale);
        cudaMemcpy(pt.data(), t.pt_coefs.data(),
                   (size_t)Q_size * N * sizeof(uint64_t),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        gpuntt::GPU_NTT_Inplace(
            pt.data(), ctx->get_ntt_table().data(),
            ctx->get_modulus().data(), cfg_ntt,
            Q_size, Q_size);
        cudaDeviceSynchronize();
        return pt;
    };

    int n_pass = 0, n_total = 0;
    auto tally = [&](double err) {
        n_total++;
        if (err < 1e-3) n_pass++;
    };

    for (int i = 0; i < (int)n_tests; i++) {
        const auto& t = tests[i];
        std::cout << "\n================ test " << i << " ================\n";

        // T0 — native encrypt with loaded sk, decrypt, compare with q.
        {
            Plaintext<S> pt(ctx);
            encoder.encode(pt, t.q, t.ct_scale);
            Ciphertext<S> ct(ctx);
            encryptor.encrypt(ct, pt);

            Plaintext<S> ptd(ctx);
            decryptor.decrypt(ptd, ct);
            std::vector<double> got(slot_count);
            encoder.decode(got, ptd);
            int w; double e = max_abs_error(got, t.q, w);
            report("T0 native encrypt+decrypt", e, w, got[w], t.q[w]); tally(e);
        }

        // T1 — bridge ct, decrypt, compare with q.
        {
            Ciphertext<S> ct = bridge_ct(t);
            Plaintext<S> ptd(ctx);
            decryptor.decrypt(ptd, ct);
            std::vector<double> got(slot_count);
            encoder.decode(got, ptd);
            int w; double e = max_abs_error(got, t.q, w);
            report("T1 bridge ct + decrypt", e, w, got[w], t.q[w]); tally(e);
        }

        // T2 — bridge pt, decode, compare with c.
        {
            Plaintext<S> pt = bridge_pt(t);
            std::vector<double> got(slot_count);
            encoder.decode(got, pt);
            int w; double e = max_abs_error(got, t.c, w);
            report("T2 bridge pt + decode", e, w, got[w], t.c[w]); tally(e);
        }

        // T3 — native ct × bridge pt, decrypt, compare with q ⊙ c.
        std::vector<double> qc(slot_count);
        for (int j = 0; j < slot_count; j++) qc[j] = t.q[j] * t.c[j];
        {
            Plaintext<S> ptq(ctx);
            encoder.encode(ptq, t.q, t.ct_scale);
            Ciphertext<S> ct(ctx);
            encryptor.encrypt(ct, ptq);
            Plaintext<S> pt = bridge_pt(t);

            Ciphertext<S> res(ctx);
            ops.multiply_plain(ct, pt, res);
            Plaintext<S> ptd(ctx);
            decryptor.decrypt(ptd, res);
            std::vector<double> got(slot_count);
            encoder.decode(got, ptd);
            int w; double e = max_abs_error(got, qc, w);
            report("T3 native ct × bridge pt", e, w, got[w], qc[w]); tally(e);
        }

        // T4 — bridge ct × native pt.
        {
            Plaintext<S> pt(ctx);
            encoder.encode(pt, t.c, t.pt_scale);
            Ciphertext<S> ct = bridge_ct(t);

            Ciphertext<S> res(ctx);
            ops.multiply_plain(ct, pt, res);
            Plaintext<S> ptd(ctx);
            decryptor.decrypt(ptd, res);
            std::vector<double> got(slot_count);
            encoder.decode(got, ptd);
            int w; double e = max_abs_error(got, qc, w);
            report("T4 bridge ct × native pt", e, w, got[w], qc[w]); tally(e);
        }

        // T5 — bridge both.
        Ciphertext<S> t5_result(ctx);
        {
            Ciphertext<S> ct = bridge_ct(t);
            Plaintext<S> pt = bridge_pt(t);
            ops.multiply_plain(ct, pt, t5_result);
            Plaintext<S> ptd(ctx);
            decryptor.decrypt(ptd, t5_result);
            std::vector<double> got(slot_count);
            encoder.decode(got, ptd);
            int w; double e = max_abs_error(got, qc, w);
            report("T5 bridge ct × bridge pt", e, w, got[w], qc[w]); tally(e);
        }

        // T6 — T5 + rescale.
        Ciphertext<S> t6_result(t5_result);
        {
            ops.rescale_inplace(t6_result);
            Plaintext<S> ptd(ctx);
            decryptor.decrypt(ptd, t6_result);
            std::vector<double> got(slot_count);
            encoder.decode(got, ptd);
            int w; double e = max_abs_error(got, qc, w);
            report("T6 T5 + rescale", e, w, got[w], qc[w]); tally(e);
        }

        // T7 — T6 + rotate by 1.
        {
            Ciphertext<S> cr(t6_result);
            ops.rotate_rows_inplace(cr, gk, 1);
            Plaintext<S> ptd(ctx);
            decryptor.decrypt(ptd, cr);
            std::vector<double> got(slot_count);
            encoder.decode(got, ptd);
            std::vector<double> want(slot_count);
            for (int j = 0; j < slot_count; j++) {
                want[j] = qc[(j + 1) % slot_count];
            }
            int w; double e = max_abs_error(got, want, w);
            report("T7 T6 + rotate_rows(1)", e, w, got[w], want[w]); tally(e);
        }

        // T8 — full dot product: rotate by 1,2,...,dim/2 and add. Slot 0 of
        // each dim-sized segment should equal the expected dot product.
        {
            Ciphertext<S> acc(t6_result);
            for (int stride = 1; stride < dim; stride *= 2) {
                Ciphertext<S> rot(acc);
                ops.rotate_rows_inplace(rot, gk, stride);
                ops.add_inplace(acc, rot);
            }
            Plaintext<S> ptd(ctx);
            decryptor.decrypt(ptd, acc);
            std::vector<double> got(slot_count);
            encoder.decode(got, ptd);
            // Check slot p*dim for each centroid pack p.
            double max_e = 0; int worst_p = 0;
            for (int p = 0; p < packs; p++) {
                double e = std::abs(got[p * dim] - t.expected[p]);
                if (e > max_e) { max_e = e; worst_p = p; }
            }
            bool pass = max_e < 1e-2;
            std::cout << "  " << (pass ? "PASS " : "FAIL ")
                      << "T8 full dot-product loop  max_err="
                      << std::scientific << std::setprecision(3) << max_e
                      << "  worst_pack=" << worst_p
                      << "  (got=" << std::fixed << std::setprecision(6)
                      << got[worst_p * dim]
                      << " want=" << t.expected[worst_p] << ")\n";
            n_total++;
            if (pass) n_pass++;

            // T9 — simulate production return path: INTT the result, copy out
            // to host, copy back in, NTT, decrypt. This models server → Go →
            // Lattigo NTT → decrypt, with the Go NTT substituted by a second
            // HEonGPU NTT (since both libraries agree on the plaintext
            // polynomial, per T2/T5). A FAIL here reveals a bug in the result
            // INTT / cudaMemcpy / NTT cycle used by main.cpp.
            {
                int depth = acc.depth();
                int res_levels = Q_size - depth;
                int res_polys = acc.size();
                size_t res_flat = (size_t)res_polys * res_levels * N;

                // INTT in place.
                gpuntt::GPU_INTT_Inplace(
                    acc.data(), ctx->get_intt_table().data(),
                    ctx->get_modulus().data(), cfg_intt,
                    res_polys * res_levels, res_levels);
                cudaDeviceSynchronize();

                // Round-trip through host (mirrors gRPC transport).
                std::vector<uint64_t> host_buf(res_flat);
                cudaMemcpy(host_buf.data(), acc.data(),
                           res_flat * sizeof(uint64_t),
                           cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                cudaMemcpy(acc.data(), host_buf.data(),
                           res_flat * sizeof(uint64_t),
                           cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();

                // Re-apply NTT, then decrypt.
                gpuntt::GPU_NTT_Inplace(
                    acc.data(), ctx->get_ntt_table().data(),
                    ctx->get_modulus().data(), cfg_ntt,
                    res_polys * res_levels, res_levels);
                cudaDeviceSynchronize();

                Plaintext<S> ptd2(ctx);
                decryptor.decrypt(ptd2, acc);
                std::vector<double> got2(slot_count);
                encoder.decode(got2, ptd2);
                double max_e2 = 0; int worst2 = 0;
                for (int p = 0; p < packs; p++) {
                    double e = std::abs(got2[p * dim] - t.expected[p]);
                    if (e > max_e2) { max_e2 = e; worst2 = p; }
                }
                bool p2 = max_e2 < 1e-2;
                std::cout << "  " << (p2 ? "PASS " : "FAIL ")
                          << "T9 result INTT + host round-trip + NTT + decrypt"
                          << "  max_err="
                          << std::scientific << std::setprecision(3) << max_e2
                          << "  worst_pack=" << worst2
                          << "  (got=" << std::fixed << std::setprecision(6)
                          << got2[worst2 * dim]
                          << " want=" << t.expected[worst2] << ")\n";
                n_total++;
                if (p2) n_pass++;
            }
        }
    }

    std::cout << "\n============== SUMMARY: " << n_pass << " / " << n_total
              << " tests PASSED ==============\n";
    return n_pass == n_total ? 0 : 10;
}
