// encode_compare.cpp — investigates whether Lattigo's CKKS encoding of a
// vector yields the same polynomial representation as HEonGPU's encoding.
//
// Flow:
//   1. Reads lattigo_dump.bin produced by cmd/encode-compare-go (coefficient
//      domain polynomials, one per prime Q, non-Montgomery).
//   2. For each test vector:
//        a. Encodes the same vector natively with HEonGPU and decodes it to
//           verify HEonGPU's round-trip produces the expected values.
//        b. Loads Lattigo's coefficient-domain polynomial into a HEonGPU
//           plaintext via cudaMemcpy, applies HEonGPU's forward NTT, and
//           decodes — this asks "does HEonGPU see what Lattigo wrote?".
//   3. Reports per-slot max error for both round-trips so we can tell whether
//      the two libraries encode the same vector as the same polynomial.
//
// If (b) recovers the original vector, Lattigo/HEonGPU encodings are
// semantically compatible. If (b) returns garbage while (a) is correct, the
// canonical-embedding convention differs and the bridge needs a conversion.
//
// Usage: ./encode_compare <lattigo_dump.bin> [out_report.txt]

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

constexpr uint32_t kMagic = 0x4F504151;

uint32_t read_u32(std::ifstream& is) {
    uint32_t v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

struct TestVec {
    std::string name;
    std::vector<double> values;
    // coefficient-domain polynomial from Lattigo (Q_size * N)
    std::vector<uint64_t> lattigo_poly;
};

std::vector<double> with_slot(int n, int idx, double v) {
    std::vector<double> out(n, 0.0);
    out[idx] = v;
    return out;
}
std::vector<double> ramp_to(int n, int k) {
    std::vector<double> out(n, 0.0);
    for (int i = 0; i < k && i < n; i++) out[i] = (double)i;
    return out;
}
std::vector<double> constant_vec(int n, double v) {
    return std::vector<double>(n, v);
}
std::vector<double> sine_wave(int n) {
    std::vector<double> out(n);
    for (int i = 0; i < n; i++)
        out[i] = 0.5 * std::sin(2.0 * M_PI * (double)i / (double)n);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const char* lattigo_path = argc > 1 ? argv[1] : "/tmp/lattigo_dump.bin";

    // Must match NewParametersGPU() on Go side.
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
    const double scale = std::pow(2.0, 45);

    std::cout << "HEonGPU / Lattigo encoding compatibility test\n"
              << "  LogN=" << log_n << " N=" << N
              << " slots=" << slot_count
              << " qSize=" << Q_size
              << " scale=2^45\n\n";

    HEEncoder<S> encoder(ctx);

    // Reconstruct the test vectors that Go wrote, in the same order. They
    // must match cmd/encode-compare-go/main.go to line up with the dump.
    std::vector<TestVec> tests = {
        {"slot0_eq_1",      with_slot(slot_count, 0, 1.0), {}},
        {"slot1_eq_1",      with_slot(slot_count, 1, 1.0), {}},
        {"slot2_eq_1",      with_slot(slot_count, 2, 1.0), {}},
        {"slot_last_eq_1",  with_slot(slot_count, slot_count - 1, 1.0), {}},
        {"ramp_0_to_15",    ramp_to(slot_count, 16), {}},
        {"constant_0.5",    constant_vec(slot_count, 0.5), {}},
        {"sine_wave",       sine_wave(slot_count), {}},
    };

    // Load Lattigo dump.
    std::ifstream is(lattigo_path, std::ios::binary);
    if (!is) {
        std::cerr << "Cannot open Lattigo dump " << lattigo_path << "\n";
        return 1;
    }
    uint32_t m = read_u32(is);
    uint32_t ver = read_u32(is);
    uint32_t logN = read_u32(is);
    uint32_t numPrimes = read_u32(is);
    uint32_t numTests = read_u32(is);
    if (m != kMagic || (int)logN != log_n || (int)numPrimes != Q_size) {
        std::cerr << "Dump header mismatch: magic=" << std::hex << m
                  << " ver=" << ver << " logN=" << std::dec << logN
                  << " primes=" << numPrimes << "\n";
        return 2;
    }
    if ((int)numTests != (int)tests.size()) {
        std::cerr << "Test count mismatch: dump=" << numTests
                  << " expected=" << tests.size() << "\n";
        return 3;
    }

    for (auto& t : tests) {
        uint32_t nl = read_u32(is);
        std::string name(nl, '\0');
        is.read(&name[0], nl);
        if (name != t.name) {
            std::cerr << "Test name mismatch: expected " << t.name
                      << " got " << name << "\n";
            return 4;
        }
        t.lattigo_poly.resize((size_t)Q_size * N);
        is.read(reinterpret_cast<char*>(t.lattigo_poly.data()),
                (size_t)Q_size * N * sizeof(uint64_t));
    }

    // NTT config for forward transform on coefficient-domain plaintexts.
    gpuntt::ntt_rns_configuration<uint64_t> cfg_ntt = {
        .n_power = log_n,
        .ntt_type = gpuntt::FORWARD,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = cudaStreamDefault,
    };

    // Error accumulators
    double worst_native_err = 0.0;
    double worst_bridge_err = 0.0;
    std::string worst_bridge_test;

    // INTT config for self-test (verify cudaMemcpy + NTT round-trip).
    gpuntt::ntt_rns_configuration<uint64_t> cfg_intt = {
        .n_power = log_n,
        .ntt_type = gpuntt::INVERSE,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = ctx->get_n_inverse().data(),
        .stream = cudaStreamDefault,
    };

    // ============================================================
    // Self-test: apply HEonGPU INTT then NTT on a native plaintext
    // via cudaMemcpy reconstruction. Verifies the bridge plumbing.
    // ============================================================
    {
        std::cout << "--- Self-test: native encode → INTT (device) → "
                     "cudaMemcpy round-trip → NTT → decode ---\n";
        Plaintext<S> pt_sanity(ctx);
        const auto& v = tests[5].values; // constant_0.5
        encoder.encode(pt_sanity, v, scale);

        // INTT in place → coefficient domain
        gpuntt::GPU_NTT_Inplace(
            pt_sanity.data(),
            ctx->get_intt_table().data(),
            ctx->get_modulus().data(),
            cfg_intt, 1, Q_size);
        cudaDeviceSynchronize();

        // Copy to host, print first 4 of prime 0.
        std::vector<uint64_t> coef_host((size_t)Q_size * N);
        cudaMemcpy(coef_host.data(), pt_sanity.data(),
                   (size_t)Q_size * N * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        std::cout << "  HEonGPU-native coefficient-domain prime Q[0] first 4: "
                  << coef_host[0] << " " << coef_host[1] << " "
                  << coef_host[2] << " " << coef_host[3] << "\n";

        // NTT back
        gpuntt::GPU_NTT_Inplace(
            pt_sanity.data(),
            ctx->get_ntt_table().data(),
            ctx->get_modulus().data(),
            cfg_ntt, 1, Q_size);
        cudaDeviceSynchronize();

        std::vector<double> decoded_rt(slot_count);
        encoder.decode(decoded_rt, pt_sanity);
        double rt_err = 0;
        for (int i = 0; i < slot_count; i++) {
            double e = std::abs(decoded_rt[i] - v[i]);
            if (e > rt_err) rt_err = e;
        }
        std::cout << "  Self-test INTT→NTT max_err = " << std::scientific
                  << std::setprecision(3) << rt_err << "\n\n";
    }

    for (const auto& t : tests) {
        std::cout << "--- Test \"" << t.name << "\" ---\n";

        // (a) Native HEonGPU encode → decode round-trip
        Plaintext<S> pt_native(ctx);
        encoder.encode(pt_native, t.values, scale);
        std::vector<double> decoded_native(slot_count);
        encoder.decode(decoded_native, pt_native);

        double native_err = 0.0;
        for (int i = 0; i < slot_count; i++) {
            double e = std::abs(decoded_native[i] - t.values[i]);
            if (e > native_err) native_err = e;
        }
        std::cout << "  Native HEonGPU encode+decode max_err = "
                  << std::scientific << std::setprecision(3) << native_err << "\n";
        if (native_err > worst_native_err) worst_native_err = native_err;

        // (b) Lattigo polynomial → HEonGPU NTT → HEonGPU decode.
        // Create a fresh plaintext of the right size and inject Lattigo's
        // coefficient-domain polynomial into its device storage.
        Plaintext<S> pt_bridge(ctx);
        // Encode zeros first to get the right device allocation + metadata.
        std::vector<double> zeros(slot_count, 0.0);
        encoder.encode(pt_bridge, zeros, scale);

        cudaMemcpy(pt_bridge.data(), t.lattigo_poly.data(),
                   (size_t)Q_size * N * sizeof(uint64_t),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        // Apply HEonGPU's forward NTT to move coefficients into the NTT
        // domain that HEonGPU's ops expect for a "normal" plaintext.
        gpuntt::GPU_NTT_Inplace(
            pt_bridge.data(),
            ctx->get_ntt_table().data(),
            ctx->get_modulus().data(),
            cfg_ntt,
            1,        // one polynomial per plaintext
            Q_size    // Q_size RNS components
        );
        cudaDeviceSynchronize();

        // Reset the scale metadata so HEonGPU's decoder divides by the right Δ.
        // (The zero-encode set scale_ to 2^45 already, so this should be fine.)

        std::vector<double> decoded_bridge(slot_count);
        encoder.decode(decoded_bridge, pt_bridge);

        double bridge_err = 0.0;
        int first_bad = -1;
        for (int i = 0; i < slot_count; i++) {
            double e = std::abs(decoded_bridge[i] - t.values[i]);
            if (e > bridge_err) {
                bridge_err = e;
                first_bad = i;
            }
        }
        std::cout << "  Lattigo→HEonGPU NTT→decode  max_err = "
                  << std::scientific << std::setprecision(3) << bridge_err
                  << "  worst_slot=" << first_bad
                  << "  (v=" << std::fixed << std::setprecision(4)
                  << t.values[first_bad < 0 ? 0 : first_bad]
                  << "  got=" << decoded_bridge[first_bad < 0 ? 0 : first_bad]
                  << ")\n";
        if (bridge_err > worst_bridge_err) {
            worst_bridge_err = bridge_err;
            worst_bridge_test = t.name;
        }

        // Print first 4 slot values side by side for quick inspection.
        std::cout << "  first 4 slots:  original=["
                  << std::setprecision(4) << std::fixed
                  << t.values[0] << ", " << t.values[1] << ", "
                  << t.values[2] << ", " << t.values[3] << "]\n"
                  << "                  bridge  =["
                  << decoded_bridge[0] << ", " << decoded_bridge[1] << ", "
                  << decoded_bridge[2] << ", " << decoded_bridge[3] << "]\n\n";
    }

    std::cout << "\n================= SUMMARY =================\n"
              << "  Worst native round-trip error: " << std::scientific
              << std::setprecision(3) << worst_native_err << "\n"
              << "  Worst bridge round-trip error: " << std::scientific
              << std::setprecision(3) << worst_bridge_err
              << "  (test: " << worst_bridge_test << ")\n";

    if (worst_bridge_err < 1e-3) {
        std::cout << "  >>> Encodings COMPATIBLE: Lattigo poly decodes correctly via HEonGPU.\n";
    } else {
        std::cout << "  >>> Encodings DIVERGE: bridge recall bug localised to canonical embedding.\n";
    }
    return 0;
}
