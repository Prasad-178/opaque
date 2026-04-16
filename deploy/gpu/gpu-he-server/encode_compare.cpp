// encode_compare.cpp — dumps polynomial coefficient representations of
// known CKKS-encoded vectors in HEonGPU's convention. The Go twin in
// cmd/encode-compare-go/main.go dumps the same vectors in Lattigo's
// convention. If the dumps disagree, the canonical embedding (IFFT +
// slot-to-coefficient mapping) differs between libraries, which is the
// prime suspect for the Lattigo→HEonGPU bridge recall collapse.
//
// Build: compile with HEonGPU CMake (add to gpu-he-server/CMakeLists.txt).
// Run:   ./encode_compare heongpu_dump.bin
//
// Dump format (little-endian, same as Go):
//   magic u32 = 0x4F504151 ("OPAQ")
//   version u32 = 1
//   logN u32
//   numPrimes u32 = Q_size
//   numTests u32
//   for each test: nameLen u32, name bytes, then numPrimes × N × u64 coeffs

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
constexpr uint32_t kVersion = 1;

struct TestVec {
    std::string name;
    std::vector<double> values;
};

void write_u32(std::ofstream& os, uint32_t v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
void write_u64(std::ofstream& os, uint64_t v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

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
    const char* out_path = argc > 1 ? argv[1] : "heongpu_dump.bin";

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

    std::cout << "HEonGPU encode dump\n"
              << "  LogN=" << log_n << " N=" << N
              << " slots=" << slot_count
              << " qSize=" << Q_size
              << " scale=" << std::fixed << std::setprecision(0) << scale << "\n";

    HEEncoder<S> encoder(ctx);

    std::vector<TestVec> tests = {
        {"slot0_eq_1",      with_slot(slot_count, 0, 1.0)},
        {"slot1_eq_1",      with_slot(slot_count, 1, 1.0)},
        {"slot2_eq_1",      with_slot(slot_count, 2, 1.0)},
        {"slot_last_eq_1",  with_slot(slot_count, slot_count - 1, 1.0)},
        {"ramp_0_to_15",    ramp_to(slot_count, 16)},
        {"constant_0.5",    constant_vec(slot_count, 0.5)},
        {"sine_wave",       sine_wave(slot_count)},
    };

    std::ofstream os(out_path, std::ios::binary);
    if (!os) {
        std::cerr << "cannot open " << out_path << "\n";
        return 1;
    }
    write_u32(os, kMagic);
    write_u32(os, kVersion);
    write_u32(os, (uint32_t)log_n);
    write_u32(os, (uint32_t)Q_size);
    write_u32(os, (uint32_t)tests.size());

    // The encoder produces plaintext in NTT domain. To compare with Lattigo
    // (coefficient domain) we need to invert the NTT. Match main.cpp INTT
    // config — N^-1 is baked into HEonGPU's INTT tables, so mod_inverse
    // need not be set explicitly.
    gpuntt::ntt_rns_configuration<uint64_t> cfg_intt = {
        .n_power = log_n,
        .ntt_type = gpuntt::INVERSE,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = cudaStreamDefault,
    };

    for (const auto& t : tests) {
        Plaintext<S> pt(ctx);
        encoder.encode(pt, t.values, scale);

        // pt is on device in NTT domain. Apply INTT in-place (batch=1, mod_count=Q_size).
        gpuntt::GPU_NTT_Inplace(
            pt.data(),
            ctx->get_intt_table().data(),
            ctx->get_modulus().data(),
            cfg_intt,
            1,       // one polynomial in a plaintext
            Q_size   // Q_size RNS components
        );
        cudaDeviceSynchronize();

        pt.store_in_host();
        std::vector<uint64_t> pt_raw;
        pt.get_data(pt_raw);

        if ((int)pt_raw.size() < Q_size * N) {
            std::cerr << "Plaintext " << t.name << " has only "
                      << pt_raw.size() << " coeffs (need " << Q_size * N << ")\n";
            return 2;
        }

        // Sanity print: first 8 coeffs of prime Q[0]
        std::cout << "\nTest \"" << t.name << "\": first 8 coeffs of prime Q[0]: ";
        for (int i = 0; i < 8; i++) std::cout << pt_raw[i] << " ";
        std::cout << "\n";

        // Write
        write_u32(os, (uint32_t)t.name.size());
        os.write(t.name.data(), t.name.size());
        os.write(reinterpret_cast<const char*>(pt_raw.data()),
                 (size_t)Q_size * N * sizeof(uint64_t));
    }

    std::cout << "\nWrote " << out_path << " (" << tests.size()
              << " tests, " << Q_size << " primes, N=" << N << ")\n";
    return 0;
}
