// intt_ordering.cpp — Produces HEonGPU's INTT output for the same known input
// as Lattigo's dump-intt-ordering tool. Compare to find index permutation.
//
// Input: NTT-domain polynomial where index i has value (i+1)
// Same input as Lattigo side.

#include <iostream>
#include <vector>
#include <heongpu/heongpu.hpp>
using namespace heongpu;
constexpr auto S = Scheme::CKKS;

int main() {
    auto ctx = GenHEContext<S>(sec_level_type::none);
    ctx->set_poly_modulus_degree(16384);
    // Use ONLY Q[0] for simplicity — single modulus comparison
    std::vector<uint64_t> q = {1152921504606748673ULL, 35184372121601ULL,
        35184372744193ULL, 35184373006337ULL, 35184371138561ULL,
        35184370941953ULL, 35184370352129ULL, 35184373989377ULL};
    std::vector<uint64_t> p = {2305843009211662337ULL};
    ctx->set_coeff_modulus_values(q, p);
    ctx->generate();

    int N = 16384;
    int QP = q.size() + p.size();
    uint64_t Q0 = q[0];

    std::cout << "N=" << N << ", Q[0]=" << Q0 << std::endl;

    // Create known NTT-domain input: value at position i = (i+1)
    // Only fill the first modulus (Q[0]), rest zeros
    std::vector<uint64_t> ntt_input(QP * N, 0);
    for (int i = 0; i < N; i++) {
        ntt_input[i] = (uint64_t)(i + 1) % Q0;
    }

    std::cout << "NTT input[0..9]: ";
    for (int i = 0; i < 10; i++) std::cout << ntt_input[i] << " ";
    std::cout << std::endl;

    DeviceVector<uint64_t> d_data(ntt_input);

    // Apply INTT using RNS version (QP moduli, but only first matters)
    gpuntt::ntt_rns_configuration<uint64_t> cfg_inv = {
        .n_power = 14, .ntt_type = gpuntt::INVERSE,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = ctx->get_n_inverse().data(),
        .stream = cudaStreamDefault
    };

    gpuntt::GPU_INTT_Inplace(d_data.data(), ctx->get_intt_table().data(),
        ctx->get_modulus().data(), cfg_inv, QP, QP);
    cudaDeviceSynchronize();

    // Read back first modulus results
    std::vector<uint64_t> coeff_out(QP * N);
    cudaMemcpy(coeff_out.data(), d_data.data(), QP * N * sizeof(uint64_t),
        cudaMemcpyDeviceToHost);

    std::cout << "\nHEonGPU INTT output (first 20):" << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << "  [" << i << "] = " << coeff_out[i] << std::endl;
    }

    std::cout << "\nHEonGPU INTT at key positions:" << std::endl;
    int positions[] = {0,1,2,3,4,8,16,32,64,128,256,512,1024,2048,4096,8192};
    for (int p : positions) {
        if (p < N) {
            std::cout << "  [" << p << "] = " << coeff_out[p] << std::endl;
        }
    }

    // Lattigo INTT values for comparison (from Go output)
    uint64_t lattigo_intt[] = {
        576460752303382529ULL, 388832870431833430ULL, 508277958646733817ULL,
        332746362894987372ULL, 325238026541738767ULL, 573097772148836905ULL,
        446259379817953885ULL, 54643056679141819ULL, 57714449758225539ULL,
        971449667303732505ULL, 273773039341795307ULL, 261115339402623680ULL,
        692441278225502872ULL, 685322643211089639ULL, 257499775151601460ULL,
        971316583803151176ULL, 461462158868921966ULL, 723668203879559796ULL,
        424436053523422799ULL, 784484859501908191ULL
    };

    std::cout << "\n=== COMPARISON ===" << std::endl;
    bool all_match = true;
    int first_mismatch = -1;
    for (int i = 0; i < 20; i++) {
        bool match = (coeff_out[i] == lattigo_intt[i]);
        std::cout << "  [" << i << "] HEonGPU=" << coeff_out[i]
                  << " Lattigo=" << lattigo_intt[i]
                  << (match ? " ✓" : " ✗") << std::endl;
        if (!match) {
            all_match = false;
            if (first_mismatch < 0) first_mismatch = i;
        }
    }

    if (all_match) {
        std::cout << "\n✓ ALL MATCH! Coefficient ordering is IDENTICAL!" << std::endl;
        std::cout << "The bridge issue is elsewhere." << std::endl;
    } else {
        std::cout << "\n✗ MISMATCH at index " << first_mismatch << std::endl;

        // Try to find Lattigo's first value in HEonGPU's output
        std::cout << "\nSearching for Lattigo[0]=" << lattigo_intt[0] << " in HEonGPU output..." << std::endl;
        for (int i = 0; i < N; i++) {
            if (coeff_out[i] == lattigo_intt[0]) {
                std::cout << "  Found at HEonGPU[" << i << "]!" << std::endl;
                // Check if it's a simple offset or bit-reversal
                std::cout << "  Checking if HEonGPU[" << i << "+j] == Lattigo[j]..." << std::endl;
                bool offset_match = true;
                for (int j = 0; j < 10 && (i+j) < N; j++) {
                    if (coeff_out[i+j] != lattigo_intt[j]) {
                        offset_match = false;
                        break;
                    }
                }
                if (offset_match) {
                    std::cout << "  → Simple offset by " << i << "!" << std::endl;
                }
            }
        }

        // Check bit-reversal
        std::cout << "\nChecking bit-reversal permutation..." << std::endl;
        auto bitrev = [](int v, int bits) -> int {
            int result = 0;
            for (int i = 0; i < bits; i++) {
                result = (result << 1) | (v & 1);
                v >>= 1;
            }
            return result;
        };

        bool is_bitrev = true;
        for (int i = 0; i < 20; i++) {
            int br = bitrev(i, 14);
            if (br < N && coeff_out[br] != lattigo_intt[i]) {
                is_bitrev = false;
                break;
            }
        }
        if (is_bitrev) {
            std::cout << "  → It IS a bit-reversal permutation!" << std::endl;
        }

        // Check inverse bit-reversal
        bool is_inv_bitrev = true;
        for (int i = 0; i < 20; i++) {
            int br = bitrev(i, 14);
            if (br < N && coeff_out[i] != lattigo_intt[br]) {
                is_inv_bitrev = false;
                break;
            }
        }
        if (is_inv_bitrev) {
            std::cout << "  → It IS an inverse bit-reversal permutation!" << std::endl;
        }

        // Just try matching Lattigo values anywhere in HEonGPU
        std::cout << "\nFinding Lattigo values in HEonGPU output:" << std::endl;
        for (int li = 0; li < 5; li++) {
            for (int hi = 0; hi < N; hi++) {
                if (coeff_out[hi] == lattigo_intt[li]) {
                    std::cout << "  Lattigo[" << li << "] = HEonGPU[" << hi << "]" << std::endl;
                    break;
                }
            }
        }
    }

    return 0;
}
