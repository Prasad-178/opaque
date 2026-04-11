// dump_ntt_roots.cpp — Extract NTT primitive roots from HEonGPU
// for comparison with Lattigo's roots.
//
// Build as HEonGPU example and run on GPU instance.

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <heongpu/heongpu.hpp>

using namespace heongpu;
constexpr auto S = Scheme::CKKS;

int main() {
    auto ctx = GenHEContext<S>(sec_level_type::none);
    ctx->set_poly_modulus_degree(16384);

    std::vector<uint64_t> q = {
        1152921504606748673ULL, 35184372121601ULL, 35184372744193ULL,
        35184373006337ULL, 35184371138561ULL, 35184370941953ULL,
        35184370352129ULL, 35184373989377ULL
    };
    std::vector<uint64_t> p = { 2305843009211662337ULL };

    ctx->set_coeff_modulus_values(q, p);
    ctx->generate();

    std::cout << "=== HEonGPU NTT Roots ===" << std::endl;
    std::cout << "LogN: 14, N: 16384" << std::endl;
    std::cout << "Q primes: " << q.size() << ", P primes: " << p.size() << std::endl;
    std::cout << std::endl;

    // Access the NTT table from context
    // The NTT table is stored on GPU — need to copy to host
    // ctx->ntt_table_ contains the forward NTT roots

    // Alternative: use the generate_primitive_root_of_unity function directly
    // which is called during context generation
    std::vector<uint64_t> all_primes;
    for (auto qi : q) all_primes.push_back(qi);
    for (auto pi : p) all_primes.push_back(pi);

    // Print the modulus values
    auto key_modulus = ctx->get_key_modulus();
    std::cout << "Key modulus chain (" << key_modulus.size() << " primes):" << std::endl;
    for (size_t i = 0; i < key_modulus.size(); i++) {
        std::cout << "  [" << i << "]: " << key_modulus[i].value << std::endl;
    }
    std::cout << std::endl;

    // The NTT roots are generated in context.cu by:
    //   generate_primitive_root_of_unity(n, prime_vector_)
    // which returns base_q_psi (the psi values for each prime).
    //
    // Then generate_ntt_table(base_q_psi, prime_vector_, n_power)
    // creates the actual NTT table stored on GPU.
    //
    // We can't easily access the GPU NTT table from here, but we CAN
    // replicate the root generation to compare with Lattigo.

    // From GPU-NTT's NTTParameters constructor:
    // The psi (primitive 2N-th root of unity) is computed as:
    //   omega = generator^((p-1)/N) mod p  (N-th root)
    //   psi = generator^((p-1)/(2N)) mod p (2N-th root, psi^2 = omega)
    //
    // This uses the smallest generator of Z/pZ*.

    // Let's compute psi for each prime manually to compare with Lattigo
    std::cout << "=== Computing psi (primitive 2N-th root) for each prime ===" << std::endl;
    int N = 16384;

    for (size_t i = 0; i < all_primes.size(); i++) {
        uint64_t p = all_primes[i];

        // Find the smallest generator of Z/pZ*
        // For NTT primes, p ≡ 1 (mod 2N), so 2N divides p-1
        // The generator g has order p-1

        // Compute psi = g^((p-1)/(2N)) mod p
        uint64_t exp = (p - 1) / (2 * N);

        // Find smallest primitive root
        // Try small values: 2, 3, 5, 6, 7, ...
        uint64_t psi = 0;
        for (uint64_t g = 2; g < 100; g++) {
            // Check if g is a primitive root mod p
            // g^((p-1)/2) should NOT be 1 (mod p)
            // g^((p-1)/q) should NOT be 1 (mod p) for all prime factors q of p-1

            // Simple check: compute g^exp mod p as candidate psi
            // Then verify psi^(2N) = 1 mod p and psi^N != 1 mod p

            // Modular exponentiation: g^exp mod p
            __uint128_t result = 1;
            __uint128_t base = g;
            uint64_t e = exp;
            while (e > 0) {
                if (e & 1) result = (result * base) % p;
                base = (base * base) % p;
                e >>= 1;
            }
            uint64_t candidate = (uint64_t)result;

            // Check: candidate^(2N) should be 1 mod p
            __uint128_t check = 1;
            base = candidate;
            for (int j = 0; j < 2 * N; j++) {
                // Too slow for large 2N, use repeated squaring
                break;
            }

            // Quick check: candidate^N mod p should NOT be 1
            // (then candidate is a primitive 2N-th root, not just N-th)
            __uint128_t checkN = 1;
            base = candidate;
            uint64_t eN = N;
            while (eN > 0) {
                if (eN & 1) checkN = (checkN * base) % p;
                base = (base * base) % p;
                eN >>= 1;
            }

            if ((uint64_t)checkN != 1) {
                psi = candidate;
                std::cout << "  Prime[" << i << "] = " << p
                          << " | generator = " << g
                          << " | psi = g^((p-1)/(2N)) = " << psi << std::endl;
                break;
            }
        }

        if (psi == 0) {
            std::cout << "  Prime[" << i << "] = " << p << " | psi NOT FOUND" << std::endl;
        }
    }

    return 0;
}
