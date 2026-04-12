// GPU HE Server — CKKS computation on GPU via HEonGPU.
//
// This server receives raw polynomial coefficients from a Go/Lattigo client,
// reconstructs CKKS ciphertexts in HEonGPU format, performs homomorphic
// operations on GPU, and returns raw result coefficients.
//
// The server NEVER has access to secret keys or plaintext data.
// Only evaluation keys (Galois + relin) and encrypted data cross the wire.
//
// Build: cmake -DCMAKE_CUDA_ARCHITECTURES=75 .. && make
// Run:   ./gpu-he-server --port 50052

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <vector>
#include <sstream>

#include <grpcpp/grpcpp.h>
#include "gpuhe.grpc.pb.h"

// HEonGPU headers
#include <heongpu/heongpu.hpp>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using namespace opaque::gpu::v1;

// Session holds per-client HEonGPU state.
struct Session {
    heongpu::HEContext<heongpu::Scheme::CKKS> context;
    std::unique_ptr<heongpu::HEEncoder<heongpu::Scheme::CKKS>> encoder;
    std::unique_ptr<heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>> ops;
    int ring_size;
    int coeff_modulus_count;
    std::vector<uint64_t> q_moduli;
    std::vector<uint64_t> p_moduli;

    // Evaluation keys (stored on GPU)
    std::unique_ptr<heongpu::Galoiskey<heongpu::Scheme::CKKS>> galois_key;
    std::unique_ptr<heongpu::Relinkey<heongpu::Scheme::CKKS>> relin_key;

    std::mutex mu; // Protects operator (not thread-safe)
};

class GPUHEServiceImpl final : public GPUHEService::Service {
public:
    GPUHEServiceImpl() {
        // Print GPU info
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "GPU HE Server starting on " << prop.name
                  << " (" << prop.totalGlobalMem / (1024*1024) << " MB VRAM)" << std::endl;
        device_name_ = prop.name;
    }

    Status InitContext(ServerContext* ctx,
                       const InitContextRequest* req,
                       InitContextResponse* resp) override {
        if (req->session_id().empty()) {
            resp->set_success(false);
            resp->set_error("session_id required");
            return Status::OK;
        }

        auto params = req->params();
        if (!params.q_moduli_size()) {
            resp->set_success(false);
            resp->set_error("CKKSParams with Q moduli required");
            return Status::OK;
        }

        std::cout << "[" << req->session_id() << "] InitContext: LogN="
                  << params.log_n() << ", Q=" << params.q_moduli_size()
                  << ", P=" << params.p_moduli_size() << std::endl;

        try {
            auto session = std::make_shared<Session>();

            std::vector<uint64_t> q_moduli(params.q_moduli().begin(), params.q_moduli().end());
            std::vector<uint64_t> p_moduli(params.p_moduli().begin(), params.p_moduli().end());
            session->q_moduli = q_moduli;
            session->p_moduli = p_moduli;

            session->context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none
            );
            session->context->set_poly_modulus_degree(1 << params.log_n());
            session->context->set_coeff_modulus_values(q_moduli, p_moduli);
            session->context->generate();

            session->ring_size = 1 << params.log_n();
            session->coeff_modulus_count = q_moduli.size();

            // Extract NTT roots (ψ per prime) using the same function
            // that the context uses internally during generate().
            std::vector<heongpu::Modulus64> prime_vector;
            for (auto q : q_moduli) prime_vector.push_back(heongpu::Modulus64(q));
            for (auto p : p_moduli) prime_vector.push_back(heongpu::Modulus64(p));

            std::vector<uint64_t> ntt_roots =
                heongpu::generate_primitive_root_of_unity(1 << params.log_n(), prime_vector);

            // Create encoder and arithmetic operator
            session->encoder = std::make_unique<heongpu::HEEncoder<heongpu::Scheme::CKKS>>(
                session->context);
            session->ops = std::make_unique<heongpu::HEArithmeticOperator<heongpu::Scheme::CKKS>>(
                session->context, *session->encoder);

            // Return roots to client
            for (auto root : ntt_roots) {
                resp->add_ntt_roots(root);
            }

            std::lock_guard<std::mutex> lock(sessions_mu_);
            sessions_[req->session_id()] = session;

            resp->set_success(true);
            std::cout << "[" << req->session_id() << "] Context created, "
                      << ntt_roots.size() << " NTT roots returned" << std::endl;

        } catch (const std::exception& e) {
            resp->set_success(false);
            resp->set_error(std::string("InitContext failed: ") + e.what());
        }

        return Status::OK;
    }

    Status RegisterEvalKeys(ServerContext* ctx,
                            const RegisterEvalKeysRequest* req,
                            RegisterEvalKeysResponse* resp) override {
        if (req->session_id().empty()) {
            resp->set_success(false);
            resp->set_error("session_id required");
            return Status::OK;
        }

        // Session must already exist (created by InitContext).
        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(sessions_mu_);
            auto it = sessions_.find(req->session_id());
            if (it == sessions_.end()) {
                resp->set_success(false);
                resp->set_error("session not found; call InitContext first");
                return Status::OK;
            }
            session = it->second;
        }

        std::cout << "[" << req->session_id() << "] RegisterEvalKeys: "
                  << req->galois_keys_heongpu().size() << " bytes HEonGPU-format keys" << std::endl;

        try {
            std::lock_guard<std::mutex> lock(session->mu);

            // Load HEonGPU-format Galois keys (NTT-converted by Go client).
            if (!req->galois_keys_heongpu().empty()) {
                std::istringstream gk_stream(req->galois_keys_heongpu());
                session->galois_key = std::make_unique<Galoiskey<Scheme::CKKS>>();
                session->galois_key->set_context(session->context);
                session->galois_key->load(gk_stream);
                std::cout << "[" << req->session_id() << "] Galois keys loaded" << std::endl;
            }

            resp->set_success(true);

        } catch (const std::exception& e) {
            resp->set_success(false);
            resp->set_error(std::string("RegisterEvalKeys failed: ") + e.what());
        }

        return Status::OK;
    }

    Status BatchDotProduct(ServerContext* ctx,
                           const BatchDotProductRequest* req,
                           BatchDotProductResponse* resp) override {
        std::shared_ptr<Session> session;
        {
            std::lock_guard<std::mutex> lock(sessions_mu_);
            auto it = sessions_.find(req->session_id());
            if (it == sessions_.end()) {
                resp->set_error("session not found; call RegisterEvalKeys first");
                return Status::OK;
            }
            session = it->second;
        }

        std::lock_guard<std::mutex> lock(session->mu);

        if (!req->has_raw_query() || req->raw_query().polynomials_size() < 2) {
            resp->set_error("raw_query with at least 2 polynomials required");
            return Status::OK;
        }

        try {
            auto total_start = std::chrono::high_resolution_clock::now();

            int dim = req->dimension();
            int ring_size = session->ring_size;
            int Q_size = session->coeff_modulus_count;

            // --- Reconstruct ciphertext from raw coefficients ---
            // Raw coefficients are already in HEonGPU's NTT domain (converted by Go client).
            // Create a Ciphertext object and overwrite its data via cudaMemcpy.
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_query(session->context);
            {
                // Dummy encrypt to allocate GPU memory with correct structure
                std::vector<double> dummy(ring_size / 2, 0.0);
                heongpu::Plaintext<heongpu::Scheme::CKKS> pt_dummy(session->context);
                session->encoder->encode(pt_dummy, dummy, pow(2.0, 45));

                // We need a public key to encrypt — but we don't have one from the client.
                // Instead, just allocate the ciphertext memory and overwrite.
                // Ciphertext needs cipher_size_ * Q_size * ring_size uint64 values.
                // For a fresh ct: 2 * Q_size * N values.

                auto& raw_q = req->raw_query();
                int num_polys = raw_q.polynomials_size(); // Should be 2
                int num_levels = raw_q.polynomials(0).num_levels();
                int ct_total = num_polys * num_levels * ring_size;

                std::vector<uint64_t> ct_data(ct_total);
                for (int p = 0; p < num_polys; p++) {
                    auto& poly = raw_q.polynomials(p);
                    for (int lvl = 0; lvl < num_levels; lvl++) {
                        int src_offset = lvl * ring_size;
                        int dst_offset = (p * num_levels + lvl) * ring_size;
                        std::memcpy(&ct_data[dst_offset],
                                    &poly.coefficients()[src_offset],
                                    ring_size * sizeof(uint64_t));
                    }
                }

                // Upload to GPU by overwriting the ciphertext data
                // First ensure ct_query has allocated GPU memory
                ct_query.store_in_device();
                cudaMemcpy(ct_query.data(), ct_data.data(),
                           ct_total * sizeof(uint64_t), cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();
            }

            auto after_ct_load = std::chrono::high_resolution_clock::now();

            // --- Reconstruct plaintext from raw coefficients ---
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt_centroids(session->context);
            if (req->has_raw_centroids()) {
                auto& raw_pt = req->raw_centroids().polynomial();
                int pt_levels = raw_pt.num_levels();
                int pt_total = pt_levels * ring_size;

                std::vector<uint64_t> pt_data(pt_total);
                std::memcpy(pt_data.data(), raw_pt.coefficients().data(),
                            pt_total * sizeof(uint64_t));

                // Encode a dummy to allocate memory, then overwrite
                std::vector<double> dummy(ring_size / 2, 0.0);
                session->encoder->encode(pt_centroids, dummy, pow(2.0, 45));
                pt_centroids.store_in_device();
                cudaMemcpy(pt_centroids.data(), pt_data.data(),
                           pt_total * sizeof(uint64_t), cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();
            }

            // --- GPU Batch Dot Product ---
            auto mul_start = std::chrono::high_resolution_clock::now();

            // Step 1: Multiply encrypted query × plaintext centroids
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_result(session->context);
            session->ops->multiply_plain(ct_query, pt_centroids, ct_result);

            auto rescale_start = std::chrono::high_resolution_clock::now();

            // Step 2: Rescale
            session->ops->rescale_inplace(ct_result);

            auto rotate_start = std::chrono::high_resolution_clock::now();

            // Step 3: Rotation sum within each dim-sized segment
            if (session->galois_key) {
                for (int stride = 1; stride < dim; stride *= 2) {
                    heongpu::Ciphertext<heongpu::Scheme::CKKS> ct_rotated(ct_result);
                    session->ops->rotate_rows_inplace(ct_rotated, *session->galois_key, stride);
                    session->ops->add_inplace(ct_result, ct_rotated);
                }
            }

            auto compute_end = std::chrono::high_resolution_clock::now();
            cudaDeviceSynchronize();

            // --- Extract result coefficients ---
            ct_result.store_in_host();
            std::vector<uint64_t> result_data;
            ct_result.get_data(result_data);

            int result_levels = ct_result.coeff_modulus_count() - ct_result.depth();
            int result_polys = ct_result.size(); // 2

            // Build raw result proto
            auto* raw_result = resp->mutable_raw_result();
            raw_result->set_scale(ct_result.scale());
            raw_result->set_is_ntt(ct_result.in_ntt_domain());
            raw_result->set_depth(ct_result.depth());

            for (int p = 0; p < result_polys; p++) {
                auto* rp = raw_result->add_polynomials();
                rp->set_num_levels(result_levels);
                rp->set_ring_size(ring_size);
                int offset = p * result_levels * ring_size;
                for (int i = 0; i < result_levels * ring_size; i++) {
                    rp->add_coefficients(result_data[offset + i]);
                }
            }

            auto total_end = std::chrono::high_resolution_clock::now();

            // Timing
            resp->set_compute_time_us(
                std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count());
            resp->set_multiply_us(
                std::chrono::duration_cast<std::chrono::microseconds>(rescale_start - mul_start).count());
            resp->set_rescale_us(
                std::chrono::duration_cast<std::chrono::microseconds>(rotate_start - rescale_start).count());
            resp->set_rotate_us(
                std::chrono::duration_cast<std::chrono::microseconds>(compute_end - rotate_start).count());

        } catch (const std::exception& e) {
            resp->set_error(std::string("Compute failed: ") + e.what());
        }

        return Status::OK;
    }

    Status HealthCheck(ServerContext* ctx,
                       const GPUHealthCheckRequest* req,
                       GPUHealthCheckResponse* resp) override {
        resp->set_status(GPUHealthCheckResponse::SERVING);
        resp->set_backend(GPUHealthCheckResponse::CUDA);
        resp->set_device_name(device_name_);

        std::lock_guard<std::mutex> lock(sessions_mu_);
        resp->set_active_sessions(sessions_.size());
        return Status::OK;
    }

private:
    std::string device_name_;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
    std::mutex sessions_mu_;
};

int main(int argc, char** argv) {
    std::string port = "50052";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            port = argv[++i];
        }
    }

    std::string addr = "0.0.0.0:" + port;

    GPUHEServiceImpl service;

    ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.SetMaxReceiveMessageSize(256 * 1024 * 1024); // 256MB for eval keys
    builder.SetMaxSendMessageSize(256 * 1024 * 1024);
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
    std::cout << "GPU HE Server listening on " << addr << std::endl;
    server->Wait();

    return 0;
}
