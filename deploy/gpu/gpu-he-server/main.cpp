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
#include <heongpu/heongpu.cuh>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using namespace opaque::gpu::v1;

// Session holds per-client HEonGPU state.
struct Session {
    heongpu::HEContext<heongpu::Scheme::CKKS> context;
    std::unique_ptr<heongpu::HEOperator<heongpu::Scheme::CKKS>> op;
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

    Status RegisterEvalKeys(ServerContext* ctx,
                            const RegisterEvalKeysRequest* req,
                            RegisterEvalKeysResponse* resp) override {
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

        std::cout << "[" << req->session_id() << "] RegisterEvalKeys: LogN="
                  << params.log_n() << ", Q=" << params.q_moduli_size()
                  << " primes, P=" << params.p_moduli_size() << " primes" << std::endl;

        try {
            auto session = std::make_shared<Session>();

            // Extract modulus primes.
            std::vector<uint64_t> q_moduli(params.q_moduli().begin(), params.q_moduli().end());
            std::vector<uint64_t> p_moduli(params.p_moduli().begin(), params.p_moduli().end());
            session->q_moduli = q_moduli;
            session->p_moduli = p_moduli;

            // Create HEonGPU context with exact same primes as Lattigo.
            session->context = heongpu::GenHEContext<heongpu::Scheme::CKKS>(
                heongpu::sec_level_type::none // We specify exact primes, skip security check
            );
            session->context->set_poly_modulus_degree(1 << params.log_n());
            session->context->set_coeff_modulus_values(q_moduli, p_moduli);
            session->context->generate();

            session->ring_size = 1 << params.log_n();
            session->coeff_modulus_count = q_moduli.size();

            // Create operator for HE computation.
            session->op = std::make_unique<heongpu::HEOperator<heongpu::Scheme::CKKS>>(
                session->context
            );

            // TODO: Deserialize Galois and relin keys from raw coefficient format.
            // For now, the server can only compute with keys generated locally.
            // The full key deserialization requires matching the serialization
            // format between Lattigo and HEonGPU for evaluation keys.

            std::lock_guard<std::mutex> lock(sessions_mu_);
            sessions_[req->session_id()] = session;

            resp->set_success(true);
            std::cout << "[" << req->session_id() << "] Session created" << std::endl;

        } catch (const std::exception& e) {
            resp->set_success(false);
            resp->set_error(std::string("Setup failed: ") + e.what());
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

            // Reconstruct ciphertext from raw coefficients.
            heongpu::Ciphertext<heongpu::Scheme::CKKS> ct(session->context);
            auto& raw_q = req->raw_query();
            int num_polys = raw_q.polynomials_size();
            int num_levels = raw_q.polynomials(0).num_levels();
            int total_coeffs = num_polys * num_levels * ring_size;

            // Pack raw coefficients into contiguous host buffer.
            std::vector<uint64_t> ct_data(total_coeffs);
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

            // Similarly reconstruct plaintext from raw coefficients.
            heongpu::Plaintext<heongpu::Scheme::CKKS> pt(session->context);
            if (req->has_raw_centroids()) {
                auto& raw_pt = req->raw_centroids().polynomial();
                int pt_levels = raw_pt.num_levels();
                std::vector<uint64_t> pt_data(pt_levels * ring_size);
                std::memcpy(pt_data.data(),
                            raw_pt.coefficients().data(),
                            pt_levels * ring_size * sizeof(uint64_t));
                // TODO: Set plaintext data on HEonGPU object
            }

            // TODO: Perform HE computation on GPU:
            // 1. Upload ct_data to GPU via HEonGPU Ciphertext API
            // 2. session->op->multiply_plain(ct, pt, result)
            // 3. session->op->rescale(result)
            // 4. for stride in [1, 2, 4, ...dim]: session->op->rotate(result, stride, galois_key)
            // 5. Extract result coefficients

            // For now, return an error indicating this needs the key bridge.
            // The CPU stub (cmd/gpu-server) handles the legacy serialized path.
            resp->set_error("GPU raw coefficient compute not yet implemented — "
                           "use CPU stub (cmd/gpu-server) for testing. "
                           "This server needs Galois key deserialization to complete the bridge.");

            auto total_end = std::chrono::high_resolution_clock::now();
            resp->set_compute_time_us(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    total_end - total_start).count());

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
