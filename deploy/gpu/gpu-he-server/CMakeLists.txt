cmake_minimum_required(VERSION 3.26)
project(gpu-he-server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find dependencies
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG QUIET)

# If gRPC not found via cmake, try pkg-config
if(NOT gRPC_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRPC REQUIRED grpc++ grpc)
endif()

# HEonGPU — assume it's built in /home/ubuntu/HEonGPU/build
set(HEONGPU_DIR "$ENV{HOME}/HEonGPU" CACHE PATH "HEonGPU source directory")
set(HEONGPU_BUILD_DIR "${HEONGPU_DIR}/build" CACHE PATH "HEonGPU build directory")

# Generate proto C++ code
set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../../api/proto/gpuhe.proto")
set(PROTO_OUT "${CMAKE_CURRENT_BINARY_DIR}/proto")
file(MAKE_DIRECTORY ${PROTO_OUT})

# Proto generation commands
add_custom_command(
    OUTPUT "${PROTO_OUT}/gpuhe.pb.cc" "${PROTO_OUT}/gpuhe.pb.h"
           "${PROTO_OUT}/gpuhe.grpc.pb.cc" "${PROTO_OUT}/gpuhe.grpc.pb.h"
    COMMAND protobuf::protoc
    ARGS --grpc_out="${PROTO_OUT}"
         --cpp_out="${PROTO_OUT}"
         -I "${CMAKE_CURRENT_SOURCE_DIR}/../../../api/proto"
         --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
         "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
    COMMENT "Generating gRPC C++ code from gpuhe.proto"
)

# Server executable
add_executable(gpu-he-server
    main.cpp
    ${PROTO_OUT}/gpuhe.pb.cc
    ${PROTO_OUT}/gpuhe.grpc.pb.cc
)

target_include_directories(gpu-he-server PRIVATE
    ${PROTO_OUT}
    ${HEONGPU_DIR}/src/include
    ${HEONGPU_BUILD_DIR}/_deps/gpu-ntt-src/src/include
    ${HEONGPU_BUILD_DIR}/_deps/rmm-src/include
    ${HEONGPU_BUILD_DIR}/_deps/spdlog-src/include
    ${HEONGPU_BUILD_DIR}/_deps/rapids_logger-src/include
)

target_link_directories(gpu-he-server PRIVATE
    ${HEONGPU_BUILD_DIR}/src
    ${HEONGPU_BUILD_DIR}/thirdparty/GPU-NTT/src
    ${HEONGPU_BUILD_DIR}/thirdparty/RNGonGPU/src
    ${HEONGPU_BUILD_DIR}/_deps/rmm-build
    ${HEONGPU_BUILD_DIR}/_deps/spdlog-build
)

target_link_libraries(gpu-he-server PRIVATE
    heongpu
    gRPC::grpc++
    protobuf::libprotobuf
    cudart
    gmp
    ntl
)
