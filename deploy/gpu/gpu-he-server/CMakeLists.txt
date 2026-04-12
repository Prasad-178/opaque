cmake_minimum_required(VERSION 3.26)
project(gpu-he-server LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find dependencies
find_package(Protobuf REQUIRED)

# Try cmake config first, fall back to pkg-config
find_package(gRPC CONFIG QUIET)

if(gRPC_FOUND)
    message(STATUS "Found gRPC via cmake config")
    set(GRPC_LIBRARIES gRPC::grpc++)
    set(GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
else()
    message(STATUS "gRPC cmake config not found, using pkg-config")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRPC REQUIRED grpc++ grpc)
    # Find grpc_cpp_plugin in PATH
    find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
    message(STATUS "grpc_cpp_plugin: ${GRPC_CPP_PLUGIN}")
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
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
    COMMENT "Generating gRPC C++ code from gpuhe.proto"
)

# Find CUDA toolkit for include paths (needed when CXX files include CUDA headers)
find_package(CUDAToolkit REQUIRED)

# Server executable — main.cpp includes CUDA headers via HEonGPU, so compile as CUDA
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(gpu-he-server
    main.cpp
    ${PROTO_OUT}/gpuhe.pb.cc
    ${PROTO_OUT}/gpuhe.grpc.pb.cc
)

target_include_directories(gpu-he-server PRIVATE
    ${PROTO_OUT}
    ${HEONGPU_DIR}/src/include
    ${HEONGPU_DIR}/thirdparty/GPU-NTT/src/include
    ${HEONGPU_DIR}/thirdparty/GPU-FFT/src/include
    ${HEONGPU_DIR}/thirdparty/RNGonGPU/src/include
    ${HEONGPU_BUILD_DIR}/_deps/gpu-ntt-src/src/include
    ${HEONGPU_BUILD_DIR}/_deps/rmm-src/include
    ${HEONGPU_BUILD_DIR}/_deps/rmm-src/cpp/include
    ${HEONGPU_BUILD_DIR}/_deps/rmm-build/include
    ${HEONGPU_BUILD_DIR}/_deps/spdlog-src/include
    ${HEONGPU_BUILD_DIR}/_deps/rapids_logger-src/include
    ${HEONGPU_BUILD_DIR}/_deps/nvtx3-src/c/include
)

# RMM requires this define
target_compile_definitions(gpu-he-server PRIVATE
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
)

# Add pkg-config include dirs if using pkg-config
if(NOT gRPC_FOUND AND GRPC_INCLUDE_DIRS)
    target_include_directories(gpu-he-server PRIVATE ${GRPC_INCLUDE_DIRS})
endif()

target_link_directories(gpu-he-server PRIVATE
    ${HEONGPU_BUILD_DIR}/src
    ${HEONGPU_BUILD_DIR}/thirdparty/GPU-NTT/src
    ${HEONGPU_BUILD_DIR}/thirdparty/RNGonGPU/src
    ${HEONGPU_BUILD_DIR}/_deps/rmm-build
    ${HEONGPU_BUILD_DIR}/_deps/spdlog-build
)

if(gRPC_FOUND)
    target_link_libraries(gpu-he-server PRIVATE
        heongpu
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
        cudart
        gmp
        ntl
    )
else()
    target_link_libraries(gpu-he-server PRIVATE
        heongpu
        ${GRPC_LINK_LIBRARIES}
        protobuf::libprotobuf
        cudart
        gmp
        ntl
    )
    # Add pkg-config link directories
    if(GRPC_LIBRARY_DIRS)
        target_link_directories(gpu-he-server PRIVATE ${GRPC_LIBRARY_DIRS})
    endif()
endif()
