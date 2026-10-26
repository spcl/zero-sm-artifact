// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Experiment-only FP32, out-of-place AllGather harness matching nccl-tests.

#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mscclpp/executor.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/utils.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "utils_internal.hpp"

namespace {
std::vector<size_t> parseSizes(const char* text) {
  std::vector<size_t> sizes;
  std::stringstream stream(text == nullptr ? "" : text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) sizes.push_back(std::stoull(item));
  }
  return sizes;
}
int envInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  return value == nullptr ? fallback : std::stoi(value);
}
}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <execution plan path>" << std::endl;
    return 1;
  }
  MPI_Init(nullptr, nullptr);
  int rank = 0;
  int world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  int deviceCount = 0;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&deviceCount));
  MSCCLPP_CUDATHROW(cudaSetDevice(rank % deviceCount));
  const auto totalSizes = parseSizes(std::getenv("MSCCLPP_TOTAL_SIZES"));
  const int warmups = envInt("MSCCLPP_WARMUPS", 5);
  const int iterations = envInt("MSCCLPP_ITERS", 20);
  const int trial = envInt("MSCCLPP_TRIAL_ID", 0);
  if (totalSizes.empty() || warmups != 5 || iterations != 20 || trial < 0) {
    if (rank == 0) std::cerr << "invalid exact-protocol configuration" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 3);
  }

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world);
  mscclpp::UniqueId id;
  if (rank == 0) id = bootstrap->createUniqueId();
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(id);
  auto communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
  auto executor = std::make_shared<mscclpp::Executor>(communicator);
  mscclpp::ExecutionPlan plan(argv[1], rank);

  for (size_t totalBytes : totalSizes) {
    if (totalBytes % static_cast<size_t>(world) != 0) {
      if (rank == 0) std::cerr << "total bytes not divisible by world size: " << totalBytes << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 4);
    }
    const size_t inputBytes = totalBytes / static_cast<size_t>(world);
    if (inputBytes < sizeof(float) || inputBytes % sizeof(float) != 0) {
      if (rank == 0) std::cerr << "per-rank input is not a positive FP32 count: " << inputBytes << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 5);
    }

    auto input = mscclpp::GpuBuffer(inputBytes).memory();
    auto output = mscclpp::GpuBuffer(totalBytes).memory();
    std::vector<float> inputHost(inputBytes / sizeof(float), static_cast<float>(rank + 1));
    std::vector<float> outputHost(totalBytes / sizeof(float), -1.0f);
    MSCCLPP_CUDATHROW(cudaMemcpy(input.get(), inputHost.data(), inputBytes, cudaMemcpyHostToDevice));
    MSCCLPP_CUDATHROW(cudaMemset(output.get(), 0xff, totalBytes));
    mscclpp::CudaStreamWithFlags stream(cudaStreamNonBlocking);
    for (int iteration = 0; iteration < warmups; ++iteration) {
      executor->execute(rank, input.get(), output.get(), inputBytes, totalBytes, mscclpp::DataType::FLOAT32, plan,
                        stream, mscclpp::PacketType::LL16);
    }
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

    MSCCLPP_CUDATHROW(cudaMemcpy(outputHost.data(), output.get(), totalBytes, cudaMemcpyDeviceToHost));
    int localWrong = 0;
    const size_t valuesPerRank = inputBytes / sizeof(float);
    for (int peer = 0; peer < world && localWrong == 0; ++peer) {
      const float expected = static_cast<float>(peer + 1);
      const size_t begin = static_cast<size_t>(peer) * valuesPerRank;
      const size_t end = begin + valuesPerRank;
      for (size_t index = begin; index < end; ++index) {
        if (outputHost[index] != expected) {
          std::cerr << "rank " << rank << " validation failure peer " << peer << " index " << index << " got "
                    << outputHost[index] << " expected " << expected << " total_bytes " << totalBytes << std::endl;
          localWrong = 1;
          break;
        }
      }
    }
    int globalWrong = 0;
    MPI_Allreduce(&localWrong, &globalWrong, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (globalWrong != 0) MPI_Abort(MPI_COMM_WORLD, 6);
    if (rank == 0) {
      std::cout << "MSCCLPP_FULL_VALIDATION_PASS,total_bytes=" << totalBytes << ",world_size=" << world
                << ",dtype=float32,out_of_place=1" << std::endl;
    }

    bootstrap->barrier();
    cudaEvent_t start{};
    cudaEvent_t stop{};
    MSCCLPP_CUDATHROW(cudaEventCreate(&start));
    MSCCLPP_CUDATHROW(cudaEventCreate(&stop));
    MSCCLPP_CUDATHROW(cudaEventRecord(start, stream));
    for (int iteration = 0; iteration < iterations; ++iteration) {
      executor->execute(rank, input.get(), output.get(), inputBytes, totalBytes, mscclpp::DataType::FLOAT32, plan,
                        stream, mscclpp::PacketType::LL16);
    }
    MSCCLPP_CUDATHROW(cudaEventRecord(stop, stream));
    MSCCLPP_CUDATHROW(cudaEventSynchronize(stop));
    float localMs = 0.0f;
    MSCCLPP_CUDATHROW(cudaEventElapsedTime(&localMs, start, stop));
    MSCCLPP_CUDATHROW(cudaEventDestroy(start));
    MSCCLPP_CUDATHROW(cudaEventDestroy(stop));
    const double localUs = static_cast<double>(localMs) * 1000.0 / iterations;
    double maxUs = 0.0;
    MPI_Reduce(&localUs, &maxUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      const double algbw = static_cast<double>(totalBytes) / 1.0e9 / (maxUs / 1.0e6);
      const double busbw = algbw * static_cast<double>(world - 1) / static_cast<double>(world);
      std::cout << "MSCCLPP_EXACT_TRIAL," << trial << ',' << totalBytes << ',' << inputBytes << ',' << maxUs << ','
                << algbw << ',' << busbw << ',' << world << std::endl;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "MSCCLPP_EXACT_PROCESS_PASS,trial=" << trial << ",sizes=" << totalSizes.size()
              << ",warmups=5,iters=20,dtype=float32,out_of_place=1,graph=0,world_size=" << world << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout.flush();
  std::cerr.flush();
  std::_Exit(0);
}
