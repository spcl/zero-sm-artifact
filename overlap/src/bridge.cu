// C ABI used by replay.py: owns the NCCL communicator, the AllGather buffers
// (ncclMemAlloc + ncclCommWindowRegister), the communication/release/join streams,
// the device-side gate that holds both streams until a trial is fully enqueued,
// and the CUDA event timing.

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace {

int g_rank = -1;

[[noreturn]] void die(const char* kind, const char* expr, const char* detail, int line) {
  std::fprintf(stderr, "bridge FATAL rank=%d kind=%s line=%d expr=%s detail=%s\n", g_rank, kind,
               line, expr, detail ? detail : "unknown");
  std::fflush(stderr);
  std::_Exit(2);
}

#define CUDA_CHECK(x)                                                         \
  do {                                                                        \
    cudaError_t _s = (x);                                                     \
    if (_s != cudaSuccess) die("CUDA", #x, cudaGetErrorString(_s), __LINE__); \
  } while (0)
#define NCCL_CHECK(x)                                                          \
  do {                                                                         \
    ncclResult_t _s = (x);                                                     \
    if (_s != ncclSuccess) die("NCCL", #x, ncclGetErrorString(_s), __LINE__); \
  } while (0)
#define CU_CHECK(x)                          \
  do {                                       \
    CUresult _s = (x);                       \
    if (_s != CUDA_SUCCESS) {                \
      const char* text = nullptr;            \
      cuGetErrorString(_s, &text);           \
      die("CUDA_DRIVER", #x, text, __LINE__); \
    }                                        \
  } while (0)

void write_all(int fd, const void* data, size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  while (bytes) {
    ssize_t n = ::write(fd, p, bytes);
    if (n < 0) {
      if (errno == EINTR) continue;
      die("POSIX", "write", std::strerror(errno), __LINE__);
    }
    p += n;
    bytes -= static_cast<size_t>(n);
  }
}

void read_all(int fd, void* data, size_t bytes) {
  auto* p = static_cast<unsigned char*>(data);
  while (bytes) {
    ssize_t n = ::read(fd, p, bytes);
    if (n < 0) {
      if (errno == EINTR) continue;
      die("POSIX", "read", std::strerror(errno), __LINE__);
    }
    if (n == 0) die("POSIX", "read", "unexpected EOF", __LINE__);
    p += n;
    bytes -= static_cast<size_t>(n);
  }
}

// Rank 0 publishes the NCCL unique id through a file on the shared output directory.
ncclUniqueId distribute_id(const char* path, int rank) {
  ncclUniqueId id{};
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
    std::string temp = std::string(path) + ".tmp";
    int fd = ::open(temp.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
    if (fd < 0) die("POSIX", "open id", std::strerror(errno), __LINE__);
    write_all(fd, &id, sizeof(id));
    if (::fsync(fd) != 0) die("POSIX", "fsync id", std::strerror(errno), __LINE__);
    if (::close(fd) != 0) die("POSIX", "close id", std::strerror(errno), __LINE__);
    if (::rename(temp.c_str(), path) != 0) die("POSIX", "rename id", std::strerror(errno), __LINE__);
  } else {
    bool ready = false;
    for (int i = 0; i < 12000 && !ready; ++i) {
      struct stat st {};
      if (::stat(path, &st) == 0 && st.st_size == static_cast<off_t>(sizeof(id))) {
        int fd = ::open(path, O_RDONLY);
        if (fd >= 0) {
          read_all(fd, &id, sizeof(id));
          if (::close(fd) != 0) die("POSIX", "close id", std::strerror(errno), __LINE__);
          ready = true;
        }
      }
      if (!ready) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!ready) die("POSIX", "id wait", "timeout", __LINE__);
  }
  return id;
}

__global__ void validate_ag(const unsigned char* recv, size_t output_bytes, size_t send_bytes,
                            unsigned long long* bad) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < output_bytes;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    if (recv[i] != static_cast<unsigned char>(i / send_bytes)) atomicAdd(bad, 1ull);
  }
}

struct State {
  ncclComm_t comm = nullptr;
  unsigned char* scale_send = nullptr;
  unsigned char* scale_recv = nullptr;
  unsigned char* data_send = nullptr;
  unsigned char* data_recv = nullptr;
  size_t scale_send_bytes = 0;
  size_t scale_output_bytes = 0;
  size_t data_send_bytes = 0;
  size_t data_output_bytes = 0;
  ncclWindow_t scale_send_window = nullptr;
  ncclWindow_t scale_recv_window = nullptr;
  ncclWindow_t data_send_window = nullptr;
  ncclWindow_t data_recv_window = nullptr;
  cudaStream_t comm_stream = nullptr;
  cudaStream_t release_stream = nullptr;
  cudaStream_t join_stream = nullptr;
  int comm_stream_priority = 0;
  cudaEvent_t start = nullptr;
  cudaEvent_t phase_ready = nullptr;
  cudaEvent_t compute_done = nullptr;
  cudaEvent_t comm_done = nullptr;
  cudaEvent_t joined_done = nullptr;
  uint32_t* gate_ptr = nullptr;
  uint32_t token = 0;
  int* barrier_value = nullptr;
  bool initialized = false;
  bool compute_armed = false;
  bool phase_marked = false;
  bool compute_marked = false;
  bool comm_enqueued = false;
};

State s;

void collective_barrier() {
  int one = 1;
  CUDA_CHECK(cudaMemcpyAsync(s.barrier_value, &one, sizeof(one), cudaMemcpyHostToDevice,
                             s.release_stream));
  NCCL_CHECK(ncclAllReduce(s.barrier_value, s.barrier_value, 1, ncclInt, ncclSum, s.comm,
                           s.release_stream));
  CUDA_CHECK(cudaStreamSynchronize(s.release_stream));
}

void alloc_register(size_t send_bytes, size_t recv_bytes, unsigned char** send, unsigned char** recv,
                    ncclWindow_t* send_win, ncclWindow_t* recv_win) {
  NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(send), send_bytes));
  NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(recv), recv_bytes));
  NCCL_CHECK(ncclCommWindowRegister(s.comm, *send, send_bytes, send_win, NCCL_WIN_COLL_SYMMETRIC));
  NCCL_CHECK(ncclCommWindowRegister(s.comm, *recv, recv_bytes, recv_win, NCCL_WIN_COLL_SYMMETRIC));
}

}  // namespace

extern "C" {

// AllGathers are int8: an optional scale member (Llama, 2 KiB) grouped with the data member.
void zs_native_init(int rank, int world, int local_rank, const char* id_file,
                    unsigned long long scale_send_bytes, unsigned long long data_send_bytes) {
  if (s.initialized) die("STATE", "init", "already initialized", __LINE__);
  g_rank = rank;
  CUDA_CHECK(cudaSetDevice(local_rank));
  CU_CHECK(cuInit(0));
  int version = 0;
  NCCL_CHECK(ncclGetVersion(&version));
  if (version != NCCL_VERSION_CODE) die("CONFIG", "ncclGetVersion", "runtime NCCL differs from build headers", __LINE__);
  ncclUniqueId id = distribute_id(id_file, rank);
  NCCL_CHECK(ncclCommInitRank(&s.comm, world, id, rank));
  s.scale_send_bytes = static_cast<size_t>(scale_send_bytes);
  s.scale_output_bytes = s.scale_send_bytes * static_cast<size_t>(world);
  s.data_send_bytes = static_cast<size_t>(data_send_bytes);
  s.data_output_bytes = s.data_send_bytes * static_cast<size_t>(world);
  if (s.data_send_bytes == 0) die("CONFIG", "data_send_bytes", "must be positive", __LINE__);
  if (s.scale_send_bytes > 0)
    alloc_register(s.scale_send_bytes, s.scale_output_bytes, &s.scale_send, &s.scale_recv,
                   &s.scale_send_window, &s.scale_recv_window);
  alloc_register(s.data_send_bytes, s.data_output_bytes, &s.data_send, &s.data_recv,
                 &s.data_send_window, &s.data_recv_window);
  // NCCL_WIN_ENABLE=0 (legacy arm) must leave the buffers unregistered; otherwise all must be.
  const char* win_env = std::getenv("NCCL_WIN_ENABLE");
  const bool windows_disabled = win_env != nullptr && std::strcmp(win_env, "0") == 0;
  const bool any_window = s.data_send_window || s.data_recv_window || s.scale_send_window ||
                          s.scale_recv_window;
  const bool all_windows = s.data_send_window && s.data_recv_window &&
                           (s.scale_send_bytes == 0 || (s.scale_send_window && s.scale_recv_window));
  if (windows_disabled ? any_window : !all_windows)
    die("CONFIG", "ncclCommWindowRegister", "unexpected window registration state", __LINE__);

  int least_priority = 0, greatest_priority = 0;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  // Megatron-FSDP runs its AllGathers on a high-priority stream.
  s.comm_stream_priority = greatest_priority;
  CUDA_CHECK(cudaStreamCreateWithPriority(&s.comm_stream, cudaStreamNonBlocking, s.comm_stream_priority));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s.release_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s.join_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&s.start));
  CUDA_CHECK(cudaEventCreate(&s.phase_ready));
  CUDA_CHECK(cudaEventCreate(&s.compute_done));
  CUDA_CHECK(cudaEventCreate(&s.comm_done));
  CUDA_CHECK(cudaEventCreate(&s.joined_done));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.gate_ptr), sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(s.gate_ptr, 0, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.barrier_value), sizeof(int)));

  // One checked AllGather before any measurement.
  if (s.scale_send_bytes > 0) {
    CUDA_CHECK(cudaMemsetAsync(s.scale_send, rank, s.scale_send_bytes, s.comm_stream));
    CUDA_CHECK(cudaMemsetAsync(s.scale_recv, 0, s.scale_output_bytes, s.comm_stream));
  }
  CUDA_CHECK(cudaMemsetAsync(s.data_send, rank, s.data_send_bytes, s.comm_stream));
  CUDA_CHECK(cudaMemsetAsync(s.data_recv, 0, s.data_output_bytes, s.comm_stream));
  NCCL_CHECK(ncclGroupStart());
  if (s.scale_send_bytes > 0)
    NCCL_CHECK(ncclAllGather(s.scale_send, s.scale_recv, s.scale_send_bytes, ncclInt8, s.comm, s.comm_stream));
  NCCL_CHECK(ncclAllGather(s.data_send, s.data_recv, s.data_send_bytes, ncclInt8, s.comm, s.comm_stream));
  NCCL_CHECK(ncclGroupEnd());
  unsigned long long* bad = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&bad), sizeof(*bad)));
  CUDA_CHECK(cudaMemsetAsync(bad, 0, sizeof(*bad), s.comm_stream));
  if (s.scale_send_bytes > 0)
    validate_ag<<<1024, 256, 0, s.comm_stream>>>(s.scale_recv, s.scale_output_bytes, s.scale_send_bytes, bad);
  validate_ag<<<1024, 256, 0, s.comm_stream>>>(s.data_recv, s.data_output_bytes, s.data_send_bytes, bad);
  CUDA_CHECK(cudaGetLastError());
  unsigned long long host_bad = 0;
  CUDA_CHECK(cudaMemcpyAsync(&host_bad, bad, sizeof(host_bad), cudaMemcpyDeviceToHost, s.comm_stream));
  CUDA_CHECK(cudaStreamSynchronize(s.comm_stream));
  CUDA_CHECK(cudaFree(bad));
  if (host_bad) die("CORRECTNESS", "AllGather", "mismatch", __LINE__);
  s.initialized = true;
  std::printf("bridge rank=%d nccl=%d windows=%s scale_bytes=%zu data_bytes=%zu comm_stream_priority=%d\n",
              rank, version, windows_disabled ? "disabled" : "registered", s.scale_send_bytes,
              s.data_send_bytes, s.comm_stream_priority);
  std::fflush(stdout);
}

void zs_native_barrier() {
  if (!s.initialized) die("STATE", "barrier", "not initialized", __LINE__);
  collective_barrier();
}

void zs_native_begin_trial(int synchronize_ranks) {
  if (!s.initialized) die("STATE", "begin", "not initialized", __LINE__);
  if (synchronize_ranks) collective_barrier();
  ++s.token;
  s.compute_armed = false;
  s.phase_marked = false;
  s.compute_marked = false;
  s.comm_enqueued = false;
}

// The compute stream waits on the gate before its first GEMM.
void zs_native_arm_compute(unsigned long long stream_value) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
  CU_CHECK(cuStreamWaitValue32(reinterpret_cast<CUstream>(stream),
                               reinterpret_cast<CUdeviceptr>(s.gate_ptr), s.token,
                               CU_STREAM_WAIT_VALUE_EQ));
  s.compute_armed = true;
}

void zs_native_mark_phase_ready(unsigned long long stream_value) {
  if (!s.compute_armed) die("STATE", "mark phase", "compute not armed", __LINE__);
  CUDA_CHECK(cudaEventRecord(s.phase_ready, reinterpret_cast<cudaStream_t>(stream_value)));
  s.phase_marked = true;
}

void zs_native_mark_compute_done(unsigned long long stream_value) {
  if (!s.compute_armed || !s.phase_marked) die("STATE", "mark compute", "not armed", __LINE__);
  CUDA_CHECK(cudaEventRecord(s.compute_done, reinterpret_cast<cudaStream_t>(stream_value)));
  s.compute_marked = true;
}

// wait_for_phase=1 (overlap mode): the AllGather starts when the compute stream reaches the target GEMM.
void zs_native_enqueue_comm(int wait_for_phase, unsigned long long scale_send_bytes,
                            unsigned long long data_send_bytes) {
  if (wait_for_phase && !s.phase_marked) die("STATE", "enqueue comm", "phase not marked", __LINE__);
  if (scale_send_bytes > s.scale_send_bytes || data_send_bytes == 0 || data_send_bytes > s.data_send_bytes)
    die("CONFIG", "enqueue comm", "payload exceeds the allocated buffers", __LINE__);
  CU_CHECK(cuStreamWaitValue32(reinterpret_cast<CUstream>(s.comm_stream),
                               reinterpret_cast<CUdeviceptr>(s.gate_ptr), s.token,
                               CU_STREAM_WAIT_VALUE_EQ));
  if (wait_for_phase) CUDA_CHECK(cudaStreamWaitEvent(s.comm_stream, s.phase_ready, 0));
  NCCL_CHECK(ncclGroupStart());
  if (scale_send_bytes > 0)
    NCCL_CHECK(ncclAllGather(s.scale_send, s.scale_recv, scale_send_bytes, ncclInt8, s.comm, s.comm_stream));
  NCCL_CHECK(ncclAllGather(s.data_send, s.data_recv, data_send_bytes, ncclInt8, s.comm, s.comm_stream));
  NCCL_CHECK(ncclGroupEnd());
  CUDA_CHECK(cudaEventRecord(s.comm_done, s.comm_stream));
  s.comm_enqueued = true;
}

// mode: 0 compute_alone, 1 comm_alone, 2 overlap. Opens the gate and waits for completion.
void zs_native_finish_trial(int mode, double* compute_ms, double* comm_ms, double* joined_ms) {
  const bool has_compute = mode != 1;
  const bool has_comm = mode != 0;
  if (has_compute && !s.compute_marked) die("STATE", "finish", "compute not marked", __LINE__);
  if (has_comm && !s.comm_enqueued) die("STATE", "finish", "comm not enqueued", __LINE__);
  if (has_compute) CUDA_CHECK(cudaStreamWaitEvent(s.join_stream, s.compute_done, 0));
  if (has_comm) CUDA_CHECK(cudaStreamWaitEvent(s.join_stream, s.comm_done, 0));
  CUDA_CHECK(cudaEventRecord(s.joined_done, s.join_stream));
  CUDA_CHECK(cudaEventRecord(s.start, s.release_stream));
  CU_CHECK(cuStreamWriteValue32(reinterpret_cast<CUstream>(s.release_stream),
                                reinterpret_cast<CUdeviceptr>(s.gate_ptr), s.token,
                                CU_STREAM_WRITE_VALUE_DEFAULT));
  CUDA_CHECK(cudaEventSynchronize(s.joined_done));
  float elapsed = 0;
  *compute_ms = 0;
  *comm_ms = 0;
  if (has_compute) {
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, s.start, s.compute_done));
    *compute_ms = elapsed;
  }
  if (has_comm) {
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, s.start, s.comm_done));
    *comm_ms = elapsed;
  }
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, s.start, s.joined_done));
  *joined_ms = elapsed;
}

void zs_native_finalize() {
  if (!s.initialized) return;
  collective_barrier();
  CUDA_CHECK(cudaDeviceSynchronize());
  NCCL_CHECK(ncclCommWindowDeregister(s.comm, s.data_recv_window));
  NCCL_CHECK(ncclCommWindowDeregister(s.comm, s.data_send_window));
  if (s.scale_send_bytes > 0) {
    NCCL_CHECK(ncclCommWindowDeregister(s.comm, s.scale_recv_window));
    NCCL_CHECK(ncclCommWindowDeregister(s.comm, s.scale_send_window));
  }
  NCCL_CHECK(ncclMemFree(s.data_recv));
  NCCL_CHECK(ncclMemFree(s.data_send));
  if (s.scale_send_bytes > 0) {
    NCCL_CHECK(ncclMemFree(s.scale_recv));
    NCCL_CHECK(ncclMemFree(s.scale_send));
  }
  CUDA_CHECK(cudaFree(s.barrier_value));
  CUDA_CHECK(cudaFree(s.gate_ptr));
  CUDA_CHECK(cudaEventDestroy(s.joined_done));
  CUDA_CHECK(cudaEventDestroy(s.comm_done));
  CUDA_CHECK(cudaEventDestroy(s.compute_done));
  CUDA_CHECK(cudaEventDestroy(s.phase_ready));
  CUDA_CHECK(cudaEventDestroy(s.start));
  CUDA_CHECK(cudaStreamDestroy(s.join_stream));
  CUDA_CHECK(cudaStreamDestroy(s.release_stream));
  CUDA_CHECK(cudaStreamDestroy(s.comm_stream));
  NCCL_CHECK(ncclCommDestroy(s.comm));
  s.initialized = false;
}

}  // extern "C"
