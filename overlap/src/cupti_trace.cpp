// Records CUPTI CONCURRENT_KERNEL activity (start/end of every kernel) and writes it as TSV at stop.
#include <cupti.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// All buffers are allocated and faulted in up front: no allocation or file I/O while measuring.
constexpr size_t kBufferBytes = 64ULL * 1024ULL * 1024ULL;
constexpr size_t kBufferCount = 4;

struct BufferSlot {
  uint8_t* data = nullptr;
  std::atomic<size_t> valid_size{0};
  std::atomic<uintptr_t> context{0};
  std::atomic<uint32_t> stream_id{0};
  std::atomic<bool> returned{false};
};

std::array<BufferSlot, kBufferCount> slots;
std::atomic<size_t> next_buffer{0};
std::atomic<size_t> returned_buffers{0};
std::atomic<bool> pool_exhausted{false};
std::atomic<bool> callback_error{false};
bool callbacks_registered = false;
bool prepared = false;
bool enabled = false;
bool capture_was_started = false;
int collector_rank = -1;
std::string output_path;

void report_cupti_error(const char* operation, CUptiResult result) {
  const char* text = nullptr;
  cuptiGetResultString(result, &text);
  std::fprintf(stderr,
               "CUPTI_TRACE ERROR rank=%d operation=%s result=%d "
               "text=%s\n",
               collector_rank, operation, static_cast<int>(result),
               text == nullptr ? "unknown" : text);
}

std::string summary_path() { return output_path + ".summary.env"; }

const char* channel_type_name(CUpti_ChannelType type) {
  switch (type) {
    case CUPTI_CHANNEL_TYPE_COMPUTE:
      return "COMPUTE";
    case CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY:
      return "ASYNC_MEMCPY";
    case CUPTI_CHANNEL_TYPE_DECOMP:
      return "DECOMP";
    default:
      return "INVALID";
  }
}

void sanitize_name(const char* input, std::string* output) {
  output->clear();
  if (input == nullptr) {
    *output = "<null>";
    return;
  }
  for (const unsigned char value : std::string(input)) {
    if (value == '\t' || value == '\n' || value == '\r') {
      output->push_back(' ');
    } else {
      output->push_back(static_cast<char>(value));
    }
  }
}

void CUPTIAPI request_buffer(uint8_t** buffer, size_t* size,
                             size_t* max_records) {
  const size_t index = next_buffer.fetch_add(1, std::memory_order_relaxed);
  if (index >= slots.size()) {
    pool_exhausted.store(true, std::memory_order_relaxed);
    *buffer = nullptr;
    *size = 0;
    *max_records = 0;
    return;
  }
  *buffer = slots[index].data;
  *size = kBufferBytes;
  *max_records = 0;
}

void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream_id,
                              uint8_t* buffer, size_t, size_t valid_size) {
  if (buffer == nullptr) {
    callback_error.store(true, std::memory_order_relaxed);
    return;
  }
  for (BufferSlot& slot : slots) {
    if (slot.data != buffer) continue;
    slot.context.store(reinterpret_cast<uintptr_t>(context),
                       std::memory_order_relaxed);
    slot.stream_id.store(stream_id, std::memory_order_relaxed);
    slot.valid_size.store(valid_size, std::memory_order_release);
    slot.returned.store(true, std::memory_order_release);
    returned_buffers.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  callback_error.store(true, std::memory_order_relaxed);
}

int write_summary(const char* status, uint64_t records, uint64_t unexpected,
                  uint64_t invalid_timestamps, uint64_t dropped) {
  FILE* summary = std::fopen(summary_path().c_str(), "w");
  if (summary == nullptr) {
    std::fprintf(stderr,
                 "CUPTI_TRACE ERROR rank=%d operation=fopen_summary "
                 "path=%s error=%s\n",
                 collector_rank, summary_path().c_str(), std::strerror(errno));
    return 1;
  }
  std::fprintf(summary, "status=%s\n", status);
  std::fprintf(summary, "rank=%d\n", collector_rank);
  std::fprintf(summary, "enabled_activity_kind=CONCURRENT_KERNEL\n");
  std::fprintf(summary, "capture_started=%d\n", capture_was_started ? 1 : 0);
  std::fprintf(summary, "buffer_count=%zu\n", kBufferCount);
  std::fprintf(summary, "buffer_bytes=%zu\n", kBufferBytes);
  std::fprintf(summary, "buffers_requested=%zu\n",
               next_buffer.load(std::memory_order_relaxed));
  std::fprintf(summary, "buffers_returned=%zu\n",
               returned_buffers.load(std::memory_order_relaxed));
  std::fprintf(summary, "pool_exhausted=%d\n",
               pool_exhausted.load(std::memory_order_relaxed) ? 1 : 0);
  std::fprintf(summary, "callback_error=%d\n",
               callback_error.load(std::memory_order_relaxed) ? 1 : 0);
  std::fprintf(summary, "kernel_records=%" PRIu64 "\n", records);
  std::fprintf(summary, "unexpected_records=%" PRIu64 "\n", unexpected);
  std::fprintf(summary, "invalid_timestamps=%" PRIu64 "\n",
               invalid_timestamps);
  std::fprintf(summary, "dropped_records=%" PRIu64 "\n", dropped);
  if (std::fclose(summary) != 0) return 1;
  return 0;
}

int write_activity() {
  FILE* output = std::fopen(output_path.c_str(), "w");
  if (output == nullptr) {
    std::fprintf(stderr,
                 "CUPTI_TRACE ERROR rank=%d operation=fopen_activity "
                 "path=%s error=%s\n",
                 collector_rank, output_path.c_str(), std::strerror(errno));
    return 1;
  }
  std::fprintf(
      output,
      "start_ns\tend_ns\tduration_ns\tcompleted_ns\tdevice_id\tcontext_id\t"
      "stream_id\tcorrelation_id\tgrid_id\tgrid_x\tgrid_y\tgrid_z\tblock_x\t"
      "block_y\tblock_z\tstatic_shared_bytes\tdynamic_shared_bytes\t"
      "registers_per_thread\tgraph_node_id\tgraph_id\tchannel_id\t"
      "channel_type\tcluster_x\tcluster_y\tcluster_z\tlaunch_priority\tname\n");

  uint64_t records = 0;
  uint64_t unexpected = 0;
  uint64_t invalid_timestamps = 0;
  uint64_t dropped = 0;
  int parse_error = 0;
  std::string clean_name;
  for (BufferSlot& slot : slots) {
    if (!slot.returned.load(std::memory_order_acquire)) continue;
    const size_t valid_size = slot.valid_size.load(std::memory_order_acquire);
    CUpti_Activity* record = nullptr;
    CUptiResult result = CUPTI_SUCCESS;
    while ((result = cuptiActivityGetNextRecord(slot.data, valid_size, &record)) ==
           CUPTI_SUCCESS) {
      if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        ++unexpected;
        continue;
      }
      const auto* kernel =
          reinterpret_cast<const CUpti_ActivityKernel11*>(record);
      if (kernel->start == 0 || kernel->end == 0 ||
          kernel->end < kernel->start) {
        ++invalid_timestamps;
      }
      sanitize_name(kernel->name, &clean_name);
      std::fprintf(
          output,
          "%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64
          "\t%u\t%u\t%u\t%u\t%" PRId64
          "\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t%" PRIu64
          "\t%u\t%u\t%s\t%u\t%u\t%u\t%d\t%s\n",
          kernel->start, kernel->end,
          kernel->end >= kernel->start ? kernel->end - kernel->start : 0,
          kernel->completed, kernel->deviceId, kernel->contextId,
          kernel->streamId, kernel->correlationId, kernel->gridId,
          kernel->gridX, kernel->gridY, kernel->gridZ, kernel->blockX,
          kernel->blockY, kernel->blockZ, kernel->staticSharedMemory,
          kernel->dynamicSharedMemory, kernel->registersPerThread,
          kernel->graphNodeId, kernel->graphId, kernel->channelID,
          channel_type_name(kernel->channelType), kernel->clusterX,
          kernel->clusterY, kernel->clusterZ, kernel->priority,
          clean_name.c_str());
      ++records;
    }
    if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      report_cupti_error("get_next_record", result);
      parse_error = 1;
    }
    size_t slot_dropped = 0;
    const auto context = reinterpret_cast<CUcontext>(
        slot.context.load(std::memory_order_relaxed));
    const uint32_t stream_id = slot.stream_id.load(std::memory_order_relaxed);
    result = cuptiActivityGetNumDroppedRecords(context, stream_id, &slot_dropped);
    if (result != CUPTI_SUCCESS) {
      report_cupti_error("get_dropped_records", result);
      parse_error = 1;
    } else {
      dropped += slot_dropped;
    }
  }
  if (std::fclose(output) != 0) parse_error = 1;

  const bool failed =
      parse_error || records == 0 || unexpected != 0 ||
      invalid_timestamps != 0 || dropped != 0 ||
      pool_exhausted.load(std::memory_order_relaxed) ||
      callback_error.load(std::memory_order_relaxed);
  if (write_summary(failed ? "FAIL" : "PASS", records, unexpected,
                    invalid_timestamps, dropped) != 0) {
    return 1;
  }
  std::fprintf(stderr,
               "CUPTI_TRACE SUMMARY rank=%d status=%s records=%" PRIu64
               " dropped=%" PRIu64 " unexpected=%" PRIu64
               " invalid_timestamps=%" PRIu64 " requested=%zu returned=%zu\n",
               collector_rank, failed ? "FAIL" : "PASS", records, dropped,
               unexpected, invalid_timestamps,
               next_buffer.load(std::memory_order_relaxed),
               returned_buffers.load(std::memory_order_relaxed));
  return failed ? 1 : 0;
}

void free_buffers() {
  for (BufferSlot& slot : slots) {
    std::free(slot.data);
    slot.data = nullptr;
  }
}

}  // namespace

extern "C" int cupti_trace_prepare(const char* path, int rank) {
  if (prepared || path == nullptr || *path == '\0' || rank < 0) return 1;
  collector_rank = rank;
  output_path = path;
  for (BufferSlot& slot : slots) {
    void* allocation = nullptr;
    if (posix_memalign(&allocation, 8, kBufferBytes) != 0) {
      free_buffers();
      return 1;
    }
    slot.data = static_cast<uint8_t*>(allocation);
    std::memset(slot.data, 0, kBufferBytes);
  }
  const CUptiResult result =
      cuptiActivityRegisterCallbacks(request_buffer, complete_buffer);
  if (result != CUPTI_SUCCESS) {
    report_cupti_error("register_callbacks", result);
    free_buffers();
    return 1;
  }
  callbacks_registered = true;
  prepared = true;
  std::fprintf(stderr,
               "CUPTI_TRACE PREPARED rank=%d buffers=%zu "
               "buffer_bytes=%zu output=%s\n",
               collector_rank, kBufferCount, kBufferBytes, output_path.c_str());
  return 0;
}

extern "C" int cupti_trace_start() {
  if (!prepared || !callbacks_registered || enabled) return 1;
  const CUptiResult result =
      cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (result != CUPTI_SUCCESS) {
    report_cupti_error("enable_concurrent_kernel", result);
    return 1;
  }
  enabled = true;
  capture_was_started = true;
  std::fprintf(stderr,
               "CUPTI_TRACE START rank=%d "
               "enabled_kind=CONCURRENT_KERNEL\n",
               collector_rank);
  return 0;
}

extern "C" int cupti_trace_stop() {
  if (!prepared || !enabled) return 1;
  CUptiResult result =
      cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  if (result != CUPTI_SUCCESS) {
    report_cupti_error("flush", result);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    enabled = false;
    return 1;
  }
  const int write_result = write_activity();
  result = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  if (result != CUPTI_SUCCESS) {
    report_cupti_error("disable_concurrent_kernel", result);
    enabled = false;
    return 1;
  }
  enabled = false;
  std::fprintf(stderr, "CUPTI_TRACE STOP rank=%d\n", collector_rank);
  return write_result;
}

extern "C" int cupti_trace_finalize() {
  int result = 0;
  if (!prepared) return 1;
  if (enabled) result = cupti_trace_stop();
  if (!capture_was_started) {
    result = write_summary("NOT_STARTED", 0, 0, 0, 0);
  }
  free_buffers();
  prepared = false;
  callbacks_registered = false;
  output_path.clear();
  std::fprintf(stderr,
               "CUPTI_TRACE FINALIZED rank=%d capture_started=%d "
               "result=%d\n",
               collector_rank, capture_was_started ? 1 : 0, result);
  return result;
}
