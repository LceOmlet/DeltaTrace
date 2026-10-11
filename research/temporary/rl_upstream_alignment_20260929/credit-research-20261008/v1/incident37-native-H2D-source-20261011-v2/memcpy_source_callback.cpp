// Diagnostic only: installed MCPTI callbacks; no runtime interception or GPU API.
// Read only the valid host source specified by an existing H2D call; no GPU data.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <mcpti/mcpti.h>
#include <atomic>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace {
constexpr uint64_t capacity = 1024;
constexpr size_t maximum_row_bytes = 8192;
std::atomic<uint64_t> selected{0};
uint64_t copy_bytes, row_offset, row_bytes;
MCpti_SubscriberHandle subscriber;
FILE* output = nullptr;
std::mutex output_mutex;
bool active = false;
const MCpti_CallbackId ids[] = {MCPTI_RUNTIME_TRACE_CBID_mcMemcpy,
    MCPTI_RUNTIME_TRACE_CBID_mcMemcpyAsync,
    MCPTI_RUNTIME_TRACE_CBID_mcMemcpyWithStream};

void callback(void*, MCpti_CallbackDomain domain, MCpti_CallbackId id,
              const void* data) {
  if (domain != MCPTI_CB_DOMAIN_RUNTIME_API || !data) return;
  const auto* info = static_cast<const MCpti_CallbackData*>(data);
  if (!info->functionParams) return;
  uintptr_t src = 0, dst = 0, stream = 0;
  uint64_t bytes = 0;
  int kind = -1;
  if (id == MCPTI_RUNTIME_TRACE_CBID_mcMemcpy) {
    const auto* p = static_cast<const mcMemcpy_params*>(info->functionParams);
    src = reinterpret_cast<uintptr_t>(p->src); dst = reinterpret_cast<uintptr_t>(p->dst);
    bytes = p->count; kind = p->kind;
  } else if (id == MCPTI_RUNTIME_TRACE_CBID_mcMemcpyAsync) {
    const auto* p = static_cast<const mcMemcpyAsync_params*>(info->functionParams);
    src = reinterpret_cast<uintptr_t>(p->src); dst = reinterpret_cast<uintptr_t>(p->dst);
    bytes = p->count; kind = p->kind; stream = reinterpret_cast<uintptr_t>(p->stream);
  } else if (id == MCPTI_RUNTIME_TRACE_CBID_mcMemcpyWithStream) {
    const auto* p = static_cast<const mcMemcpyWithStream_params*>(info->functionParams);
    src = reinterpret_cast<uintptr_t>(p->src); dst = reinterpret_cast<uintptr_t>(p->dst);
    bytes = p->sizeBytes; kind = p->kind; stream = reinterpret_cast<uintptr_t>(p->stream);
  } else return;
  if (kind != mcMemcpyHostToDevice || bytes != copy_bytes) return;
  const uint64_t index = selected.fetch_add(1, std::memory_order_relaxed);
  if (index >= capacity) return;
  unsigned char row[maximum_row_bytes];
  char hex[maximum_row_bytes * 2 + 1];
  timespec now{}; clock_gettime(CLOCK_MONOTONIC, &now);
  const uint64_t time_ns = uint64_t(now.tv_sec) * 1000000000 + now.tv_nsec;
  // mcMemcpyHostToDevice owns the host-source validity contract at this call.
  std::memcpy(row, reinterpret_cast<const void*>(src + row_offset), row_bytes);
  const ssize_t read_bytes = static_cast<ssize_t>(row_bytes);
  const int read_errno = 0;
  const size_t retained = row_bytes;
  const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < retained; ++i) {
    hex[2*i] = digits[row[i] >> 4]; hex[2*i+1] = digits[row[i] & 15];
  }
  hex[2*retained] = 0;
  int result = -1;
  if (info->callbackSite == MCPTI_API_EXIT && info->functionReturnValue)
    result = *static_cast<const int*>(info->functionReturnValue);
  std::lock_guard<std::mutex> lock(output_mutex);
  fprintf(output, "{\"index\":%llu,\"time_ns\":%llu,\"thread\":%ld,"
      "\"src\":%llu,\"dst\":%llu,\"bytes\":%llu,\"stream\":%llu,"
      "\"cbid\":%u,\"site\":%u,\"correlation\":%u,\"kind\":%d,"
      "\"result\":%d,\"row_offset\":%llu,\"row_bytes\":%llu,"
      "\"read_bytes\":%lld,\"read_errno\":%d,\"CPU_row_hex\":\"%s\"}\n",
      (unsigned long long)index, (unsigned long long)time_ns, syscall(SYS_gettid),
      (unsigned long long)src, (unsigned long long)dst, (unsigned long long)bytes,
      (unsigned long long)stream, id, info->callbackSite, info->correlationId, kind,
      result, (unsigned long long)row_offset, (unsigned long long)row_bytes,
      (long long)read_bytes, read_errno, hex);
  fflush(output); // Persist bounded CPU observations even if the Ray worker exits abruptly.
}
}

extern "C" int dt_source_start(const char* path, uint64_t bytes,
                               uint64_t offset, uint64_t row_size) {
  if (active || !path || row_size == 0 || row_size > maximum_row_bytes ||
      offset > bytes || row_size > bytes - offset) return -1;
  output = fopen(path, "wx");
  if (!output) return -2;
  copy_bytes = bytes; row_offset = offset; row_bytes = row_size; selected.store(0);
  fprintf(output, "{\"filter_copy_bytes\":%llu,\"row_offset\":%llu,"
      "\"row_bytes\":%llu,\"capacity\":%llu,\"pid\":%d}\n",
      (unsigned long long)bytes, (unsigned long long)offset,
      (unsigned long long)row_size, (unsigned long long)capacity, getpid());
  fflush(output);
  auto status = mcptiSubscribe(&subscriber, callback, nullptr);
  if (status != MCPTI_SUCCESS) { fclose(output); output = nullptr; return status; }
  for (auto id : ids) {
    status = mcptiEnableCallback(1, subscriber, MCPTI_CB_DOMAIN_RUNTIME_API, id);
    if (status != MCPTI_SUCCESS) {
      mcptiUnsubscribe(subscriber); fclose(output); output = nullptr; return status;
    }
  }
  active = true;
  return 0;
}

extern "C" int dt_source_stop() {
  if (!active) return -1;
  for (auto id : ids) mcptiEnableCallback(0, subscriber, MCPTI_CB_DOMAIN_RUNTIME_API, id);
  const auto status = mcptiUnsubscribe(subscriber);
  active = false;
  const uint64_t total = selected.load();
  fprintf(output, "{\"selected\":%llu,\"capacity\":%llu,\"dropped\":%llu,"
      "\"unsubscribe_status\":%d}\n", (unsigned long long)total,
      (unsigned long long)capacity, (unsigned long long)(total > capacity ? total-capacity : 0), status);
  const int close_status = fclose(output); output = nullptr;
  return status != MCPTI_SUCCESS ? status : close_status;
}
