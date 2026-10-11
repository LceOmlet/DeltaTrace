// Diagnostic only. Subscribe to installed MetaX MCPTI; do not intercept or
// replace a runtime function, inspect tensor values, or synchronize a GPU.
// Parameter layouts come from generated_maca_runtime_api_meta.h in MACA 3.5.3.
#include <mcpti/mcpti.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
constexpr uint64_t capacity = 32768;
struct Record {
  uint64_t time_ns, thread, src, dst, bytes, stream;
  uint32_t cbid, site, correlation;
  int kind, result;
};
Record records[capacity];
std::atomic<uint64_t> count{0};
MCpti_SubscriberHandle subscriber;
std::string output_path;
bool active = false;
const MCpti_CallbackId ids[] = {
    MCPTI_RUNTIME_TRACE_CBID_mcMemcpy,
    MCPTI_RUNTIME_TRACE_CBID_mcMemcpyAsync,
    MCPTI_RUNTIME_TRACE_CBID_mcMemcpyWithStream};

void callback(void*, MCpti_CallbackDomain domain, MCpti_CallbackId id,
              const void* data) {
  if (domain != MCPTI_CB_DOMAIN_RUNTIME_API || !data) return;
  auto* info = static_cast<const MCpti_CallbackData*>(data);
  if (!info->functionParams) return;
  Record r{};
  r.cbid = id;
  r.site = info->callbackSite;
  r.correlation = info->correlationId;
  r.result = -1;
  if (id == MCPTI_RUNTIME_TRACE_CBID_mcMemcpy) {
    auto* p = static_cast<const mcMemcpy_params*>(info->functionParams);
    r.src = reinterpret_cast<uintptr_t>(p->src);
    r.dst = reinterpret_cast<uintptr_t>(p->dst);
    r.bytes = p->count;
    r.kind = p->kind;
  } else if (id == MCPTI_RUNTIME_TRACE_CBID_mcMemcpyAsync) {
    auto* p = static_cast<const mcMemcpyAsync_params*>(info->functionParams);
    r.src = reinterpret_cast<uintptr_t>(p->src);
    r.dst = reinterpret_cast<uintptr_t>(p->dst);
    r.bytes = p->count;
    r.kind = p->kind;
    r.stream = reinterpret_cast<uintptr_t>(p->stream);
  } else if (id == MCPTI_RUNTIME_TRACE_CBID_mcMemcpyWithStream) {
    auto* p = static_cast<const mcMemcpyWithStream_params*>(info->functionParams);
    r.src = reinterpret_cast<uintptr_t>(p->src);
    r.dst = reinterpret_cast<uintptr_t>(p->dst);
    r.bytes = p->sizeBytes;
    r.kind = p->kind;
    r.stream = reinterpret_cast<uintptr_t>(p->stream);
  } else {
    return;
  }
  if (info->callbackSite == MCPTI_API_EXIT && info->functionReturnValue)
    r.result = *static_cast<const int*>(info->functionReturnValue);
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  r.time_ns = uint64_t(now.tv_sec) * 1000000000 + now.tv_nsec;
  r.thread = syscall(SYS_gettid);
  uint64_t i = count.fetch_add(1, std::memory_order_relaxed);
  if (i < capacity) records[i] = r;
}
}

extern "C" int dt_diagnostic_start(const char* path) {
  if (active || !path) return -1;
  output_path = path;
  count.store(0);
  auto status = mcptiSubscribe(&subscriber, callback, nullptr);
  if (status != MCPTI_SUCCESS) return status;
  for (auto id : ids) {
    status = mcptiEnableCallback(1, subscriber, MCPTI_CB_DOMAIN_RUNTIME_API, id);
    if (status != MCPTI_SUCCESS) {
      mcptiUnsubscribe(subscriber);
      return status;
    }
  }
  active = true;
  return 0;
}

extern "C" int dt_diagnostic_stop() {
  if (!active) return -1;
  for (auto id : ids)
    mcptiEnableCallback(0, subscriber, MCPTI_CB_DOMAIN_RUNTIME_API, id);
  auto status = mcptiUnsubscribe(subscriber);
  active = false;
  if (status != MCPTI_SUCCESS) return status;
  FILE* f = fopen(output_path.c_str(), "w");
  if (!f) return -2;
  uint64_t total = count.load();
  fprintf(f, "{\"total\":%llu,\"capacity\":%llu,\"dropped\":%llu}\n",
          (unsigned long long)total, (unsigned long long)capacity,
          (unsigned long long)(total > capacity ? total - capacity : 0));
  for (uint64_t i = 0; i < total && i < capacity; ++i) {
    const auto& r = records[i];
    fprintf(f, "{\"time_ns\":%llu,\"thread\":%llu,\"src\":%llu,"
               "\"dst\":%llu,\"bytes\":%llu,\"stream\":%llu,"
               "\"cbid\":%u,\"site\":%u,\"correlation\":%u,"
               "\"kind\":%d,\"result\":%d}\n",
            (unsigned long long)r.time_ns, (unsigned long long)r.thread,
            (unsigned long long)r.src, (unsigned long long)r.dst,
            (unsigned long long)r.bytes, (unsigned long long)r.stream,
            r.cbid, r.site, r.correlation, r.kind, r.result);
  }
  return fclose(f) == 0 ? 0 : -3;
}
