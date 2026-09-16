// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Build with `gcc -O2 nccl_telemetry_benchmark.cc -o nccl_telemetry_benchmark
// -lstdc++ -ldl`. Run `./nccl_telemetry_benchmark` to see results.

#include <dlfcn.h>
#include <stdarg.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "../src/nccl_stats.h"

#define NUM_CONNECTIONS 10000
#define NUM_SAMPLES (10000 * NUM_CONNECTIONS)
const std::vector<uint32_t> LOCAL_ADDR = {0x0100007F, 0x123456, 0x234567,
                                          0x567890,   0x11111,  0x222222,
                                          0x3333333,  0x44444};
#define MAX_LATENCY (1000 * 1000 * 100)
#define MAX_SIZE (512 * 1024)

uintptr_t createRandomConnection(ncclStatsPlugin_t* nccl_stats_plugin,
                                 uintptr_t statsGlobalHandle) {
  ncclStatsTCPConnection connection;
  uintptr_t statsConnectionHandle;
  // Local Endpoint Creation
  struct sockaddr_storage local_addr;
  memset(&local_addr, 0, sizeof(local_addr));
  local_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&local_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&local_addr)->sin_addr.s_addr =
      LOCAL_ADDR[rand() % LOCAL_ADDR.size()];

  // Remote Endpoint Creation
  struct sockaddr_storage remote_addr;
  memset(&remote_addr, 0, sizeof(remote_addr));
  remote_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&remote_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&remote_addr)->sin_addr.s_addr = rand();

  connection.local_endpoint = local_addr;
  connection.remote_endpoint = remote_addr;

  ncclStatsConnectionIdentifier connection_identifier = {};
  connection_identifier.connection.tcp_conn = connection;
  connection_identifier.nccl_plugin_name = "StubPlugin";
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.nccl_plugin_type = CollNetPlugin;
  connection_identifier.nccl_plugin.coll_net_plugin = 0;
  connection_identifier.gpu_pci_addr = NULL;
  nccl_stats_plugin->addConnection(statsGlobalHandle, &connection_identifier,
                                   &statsConnectionHandle);
  std::cout << "statsConnectionHandle in main: " << statsConnectionHandle
            << std::endl;
  return statsConnectionHandle;
}

void addRandomEvent(ncclStatsPlugin_t* nccl_stats_plugin,
                    const std::vector<uintptr_t>& statsConnectionHandles) {
  ncclStatsOperationMetric nccl_stats_operation_metric;
  nccl_stats_operation_metric.type = OperationTypeChunkSend;
  nccl_stats_operation_metric.num_measurements = 1;
  nccl_stats_operation_metric.op_id = rand();
  nccl_stats_operation_metric.collective_id = rand();
  nccl_stats_operation_metric.op_start_time = 1234567890;
  nccl_stats_operation_metric.op_sz = rand() % MAX_SIZE;
  ncclStatsLatencyMeasurement nccl_stats_latency_measurement[1];

  // Software Latency
  nccl_stats_latency_measurement[0].latency_type = LatencySoftware;
  nccl_stats_latency_measurement[0].latency_in_nanoseconds =
      rand() % MAX_LATENCY;

  nccl_stats_operation_metric.measurements = nccl_stats_latency_measurement;
  nccl_stats_plugin->notifyOperationMeasurement(
      statsConnectionHandles[rand() % NUM_CONNECTIONS],
      &nccl_stats_operation_metric);
}

enum { NS_PER_SECOND = 1000000000 };

void sub_timespec(struct timespec t1, struct timespec t2, struct timespec* td) {
  td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
  td->tv_sec = t2.tv_sec - t1.tv_sec;
  if (td->tv_sec > 0 && td->tv_nsec < 0) {
    td->tv_nsec += NS_PER_SECOND;
    td->tv_sec--;
  } else if (td->tv_sec < 0 && td->tv_nsec > 0) {
    td->tv_nsec -= NS_PER_SECOND;
    td->tv_sec++;
  }
}

static void dummyDebugLog(ncclDebugLogLevel level, uint64_t flags,
                          const char* filefunc, int line, const char* fmt,
                          ...) {
  (void)level;
  (void)flags;
  // printf("%s:%d ", filefunc, line);
  va_list args;
  va_start(args, fmt);
  // vprintf(fmt, args);
  va_end(args);
  // printf("\n");
}

std::string GetGpuVizLocation() {
  char* location = getenv("GPUVIZ_LOCATION");
  if (!location || strlen(location) == 0) {
    return "../build/libGPUViz.so";
  }
  return std::string(location);
}

int main() {
  std::string gpuviz_location = GetGpuVizLocation();
  void* ncclTelemetryPluginLib = dlopen(gpuviz_location.c_str(), RTLD_LAZY);
  if (!ncclTelemetryPluginLib) {
    std::cout << "Failed to load library " << dlerror() << std::endl;
    return -1;
  }
  srand(time(NULL));
  ncclStatsPlugin_t* nccl_stats_plugin = (ncclStatsPlugin_t*)dlsym(
      ncclTelemetryPluginLib, "nccl_telemetry_stats_plugin_v1");
  std::cout << nccl_stats_plugin->name << std::endl;
  uintptr_t statsGlobalHandle = 0;
  std::vector<uintptr_t> statsConnectionHandles;

  std::cout << "<=============================== Test Init "
               "===============================>"
            << std::endl;
  uint64_t distributionTypeList = 0;

  distributionTypeList |= RecvLatencySW;
  distributionTypeList |= SendLatencySW;
  distributionTypeList |= SendMessageSize;
  distributionTypeList |= RecvMessageSize;
  const ncclResult_t init_result = nccl_stats_plugin->init(
      dummyDebugLog, distributionTypeList, &statsGlobalHandle);
  if (init_result != ncclSuccess) {
    std::cerr << "Failed to initialize plugin: " << init_result << std::endl;
    return -1;
  }

  std::cout << "statsGlobalHandle in main: " << statsGlobalHandle << std::endl;

  std::cout << "<=============================== Test Add Connections "
               "===============================>"
            << std::endl;

  for (int i = 0; i < NUM_CONNECTIONS; i++) {
    statsConnectionHandles.push_back(
        createRandomConnection(nccl_stats_plugin, statsGlobalHandle));
  }

  std::cout << "<=============================== Test Notify Measurement "
               "===============================>"
            << std::endl;
  struct timespec start, end, delta;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < NUM_SAMPLES; i++) {
    addRandomEvent(nccl_stats_plugin, statsConnectionHandles);
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  sub_timespec(start, end, &delta);
  std::cout << "Ran " << NUM_SAMPLES << " iterations at " << delta.tv_sec
            << " seconds and " << delta.tv_nsec << " nanoseconds" << std::endl;
  std::cout << "<=============================== Test Delete Connection "
               "===============================>"
            << std::endl;
  for (auto connection : statsConnectionHandles) {
    nccl_stats_plugin->deleteConnection(
        connection, ConnectionCloseLocalTerminate, "Normal Terminate");
  }

  std::cout << "<=============================== Test Destroy "
               "===============================>"
            << std::endl;
  nccl_stats_plugin->destroy(statsGlobalHandle);
}
