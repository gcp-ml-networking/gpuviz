// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run `gcc nccl_telemetry_stub_main.cc -o nccl_telemetry_stub_main -lstdc++
// -ldl`. Run `./nccl_telemetry_stub_main` to see results.

#include <dlfcn.h>
#include <string.h>
#include <sys/socket.h>

#include <iostream>

#include "../src/nccl_stats.h"

using namespace std;
int main() {
  void* ncclTelemetryPluginLib = dlopen("../build/libGPUVizStub.so", RTLD_LAZY);
  if (!ncclTelemetryPluginLib) {
    cout << "Failed to load library " << dlerror() << endl;
    return -1;
  }
  ncclStatsPlugin_t* nccl_stats_plugin = (ncclStatsPlugin_t*)dlsym(
      ncclTelemetryPluginLib, "nccl_telemetry_stats_plugin_v1");
  std::cout << nccl_stats_plugin->name << endl;
  ncclDebugLogger_t logFunction;
  uintptr_t statsGlobalHandle = 0;
  uintptr_t statsConnectionHandle = 1;

  cout << "<=============================== Test Init "
          "===============================>"
       << endl;
  uint64_t distributionTypeList = 0;

  distributionTypeList |= RecvLatencyNetHW;
  distributionTypeList |= RecvLatencySW;
  distributionTypeList |= SendLatencyNetHW;
  nccl_stats_plugin->init(logFunction, distributionTypeList,
                          &statsGlobalHandle);

  std::cout << "statsGlobalHandle in main: " << statsGlobalHandle << endl;

  cout << "<=============================== Test Add Connection "
          "===============================>"
       << endl;
  ncclStatsTCPConnection connection;

  // Local Endpoint Creation
  struct sockaddr_storage local_addr;
  memset(&local_addr, 0, sizeof(local_addr));
  local_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&local_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&local_addr)->sin_addr.s_addr = inet_addr("127.0.0.1");

  // Remote Endpoint Creation
  struct sockaddr_storage remote_addr;
  memset(&remote_addr, 0, sizeof(remote_addr));
  remote_addr.ss_family = AF_INET;
  ((struct sockaddr_in*)&remote_addr)->sin_port = htons(80);
  ((struct sockaddr_in*)&remote_addr)->sin_addr.s_addr = inet_addr("128.0.0.1");

  connection.local_endpoint = local_addr;
  connection.remote_endpoint = remote_addr;

  ncclStatsConnectionIdentifier connection_identifier;
  connection_identifier.connection.tcp_conn = connection;
  connection_identifier.nccl_plugin_name = "StubPlugin";
  connection_identifier.conn_type = EntityTCPConnection;
  connection_identifier.nccl_plugin_type = CollNetPlugin;
  connection_identifier.nccl_plugin.coll_net_plugin = (ncclCollNet_t*)32547698;
  nccl_stats_plugin->addConnection(statsGlobalHandle, &connection_identifier,
                                   &statsConnectionHandle);
  std::cout << "statsConnectionHandle in main: " << statsConnectionHandle
            << endl;

  cout << "<=============================== Test Notify Measurement "
          "===============================>"
       << endl;

  ncclStatsOperationMetric nccl_stats_operation_metric;
  nccl_stats_operation_metric.type = OperationTypeChunkSend;
  nccl_stats_operation_metric.num_measurements = 2;
  nccl_stats_operation_metric.op_id = 763521;
  nccl_stats_operation_metric.collective_id = 89321;
  nccl_stats_operation_metric.op_start_time = 1234567890;
  ncclStatsLatencyMeasurement nccl_stats_latency_measurement[2];

  // Software Latency
  nccl_stats_latency_measurement[0].latency_type = LatencySoftware;
  nccl_stats_latency_measurement[0].latency_in_nanoseconds = 10;

  // Hardware Latency
  nccl_stats_latency_measurement[1].latency_type = LatencyNetHW;
  nccl_stats_latency_measurement[1].latency_in_nanoseconds = 12;

  nccl_stats_operation_metric.measurements = nccl_stats_latency_measurement;
  nccl_stats_plugin->notifyOperationMeasurement(statsConnectionHandle,
                                                &nccl_stats_operation_metric);

  cout << "<=============================== Test Delete Connection "
          "===============================>"
       << endl;
  nccl_stats_plugin->deleteConnection(
      statsConnectionHandle, ConnectionCloseLocalTerminate, "Normal Terminate");

  cout << "<=============================== Test Destroy "
          "===============================>"
       << endl;
  nccl_stats_plugin->destroy(statsGlobalHandle);
}