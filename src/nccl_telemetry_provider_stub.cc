// Copyright 2026 Google LLC
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "nccl_telemetry_provider_stub.h"

namespace gpuviz {

void NcclTelemetryStatsStub::print_ncclStatsOperationMetric(
    ncclStatsOperationMetric operationMetric) {
  std::cout << "opIdentifier: " << operationMetric.op_id << std::endl;
  std::cout << "opSize: " << operationMetric.op_sz << std::endl;
  std::cout << "numMeasurements: " << operationMetric.num_measurements
            << std::endl;
  for (uint32_t i = 0; i < operationMetric.num_measurements; i++) {
    std::cout << "measurements[" << i
              << "]: " << operationMetric.measurements[i].latency_in_nanoseconds
              << " ns, " << operationMetric.measurements[i].latency_type
              << std::endl;
  }
}

void NcclTelemetryStatsStub::print_sockaddr_storage(sockaddr_storage sockaddr,
                                                    std::ostream& os) {
  if (sockaddr.ss_family == AF_INET) {
    struct sockaddr_in* in = reinterpret_cast<struct sockaddr_in*>(&sockaddr);
    os << in->sin_addr.s_addr << ":" << in->sin_port << std::endl;
  } else if (sockaddr.ss_family == AF_INET6) {
    struct sockaddr_in6* in6 =
        reinterpret_cast<struct sockaddr_in6*>(&sockaddr);
    os << in6->sin6_addr.s6_addr << ":" << in6->sin6_port << std::endl;
  }
}
void NcclTelemetryStatsStub::print_ncclStatsConnectionIdentifier(
    ncclStatsConnectionIdentifier connectionIdentifier) {
  std::cout << "connectionType: " << connectionIdentifier.conn_type
            << std::endl;
  std::cout << "nccl_plugin_name: " << connectionIdentifier.nccl_plugin_name
            << std::endl;
  switch (connectionIdentifier.nccl_plugin_type) {
    case ncclStatsPluginType::NetPlugin:
      std::cout << "Netplugin pointer: "
                << connectionIdentifier.nccl_plugin.net_plugin << std::endl;
      break;
    case ncclStatsPluginType::CollNetPlugin:
      std::cout << "Collnetplugin"
                << connectionIdentifier.nccl_plugin.coll_net_plugin
                << std::endl;
      break;
    case ncclStatsPluginType::ProfilerPlugin:
      std::cout << "ProfilerPlugin" << std::endl;
  }
  switch (connectionIdentifier.conn_type) {
    case EntityNVLConnection:
      std::cout << "NVLConnection: " << std::endl;
      break;
    case EntityPCIConnection:
      std::cout << "PCIConnection: " << std::endl;
      break;
    case EntityTCPConnection:
      std::cout << "TCPConnection local: ";
      print_sockaddr_storage(
          connectionIdentifier.connection.tcp_conn.local_endpoint, std::cout);
      std::cout << "TCPConnection remote: ";
      print_sockaddr_storage(
          connectionIdentifier.connection.tcp_conn.remote_endpoint, std::cout);
      break;
    case EntityRDMAConnection:
      std::cout << "RDMAConnection local: ";
      print_sockaddr_storage(
          connectionIdentifier.connection.rdma_conn.local_endpoint, std::cout);
      std::cout << "RDMAConnection local QPN: "
                << connectionIdentifier.connection.rdma_conn.local_qpn
                << std::endl;
      std::cout << "RDMAConnection remote: ";
      print_sockaddr_storage(
          connectionIdentifier.connection.rdma_conn.remote_endpoint, std::cout);
      std::cout << "RDMAConnection remote QPN: "
                << connectionIdentifier.connection.rdma_conn.remote_qpn
                << std::endl;
      break;
    case EntityProfilerPluginConnection:
      std::cout << "Profiler Plugin Connection" << std::endl;
  }
}

ncclResult_t NcclTelemetryStatsStub::init(
    ncclDebugLogger_t logFunction, uint64_t distributionCollectorBitmap,
    uintptr_t* statsGlobalHandle /* Out */) {
  (void)logFunction;
  *statsGlobalHandle = 12345;
  cout << "initImpl" << endl;
  for (int i = 0; i < TotalDistributionType; i++) {
    if ((distributionCollectorBitmap & (1 << i))) {
      switch (1 << i) {
        case RecvLatencyNetHW:
          cout << "RecvLatencyNetHW" << endl;
          break;
        case RecvLatencySW:
          cout << "RecvLatencySW" << endl;
          break;
        case RecvMessageSize:
          cout << "RecvMessageSize" << endl;
          break;
        case SendLatencyNetHW:
          cout << "SendLatencyNetHW" << endl;
          break;
        case RecvReadyLatency:
          cout << "RecvReadyLatency" << endl;
          break;
        case SendLatencySW:
          cout << "SendLatencySW" << endl;
          break;
        case SendMessageSize:
          cout << "SendMessageSize" << endl;
          break;
        default:
          cout << "Incorrect option";
      }
    }
  }
  return ncclSuccess;
}
ncclResult_t NcclTelemetryStatsStub::destroy(uintptr_t statsGlobalHandle) {
  cout << "destroyImpl: " << endl;
  cout << "statsGlobalHandle: " << statsGlobalHandle << endl;
  return ncclSuccess;
}

void NcclTelemetryStatsStub::addLogMsg(absl::string_view message,
                                       ncclDebugLogLevel level,
                                       const char* pretty_func, uint32_t line,
                                       ncclDebugLogSubSys subsys,
                                       std::string category_name,
                                       int32_t category, int32_t msg_id) {
  cout << message << endl;
}
ncclResult_t NcclTelemetryStatsStub::addConnection(
    uintptr_t statsGlobalHandle,
    const ncclStatsConnectionIdentifier* connectionIdentifier,
    uintptr_t* statsConnectionHandle) {
  cout << "addConnectionImpl" << endl;
  cout << "statsGlobalHandle: " << statsGlobalHandle << endl;
  print_ncclStatsConnectionIdentifier(*connectionIdentifier);
  *statsConnectionHandle = 54321;
  return ncclSuccess;
}
ncclResult_t NcclTelemetryStatsStub::deleteConnection(
    uintptr_t statsConnectionHandle, ncclStatsConnectionCloseType closeType,
    const char* verboseReason) {
  cout << "deleteConnectionImpl" << endl;
  cout << "statsConnectionHandle: " << statsConnectionHandle
       << "closeType: " << closeType << endl;
  printf("Verbose Reason: %s\n", verboseReason);
  return ncclSuccess;
}
ncclResult_t NcclTelemetryStatsStub::notifyOperationMeasurement(
    uintptr_t statsConnectionHandle,
    const ncclStatsOperationMetric* measurement) {
  cout << "notifyOperationMeasurementImpl" << endl;
  cout << "statsConnectionHandle: " << statsConnectionHandle << endl;
  print_ncclStatsOperationMetric(*measurement);
  return ncclSuccess;
}

absl::StatusOr<std::unique_ptr<protoDist::AllStats>>
NcclTelemetryStatsStub::readTelemetry() {
  cout << "getTelemetryImpl. return an empty AllStats for testing purpose"
       << endl;
  std::unique_ptr<protoDist::AllStats> stats =
      std::make_unique<protoDist::AllStats>();
  return stats;
}

absl::Status NcclTelemetryStatsStub::processHighFrequencyTelemetry() {
  return absl::OkStatus();
}

}  // namespace gpuviz
