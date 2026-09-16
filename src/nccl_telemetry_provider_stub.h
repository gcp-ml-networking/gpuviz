/*
 * Copyright 2026 Google LLC
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef NCCL_TELEMETRY_PROVIDER_H_
#define NCCL_TELEMETRY_PROVIDER_H_

#include <iostream>
#include <map>
#include <vector>

#include "nccl_stats.h"
#include "telemetry_provider.h"

namespace gpuviz {

using namespace std;

class NcclTelemetryStatsStub : public TelemetryProvider {
 public:
  static ncclResult_t init(ncclDebugLogger_t logFunction,
                           uint64_t distributionCollectorBitmap,
                           uintptr_t* statsGlobalHandle /* Out */);
  static ncclResult_t destroy(uintptr_t statsGlobalHandle);
  static ncclResult_t addConnection(
      uintptr_t statsGlobalHandle,
      const ncclStatsConnectionIdentifier* connectionIdentifier,
      uintptr_t* statsConnectionHandle);
  static ncclResult_t deleteConnection(uintptr_t statsConnectionHandle,
                                       ncclStatsConnectionCloseType closeType,
                                       const char* verboseReason);
  static ncclResult_t notifyOperationMeasurement(
      uintptr_t statsConnectionHandle,
      const ncclStatsOperationMetric* measurement);
  absl::StatusOr<std::unique_ptr<protoDist::AllStats>> readTelemetry();
  absl::Status processHighFrequencyTelemetry();
  void addLogMsg(absl::string_view message, ncclDebugLogLevel level,
                 const char* pretty_func, uint32_t line,
                 ncclDebugLogSubSys subsys, std::string category_name,
                 int32_t category, int32_t msg_id);
  void addLogMsg(absl::string_view message, ncclDebugLogLevel level,
                 const char* pretty_func, uint32_t line) {
    addLogMsg(message, level, pretty_func, line, NCCL_NET, "", 0, 0);
  }

 private:
  static void print_sockaddr_storage(sockaddr_storage sockaddr,
                                     std::ostream& os);
  static void print_ncclStatsConnectionIdentifier(
      ncclStatsConnectionIdentifier connectionIdentifier);
  static void print_ncclStatsOperationMetric(
      ncclStatsOperationMetric operationMetric);
};

}  // namespace gpuviz

#endif  // NCCL_TELEMETRY_PROVIDER_H_
